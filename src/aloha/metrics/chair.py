"""
This code is modified directly from the original CHAIR codebase:
https://github.com/LisaAnne/Hallucination/blob/master/utils/chair.py

With credit to the paper authors:
Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, Kate Saenko.
Object hallucination in image captioning. In EMNLP, 2018.
"""

import json
import logging
import os
import string
from copy import deepcopy
from typing import Any, Dict, List

import nltk
import numpy as np
import pandas as pd
import spacy
from rich.progress import Progress

from aloha.metrics.base import Metric
from aloha.utils.pattern import singularize

nltk.download("averaged_perceptron_tagger")


def combine_coco_captions(annotation_path):
    annotation_path = os.path.join(annotation_path, "coco_data")
    if not os.path.exists("{}/captions_{}2014.json".format(annotation_path, "val")):
        raise Exception("Please download MSCOCO caption annotations for val set")
    if not os.path.exists("{}/captions_{}2014.json".format(annotation_path, "train")):
        raise Exception("Please download MSCOCO caption annotations for train set")

    val_caps = json.load(open("{}/captions_{}2014.json".format(annotation_path, "val")))
    train_caps = json.load(open("{}/captions_{}2014.json".format(annotation_path, "train")))
    all_caps = {
        "info": train_caps["info"],
        "licenses": train_caps["licenses"],
        "images": val_caps["images"] + train_caps["images"],
        "annotations": val_caps["annotations"] + train_caps["annotations"],
    }

    return all_caps


def combine_coco_instances(annotation_path):
    annotation_path = os.path.join(annotation_path, "coco_data")
    if not os.path.exists("{}/instances_{}2014.json".format(annotation_path, "val")):
        raise Exception("Please download MSCOCO instance annotations for val set")
    if not os.path.exists("{}/instances_{}2014.json".format(annotation_path, "train")):
        raise Exception("Please download MSCOCO instance annotations for train set")

    val_instances = json.load(open("{}/instances_{}2014.json".format(annotation_path, "val")))
    train_instances = json.load(open("{}/instances_{}2014.json".format(annotation_path, "train")))
    all_instances = {
        "info": train_instances["info"],
        "licenses": train_instances["licenses"],
        "type": train_instances["licenses"],
        "categories": train_instances["categories"],
        "images": train_instances["images"] + val_instances["images"],
        "annotations": val_instances["annotations"] + train_instances["annotations"],
    }

    return all_instances


def _get_postfix(filter_for_nouns: bool):
    return "" if filter_for_nouns else "_unfiltered"


def _update_scores(samples, method_postfix, filter_for_nouns=True):
    postfix = _get_postfix(filter_for_nouns)
    for i, sample in enumerate(samples):
        if sample.get("metrics") is None:
            sample["metrics"] = {}

        sample["metrics"][f"CHAIRs{postfix}{method_postfix}"] = sample[f"chair_data{postfix}{method_postfix}"][
            "metrics"
        ]["CHAIRs"]
        sample["metrics"][f"CHAIRi{postfix}{method_postfix}"] = sample[f"chair_data{postfix}{method_postfix}"][
            "metrics"
        ]["CHAIRi"]
    return samples


def create_coco_format_annotations(samples, reference_key):
    output = {"annotations": []}
    for i, sample in enumerate(samples):
        refs = sample[reference_key]
        imid = sample["image_id"]
        for r in refs:
            output["annotations"].append({"image_id": imid, "caption": r})
    return output


def _cm_kys(
    samples: List[Dict[str, Any]],
    candidate_key: str,
    reference_key: str,
    image_key: str,
    annotation_root: str = None,
    nocaps: bool = False,
    is_coco: bool = True,
) -> List[Dict[str, Any]]:
    """
    Compute and add CHAIR metrics to samples.

    Args:
        samples: List of samples.
        reference_key: Key of reference sentences in samples.
        candidate_key: Key of candidate sentences in samples.
        image_key: Key of COCO-format file names in samples (e.g., file name could be "COCO_val2014_000000360772.jpg").
        annotation_root: Path to directory containing COCO-format annotations (e.g., captions_val2014.json) and CHAIR file (synonyms.txt or nocaps_synonyms.txt).
        nocaps: Whether to use the nocaps version of CHAIR. Defaults to False.

    Returns:
        List of samples with CHAIR metrics added.
    """
    # Load COCO annotations
    if is_coco:
        # coco_caption_data = combine_coco_captions(annotation_root)
        coco_instance_data = combine_coco_instances(annotation_root)

        # Convert image paths to image ids.
        image_paths = [sample[image_key] for sample in samples]
        image_path2id = {c["file_name"]: c["id"] for c in coco_instance_data["images"]}
        image_ids = [image_path2id[image_path] for image_path in image_paths]

        # Add 'image_id' key to samples.  Used by CHAIR to add instance data.
        samples = [{**sample, "image_id": image_path2id[sample[image_key]]} for sample in samples]
    else:
        coco_instance_data = None
        image_ids = [sample["image_id"] for sample in samples]

    # Create CHAIR object
    if nocaps:
        evaluator = NocapsCHAIR(image_ids, annotation_root)
    else:
        caption_data = create_coco_format_annotations(samples, reference_key)
        evaluator = CHAIR(
            image_ids, annotation_root, coco_instance_data=coco_instance_data, coco_caption_data=caption_data
        )

    # Compute while filtering for nouns
    evaluator.get_annotations(image_ids, filter_for_nouns=True)
    samples = evaluator.compute_chair(samples, candidate_key, filter_for_nouns=True)
    samples = _update_scores(samples, evaluator.method_postfix, filter_for_nouns=True)

    # Compute without filtering for nouns (original CHAIR implementation)
    evaluator.reset(image_ids)
    evaluator.get_annotations(image_ids, filter_for_nouns=False)
    samples = evaluator.compute_chair(samples, candidate_key, filter_for_nouns=False)
    samples = _update_scores(samples, evaluator.method_postfix, filter_for_nouns=False)

    return samples


class CHAIR:
    def __init__(self, imids, data_path, coco_instance_data=None, coco_caption_data=None):
        logging.info("Initializing CHAIR")
        coco_path = os.path.join(data_path, "coco_data")
        self.coco_path = coco_path
        self.coco_instance_data = coco_instance_data
        self.coco_caption_data = coco_caption_data
        self.imid_to_objects = {imid: [] for imid in imids}
        self.method_postfix = ""

        # Read in synonyms
        logging.info("Reading in synonyms")
        synonyms = open(os.path.join(coco_path, "synonyms.txt")).readlines()
        synonyms = [s.strip().split(", ") for s in synonyms]
        self.root2synonyms = {s[0]: s for s in synonyms}
        self.mscoco_objects = []  # mscoco objects and *all* synonyms
        self.inverse_synonym_dict = {}
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        # Some hard coded rules for implementing CHAIR metrics on MSCOCO

        # Common 'double words' in MSCOCO that should be treated as a single word
        coco_double_words = [
            "motor bike",
            "motor cycle",
            "air plane",
            "traffic light",
            "street light",
            "traffic signal",
            "stop light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "suit case",
            "sports ball",
            "baseball bat",
            "baseball glove",
            "tennis racket",
            "wine glass",
            "hot dog",
            "cell phone",
            "mobile phone",
            "teddy bear",
            "hair drier",
            "potted plant",
            "bow tie",
            "laptop computer",
            "stove top oven",
            "hot dog",
            "teddy bear",
            "home plate",
            "train track",
        ]

        # Hard code some rules for special cases in MSCOCO
        # Qualifiers like 'baby' or 'adult' animal will lead to a false fire for the MSCOCO object 'person'.  'baby bird' --> 'bird'.
        animal_words = [
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "animal",
            "cub",
        ]
        # Qualifiers like 'passenger' vehicle will lead to a false fire for the MSCOCO object 'person'.  'passenger jet' --> 'jet'.
        vehicle_words = ["jet", "train"]

        # double_word_dict will map double words to the word they should be treated as in our analysis
        self.double_word_dict = {}
        for double_word in coco_double_words:
            self.double_word_dict[double_word] = double_word
        for animal_word in animal_words:
            self.double_word_dict[f"baby {animal_word}"] = animal_word
            self.double_word_dict[f"adult {animal_word}"] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict[f"passenger {vehicle_word}"] = vehicle_word
        self.double_word_dict["bow tie"] = "tie"
        self.double_word_dict["toilet seat"] = "toilet"
        self.double_word_dict["wine glas"] = "wine glass"

        # Hard-code common failure cases for noun parser. These words will always be counted as nouns.
        self.common_noun_parser_failures = ["moped", "urinal", "toaster", "oven"]

        # Load spacy
        logging.info("Loading spacy")
        self.nlp = spacy.load("en_core_web_sm")

    def reset(self, imids):
        """
        Reset annotations.
        """
        self.imid_to_objects = {imid: [] for imid in imids}

    def caption_to_words(self, caption, filter_for_nouns=True):  # noqa: C901,PLR0912
        """
        Input: caption
        Output: MSCOCO words in the caption
        """
        # Standard preprocessing
        transtab = str.maketrans({key: None for key in string.punctuation})
        caption = caption.translate(transtab)
        caption = caption.lower()
        words = nltk.word_tokenize(caption)
        words = [singularize(w) for w in words]

        # Remove punctuation
        words = [w for w in words if w not in string.punctuation]

        # Replace double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
            idxs.append(i)
            double_word = " ".join(words[i : i + 2])
            if double_word in self.double_word_dict:
                double_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        # toilet seat is not chair (sentences like "the seat of the toilet"
        # will fire for "chair" if we do not include this line)
        if ("toilet" in words) & ("seat" in words):
            words = [word for word in words if word != "seat"]

        # Get synonyms for all words in the caption
        idxs = [idxs[idx] for idx, word in enumerate(words) if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.mscoco_objects)]

        if filter_for_nouns:
            doc = self.nlp(caption)
            new_idxs = []
            new_words = []
            for idx, word in zip(idxs, words):
                if doc[idx].pos_ == "NOUN" or len(word.split(" ")) > 1 or word in self.common_noun_parser_failures:
                    # Double COCO words are likely to be used as nouns anyway
                    new_idxs.append(idx)
                    new_words.append(word)
                else:
                    # Catch some nouns that may have been missed
                    use = False

                    # If NLTK tags it as a noun, use it
                    toks = nltk.word_tokenize(caption.lower())
                    tagged = nltk.pos_tag(toks)
                    if tagged[idx][1] in ["NN", "NNS"]:
                        use = True

                    # If the word is the root of a noun phrase, use it
                    noun_chunks = list(doc.noun_chunks)
                    for chunk in noun_chunks:
                        if idx >= chunk.start and idx < chunk.end:
                            # In noun phrase
                            if chunk.root == doc[idx]:
                                use = True
                    if use:
                        new_idxs.append(idx)
                        new_words.append(word)
            words = new_words
            idxs = new_idxs

        node_words = []
        for word in words:
            node_words.append(self.inverse_synonym_dict[word])
        # Return all the MSCOCO objects in the caption
        return words, node_words, idxs, double_words

    def get_annotations(self, imids, filter_for_nouns=True):
        """
        Get annotations from both segmentation and captions.
        Need both annotation types for CHAIR metric.
        """
        self.get_annotations_from_segments(imids)
        self.get_annotations_from_captions(imids, filter_for_nouns=filter_for_nouns)

    def get_annotations_from_segments(self, imids):
        """
        Add objects taken from MSCOCO segmentation masks
        """
        if self.coco_instance_data is None:
            for imid in imids:
                self.imid_to_objects[imid] = set()
            return
        segment_annotations = self.coco_instance_data["annotations"]
        segment_annotations = [ann for ann in segment_annotations if ann["image_id"] in imids]

        # Make dict linking object name to ids
        id_to_name = {}  # dict with id to synsets
        for cat in self.coco_instance_data["categories"]:
            id_to_name[cat["id"]] = cat["name"]

        for i, annotation in enumerate(segment_annotations):
            imid = annotation["image_id"]
            if imid in self.imid_to_objects:
                node_word = self.inverse_synonym_dict[id_to_name[annotation["category_id"]]]
                self.imid_to_objects[imid].append(node_word)

        for imid in self.imid_to_objects:
            self.imid_to_objects[imid] = set(self.imid_to_objects[imid])

    def get_annotations_from_captions(self, imids, filter_for_nouns=True):
        """
        Add objects from references.
        """
        caption_annotations = self.coco_caption_data["annotations"]
        caption_annotations = [ann for ann in caption_annotations if ann["image_id"] in imids]

        for i, annotation in enumerate(caption_annotations):
            imid = annotation["image_id"]
            if imid in self.imid_to_objects:
                _, node_words, _, _ = self.caption_to_words(annotation["caption"], filter_for_nouns=filter_for_nouns)
                self.imid_to_objects[imid].update(node_words)

        for imid in self.imid_to_objects:
            self.imid_to_objects[imid] = set(self.imid_to_objects[imid])

    def compute_chair(self, samples: List[Dict[str, Any]], candidate_key: str, filter_for_nouns=True):
        """
        Given ground truth objects and generated captions, determine which sentences have hallucinated words.

        Args:
            samples: List of samples. Each sample must have keys 'image_id' and candidate_key.
            candidate_key: Key in sample to get generated caption
            reference_key: Key in sample to get ground truth caption
            filter_for_nouns: Whether to filter out non-nouns from the ground truth objects. If False, becomes original CHAIR implementation.
        """
        imid_to_objects = self.imid_to_objects
        postfix = _get_postfix(filter_for_nouns)

        num_caps = 0.0
        num_hallucinated_caps = 0.0
        hallucinated_word_count = 0.0
        coco_word_count = 0.0

        with Progress() as progress:
            task = progress.add_task("Computing CHAIR...", total=len(samples))
            for i, sample in enumerate(samples):
                sample[f"chair_data{postfix}{self.method_postfix}"] = {}

                cap = sample[candidate_key]
                imid = sample["image_id"]

                # Get all words in the caption, as well as corresponding node word
                words, node_words, idxs, raw_words = self.caption_to_words(cap, filter_for_nouns=filter_for_nouns)

                gt_objects = imid_to_objects[imid]
                cap_dict = {
                    "image_id": sample["image_id"],
                    "caption": cap,
                    "mscoco_hallucinated_words": [],
                    "mscoco_gt_words": list(gt_objects),
                    "mscoco_generated_words": list(node_words),
                    "hallucination_idxs": [],
                    "words": raw_words,
                }

                cap_dict["metrics"] = {"CHAIRs": 0, "CHAIRi": 0}

                # Count hallucinated words
                coco_word_count += len(node_words)
                hallucinated = False
                for word, node_word, idx in zip(words, node_words, idxs):
                    if node_word not in gt_objects:
                        # Incorrect (hallucinated)
                        hallucinated_word_count += 1
                        cap_dict["mscoco_hallucinated_words"].append((word, node_word))
                        cap_dict["hallucination_idxs"].append(idx)
                        hallucinated = True

                # Count hallucinated caps
                num_caps += 1
                if hallucinated:
                    num_hallucinated_caps += 1

                cap_dict["metrics"]["CHAIRs"] = int(hallucinated)
                cap_dict["metrics"]["CHAIRi"] = 0.0
                if len(words) > 0:
                    cap_dict["metrics"]["CHAIRi"] = len(cap_dict["mscoco_hallucinated_words"]) / float(len(words))

                sample[f"chair_data{postfix}{self.method_postfix}"].update(cap_dict)

                progress.advance(task)

        return samples


class NocapsCHAIR(CHAIR):
    def __init__(self, imids, data_path):  # noqa: C901
        self.method_postfix = "_nocaps"

        self.nlp = spacy.load("en_core_web_sm")
        self.imid_to_objects = {imid: [] for imid in imids}
        with open(os.path.join(data_path, "nocaps_data", "nocaps_val_4500_captions.json")) as f:
            self.nocaps_anns = json.load(f)
        self.nocaps_bbox_anns = pd.read_csv(os.path.join(data_path, "nocaps_data", "validation-annotations-bbox.csv"))

        # Nocaps objects
        self.descriptions = pd.read_csv(
            os.path.join(data_path, "nocaps_data", "class-descriptions-boxable.csv"), header=None
        )
        nocaps_classes = self.descriptions[1].values
        nocaps_classes = [c.lower() for c in nocaps_classes]

        # Map full class id, e.g. '/m/011k07' to class name, e.g. 'tortoise'
        self.fullid2class = {}
        for i in range(self.descriptions.shape[0]):
            self.fullid2class[self.descriptions[0][i]] = self.descriptions[1][i].lower()

        # Synonyms
        synonyms = open(os.path.join(data_path, "nocaps_data", "nocaps_synonyms.txt")).readlines()
        synonyms = [s.strip().split(", ") for s in synonyms]
        {s[0]: s for s in synonyms}
        [s[0] for s in synonyms]

        self.inverse_synonym_dict = {}
        self.mscoco_objects = []  # nocaps objects and *all* synonyms. name is still mscoco_objects for compatibility.
        for synonym in synonyms:
            self.mscoco_objects.extend(synonym)
            for s in synonym:
                self.inverse_synonym_dict[s] = synonym[0]

        # Additional acceptable synonyms, which may appear as other classes.
        # E.g. "person" and "woman" are separate classes in nocaps, but let "person" be an acceptable
        # synonym for "woman".
        additional_acceptable_classes = open(
            os.path.join(data_path, "nocaps_data", "nocaps_acceptable.txt")
        ).readlines()

        additional_acceptable_classes = [s.strip().split(",") for s in additional_acceptable_classes]
        for i, class_list in enumerate(additional_acceptable_classes):
            additional_acceptable_classes[i] = [c.strip() for c in class_list]

        for class_list in additional_acceptable_classes:
            for c in class_list:
                if isinstance(c, str):
                    c = [c]  # noqa: PLW2901
                self.mscoco_objects.extend(c)
        self.root2acceptable = {s[0]: s for s in additional_acceptable_classes}

        animal_words = [
            "tortoise",
            "magpie",
            "sea turtle",
            "bird",
            "bear",
            "brown bear",
            "blue jay",
            "bee",
            "bat",
            "starfish",
            "tick",
            "centipede",
            "camel",
            "cat",
            "beetle",
            "dinosaur",
            "dolphin",
            "harbor seal",
            "fox",
            "panda",
            "giraffe",
            "rhinoceros",
            "goldfish",
            "goat",
            "jaguar",
            "kangaroo",
            "koala",
            "lynx",
            "lizard",
            "mouse",
            "ostrich",
            "penguin",
            "polar bear",
            "pig",
            "raven",
            "red panda",
            "rabbit",
            "squirrel",
            "snake",
            "sheep",
            "tiger",
            "worm",
            "whale",
            "zebra",
            "monkey",
            "lion",
            "chicken",
            "eagle",
            "owl",
            "duck",
            "turtle",
            "hippopotamus",
            "crocodile",
            "squid",
            "spider",
            "deer",
            "frog",
            "dog",
            "elephant",
            "shark",
            "leopard",
            "hedgehog",
            "otter",
            "bull",
            "caterpillar",
            "butterfly",
            "antelope",
            "moth",
            "butterfly",
            "jellyfish",
            "goose",
            "mule",
            "swan",
            "raccoon",
            "falcon",
            "snail",
            "dragonfly",
            "sea lion",
            "ladybug",
            "parrot",
            "sparrow",
            "turkey",
            "ant",
            "skunk",
            "shrimp",
            "crab",
            "seahorse",
            "alpaca",
            "armadillo",
        ]

        vehicle_words = [
            "ambulance",
            "tank",
            "land craft",
            "airplane",
            "train",
            "boat",
            "bus",
            "barge",
            "limousine",
            "land vehicle",
            "watercraft",
            "snowmobile",
            "gondola",
            "helicopter",
            "taxi",
        ]

        # Double_word_dict will map double words to the word they should be treated as in our analysis
        nocaps_double_words = [c for c in nocaps_classes if len(c.split(" ")) > 1]
        self.double_word_dict = {}
        for double_word in nocaps_double_words:
            self.double_word_dict[double_word] = double_word

        for animal_word in animal_words:
            self.double_word_dict[f"baby {animal_word}"] = animal_word
            self.double_word_dict[f"adult {animal_word}"] = animal_word
        for vehicle_word in vehicle_words:
            self.double_word_dict[f"passenger {vehicle_word}"] = vehicle_word
        self.double_word_dict["bow tie"] = "tie"
        self.double_word_dict["toilet seat"] = "toilet"
        self.double_word_dict["wine glas"] = "wine glass"

    def caption_to_words(self, caption, filter_for_nouns=True):  # noqa: C901,PLR0912,PLR0915
        """
        Input: caption
        Output: nocaps words in the caption
        """
        transtab = str.maketrans({key: None for key in string.punctuation})
        caption = caption.translate(transtab)
        caption = caption.lower()
        caption = "".join(ch for ch in caption if ch.isalnum() or ch == " ")
        words = nltk.word_tokenize(caption)
        words = [singularize(w) for w in words]

        # Remove punctuation
        words = [w for w in words if w not in string.punctuation]

        # Replace double words
        i = 0
        double_words = []
        idxs = []
        while i < len(words):
            idxs.append(i)
            double_word = " ".join(words[i : i + 2])
            if double_word in self.double_word_dict:
                double_words.append(self.double_word_dict[double_word])
                i += 2
            else:
                double_words.append(words[i])
                i += 1
        words = double_words

        # toilet seat is not chair (sentences like "the seat of the toilet"
        # will fire for "chair" if we do not include this line)
        if ("toilet" in words) & ("seat" in words):
            words = [word for word in words if word != "seat"]

        # Get synonyms for all words in the caption
        idxs = [idxs[idx] for idx, word in enumerate(words) if word in set(self.mscoco_objects)]
        words = [word for word in words if word in set(self.mscoco_objects)]

        # Filter out words that are not nouns
        if filter_for_nouns:
            doc = self.nlp(caption)
            new_idxs = []
            new_words = []
            for idx, word in zip(idxs, words):  # or len(word.split(' ')) > 1:
                if doc[idx].pos_ == "NOUN":
                    new_idxs.append(idx)
                    new_words.append(word)
                elif doc[idx].pos_ == "PROPN":
                    # Catch some nouns that may have been missed (tagged as PROPN),
                    # but try to minimize false positives
                    use = True
                    # If NLTK doesn't tag it as a noun, don't count.
                    toks = nltk.word_tokenize(caption.lower())
                    tagged = nltk.pos_tag(toks)
                    if tagged[idx][1] not in ["NN", "NNS"]:
                        use = False

                    # If the word is not part of a noun phrase, don't count.
                    # If the word is not the root of the noun phrase it is a part of, don't count.
                    # E.g. prevent ORANGE from being classified as a noun
                    # in the noun phrase 'a black and yellow ORANGE insect'
                    noun_chunks = list(doc.noun_chunks)
                    in_noun_phrase = False
                    for chunk in noun_chunks:
                        if idx >= chunk.start and idx < chunk.end:
                            in_noun_phrase = True
                            if chunk.root != doc[idx]:
                                use = False
                    if not in_noun_phrase:
                        use = False
                    if use:
                        new_idxs.append(idx)
                        new_words.append(word)

            words = new_words
            idxs = new_idxs
        node_words = []
        for word in words:
            if word in self.inverse_synonym_dict:
                node_words.append(self.inverse_synonym_dict[word])
            else:
                node_words.append(word)

        return words, node_words, idxs, double_words

    def get_annotations_from_captions(self, filter_for_nouns=True):
        """
        Add objects taken from nocaps ground truth captions
        """
        caption_annotations = self.nocaps_anns["annotations"]

        for i, annotation in enumerate(caption_annotations):
            imid = annotation["image_id"]

            if imid in self.imid_to_objects:
                _, node_words, _, _ = self.caption_to_words(annotation["caption"], filter_for_nouns=filter_for_nouns)
                for n in node_words:
                    self.imid_to_objects[imid].extend(node_words)

    def get_annotations_from_boxes(self, filter_for_nouns=True):
        """
        Add objects taken from nocaps ground truth bounding boxes
        """
        # Map full class id, e.g. '/m/011k07' to class name, e.g. 'tortoise'
        fullid2class = self.fullid2class

        # Map image id, e.g. 0, to full image id, e.g. fe600639ac5f36c1
        imid2fullid = {}
        for i in range(len(self.nocaps_anns["images"])):
            ann = self.nocaps_anns["images"][i]
            imid2fullid[ann["id"]] = ann["open_images_id"]

        for imid in self.imid_to_objects.keys():
            fullid = imid2fullid[imid]
            boxes = self.nocaps_bbox_anns[self.nocaps_bbox_anns["ImageID"] == fullid]
            classes = boxes["LabelName"].values
            classes = [fullid2class[c] for c in classes]
            for c in classes:
                self.imid_to_objects[imid].extend([c])

    def get_annotations(self, imids, filter_for_nouns=True):
        """
        Get annotations from both bounding boxes and captions.
        Need both annotation types for CHAIR metric.
        """
        self.get_annotations_from_boxes()
        self.get_annotations_from_captions(filter_for_nouns=filter_for_nouns)

        for imid in self.imid_to_objects:
            # # Add additional acceptable words to set of objects
            objects = deepcopy(self.imid_to_objects[imid])
            for obj in objects:
                if obj in self.root2acceptable:
                    self.imid_to_objects[imid].extend(self.root2acceptable[obj])
            self.imid_to_objects[imid] = set(self.imid_to_objects[imid])


class CHAIRMetrics(Metric):
    def __init__(
        self,
        name: str = "chair",
        nocaps: bool = False,
        is_coco: bool = True,
    ):
        self._name = name
        self._nocaps = nocaps  # Whether to use the nocaps version of CHAIR
        self._is_coco = is_coco  # Whether to use COCO annotations

    def evaluate_dataset(
        self,
        samples: List[Dict[str, Any]],
        candidate_key: str,
        reference_key: str,
        image_key: str,
        image_root_dir: str,
        annotation_root: str,
    ) -> List[Dict[str, Any]]:
        """
        Compute and add CHAIR metrics to samples.

        Args:
            samples: List of samples.
            reference_key: Key of reference sentences in samples.
            candidate_key: Key of candidate sentences in samples.
            image_key: Key of COCO-format file names in samples (e.g., file name could be "COCO_val2014_000000360772.jpg").
            annotation_root: Path to directory containing COCO-format annotations (e.g., captions_val2014.json) and CHAIR file (synonyms.txt or nocaps_synonyms.txt).

        Returns:
            List of samples with CHAIR metrics added.
        """
        return _cm_kys(samples, candidate_key, reference_key, image_key, annotation_root, self._nocaps, self._is_coco)

    def aggregate(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Compute overall CHAIRs and CHAIRi scores, for both filtered and unfiltered versions
        output = {}
        output.update(self._aggregate(samples, filter_for_nouns=True))
        output.update(self._aggregate(samples, filter_for_nouns=False))
        for k, v in output.items():
            logging.info(f"{k}: {np.round(100*v, 2)}")

        return output

    def _aggregate(self, samples, filter_for_nouns):
        postfix = _get_postfix(filter_for_nouns)
        postfix += "_nocaps" if self._nocaps else ""

        num_hallucinated_caps = 0.0
        num_caps = 0.0
        hallucinated_word_count = 0.0
        coco_word_count = 0.0

        for i, sample in enumerate(samples):
            hallucinated_objs = sample[f"chair_data{postfix}"]["mscoco_hallucinated_words"]
            all_objs = sample[f"chair_data{postfix}"]["mscoco_generated_words"]

            num_hallucinated_caps += int(len(hallucinated_objs) > 0)
            num_caps += 1
            hallucinated_word_count += len(hallucinated_objs)
            coco_word_count += len(all_objs)

        # Overall metrics
        chair_s = num_hallucinated_caps / num_caps
        chair_i = hallucinated_word_count / coco_word_count

        return {f"CHAIRs{postfix}": chair_s, f"CHAIRi{postfix}": chair_i}
