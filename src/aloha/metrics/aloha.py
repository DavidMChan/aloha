import os
from typing import Any, Dict, Generator, List, Literal, Optional, Tuple, Type, TypedDict, Union

import numpy as np
import spacy
import torch
from PIL import Image
from rich.progress import Progress
from scipy.optimize import linear_sum_assignment

from aloha.metrics.base import Metric
from aloha.object_parser import GPT35TurboObjectParser, ObjectParser
from aloha.string_similarity import MPNetSimilarity, StringSimilarityMeasure
from aloha.utils import COCO_OBJECTS


class ALOHaMetricInfoDict(TypedDict):
    # The objects extracted by the object extractor (LLM) prior to any parsing or filtering
    unparsed_target_objects: str
    unparsed_reference_objects: str
    target_objects: List[List[str]]
    reference_objects: List[List[str]]
    matches: Optional[List[Dict[str, Any]]]


def _flatten(input: List[List[str]]) -> Generator[List[str], None, None]:
    # Generate all possible combinations of the list, where each element is a list,  taking one element from each list
    if len(input) == 0:
        return

    for item in input[0]:
        continuations = list(_flatten(input[1:]) or [])
        if len(continuations) == 0:
            yield [item]
        else:
            for rest in continuations:
                yield [item, *rest]


class ALOHa(Metric):
    def __init__(
        self,
        name: str = "aloha",
        object_parser: Type[ObjectParser] = GPT35TurboObjectParser,
        similarity_measure: Type[StringSimilarityMeasure] = MPNetSimilarity,
        num_reference_examples: Optional[int] = None,
        num_target_examples: Optional[int] = None,
        coco_objects: bool = False,
        debug: bool = True,
        min_object_length: int = 3,
        detect_objects: bool = False,
        similarity_aggregate_method: Union[Literal["mean"], Literal["min"]] = "min",
    ):
        # Construct the LLM engine and similarity measure
        self._name = name
        self._object_parser = object_parser(
            num_target_examples=num_target_examples, num_reference_examples=num_reference_examples
        )
        self._similarity_measure = similarity_measure()

        # Load the spaCy model for parsing objects
        self._nlp = spacy.load("en_core_web_lg")

        # Setup the object detector if enabled
        self._detect_objects = detect_objects
        if self._detect_objects:
            from transformers import AutoImageProcessor, DetrForObjectDetection

            self._od_image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
            self._od_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
            self._od_threshold = 0.9
        else:
            self._od_image_processor = None
            self._od_model = None
            self._od_threshold = 0.0

        # Options
        self._min_object_length = min_object_length
        self._similarity_aggregate_method: Union[Literal["mean"], Literal["min"]] = similarity_aggregate_method
        self._debug = debug

        # Determine if we're using object detection categories
        self._coco = coco_objects

    def _extract_objects_from_image(self, image_path: str) -> List[str]:
        if self._detect_objects is False or self._od_model is None or self._od_image_processor is None:
            raise AssertionError("Object detection is not enabled")

        image = Image.open(image_path).convert("RGB")
        inputs = self._od_image_processor(images=image, return_tensors="pt")
        outputs = self._od_model(**inputs)  # type: ignore
        target_sizes = torch.tensor([image.size[::-1]])
        results = self._od_image_processor.post_process_object_detection(
            outputs, threshold=self._od_threshold, target_sizes=target_sizes
        )[0]

        # Convert labels to strings
        labels = []
        for _, label, _ in zip(results["scores"], results["labels"], results["boxes"]):
            labels.append(self._od_model.config.id2label[label.item()])  # type: ignore

        return labels

    def _parse_objects_from_llm_response(
        self,
        object_string: Optional[str],
        reference_mode: bool = False,
    ) -> Generator[List[str], None, None]:
        if object_string is None or not object_string.strip():
            return

        for obj in object_string.strip().split("\n"):
            # Handle some edge cases where the object is not formatted correctly
            if len(obj) < self._min_object_length or obj[0] != "-" or ("possibly" in obj and not reference_mode):
                continue

            # Remove the possibly from the object
            clean_obj = (
                obj.replace("(possibly)", "").strip().replace("possibly", "").strip() if "possibly" in obj else obj
            )
            # Remove the leading dash
            clean_obj = clean_obj[1:].strip()
            # Remove parentheticals if they exist
            clean_obj = clean_obj.split("(")[0].strip()

            # If it's a list of objects, we need to handle it differently
            _objs = clean_obj.split(",") if "," in clean_obj else [clean_obj]
            all_objects = []
            for o in _objs:
                all_objects += o.split(" or ") if " or " in o else [o]

            # Yield the nounds or the root nouns (depending on mode)
            yield [o.lower().strip() for o in all_objects if o]

            # If we're in reference mode, allow for root nouns
            if reference_mode or self._coco:
                nouns = []
                for o in all_objects:
                    # Remove any pluralization
                    _doc = self._nlp(o)
                    root_noun = next((token for token in _doc if token.head == token), None)
                    if root_noun is not None:
                        nouns += [root_noun.text.lower().strip()]
                yield nouns

    def _compute_alohao_scores(
        self,
        target_objects: List[str],
        reference_objects: List[str],
        aggregate_method: Union[Literal["mean"], Literal["min"]] = "min",
    ) -> Tuple[float, List[Dict[str, Any]]]:
        # If no target objects predicted, no hallucinations.
        if len(target_objects) == 0:
            return 1.0 if aggregate_method == "mean" else 0.0, []

        #  If no reference objects, but target objects predicted, all hallucinations
        if len(reference_objects) == 0:
            matches = [
                {"target_word": target_word, "ref_word": None, "similarity": 0} for target_word in target_objects
            ]
            return 0.0, matches

        # Compute the similarity matrix between the target and reference objects
        similarity = self._similarity_measure(target_objects, reference_objects)
        row_ind, col_ind = linear_sum_assignment(similarity, maximize=True)
        similarity_scores = similarity[row_ind, col_ind]

        matches = []
        for i, target_word in enumerate(target_objects):
            if i in row_ind:
                # Find reference word that target_word was matched to
                i_row_ind = np.where(np.array(row_ind) == i)[0][0]
                cind = col_ind[i_row_ind]
                ref_word = reference_objects[cind]
                target_sim = similarity_scores[i_row_ind]
                matches.append({"target_word": target_word, "ref_word": ref_word, "similarity": float(target_sim)})
            else:
                # Target wasn't matched to any reference word; more targets than references
                matches.append({"target_word": target_word, "ref_word": None, "similarity": float(0)})

        # Add entries of 0 similarity for every unmatched target word
        n_unmatched = len(target_objects) - len(similarity_scores)
        similarity_scores = similarity_scores + np.array([0]) * n_unmatched
        aggregate_score = float(np.mean(similarity_scores) if aggregate_method == "mean" else np.min(similarity_scores))

        return aggregate_score, matches

    def __call__(
        self,
        target: str,
        references: List[str],
        image_path: Optional[str] = None,
        object_detections: Optional[List[str]] = None,
    ) -> Tuple[float, ALOHaMetricInfoDict]:
        # Step 1: Parse objects from the target and reference captions
        unparsed_target_objects = self._object_parser.extract_objects_single_caption(target)
        unparsed_reference_objects = self._object_parser.extract_objects_multiple_captions(references)

        # Step 2: Extract objects from the unparsed objects with some filtering/parsing
        target_objects = list(self._parse_objects_from_llm_response(unparsed_target_objects))
        reference_objects = list(self._parse_objects_from_llm_response(unparsed_reference_objects, reference_mode=True))

        # Step 3: Merge with object detections if available and enabled
        if object_detections is not None:
            detections = [[item] for item in list(set(object_detections))]
            reference_objects += detections
        elif image_path is not None and self._detect_objects:
            # If we have an image path, we can use the object detector to extract objects
            detections = self._extract_objects_from_image(image_path)

        # Step 4: Compute the ALOHa score from each of the possible mappings
        best_score = float("-inf")
        best_matches: Optional[List[Dict[str, Any]]] = None

        for ftarget_ in _flatten(target_objects):
            for freferences_ in _flatten(reference_objects):
                # Remove duplicates
                ftarget = list(set(ftarget_))
                freferences = list(set(freferences_))

                # If in COCO mode, filter out any objects that are not in the COCO categories
                if self._coco:
                    ftarget = [f for f in ftarget if f in COCO_OBJECTS]
                    freferences = [f for f in freferences if f in COCO_OBJECTS]

                # Compute the pairwise ALOHa-O scores, and keep the best one
                similarity, matches = self._compute_alohao_scores(
                    ftarget, freferences, self._similarity_aggregate_method
                )
                if similarity > best_score:
                    best_score = similarity
                    best_matches = matches

        return best_score, {
            "unparsed_target_objects": unparsed_target_objects,
            "unparsed_reference_objects": unparsed_reference_objects,
            "target_objects": target_objects,
            "reference_objects": reference_objects,
            "matches": best_matches,
        }

    def evaluate_dataset(
        self,
        samples: List[Dict[str, Any]],
        candidate_key: str,
        reference_key: str,
        image_key: str,
        image_root_dir: str,
        annotation_root: str,
    ) -> List[Dict[str, Any]]:
        outputs = []

        with Progress() as progress:
            task = progress.add_task(f"Computing {self._name} scores...", total=len(samples))
            for sample in samples:
                if sample.get("metrics", {}).get(self._name, {}).get("score") is not None:
                    # Skip samples that have already been processed
                    outputs.append(sample)
                    continue

                # Setup the sample output
                if "metrics" not in sample:
                    sample["metrics"] = {}
                if self._name not in sample["metrics"]:
                    sample["metrics"][self._name] = {}

                # Extract the candidates, references, and image paths
                target = sample[candidate_key]
                references = sample[reference_key]
                image_path = os.path.join(image_root_dir, sample[image_key])
                scores, info = self(target, references, image_path, sample.get("detections", None))

                sample["metrics"][self._name]["score"] = scores
                if self._debug:
                    sample["metrics"][self._name]["info"] = info

                outputs.append(sample)
                progress.advance(task)

        return outputs

    def aggregate(self, samples: List[Dict[str, Any]]) -> Dict[str, float]:
        return {f"{self._name}": float(np.nanmean([sample["metrics"][self._name]["score"] for sample in samples]))}
