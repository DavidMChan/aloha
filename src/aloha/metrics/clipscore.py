"""
This code is modified directly from:
https://github.com/jmhessel/clipscore/blob/main/clipscore.py

With credit to the CLIPScore paper authors:
Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, Yejin Choi.
CLIPScore: A Reference-free Evaluation Metric for Image Captioning. In EMNLP 2021.
"""

import collections
import logging
import os
from typing import Dict, List

try:
    import clip
except ImportError:
    clip = None


import numpy as np
import torch
import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from aloha.metrics.base import Metric
from aloha.types import Sample


class CLIPImageDataset(Dataset):
    def __init__(self, data, preprocess):
        self.data = data
        self.preprocess = preprocess

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {"image": image}

    def __len__(self):
        return len(self.data)


class CLIPCapDataset(Dataset):
    def __init__(self, data, prefix="A photo depicts"):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != " ":
            self.prefix += " "

    def __getitem__(self, idx):
        if clip is None:
            raise ImportError("CLIPScoreMetrics requires the `clip` package to be installed.")

        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {"caption": c_data}

    def __len__(self):
        return len(self.data)


def extract_all_images(images, model, preprocess, device, batch_size=64, num_workers=8):
    data = DataLoader(
        CLIPImageDataset(images, preprocess), batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    all_image_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b["image"].to(device)  # noqa: PLW2901
            if device == "cuda":
                b = b.to(torch.float16)  # noqa: PLW2901
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def extract_all_captions(captions, model, device, batch_size=256, num_workers=8):
    data = DataLoader(CLIPCapDataset(captions), batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm.tqdm(data):
            b = b["caption"].to(device)  # noqa: PLW2901
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def get_clip_score(images, model, preprocess, candidates, device, w=2.5):
    """
    Get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    """
    if isinstance(images, list):
        # Need to extract image features
        images = extract_all_images(images, model, preprocess, device)

    candidates = extract_all_captions(candidates, model, device)

    images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
    candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))

    per = w * np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates


def get_refonlyclipscore(model, references, candidates, device):
    """
    The text only side for refclipscore
    """
    if isinstance(candidates, list):
        candidates = extract_all_captions(candidates, model, device)

    flattened_refs = []
    flattened_refs_idxs = []
    for idx, refs in enumerate(references):
        flattened_refs.extend(refs)
        flattened_refs_idxs.extend([idx for _ in refs])

    print(flattened_refs)

    flattened_refs = extract_all_captions(flattened_refs, model, device)

    candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))
    flattened_refs = flattened_refs / np.sqrt(np.sum(flattened_refs**2, axis=1, keepdims=True))

    cand_idx2refs = collections.defaultdict(list)
    for ref_feats, cand_idx in zip(flattened_refs, flattened_refs_idxs):
        cand_idx2refs[cand_idx].append(ref_feats)

    assert len(cand_idx2refs) == len(candidates)

    cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}

    per = []
    for c_idx, cand in tqdm.tqdm(enumerate(candidates)):
        cur_refs = cand_idx2refs[c_idx]
        all_sims = cand.dot(cur_refs.transpose())
        per.append(np.max(all_sims))

    return np.mean(per), per


def _cm_kys(
    samples: List[Sample],
    candidate_key: str,
    reference_key: str,
    image_key: str,
    image_root_dir: str,
    name: str = "CLIPScore",
) -> List[Sample]:
    """
    Compute and add CHAIR metrics to samples.

    Args:
        samples: List of samples.
        reference_key: Key of reference sentences in samples.
        candidate_key: Key of candidate sentences in samples.
        image_key: Key of COCO-format file names in samples (e.g., file name could be "COCO_val2014_000000360772.jpg").
        image_root_dir: Path to directory containing images.
        name: Name of metric.

    Returns:
        List of samples with CLIPScore metrics added.
    """

    if clip is None:
        raise ImportError("CLIPScoreMetrics requires the `clip` package to be installed.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # clip_type = "RN50x64"
    clip_type = "ViT-B/32"
    model, transform = clip.load(clip_type, device=device, jit=False)
    model.eval()

    image_paths = [os.path.join(image_root_dir, sample[image_key]) for sample in samples]
    image_features = extract_all_images(image_paths, model, transform, device)

    # Get image-text clipscore
    candidates = [sample[candidate_key] for sample in samples]
    _, per_instance_image_text, candidate_features = get_clip_score(
        image_features, model, transform, candidates, device
    )
    clipscores = per_instance_image_text

    # Get text-text clipscore
    references = [sample[reference_key] for sample in samples]
    _, per_instance_text_text = get_refonlyclipscore(model, references, candidate_features, device)
    # F-score
    refclipscores = (
        2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)
    )

    # Add scores to each sample
    for i, sample in enumerate(samples):
        if "metrics" not in sample:
            sample["metrics"] = {}
        if name not in sample["metrics"]:
            sample["metrics"][name] = {}
        sample["metrics"][name]["CLIPScore"] = float(clipscores[i])
        sample["metrics"][name]["RefCLIPScore"] = float(refclipscores[i])

    return samples


class CLIPScoreMetrics(Metric):
    def __init__(self) -> None:
        if clip is None:
            raise ImportError("CLIPScoreMetrics requires the `openai-clip` package to be installed.")

    def evaluate_dataset(
        self,
        samples: List[Sample],
        candidate_key: str,
        reference_key: str,
        image_key: str,
        image_root_dir: str,
        annotation_root: str,
    ) -> List[Sample]:
        """
        Compute and add CLIPScore metrics to samples.

        Args:
            samples: List of samples.
            reference_key: Key of reference sentences in samples.
            candidate_key: Key of candidate sentences in samples.
            image_key: Key of COCO-format file names in samples (e.g., file name could be "COCO_val2014_000000360772.jpg").
            image_root_dir: Path to directory containing images.

        Returns:
            List of samples with CLIPScore metrics added.
        """
        return _cm_kys(samples, candidate_key, reference_key, image_key, image_root_dir)

    def aggregate(self, samples: List[Sample]) -> Dict[str, float]:
        output = {
            "CLIPScore": float(sum([d["metrics"]["CLIPScore"]["CLIPScore"] for d in samples])) / (len(samples) + 1e-12),
            "RefCLIPScore": float(sum([d["metrics"]["CLIPScore"]["CLIPScore"] for d in samples]))
            / (len(samples) + 1e-12),
        }
        for k, v in output.items():
            logging.info(f"{k}: {np.round(100*v, 2)}")
        return output
