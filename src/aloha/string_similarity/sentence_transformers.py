# Copyright (c) 2023 David Chan
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from aloha.string_similarity.base import StringSimilarityMeasure


class MPNetSimilarity(StringSimilarityMeasure):
    def __init__(
        self,
    ):
        self._model = SentenceTransformer("all-mpnet-base-v2")

    def __call__(self, strings_target: List[str], strings_reference: List[str]) -> np.ndarray:
        reference_encodings = self._model.encode(strings_reference, show_progress_bar=False)
        target_encodings = self._model.encode(strings_target, show_progress_bar=False)

        similarity = target_encodings @ reference_encodings.T

        return similarity
