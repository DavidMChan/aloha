# Copyright (c) 2023 David Chan
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import List

import numpy as np

from aloha.string_similarity.base import StringSimilarityMeasure


class StringMatchingSimilarity(StringSimilarityMeasure):
    def __call__(self, strings_target: List[str], strings_reference: List[str]) -> np.ndarray:
        target_encodings = np.array([s.lower() for s in strings_target])
        reference_encodings = np.array([s.lower() for s in strings_reference])

        similarity = np.array(
            [[1 if target == reference else 0 for reference in reference_encodings] for target in target_encodings]
        )

        return similarity
