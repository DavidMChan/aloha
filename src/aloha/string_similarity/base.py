# Copyright (c) 2023 David Chan
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


from abc import ABC, abstractmethod
from typing import List

import numpy as np


class StringSimilarityMeasure(ABC):
    @abstractmethod
    def __call__(self, strings_target: List[str], strings_reference: List[str]) -> np.ndarray:
        # Generate the pairwise similarity matrix between the target and reference strings
        raise NotImplementedError
