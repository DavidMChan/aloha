# Copyright (c) 2023 David Chan
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import List

import numpy as np
import openai

from aloha.string_similarity.base import StringSimilarityMeasure
from aloha.utils import retry


class GPT3EmbeddingSimilarity(StringSimilarityMeasure):
    def __init__(self, model="text-embedding-3-small"):
        self._model = model
        self._client = openai.OpenAI()

    @retry
    def get_embedding(self, text: str) -> np.ndarray:
        text = text.replace("\n", " ")
        return np.array(self._client.embeddings.create(input=text, model=self._model).data[0].embedding)

    def __call__(self, strings_target: List[str], strings_reference: List[str]) -> np.ndarray:
        target_encodings = np.array([self.get_embedding(s) for s in strings_target])
        reference_encodings = np.array([self.get_embedding(s) for s in strings_reference])

        similarity = target_encodings @ reference_encodings.T

        return similarity
