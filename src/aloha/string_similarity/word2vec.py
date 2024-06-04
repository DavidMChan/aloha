# Copyright (c) 2023 David Chan
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import List

import numpy as np

from aloha.string_similarity.base import StringSimilarityMeasure


class Word2VecSimilarity(StringSimilarityMeasure):
    def __init__(
        self,
    ):
        try:
            from gensim.models import KeyedVectors

            self._model = KeyedVectors.load_word2vec_format("word2vec.bin", binary=True)
        except ImportError:
            raise ImportError("Please install gensim to use Word2VecSimilarity")

    def sentence_vector(self, sentence: str):
        words = sentence.split()
        word_vectors = []

        for word in words:
            try:
                word_vectors.append(self._model[word])
            except KeyError:
                pass

        if word_vectors:
            return np.mean(word_vectors, axis=0)

        return np.zeros(self._model.vector_size)

    def __call__(self, strings_target: List[str], strings_reference: List[str]) -> np.ndarray:
        target_encodings = np.array([self.sentence_vector(s) for s in strings_target])
        reference_encodings = np.array([self.sentence_vector(s) for s in strings_reference])

        # Normalize non-zero vectors
        target_encodings = np.array(
            [vec / np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else vec for vec in target_encodings]
        )
        reference_encodings = np.array(
            [vec / np.linalg.norm(vec) if np.linalg.norm(vec) != 0 else vec for vec in reference_encodings]
        )

        similarity = target_encodings @ reference_encodings.T

        return similarity
