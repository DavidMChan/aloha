from typing import List, Optional

import spacy

from aloha.object_parser.base import ObjectParser
from aloha.utils.pattern import singularize


class SpacyObjectParser(ObjectParser):
    def __init__(
        self,
        num_target_examples: Optional[int] = 3,
        num_reference_examples: Optional[int] = 3,
        model: str = "en_core_web_lg",
    ):
        super().__init__(num_target_examples, num_reference_examples)

        self._model = spacy.load(model)

    def extract_objects_single_caption(self, caption: str) -> str:
        # Extract all of the nouns from the caption
        doc = self._model(caption)
        nouns = [token.text for token in doc if token.pos_ == "NOUN"]

        # Singularize all of the nouns
        return "\n".join(f"- {singularize(noun)}" for noun in nouns)

    def extract_objects_multiple_captions(self, captions: List[str]) -> str:
        # Extract all of the nouns from the caption
        docs = [self._model(caption) for caption in captions]
        all_nouns = []
        for doc in docs:
            nouns = [token.text for token in doc if token.pos_ == "NOUN"]
            all_nouns.extend(nouns)

        # Singularize all of the nouns
        return "\n".join(f"- {singularize(noun)}" for noun in all_nouns)
