from abc import ABC, abstractmethod
from typing import List, Optional

from aloha.object_parser.prompts import (
    MULTI_OBJECT_EXAMPLES,
    SINGLE_OBJECT_EXAMPLES,
)


class ObjectParser(ABC):
    def __init__(self, num_target_examples: Optional[int] = 3, num_reference_examples: Optional[int] = 3):
        # Validate and set the number of reference/target examples for the prompting
        if num_reference_examples and num_reference_examples > len(MULTI_OBJECT_EXAMPLES):
            raise ValueError(
                f"Number of requested reference examples {num_reference_examples} is greater than the number of examples {len(MULTI_OBJECT_EXAMPLES)}"
            )
        if num_target_examples and num_target_examples > len(SINGLE_OBJECT_EXAMPLES):
            raise ValueError(
                f"Number of requested target examples {num_target_examples} is greater than the number of examples {len(SINGLE_OBJECT_EXAMPLES)}"
            )

        self._num_reference_examples = num_reference_examples or len(MULTI_OBJECT_EXAMPLES)
        self._num_target_examples = num_target_examples or len(SINGLE_OBJECT_EXAMPLES)

    @abstractmethod
    def extract_objects_single_caption(self, caption: str) -> str:
        pass

    @abstractmethod
    def extract_objects_multiple_captions(self, captions: List[str]) -> str:
        pass
