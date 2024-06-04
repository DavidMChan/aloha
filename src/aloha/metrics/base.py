from abc import ABC, abstractmethod
from typing import Dict, List

from aloha.types import Sample


class Metric(ABC):
    @abstractmethod
    def evaluate_dataset(
        self,
        samples: List[Sample],
        candidate_key: str,
        reference_key: str,
        image_key: str,
        image_root_dir: str,
        annotation_root: str,
    ) -> List[Sample]:
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, samples: List[Sample]) -> Dict[str, float]:
        raise NotImplementedError
