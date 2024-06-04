from typing import List, TypedDict


class Sample(TypedDict):
    image_path: str
    candidate: str
    references: List[str]
