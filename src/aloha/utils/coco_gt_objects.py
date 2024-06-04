from typing import List

from pycocotools.coco import COCO
from rich.progress import Progress

from aloha.types import Sample


def add_coco_gt_detections(
    samples: List[Sample],
    image_key: str = "image_path",
    annotations_path: str = "annotations",
) -> List[Sample]:
    coco = COCO(f"{annotations_path}/instances_val2014.json")

    # 1. Load the images
    with Progress() as progress:
        task = progress.add_task("Retrieving object detections...", total=len(samples))
        for i, sample in enumerate(samples):
            id_ = int(sample[image_key].split("_")[-1].split(".")[0])
            anns = coco.loadAnns(coco.getAnnIds(imgIds=[id_]))
            gt_objects = [coco.loadCats([ann["category_id"]])[0]["name"] for ann in anns]

            samples[i]["detections"] = gt_objects
            progress.advance(task)

    return samples


COCO_OBJECTS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]
