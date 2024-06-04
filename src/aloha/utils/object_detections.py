import os
from typing import List

import torch
from PIL import Image
from rich.progress import Progress
from transformers import AutoImageProcessor, DetrForObjectDetection

from aloha.types import Sample


def add_object_detections(
    samples: List[Sample],
    image_root_dir: str,
    image_key: str = "image_path",
) -> List[Sample]:
    image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

    # 1. Load the images
    with Progress() as progress:
        task = progress.add_task("Retrieving object detections...", total=len(samples))
        for i, sample in enumerate(samples):
            image_path = os.path.join(image_root_dir, sample[image_key])

            image = Image.open(image_path).convert("RGB")

            inputs = image_processor(images=image, return_tensors="pt")

            outputs = model(**inputs)  # type: ignore

            target_sizes = torch.tensor([image.size[::-1]])
            results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
                0
            ]

            # Convert labels to strings
            labels = []
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                labels.append(model.config.id2label[label.item()])  # type: ignore

            samples[i]["detections"] = labels
            progress.advance(task)

    return samples
