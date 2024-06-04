import json
import logging
from typing import Dict, List, Union

import click
from rich.logging import RichHandler

from aloha.metrics import METRICS
from aloha.types import Sample
from aloha.utils import add_coco_gt_detections, add_object_detections


@click.command()
@click.argument("dataset-path", type=str, required=True)
@click.option("-m", "--metric", "metrics", type=click.Choice(list(METRICS.keys())), multiple=True)
@click.option("--overwrite-cached-metrics/--no-overwrite-cached-metrics", default=False)
@click.option("--output-path", type=str, default="output.json")
@click.option("--candidate-key", type=str, default="baseline")
@click.option("--reference-key", type=str, default="references")
@click.option("--image-key", type=str, default="image_path")
@click.option("--image-root-dir", type=str, default="")
@click.option("--annotation-root-dir", type=str, default=None)
@click.option("--add-detections", type=bool, default=False)
@click.option("--add-coco-gt", type=bool, default=False)
@click.option("--debug", is_flag=True, default=False)
def evaluate_dataset(
    dataset_path: str,
    metrics: List[str],
    overwrite_cached_metrics: bool = False,
    output_path: str = "output.json",
    candidate_key: str = "caption",
    reference_key: str = "references",
    image_key: str = "image_path",
    image_root_dir: str = "",
    annotation_root_dir: str = "",
    add_detections: bool = False,
    add_coco_gt: bool = False,
    debug: bool = False,
) -> None:
    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )

    # 1. Load the dataset
    with open(dataset_path) as f:
        logging.info(f"Loading dataset from {dataset_path}")
        dataset: Union[List[Sample], Dict[str, List[Sample]]] = json.load(f)
        samples: List[Sample] = []
        if isinstance(dataset, dict):
            if "samples" in dataset:
                samples = dataset["samples"]
                output_metrics: Dict[str, float] = dataset.get("metrics", {})  # type: ignore
            else:
                logging.error("Dataset must contain a 'samples' key")
                return
        else:
            samples = dataset
            output_metrics = {}

    # 1.5. Add object detections
    if add_detections:
        logging.info("Adding object detections to samples")
        samples = add_object_detections(
            samples,
            image_root_dir,
            image_key=image_key,
        )
    elif add_coco_gt:
        logging.info("Adding COCO ground truth to samples")
        samples = add_coco_gt_detections(
            samples,
            image_key=image_key,
            annotations_path=annotation_root_dir,
        )

    # 2. For each metric, compute the metric, and update the samples with the metric
    for metric in metrics:
        logging.info(f"Computing metric: {metric}")
        if metric in output_metrics and not overwrite_cached_metrics:
            logging.info(f"Skipping metric: {metric}, as it is already computed")
            continue

        # Grab the metric function
        _mf = METRICS.get(metric, None)
        if _mf is None:
            logging.warning(f"Metric {metric} not found, skipping")
            continue
        else:
            _mf = _mf()

        # Compute the metric
        samples = _mf.evaluate_dataset(
            samples,
            candidate_key=candidate_key,
            reference_key=reference_key,
            image_key=image_key,
            image_root_dir=image_root_dir,
            annotation_root=annotation_root_dir,
        )
        aggregated_metrics = _mf.aggregate(samples)
        output_metrics |= aggregated_metrics

    # 3. Write the output
    logging.info(f"Writing output to {output_path}")
    with open(output_path, "w") as jf:
        json.dump(
            {
                "samples": samples,
                "metrics": output_metrics,
            },
            jf,
            indent=2,
        )
