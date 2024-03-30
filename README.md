# ALOHa: A New Measure for Hallucination in Captioning Models

### [Project](https://davidmchan.github.io/aloha/) | [Paper]()

Official implementation of the paper: "ALOHa: A New Measure for Hallucination in Captioning Models".
<br>

Despite recent advances in multimodal pre-training for visual description, state-of-the-art models still produce captions containing errors, such as hallucinating objects not present in a scene. The existing prominent metric for object hallucination, CHAIR, is limited to a fixed set of MS COCO objects and synonyms. In this work, we propose a modernized open-vocabulary metric, ALOHa, which leverages large language models (LLMs) to measure object hallucinations. Specifically, we use an LLM to extract groundable objects from a candidate caption, measure their semantic similarity to reference objects from captions and object detections, and use Hungarian matching to produce a final hallucination score. We show that ALOHa correctly identifies 13.6\% more hallucinated objects than CHAIR on HAT, a new gold-standard subset of MS COCO Captions annotated for hallucinations, and 30.8% more on nocaps, where objects extend beyond MS COCO categories.

## Getting started

### Setup

[COMING SOON]

### Usage

[COMING SOON]

## Citation

If you find this repository useful, please cite our paper:

```bibtex
@inproceedings{petryk2024aloha,
    title = "ALOHa: A New Measure for Hallucination in Captioning Models",
    author = "Petryk, Suzanne and
        Chan, David M and
        Kachinthaya, Anish and
        Zou, Haodi and
        Canny, John and
        Gonzalez, Joseph E and
        Darrell, Trevor",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    year = "2024",
    publisher = "Association for Computational Linguistics",
}
```
