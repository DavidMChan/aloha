# ALOHa: A New Measure for Hallucination in Captioning Models

### [Project](https://davidmchan.github.io/aloha/) | [Paper](https://arxiv.org/abs/2404.02904)

Official implementation of the paper: ["ALOHa: A New Measure for Hallucination in Captioning Models"](https://arxiv.org/abs/2404.02904).
<br>

Despite recent advances in multimodal pre-training for visual description, state-of-the-art models still produce captions containing errors, such as hallucinating objects not present in a scene. The existing prominent metric for object hallucination, CHAIR, is limited to a fixed set of MS COCO objects and synonyms. In this work, we propose a modernized open-vocabulary metric, ALOHa, which leverages large language models (LLMs) to measure object hallucinations. Specifically, we use an LLM to extract groundable objects from a candidate caption, measure their semantic similarity to reference objects from captions and object detections, and use Hungarian matching to produce a final hallucination score. We show that ALOHa correctly identifies 13.6\% more hallucinated objects than CHAIR on HAT, a new gold-standard subset of MS COCO Captions annotated for hallucinations, and 30.8% more on nocaps, where objects extend beyond MS COCO categories.

## Getting started

### Setup

```bash

# Install this package from github
pip install git+https://github.com/DavidMChan/aloha.git

# Install the Spacy model if you haven't already
pip install -U spacy
python -m spacy download en_core_web_lg
```

### Usage

To compute the ALOHa score for a single caption:

```python
from aloha.metrics import ALOHa
from aloha.object_parser import GPT35TurboObjectParser
from aloha.string_similarity import MPNetSimilarity

# Initialize the ALOHa metric
evaluator = ALOHa(
    name="aloha",
    object_parser=GPT35TurboObjectParser,
    similarity_measure=MPNetSimilarity,
    num_reference_examples=3,
    num_target_examples=3,
    detect_objects=True,
)

candidate_caption = "A cat is sitting on a table"
reference_captions = ["A dog is sitting on a table", "A hound is sitting on a table"]
optional_image_path = None
optional_precomputed_detections = None

# Compute the ALOHa score
score, matches = evaluator(
    target=candidate_caption,
    references=reference_captions,
    image_path=optional_image_path,
    object_detections=optional_precomputed_detections,
)

print(score)
# 0.6081229448318481

print(matches)
# {'matches': [{'ref_word': 'table', 'similarity': 1.0, 'target_word': 'table'},
#              {'ref_word': 'dog',
#               'similarity': 0.6081229448318481,
#               'target_word': 'cat'}],
#  'reference_objects': [['dog'],
#                        ['dog'],
#                        ['table'],
#                        ['table'],
#                        ['hound'],
#                        ['hound']],
#  'target_objects': [['cat'], ['table']],
#  'unparsed_reference_objects': '- dog\n- table\n- hound',
#  'unparsed_target_objects': '- cat\n- table'}
```

To compute it for a full dataset of samples, you can use the `evaluate-dataset` script. First, prepare your dataset in
a JSON file with the following format:

```json
[
    {
        "caption": "A caption",
        "references": ["Ref 1", "Ref 2", ...],
        "image_path": "path/to/image.jpg",
    },
    ...
]
```

Then, run the following command:

```bash
aloha evaluate-dataset -m aloha path/to/dataset.json
```

The above command has many options to customize the evaluation. You can see them by running:

```bash
aloha evaluate-dataset --help
```

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
