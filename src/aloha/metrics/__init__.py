from typing import Dict, Type

from aloha.metrics.aloha import ALOHa
from aloha.metrics.base import Metric
from aloha.metrics.chair import CHAIRMetrics
from aloha.metrics.clipscore import CLIPScoreMetrics
from aloha.object_parser import Claude3OpusObjectParser, GPT35TurboObjectParser, SpacyObjectParser
from aloha.string_similarity import (
    GPT3EmbeddingSimilarity,
    MPNetSimilarity,
    StringMatchingSimilarity,
    Word2VecSimilarity,
)
from aloha.utils import partialclass

METRICS: Dict[str, Type[Metric]] = {
    "aloha": partialclass(
        ALOHa,
        name="aloha",
        object_parser=GPT35TurboObjectParser,
        similarity_measure=MPNetSimilarity,
        num_reference_examples=3,
        num_target_examples=3,
        detect_objects=True,
    ),
    "aloha-w2v": partialclass(
        ALOHa,
        name="aloha-w2v",
        object_parser=GPT35TurboObjectParser,
        similarity_measure=Word2VecSimilarity,
        num_reference_examples=3,
        num_target_examples=3,
        detect_objects=True,
    ),
    "aloha-gpt3sim": partialclass(
        ALOHa,
        name="aloha-gpt3sim",
        object_parser=GPT35TurboObjectParser,
        similarity_measure=GPT3EmbeddingSimilarity,
        num_reference_examples=3,
        num_target_examples=3,
        detect_objects=True,
    ),
    "aloha-sm": partialclass(
        ALOHa,
        name="aloha-sm",
        object_parser=GPT35TurboObjectParser,
        similarity_measure=StringMatchingSimilarity,
        num_reference_examples=3,
        num_target_examples=3,
        detect_objects=True,
    ),
    "aloha-claude": partialclass(
        ALOHa,
        name="aloha-claude",
        object_parser=Claude3OpusObjectParser,
        similarity_measure=MPNetSimilarity,
        num_reference_examples=3,
        num_target_examples=3,
        detect_objects=True,
    ),
    "aloha-claude-w2v": partialclass(
        ALOHa,
        name="aloha-claude-w2v",
        object_parser=Claude3OpusObjectParser,
        similarity_measure=Word2VecSimilarity,
        num_reference_examples=3,
        num_target_examples=3,
        detect_objects=True,
    ),
    "aloha-claude-gpt3sim": partialclass(
        ALOHa,
        name="aloha-claude-gpt3sim",
        object_parser=Claude3OpusObjectParser,
        similarity_measure=GPT3EmbeddingSimilarity,
        num_reference_examples=3,
        num_target_examples=3,
        detect_objects=True,
    ),
    "aloha-claude-sm": partialclass(
        ALOHa,
        name="aloha-claude-sm",
        object_parser=Claude3OpusObjectParser,
        similarity_measure=StringMatchingSimilarity,
        num_reference_examples=3,
        num_target_examples=3,
        detect_objects=True,
    ),
    "aloha-spacy": partialclass(
        ALOHa,
        name="aloha-spacy",
        object_parser=SpacyObjectParser,
        similarity_measure=MPNetSimilarity,
        num_reference_examples=3,
        num_target_examples=3,
        detect_objects=True,
    ),
    "aloha-spacy-w2v": partialclass(
        ALOHa,
        name="aloha-spacy-w2v",
        object_parser=SpacyObjectParser,
        similarity_measure=Word2VecSimilarity,
        num_reference_examples=3,
        num_target_examples=3,
        detect_objects=True,
    ),
    "aloha-spacy-sm": partialclass(
        ALOHa,
        name="aloha-spacy-sm",
        object_parser=SpacyObjectParser,
        similarity_measure=StringMatchingSimilarity,
        num_reference_examples=3,
        num_target_examples=3,
        detect_objects=True,
    ),
    "aloha-spacy-gpt3sim": partialclass(
        ALOHa,
        name="aloha-spacy-gpt3sim",
        object_parser=SpacyObjectParser,
        similarity_measure=GPT3EmbeddingSimilarity,
        num_reference_examples=3,
        num_target_examples=3,
        detect_objects=True,
    ),
    "chair": CHAIRMetrics,
    "chair-nocaps": partialclass(CHAIRMetrics, nocaps=True, is_coco=False, name="chair-nocaps"),
    "chair-noncoco": partialclass(CHAIRMetrics, is_coco=False, name="chair-noncoco"),
    "clipscore": CLIPScoreMetrics,
}
