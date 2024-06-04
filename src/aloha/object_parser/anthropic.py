import os
from typing import List, Optional

import anthropic
from anthropic.types.message_param import MessageParam

from aloha.object_parser.base import ObjectParser
from aloha.object_parser.prompts import (
    MULTI_OBJECT_EXAMPLES,
    MULTI_OBJECT_SYSTEM_PROMPT,
    SINGLE_OBJECT_EXAMPLES,
    SINGLE_OBJECT_SYSTEM_PROMPT,
)


class AnthropomorphicObjectParser(ObjectParser):
    def __init__(
        self,
        num_target_examples: Optional[int] = 3,
        num_reference_examples: Optional[int] = 3,
        model: str = "claude-3-opus",
    ):
        super().__init__(num_target_examples, num_reference_examples)

        self._anthropic_client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY", None))
        self._model = model

    def extract_objects_single_caption(self, caption: str) -> str:
        messages: List[MessageParam] = []
        for idx in range(self._num_target_examples):
            user, system = SINGLE_OBJECT_EXAMPLES[idx]
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": system})
        messages.append({"role": "user", "content": f"Caption: {caption}\nObjects:"})

        message = self._anthropic_client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=0.0,
            system=SINGLE_OBJECT_SYSTEM_PROMPT,
            messages=messages,
        )

        if message.content[0].type != "text":
            raise ValueError(f"Unexpected message type: {message.content[0].type}")

        return message.content[0].text

    def extract_objects_multiple_captions(self, captions: List[str]) -> str:
        messages: List[MessageParam] = []
        for idx in range(self._num_reference_examples):
            user, system = MULTI_OBJECT_EXAMPLES[idx]
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": system})

        # Add the caption to the end of the messages
        formatted_captions = "\n".join(f"- {caption}" for caption in captions)
        messages.append({"role": "user", "content": f"Captions:\n{formatted_captions}\nObjects:"})

        message = self._anthropic_client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=0.0,
            system=MULTI_OBJECT_SYSTEM_PROMPT,
            messages=messages,
        )

        if message.content[0].type != "text":
            raise ValueError(f"Unexpected message type: {message.content[0].type}")

        return message.content[0].text


class Claude3OpusObjectParser(AnthropomorphicObjectParser):
    def __init__(
        self,
        num_target_examples: Optional[int] = 3,
        num_reference_examples: Optional[int] = 3,
    ):
        super().__init__(num_target_examples, num_reference_examples, model="claude-3-opus-20240229")


class Claude3SonnetObjectParser(AnthropomorphicObjectParser):
    def __init__(
        self,
        num_target_examples: Optional[int] = 3,
        num_reference_examples: Optional[int] = 3,
    ):
        super().__init__(num_target_examples, num_reference_examples, model="claude-3-sonnet-20240229")


class Claude3HaikuObjectParser(AnthropomorphicObjectParser):
    def __init__(
        self,
        num_target_examples: Optional[int] = 3,
        num_reference_examples: Optional[int] = 3,
    ):
        super().__init__(num_target_examples, num_reference_examples, model="claude-3-haiku-20240307")
