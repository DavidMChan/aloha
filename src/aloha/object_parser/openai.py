from typing import List, Optional

from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

from aloha.object_parser.base import ObjectParser
from aloha.object_parser.prompts import (
    MULTI_OBJECT_EXAMPLES,
    MULTI_OBJECT_SYSTEM_PROMPT,
    SINGLE_OBJECT_EXAMPLES,
    SINGLE_OBJECT_SYSTEM_PROMPT,
)


class OpenAIObjectParser(ObjectParser):
    def __init__(
        self,
        num_target_examples: Optional[int] = 3,
        num_reference_examples: Optional[int] = 3,
        model: str = "gpt-3.5-turbo",
    ):
        super().__init__(num_target_examples, num_reference_examples)

        self._client = OpenAI()
        self._model = model

    def extract_objects_single_caption(self, caption: str) -> str:
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": SINGLE_OBJECT_SYSTEM_PROMPT},
        ]
        for idx in range(self._num_target_examples):
            user, system = SINGLE_OBJECT_EXAMPLES[idx]
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": system})

        # Add the caption to the end of the messages
        messages.append({"role": "user", "content": f"Caption: {caption}\nObjects:"})

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.0,
        )

        return completion.choices[0].message.content or ""

    def extract_objects_multiple_captions(self, captions: List[str]) -> str:
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": MULTI_OBJECT_SYSTEM_PROMPT},
        ]
        for idx in range(self._num_reference_examples):
            user, system = MULTI_OBJECT_EXAMPLES[idx]
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": system})

        # Add the caption to the end of the messages
        formatted_captions = "\n".join(f"- {caption}" for caption in captions)
        messages.append({"role": "user", "content": f"Captions:\n{formatted_captions}\nObjects:"})

        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.0,
        )
        return completion.choices[0].message.content or ""


class GPT35TurboObjectParser(OpenAIObjectParser):
    def __init__(self, num_target_examples: Optional[int] = 3, num_reference_examples: Optional[int] = 3):
        super().__init__(num_target_examples, num_reference_examples, model="gpt-3.5-turbo")


class GPT4TurboObjectParser(OpenAIObjectParser):
    def __init__(self, num_target_examples: Optional[int] = 3, num_reference_examples: Optional[int] = 3):
        super().__init__(num_target_examples, num_reference_examples, model="gpt-4-turbo")


class GPT4ObjectParser(OpenAIObjectParser):
    def __init__(self, num_target_examples: Optional[int] = 3, num_reference_examples: Optional[int] = 3):
        super().__init__(num_target_examples, num_reference_examples, model="gpt-4")


class GPT4OObjectParser(OpenAIObjectParser):
    def __init__(self, num_target_examples: Optional[int] = 3, num_reference_examples: Optional[int] = 3):
        super().__init__(num_target_examples, num_reference_examples, model="gpt-4o")
