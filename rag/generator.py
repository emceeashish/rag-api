"""OpenAI Chat Completions wrapper (async, streaming + non-streaming) with error handling."""

import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI, APIError, APIConnectionError, RateLimitError

logger = logging.getLogger(__name__)


class GeneratorError(Exception):
    """Raised when OpenAI generation fails."""


class Generator:
    """Async wrapper around OpenAI Chat Completions."""

    def __init__(self, model: str = "gpt-3.5-turbo") -> None:
        self.client = AsyncOpenAI()  # reads OPENAI_API_KEY from env
        self.model = model

    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Non-streaming completion.  Returns the full answer string."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0,
            )
            return response.choices[0].message.content or ""
        except RateLimitError as exc:
            logger.error("OpenAI rate limit hit: %s", exc)
            raise GeneratorError("OpenAI rate limit exceeded. Please retry later.") from exc
        except APIConnectionError as exc:
            logger.error("OpenAI connection error: %s", exc)
            raise GeneratorError("Could not connect to OpenAI API.") from exc
        except APIError as exc:
            logger.error("OpenAI API error: %s", exc)
            raise GeneratorError(f"OpenAI error: {exc.message}") from exc

    async def generate_stream(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """Streaming completion.  Yields content tokens as they arrive."""
        try:
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=0,
                stream=True,
            )
            async for event in stream:
                delta = event.choices[0].delta
                if delta.content:
                    yield delta.content
        except RateLimitError as exc:
            logger.error("OpenAI rate limit hit (stream): %s", exc)
            raise GeneratorError("OpenAI rate limit exceeded. Please retry later.") from exc
        except APIConnectionError as exc:
            logger.error("OpenAI connection error (stream): %s", exc)
            raise GeneratorError("Could not connect to OpenAI API.") from exc
        except APIError as exc:
            logger.error("OpenAI API error (stream): %s", exc)
            raise GeneratorError(f"OpenAI error: {exc.message}") from exc
