"""
Vertex AI completions using Google GenAI SDK.

Uses the new google-genai SDK with Vertex AI endpoint for better performance
and rate limits compared to the OpenAI-compatible Gemini API.

Authentication:
    - Google Cloud credentials (Application Default Credentials)
    - Service account key file (GOOGLE_APPLICATION_CREDENTIALS)
    - gcloud auth application-default login

Environment variables:
    - GOOGLE_APPLICATION_CREDENTIALS: Path to service account key file
    - GOOGLE_CLOUD_PROJECT: Google Cloud project ID
    - GOOGLE_CLOUD_LOCATION: Vertex AI location (e.g., us-central1)
"""

import copy
import os
import logging
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union

from lm_eval.api.registry import register_model
from lm_eval.api.model import LM
from lm_eval.models.api_models import TemplateAPI
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.utils import eval_logger

try:
    from google import genai
    from google.genai import types
    from google.api_core import exceptions as google_exceptions
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False


def _handle_vertex_errors(e: Exception) -> str:
    """Convert Vertex AI exceptions to error messages."""
    if isinstance(e, google_exceptions.InvalidArgument):
        return f"Invalid argument: {e.message}"
    elif isinstance(e, google_exceptions.NotFound):
        return f"Not found: {e.message}"
    elif isinstance(e, google_exceptions.PermissionDenied):
        return f"Permission denied: {e.message}"
    elif isinstance(e, google_exceptions.ResourceExhausted):
        return f"Rate limit exceeded: {e.message}"
    elif isinstance(e, google_exceptions.ServiceUnavailable):
        return f"Service unavailable: {e.message}"
    elif isinstance(e, google_exceptions.DeadlineExceeded):
        return f"Deadline exceeded: {e.message}"
    else:
        return f"Unexpected error: {str(e)}"


@register_model("vertex-chat-completions")
class VertexChatCompletion(LocalCompletionsAPI):
    """
    Vertex AI Chat Completions using Google GenAI SDK.

    Uses Vertex AI endpoint which has better performance and rate limits
    than the OpenAI-compatible Gemini API.
    """

    def __init__(
        self,
        project: Optional[str] = None,
        location: Optional[str] = None,
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        if not HAS_GENAI:
            raise ModuleNotFoundError(
                "google-genai package is required for Vertex AI. "
                "Please install it with: pip install google-genai"
            )

        # Get project and location from env or kwargs
        self._project = project or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self._location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

        if not self._project:
            raise ValueError(
                "GOOGLE_CLOUD_PROJECT must be set as environment variable "
                "or passed as project= parameter"
            )

        eval_logger.info(f"Using Vertex AI: project={self._project}, location={self._location}")

        # Store client config for creating new clients per request
        # The GenAI SDK client may not be thread-safe, so we create new clients per request
        self._client_config = {
            "vertexai": True,
            "project": self._project,
            "location": self._location,
        }

        # Create a client for initial validation
        try:
            genai.Client(**self._client_config)
        except Exception as e:
            raise ValueError(
                f"Failed to initialize Vertex AI client: {e}\n"
                "Make sure you have authenticated with:\n"
                "  - gcloud auth application-default login\n"
                "  - Or set GOOGLE_APPLICATION_CREDENTIALS to service account key file"
            )

        # Call parent init but override base_url (not used for Vertex AI)
        # Set tokenizer_backend=None to skip HuggingFace tokenizer loading
        kwargs.pop("base_url", None)  # Remove base_url if provided
        kwargs["tokenizer_backend"] = tokenizer_backend
        kwargs["tokenized_requests"] = tokenized_requests
        super().__init__(base_url="vertex://unused", **kwargs)

        # Disable batching for chat completions
        if self._batch_size > 1:
            eval_logger.warning(
                "Chat completions does not support batching. Defaulting to batch size 1."
            )
            self._batch_size = 1

    @cached_property
    def api_key(self):
        """Vertex AI doesn't use API keys like OpenAI."""
        return "vertex_ai"

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos=None,
        **kwargs,
    ) -> dict:
        """
        Create payload for Vertex AI API.

        Vertex AI uses different parameter names than OpenAI:
        - max_completion_tokens instead of max_tokens
        - Supports temperature, top_p, top_k
        - Does NOT support seed parameter
        """
        assert (
            type(messages) is not str
        ), "chat-completions require the --apply_chat_template flag."

        gen_kwargs = copy.deepcopy(gen_kwargs) if gen_kwargs else {}
        gen_kwargs.pop("do_sample", False)

        temperature = gen_kwargs.pop("temperature", 0)
        top_p = gen_kwargs.pop("top_p", None)
        top_k = gen_kwargs.pop("top_k", None)

        # Handle stop sequences
        from lm_eval.models.openai_completions import handle_stop_sequences
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]

        # Remove unsupported parameters
        gen_kwargs.pop("max_gen_toks", None)
        gen_kwargs.pop("max_tokens", None)

        # Build config
        config = {
            "temperature": temperature,
        }
        if top_p is not None:
            config["top_p"] = top_p
        if top_k is not None:
            config["top_k"] = top_k

        # Only add stop if non-empty
        if stop and any(s for s in stop[:4] if s):
            config["stop_sequences"] = stop[:4]

        return {
            "messages": messages,
            "config": config,
            "gemini_config": gen_kwargs,  # Pass any remaining kwargs as Gemini config
        }

    @staticmethod
    def _extract_final_code_block(content: str) -> str:
        """
        Extract only the final code block from Gemini 2.5 Pro responses.

        Gemini 2.5 Pro includes reasoning/thought process sections that may contain
        example code blocks. We want to return only the final/last code block which
        contains the actual implementation.

        Args:
            content: The full response content

        Returns:
            The content filtered to only include the last code block, or original
            content if no code blocks are found.
        """
        import re

        # Find all code blocks with language specification
        # Pattern matches: ```python or ``` followed by code, then ```
        code_block_pattern = r'```(?:python|py)?\n(.*?)\n```'
        matches = list(re.finditer(code_block_pattern, content, re.DOTALL))

        if not matches:
            # No code blocks found, return original content
            return content

        if len(matches) == 1:
            # Only one code block, return as-is
            return content

        # Multiple code blocks found - return only the last one
        # Include some context before the code block (like "Here's the solution:")
        last_match = matches[-1]
        last_code_start = last_match.start()

        # Look backwards for a natural break point (like "### Solution", "Here's", etc.)
        # to provide some context before the final code block
        context_pattern = r'(?:\n\n|\n### |\nHere|\nFinal|\nThe solution|\nImplementation)'
        context_before = content[:last_code_start]
        context_matches = list(re.finditer(context_pattern, context_before))

        if context_matches:
            # Start from the last context break
            context_start = context_matches[-1].end()
            # Find the actual start position
            start_pos = len(context_before[:context_start].rstrip())
        else:
            # No clear break point, just go back a bit for minimal context
            # Look for the last newline before the code block
            newline_before = content.rfind('\n', 0, last_code_start)
            if newline_before > last_code_start - 500:  # Within reasonable distance
                start_pos = newline_before
            else:
                start_pos = last_code_start

        filtered = content[start_pos:].lstrip()
        eval_logger.info(f"[VERTEX-FILTER] Extracted final code block (was {len(matches)} blocks total)")

        return filtered

    def model_call(
        self,
        messages: Union[List[List[int]], List[str], List[Dict]],
        *,
        generate=True,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[Union[List[str], List[Tuple[float, bool]]]]:
        """Synchronous model call using Vertex AI SDK."""
        from lm_eval.models.api_models import JsonChatStr

        # Handle message format - messages might be wrapped in a list
        actual_messages = messages
        if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[0], list):
            # Unwrap one level: [[{...}]] -> [{...}]
            actual_messages = messages[0]

        # Handle message format
        if isinstance(actual_messages, list) and len(actual_messages) > 0:
            if isinstance(actual_messages[0], JsonChatStr):
                # Parse JsonChatStr to dict
                import json
                actual_messages = [json.loads(msg.prompt) for msg in actual_messages]
            elif isinstance(actual_messages[0], str):
                # List of strings - convert to chat format
                actual_messages = [{"role": "user", "content": msg} for msg in actual_messages]

        # Create payload
        payload = self._create_payload(
            actual_messages,
            generate=generate,
            gen_kwargs=gen_kwargs,
            seed=self._seed,
            eos=self.eos_string,
            **kwargs,
        )

        try:
            # Extract config and messages
            config = payload.pop("config", {})
            gemini_config = payload.pop("gemini_config", {})
            messages_list = payload["messages"]

            # Debug: log message format
            eval_logger.debug(f"Vertex AI: messages_list type: {type(messages_list)}, length: {len(messages_list) if isinstance(messages_list, list) else 'N/A'}")
            if isinstance(messages_list, list) and len(messages_list) > 0:
                eval_logger.debug(f"Vertex AI: first message type: {type(messages_list[0])}")

            # Handle nested list wrapping - unwrap until we get to the actual messages
            # [[{...}]] -> [{...}] or even more nesting
            while (isinstance(messages_list, list) and len(messages_list) > 0 and
                   isinstance(messages_list[0], list)):
                eval_logger.debug(f"Vertex AI: unwrapping nested list, was length {len(messages_list)}")
                messages_list = messages_list[0]

            # Vertex AI expects messages as a list of Content objects
            # Convert from OpenAI format to Vertex format
            # Note: Vertex AI Gemini doesn't support 'system' role, so we handle it specially
            contents = []
            system_content = None  # Store system message to prepend to first user message

            for msg in messages_list:
                # Handle JsonChatStr - parse JSON string to list of dicts
                if isinstance(msg, JsonChatStr):
                    import json
                    try:
                        parsed = json.loads(msg.prompt)
                        # parsed should be a list of dicts like [{"role": "user", "content": "..."}]
                        if isinstance(parsed, list):
                            for item in parsed:
                                if isinstance(item, dict):
                                    role = item.get("role", "user")
                                    content = item.get("content", "")
                                    # Process this parsed message
                                    if role == "assistant":
                                        role = "model"
                                    elif role == "system":
                                        system_content = content
                                        continue
                                    if role == "user" and system_content:
                                        content = f"{system_content}\n\n{content}"
                                        system_content = None
                                    contents.append(
                                        types.Content(
                                            role=role,
                                            parts=[types.Part.from_text(text=content)]
                                        )
                                    )
                    except json.JSONDecodeError as e:
                        eval_logger.warning(f"Vertex AI: failed to parse JsonChatStr: {e}")
                    continue  # Skip the rest of the loop for JsonChatStr

                # Handle both dict and list formats
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                elif isinstance(msg, str):
                    role = "user"
                    content = msg
                elif isinstance(msg, list) and len(msg) >= 2:
                    # Handle [role, content] format
                    role = msg[0] if isinstance(msg[0], str) else "user"
                    content = msg[1] if isinstance(msg[1], str) else str(msg[1])
                else:
                    # Unexpected format, log and skip
                    eval_logger.warning(f"Vertex AI: skipping unexpected message format: {type(msg)} = {msg}")
                    continue

                # Map OpenAI roles to Vertex AI roles
                if role == "assistant":
                    role = "model"
                elif role == "system":
                    # Store system content to prepend to first user message
                    system_content = content
                    continue  # Don't add system message as a separate content

                # If this is the first user message and we have system content, prepend it
                if role == "user" and system_content:
                    content = f"{system_content}\n\n{content}"
                    system_content = None  # Only prepend to first user message

                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=content)]
                    )
                )

            if not contents:
                raise ValueError("No valid messages found after parsing. Input messages: " + str(messages_list)[:500])

            # Get max tokens from gen_kwargs
            max_tokens_from_kwargs = gen_kwargs.get("max_tokens")
            max_gen_toks_from_kwargs = gen_kwargs.get("max_gen_toks")
            max_tokens = max_tokens_from_kwargs or max_gen_toks_from_kwargs or self._max_gen_toks

            eval_logger.info(f"[VERTEX-DEBUG] Token limits: max_tokens={max_tokens_from_kwargs}, max_gen_toks={max_gen_toks_from_kwargs}, self._max_gen_toks={self._max_gen_toks}, using={max_tokens}")
            eval_logger.info(f"[VERTEX-DEBUG] gen_kwargs keys: {list(gen_kwargs.keys())}")
            eval_logger.info(f"[VERTEX-DEBUG] gemini_config: {gemini_config}")

            # Remove invalid parameters from gemini_config
            # These are either already handled or not supported by Vertex AI
            invalid_params = [
                "max_new_tokens", "max_gen_toks", "max_tokens",
                "until", "eos", "seed", "do_sample"
            ]
            for param in invalid_params:
                gemini_config.pop(param, None)

            eval_logger.info(f"[VERTEX-DEBUG] After cleanup: gemini_config={gemini_config}, max_output_tokens={max_tokens}")

            # Build the config - only use supported parameters
            # Note: We merge config (temperature, top_p, etc.) with gemini_config
            # But we explicitly set max_output_tokens
            api_config = {"max_output_tokens": max_tokens}
            api_config.update(config)
            api_config.update(gemini_config)

            eval_logger.info(f"[VERTEX-DEBUG] Final api_config: {api_config}")

            # Create a new client for each request to ensure thread safety
            # The GenAI SDK client may have issues with concurrent access
            client = genai.Client(**self._client_config)

            # Make the call
            response = client.models.generate_content(
                model=self.model,
                contents=contents,
                config=types.GenerateContentConfig(**api_config),
            )

            # Parse response
            if response.candidates and len(response.candidates) > 0:
                candidate = response.candidates[0]
                # Concatenate all parts from the response (Vertex AI may split content across multiple parts)
                parts = candidate.content.parts
                content = "".join([part.text for part in parts])

                # Check token usage metadata
                token_count = None
                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    token_count = getattr(response.usage_metadata, 'candidates_token_count', None)
                    if token_count is None and hasattr(response.usage_metadata, 'total_token_count'):
                        # Try to get output tokens from total
                        token_count = response.usage_metadata.total_token_count

                eval_logger.info(f"[VERTEX-DEBUG] Response: num_parts={len(parts)}, content_length={len(content)}, tokens={token_count}, finish_reason={candidate.finish_reason if hasattr(candidate, 'finish_reason') else 'N/A'}")
                if len(content) > 0:
                    eval_logger.info(f"[VERTEX-DEBUG] First 500 chars: {content[:500]}")

                # Filter out reasoning/thought process section
                # Gemini 2.5 Pro includes reasoning tokens that create multiple code blocks
                # We only want the final/last code block which contains the actual solution
                filtered_content = self._extract_final_code_block(content)
                if filtered_content != content:
                    eval_logger.info(f"[VERTEX-DEBUG] Filtered reasoning tokens: {len(content)} -> {len(filtered_content)} chars")
                    content = filtered_content

                # Return OpenAI-format response for compatibility with parse_generations
                # Note: Uses "text" format for completions API, not "message" format for chat
                result = {
                    "choices": [
                        {
                            "index": 0,
                            "text": content,
                            "finish_reason": candidate.finish_reason if hasattr(candidate, 'finish_reason') else "STOP"
                        }
                    ]
                }
                # Log the full response without trimming
                for line in content.split('\n'):
                    eval_logger.info(f"[FULL-RESPONSE] {line}")
                eval_logger.info(f"[VERTEX-DEBUG] ===== END FULL RESPONSE ({len(content)} chars total) =====")
                eval_logger.info(f"[VERTEX-DEBUG] Returning response with text length: {len(content)}, has_code_block: {'```' in content}")
                if '```' not in content:
                    eval_logger.warning(f"[VERTEX-DEBUG] Response does not contain code block! First 200 chars: {content[:200]}")
                return result
            else:
                eval_logger.warning(f"Empty response from Vertex AI: {response}")
                return None

        except Exception as e:
            error_msg = _handle_vertex_errors(e)
            eval_logger.warning(f"Vertex AI request failed: {error_msg}")
            raise

    async def amodel_call(
        self,
        messages: Union[List[List[int]], List[str], List[Dict]],
        *,
        generate=True,
        gen_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> Optional[Union[List[str], List[Tuple[float, bool]]]]:
        """Async model call using Vertex AI SDK (wraps sync call)."""
        # The google-genai SDK doesn't have native async support,
        # so we run the sync call in an executor
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.model_call(messages, generate=generate, gen_kwargs=gen_kwargs, **kwargs)
        )


# Also register as vertex-direct for convenience
@register_model("vertex-direct")
class VertexDirect(VertexChatCompletion):
    """Alias for vertex-chat-completions for consistency with other providers."""
    pass
