import os
from functools import cached_property
from typing import Any, Dict, List, Optional, Tuple, Union

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI
from lm_eval.models.utils import handle_stop_sequences
from lm_eval.utils import eval_logger


eval_logger = eval_logger


def _normalize_max_tokens(gen_kwargs: dict) -> dict:
    """
    MultiPL-E / some tasks pass max_new_tokens.
    lm-eval API models expect max_gen_toks or max_tokens.
    Normalize here so all backends behave consistently.
    """
    if gen_kwargs is None:
        return {}
    gen_kwargs = dict(gen_kwargs)

    # Prefer explicit max_tokens/max_gen_toks if already present.
    if "max_new_tokens" in gen_kwargs and "max_tokens" not in gen_kwargs and "max_gen_toks" not in gen_kwargs:
        gen_kwargs["max_gen_toks"] = gen_kwargs.pop("max_new_tokens")

    return gen_kwargs


def google_chat(
    client,  # genai.Client
    model: str,
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    stop: List[str],
    system_instruction: str = None,
    **kwargs: Any,
) -> str:
    """Wrapper function around the Google GenAI SDK with retry logic.

    params:
        client: genai.Client
            Google GenAI API client
        model: str
            Google model e.g. 'gemini-2.5-flash', 'gemini-2.5-pro', 'gemini-3-pro-preview'
        messages: List[Dict]
            Chat messages with 'role' and 'content' keys
        max_tokens: int
            Maximum number of tokens to sample from the model
        temperature: float
            Sampling temperature
        stop: List[str]
            List of stop sequences
        system_instruction: str
            Optional system instruction
        kwargs: Any
            Additional model_args to pass to the API client
    """

    try:
        from google import genai
        from google.genai import types
    except ModuleNotFoundError as exception:
        raise type(exception)(
            "attempted to use 'google' LM type, but package `google-genai` is not installed. "
            "please install it via `pip install google-genai`"
        )

    def _exception_callback(e: Exception, sleep_time: float) -> None:
        eval_logger.warning(
            f"API error occurred: {e}\nRetrying in {sleep_time} seconds"
        )

    # Simple retry with exponential backoff
    import time
    max_retries = 5
    base_delay = 1

    for attempt in range(max_retries):
        try:
            from google.genai import types

            # Build config
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )

            # Add stop sequences if provided
            if stop:
                stop_sequences = [s for s in stop if s]
                if stop_sequences:
                    config.stop_sequences = stop_sequences

            # Add system instruction if provided
            if system_instruction:
                config.system_instruction = system_instruction

            # Convert messages to Google's Content format
            contents = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                # Handle non-string content (e.g., JsonChatStr objects)
                if not isinstance(content, str):
                    # Try to get prompt attribute or convert to string
                    if hasattr(content, 'prompt'):
                        content = content.prompt
                    else:
                        content = str(content)

                if role == "system":
                    # System messages are handled via system_instruction
                    if not hasattr(config, 'system_instruction') or not config.system_instruction:
                        config.system_instruction = content
                    continue
                elif role == "user":
                    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=content)]))
                elif role == "assistant":
                    contents.append(types.Content(role="model", parts=[types.Part.from_text(text=content)]))

            # Generate content
            response = client.models.generate_content(
                model=model,
                contents=contents if contents else "Please respond.",
                config=config,
            )
            return response.text

        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                _exception_callback(e, delay)
                time.sleep(delay)
            else:
                raise


@register_model("google-chat-completions")
class GoogleChatCompletion(TemplateAPI):
    def __init__(
        self,
        base_url=None,  # Not used for Google GenAI SDK
        tokenizer_backend=None,
        **kwargs,
    ):
        eval_logger.warning(
            "Google chat-completions requires the `--apply_chat_template` flag."
        )
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            **kwargs,
        )

        try:
            from google import genai
        except ModuleNotFoundError as exception:
            raise type(exception)(
                "attempted to use 'google' LM type, but package `google-genai` is not installed. "
                "please install it via `pip install google-genai`"
            )

        if self._batch_size > 1:
            eval_logger.warning(
                "Google GenAI does not support batching. Defaulting to batch size 1."
            )
            self._batch_size = 1

        eval_logger.info(
            f"Using Google GenAI SDK with model: {self.model}"
        )

    @cached_property
    def api_key(self):
        """Return the API key for Google GenAI."""
        key = os.environ.get("GOOGLE_API_KEY", None)
        if key is None:
            key = os.environ.get("GEMINI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
            )
        return key

    @cached_property
    def client(self):
        """Create and cache the Google GenAI client."""
        from google import genai

        return genai.Client(api_key=self.api_key)

    async def amodel_call(self, *args, **kwargs):
        """
        Override async model call to use Google GenAI SDK directly.
        This bypasses the HTTP request mechanism of TemplateAPI.
        """
        # Handle different calling patterns due to retry decorator
        payload = None

        # First arg could be the payload dict
        if args and isinstance(args[0], dict):
            payload = args[0]
        # First arg could be a JsonChatStr or similar object with .prompt attribute
        elif args and hasattr(args[0], 'prompt'):
            payload = {"messages": args[0].prompt}
        # Payload could be in kwargs
        elif 'payload' in kwargs:
            payload = kwargs['payload']
        # Messages could be directly in kwargs
        elif 'messages' in kwargs:
            payload = kwargs

        # If still no payload, use kwargs as payload
        if payload is None:
            payload = kwargs

        # Extract parameters from payload (created by _create_payload)
        messages = payload.get("messages", [])
        max_tokens = payload.get("max_tokens", self._max_gen_toks)
        temperature = payload.get("temperature", 0)
        stop = payload.get("stop", [])
        system_instruction = payload.get("system_instruction")

        # Handle messages as a list or convert string to list
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        elif not isinstance(messages, list):
            messages = [messages]

        # Run the blocking Google API call in a thread pool to avoid blocking the event loop
        import asyncio
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: google_chat(
                client=self.client,
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                system_instruction=system_instruction,
            )
        )

        # Return in the format expected by parse_generations (dict with choices)
        return {
            "choices": [
                {
                    "message": {
                        "content": response
                    },
                    "text": response
                }
            ]
        }

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos=None,
        **kwargs,
    ) -> dict:
        assert (
            type(messages) is not str
        ), "chat-completions require the --apply_chat_template flag."

        gen_kwargs = _normalize_max_tokens(gen_kwargs)

        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)

        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]

        # Filter out empty stop sequences
        stop = [s for s in stop if s]

        # Extract system instruction if present
        system_instruction = None
        if messages and messages[0].get("role") == "system":
            system_instruction = messages[0].get("content")
            messages = messages[1:]

        out = {
            "messages": messages,
            "model": self.model,
            "max_tokens": int(max_tokens),
            "temperature": temperature,
            "stop": stop,
            "seed": seed,
            "system_instruction": system_instruction,
            **gen_kwargs,
        }

        return out

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        """
        Google GenAI does not support logprobs, so this raises NotImplementedError.
        This method is required by TemplateAPI but is not used for chat completions.
        """
        raise NotImplementedError(
            "Google GenAI does not support logprobs (token probabilities)."
        )

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            if isinstance(out, str):
                res.append(out)
            elif isinstance(out, dict):
                if "choices" in out:
                    for choices in out["choices"]:
                        if "message" in choices:
                            res.append(choices["message"]["content"])
                        else:
                            res.append(choices.get("text", ""))
                else:
                    res.append(str(out))
            else:
                res.append(str(out))
        return res

    def _batch_generate_request(self, requests, disable_tqdm: bool = False):
        """
        Override to use Google GenAI SDK directly instead of HTTP requests.
        Returns list of (index, response_string) tuples.
        """
        from tqdm import tqdm

        res = []
        for idx, request in enumerate(tqdm(requests, disable=disable_tqdm)):
            try:
                # request is the payload dict returned by _create_payload
                # Extract parameters from the payload
                messages = request.get("messages", [])
                max_tokens = request.get("max_tokens", self._max_gen_toks)
                temperature = request.get("temperature", 0)
                stop = request.get("stop", [])
                system_instruction = request.get("system_instruction")

                # Handle messages as a list or convert string to list
                if isinstance(messages, str):
                    messages = [{"role": "user", "content": messages}]
                elif not isinstance(messages, list):
                    messages = [messages]

                # Call the Google API
                response = google_chat(
                    client=self.client,
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=stop,
                    system_instruction=system_instruction,
                )

                res.append((idx, response))

            except Exception as e:
                eval_logger.error(f"Error generating response: {e}")
                import traceback
                traceback.print_exc()
                res.append((idx, ""))

        return res

    def tok_encode(
        self,
        string: Union[str, Any],
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> Union[List[str], List[int], Any]:
        # Google GenAI uses token counting via the API
        # For chat completions, we return the string as-is
        return string

    def tok_decode(self, tokens: List[int]) -> str:
        # Not applicable for Google GenAI
        return "".join(str(t) for t in tokens)

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Google GenAI does not support loglikelihood (token probabilities)."
        )

    def loglikelihood_rolling(self, requests, **kwargs):
        raise NotImplementedError(
            "Google GenAI does not support loglikelihood (token probabilities)."
        )
