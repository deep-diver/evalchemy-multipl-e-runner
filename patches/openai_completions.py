import os
from functools import cached_property
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Union

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI
from lm_eval.models.utils import handle_stop_sequences
from lm_eval.utils import eval_logger

def _is_openrouter(base_url: str) -> bool:
    return "openrouter.ai" in (base_url or "").lower()

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

def _model_reasoning_effort(model_name: str) -> bool:
    """
    Some OpenAI chat-completions models require the `reasoning_effort` parameter.
    """
    if not model_name:
        return False
    m = model_name.lower()
    if "o1" in m or "o3" in m or "o4" in m or "gpt-5" in m:
        return True
    return False

def _double_max_tokens(model_name: str, max_tokens: int, factor: int = 2) -> int:
    """
    Some OpenAI chat-completions models require the `max_tokens` parameter to be doubled.
    """
    m = model_name.lower()
    if "o1" in m or "o3" in m or "o4" in m:
        return max_tokens * factor
    
    if "gpt-5" in m:
        return max_tokens * factor
    
    return max_tokens

def _model_disallows_stop(model_name: str) -> bool:
    """
    Some OpenAI chat-completions models reject the `stop` parameter.
    We defensively disable `stop` for those models to avoid 400 errors.
    """
    if not model_name:
        return False
    m = model_name.lower()

    # Known families that often disallow stop on chat-completions:
    # - o1 (already handled originally)
    # - GPT-5 family (gpt-5, gpt-5.1, gpt-5.2, chat-latest variants, etc.)
    if "o1" in m or "o3" in m or "o4" in m:
        return True
    if m.startswith("gpt-5"):
        return True
    if "gpt-5" in m:
        return True

    # If you later find other families disallow `stop`, add them here.
    return False


def _model_disaalows_temperature(model_name: str) -> bool:
    """
    Some OpenAI chat-completions models reject the `temperature` parameter.
    We defensively disable `temperature` for those models to avoid 400 errors.
    """
    if not model_name:
        return False
    m = model_name.lower()
    if "o1" in m or "o3" in m or "o4" in m:
        return True
    if "gpt-5" in m:
        return True
    return False

@register_model("local-completions")
class LocalCompletionsAPI(TemplateAPI):
    def __init__(
        self,
        base_url=None,
        tokenizer_backend="huggingface",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )

    def _create_payload(
        self,
        messages: Union[List[List[int]], List[dict], List[str], str],
        generate=False,
        gen_kwargs: Optional[dict] = None,
        seed: int = 1234,
        eos=None,
        **kwargs,
    ) -> dict:
        gen_kwargs = {} if gen_kwargs is None else dict(gen_kwargs)
        # gen_kwargs = _normalize_max_tokens(gen_kwargs)
        # print("--------------------------------")
        # print(gen_kwargs)
        # print("--------------------------------")

        if generate:
            gen_kwargs.pop("do_sample", False)
            if "max_tokens" in gen_kwargs:
                max_tokens = gen_kwargs.pop("max_tokens")
            else:
                max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
            temperature = gen_kwargs.pop("temperature", 0)
            stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
            return {
                "prompt": messages,
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop,
                "seed": seed,
                **gen_kwargs,
            }
        else:
            return {
                "model": self.model,
                "prompt": messages,
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": 1,
                "seed": seed,
                "echo": True,
            }

    @staticmethod
    def parse_logprobs(
        outputs: Union[Dict, List[Dict]],
        tokens: List[List[int]] = None,
        ctxlens: List[int] = None,
        **kwargs,
    ) -> List[Tuple[float, bool]]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            for choice, ctxlen in zip(
                sorted(out["choices"], key=itemgetter("index")), ctxlens
            ):
                assert ctxlen > 0, "Context length must be greater than 0"
                logprobs = sum(choice["logprobs"]["token_logprobs"][ctxlen:-1])
                tokens_logprobs = choice["logprobs"]["token_logprobs"][ctxlen:-1]
                top_logprobs = choice["logprobs"]["top_logprobs"][ctxlen:-1]
                is_greedy = True
                for tok, top in zip(tokens_logprobs, top_logprobs):
                    if tok != max(top.values()):
                        is_greedy = False
                        break
                res.append((logprobs, is_greedy))
        return res

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        from lm_eval.models.eval_logger import eval_logger
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                text = choices["text"]
                tmp[choices["index"]] = text
                eval_logger.info(f"[PARSE-DEBUG] Extracted text length: {len(text)}, has_code_block: {'```' in text}")
                if len(text) > 0:
                    eval_logger.info(f"[PARSE-DEBUG] First 200 chars: {text[:200]}")
            res = res + tmp
        eval_logger.info(f"[PARSE-DEBUG] Total parsed outputs: {len(res)}")
        return res

    @property
    def api_key(self):
        return os.environ.get("OPENAI_API_KEY", "")


@register_model("local-chat-completions")
class LocalChatCompletion(LocalCompletionsAPI):
    def __init__(
        self,
        base_url=None,
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        eval_logger.warning(
            "chat-completions endpoint requires the `--apply_chat_template` flag."
        )
        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )
        # Chat completions API doesn't support true batching (multiple requests in one API call)
        # batch_size here controls how many requests are grouped into one async task
        # Keep it at 1 for compatibility, but note that rate limiting affects performance
        if self._batch_size > 1:
            eval_logger.warning(
                "Chat completions does not support batching. Defaulting to batch size 1."
            )
            self._batch_size = 1

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

        # Extract max_tokens or max_gen_toks and convert to max_tokens
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
            # Cap max_tokens to avoid exceeding model context length
            # Most models have 32k-128k context; use conservative value
            if max_tokens > 8192:
                eval_logger.info(f"[PAYLOAD-DEBUG] Capping max_tokens from {max_tokens} to 8192 to avoid exceeding context length")
                max_tokens = 8192
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)

        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]

        # Gemini OpenAI-compatible endpoint doesn't accept seed and max_gen_toks
        is_gemini = self.base_url and "generativelanguage.googleapis.com" in self.base_url

        # Filter out unsupported parameters for Gemini
        if is_gemini:
            gen_kwargs.pop("max_gen_toks", None)
            gen_kwargs.pop("max_tokens", None)

        out = {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **gen_kwargs,
        }

        # Only add stop if non-empty
        if stop and any(s for s in stop[:4] if s):
            out["stop"] = stop[:4]

        # Only add seed for non-Gemini endpoints
        if not is_gemini:
            out["seed"] = seed

        return out

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        from lm_eval.utils import eval_logger

        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                try:
                    # Try standard OpenAI format
                    tmp[choices["index"]] = choices["message"]["content"]
                except (KeyError, TypeError) as e:
                    # Handle edge cases: Gemini might have different format
                    eval_logger.warning(f"Unexpected response format: {choices}. Error: {e}")
                    # Try to extract content from alternative formats
                    content = None
                    if isinstance(choices, dict):
                        # Try direct content field
                        content = choices.get("content")
                        # Try text field
                        if content is None:
                            content = choices.get("text")
                        # Try nested message
                        if content is None and "message" in choices:
                            msg = choices["message"]
                            if isinstance(msg, dict):
                                content = msg.get("content") or msg.get("text")
                    # Fallback to string representation
                    tmp[choices["index"]] = content if content is not None else str(choices)
            res = res + tmp
        return res

    def tok_encode(
        self,
        string: Union[str, Any],
        left_truncate_len=None,
        add_special_tokens=None,
        **kwargs,
    ) -> Union[List[str], List[int], Any]:
        return string

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood is not supported for chat completions. Consider using the completions API instead."
        )


@register_model(
    "openai-completions",
)
class OpenAICompletionsAPI(LocalCompletionsAPI):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/completions",
        tokenizer_backend="tiktoken",
        **kwargs,
    ):
        super().__init__(
            base_url=base_url, tokenizer_backend=tokenizer_backend, **kwargs
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        assert (
            self.model
            in [
                "babbage-002",
                "davinci-002",
            ]
        ), f"Prompt loglikelihoods are only supported by OpenAI's API for {['babbage-002', 'davinci-002']}."
        return super().loglikelihood(requests, **kwargs)

    def chat_template(self, chat_template: Union[bool, str] = False) -> Optional[str]:
        return ""


@register_model("openai-chat-completions")
class OpenAIChatCompletion(LocalChatCompletion):
    def __init__(
        self,
        base_url="https://api.openai.com/v1/chat/completions",
        tokenizer_backend=None,
        tokenized_requests=False,
        **kwargs,
    ):
        model_name = kwargs.get("model", "") or ""
        if "o1" in model_name:
            eval_logger.warning(
                "o1 models do not support `stop` and only support temperature=1"
            )
        if _model_disallows_stop(model_name) and "o1" not in model_name.lower():
            eval_logger.warning(
                f"{model_name} may not support `stop` on chat-completions; disabling `stop` in request payload."
            )

        super().__init__(
            base_url=base_url,
            tokenizer_backend=tokenizer_backend,
            tokenized_requests=tokenized_requests,
            **kwargs,
        )

    @cached_property
    def api_key(self):
        """Override this property to return the API key for the API request."""
        key = os.environ.get("OPENAI_API_KEY", None)
        if key is None:
            raise ValueError(
                "API key not found. Please set the `OPENAI_API_KEY` environment variable."
            )
        return key

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError(
            "Loglikelihood (and therefore `multiple_choice`-type tasks) is not supported for chat completions as OpenAI does not provide prompt logprobs. See https://github.com/EleutherAI/lm-evaluation-harness/issues/942#issuecomment-1777836312 or https://github.com/EleutherAI/lm-evaluation-harness/issues/1196 for more background on this limitation."
        )

    def _create_payload(
        self,
        messages: List[Dict],
        generate=False,
        gen_kwargs: dict = None,
        seed=1234,
        eos="<|endoftext|>",
        **kwargs,
    ) -> dict:
        assert (
            type(messages) is not str
        ), "chat-completions require the --apply_chat_template flag."

        gen_kwargs = {} if gen_kwargs is None else dict(gen_kwargs)

        # Debug: log incoming gen_kwargs
        eval_logger.info(f"[PAYLOAD-DEBUG] Incoming gen_kwargs keys: {list(gen_kwargs.keys())}, max_tokens={gen_kwargs.get('max_tokens', 'N/A')}, max_gen_toks={gen_kwargs.get('max_gen_toks', 'N/A')}, max_new_tokens={gen_kwargs.get('max_new_tokens', 'N/A')}")

        # For Qwen3 models on vLLM, ensure chat_template_kwargs is set to disable thinking by default
        # Qwen3 has thinking mode enabled by default, which outputs </think> tags
        # If chat_template_kwargs is not explicitly set, disable thinking mode
        is_vllm = self.base_url and ("localhost" in self.base_url or "129.254" in self.base_url or "vllm" in self.base_url.lower())
        is_qwen3 = "qwen3" in self.model.lower()

        if is_vllm and is_qwen3 and "chat_template_kwargs" not in gen_kwargs:
            # Explicitly disable thinking mode for Qwen3 by default
            gen_kwargs["chat_template_kwargs"] = {"enable_thinking": False}
            eval_logger.info(f"[PAYLOAD-DEBUG] Auto-setting chat_template_kwargs={{'enable_thinking': False}} for Qwen3 model")

        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
            # Cap max_tokens to avoid exceeding model context length
            # Most models have 32k-128k context; use conservative value
            if max_tokens > 8192:
                eval_logger.info(f"[PAYLOAD-DEBUG] Capping max_tokens from {max_tokens} to 8192 to avoid exceeding context length")
                max_tokens = 8192
            eval_logger.info(f"[PAYLOAD-DEBUG] Using max_tokens from gen_kwargs: {max_tokens}")
        else:
            max_tokens_from_kwargs = gen_kwargs.pop("max_gen_toks", None)
            if max_tokens_from_kwargs is not None:
                max_tokens = max_tokens_from_kwargs
                eval_logger.info(f"[PAYLOAD-DEBUG] Using max_gen_toks from gen_kwargs: {max_tokens}")
            else:
                max_tokens = self._max_gen_toks
                eval_logger.info(f"[PAYLOAD-DEBUG] Using default self._max_gen_toks: {max_tokens}")
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", ["<|endoftext|>"]), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]

        output = {
            "messages": messages,
            "model": self.model,
            # NOTE: keep using max_completion_tokens (your original code already does this)
            "max_completion_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }

        # # 4) IMPORTANT: openrouter prefers max_tokens key
        # if "max_new_tokens" in gen_kwargs and "max_tokens" not in gen_kwargs and "max_gen_toks" not in gen_kwargs:
        #     gen_kwargs["max_gen_toks"] = gen_kwargs.pop("max_new_tokens")

        # o1: original behavior
        if "o1" in self.model.lower():
            output.pop("stop", None)
            output["temperature"] = 1
            return output

        # GPT-5 family (and any other models we mark as disallowing stop):
        if _model_disallows_stop(self.model):
            output.pop("stop", None)
            
        if _model_disaalows_temperature(self.model):
            # output["temperature"] = 1.0
            output.pop("temperature", None)
        
        # if _model_reasoning_effort(self.model):
        #     output["reasoning_effort"] = "high"

        original_max_tokens = output["max_completion_tokens"]
        output["max_completion_tokens"] = _double_max_tokens(self.model, output["max_completion_tokens"])
        if output["max_completion_tokens"] != original_max_tokens:
            eval_logger.info(f"[PAYLOAD-DEBUG] Doubled max_tokens: {original_max_tokens} -> {output['max_completion_tokens']}")

        return output

