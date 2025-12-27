import os
from functools import cached_property
from operator import itemgetter
from typing import Any, Dict, List, Optional, Tuple, Union

from lm_eval.api.registry import register_model
from lm_eval.models.api_models import TemplateAPI
from lm_eval.models.utils import handle_stop_sequences
from lm_eval.utils import eval_logger

def _double_max_tokens(model_name: str, max_tokens: int, factor: int = 2) -> int:
    """
    Some OpenAI chat-completions models require the `max_tokens` parameter to be doubled.
    """
    m = model_name.lower()
    if "o1" in m or "o3" in m or "o4" in m:
        return max_tokens * factor

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
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                tmp[choices["index"]] = choices["text"]
            res = res + tmp
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

        gen_kwargs = {} if gen_kwargs is None else dict(gen_kwargs)

        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
        temperature = gen_kwargs.pop("temperature", 0)
        stop = handle_stop_sequences(gen_kwargs.pop("until", None), eos)
        if not isinstance(stop, (list, tuple)):
            stop = [stop]
        return {
            "messages": messages,
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": stop[:4],
            "seed": seed,
            **gen_kwargs,
        }

    @staticmethod
    def parse_generations(outputs: Union[Dict, List[Dict]], **kwargs) -> List[str]:
        res = []
        if not isinstance(outputs, list):
            outputs = [outputs]
        for out in outputs:
            tmp = [None] * len(out["choices"])
            for choices in out["choices"]:
                tmp[choices["index"]] = choices["message"]["content"]
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

        gen_kwargs.pop("do_sample", False)
        if "max_tokens" in gen_kwargs:
            max_tokens = gen_kwargs.pop("max_tokens")
        else:
            max_tokens = gen_kwargs.pop("max_gen_toks", self._max_gen_toks)
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

        # o1: original behavior
        if "o1" in self.model.lower():
            output.pop("stop", None)
            output["temperature"] = 1
            return output

        # GPT-5 family (and any other models we mark as disallowing stop):
        if _model_disallows_stop(self.model):
            output.pop("stop", None)
            
        if _model_disaalows_temperature(self.model):
            output["temperature"] = 1.0
            
        output["max_completion_tokens"] = _double_max_tokens(self.model, output["max_completion_tokens"])

        return output

