"""Kapel, an inference engine for small language models: load, generate, serve."""

from typing import Iterator, Literal, Mapping, Optional, Sequence, Union, overload

__version__: str

class KapelError(Exception):
    """A model that cannot be read, a pass of a model that failed."""

class Stream:
    """Pieces of text as they are generated. Dropping it stops the generation."""

    def __iter__(self) -> "Stream": ...
    def __next__(self) -> str: ...

class Model:
    """A model loaded on its device, with its tokenizer and chat template.

    ``name`` is a model directory, a GGUF file, or a repository of the Hugging
    Face Hub (``owner/name``, downloaded once into ``~/.cache/kapel/hub``, or
    ``$KAPEL_CACHE``; a gated model needs ``HF_TOKEN``).

    ``device`` is ``"cpu"``, ``"cuda"`` or ``"cuda:N"``; ``dtype`` is ``"f32"``,
    ``"bf16"`` or ``"f16"`` (by default f32 on the processor, bf16 on a card);
    ``quantize`` is ``"int8"``, ``"int4"`` (or ``"int8/G"``, ``"int4/G"``),
    ``"q8_0"`` or ``"q4_0"``; ``kv_cache`` is ``"f32"``, ``"bf16"``, ``"f16"``,
    ``"f8e4m3"`` or ``"f8e5m2"``.
    """

    def __init__(
        self,
        name: str,
        device: str = "cpu",
        dtype: Optional[str] = None,
        quantize: Optional[str] = None,
        kv_cache: Optional[str] = None,
        revision: str = "main",
    ) -> None: ...
    @property
    def has_chat_template(self) -> bool:
        """Whether the model has a chat template."""
    def tokenize(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """The tokens of ``text``."""
    def detokenize(self, ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        """The text of ``ids``."""
    def chat_prompt(self, messages: Sequence[Mapping[str, str]]) -> list[int]:
        """The tokens of a conversation in the model's chat template, ending
        with the start of the assistant's answer."""
    @overload
    def generate(
        self,
        prompt: Union[str, Sequence[int]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        stream: Literal[False] = False,
    ) -> str: ...
    @overload
    def generate(
        self,
        prompt: Union[str, Sequence[int]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        *,
        stream: Literal[True],
    ) -> Iterator[str]: ...
    @overload
    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        stream: Literal[False] = False,
    ) -> str: ...
    @overload
    def chat(
        self,
        messages: Sequence[Mapping[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        seed: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        *,
        stream: Literal[True],
    ) -> Iterator[str]: ...

def serve(
    name: str,
    host: str = "127.0.0.1",
    port: int = 8000,
    device: str = "cpu",
    dtype: Optional[str] = None,
    quantize: Optional[str] = None,
    kv_cache: Optional[str] = None,
    revision: str = "main",
    served_model_name: Optional[str] = None,
    max_model_len: Optional[int] = None,
    cache_memory: Optional[float] = None,
    gpu_memory_utilization: float = 0.85,
    max_sequences: int = 256,
    max_pass_tokens: int = 2048,
    max_requests: int = 1024,
    prefix_sharing: bool = True,
    timeout: float = 600.0,
) -> None:
    """Loads ``name`` and serves it behind the HTTP API compatible with
    OpenAI's, until Ctrl-C. The options are those of ``kapel serve``."""
