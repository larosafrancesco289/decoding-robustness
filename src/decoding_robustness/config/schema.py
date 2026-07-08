"""Pydantic schema for decoding-robustness experiment configs.

Encodes the factorial design: model checkpoints, quantization
levels, decoding methods, the (asymmetric) temperature grid, and tasks. This module
is pure data + validation: no I/O, no model loading. The validators deliberately
enforce the study's confound controls (e.g. a decoding method may only set its own
parameter, keeping method and temperature orthogonal).
"""

from __future__ import annotations

from collections.abc import Iterator
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QuantLevel(StrEnum):
    """Quantization levels (one provider's imatrix GGUFs). BF16 is the small-model anchor."""

    BF16 = "BF16"
    Q8_0 = "Q8_0"
    Q6_K = "Q6_K"
    Q4_K_M = "Q4_K_M"
    Q3_K_M = "Q3_K_M"


class DecodingMethod(StrEnum):
    """The decoding methods under study."""

    GREEDY = "greedy"  # argmax, deterministic anchor (T=0)
    TEMPERATURE = "temperature"  # pure temperature scaling, no truncation
    TOP_P = "top_p"
    TOP_K = "top_k"
    MIN_P = "min_p"
    TOP_N_SIGMA = "top_n_sigma"


# Which truncation parameter each method requires. Methods not listed take none.
_REQUIRED_PARAM: dict[DecodingMethod, str | None] = {
    DecodingMethod.GREEDY: None,
    DecodingMethod.TEMPERATURE: None,
    DecodingMethod.TOP_P: "top_p",
    DecodingMethod.TOP_K: "top_k",
    DecodingMethod.MIN_P: "min_p",
    DecodingMethod.TOP_N_SIGMA: "top_n_sigma",
}
_PARAM_FIELDS = ("top_p", "top_k", "min_p", "top_n_sigma")


class Checkpoint(BaseModel):
    """A specific model checkpoint (family-size), with its GGUF source."""

    model_config = ConfigDict(extra="forbid")

    name: str  # canonical slug, e.g. "llama-3.1-8b-instruct"
    family: str  # "llama" | "mistral" | "qwen3"
    params_b: float = Field(gt=0)  # size in billions of parameters (the size axis)
    hf_repo: str  # GGUF repo, e.g. "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF"
    # level -> GGUF filename. Resolved at M1 when files are fetched; may be empty in M0.
    quant_files: dict[QuantLevel, str] = Field(default_factory=dict)
    # None => render with the model's own tokenizer chat template (the default).
    chat_template: str | None = None

    def gguf_filename(self, level: QuantLevel) -> str:
        try:
            return self.quant_files[level]
        except KeyError as exc:
            raise KeyError(
                f"{self.name}: no GGUF filename registered for {level.value} "
                f"(populate quant_files before generation)"
            ) from exc


class SamplerSpec(BaseModel):
    """A decoding method with its (single) canonical parameter.

    Temperature is intentionally NOT stored here; it is the shared swept axis
    (`ExperimentConfig.temperatures`), keeping method and temperature orthogonal.
    """

    model_config = ConfigDict(extra="forbid")

    method: DecodingMethod
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, gt=0)
    min_p: float | None = Field(default=None, ge=0, le=1)
    top_n_sigma: float | None = Field(default=None, gt=0)
    # Explicit llama-server sampler-chain order to pin. None => runner default.
    chain: list[str] | None = None

    @model_validator(mode="after")
    def _enforce_orthogonal_params(self) -> SamplerSpec:
        required = _REQUIRED_PARAM[self.method]
        if required is not None and getattr(self, required) is None:
            raise ValueError(f"method '{self.method.value}' requires '{required}' to be set")
        for field in _PARAM_FIELDS:
            if field != required and getattr(self, field) is not None:
                raise ValueError(
                    f"method '{self.method.value}' must not set '{field}' "
                    f"(methods are kept orthogonal; only its own parameter is allowed)"
                )
        return self

    @property
    def label(self) -> str:
        """Stable identifier for filenames and result keys, e.g. 'min_p_min_p0.05'.

        An explicit chain that applies temperature *last* (the llama.cpp default, departing
        from our temperature-first default) gets a '_tlast' suffix, so the two ordering arms
        of the same method stay distinct in record ids and the analysis table.
        """
        parts = [self.method.value]
        for field in _PARAM_FIELDS:
            value = getattr(self, field)
            if value is not None:
                parts.append(f"{field}{value}")
        label = "_".join(parts)
        if self.chain is not None and len(self.chain) > 1 and self.chain[-1] == "temperature":
            label += "_tlast"
        return label


class TaskSpec(BaseModel):
    """An objective/verifiable benchmark task with a deterministic answer parser."""

    model_config = ConfigDict(extra="forbid")

    name: str  # "gsm8k" | "mmlu_pro" | "gpqa_diamond"
    hf_dataset: str  # HF datasets id
    revision: str  # HF dataset revision; exact items and prompts are logged per record
    hf_config: str | None = None  # HF dataset config/subset name, if any
    split: str = "test"
    subset_size: int | None = Field(default=None, gt=0)  # None => full split
    # "head" = first subset_size rows (the frozen-matrix behaviour; NOTE: MMLU-Pro's test
    # split is sorted by category, so head-50 is single-category). "stratified_category" =
    # deterministic round-robin over categories in row order (loaders that have a category
    # column only). The strategy is part of the item identity, so changing it changes ids.
    subset_strategy: str = Field(default="head", pattern="^(head|stratified_category)$")
    max_new_tokens: int = Field(default=512, gt=0)
    parser: str  # registered answer-parser name (implemented in tasks/, M2)


class ServerConfig(BaseModel):
    """llama-server launch settings."""

    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8080, gt=0, lt=65536)
    llama_cpp_commit: str | None = None  # pinned build commit, recorded in the manifest
    n_ctx: int = Field(default=4096, gt=0)
    n_gpu_layers: int = -1  # -1 => offload all layers to GPU
    # Parallel slots / continuous batching. >1 boosts throughput but breaks bit-exactness
    # 1 => deterministic single-stream.
    parallel: int = Field(default=1, gt=0)
    # We render the official chat template client-side, which already emits the model's BOS
    # when the template calls for it (Llama/Mistral do, Qwen does not). llama-server's
    # /completion path otherwise tokenizes with add_special=true and prepends *another* BOS,
    # a double-BOS that drifts from the canonical training prompt. Overriding the model's
    # add_bos metadata off makes server tokenization match HF apply_chat_template exactly,
    # keeping the rendered+hashed prompt faithful. Disable only to debug.
    disable_server_add_bos: bool = True
    extra_args: list[str] = Field(default_factory=list)


class GenerationConfig(BaseModel):
    """Sampling/seeding settings for a run."""

    model_config = ConfigDict(extra="forbid")

    global_seed: int = 0
    repetitions: int = Field(default=1, gt=0)  # samples per (condition, item)


class ExperimentConfig(BaseModel):
    """A full experiment specification (one YAML file under configs/)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    phase: str  # "pilot" | "phase1" | "phase2" | ...
    output_dir: str = "results"
    checkpoints: list[Checkpoint] = Field(min_length=1)
    quant_levels: list[QuantLevel] = Field(min_length=1)
    samplers: list[SamplerSpec] = Field(min_length=1)
    temperatures: list[float] = Field(min_length=1)  # grid for stochastic methods only
    tasks: list[TaskSpec] = Field(min_length=1)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)

    @model_validator(mode="after")
    def _check(self) -> ExperimentConfig:
        for temp in self.temperatures:
            if temp <= 0:
                raise ValueError(
                    "temperatures is the grid for stochastic methods and must be > 0; "
                    "use the 'greedy' method for deterministic T=0 decoding"
                )
        if len(self.temperatures) != len(set(self.temperatures)):
            raise ValueError("temperatures must be unique")
        return self

    def decoding_conditions(self) -> Iterator[tuple[SamplerSpec, float]]:
        """Yield (sampler, temperature) pairs per the asymmetric-grid rule.

        Greedy is emitted exactly once at T=0; every other method is crossed with the
        full temperature grid (so we don't waste samples in the low-T region where all
        truncation methods collapse to greedy).
        """
        for sampler in self.samplers:
            if sampler.method == DecodingMethod.GREEDY:
                yield (sampler, 0.0)
            else:
                for temp in self.temperatures:
                    yield (sampler, temp)
