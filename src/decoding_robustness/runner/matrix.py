"""Matrix expansion: loop (model, quant) server loads, and run
the full decoding grid for every task through the resumable loop.

Loop order minimises weight reloads: the outer loop is (checkpoint, quant), one server
load, and the sampler/temperature/task conditions are issued as request params against
that single running server. Everything appends to one id-keyed JSONL, so the whole pilot
is interruptible and resumable as a unit (an added quant or task just contributes new ids).

Built to run *incrementally*: quants whose GGUF is not yet on disk are skipped (with a
note), and tasks whose parser/loader is not yet registered are skipped, so the run can
start on the quants already downloaded and widen as more arrive.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..config.schema import Checkpoint, ExperimentConfig
from ..inference import ChatTemplate, llama_server
from ..tasks import load_task
from ..tasks.parsers import _PARSERS
from .loop import GenerationRecord, run_conditions
from .records import RecordStore


@dataclass
class MatrixProgress:
    """What the matrix run did, for the M3 self-check."""

    server_loads: int = 0
    generated: int = 0
    skipped: int = 0
    missing_quants: list[str] = field(default_factory=list)
    skipped_tasks: list[str] = field(default_factory=list)


def _resolve_template(ckpt: Checkpoint, model_path: Path) -> ChatTemplate:
    if ckpt.chat_template is not None:
        return ChatTemplate(template=ckpt.chat_template)
    meta_path = model_path.with_suffix(model_path.suffix + ".meta.json")
    if meta_path.is_file():
        return ChatTemplate.from_meta_file(meta_path)
    raise FileNotFoundError(
        f"no chat template for {ckpt.name}: set Checkpoint.chat_template or fetch "
        f"{model_path.name} (fetch_models.py writes {meta_path.name})"
    )


def run_matrix(
    config: ExperimentConfig,
    *,
    model_dir: str | Path = "models",
    binary: str | None = None,
    out_path: str | Path | None = None,
    limit: int | None = None,
    only_models: set[str] | None = None,
    only_quants: set[str] | None = None,
    enable_thinking: bool = False,
    on_record: Callable[[GenerationRecord], None] | None = None,
) -> MatrixProgress:
    """Run the configured (checkpoint × quant × decode-grid × task) pilot matrix.

    ``limit`` overrides every task's subset size (handy for a quick smoke); ``None`` uses
    each TaskSpec's own ``subset_size``. Returns a MatrixProgress with what ran vs. skipped.
    """
    model_dir = Path(model_dir)
    out_path = Path(out_path) if out_path else Path(config.output_dir) / f"{config.phase}.jsonl"
    store = RecordStore(out_path)
    progress = MatrixProgress()

    conditions = list(config.decoding_conditions())
    runnable_tasks = [t for t in config.tasks if t.parser in _PARSERS]
    progress.skipped_tasks = [t.name for t in config.tasks if t.parser not in _PARSERS]

    for ckpt in config.checkpoints:
        if only_models and ckpt.name not in only_models:
            continue
        for quant in config.quant_levels:
            if only_quants and quant.value not in only_quants:
                continue
            filename = ckpt.quant_files.get(quant)
            model_path = model_dir / filename if filename else None
            if model_path is None or not model_path.is_file():
                progress.missing_quants.append(f"{ckpt.name}/{quant.value}")
                continue

            template = _resolve_template(ckpt, model_path)
            stop = [template.eos_token] if template.eos_token else None

            with llama_server(model_path, config.server, binary=binary) as client:
                progress.server_loads += 1
                for task in runnable_tasks:
                    task_spec = task.model_copy(update={"subset_size": limit}) if limit else task
                    items = load_task(task_spec)
                    summary = run_conditions(
                        client=client,
                        template=template,
                        checkpoint=ckpt,
                        quant=quant,
                        task=task_spec,
                        items=items,
                        conditions=conditions,
                        store=store,
                        global_seed=config.generation.global_seed,
                        repetitions=config.generation.repetitions,
                        concurrency=config.server.parallel,
                        server_commit=config.server.llama_cpp_commit,
                        stop=stop,
                        enable_thinking=enable_thinking,
                        on_record=on_record,
                    )
                    progress.generated += summary.generated
                    progress.skipped += summary.skipped

    return progress
