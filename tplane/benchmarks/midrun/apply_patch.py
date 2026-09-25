"""Patch rl-rewardhacking-ext and its bundled verl 0.6.1 for the mid-run benchmark, at named anchors, idempotently.

Every anchor must occur the stated number of times, or nothing is written: a changed upstream file stops the
patch instead of being edited in the wrong place. Run: python apply_patch.py --repo PATH --verl PATH.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

MARKER: Final[str] = "tplane-benchmark"
LR_METHOD: Final[str] = '''    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def scale_learning_rate(self, factor: float) -> None:  # tplane-benchmark
        """Scale the scheduler's base learning rates; LambdaLR recomputes each step's rate from them."""
        self.actor_lr_scheduler.base_lrs = [rate * factor for rate in self.actor_lr_scheduler.base_lrs]

'''


@dataclass(frozen=True, kw_only=True)
class Edit:
    """Insert text after (or replace) an anchor that must occur exactly `count` times."""

    path: str
    anchor: str
    text: str
    count: int = 1
    replace: bool = False
    before: bool = False


def edits(repo: Path, verl: Path) -> list[tuple[Path, Edit]]:
    trainer = repo / "src/train/verl/trainer.py"
    workers = repo / "src/train/verl/workers/extended_workers.py"
    template = repo / "src/train/verl/grpo_config.jinja2"
    rollout = verl / "verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py"
    return [
        (
            trainer,
            Edit(
                path="trainer",
                anchor="from verl.trainer.ppo.reward import load_reward_manager, compute_reward, compute_reward_async\n",
                text="from benchmarks.midrun.verl_bridge import BenchmarkBridge  # tplane-benchmark\n",
            ),
        ),
        (
            trainer,
            Edit(
                path="trainer",
                anchor='        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")\n',
                text="        bridge = BenchmarkBridge.from_environment()  # tplane-benchmark\n        bridge.supersede_from(self.global_steps + 1)\n",
            ),
        ),
        (
            trainer,
            Edit(
                path="trainer",
                anchor='                gen_batch.meta_info["global_steps"] = self.global_steps\n',
                text=(
                    '                gen_batch.meta_info["rollout_overrides"] = bridge.rollout_overrides(self.global_steps)  # tplane-benchmark\n'
                    "                lr_factor = bridge.learning_rate_change(self.global_steps)\n"
                    "                if lr_factor is not None:\n"
                    "                    self.actor_rollout_wg.scale_learning_rate(lr_factor)\n"
                ),
            ),
        ),
        (
            trainer,
            Edit(
                path="trainer",
                anchor="""                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'\n""",
                text="                    bridge.on_old_log_probs(batch)  # tplane-benchmark\n",
            ),
        ),
        (
            trainer,
            Edit(
                path="trainer",
                anchor='                        batch.batch["token_level_scores"] = reward_tensor\n',
                text="                        reward_tensor = bridge.on_rewards(self.global_steps, batch, reward_tensor)  # tplane-benchmark\n",
                before=True,
            ),
        ),
        (
            workers,
            Edit(
                path="workers",
                anchor="    def init_model(self):\n",
                text=LR_METHOD,
                count=2,
                before=True,
            ),
        ),
        (
            template,
            Edit(
                path="template",
                anchor="        calculate_log_probs: false\n",
                text="        calculate_log_probs: true  # tplane-benchmark\n",
                replace=True,
            ),
        ),
        (
            template,
            Edit(
                path="template",
                anchor="            vllm: {}\n",
                text="            vllm: {logprobs_mode: processed_logprobs}  # tplane-benchmark\n",
                replace=True,
            ),
        ),
        (
            rollout,
            Edit(
                path="rollout",
                anchor='                "n": 1,  # if validate, already repeat in ray_trainer\n            }\n',
                text='        elif prompts.meta_info.get("rollout_overrides"):  # tplane-benchmark\n            kwargs = dict(prompts.meta_info["rollout_overrides"])\n',
            ),
        ),
    ]


def apply(repo: Path, verl: Path) -> list[str]:
    """Return one line per file: patched, or already patched. Raise before writing if any anchor is off."""
    planned: dict[Path, str] = {}
    already = {path for path, _ in edits(repo, verl) if MARKER in path.read_text()}
    for path, edit in edits(repo, verl):
        if path in already:
            planned[path] = path.read_text()
            continue
        text = planned.get(path, path.read_text())
        found = text.count(edit.anchor)
        if found != edit.count:
            raise SystemExit(
                f"{path}: anchor for {edit.path} occurs {found} times, expected {edit.count}; upstream changed"
            )
        if edit.replace:
            text = text.replace(edit.anchor, edit.text)
        elif edit.before:
            text = text.replace(edit.anchor, edit.text + edit.anchor)
        else:
            text = text.replace(edit.anchor, edit.anchor + edit.text)
        planned[path] = text
    report = []
    for path, text in planned.items():
        if path.read_text() == text:
            report.append(f"{path}: already patched")
            continue
        path.write_text(text)
        report.append(f"{path}: patched")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--verl", type=Path, required=True)
    arguments = parser.parse_args(argv)
    for line in apply(arguments.repo, arguments.verl):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
