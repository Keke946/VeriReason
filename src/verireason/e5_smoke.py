"""Private, fail-closed E5 capture/audit/update journal. No model or CUDA import."""

from __future__ import annotations

import fcntl
import json
import math
import os
import subprocess
import time
import uuid
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from verireason.e5_credit import TokenCredit
from verireason.e5_interface import qwen_token_credit, terminal_advantages

EXPERIMENT = "e5_capture_audit_smoke_001"


def validate_selection(rows: list[dict]) -> None:
    if len(rows) != 8 or len({r["example_id"] for r in rows}) != 8:
        raise ValueError("exactly eight distinct training examples required")
    for r in rows:
        if r.get("split") != "train" or r.get("operational_eligible") is not True:
            raise ValueError("validation/heldout/ineligible input forbidden")
        if r.get("source") == "heldout":
            raise ValueError("heldout input forbidden")
        if not all(isinstance(r.get(k), str) and r[k] for k in (
            "example_id", "prompt", "gold_sql", "database_path"
        )):
            raise ValueError("training row identity or payload missing")


def validate_record(r: dict, row: dict, group: int, rollout: int) -> None:
    if (r.get("example_id"), r.get("group"), r.get("rollout")) != (
        row["example_id"], group, rollout
    ):
        raise ValueError("capture identity/order drift")
    if not isinstance(r.get("text"), str) or r.get("reward") not in (0, 1):
        raise ValueError("invalid completion/reward")
    if type(r.get("pad_id")) is not int or not isinstance(r.get("terminal_status"), str):
        raise ValueError("missing padding/terminal metadata")
    ids, mask, old = (r.get(k) for k in ("completion_ids", "completion_mask", "old_logps"))
    prompt_ids = r.get("prompt_ids")
    if not prompt_ids or any(type(x) is not int or x < 0 for x in prompt_ids):
        raise ValueError("missing original prompt IDs")
    if not ids or not len(ids) == len(mask) == len(old):
        raise ValueError("capture token rows misaligned")
    if any(type(x) is not int or x < 0 for x in ids) or any(x not in (0, 1) for x in mask):
        raise ValueError("invalid original IDs/mask")
    if any(not math.isfinite(x) or x > 0 for x in old):
        raise ValueError("nonfinite/invalid old log-probability")
    if r.get("old_logps_dtype") not in {"float32", "float64", "float16", "bfloat16"}:
        raise ValueError("original old-logp dtype required")
    if type(r.get("truncated")) is not bool or type(r.get("eos_id")) is not int:
        raise ValueError("missing EOS/truncation metadata")
    if r["truncated"] and any(mask):
        raise ValueError("truncated completion must retain E2 zero loss mask")
    n = sum(mask)
    if mask != [1] * n + [0] * (len(mask) - n):
        raise ValueError("noncontiguous original mask")


class PrivateJournal:
    """One atomic, fsynced 0600 journal under a 0700 directory, with an exclusive lease.

    Intent is durable BEFORE generation or an optimizer side effect. An interrupted intent
    is never replayed automatically. No rollback claims or repeated opaque generation.
    """

    def __init__(self, directory: Path):
        directory = Path(directory)
        if directory.is_symlink():
            raise ValueError("private directory cannot be a symlink")
        self.directory = directory.resolve()
        repo = Path(__file__).resolve().parents[2]
        if self.directory.is_relative_to(repo):
            ignored = subprocess.run(
                ["git", "check-ignore", "-q", str(self.directory / "replay.json")], cwd=repo
            )
            if ignored.returncode != 0:
                raise ValueError("private replay destination must be ignored/outside Git")
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.directory.stat().st_mode & 0o077:
            raise ValueError("private evidence directory must be mode 0700")
        self.path = self.directory / "replay.json"
        self.lock = os.open(self.directory / "session.lock", os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self.lock)
            raise

    def close(self):
        if self.lock is not None:
            os.close(self.lock)
            self.lock = None

    def read(self):
        if self.path.stat().st_mode & 0o077:
            raise ValueError("private replay must be mode 0600")
        with self.path.open() as f:
            return json.load(f)

    def write(self, value):
        data = json.dumps(value, allow_nan=False).encode()
        temporary = self.directory / "replay.partial"
        # A stale partial is evidence of an interrupted write, never an executable checkpoint.
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, self.path)
            dirfd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
        except BaseException:
            # Leave partial evidence. Resume must stop rather than guess write completion.
            raise


class SmokeSession:
    def __init__(self, directory: Path, rows: list[dict], context: dict):
        validate_selection(rows)
        self.journal = PrivateJournal(directory)
        if (self.journal.directory / "replay.partial").exists():
            self.journal.close()
            raise RuntimeError("partial evidence: unsafe to resume")
        if self.journal.path.exists():
            self.data = self.journal.read()
            if self.data["selection"] != rows or self.data["context"] != context:
                self.close()
                raise ValueError("resume selection/parent/source drift")
            if self.data["pending"] is not None or self.data["phase"] in {
                "updating", "finishing"
            }:
                self.close()
                raise RuntimeError("uncertain side effect: stop, never replay")
            for i, record in enumerate(self.data["records"]):
                validate_record(record, rows[i // 4], i // 4, i % 4)
        else:
            self.data = dict(
                schema_version=1, experiment_id=EXPERIMENT, phase="preflight",
                session_id=str(uuid.uuid4()),
                selection=rows, context=context, records=[], audit=[], gate=None,
                optimizer_steps=0, pending=None,
                started_at=context.get("paid_started_at", time.time()), updates=[],
            )
            self.journal.write(self.data)

    def close(self):
        self.journal.close()

    def save(self):
        self.journal.write(self.data)

    def check_deadline(self):
        if time.time() - self.data["started_at"] >= 3600:
            raise RuntimeError("cumulative_deadline")

    def audit(self, backend):
        if self.data["optimizer_steps"] or backend.optimizer_steps:
            raise RuntimeError("audit must precede all optimizer updates")
        records = self.data["records"]
        advantages = terminal_advantages(tuple(r["reward"] for r in records))
        audit, credits = [], []
        for i, (r, a) in enumerate(zip(records, advantages, strict=True)):
            self.check_deadline()
            credit = backend.localize(r, self.data["selection"][i // 4])
            if credit.terminal_reward != r["reward"]:
                raise ValueError("terminal reward drift")
            kwargs = dict(eos_id=r["eos_id"], truncated=r["truncated"])
            args = (credit, r["text"], tuple(r["completion_ids"]),
                    tuple(r["completion_mask"]), backend.tokenizer)
            geometry = qwen_token_credit(*args, terminal_advantage=-1.0, **kwargs)
            actual = qwen_token_credit(*args, terminal_advantage=a, **kwargs)
            k = sum(m * v for m, v in zip(geometry.mask, r["completion_mask"], strict=True))
            mapped = credit.span is not None and geometry.reason == "localized" and k > 0
            n = sum(r["completion_mask"])
            audit.append(dict(
                row=i, semantic=credit.span is not None, mapped=mapped,
                useful=mapped and a < 0, clause=credit.span.clause if credit.span else None,
                character_span=asdict(credit.span) if credit.span else None,
                semantic_reason=credit.reason, token_reason=actual.reason,
                advantage=a, target_tokens=k if mapped else 0, valid_tokens=n,
                n_over_k=n / k if mapped else None,
                credit_mass_ratio=k / n if mapped and a < 0 else 1.0,
                mask=list(actual.mask), advantages=list(actual.advantages),
            ))
            credits.append(actual)
        self.data["audit"] = audit
        backend.check_numerics(records, tuple(credits))
        self.data["numerics"] = backend.numerics
        self.data["phase"] = "awaiting_review"
        self.save()

    def decide(self, review):
        """Frozen 2026-10-21 thresholds; no runtime override or outcome filtering."""
        d, audit = self.data, self.data["audit"]
        if d["phase"] != "awaiting_review" or len(audit) != 32 or d["optimizer_steps"]:
            raise RuntimeError("gate must follow complete pre-update audit")
        rewards = [r["reward"] for r in d["records"]]
        w = rewards.count(0)
        s = sum(x["semantic"] for x in audit)
        m = sum(x["mapped"] for x in audit)
        u = sum(x["useful"] for x in audit)
        mixed = sum(len(set(rewards[i:i + 4])) == 2 for i in range(0, 32, 4))
        candidates = [a["row"] for a in audit if a["semantic"]]
        reviewed = (
            isinstance(review, dict) and bool(review.get("reviewer"))
            and review.get("session_id") == d["session_id"]
            and review.get("candidate_rows") == candidates
            and review.get("wrong_rows") == [] and review.get("unresolved_rows") == []
        )
        checks = dict(
            complete=len(d["records"]) == 32, signal=w >= 4 and mixed >= 2,
            coverage=m >= 2 and w > 0 and m / w >= .2,
            abstention=w > 0 and (w - m) / w <= .8,
            token_mapping=s > 0 and m / s >= .8,
            useful=u >= 2 and len({a["row"] // 4 for a in audit if a["useful"]}) >= 2,
            attribution=bool(reviewed),
        )
        d["gate"] = dict(
            passed=all(checks.values()), checks=checks, W=w, S=s, M=m, U=u,
            mixed_groups=mixed, coverage=m / w if w else None,
            abstention=(w - m) / w if w else None,
            token_mapping=m / s if s else None,
            reasons=dict(Counter(a["semantic_reason"] for a in audit)), review=review,
            optimizer_steps_before_gate=d["optimizer_steps"],
        )
        d["phase"] = "gated" if d["gate"]["passed"] else "stopped"
        self.save()

    def before_update(self, index):
        self.check_deadline()
        d = self.data
        if (d["phase"] not in {"gated", "updating"} or not d["gate"]["passed"]
                or d["pending"] is not None or index != d["optimizer_steps"] or index >= 8):
            raise RuntimeError("optimizer before gate, duplicate or excess update")
        d["phase"], d["pending"] = "updating", {"update": index}
        self.save()  # must succeed before the optimizer side effect

    def after_update(self, index, metrics):
        d = self.data
        if d["pending"] != {"update": index} or index != d["optimizer_steps"]:
            raise RuntimeError("optimizer evidence order drift")
        if not metrics or any(not math.isfinite(x) for x in metrics.values()):
            raise ValueError("nonfinite update evidence")
        d["updates"].append(metrics)
        d["optimizer_steps"] += 1
        d["pending"] = None
        self.save()

    def run(self, backend, review=None):
        """Backend side effects run only after a durable intent; exceptions never retry."""
        d = self.data
        try:
            self.check_deadline()
            if d["phase"] in {"complete", "stopped"}:
                return self.summary()
            backend.preflight(d["context"])
            if d["optimizer_steps"] != 0 or backend.optimizer_steps != 0:
                raise RuntimeError("pre-audit optimizer count must be zero")
            if d["phase"] in {"preflight", "capturing"}:
                if len(d["records"]) % 4:
                    raise ValueError("partial group cannot resume")
                for group in range(len(d["records"]) // 4, 8):
                    self.check_deadline()
                    d["phase"], d["pending"] = "capturing", {"generation": group}
                    self.save()
                    result = backend.capture_group(d["selection"][group], group)
                    if len(result) != 4 or backend.optimizer_steps != 0:
                        raise ValueError("capture count or optimizer drift")
                    for j, r in enumerate(result):
                        validate_record(r, d["selection"][group], group, j)
                    d["records"].extend(result)
                    d["pending"] = None
                    self.save()
                d["phase"] = "captured"
                self.save()
            if d["phase"] == "captured":
                self.audit(backend)
            if d["phase"] == "awaiting_review":
                if review is None:
                    return self.summary()  # no implicit review; unload/stop resources
                self.decide(review)
            if d["phase"] == "gated":
                credits = tuple(TokenCredit(tuple(a["advantages"]), tuple(a["mask"]),
                                            a["token_reason"]) for a in d["audit"])
                backend.update(d["records"], credits, self)
                if d["optimizer_steps"] != 8 or backend.optimizer_steps != 8:
                    raise RuntimeError("incomplete update evidence")
                d["phase"] = "finishing"
                self.save()
                result = backend.finish(self.journal.directory)
                if not (math.isfinite(result["adapter_delta_l2"])
                        and result["adapter_delta_l2"] > 0 and result["independent_reload"]):
                    raise RuntimeError("adapter delta or independent reload failed")
                d["finish"] = result
                d["phase"] = "complete"
                self.save()
        except BaseException as error:
            # Never log raw exception strings: SQL, prompts and paths may be embedded.
            d["phase"] = "stopped"
            d["stop_reason"] = (
                "oom" if "outofmemory" in type(error).__name__.lower()
                else type(error).__name__
            )
            try:
                self.save()
            except OSError:
                pass  # last durable intent/partial remains unsafe-to-resume evidence
            raise
        return self.summary()

    def summary(self):
        d = self.data
        gate = d["gate"]
        return dict(
            experiment_id=EXPERIMENT, phase=d["phase"], optimizer_steps=d["optimizer_steps"],
            rollouts=len(d["records"]), gate_passed=gate["passed"] if gate else None,
            coverage=gate["coverage"] if gate else None,
            validation_payloads_read=0, heldout_payloads_read=0,
            stop_reason=d.get("stop_reason"),
            uncertain_side_effect=d["pending"] is not None,
        )
