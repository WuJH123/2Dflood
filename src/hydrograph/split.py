from __future__ import annotations

import random
from dataclasses import dataclass, asdict
from typing import Iterable

from .data import EventRef


@dataclass
class SplitManifest:
    protocol: str
    target_model: str | None
    seed: int
    train: list[EventRef]
    val: list[EventRef]
    test: list[EventRef]

    def validate_no_leakage(self) -> None:
        def keys(refs: Iterable[EventRef]):
            return {(r.model_id, r.event_id, r.split) for r in refs}
        a, b, c = keys(self.train), keys(self.val), keys(self.test)
        if a & b or a & c or b & c:
            raise ValueError("Event leakage detected between train/val/test splits")
        if self.protocol == "leave_one_city_out" and self.target_model:
            if any(r.model_id == self.target_model for r in self.train):
                raise ValueError("LOCO leakage: target model appears in source-city training set")
            if any(r.model_id != self.target_model for r in self.test):
                raise ValueError("LOCO contract: test set must contain only the held-out target model")

    def serializable(self) -> dict:
        def conv(xs):
            return [{"model_id": x.model_id, "split": x.split, "event_id": x.event_id, "path": str(x.path)} for x in xs]
        return {"protocol": self.protocol, "target_model": self.target_model, "seed": self.seed,
                "train": conv(self.train), "val": conv(self.val), "test": conv(self.test)}


def make_manifest(
    refs: list[EventRef],
    protocol: str,
    seed: int,
    target_model: str | None = None,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
) -> SplitManifest:
    if not refs:
        raise ValueError("No events supplied to split")
    rng = random.Random(seed)
    models = sorted({r.model_id for r in refs})
    if protocol == "leave_one_city_out":
        if target_model is None:
            if len(models) < 2:
                raise ValueError("leave_one_city_out requires >=2 city/model domains")
            target_model = models[-1]
        if target_model not in models:
            raise ValueError(f"Unknown target_model={target_model}; choices={models}")
        source = [r for r in refs if r.model_id != target_model]
        target = [r for r in refs if r.model_id == target_model]
        rng.shuffle(source)
        rng.shuffle(target)
        nval = max(1, int(round(len(source) * val_fraction))) if len(source) > 1 else 0
        train, val = source[nval:], source[:nval]
        test = target
    elif protocol == "event_holdout":
        work = refs[:]
        rng.shuffle(work)
        ntest = max(1, int(round(len(work) * test_fraction)))
        nval = max(1, int(round(len(work) * val_fraction)))
        test = work[:ntest]
        val = work[ntest:ntest+nval]
        train = work[ntest+nval:]
    else:
        raise ValueError(f"Unsupported split protocol: {protocol}")
    if not train or not test:
        raise ValueError("Split produced an empty train or test set")
    m = SplitManifest(protocol, target_model, seed, train, val, test)
    m.validate_no_leakage()
    return m


def few_shot_target_split(target_refs: list[EventRef], k: int, seed: int) -> tuple[list[EventRef], list[EventRef]]:
    if k < 0:
        raise ValueError("k must be non-negative")
    work = target_refs[:]
    random.Random(seed).shuffle(work)
    k = min(k, max(0, len(work) - 1))
    return work[:k], work[k:]
