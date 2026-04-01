"""Decision-layer state machine for stable HUD-facing audio events."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import (
    ACTIONABLE_LABELS,
    ACTIONABLE_PRIORITY_MAP,
    CRITICAL_LABELS,
    DECISION_AWARENESS_CONF_THRESH,
    DECISION_EMA_LAMBDA,
    DECISION_ENTER_THRESH,
    DECISION_EXIT_THRESH,
    DECISION_IDLE_CONF_THRESH,
    DECISION_MIN_HOLD_S,
    DECISION_NOISE_FLOOR_INIT_DBFS,
    DECISION_OOD_ENTROPY_THRESH,
    DECISION_OOD_MIN_MARGIN,
    DECISION_OOD_MIN_TOP_PROB,
    DECISION_PRIORITY_SWITCH_MARGIN,
    DECISION_TOPK,
    DECISION_SILENCE_MARGIN_DB,
    IDLE_LABELS,
    MODEL_SILENCE_LABEL,
    NON_ACTIONABLE_LABELS,
    SPEECH_LABEL,
)
from .spatial_audio import SpatialSnapshot


PRIORITY_RANK = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}


@dataclass(slots=True)
class DecisionSnapshot:
    """One stabilized event snapshot emitted after EMA + hysteresis + OOD gate."""

    label: str
    confidence: float
    raw_label: str
    raw_confidence: float
    state: str
    hud_action: str
    priority: str
    onset_unix: float | None
    offset_unix: float | None
    entropy: float
    entropy_norm: float
    margin: float
    is_silence: bool
    is_ood: bool
    is_actionable: bool
    noise_floor_dbfs: float
    spl_dbfs: float
    direction: str
    direction_confidence: float
    speech_prob: float
    top5: list[dict]

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": float(self.confidence),
            "raw_label": self.raw_label,
            "raw_confidence": float(self.raw_confidence),
            "state": self.state,
            "hud_action": self.hud_action,
            "priority": self.priority,
            "onset_unix": None if self.onset_unix is None else float(self.onset_unix),
            "offset_unix": None if self.offset_unix is None else float(self.offset_unix),
            "entropy": float(self.entropy),
            "entropy_norm": float(self.entropy_norm),
            "margin": float(self.margin),
            "is_silence": bool(self.is_silence),
            "is_ood": bool(self.is_ood),
            "is_actionable": bool(self.is_actionable),
            "noise_floor_dbfs": float(self.noise_floor_dbfs),
            "spl_dbfs": float(self.spl_dbfs),
            "direction": self.direction,
            "direction_confidence": float(self.direction_confidence),
            "speech_prob": float(self.speech_prob),
            "top5": self.top5,
        }


class PriorityDecisionLayer:
    """
    EMA + hysteresis + OOD gate + priority:
    ham p(c|z) skorlarini HUD'a dogrudan baglamak yerine once kararlastir.
    """

    def __init__(self) -> None:
        self._labels: list[str] = []
        self._ema_probs: np.ndarray | None = None
        self.noise_floor_dbfs = float(DECISION_NOISE_FLOOR_INIT_DBFS)

        self._active_label: str | None = None
        self._active_since_unix: float | None = None
        self._last_offset_unix: float | None = None

    def update(
        self,
        labels: list[str],
        probabilities: np.ndarray,
        spl_dbfs: float,
        spatial: SpatialSnapshot,
        timestamp_unix: float,
    ) -> DecisionSnapshot:
        if len(labels) != int(probabilities.size):
            raise ValueError("labels/probabilities size mismatch")

        probs = np.asarray(probabilities, dtype=np.float64)
        total = float(np.sum(probs))
        if total <= 0.0:
            probs = np.full(len(labels), 1.0 / max(len(labels), 1), dtype=np.float64)
        else:
            probs = probs / total

        if labels != self._labels or self._ema_probs is None or self._ema_probs.shape[0] != len(labels):
            self._labels = list(labels)
            self._ema_probs = probs.copy()
        else:
            self._ema_probs = DECISION_EMA_LAMBDA * self._ema_probs + (1.0 - DECISION_EMA_LAMBDA) * probs
            self._ema_probs = self._ema_probs / (float(np.sum(self._ema_probs)) + 1e-12)

        order = np.argsort(self._ema_probs)[::-1]
        top_index = int(order[0])
        top_label = self._labels[top_index]
        top_prob = float(self._ema_probs[top_index])
        raw_top_label = self._labels[int(np.argmax(probs))]
        raw_top_prob = float(np.max(probs))
        semantic_label, semantic_prob = self._pick_semantic_candidate(order)
        actionable_label, actionable_prob = self._pick_actionable_candidate(order)

        second_prob = float(self._ema_probs[int(order[1])]) if len(order) > 1 else 0.0
        margin = top_prob - second_prob
        entropy = _entropy(self._ema_probs)
        entropy_norm = _normalized_entropy(self._ema_probs)
        top5 = [
            {"label": self._labels[index], "prob": float(self._ema_probs[index])}
            for index in order[: min(DECISION_TOPK, len(order))]
        ]

        speech_prob = self._prob_for(SPEECH_LABEL)
        self._update_noise_floor(spl_dbfs, top_label)

        is_silence = (
            spl_dbfs <= (self.noise_floor_dbfs + DECISION_SILENCE_MARGIN_DB)
            or (top_label == MODEL_SILENCE_LABEL and top_prob >= DECISION_EXIT_THRESH)
        )
        is_ood = (entropy_norm >= DECISION_OOD_ENTROPY_THRESH and top_prob < DECISION_ENTER_THRESH) or (
            top_prob < DECISION_OOD_MIN_TOP_PROB and margin < DECISION_OOD_MIN_MARGIN
        )
        is_actionable = actionable_label is not None and actionable_prob >= DECISION_ENTER_THRESH

        if is_ood:
            self._deactivate(timestamp_unix)
            return self._snapshot(
                label="ood",
                confidence=top_prob,
                raw_label=raw_top_label,
                raw_confidence=raw_top_prob,
                state="ood",
                hud_action="suppress",
                priority="none",
                onset_unix=None,
                offset_unix=self._last_offset_unix,
                entropy=entropy,
                entropy_norm=entropy_norm,
                margin=margin,
                is_silence=False,
                is_ood=True,
                is_actionable=False,
                spl_dbfs=spl_dbfs,
                spatial=spatial,
                speech_prob=speech_prob,
                top5=top5,
            )

        if self._active_label is not None:
            active_prob = self._prob_for(self._active_label)
            active_priority = _priority_for(self._active_label)
            candidate_priority = _priority_for(actionable_label or "")
            priority_jump = (
                PRIORITY_RANK.get(candidate_priority, 0) > PRIORITY_RANK.get(active_priority, 0)
                and actionable_prob >= (DECISION_ENTER_THRESH + DECISION_PRIORITY_SWITCH_MARGIN)
            )

            if priority_jump:
                self._activate(actionable_label, timestamp_unix)
                return self._active_snapshot(
                    actionable_label,
                    actionable_prob,
                    raw_top_label,
                    raw_top_prob,
                    entropy,
                    entropy_norm,
                    margin,
                    spl_dbfs,
                    spatial,
                    speech_prob,
                    top5,
                    state="active",
                    hud_action=_hud_action_for(actionable_label),
                )

            if active_prob >= DECISION_EXIT_THRESH and not is_silence:
                return self._active_snapshot(
                    self._active_label,
                    active_prob,
                    raw_top_label,
                    raw_top_prob,
                    entropy,
                    entropy_norm,
                    margin,
                    spl_dbfs,
                    spatial,
                    speech_prob,
                    top5,
                    state="active",
                    hud_action=_hud_action_for(self._active_label),
                )

            if self._active_since_unix is not None and (timestamp_unix - self._active_since_unix) < DECISION_MIN_HOLD_S:
                return self._active_snapshot(
                    self._active_label,
                    active_prob,
                    raw_top_label,
                    raw_top_prob,
                    entropy,
                    entropy_norm,
                    margin,
                    spl_dbfs,
                    spatial,
                    speech_prob,
                    top5,
                    state="hold",
                    hud_action="hold",
                )

            self._deactivate(timestamp_unix)

        if is_silence:
            return self._snapshot(
                label="silence",
                confidence=top_prob,
                raw_label=raw_top_label,
                raw_confidence=raw_top_prob,
                state="silence",
                hud_action="neutral",
                priority="none",
                onset_unix=None,
                offset_unix=self._last_offset_unix,
                entropy=entropy,
                entropy_norm=entropy_norm,
                margin=margin,
                is_silence=True,
                is_ood=False,
                is_actionable=False,
                spl_dbfs=spl_dbfs,
                spatial=spatial,
                speech_prob=speech_prob,
                top5=top5,
            )

        if is_actionable:
            self._activate(actionable_label, timestamp_unix)
            return self._active_snapshot(
                actionable_label,
                actionable_prob,
                raw_top_label,
                raw_top_prob,
                entropy,
                entropy_norm,
                margin,
                spl_dbfs,
                spatial,
                speech_prob,
                top5,
                state="active",
                hud_action=_hud_action_for(actionable_label),
            )

        if semantic_label is None:
            return self._snapshot(
                label="idle",
                confidence=top_prob,
                raw_label=raw_top_label,
                raw_confidence=raw_top_prob,
                state="idle",
                hud_action="neutral",
                priority="none",
                onset_unix=None,
                offset_unix=self._last_offset_unix,
                entropy=entropy,
                entropy_norm=entropy_norm,
                margin=margin,
                is_silence=False,
                is_ood=False,
                is_actionable=False,
                spl_dbfs=spl_dbfs,
                spatial=spatial,
                speech_prob=speech_prob,
                top5=top5,
            )

        if semantic_label in IDLE_LABELS and semantic_prob < DECISION_IDLE_CONF_THRESH:
            return self._snapshot(
                label="idle",
                confidence=semantic_prob,
                raw_label=raw_top_label,
                raw_confidence=raw_top_prob,
                state="idle",
                hud_action="neutral",
                priority="none",
                onset_unix=None,
                offset_unix=self._last_offset_unix,
                entropy=entropy,
                entropy_norm=entropy_norm,
                margin=margin,
                is_silence=False,
                is_ood=False,
                is_actionable=False,
                spl_dbfs=spl_dbfs,
                spatial=spatial,
                speech_prob=speech_prob,
                top5=top5,
            )

        if semantic_label in IDLE_LABELS:
            return self._snapshot(
                label=semantic_label,
                confidence=semantic_prob,
                raw_label=raw_top_label,
                raw_confidence=raw_top_prob,
                state="ambient",
                hud_action="ambient",
                priority="none",
                onset_unix=None,
                offset_unix=self._last_offset_unix,
                entropy=entropy,
                entropy_norm=entropy_norm,
                margin=margin,
                is_silence=False,
                is_ood=False,
                is_actionable=False,
                spl_dbfs=spl_dbfs,
                spatial=spatial,
                speech_prob=speech_prob,
                top5=top5,
            )

        state_label = semantic_label if semantic_label not in NON_ACTIONABLE_LABELS else semantic_label
        return self._snapshot(
            label=state_label,
            confidence=semantic_prob,
            raw_label=raw_top_label,
            raw_confidence=raw_top_prob,
            state="aware",
            hud_action=("caption" if semantic_label == SPEECH_LABEL else "context"),
            priority="none",
            onset_unix=None,
            offset_unix=self._last_offset_unix,
            entropy=entropy,
            entropy_norm=entropy_norm,
            margin=margin,
            is_silence=False,
            is_ood=False,
            is_actionable=False,
            spl_dbfs=spl_dbfs,
            spatial=spatial,
            speech_prob=speech_prob,
            top5=top5,
        )

    def _update_noise_floor(self, spl_dbfs: float, top_label: str) -> None:
        if top_label in ACTIONABLE_LABELS or spl_dbfs > (self.noise_floor_dbfs + 18.0):
            return
        if spl_dbfs <= self.noise_floor_dbfs:
            alpha = 0.78
        else:
            alpha = 0.992
        self.noise_floor_dbfs = alpha * self.noise_floor_dbfs + (1.0 - alpha) * float(spl_dbfs)

    def _prob_for(self, label: str) -> float:
        if self._ema_probs is None or label not in self._labels:
            return 0.0
        return float(self._ema_probs[self._labels.index(label)])

    def _pick_semantic_candidate(self, order: np.ndarray) -> tuple[str | None, float]:
        for index in order:
            label = self._labels[int(index)]
            prob = float(self._ema_probs[int(index)])
            if label in {MODEL_SILENCE_LABEL, "other"}:
                continue
            if prob >= DECISION_AWARENESS_CONF_THRESH:
                return label, prob
        return None, 0.0

    def _pick_actionable_candidate(self, order: np.ndarray) -> tuple[str | None, float]:
        for index in order:
            label = self._labels[int(index)]
            if label not in ACTIONABLE_LABELS:
                continue
            return label, float(self._ema_probs[int(index)])
        return None, 0.0

    def _activate(self, label: str, timestamp_unix: float) -> None:
        if self._active_label != label:
            self._active_since_unix = timestamp_unix
        self._active_label = label

    def _deactivate(self, timestamp_unix: float) -> None:
        if self._active_label is not None:
            self._last_offset_unix = timestamp_unix
        self._active_label = None
        self._active_since_unix = None

    def _active_snapshot(
        self,
        label: str,
        confidence: float,
        raw_label: str,
        raw_confidence: float,
        entropy: float,
        entropy_norm: float,
        margin: float,
        spl_dbfs: float,
        spatial: SpatialSnapshot,
        speech_prob: float,
        top5: list[dict],
        state: str,
        hud_action: str,
    ) -> DecisionSnapshot:
        return self._snapshot(
            label=label,
            confidence=confidence,
            raw_label=raw_label,
            raw_confidence=raw_confidence,
            state=state,
            hud_action=hud_action,
            priority=_priority_for(label),
            onset_unix=self._active_since_unix,
            offset_unix=self._last_offset_unix,
            entropy=entropy,
            entropy_norm=entropy_norm,
            margin=margin,
            is_silence=False,
            is_ood=False,
            is_actionable=True,
            spl_dbfs=spl_dbfs,
            spatial=spatial,
            speech_prob=speech_prob,
            top5=top5,
        )

    def _snapshot(
        self,
        *,
        label: str,
        confidence: float,
        raw_label: str,
        raw_confidence: float,
        state: str,
        hud_action: str,
        priority: str,
        onset_unix: float | None,
        offset_unix: float | None,
        entropy: float,
        entropy_norm: float,
        margin: float,
        is_silence: bool,
        is_ood: bool,
        is_actionable: bool,
        spl_dbfs: float,
        spatial: SpatialSnapshot,
        speech_prob: float,
        top5: list[dict],
    ) -> DecisionSnapshot:
        return DecisionSnapshot(
            label=label,
            confidence=float(confidence),
            raw_label=raw_label,
            raw_confidence=float(raw_confidence),
            state=state,
            hud_action=hud_action,
            priority=priority,
            onset_unix=onset_unix,
            offset_unix=offset_unix,
            entropy=float(entropy),
            entropy_norm=float(entropy_norm),
            margin=float(margin),
            is_silence=is_silence,
            is_ood=is_ood,
            is_actionable=is_actionable,
            noise_floor_dbfs=float(self.noise_floor_dbfs),
            spl_dbfs=float(spl_dbfs),
            direction=spatial.direction,
            direction_confidence=float(spatial.direction_confidence),
            speech_prob=float(speech_prob),
            top5=top5,
        )


def _entropy(probs: np.ndarray) -> float:
    values = np.asarray(probs, dtype=np.float64)
    safe = np.clip(values, 1e-12, 1.0)
    return float(-np.sum(safe * np.log(safe)))


def _normalized_entropy(probs: np.ndarray) -> float:
    if probs.size <= 1:
        return 0.0
    return float(_entropy(probs) / math.log(float(probs.size)))


def _priority_for(label: str) -> str:
    return ACTIONABLE_PRIORITY_MAP.get(label, "none")


def _hud_action_for(label: str) -> str:
    return "critical" if label in CRITICAL_LABELS else "show"
