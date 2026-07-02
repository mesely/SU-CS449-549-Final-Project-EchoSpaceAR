"""Helpers for optional BEME integration inside the semantic audio pipeline."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


AutoBEME: type[Any] | None = None
_get_market_prices: Any | None = None
_BEME_IMPORT_ERROR: Exception | None = None


def _candidate_repo_roots() -> list[Path]:
    """Return likely local checkout locations for the sibling BEME repo."""
    here = Path(__file__).resolve()
    return [
        Path(os.environ.get("BEME_REPO_DIR", "")).expanduser() if os.environ.get("BEME_REPO_DIR") else None,
        here.parents[4] / "beme",
        Path.home() / "beme",
    ]


def ensure_beme_available() -> None:
    """Import the local or installed BEME package, adding a sibling repo if needed."""
    global AutoBEME, _get_market_prices, _BEME_IMPORT_ERROR
    if AutoBEME is not None and _get_market_prices is not None:
        return

    try:
        from beme import AutoBEME as _AutoBEME
        from beme.pricing import get_market_prices as _pricing_helper
    except Exception as first_error:
        _BEME_IMPORT_ERROR = first_error
        for module_name in list(sys.modules):
            if module_name == "beme" or module_name.startswith("beme."):
                sys.modules.pop(module_name, None)
        for candidate in _candidate_repo_roots():
            if candidate is None:
                continue
            package_init = candidate / "beme" / "__init__.py"
            if not package_init.exists():
                continue
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            try:
                from beme import AutoBEME as _AutoBEME
                from beme.pricing import get_market_prices as _pricing_helper
                break
            except Exception as retry_error:
                _BEME_IMPORT_ERROR = retry_error
                for module_name in list(sys.modules):
                    if module_name == "beme" or module_name.startswith("beme."):
                        sys.modules.pop(module_name, None)
        else:
            raise RuntimeError(
                "BEME package is not available. Install it or point BEME_REPO_DIR at the local repo."
            ) from _BEME_IMPORT_ERROR

    AutoBEME = _AutoBEME
    _get_market_prices = _pricing_helper


def make_autobeme(mode: str = "balanced", n_funds: int = 12):
    """Construct an AutoBEME model after resolving the optional dependency."""
    ensure_beme_available()
    return AutoBEME(mode=mode, n_funds=n_funds)


def _normalize_probabilities(raw: Any) -> np.ndarray:
    """Convert assorted sklearn predict_proba outputs into a dense [N, K] array."""
    if isinstance(raw, list):
        n_samples = int(raw[0].shape[0]) if raw else 0
        columns: list[np.ndarray] = []
        for item in raw:
            arr = np.asarray(item, dtype=np.float32)
            if arr.ndim == 1:
                columns.append(arr.astype(np.float32, copy=False))
            elif arr.shape[1] >= 2:
                columns.append(arr[:, 1].astype(np.float32, copy=False))
            else:
                columns.append(np.zeros(n_samples, dtype=np.float32))
        return np.column_stack(columns).astype(np.float32, copy=False)

    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 3:
        return arr[:, :, -1].astype(np.float32, copy=False)
    if arr.ndim == 1:
        return arr.reshape(-1, 1).astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def _collect_agent_probabilities(agent_model: Any, X: np.ndarray) -> np.ndarray:
    """Read probabilities from a BEME fund model or its wrapped estimator."""
    if hasattr(agent_model, "predict_proba"):
        return _normalize_probabilities(agent_model.predict_proba(X))

    for attr in ("estimator", "base_estimator", "model", "clf", "_estimator", "classifier"):
        if hasattr(agent_model, attr):
            inner = getattr(agent_model, attr)
            if hasattr(inner, "predict_proba"):
                return _normalize_probabilities(inner.predict_proba(X))

    raise RuntimeError("Could not extract predict_proba output from a BEME agent.")


def _predict_balanced_market_probabilities(model: Any, X: np.ndarray) -> np.ndarray:
    """Return the Apocalypse ensemble's capital-weighted class probabilities."""
    funds = getattr(model.market, "funds", [])
    if not funds:
        raise RuntimeError("AutoBEME model has no trained funds.")

    n_samples = int(X.shape[0])
    n_classes = len(funds[0]["active_classes"])
    total_balance = float(sum(float(fund["balance"]) for fund in funds))
    if total_balance <= 0.0:
        raise RuntimeError("AutoBEME market has non-positive total balance.")

    market_probs = np.zeros((n_samples, n_classes), dtype=np.float32)
    for fund in funds:
        probabilities = _collect_agent_probabilities(fund["model"], X)
        if probabilities.shape[1] != n_classes:
            raise RuntimeError(
                f"BEME agent probability width mismatch: expected {n_classes}, got {probabilities.shape[1]}"
            )
        market_probs += probabilities * (float(fund["balance"]) / total_balance)
    return np.clip(market_probs, 0.0, 1.0).astype(np.float32, copy=False)


def predict_autobeme_proba(model: Any, X: np.ndarray) -> np.ndarray:
    """Expose soft class probabilities for a trained AutoBEME instance."""
    ensure_beme_available()
    if getattr(model, "market", None) is None:
        raise RuntimeError("AutoBEME model must be fitted before probability inference.")

    if getattr(model, "mode", "").lower() == "balanced":
        probabilities = _predict_balanced_market_probabilities(model, X)
    else:
        funds = list(model.market.funds)
        if not funds:
            raise RuntimeError("AutoBEME model has no trained funds.")
        agent_predictions = np.stack(
            [np.clip(_collect_agent_probabilities(fund["model"], X), 0.01, 0.99) for fund in funds],
            axis=0,
        )
        per_sample = [
            _get_market_prices(funds, agent_predictions[:, sample_index, :], getattr(model.market, "T", 1.0))
            for sample_index in range(int(X.shape[0]))
        ]
        probabilities = np.asarray(per_sample, dtype=np.float32)

    if getattr(model, "_is_1d_target", False) and probabilities.ndim == 2 and probabilities.shape[1] >= 2:
        return probabilities[:, 1].reshape(-1, 1).astype(np.float32, copy=False)
    return probabilities.astype(np.float32, copy=False)
