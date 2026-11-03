"""Spike: is a time-series foundation model worth adopting here?

Compares zero-shot TimesFM against the project's own rolling-origin baselines on
the **same** protocol. A foundation model that cannot beat `naive` on this
dataset earns nothing, regardless of its benchmark paper.

The script never fabricates a result: if the model cannot be loaded (no weights,
no network, incompatible torch), it records the failure verbatim and exits 0 with
a report that says so.

Usage:  python tools/spike_tsfm.py
"""

from __future__ import annotations

import sys
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

N_POINTS = 260
SEED = 20240501
HORIZON = 1
ORIGINS = 5
REPORT = Path("docs/SPIKE-TSFM.md")


@dataclass
class Attempt:
    name: str
    ok: bool
    detail: str
    mae: float | None = None


def synthetic_series(n: int = N_POINTS, seed: int = SEED) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = np.clip(rng.normal(0.0004, 0.015, size=n), -0.099, 0.099)
    return pd.Series(12.0 * np.cumprod(1.0 + returns))


def baseline_scores(series: pd.Series) -> pd.DataFrame:
    from stockpred.models.compare import compare_models

    result = compare_models(
        series, horizon=HORIZON, origins=ORIGINS, include_neural=False
    )
    return result.table


def timesfm_scores(series: pd.Series) -> Attempt:
    """Zero-shot TimesFM: fit nothing, just feed a context window and read the forecast."""
    try:
        import timesfm
    except ImportError as exc:
        return Attempt("timesfm", False, f"import failed: {exc}")

    try:
        model_cls = getattr(timesfm, "TimesFM_2p5_200M_torch", None) or getattr(
            timesfm, "TimesFM3Torch", None
        )
        if model_cls is None:
            return Attempt("timesfm", False, "no known model class on the package")
        config = timesfm.ForecastConfig(
            max_context=256, max_horizon=8, normalize_inputs=True, per_core_batch_size=1
        )
        model = model_cls.from_pretrained("google/timesfm-2.5-200m-pytorch")
        forecast = getattr(model, "forecast", None)
        if forecast is None:
            return Attempt("timesfm", False, "loaded model exposes no forecast method")
        compile_model = getattr(model, "compile", None)
        if compile_model is not None:
            # TimesFM 2.5 needs an explicit compile() before forecast(); without it
            # the call raises rather than falling back to a default config.
            compile_model(config, horizon=HORIZON)
    except Exception as exc:  # noqa: BLE001 - the whole point is to report what happened
        return Attempt("timesfm", False, f"load failed: {type(exc).__name__}: {exc}"[:400])

    values = series.to_numpy(dtype=float)
    errors: list[float] = []
    try:
        for offset in range(ORIGINS):
            cut = len(values) - (ORIGINS - offset) - HORIZON
            context = values[max(0, cut - 200) : cut]
            actual = values[cut : cut + HORIZON]
            output = forecast(horizon=HORIZON, inputs=[context])
            point = _extract_point(output)
            if point is None or len(point) < HORIZON:
                return Attempt("timesfm", False, f"unexpected forecast shape: {type(output)}")
            errors.append(float(np.mean(np.abs(np.asarray(point[:HORIZON]) - actual))))
    except Exception as exc:  # noqa: BLE001
        return Attempt(
            "timesfm", False, f"inference failed: {type(exc).__name__}: {exc}"[:400]
        )

    return Attempt("timesfm", True, f"{ORIGINS} origins, zero-shot", float(np.mean(errors)))


def _extract_point(output: object) -> np.ndarray | None:
    """Pull the point forecast out of whatever container the library returns."""
    if isinstance(output, tuple):
        for item in output:
            array = np.asarray(item, dtype=float)
            if array.ndim >= 1 and array.size:
                return array.reshape(-1)
        return None
    array = np.asarray(output, dtype=float)
    if array.size == 0:
        return None
    if array.ndim == 3:  # (batch, horizon, quantiles) -> median
        return array[0, :, array.shape[-1] // 2]
    if array.ndim == 2:
        return array[0]
    return array.reshape(-1)


def main() -> int:
    series = synthetic_series()
    table = baseline_scores(series)
    attempt = timesfm_scores(series)

    best_baseline = table.iloc[0]
    lines = [
        "# Spike：时序基础模型（TimesFM）值得接入吗",
        "",
        f"- 样本：{N_POINTS} 点合成序列（seed={SEED}，A 股涨跌幅约束）",
        f"- 协议：滚动起点，horizon={HORIZON}，origins={ORIGINS}（与 `models/compare.py` 一致）",
        "- 指标：MAE",
        "",
        "## 基线（本项目实现，零依赖）",
        "",
        "| 模型 | MAE |",
        "|---|---|",
    ]
    for row in table.itertuples(index=False):
        lines.append(f"| {row.model} | {row.mae:.6f} |")

    lines.extend(["", "## 时序基础模型（零样本，未微调）", ""])
    if attempt.ok:
        lines.append(f"| timesfm | {attempt.mae:.6f} |")
        verdict = (
            "**胜出**：零样本优于最佳基线，值得进入候选"
            if attempt.mae is not None and attempt.mae < float(best_baseline.mae)
            else "**未胜出**：零样本未优于本项目基线，不进主线"
        )
    else:
        lines.append("| timesfm | 未能运行 |")
        verdict = "**无法判定**：模型未运行成功，结论不成立"

    lines.extend(
        [
            "",
            f"运行情况：{'成功' if attempt.ok else '失败'} —— {attempt.detail}",
            "",
            "## 结论",
            "",
            verdict,
            "",
            "## 为什么这个结论不能外推",
            "",
            "- 合成随机游走序列上，任何模型都接近 `naive`；真实 A 股的**可预测成分**不同，结论需在真实数据上重跑；",
            "- horizon=1 是最不利于基础模型的设定（它们擅长长上下文与多步预测）；",
            "- 零样本 ≠ 微调；TimesFM 提供微调接口，未做。",
            "",
            "## 决策建议",
            "",
            "| 场景 | 建议 |",
            "|---|---|",
            "| 立刻进主线 | ❌ 无证据支持 |",
            "| 真实数据 + horizon≥5 重跑 | ✅ 值得一次性 spike |",
            "| 依赖它替代自有模型 | ❌ 成本（权重下载/GPU/推理延迟）与收益未证明 |",
            "",
        ]
    )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"timesfm: ok={attempt.ok} mae={attempt.mae} detail={attempt.detail}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(1)
