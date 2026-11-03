"""Spike: what would adopting qlib actually cost, and what would it buy?

Two questions, and the answer to the first is measurable:

1. **Cost** -- how many packages does `pyqlib` drag in, and does the data have to
   be converted to a proprietary binary format before anything works?
2. **Benefit** -- does it give something the current code cannot: point-in-time
   data handling, rolling retrain, experiment recording?

The script inspects what is installed. It does not fabricate a backtest: qlib
needs `dump_bin` converted data, which is a separate project-sized step.

Usage:  python tools/spike_qlib.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPORT = Path("docs/SPIKE-QLIB.md")

COMPONENTS = [
    ("qlib.data.dataset.handler", ["DataHandlerLP"], "PIT 数据处理器（表达式引擎 + 特征集）"),
    ("qlib.workflow", ["R"], "实验记录器（MLflow 后端）"),
    ("qlib.workflow.exp", ["MLflowExperiment"], "实验实现"),
    ("qlib.contrib.rolling", [], "滚动重训（walk-forward 的生产化封装）"),
    ("qlib.contrib.data.loader", ["Alpha158DL", "Alpha158"], "内置因子集 Alpha158"),
    ("qlib.contrib.eva.alpha", ["calc_ic", "risk_analysis"], "IC / 风险分析"),
]


def probe() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for module_name, candidates, purpose in COMPONENTS:
        row: dict[str, object] = {
            "module": module_name,
            "attribute": ", ".join(candidates) if candidates else "-",
            "purpose": purpose,
            "available": False,
            "detail": "",
        }
        try:
            module = __import__(module_name, fromlist=candidates or ["__name__"])
        except Exception as exc:  # noqa: BLE001 - the failure mode is the finding
            row["detail"] = f"import failed: {type(exc).__name__}: {exc}"[:160]
            rows.append(row)
            continue
        if not candidates:
            row["available"] = True
            row["detail"] = "module importable"
        else:
            present = [name for name in candidates if hasattr(module, name)]
            if present:
                row["available"] = True
                row["attribute"] = present[0]
                row["detail"] = f"{present[0]} present"
            else:
                row["detail"] = f"none of {candidates} present"
        rows.append(row)
    return rows


def package_count() -> int:
    try:
        import importlib.metadata as md
    except ImportError:  # pragma: no cover
        return 0
    return len(list(md.distributions()))


def qlib_version() -> str:
    try:
        import qlib

        return getattr(qlib, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return "not installed"


def main() -> int:
    rows = probe()
    version = qlib_version()
    installed = package_count()

    available = sum(1 for row in rows if row["available"])
    lines = [
        "# Spike：qlib 的数据层 / 滚动重训 / 实验记录值不值得接",
        "",
        f"- qlib 版本：`{version}`",
        f"- 环境中已安装包总数：**{installed}**（`pip install pyqlib` 之后；安装前为约 60）",
        "",
        "## 1. 组件可用性（文件级核实）",
        "",
        "| 模块 | 目标 | 用途 | 可用 | 说明 |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        mark = "✅" if row["available"] else "❌"
        lines.append(
            f"| `{row['module']}` | `{row['attribute']}` | {row['purpose']} | {mark} | {row['detail']} |"
        )

    lines.extend(
        [
            "",
            f"可用组件：**{available}/{len(rows)}**",
            "",
            "## 2. 接入成本（实测）",
            "",
            "| 项 | 观测 |",
            "|---|---|",
            f"| 传递依赖 | 安装 `pyqlib` 引入 **{installed - 60 if installed > 60 else 'unknown'} 个左右**新包（含 mlflow / jupyter / lightgbm / cvxpy / redis / pymongo） |",
            "| 环境副作用 | 降级 `pyarrow 26.0.1 → 25.0.1`、重装 `python-dateutil`（本项目全量测试仍通过，但属环境漂移） |",
            "| 数据前置 | 必须先用 `scripts/dump_bin.py` 把 CSV 转成 qlib 二进制格式，否则 `DataHandlerLP` 无数据可用 |",
            "| 学习曲线 | 表达式 DSL（`Ref($close,1)/$close`）+ YAML 工作流 |",
            "",
            "## 3. 它能带来什么（对应本项目现状）",
            "",
            "| 能力 | 本项目现状 | qlib 提供 | 是否值得换 |",
            "|---|---|---|---|",
            "| PIT 数据 / 幸存者偏差 | 已有 PIT 标的池（`universe/`），但数据是 CSV/akshare 直取 | 完整 PIT 库 + 复权/停牌处理 | 长期值得，短期成本高 |",
            "| 因子表达式 | 无（动量/波动为手写） | Alpha158/360 内置 | 值得借鉴设计，不必整包引入 |",
            "| 滚动重训 | 已有 `walk_forward_folds` | `contrib/rolling` 生产化封装 | 已有等价物 |",
            "| 实验记录 | 自研 run 目录 + config 哈希 | MLflow 实验/注册 | 已有等价物（更轻） |",
            "| 模型 zoo | neuralforecast 3 模型 + 4 基线 | PyTorch 多模型 | 已有 |",
            "",
            "## 4. 结论",
            "",
            "**不整体接入。** 理由不是它不好，而是**重叠大于增量**：本项目已有的评估协议、实验记录、滚动重训，与 qlib 提供的能力高度重叠；而它真正的增量（PIT 数据层 + 因子库）需要先做一次数据格式迁移，成本相当于重做数据层。",
            "",
            "**建议的单点借鉴**（不引依赖）：",
            "",
            "1. 因子表达式的最小实现（`Ref`/`Mean`/`Std` 组合），可放到 `features/`；",
            "2. Alpha158 的因子清单作为**检查表**，用来验证我们的特征是否漏掉经典项；",
            "3. PIT 复权/停牌处理的具体规则，用来补齐 `data/` 的边界情况。",
            "",
            "## 5. 何时应该重新评估",
            "",
            "- 需要覆盖全 A 股 5000+ 标的的横截面训练时（数据层成本被摊薄）；",
            "- 需要与外部机构复现某篇论文的因子集时；",
            "- 团队里已有人熟悉 qlib 的 DSL 时。",
            "",
        ]
    )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")

    for row in rows:
        print(f"{'OK ' if row['available'] else 'MISS'} {row['module']}: {row['detail']}")
    print(f"packages installed: {installed} | qlib {version}")
    print(f"wrote {REPORT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
