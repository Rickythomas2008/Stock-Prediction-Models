# BRIEF — A 股实盘决策支持系统（stockpred）

> 本文件是项目的对齐产物（S0），此前只存在于会话与计划文件中。目标是让**新接手的人在不读任何聊天记录的情况下**知道：做什么、不做什么、为什么这么定、以及怎么验证。

最后更新：2026-11-04

## 1. 目标

在 `Stock-Prediction-Models` fork 内构建一套 **A 股日频决策支持系统**：产出可执行的决策包（信号 → 目标仓位 → 风控校验 → 理由 → 数据时间戳），**经人工确认后由系统自动下单**。

## 2. 用户与场景

- 使用者：单人（个人账户，本金约 10 万元人民币），自行承担全部投资风险。
- 场景：收盘后生成次日委托计划 → 人工确认 → 下单 → 次日对账。
- 终端：本机 Windows；服务仅回环。

## 3. 成功标准（可检查）

| # | 标准 | 当前状态 |
|---|---|---|
| 1 | 任何决策都能回答「依据哪一根 bar、依据什么规则、被哪条风控放行或拒绝」 | ✅ `DecisionPackage` 五项必填字段 + 哈希链审计 |
| 2 | 评估无可利用的前视泄漏，且结论必须带成本前后双数字与三条基准 | ✅ `eval/leakage.py` + `eval/report.py` |
| 3 | A 股规则（T+1 / 100 股 / 涨跌停 / 停牌 / 费用）在回测与纸面撮合中一致 | ✅ `backtest/` + `execution/paper.py` |
| 4 | 连续多日纸面运行可与「券商回报」对账一致 | ✅ 12/12 日一致（合成数据） |
| 5 | 自动下单前存在无法绕过的合规门 | ✅ `broker_confirmed_at` 门 + `require_broker_confirmation()` |
| 6 | 真实行情端到端跑通 | ❌ **未做**（见 [TECH-DEBT](docs/TECH-DEBT.md) P1-6） |
| 7 | 实盘自动下单 | ❌ **未做**（阻塞于券商书面答复） |

## 4. 非目标

- 不做收益承诺，不提供投顾服务。
- 不在未取得券商确认前发送任何自动委托。
- 不引入 AGPL / GPL / Commons Clause 依赖（CI 强制）。
- 不重写 18 个 TF1 模型与 11 个保留为参考的 agent（见 [AGENT-MIGRATION](docs/AGENT-MIGRATION.md)）。
- 不做多用户、不做公网服务（当前无鉴权）。

## 5. 已确认的关键决策

| # | 决策 | 依据 |
|---|---|---|
| 1 | 市场 = A 股，本金 ≈ 10 万 | 用户定 |
| 2 | 执行 = 人工确认 → 自动下单（半自动） | 用户定；CSRC 2024 No.8 允许客户程序化交易，但须先报告后交易 |
| 3 | 先纸面跑通再切真金 | 唯一低成本灰度方式 |
| 4 | 日频、低换手 | 往返成本 0.152%，月 2 次往返即年化 3.65% 拖累 |
| 5 | 框架 = PyTorch | 同类高星项目（qlib/FinRL/SB3/neuralforecast）均为 torch |
| 6 | 依赖白名单 = MIT / Apache-2.0 / BSD / PSF / ISC | 见 `tools/license_allowlist.toml` |
| 7 | 服务用标准库而非 FastAPI | 契约层是纯函数，传输层约 100 行 |
| 8 | OpenAI 类外发接口一律不接 | 合规与隐私 |

## 6. 硬约束

- **合规**：《证券市场程序化交易管理规定（试行）》（证监会公告〔2024〕8 号，2024-10-08 施行）第 2/7/8/19 条；沪深北实施细则（2025-07-07 施行）。
- **禁止**：easytrader / UI 模拟点击 / 任何外挂层（第 8、19 条）。
- **账户**：实名自有，不出借、不代他人交易。
- **不可逆操作**（删改 notebook、改写 `dataset/`、对外发布）需人工门 + 快照。

## 7. 怎么验证这个项目是真的

```bash
python -m venv .venv && .venv/Scripts/python -m pip install -e ".[dev]"
.venv/Scripts/python -m pytest -q                     # 415 passed
.venv/Scripts/python -m ruff check src tests tools     # All checks passed
.venv/Scripts/python tools/check_licenses.py           # RESULT: PASS
.venv/Scripts/python tools/demo_decide.py              # 一个决策包
.venv/Scripts/python tools/demo_paper_loop.py 12       # 12 日纸面 + 对账 + 审计链
```

详细交接见 [docs/HANDOFF.md](docs/HANDOFF.md)。
