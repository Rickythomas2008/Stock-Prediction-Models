# 技术债与未决项

按「会不会伤到钱」排序，而不是按工作量。

## P0 — 会影响真实资金（上线前必须解决）

| # | 债务 | 当前状态 | 为什么是 P0 |
|---|---|---|---|
| 1 | 无券商执行适配器 | `execution/` 只有委托单与纸面撮合 | 自动下单是已确认目标，缺这一层就无法落地 |
| 2 | 券商书面答复未取得 | `docs/BROKER-INQUIRY.md` 模板已备；询价 2026-11-04 启动，跟踪表见该文「发送与跟踪」 | CSRC 第 8 条「先报告后交易」，未确认即下单属违规 |
| 3 | 报备材料未实际提交 | `compliance/` 能生成五类字段 | 生成 ≠ 提交；券商确认是人工门 |
| 4 | 服务无鉴权 | `serving/` 设计为本机回环 | 上公网前必须加认证 + TLS |
| 5 | 对账的「券商回报」来源是自身 | `demo_paper_loop` 用 PaperBroker 的 fills | 真对账必须接券商对账单，否则等于自己对自己 |

## P1 — 会让结果不可信

| # | 债务 | 说明 |
|---|---|---|
| 6 | ~~无真实行情端到端~~（数据层已关闭） | 2026-11-03 单标的 + 2026-11-04 二十标的 × 90 日实测：`tools/live_smoke.py`、`tools/live_universe_run.py`，**20/20 通过**（零缺日、零成交量、复权不变式、双源一致，volume 比值 1.0000），报告 `docs/LIVE-RUN-2026-11-04.md`。**决策链仍未吃真实数据**（演示/回测用合成序列）——那部分留 #7 |
| 7 | ~~策略未在真实数据上验证~~（已评估，α 未获证实） | 三个 α 在同一条 walk-forward 上比较（双源各跑一遍）：**全部 2/3 样本外超额为正，且全部在涨市跑输、跌市少亏**（momentum +0.50%/+1.20%、reversal -1.27%/-1.04%、low_vol +2.61%/+2.62% 双源均值）。换 α 不改变这个形态——**这是结构问题（低 beta 大市值多头 vs 指数），不是因子问题** |
| 8 | TopN 目标不随权益再平衡 | 纸面运行 12 日后现金降至 1,222 元，后续买入被现金缓冲拒；应改为按当前权益重算目标 |
| 9 | 分数阶差分未实现 | 平稳性与记忆的取舍仍用简单收益率 |
| 10 | meta-labeling 未接进仓位 | `eval/labeling.py` 能算标签，但没用它定 size |
| 11 | 横截面三重障碍未实现 | 目前是单标的时序口径 |
| 12 | 多标的对账键不足 | 键为 `(date, symbol, side)`，同一天同标的同方向多笔会判重复并抛错 |
| 19 | ~~`volume` 单位未定义，两家 provider 差 100 倍~~（已修 2026-11-04） | 口径定为**股**：akshare 适配层 ×100（`_LOT_SIZE`），baostock 原生即为股；`schema.VOLUME_UNIT` / `AMOUNT_UNIT` 声明口径；`compare_bars` 以 1.0 比值硬门（容差 1%），两处 live 工具与单测覆盖 |
| 20 | ~~策略配置在评估窗口内选定~~（已做 α+配置 walk-forward 2026-11-04） | 三块不重叠窗口 × 24 组网格（α × top × stay × lookback），选择规则事先固定。结果：配置选择稳定，但**三个 α 都是 2/3 正超额、都在涨市跑输跌市少亏**；low_vol 在双源均最优（+2.6%）。结论：**换 α 解决不了「跑赢指数」**，下一层问题在结构（beta/集中度/是否做指数增强） |
| 21 | 结果对数据源敏感（同配置同区间差 ~5pp） | akshare +28.27% vs baostock +23.40%（另含 baostock 缺 `000651`）；换源即换结论，接口层面无法消除，只能靠 S4 一致性门 + 双源复算 |

## P2 — 工程质量

| # | 债务 | 说明 |
|---|---|---|
| 13 | CI 未在 GitHub runner 实跑 | 本地 Windows 全绿，`ubuntu-latest` 未验证；首次 push 后需补日志 |
| 14 | baostock 许可证未核实 | 2026-11-03 复核 0.9.4 wheel：METADATA 仅写 `License: BSD License` + `License :: OSI Approved :: BSD License`，**包内无 LICENSE 文件**，无法判定 BSD-2 还是 BSD-3 ⇒ 仍为 `UNVERIFIED`，上线前须向作者索取或替换 |
| 15 | pyqlib 未进依赖 | spike 期间临时安装（引入 200+ 传递依赖），未写入 `pyproject.toml` |
| 16 | 无 `nbmake` notebook 测试 | 旧 notebook 仍未经 CI 验证（有意如此：TF1 环境与主环境冲突） |
| 17 | 无 Dockerfile | 部署形态未定 |
| 18 | 决策日志无轮转 | `logs/decisions.jsonl` 与 `runs/paper-demo/audit.jsonl` 会无限增长 |

## 环境偏差记录（S7 spike 造成）

`pip install pyqlib` 引入 **222 个包**，并降级 `pyarrow 26.0.1 → 25.0.1`、重装 `python-dateutil`。

- 全量测试仍全绿（已验证），但这属于**环境漂移**；
- qlib **未**写入 `pyproject.toml`，主线不依赖它；
- 若要恢复精简环境：`pip uninstall pyqlib` 后重装 `pip install -e ".[dev]"`（pyarrow 需手动回到 26）。

这条记录本身就是 spike 结论的一部分：**qlib 的接入成本不是「装一个库」**。

## 已知的「不是 bug」

| 现象 | 解释 |
|---|---|
| 合成数据上策略跑输买入持有 | `benchmark_warnings` 如实警示；这是正确输出，不是缺陷 |
| 涨停封板导致拒单 | A 股规则建模结果 |
| 组合层与风控上限不一致时抛错 | 装配期失败优于静默全拒 |
| 空 `risk` 字典合法 | 「无拒单」是完整答案 |

## 口径与决策（2026-11-04）

| 编号 | 决策 | 依据 | 落地 |
|---|---|---|---|
| D1 | `volume` 口径 = **股**，`amount` 口径 = **元** | 双源实测恒差 100 倍；schema 只校验 `>=0`，单位错不会被任何检查抓到 | `schema.VOLUME_UNIT`；akshare 适配层 ×100；`compare_bars` 1% 容差硬门；单测 + 两处 live 工具 |
| D2 | **不接 Tushare** | akshare + baostock 已构成双源互证；Tushare 需 token/积分与调用上限，增量是更长的历史，不是更可信的当日数据 | 依赖不变，`[data]` extra 保持两源 |
| D3 | 测试 run 范围 = **20 标的 × 90 天** | 覆盖 SH / STAR / SZ / ChiNext 四板块，压停牌、复权、双源三类边界；全 A 横截面之前先把单标的边界钉死 | `tools/live_universe_run.py`，报告 `docs/LIVE-RUN-2026-11-04.md` |
| D4 | 启动 **QMT / miniQMT 模拟盘询价** | 交易侧唯一合规路径；报备是「先报告后交易」，无法绕过 | `docs/BROKER-INQUIRY.md`「发送与跟踪」 |
| D5 | 换手控制三层：**加宽持仓（top12）· 留住强势股（stay-band 4）· 免交易带 2%** | 每月重建 top4 等于每月换掉半本书，成本吃 8.2% 毛收益；留强 + 不追零头是唯一不改 α 的降换手法 | `BufferedTopNConstructor`、`plan_execution(min_trade_notional=…)`、`tools/real_data_eval.py` |
| D6 | 资金曲线用**连续资金**（持仓/现金跨期结转、按当期权益定仓），停用「每期重置」 | 重置协议既不付卖出成本、也不结转持仓，换手与成本都被系统性低估；A/B 同信号下 +2.93% vs +28.27% | `run_backtest(initial_positions=…)`、`Pipeline.decide(capital=…)`；重置仅保留作对照（`--protocol reset`） |
| D7 | 换手目标 **<400%/年**：达标（381%/年；第二窗口 165%/年） | 换手是可控成本，不是策略属性 | 报告「年化换手」字段 |
| D8 | 评价基准改为 **沪深300 指数**买入持有（等权 20 只仍同表列出作参考） | 定位是「比拿指数更多的收益」，等权自建篮子是更强的对手但不对应可投资标的 | `--benchmark 000300`；报告「超额收益」字段 |
| D9 | 配置也走 walk-forward：**按事先固定的规则选、在不参与选择的窗口量** | 否则「跑赢」= 在同一窗口挑配置 | `tools/config_walkforward.py`，三块不重叠窗口 × 8 组网格 |

| D10 | 数据源一致性做成 **S4 门**（`tools/source_parity.py`）：不复权逐日必须一致（硬），前复权区间收益差 ≤0.2% 一致 / ≤0.5% 警告 / >0.5% 失败 | 两家拿到同一个市场，但复权因子各算各的；不设门就会出现「换源换结论」 | 退出码 0/1/2（2=未执行，不得当通过）；`docs/SOURCE-PARITY.md` |
| D11 | α 结论以**双源交叉验证**为准：low_vol > momentum > reversal 的排序在 akshare 与 baostock 上一致 | 单源排名可能是数据假象 | `docs/CONFIG-WALKFORWARD.md`（baostock）与 `-akshare.md` |

## 数据源可用性（2026-11-04 实测）

| 源 | 状态 | 处置 |
|---|---|---|
| baostock | 21:30 起**恢复可用**（20:40 曾 10030 连接被拒）；3 年窗口下 **19/20 只**有数据，`000651` 三次重试均返回空 | 默认 provider；缺数据的标的必须在报告里点名，不得静默缩小宇宙 |
| akshare | 可用（股票 1s/只；指数走 Sina 单请求） | `index_zh_a_hist` 每只翻 17 页、负载高即超时，指数改用 `stock_zh_index_daily`（成交量已是股，不再 ×100） |
| 本机代理 | `HTTP_PROXY/HTTPS_PROXY=127.0.0.1:7890` 失效时，requests 系 provider 全部超时（系统代理正常） | 运行前清空：`$env:HTTP_PROXY=""; $env:HTTPS_PROXY=""` |

### 数据源差异实测（2026-11-04，20 标的 × 180 日）

| 判据 | 结果 |
|---|---|
| 不复权逐日（价/量/额） | **20/20 完全一致** —— 两家对同一个市场的刻画一致 |
| 前复权区间收益 | 0.00%~0.50%；4 只 >0.2%（WARN），`000858` 0.50%（FAIL 线）—— 复权因子各厂商自算 |
| 全期策略收益（同配置同区间） | akshare **+28.27%** vs baostock **+23.40%**（差 4.9pp，含 baostock 少一只标的的影响）→ **换源会换结论**，故必须设门 |

### 顺带修复的仓库缺陷

`.gitignore` 里裸写 `data/` 会匹配任意层级的同名目录，`src/stockpred/data/`（整个数据层：schema / cache / providers / loader）**因此从未被 git 跟踪**。已改为 `/data/`、`/logs/`、`/runs/` 仅锚定仓库根；该包需随本次提交一并纳入版本控制。
