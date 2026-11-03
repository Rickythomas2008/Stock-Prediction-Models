# 决策服务契约（S5）

## 1. 分层理由

契约与传输分离：

```
serving/contract.py   纯函数：请求校验、状态码、响应体   ← 单测在这里
serving/app.py        薄封装：http.server 路由到契约层    ← 只测冒烟
```

这样 90% 的测试不需要开端口，CI 不会因为端口占用或防火墙随机变红。

## 2. 端点

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/health` | `{"status":"ok","version":<包版本>}` |
| POST | `/decide` | 决策包（五项必填字段）+ 委托单 `orders` |
| GET | `/trade` | **旧 Flask 兼容层**，语义见 §4 |

`openapi_schema()` 手写 OpenAPI 3.1 描述，供快照测试与外部对接。

## 3. `/decide`

请求体（必填）：`as_of`、`bar_date`、`capital`、`bars`、`prices`、`lookback`、`top_n`、`max_position_pct`、`cash`。

响应 200：

```json
{
  "api_version": "v1",
  "as_of": "2024-03-01", "bar_date": "2024-03-01",
  "signal": {"600519": 0.012}, "target_shares": {"000001": 1200},
  "risk": {}, "rationale": "alpha=momentum; ...",
  "target_invested": 32784.0, "executed_notional": 32784.0, "cash_after": 67211.0,
  "orders": [{"as_of":"2024-03-01","symbol":"000001","side":"buy","shares":1200,
              "limit_price":27.32,"notional":32784.0,"valid_until":"2024-03-02",
              "risk_status":"approved","note":"ok"}]
}
```

错误语义（有意区分）：

| 状态码 | 何时 | 含义 |
|---|---|---|
| 400 | 缺字段 / 日期不可解析 / `bars` 空 / 参数非正 | **调用方**的请求错了 |
| 422 | `PipelineError` 等业务失败（如 lookback 超过历史） | 请求合法但当前数据无法决策 |
| 404 / 405 | 路径不存在 / 方法不匹配 | 路由层 |

400 与 422 分开，是为了让调用方能自动判断「改请求」还是「补数据」。

## 4. `/trade` 兼容层

旧版（`realtime-agent/app.py`）语义：`GET /trade?data=[close, volume]` → `{action, balance, status, timestamp}`，一次 1 单位。

新实现保留这四个字段与类型，规则：

- 历史不足 20 个 → `action="nothing"`，`status` 含 `not enough`；
- `close > MA20` 且空仓 → `buy`（1 手 = 100 股）；
- `close < MA20` 且持仓 → `sell`；
- 否则 `nothing`。

**这是兼容层，不参与新管线。** 新管线走 `/decide`。保留它的唯一理由是：现有调用方（`realtime-agent/request.ipynb`）不该因为这次重构而断掉。

## 5. 没有的东西（重要）

- **没有下单端点**。任何自动下单必须先过 `Settings.assert_auto_order_allowed()`（`broker_confirmed_at` 非空），而该适配器尚未实现（`broker-adapter` = 阻塞于券商书面答复）。
- 没有鉴权 —— 本服务设计为**本机回环**使用（`127.0.0.1`）。上公网前必须加认证与 TLS，这一条写在这里是为了不被忘记。

## 6. 验证

| 项 | 结果 |
|---|---|
| `/health` | 200，version = 0.1.0 |
| `/decide` 完整请求 | 200，五项字段齐全，`orders` 含 `approved` 委托单 |
| 缺 `cash` / `max_position_pct` | 400 `missing_field`（逐字段报出） |
| 缺字段逐个补齐 | 依次报 `cash` → `max_position_pct`，无静默默认值 |
| `/trade` | 四字段齐备，`buy` + `balance=99000.0` + 信号说明 |
| OpenAPI | `paths` = `/decide` `/health` `/trade` |
| HTTP 往返 | `127.0.0.1:<ephemeral>/health` → 200 `ok` |

测试：`tests/test_serving.py`，19 个用例，含真实 server 往返（ephemeral port + `finally` 关闭）。
