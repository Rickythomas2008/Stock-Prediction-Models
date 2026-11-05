# xtquant 探针

- 运行时间：2026-11-05
- 结论：**PRESENT-BUT-NO-CLIENT**

| 项 | 值 |
|---|---|
| 平台 | Windows 10 |
| 解释器 | C:\Users\Ricky\GitHub\Stock-Prediction-Models\.venv\Scripts\python.exe (3.11.9) |
| xtquant | xtquant_250807 @ C:\Users\Ricky\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages\xtquant |
| 客户端 | 未连接：Exception: 无法连接xtquant服务，请检查QMT-投研版或QMT-极简版是否开启 |

| 周期 | 标的 | 行数 | 错误 |
|---|---|---|---|
| 1d | - | - | Exception: 无法连接xtquant服务，请检查QMT-投研版或QMT-极简版是否开启 |
| 1m | - | - | Exception: 无法连接xtquant服务，请检查QMT-投研版或QMT-极简版是否开启 |

## 探测过程

- 直接 import 失败：ModuleNotFoundError: No module named 'xtquant'
- 已从 C:\Users\Ricky\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\site-packages 注入路径后导入成功

## 退出码

0 = 可用；1 = 已安装但客户端未连接/无数据；2 = 未安装（**未执行**，不得当通过）。

## 边界

- 本探针不调用 `xttrader`，不登录、不下单。下单需要已登录账户 + 券商书面确认 （CSRC 2024 No.8 第 8 条），本工具不是这一步的前置动作。
- `xtquant` 随 QMT 客户端分发、仅 Windows、许可证未核实：**不得加入 `pyproject.toml` 或 `tools/license_allowlist.toml`**。
