# A-Share MCP Server

A-Share (中国A股) 市场数据 MCP 服务器，基于 [baostock](http://baostock.com) 数据源。

> **Vibe Coded** -- 本项目由 AI 辅助编写 (vibe coding)，代码质量和计算公式**未经严格审计**。
> 估值模型 (DCF / DDM / PEG)、技术指标、风险指标等涉及的数学公式**可能存在错误**，
> 产出的数字仅供学习参考，**请勿作为任何投资决策的依据**。
> 如果你发现了 bug，欢迎提 issue 或 PR。

---

## 功能概览

| 分类 | 工具 |
|------|------|
| 行情数据 | K线、快照、复权因子、股票列表、行业分类、交易日历、分红 |
| 财务数据 | 利润、增长、资产负债、现金流、杜邦分析、营运能力 |
| 财务报表 (akshare 可选) | 资产负债表、利润表、现金流量表 (绝对值全科目)、净负债 |
| 指数成分 | 沪深300、上证50、中证500 |
| 宏观数据 | 货币供应量 (M0/M1/M2)、存贷款利率、存款准备金率 |
| 技术分析 | MACD、RSI、KDJ、BOLL、WR、CCI、ATR、ADX、OBV、MFI、均线 (SMA/EMA) |
| 估值分析 | PE/PB/PS 历史分位、行业对比、DCF、DDM、PEG |
| 风险指标 | Beta、Sharpe、最大回撤、波动率、信息比率 |

## 快速开始

需要 Python >= 3.12 和 [uv](https://docs.astral.sh/uv/)。

```bash
# 克隆
git clone <repo-url>
cd ashare-mcp

# 安装依赖
uv sync

# 启动 (stdio 模式，供 MCP 客户端对接)
uv run python -m ashare_mcp

# 或 HTTP 模式
uv run python -m ashare_mcp --transport http --port 3000
```

### 可选: akshare 扩展

```bash
# 安装 akshare 可选依赖后多出 2 个工具 + DCF 自动改用真实现金流/Capex/净负债
uv sync --extra akshare
```

安装后:
- 新增 `get_financial_statement` (参数 `statement`: balance/income/cash_flow) 和 `get_net_debt` 工具 (数据源: 东方财富)
- `calculate_dcf_valuation` 自动改用真实经营现金流 (`NETCASH_OPERATE`) 和 Capex (`CONSTRUCT_LONG_ASSET`)，`net_debt` 自动从资产负债表计算；`capex_to_ocf_ratio` 仍需提供，作为 akshare 运行时失败/未安装时的回退估算
- `data_provenance` 字段如实标注数据来源
- 不安装则行为与之前完全一致

## 注意事项

- 数据来源为 baostock **免费接口**，无需注册、无需 API Key
- baostock 仅提供 A 股历史数据，**不提供实时行情**
- 技术指标需要足够的历史数据做 warmup（如 MACD 至少需要 33 个交易日），窗口过短会返回 null
- 复权因子仅在除权除息日存在记录，非除权日期查询会返回空
- DCF 中的 OCF 是由 `MBRevenue * CFOToOR` 推算的（约 2% 精度），Capex 需调用者自行提供；安装 akshare 后自动改用真实数据
- DDM 的股利 CAGR 仅上限 clamp 到 20%（防过度外推），负 CAGR 透传
- akshare 数据源走东方财富网页接口，有反爬/限频，不保证永久稳定

## Credits

- [firmmaple/a-share-mcp-server](https://github.com/firmmaple/a-share-mcp-server) -- 原版实现，本项目从零重写但灵感来源于此
- [LINUX DO](https://linux.do) 社区 -- 感谢社区的讨论和灵感
- [baostock](http://baostock.com) -- 免费开源的 A 股数据源
- [FastMCP](https://github.com/modelcontextprotocol) -- MCP 协议实现
