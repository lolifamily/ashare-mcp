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

## 注意事项

- 数据来源为 baostock **免费接口**，无需注册、无需 API Key
- baostock 仅提供 A 股历史数据，**不提供实时行情**
- 技术指标需要足够的历史数据做 warmup（如 MACD 至少需要 33 个交易日），窗口过短会返回 null
- 复权因子仅在除权除息日存在记录，非除权日期查询会返回空
- DCF 中的 OCF 是由 `MBRevenue * CFOToOR` 推算的（约 2% 精度），Capex 需调用者自行提供
- DDM 的股利增长率会被 clamp 到 [1%, 20%] 区间，防止极端外推

## Credits

- [firmmaple/a-share-mcp-server](https://github.com/firmmaple/a-share-mcp-server) -- 原版实现，本项目从零重写但灵感来源于此
- [LINUX DO](https://linux.do) 社区 -- 感谢社区的讨论和灵感
- [baostock](http://baostock.com) -- 免费开源的 A 股数据源
- [FastMCP](https://github.com/modelcontextprotocol) -- MCP 协议实现
