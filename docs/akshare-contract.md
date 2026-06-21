# Akshare Field Contract (EastMoney)

All fields verified against akshare (akfamily/akshare main branch, 2026-06)
with real API calls on SH600519 (贵州茅台).

Source: EastMoney (东方财富) F10 financial statements.
Symbol format: `SH600519` / `SZ000001` (uppercase market prefix, no dot).
Conversion from baostock: `sh.600519` → remove `.`, uppercase → `SH600519`.

> **Fetch path:** `AkshareSource` calls the `*_by_report_delisted_em` variants,
> which hit EastMoney's datacenter API (one `ps=200` bulk fetch, ~10x fewer
> requests) instead of the paginated `*_by_report_em` endpoint. The function
> names below document the field contract; the delisted variants are field-
> compatible and verified to return identical values. One difference: they leave
> `REPORT_DATE` as a `datetime.date` (the `*_by_report_em` ones stringify it) —
> `scalar()` normalizes it to an ISO string at the JSON boundary.

Each function returns a multi-period DataFrame (one row per report date,
`iloc[0]` = most recent). Every absolute-value column has a `*_YOY`
companion (同比增长率). Column names are raw EastMoney English uppercase
keys — akshare source code does NOT `.rename()`.

## stock_balance_sheet_by_report_em

DCF-critical fields:
```
MONETARYFUNDS          货币资金
SHORT_LOAN             短期借款
LONG_LOAN              长期借款
BOND_PAYABLE           应付债券
NONCURRENT_LIAB_1YEAR  一年内到期的非流动负债
TOTAL_ASSETS           资产总计
TOTAL_LIABILITIES      负债合计
TOTAL_EQUITY           所有者权益合计
TOTAL_PARENT_EQUITY    归属于母公司所有者权益合计
MINORITY_EQUITY        少数股东权益
```

Net debt derivation:
```
net_debt = SHORT_LOAN + LONG_LOAN + BOND_PAYABLE + NONCURRENT_LIAB_1YEAR - MONETARYFUNDS
```

Metadata columns (shared across all three statements):
```
SECUCODE               证券代码 (e.g. '600519.SH')
SECURITY_CODE          纯数字代码
SECURITY_NAME_ABBR     简称
REPORT_DATE            报告期 (datetime.date after akshare conversion)
REPORT_TYPE            报告类型
REPORT_DATE_NAME       报告期名称 (e.g. '2024年年报')
NOTICE_DATE            公告日期
UPDATE_DATE            更新日期
```

Total columns: ~160 absolute + ~160 YOY = ~320.

## stock_profit_sheet_by_report_em

DCF-relevant fields:
```
OPERATE_INCOME         营业收入
TOTAL_OPERATE_INCOME   营业总收入
OPERATE_COST           营业成本
TOTAL_OPERATE_COST     营业总成本
OPERATE_PROFIT         营业利润
TOTAL_PROFIT           利润总额
NETPROFIT              净利润
PARENT_NETPROFIT       归属于母公司净利润
DEDUCT_PARENT_NETPROFIT 扣非净利润
BASIC_EPS              基本每股收益
```

Total columns: ~100 absolute + ~100 YOY = ~200.

## stock_cash_flow_sheet_by_report_em

DCF-critical fields:
```
NETCASH_OPERATE        经营活动产生的现金流量净额
CONSTRUCT_LONG_ASSET   购建固定资产、无形资产和其他长期资产支付的现金 (= Capex)
NETCASH_INVEST         投资活动产生的现金流量净额
NETCASH_FINANCE        筹资活动产生的现金流量净额
SALES_SERVICES         销售商品、提供劳务收到的现金
```

Total columns: ~130 absolute + ~130 YOY = ~260.

## Verified wrong guesses (never guess!)

These field names appear in third-party code but do NOT exist in the
actual akshare DataFrame:
```
NET_OPERATE_CASH_FLOW  → actual: NETCASH_OPERATE
PURCHASE_ASSETS        → actual: CONSTRUCT_LONG_ASSET
MONETARY_FUNDS         → actual: MONETARYFUNDS
SHORT_TERM_LOAN        → actual: SHORT_LOAN
LONG_TERM_LOAN         → actual: LONG_LOAN
```
