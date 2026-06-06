# Baostock Field Contract

All fields verified against baostock 0.9.1 (version string "00.9.10") with real API calls.
Note: query_shibor_data was removed in 0.9.1 (existed in 0.8.9).
This document is the single source of truth for field names — never guess.

## query_history_k_data_plus
```
date, code, open, high, low, close, preclose, volume, amount,
adjustflag, turn, tradestatus, pctChg, peTTM, pbMRQ, psTTM,
pcfNcfTTM, isST
```
- Valuation fields (peTTM, pbMRQ, psTTM, pcfNcfTTM) are per-day snapshots
- Latest trading day close = de facto "current price" (no real-time API exists)

## query_stock_basic
```
code, code_name, ipoDate, outDate, type, status
```
- Only 6 fields. NO industry, NO totalShare, NO listingDate.

## query_stock_industry
```
updateDate, code, code_name, industry, industryClassification
```
- industry field lives HERE, not in stock_basic.

## query_profit_data (has absolute values!)
```
code, pubDate, statDate, roeAvg, npMargin, gpMargin,
netProfit, epsTTM, MBRevenue, totalShare, liqaShare
```
- netProfit: absolute net profit (元)
- MBRevenue: absolute main business revenue (元)
- totalShare: total shares outstanding (股) — NOT in stock_basic

## query_balance_data (ratios only!)
```
code, pubDate, statDate, currentRatio, quickRatio, cashRatio,
YOYLiability, liabilityToAsset, assetToEquity
```
- No absolute amounts. totalLiability / totalDebt do NOT exist.

## query_cash_flow_data (ratios only!)
```
code, pubDate, statDate, CAToAsset, NCAToAsset, tangibleAssetToAsset,
ebitToInterest, CFOToOR, CFOToNP, CFOToGr
```
- No absolute cash flow. netCashOperating / NCFOperateA do NOT exist.
- OCF can be derived: MBRevenue × CFOToOR (precision ~2%, verified against 茅台 2019-2024)

## query_growth_data
```
code, pubDate, statDate, YOYEquity, YOYAsset, YOYNI, YOYEPSBasic, YOYPNI
```
- All YOY* fields are **ratios**, not percentages.
  Verified: sh.600519 2023 Q4 YOYNI = 0.185778 (+18.58% YoY net profit).
- Consumers that need percent-number form (e.g. PEG = PE / G%) must multiply by 100.

## query_operation_data
```
code, pubDate, statDate, NRTurnRatio, NRTurnDays, INVTurnRatio,
INVTurnDays, CATurnRatio, AssetTurnRatio
```
- Total assets can be derived: MBRevenue / AssetTurnRatio (precision 5-15%)
- Note: AssetTurnRatio uses average assets, so derived value ≈ average, not period-end

## query_dupont_data
```
code, pubDate, statDate, dupontROE, dupontAssetStoEquity, dupontAssetTurn,
dupontPnitoni, dupontNitogr, dupontTaxBurden, dupontIntburden, dupontEbittogr
```

## query_dividend_data
```
code, dividPreNoticeDate, dividAgmPumDate, dividPlanAnnounceDate,
dividPlanDate, dividRegistDate, dividOperateDate, dividPayDate,
dividStockMarketDate, dividCashPsBeforeTax, dividCashPsAfterTax,
dividStocksPs, dividCashStock, dividReserveToStockPs
```
- WARNING: dividCashPsBeforeTax / dividCashPsAfterTax may contain
  semicolon-separated multiple values (e.g. '23.3199；25.911' for
  mid-year + year-end dividends). Must split and sum.
- dividendPerShare / dividend_per_share / dividendsPerShare do NOT exist.

## query_performance_express_report
```
code, performanceExpPubDate, performanceExpStatDate, performanceExpUpdateDate,
performanceExpressTotalAsset, performanceExpressNetAsset,
performanceExpressEPSChgPct, performanceExpressROEWa,
performanceExpressEPSDiluted, performanceExpressGRYOY, performanceExpressOPYOY
```
- Has absolute TotalAsset / NetAsset, but only for companies that publish express reports

## query_forecast_report
```
code, profitForcastExpPubDate, profitForcastExpStatDate,
profitForcastType, profitForcastAbstract,
profitForcastChgPctUp, profitForcastChgPctDwn
```

## query_trade_dates
```
calendar_date, is_trading_day
```
- Values are strings: '1' (trading) / '0' (non-trading)

## query_all_stock
```
code, tradeStatus, code_name
```
- Includes indices. Typically 5000+ rows.

## query_hs300_stocks / query_sz50_stocks / query_zz500_stocks
```
updateDate, code, code_name
```
- Code and name only. No weights.

## query_adjust_factor
```
code, dividOperateDate, foreAdjustFactor, backAdjustFactor, adjustFactor
```

## Derivation formulas (verified)
```
OCF         = MBRevenue × CFOToOR             (precision ~2%)
TotalAsset  = MBRevenue / AssetTurnRatio      (precision 5-15%, gives average not period-end)
TotalDebt   = TotalAsset × liabilityToAsset   (precision propagated from TotalAsset)
Equity      = TotalAsset / assetToEquity       (precision propagated from TotalAsset)
CurrentPrice = query_history_k_data_plus(end_date=today).tail(1).close
```

## Methods that do NOT exist
- query_real_time_quotes / any real-time quote API
- query_capex / query_depreciation / any capital expenditure API
- Any API returning absolute balance sheet or cash flow amounts
  (only performance_express has partial absolute values)
