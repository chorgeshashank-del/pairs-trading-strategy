# Citations and data sources

## Research methods and literature
The memo cites the main methodological sources used to frame the research:
- Gatev, E., Goetzmann, W.N. & Rouwenhorst, K.G. (2006), *Pairs Trading: Performance of a Relative-Value Arbitrage Rule*, Review of Financial Studies 19(3).
- Engle, R.F. & Granger, C.W.J. (1987), *Co-integration and Error Correction: Representation, Estimation, and Testing*, Econometrica 55(2).
- Krauss, C. (2017), *Statistical Arbitrage Pairs Trading Strategies: Review and Outlook*, Journal of Economic Surveys 31(2).
- Do, B. & Faff, R. (2010, 2012), *Does Simple Pairs Trading Still Work?* and *Are Pairs Trading Profits Robust to Trading Costs?*

## Primary market-data sources
- NSE daily capital-market bhavcopy / UDiFF reports: official NSE historical end-of-day price, volume and traded-value fields.
- NSE corporate-actions reports: dividends, splits, bonuses, rights, buybacks, mergers/demergers and administrative events used in the treatment ledger.
- NSE derivatives bhavcopy / UDiFF reports: historical single-stock futures eligibility checks.
- NSE Indices / NIFTY historical constituent and reconstitution information: point-in-time NIFTY Pharma membership.
- NIFTY 500 benchmark: the supplied `benchmark/NIFTY500_2017_2026.csv` is used for market-exposure calculations. Its original download source is not documented in the packaged file and remains to be confirmed. Both methods calculate strategy and benchmark returns over matching shared-date intervals.

Official portals used in the data workflow include [NSE reports](https://www.nseindia.com/all-reports) and [NIFTY Indices](https://www.niftyindices.com/).

## Implementation and software dependencies

The repository contains project-specific Python scripts using the open-source libraries listed in `requirements.txt`. The methodological references above describe the approaches used; they do not certify the correctness of this implementation.

This source list records the sources described in the project documentation. It is not a complete file-by-file record of download URLs, dates and checksums. Rebuilding the prepared datasets from original exchange downloads remains a separate reproducibility task.
