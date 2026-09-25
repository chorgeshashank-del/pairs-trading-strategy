# NIFTY 500 benchmark

`NIFTY500_2017_2026.csv` supplies the benchmark closing prices used to calculate market beta and correlation.

The file lacks 11 dates present in the strategy calendar. Both methods calculate strategy and benchmark returns between the same shared observation dates. Intervals spanning missing observations therefore cover multiple sessions; missing prices are not filled.

Full-period market exposure:
- SSD: beta approximately 0.00743; correlation approximately 0.04219.
- Engle–Granger: beta approximately 0.00116; correlation approximately 0.00965.

Beta measures sensitivity to benchmark returns. Correlation measures how closely the strategy and benchmark returns move together.

The original download source of this benchmark file remains to be confirmed.
