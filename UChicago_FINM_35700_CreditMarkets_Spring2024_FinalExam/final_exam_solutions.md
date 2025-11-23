# Credit Markets Final Exam – Step-by-Step Walkthrough

This document rewrites each question, outlines the calculations, and records the intermediate results and visual outputs generated in the companion notebook.

## Problem 1 – Overall understanding of credit models (True/False)
Evaluated each dependency ceteris paribus using the hazard-rate and Merton structural frameworks.

```latex
\begin{document}
\begin{tabular}{|c|c|}
\hline
Statement & True? \\ \hline
Price vs interest rate & True \\ \hline
Price vs hazard rate & True \\ \hline
Price vs expected recovery & False \\ \hline
Price vs coupon & False \\ \hline
Price vs maturity & True \\ \hline
\end{tabular}
\end{document}
```

```latex
\begin{document}
\begin{tabular}{|c|c|}
\hline
Statement & True? \\ \hline
Yield vs interest rate & False \\ \hline
Yield vs hazard rate & False \\ \hline
Yield vs expected recovery & True \\ \hline
Yield vs coupon & False \\ \hline
Yield vs maturity & False \\ \hline
\end{tabular}
\end{document}
```

```latex
\begin{document}
\begin{tabular}{|c|c|}
\hline
Statement & True? \\ \hline
Equity value vs assets & False \\ \hline
Equity vol vs assets & True \\ \hline
Equity value vs asset vol & False \\ \hline
Equity value vs liabilities & True \\ \hline
Equity vol vs liabilities & False \\ \hline
\end{tabular}
\end{document}
```

```latex
\begin{document}
\begin{tabular}{|c|c|}
\hline
Statement & True? \\ \hline
Yield vs liabilities & False \\ \hline
Expected recovery vs liabilities & True \\ \hline
Yield vs asset vol & False \\ \hline
Credit spread vs assets & True \\ \hline
Credit spread vs asset vol & False \\ \hline
\end{tabular}
\end{document}
```

## Problem 2 – AAPL fixed-rate corporate bond (US037833AT77)
* Built the bond object from symbology and listed the future cash flows.
* Priced off the mid-yield (5.1175%) to obtain analytics.
* Ran scenario grids from 2%–10% in 0.5% steps for price, duration, and convexity curves (Plotly lines in the notebook).

```latex
\begin{document}
\begin{tabular}{|c|c|c|c|c|}
\hline
Mid Yield & Price & Duration & Convexity & DV01 \\ \hline
5.1175 & 91.7047 & 12.7916 & 217.6269 & 0.1243 \\ \hline
\end{tabular}
\end{document}
```

## Problem 3 – Ford CDS calibration and valuation
* Bootstrapped the SOFR discount curve from the swap strip and plotted zero/DF curves.
* Loaded Ford CDS history and visualized tenor spreads through time.
* Calibrated hazard rates to par spreads dated 2024-05-03 and plotted hazard/survival curves.
* Valued a 100 bps CDS maturing 2029-06-20 using ISDA engine.

```latex
\begin{document}
\begin{tabular}{|c|c|c|c|c|}
\hline
CDS PV & Premium Leg PV & Default Leg PV & Par Spread (bps) & Survival to Maturity \\ \hline
-213271.36 & 440030.39 & -653301.75 & 148.47 & 0.8721 \\ \hline
\end{tabular}
\end{document}
```

## Problem 4 – Sympy derivations for flat-yield bonds
* Derived analytic PV and DV01 expressions:
  * Zero coupon: \(P_{ZC}=e^{-yT},\;\text{DV01}_{ZC}=0.0001\,T\,e^{-yT}\).
  * Interest-only: \(P_{IO}=\tfrac{c}{2}\dfrac{1-e^{-yT}}{e^{y/2}-1}\), DV01 via symbolic differentiation.
* Solved \(c^*\) satisfying \(P_{IO}=P_{ZC}\) and rendered Plotly surfaces for PV/DV01 across yield–maturity grids.

## Problem 5 – LQD ETF basket DV01 analysis
* Merged basket composition with corporate symbology; computed per-bond DV01s from mid-yields and scaled to basket notionals.
* Aggregated DV01s by underlying benchmark Treasury buckets and visualized counts, notionals, and DV01 bars.

```latex
\begin{document}
\begin{tabular}{|c|c|c|c|}
\hline
Benchmark ISIN & Basket Count & Face Notional & Basket DV01 \\ \hline
US912810TV08 & 586 & 6{,}816{,}956{,}000 & 96{,}576.286487 \\ \hline
US912810TZ12 & 397 & 4{,}395{,}740{,}000 & 52{,}589.401046 \\ \hline
US91282CJZ59 & 645 & 6{,}572{,}948{,}000 & 49{,}579.579343 \\ \hline
US91282CKJ98 & 140 & 1{,}099{,}876{,}000 & 3{,}629.699545 \\ \hline
US91282CKK61 & 1 & 11{,}124{,}000 & 35.478937 \\ \hline
US91282CKN01 & 21 & 191{,}166{,}000 & 1{,}117.098538 \\ \hline
US91282CKP58 & 671 & 7{,}113{,}477{,}000 & 32{,}724.070158 \\ \hline
\end{tabular}
\end{document}
```

## Problem 6 – Nelson–Siegel smooth hazard curve for ORCL
* Calibrated the on-the-run Treasury curve from mid prices (flat-forward bootstrapping).
* Filtered ORCL fixed-rate issues (amt_out > 100) and plotted market yields vs. time-to-maturity.
* Optimized Nelson–Siegel parameters against market prices/yields, then priced bonds on the smooth curve to compute price/yield edges.
* Compared model vs. market via Plotly scatters and edge bars.

```latex
\begin{document}
\begin{tabular}{|c|c|}
\hline
Parameter & Value \\ \hline
theta1 & 0.0427 \\ \hline
theta2 & -0.0382 \\ \hline
theta3 & -0.00225 \\ \hline
lambda & 10.0 \\ \hline
\end{tabular}
\end{document}
```
