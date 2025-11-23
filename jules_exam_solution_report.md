# FINM 35700 - Credit Markets Final Exam Solution

**Branch:** `jules`

## Problem 1: Overall understanding of credit models

### 1a. Fixed rate bond prices in the hazard rate model
1. Fixed rate bond price is decreasing in interest rate: **True**
2. Fixed rate bond price is decreasing in hazard rate: **True**
3. Fixed rate bond price is decreasing in expected recovery rate: **False**
4. Fixed rate bond price is decreasing in coupon: **False**
5. Fixed rate bond price is decreasing in bond maturity: **True**

### 1b. Fixed rate bond yields in the hazard rate model
1. Fixed rate bond yield is decreasing in interest rate: **False**
2. Fixed rate bond yield is decreasing in hazard rate: **False**
3. Fixed rate bond yield is decreasing in expected recovery rate: **True**
4. Fixed rate bond yield is independent of the coupon: **True**
5. Fixed rate bond yield is decreasing in bond maturity: **False**

### 1c. Equity and equity volatility in the Merton Structural Credit Model
1. Equity value is decreasing with company assets: **False**
2. Equity volatility is decreasing with company assets: **True**
3. Equity value is decreasing with assets volatility: **False**
4. Equity value is decreasing with company liabilities: **True**
5. Equity volatility is decreasing with company liabilities: **False**

### 1d. Yield and expected recovery rate in the Merton Structural Credit Model
1. Yield is decreasing with company liabilities: **False**
2. Expected recovery rate is decreasing with company liabilities: **True**
3. Yield is decreasing with assets volatility: **False**
4. Credit spread is decreasing with asset values: **True**
5. Credit spread is decreasing with assets volatility: **False**

## Problem 2: Risk and scenario analysis for a fixed rate corporate bond

Using the AAPL bond (`AAPL 2.2 09/11/29`):

latex
\begin{document}
\begin{tabular}{|l|c|}
\hline
Metric & Value \\ \hline
Price & 91.7047 \\ \hline
Yield & 5.1175\% \\ \hline
DV01 & 0.117305 \\ \hline
Mod Duration & 12.7916 \\ \hline
Convexity & 217.6269 \\ \hline
\end{tabular}
\end{document}


Scenario analysis shows the inverse relationship between price and yield, and the convexity effect.

## Problem 3: CDS calibration and pricing

**Ford CDS Valuation:**

latex
\begin{document}
\begin{tabular}{|l|c|}
\hline
Metric & Value \\ \hline
PV & 471,929.59 \\ \hline
Premium Leg PV & -441,138.25 \\ \hline
Default Leg PV & 913,067.84 \\ \hline
Par Spread & 206.98 bps \\ \hline
Survival Probability & 0.8287 \\ \hline
\end{tabular}
\end{document}


## Problem 4: Derivation of fixed rate bond PVs and DV01s in sympy

### 4a. Zero Coupon Bond PV
$$
PV_{ZC} = e^{-yT}
$$

### 4b. Zero Coupon Bond DV01
$$
DV01_{ZC} = - \frac{d PV_{ZC}}{dy} \times 0.0001 = T e^{-yT} \times 0.0001
$$

### 4c. Interest Only Bond PV
$$
PV_{IO} = \sum_{k=1}^{2T} \frac{c}{2} e^{-k y / 2} = \frac{c}{2} e^{-y/2} \frac{1 - e^{-yT}}{1 - e^{-y/2}}
$$

### 4d. Interest Only Bond DV01
$$
DV01_{IO} = - \frac{d PV_{IO}}{dy} \times 0.0001
$$

### 4e. Equilibrium Coupon $c^*$
The coupon $c^*$ where $PV_{IO} = PV_{ZC}$ is found by solving:
$$
\frac{c^*}{2} e^{-y/2} \frac{1 - e^{-yT}}{1 - e^{-y/2}} = e^{-yT}
$$

## Problem 5: LQD ETF basket analysis

latex
\begin{document}
\begin{tabular}{|l|c|}
\hline
Metric & Value \\ \hline
Number of bonds & 2148 \\ \hline
Average Notional & 272,000,000 \\ \hline
Median Notional & 272,000,000 \\ \hline
\end{tabular}
\end{document}


Aggregated risk by benchmark treasury shows the distribution of DV01 risk across the curve.

## Problem 6: Nelson-Siegel model for smooth hazard rates: ORCL curve

- **Calibrated TSY Curve:** Successfully calibrated to on-the-run treasuries.
- **ORCL Bonds:** 33 fixed-rate bonds used.
- **Optimal Nelson-Siegel Params:**
  - $\theta_1$: 0.0428
  - $\theta_2$: -0.0382
  - $\theta_3$: -0.0024
  - $\lambda$: 10.00

The calibrated model prices match market prices closely, allowing for identification of "rich" and "cheap" bonds via the "edge" metric.
