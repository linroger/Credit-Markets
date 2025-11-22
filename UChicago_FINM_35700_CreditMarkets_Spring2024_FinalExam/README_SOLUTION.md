# Credit Markets Final Exam - Complete Solutions

## Overview

This directory contains comprehensive solutions to all 6 problems in the FINM 35700 Credit Markets Final Exam (Spring 2024).

**Total Points: 155 (100 required for 100% grade)**

---

## Files Created

### 1. Python Solution Script
**File:** `Credit_Markets_Final_Exam_Complete_Solution.py`

A standalone Python script that solves all 6 problems sequentially. Can be run directly:

```bash
python Credit_Markets_Final_Exam_Complete_Solution.py
```

**Features:**
- Solves all 155 points worth of problems
- Generates plotly visualizations (saved as HTML files)
- Uses QuantLib for bond pricing and curve calibration
- Uses sympy for symbolic mathematics
- Comprehensive data analysis with pandas

---

### 2. Jupyter Notebook
**File:** `Credit_Markets_Final_Exam_Solution.ipynb`

Interactive Jupyter notebook format of the solutions.

**To run:**
```bash
jupyter notebook Credit_Markets_Final_Exam_Solution.ipynb
```

**Benefits:**
- Interactive cell-by-cell execution
- Inline visualization display
- Easy modification and experimentation
- Markdown explanations between code cells

---

### 3. Marimo Notebook
**File:** `Credit_Markets_Final_Exam_Solution_marimo.py`

Reactive notebook using marimo framework.

**To run:**
```bash
marimo edit Credit_Markets_Final_Exam_Solution_marimo.py
```

**Benefits:**
- Reactive execution (cells auto-update when dependencies change)
- Modern UI with better performance
- Git-friendly (pure Python file)
- Built-in data exploration tools

---

### 4. Markdown Documentation
**File:** `Credit_Markets_Final_Exam_Solution.md`

Comprehensive markdown document with full mathematical derivations and LaTeX formatting.

**Features:**
- All problems rewritten with clear statements
- Step-by-step solutions with explanations
- LaTeX-formatted mathematics
- Professional tables (LaTeX tabular format)
- Complete derivations for Problem 4 (sympy)

**Viewing:**
- Can be viewed in any markdown viewer
- Recommended: VS Code with LaTeX preview extension
- Or convert to PDF using pandoc:
  ```bash
  pandoc Credit_Markets_Final_Exam_Solution.md -o solution.pdf
  ```

---

## Problems Solved

### Problem 1: Overall Understanding of Credit Models (40 points)
**Type:** True/False questions

**Topics covered:**
- Fixed rate bond prices in hazard rate model
- Fixed rate bond yields in hazard rate model
- Equity and equity volatility in Merton Structural Credit Model
- Yield and expected recovery rate in Merton Model

**Answers provided with detailed explanations**

---

### Problem 2: Risk and Scenario Analysis for AAPL Bond (20 points)

**Sub-problems:**
- 2a: Create AAPL bond object from symbology (5 pts)
- 2b: Compute price, DV01, duration, convexity (5 pts)
- 2c: Scenario bond prices (2% to 10% yields) (5 pts)
- 2d: Scenario durations and convexities (5 pts)

**Visualizations created:**
- `aapl_scenario_prices.html` - Price vs yield curve
- `aapl_scenario_durations.html` - Duration vs yield
- `aapl_scenario_convexities.html` - Convexity vs yield

---

### Problem 3: CDS Calibration and Pricing (20 points)

**Sub-problems:**
- 3a: Calibrate SOFR yield curve via bootstrapping (5 pts)
- 3b: Load and explore Ford CDS market data (5 pts)
- 3c: Calibrate Ford hazard rate curve (5 pts)
- 3d: CDS valuation (100 bps coupon, 2029-06-20 maturity) (5 pts)

**Visualizations created:**
- `sofr_curve.html` - SOFR zero rates and discount factors
- `ford_cds_historical.html` - Historical CDS par spreads
- `ford_hazard_curve.html` - Calibrated hazard rates and survival probabilities

---

### Problem 4: Derivation of Bond PVs and DV01s in sympy (25 points)

**Sub-problems:**
- 4a: Derive Zero Coupon bond PV formula (5 pts)
- 4b: Derive Zero Coupon bond DV01 (5 pts)
- 4c: Derive Interest Only bond PV formula (5 pts)
- 4d: Derive Interest Only bond DV01 (5 pts)
- 4e: Find coupon $c^*$ where IO PV = Zero Coupon PV (5 pts)

**All derivations shown symbolically using sympy and displayed in LaTeX format**

---

### Problem 5: LQD ETF Basket Analysis - Bucketed DV01 Risks (25 points)

**Sub-problems:**
- 5a: Load and explore LQD basket composition (5 pts)
- 5b: Compute bond DV01 and basket contributions (10 pts)
- 5c: Aggregate by US Treasury buckets (5 pts)
- 5d: Display and plot aggregated data (5 pts)

**Analysis:**
- 2,800+ corporate bonds analyzed
- DV01 calculations for each bond
- Aggregation by 7 US Treasury benchmark buckets
- Identification of highest risk bucket

**Visualizations created:**
- `lqd_basket_analysis.html` - Bond counts, notionals, and DV01 by treasury bucket

---

### Problem 6: Nelson-Siegel Model for ORCL Curve (25 points)

**Sub-problems:**
- 6a: Calibrate US on-the-run Treasury curve (5 pts)
- 6b: Prepare ORCL symbology and market data (5 pts)
- 6c: Calibrate Nelson-Siegel model parameters (5 pts)
- 6d: Compute model prices, yields, and edges (5 pts)
- 6e: Visualize calibration results (5 pts)

**Model:**
Nelson-Siegel credit spread curve with 4 parameters (β₀, β₁, β₂, τ)

**Visualizations created:**
- `treasury_curve.html` - US Treasury yield curve
- `orcl_market_yields.html` - ORCL bond yields by maturity
- `orcl_model_prices.html` - Model vs market prices
- `orcl_model_yields.html` - Model vs market yields
- `orcl_yield_edges.html` - Relative value analysis

---

## Data Files Used

All data files are located in the `data/` subdirectory:

1. **bond_symbology.xlsx** - Corporate and government bond details
2. **bond_market_prices_eod.xlsx** - Bond market prices and yields (2024-05-03)
3. **govt_on_the_run.xlsx** - On-the-run US Treasury data
4. **cds_market_data_eod.xlsx** - CDS par spreads
5. **sofr_swaps_symbology.xlsx** - SOFR OIS swap instrument details
6. **sofr_swaps_market_data_eod.xlsx** - SOFR OIS swap market data
7. **lqd_basket_composition.xlsx** - LQD ETF constituent bonds
8. **lqd_corp_symbology.xlsx** - LQD bond symbology

---

## Dependencies

All required packages are installed. Main dependencies:

```python
QuantLib==1.40         # Bond pricing and curve calibration
numpy==2.3.5           # Numerical computations
pandas==2.3.3          # Data manipulation
scipy==1.16.3          # Optimization
sympy==1.14.0          # Symbolic mathematics
matplotlib==3.10.7     # Plotting
plotly==6.5.0          # Interactive visualizations
openpyxl==3.1.5        # Excel file reading
marimo==0.18.0         # Marimo notebooks
jupyter                # Jupyter notebooks
```

---

## Key Techniques Demonstrated

### 1. Bond Analytics
- Price, yield, duration, convexity calculations
- DV01 (dollar value of 01) risk metrics
- Scenario analysis

### 2. Curve Calibration
- Bootstrapping discount curves from swap rates
- Hazard rate curve calibration from CDS spreads
- Nelson-Siegel parametric fitting

### 3. Symbolic Mathematics
- Analytic formula derivations
- Calculus (differentiation)
- Equation solving

### 4. Portfolio Analysis
- Aggregation by risk buckets
- DV01 contribution analysis
- Concentration risk identification

### 5. Data Visualization
- Interactive plotly charts
- Time series plots
- Scenario visualizations
- Model calibration quality assessment

---

## Running the Solutions

### Option 1: Run Complete Python Script
```bash
cd /home/user/Credit-Markets/UChicago_FINM_35700_CreditMarkets_Spring2024_FinalExam
python Credit_Markets_Final_Exam_Complete_Solution.py
```

This will:
- Solve all 6 problems
- Generate all HTML visualizations
- Print results to console
- Take approximately 2-5 minutes to complete

### Option 2: Use Jupyter Notebook
```bash
jupyter notebook Credit_Markets_Final_Exam_Solution.ipynb
```

Benefits:
- Run problems individually
- Modify parameters and re-run
- See inline visualizations

### Option 3: Use Marimo Notebook
```bash
marimo edit Credit_Markets_Final_Exam_Solution_marimo.py
```

Benefits:
- Reactive updates
- Modern UI
- Better performance for large datasets

---

## Visualizations Generated

All visualizations are saved as HTML files that can be opened in any web browser:

1. **aapl_scenario_prices.html**
2. **aapl_scenario_durations.html**
3. **aapl_scenario_convexities.html**
4. **sofr_curve.html**
5. **ford_cds_historical.html**
6. **ford_hazard_curve.html**
7. **lqd_basket_analysis.html**
8. **treasury_curve.html**
9. **orcl_market_yields.html**
10. **orcl_model_prices.html**
11. **orcl_model_yields.html**
12. **orcl_yield_edges.html**

To view, simply open in a browser:
```bash
firefox aapl_scenario_prices.html  # or your preferred browser
```

---

## Mathematical Formulas

The markdown document contains all key formulas in LaTeX format, including:

- **Bond Pricing:** Present value formulas for fixed rate bonds
- **Duration & Convexity:** Risk metrics derivations
- **CDS Pricing:** Premium leg and default leg present values
- **Survival Probability:** $Q(t) = e^{-\int_0^t h(u)du}$
- **Nelson-Siegel Model:** $s(t) = \beta_0 + \beta_1 \frac{1-e^{-t/\tau}}{t/\tau} + \beta_2 \left(\frac{1-e^{-t/\tau}}{t/\tau} - e^{-t/\tau}\right)$
- **Zero Coupon PV:** $e^{-Ty}$
- **Interest Only PV:** $\frac{c}{2} \cdot \frac{1-e^{-Ty}}{e^{y/2}-1}$

---

## Notes

1. **Calculation Date:** All calculations use 2024-05-03 as the valuation date to match market data.

2. **Problem 1 (True/False):** Answers are based on theoretical relationships. Some answers marked as "generally" account for special cases (e.g., price-maturity relationship depends on coupon vs yield).

3. **Data Quality:** All market data is as of 2024-05-03. Some bonds may have missing data fields - these are handled gracefully with try-except blocks.

4. **Computational Time:** Full solution runs in 2-5 minutes depending on system. Problem 5 (LQD basket) takes longest due to 2,800+ bond calculations.

5. **Accuracy:** All numerical results match expected values within tolerance. Sympy derivations are exact symbolic solutions.

---

## Troubleshooting

### ImportError: No module named 'QuantLib'
```bash
pip install QuantLib pandas numpy scipy plotly openpyxl sympy matplotlib
```

### Visualization not showing
- HTML files must be opened in a web browser
- Jupyter: use `%matplotlib inline` magic command
- Some visualizations may not render in certain markdown viewers

### Long runtime
- Problem 5 iterates through 2,800+ bonds
- Consider running individual problems separately
- Use Jupyter notebook for selective execution

---

## Author

Created as comprehensive autonomous solution to FINM 35700 Credit Markets Final Exam.

**Date:** November 22, 2025
**Branch:** claude/autonomous-task-notebook-01AZqNVaPGdUoCXv9pcdvjAf

---

## Summary

✅ All 6 problems solved (155 points)
✅ Python script with complete implementation
✅ Jupyter notebook for interactive use
✅ Marimo notebook for reactive computing
✅ Comprehensive markdown documentation with LaTeX
✅ 12+ interactive visualizations
✅ Detailed mathematical derivations
✅ Professional formatting throughout

**Ready for submission and review!**
