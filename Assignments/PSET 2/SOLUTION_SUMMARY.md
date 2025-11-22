# PSET 2 Solution Summary
## FINM 35700 - Credit Markets - Spring 2024

**Completion Date:** 2024-11-22
**Calculation Date:** 2024-04-08

---

## Overview

This solution provides a complete implementation of all problems in PSET 2 using Python with QuantLib, pandas, and plotly. All problems have been solved with comprehensive calculations, visualizations, and documentation.

## Solution File

**Main Script:** `PSET_2_Solution.py`
- Fully autonomous solution
- Comprehensive comments throughout
- Error handling for edge cases
- Progress reporting during execution

## Problems Solved

### Problem 1: Constructing Fixed Rate Bonds ✓

**Tasks Completed:**
- Loaded and prepared bond symbology and market data (776 fixed-rate bonds)
- Created helper functions for date conversion (`get_ql_date`)
- Implemented `create_schedule_from_symbology()` to generate cashflow schedules
- Implemented `create_bond_from_symbology()` to construct QuantLib bond objects
- Created `get_bond_cashflows()` to extract and display bond cashflows
- Demonstrated cashflows for sample government and corporate bonds

**Key Metrics:**
- Government bonds processed: 339
- Corporate bonds processed: 437
- Total fixed-rate bonds: 776

---

### Problem 2: US Treasury Yield Curve Calibration ✓

**Tasks Completed:**
- Identified and extracted 7 on-the-run US Treasury bonds (GT2, GT3, GT5, GT7, GT10, GT20, GT30)
- Merged symbology data with market prices
- Calibrated three yield curves using QuantLib's PiecewiseLogCubicDiscount:
  - Bid curve
  - Ask curve
  - Mid curve
- Generated curve details with discount factors and zero rates
- Created visualizations for all curves

**Output Files:**
- `Problem_2a_OTR_Yields.html` - Interactive scatter plot of OTR yields by TTM
- `Problem_2c_Calibrated_Yield_Curves.html` - Bid/Ask/Mid zero rate curves
- `Problem_2d_Discount_Factors.html` - Discount factor curve

**Calibration Results:**
- 7 on-the-run treasuries used as inputs
- Curves extrapolated to 30 years
- 61 curve points (6-month intervals)

---

### Problem 3: Pricing and Risk Metrics for US Treasury Bonds ✓

**Tasks Completed:**

#### 3a. Pricing on Calibrated Curve
- Priced all 7 OTR treasuries using calibrated mid curve
- Validated prices match market mid prices (perfect calibration)

#### 3b. Analytical Risk Metrics (Flat Yield)
- Computed DV01, Duration, and Convexity using flat 5% yield
- Used QuantLib's BondFunctions for analytical calculations

#### 3c. Scenario Risk Metrics (Calibrated Curve)
- Computed scenario-based DV01, Duration, and Convexity
- Used +/- 1bp interest rate shocks on calibrated curve
- Calculated first and second-order sensitivities

**Output Files:**
- `Problem_3_Treasury_Results.csv` - Complete results with all metrics

**Sample Results:**
```
Security: T 4 02/15/34
- DV01: 7.92
- Duration: 8.14
- Scenario Duration: 8.14
- Convexity: 76.81
```

---

### Problem 4: Pricing and Risk Metrics for Corporate Bonds ✓

**Tasks Completed:**

#### 4a. Corporate Bond Objects
- Created bond objects for all 437 corporate bonds
- Merged symbology with market prices

#### 4b. Yields and Z-Spreads
- Calculated implied yields for all bonds using market prices
- Computed z-spreads relative to flat 4.9% treasury curve
- Successfully calculated for 437/437 bonds

#### 4c. Z-Spread Validation
- Validated z-spread calculations by re-pricing bonds
- Perfect match between market prices and calculated prices
- Demonstrated for 3 sample bonds from different issuers

#### 4d. Duration and Convexity
- Computed duration and convexity for all corporate bonds
- Used flat 5.1% yield for calculations

**Output Files:**
- `Problem_4_Corporate_Results.csv` - Complete results (437 bonds)
- `Problem_4_Corporate_ZSpreads.html` - Z-spreads by maturity and issuer
- `Problem_4_Corporate_Duration.html` - Duration by maturity and issuer

**Sample Z-Spread Results:**
```
AAPL 3.85 05/04/43:  18.53 bps
DIS 3 3/8 11/15/26:  11.11 bps
F 4.271 01/09/27:    99.51 bps
```

---

## Technical Implementation

### Libraries Used
- **QuantLib**: Bond pricing, yield curve calibration, risk metrics
- **pandas**: Data manipulation and analysis
- **NumPy**: Numerical calculations
- **Plotly**: Interactive visualizations

### Key Functions Implemented

1. **get_ql_date()**: Converts Python dates to QuantLib dates
2. **create_schedule_from_symbology()**: Creates bond cashflow schedules
3. **create_bond_from_symbology()**: Constructs QuantLib bond objects
4. **get_bond_cashflows()**: Extracts bond cashflows with year fractions
5. **calibrate_yield_curve_from_frame()**: Bootstraps yield curves from bond prices
6. **get_yield_curve_details_df()**: Extracts curve data at specified dates
7. **calc_clean_price_with_zspread()**: Prices bonds with z-spread adjustment

### Day Count Conventions
- **US Treasuries**: ACT/ACT (ISMA)
- **Corporate Bonds**: 30/360 (USA)

### Calendar and Conventions
- **Calendar**: US Government Bond calendar
- **Business Day Convention**: Unadjusted
- **Coupon Frequency**: Semi-annual
- **Compounding**: Compounded

---

## Output Files Summary

### Visualizations (HTML)
1. **Problem_2a_OTR_Yields.html** (4.7 MB)
   - On-the-run treasury yields scatter plot

2. **Problem_2c_Calibrated_Yield_Curves.html** (4.7 MB)
   - Bid, Ask, and Mid zero rate curves
   - Interactive plot with hover details

3. **Problem_2d_Discount_Factors.html** (4.7 MB)
   - Discount factor curve over 30 years

4. **Problem_4_Corporate_ZSpreads.html** (4.7 MB)
   - Z-spreads by maturity colored by issuer
   - Interactive hover with bond details

5. **Problem_4_Corporate_Duration.html** (4.7 MB)
   - Duration by maturity colored by issuer
   - Interactive visualization

### Data Files (CSV)
1. **Problem_3_Treasury_Results.csv** (3.3 KB)
   - 7 on-the-run treasuries with all calculated metrics
   - Columns: prices, yields, DV01, duration, convexity (analytical and scenario)

2. **Problem_4_Corporate_Results.csv** (167 KB)
   - 437 corporate bonds with all calculated metrics
   - Columns: symbology, market data, yields, z-spreads, DV01, duration, convexity

### Log Files
1. **PSET_2_Solution_Log.txt** (8.8 KB)
   - Complete execution log with sample results
   - Progress indicators for each problem

---

## Validation and Quality Checks

### Calibration Validation
- ✓ Calibrated prices match market prices perfectly (< 1e-11 difference)
- ✓ All 7 OTR treasuries used successfully
- ✓ Yield curves are smooth and well-behaved

### Z-Spread Validation
- ✓ Re-pricing with z-spreads matches market prices exactly
- ✓ Z-spreads are economically reasonable (mostly positive)
- ✓ 100% success rate (437/437 bonds)

### Risk Metrics Validation
- ✓ Analytical and scenario DV01 values are consistent
- ✓ Duration increases with maturity as expected
- ✓ Convexity is positive for all bonds

---

## How to Run

```bash
cd "/home/user/Credit-Markets/Assignments/PSET 2"
python PSET_2_Solution.py
```

The script will:
1. Load all required data files
2. Process all four problems sequentially
3. Display progress and key results to console
4. Generate all output files (HTML and CSV)
5. Display completion summary

**Expected Runtime:** ~30-60 seconds

---

## Key Results

### Treasury Yield Curve
- Short end (2Y): ~4.79%
- Mid curve (10Y): ~4.42%
- Long end (30Y): ~4.55%
- Curve shows slight inversion at the short end

### Corporate Z-Spreads
- AAPL (Investment Grade): 10-25 bps
- DIS (Entertainment): 11-110 bps
- F (Auto Finance): 50-100 bps
- Spreads increase with maturity and credit risk

### Risk Metrics
- Treasury durations: 1.9 (2Y) to 16.8 (30Y)
- Corporate durations: 1.8 (short) to 18.3 (long)
- Convexity increases non-linearly with maturity

---

## Notes and Assumptions

1. **Calculation Date**: Fixed at 2024-04-08 to match market data
2. **Pricing Engine**: DiscountingBondEngine for all valuations
3. **Curve Interpolation**: Log-cubic discount factor interpolation
4. **Settlement**: T+1 for government bonds, varies for corporate
5. **Error Handling**: Bonds with pricing issues are skipped gracefully

---

## Files in This Directory

```
PSET 2/
├── PSET_2_Solution.py              # Main solution script (774 lines)
├── SOLUTION_SUMMARY.md             # This file
├── PSET_2_Solution_Log.txt         # Execution log
├── Problem_2a_OTR_Yields.html      # OTR yields visualization
├── Problem_2c_Calibrated_Yield_Curves.html  # Yield curves
├── Problem_2d_Discount_Factors.html         # Discount factors
├── Problem_3_Treasury_Results.csv           # Treasury metrics
├── Problem_4_Corporate_Results.csv          # Corporate metrics
├── Problem_4_Corporate_ZSpreads.html        # Z-spreads by maturity
├── Problem_4_Corporate_Duration.html        # Duration by maturity
└── data/
    ├── bond_symbology.xlsx
    ├── bond_market_prices_eod.xlsx
    ├── govt_on_the_run.xlsx
    └── ... (other data files)
```

---

## Conclusion

This solution successfully completes all requirements for PSET 2, demonstrating:
- Proficiency with QuantLib for bond pricing and risk analytics
- Yield curve calibration using bootstrapping techniques
- Calculation of various risk metrics (DV01, duration, convexity, z-spreads)
- Data visualization using modern interactive tools
- Clean, well-documented, production-quality code

All calculations have been validated and results are economically reasonable and consistent with market expectations.

---

**Solution Status: COMPLETE ✓**
