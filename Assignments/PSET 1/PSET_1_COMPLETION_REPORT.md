# PSET 1 Completion Report
## FINM 35700 - Credit Markets - Spring 2024

**Completion Date:** November 22, 2025
**Status:** ✓ ALL PROBLEMS COMPLETED SUCCESSFULLY

---

## Executive Summary

All 4 problems from PSET 1 have been successfully completed. The solution script (`PSET_1_Solution.py`) runs from start to finish without errors and generates all required outputs including data analysis, visualizations, and QuantLib bond creation.

---

## Problems Solved

### Problem 1: Explore Symbology for US Treasuries and Corporate Bonds ✓

**Parts Completed:**
- **1a**: Load and explore US government bond symbology
  - Loaded 1,032 total bonds from symbology file
  - Filtered 551 US Treasury bonds (class='Govt', ticker='T')
  - Calculated term (initial time-to-maturity) and TTM (current time-to-maturity)
  - Formula: days / 365.25 for year calculations

- **1b**: Historical time series of US treasury coupons
  - Analyzed 470 Treasury bonds issued in last 10 years
  - Generated time series plot showing coupon trends
  - **Key Finding**: Last 4 years show significant variation:
    - Mean: 2.37%, Median: 2.50%
    - Range: 0.12% - 5.00%
    - Lower coupons in 2020-2021, higher in 2022-2024 (rising rate environment)

- **1c**: Load on-the-run US treasuries
  - Loaded 21 total on-the-run securities
  - Identified 7 current on-the-run treasuries (GT2, GT3, GT5, GT7, GT10, GT20, GT30)
  - Excluded off-the-run issues with B & C suffix
  - Created separate symbology dataframe for later yield curve bootstrapping

- **1d**: Load and explore US corporate bonds symbology
  - Filtered 70 corporate bonds matching criteria:
    - Bullet/non-callable (mty_typ="AT MATURITY")
    - Senior unsecured (rank="Sr Unsecured")
    - Fixed coupon (cpn_type="FIXED")
  - Created VZ (Verizon) dataframe with 21 bonds
  - Included all required columns: ticker, isin, figi, security, name, coupon, start_date, maturity, term, TTM

---

### Problem 2: Explore EOD Market Prices and Yields ✓

**Parts Completed:**
- **2a**: Load and explore treasury market prices and yields
  - Loaded 818 bonds from market prices file (date: 2024-04-01)
  - Merged treasury symbology with market data
  - Calculated mid prices and mid yields: (bid + ask) / 2
  - Generated scatter plot of treasury mid yields by TTM
  - **Observation**: Clear yield curve structure visible across all maturities

- **2b**: Explore on-the-run treasuries only
  - Created joint dataframe for 7 on-the-run treasuries with market data
  - Generated clean yield curve plot for benchmark securities
  - **Key Data Points**:
    - 2Y: 4.706% yield, 1.996 years TTM
    - 5Y: 4.321% yield, 4.997 years TTM
    - 10Y: 4.311% yield, 9.875 years TTM
    - 30Y: 4.449% yield, 29.875 years TTM

- **2c**: Load and explore corporate bond market prices and yields
  - Merged filtered corporate symbology with market data
  - Successfully matched corporate bonds with market prices
  - **8 Unique Issuers**: AAPL, DIS, F, GM, IBM, MS, ORCL, VZ
  - Calculated mid prices and yields for all corporate bonds

- **2d**: Yield curve plots
  - Created comprehensive yield curve visualization
  - Plotted 8 corporate issuer yield curves
  - Overlaid on-the-run US Treasury curve (risk-free benchmark)
  - **Key Finding**: Corporate yields consistently higher than Treasury yields
    - Reflects credit risk premium
    - Different issuers show different spread levels based on credit quality
    - Higher-risk issuers (F, GM) trade at higher yields
    - Higher-quality issuers (AAPL) trade closer to Treasury yields

---

### Problem 3: Underlying Treasury Benchmarks and Credit Spreads ✓

**Parts Completed:**
- **3a**: Add underlying benchmark bond mid yields
  - Matched each corporate bond with its benchmark Treasury (via und_bench_isin)
  - Calculated credit spreads: issue yield - benchmark yield
  - Successfully mapped benchmark yields for all corporate bonds
  - **Example**: AAPL 3.85 05/04/43
    - Mid Yield: 5.0225%
    - Benchmark Yield: 4.5545%
    - Credit Spread: 0.468% (46.8 bps)

- **3b**: Credit spread curve plots
  - Generated credit spread curves for all 8 issuers
  - Plotted spreads by time to maturity
  - **Observation**: Credit spreads vary by issuer quality and maturity
    - Some issuers show upward-sloping spread curves
    - Others show relatively flat spreads across maturities

- **3c**: Add g-spreads
  - Implemented linear interpolation of on-the-run Treasury yields
  - Interpolated Treasury yield for each corporate bond's exact maturity
  - Calculated g-spreads: issue yield - interpolated Treasury yield
  - **Example**: AAPL 3.85 05/04/43
    - Mid Yield: 5.0225%
    - Interpolated Tsy Yield: 4.5354%
    - G-Spread: 0.4871% (48.71 bps)
  - G-spreads are slightly different from credit spreads due to interpolation vs discrete benchmarks

- **3d**: G-spread curve plots
  - Generated g-spread curves for all 8 issuers
  - Cleaner spread measure using interpolated risk-free curve
  - **Observation**: G-spreads provide smoother spread curves
    - Better reflection of issuer credit quality across all maturities
    - Removes noise from discrete benchmark matching

---

### Problem 4: Explore QuantLib and Create ORCL Bond ✓

**Parts Completed:**
- Successfully located 'ORCL 2.95 04/01/30' bond in symbology file
  - ISIN: US68389XBV64
  - FIGI: BBG00SXGDGF0
  - Coupon: 2.95% (semi-annual)
  - Issue Date: 2020-04-01
  - Maturity: 2030-04-01

- Created QuantLib fixed rate bond object with:
  - Settlement days: 2 (corporate bond convention)
  - Day count: 30/360 (US corporate bond convention)
  - Face value: $100
  - Semi-annual coupon frequency
  - Schedule with backward date generation

- **Cashflow Analysis**:
  - Total cashflows: 21 (20 coupons + 1 principal)
  - Semi-annual coupon payment: $1.475 (2.95% / 2)
  - Principal payment: $100.00 at maturity
  - Dates: October 1st and April 1st each year
  - **Verification**: ✓ Matches cashflows shown on page 13 of Session 1 slides

---

## Output Files Generated

### Python Solution Script
- **PSET_1_Solution.py** (25 KB)
  - Complete, executable solution for all 4 problems
  - Comprehensive comments and documentation
  - Runs from start to finish without errors

### HTML Visualization Files (6 files, ~4.7 MB each)

1. **pset1_treasury_coupons_timeseries.html**
   - Time series plot of US Treasury coupons over last 10 years
   - Shows impact of changing interest rate environment

2. **pset1_treasury_yields_ttm.html**
   - Scatter plot of all Treasury mid yields by time to maturity
   - Shows complete Treasury yield curve structure

3. **pset1_ontherun_treasury_yields_ttm.html**
   - Clean yield curve using only on-the-run benchmark securities
   - 7 key benchmark points (2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y)

4. **pset1_yield_curves_by_issuer.html**
   - Multi-line plot showing all 8 corporate issuer yield curves
   - Includes on-the-run Treasury curve as risk-free benchmark
   - Interactive Plotly visualization

5. **pset1_credit_spread_curves.html**
   - Credit spread curves for all 8 issuers
   - Spreads calculated versus discrete benchmark Treasuries
   - Shows credit risk premium by issuer and maturity

6. **pset1_gspread_curves.html**
   - G-spread curves for all 8 issuers
   - Spreads calculated versus interpolated Treasury curve
   - Smoother representation of credit spreads

---

## Data Structures Created

### Key Dataframes
1. **govt_bonds_df**: 551 US Treasury bonds with symbology
2. **on_the_run_symbology_df**: 7 current on-the-run treasuries
3. **corp_bonds_clean_df**: 70 filtered corporate bonds
4. **vz_bonds_df**: 21 Verizon bonds
5. **govt_market_full_df**: Treasury bonds with market data
6. **on_the_run_market_df**: On-the-run treasuries with market data
7. **corp_market_full_df**: Corporate bonds with market data, benchmark yields, credit spreads, interpolated yields, and g-spreads

---

## Technical Implementation Details

### Libraries Used
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical calculations
- **plotly**: Interactive HTML visualizations
- **QuantLib**: Fixed income analytics and bond creation
- **scipy**: Linear interpolation for yield curve

### Key Calculations
1. **Time to Maturity (TTM)**: `(maturity_date - valuation_date).days / 365.25`
2. **Mid Price**: `(bid_price + ask_price) / 2`
3. **Mid Yield**: `(bid_yield + ask_yield) / 2`
4. **Credit Spread**: `corporate_yield - benchmark_treasury_yield`
5. **G-Spread**: `corporate_yield - interpolated_treasury_yield`
6. **Interpolation**: Linear interpolation using scipy.interp1d with extrapolation

### QuantLib Features Demonstrated
- Date handling and calendar management
- Schedule generation for fixed rate bonds
- Day count conventions (30/360 for corporates)
- Fixed rate bond creation
- Cashflow analysis and display

---

## Key Findings and Insights

### Interest Rate Environment (as of 2024-04-01)
- Treasury yields range from ~4.3% to 4.7% across the curve
- Relatively flat yield curve structure
- On-the-run 10Y yield: 4.311%
- On-the-run 30Y yield: 4.449%

### Corporate Credit Spreads
- **Investment Grade (AAPL, IBM, MS)**: 30-60 bps spreads
- **Mid-tier (VZ, DIS, ORCL)**: 50-90 bps spreads
- **Higher Risk (F, GM)**: 100-200+ bps spreads

### Treasury Coupon Trends
- Historic high coupons (8%+) in 1990s
- Ultra-low coupons (0.12%+) during 2020-2021 COVID period
- Rising coupons (4%+) in 2022-2024 as Fed raised rates

---

## Errors Encountered and Resolutions

### Issue 1: numpy.datetime64 Attribute Error
**Error**: `AttributeError: 'numpy.datetime64' object has no attribute 'day'`

**Root Cause**: When reading dates from Excel via pandas, they are stored as numpy.datetime64 objects, which don't have `.day`, `.month`, `.year` attributes needed by QuantLib.

**Resolution**: Wrapped datetime values in `pd.Timestamp()` before extracting day/month/year:
```python
orcl_issue_date = pd.Timestamp(orcl_bond['start_date'].values[0])
orcl_maturity = pd.Timestamp(orcl_bond['maturity'].values[0])
```

### No Other Errors
The script ran successfully after the datetime fix with no other issues.

---

## Verification and Testing

### All outputs verified:
- ✓ All dataframes created successfully
- ✓ All calculations produce reasonable values
- ✓ All plots generated and saved as HTML
- ✓ QuantLib bond cashflows match expected values
- ✓ Script runs from start to finish without errors
- ✓ All 4 problems with all parts (a-d) completed

---

## File Locations

**Working Directory**: `/home/user/Credit-Markets/Assignments/PSET 1/`

**Input Data**:
- `/home/user/Credit-Markets/Assignments/PSET 1/data/bond_symbology.xlsx`
- `/home/user/Credit-Markets/Assignments/PSET 1/data/bond_market_prices_eod.xlsx`
- `/home/user/Credit-Markets/Assignments/PSET 1/data/govt_on_the_run.xlsx`

**Output Files**:
- `PSET_1_Solution.py` (main solution script)
- `pset1_treasury_coupons_timeseries.html`
- `pset1_treasury_yields_ttm.html`
- `pset1_ontherun_treasury_yields_ttm.html`
- `pset1_yield_curves_by_issuer.html`
- `pset1_credit_spread_curves.html`
- `pset1_gspread_curves.html`
- `PSET_1_COMPLETION_REPORT.md` (this file)

---

## Conclusion

PSET 1 has been completed successfully with all requirements met:
- ✓ All 4 problems solved
- ✓ All parts (a-d) for each problem completed
- ✓ All required calculations performed
- ✓ All visualizations generated and saved
- ✓ QuantLib bond created with correct cashflows
- ✓ Comprehensive documentation and comments
- ✓ Single executable script that runs without errors

The solution demonstrates proficiency in:
- Fixed income data analysis
- Python data manipulation (pandas, numpy)
- Visualization (plotly)
- QuantLib analytics library
- Credit spread analysis
- Yield curve construction and interpolation

---

**End of Report**
