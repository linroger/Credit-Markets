"""
FINM 35700 - Credit Markets
PSET 3 Solution
Spring 2024

Complete solution for all 4 problems:
1. Risk & Scenario analysis for a fixed rate corporate bond (yield model)
2. Perpetual bonds
3. US SOFR swap curve calibration as of 2024-04-15
4. CDS Hazard Rate calibration and valuation

Author: Autonomous Solution
Date: 2024-11-22
"""

# ============================================================================
# IMPORTS
# ============================================================================
import QuantLib as ql
import numpy as np
import pandas as pd
import datetime as dt
import sympy as sp
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Import helper functions from credit_market_tools
from credit_market_tools import *

# ============================================================================
# SETUP
# ============================================================================
print("="*80)
print("FINM 35700 - Credit Markets - PSET 3 Solution")
print("="*80)

# Set the static calculation/valuation date
calc_date = ql.Date(15, 4, 2024)
ql.Settings.instance().evaluationDate = calc_date
print(f"\nCalculation Date: {calc_date}")

# Create output directory for plots
output_dir = "/home/user/Credit-Markets/Assignments/PSET 3/output"
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# PROBLEM 1: RISK & SCENARIO ANALYSIS FOR A FIXED RATE CORPORATE BOND
# ============================================================================
print("\n" + "="*80)
print("PROBLEM 1: RISK & SCENARIO ANALYSIS FOR A FIXED RATE CORPORATE BOND")
print("="*80)

# ----------------------------------------------------------------------------
# Problem 1a: Create generic fixed-rate corporate bond
# ----------------------------------------------------------------------------
print("\n--- Problem 1a: Create generic fixed-rate corporate bond ---")

# Bond specifications: 5% coupon, 10-year maturity from April 15, 2024
test_bond_details = {
    'class': 'Corp',
    'start_date': '2024-04-15',
    'acc_first': '2024-04-15',
    'maturity': '2034-04-15',
    'coupon': 5,
    'dcc': '30/360',
    'days_settle': 2
}

# Create the bond object using helper function
fixed_rate_bond = create_bond_from_symbology(test_bond_details)

# Display bond cashflows
cashflows_df = get_bond_cashflows(fixed_rate_bond, calc_date)
print("\nBond Cashflows:")
print(cashflows_df.to_string())

# ----------------------------------------------------------------------------
# Problem 1b: Compute bond price, DV01, duration and convexity
# ----------------------------------------------------------------------------
print("\n--- Problem 1b: Compute bond price, DV01, duration and convexity ---")

# Market yield of 6%
market_yield = 0.06
compounding = ql.Compounded
frequency = ql.Semiannual
day_count = ql.Thirty360(ql.Thirty360.USA)

# Calculate bond price
bond_price = fixed_rate_bond.cleanPrice(market_yield, day_count, compounding, frequency)
print(f"\nBond Price at 6% yield: ${bond_price:.4f}")

# Calculate DV01 (dollar value of 1 basis point)
price_up = fixed_rate_bond.cleanPrice(market_yield + 0.0001, day_count, compounding, frequency)
dv01 = (price_up - bond_price) / 100
print(f"DV01: ${dv01:.8f}")

# Calculate Modified Duration
modified_duration = ql.BondFunctions.duration(fixed_rate_bond, market_yield,
                                              day_count, compounding, frequency,
                                              ql.Duration.Modified)
print(f"Modified Duration: {modified_duration:.6f} years")

# Calculate Convexity
convexity = ql.BondFunctions.convexity(fixed_rate_bond, market_yield,
                                       day_count, compounding, frequency)
print(f"Convexity: {convexity:.6f}")

# ----------------------------------------------------------------------------
# Problem 1c: Scenario bond prices: "re-pricing" vs "second-order approximations"
# ----------------------------------------------------------------------------
print("\n--- Problem 1c: Scenario bond prices ---")

# Scenario yield grid: 2% to 10% in steps of 0.5%
yields = np.arange(0.02, 0.105, 0.005)

# Calculate scenario bond prices using re-pricing
scenario_prices = [fixed_rate_bond.cleanPrice(y, day_count, compounding, frequency)
                   for y in yields]

# Calculate second-order scenario price approximations
delta_y = yields - market_yield
approx_prices = bond_price * (1 - modified_duration * delta_y + 0.5 * convexity * delta_y**2)

# Create DataFrame for results
scenario_df = pd.DataFrame({
    'Yield (%)': yields * 100,
    'Re-priced': scenario_prices,
    'Approximation': approx_prices,
    'Difference': np.array(scenario_prices) - approx_prices
})

print("\nScenario Analysis (selected yields):")
print(scenario_df.iloc[::4].to_string(index=False))  # Show every 4th row

# Plot scenario prices
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=yields*100, y=scenario_prices,
                          mode='lines+markers', name='Re-priced Bond Prices',
                          line=dict(color='blue', width=2)))
fig1.add_trace(go.Scatter(x=yields*100, y=approx_prices,
                          mode='lines+markers', name='Second-Order Approximation',
                          line=dict(color='red', width=2, dash='dash')))
fig1.update_layout(
    title='Bond Price Scenarios: Re-pricing vs Second-Order Approximations',
    xaxis_title='Yield (%)',
    yaxis_title='Price ($)',
    legend_title='Method',
    template='plotly_white',
    hovermode='x unified'
)
output_file = os.path.join(output_dir, 'problem1c_scenario_prices.html')
fig1.write_html(output_file)
print(f"\nPlot saved to: {output_file}")

# ----------------------------------------------------------------------------
# Problem 1d: Extreme events scenarios
# ----------------------------------------------------------------------------
print("\n--- Problem 1d: Extreme events scenarios ---")

# Extreme yield of 15%
extreme_yield = 0.15

# Calculate bond price at extreme yield
extreme_scenario_price = fixed_rate_bond.cleanPrice(extreme_yield, day_count,
                                                     compounding, frequency)
print(f"\nExtreme Scenario Bond Price (15% yield): ${extreme_scenario_price:.4f}")

# Calculate second-order approximation at extreme yield
delta_y_extreme = extreme_yield - market_yield
extreme_approx_price = bond_price * (1 - modified_duration * delta_y_extreme +
                                     0.5 * convexity * delta_y_extreme**2)
print(f"Second-Order Approximation: ${extreme_approx_price:.4f}")

# Calculate error
error = extreme_scenario_price - extreme_approx_price
error_pct = (error / extreme_scenario_price) * 100
print(f"Approximation Error: ${error:.4f} ({error_pct:.2f}%)")
print("\nThe second-order approximation is less accurate for large yield changes")
print("due to higher-order effects not captured by duration and convexity.")

# Calculate DV01, duration and convexity at extreme yield
extreme_price_up = fixed_rate_bond.cleanPrice(extreme_yield + 0.0001, day_count,
                                              compounding, frequency)
extreme_dv01 = (extreme_price_up - extreme_scenario_price) / 100
extreme_modified_duration = ql.BondFunctions.duration(fixed_rate_bond, extreme_yield,
                                                      day_count, compounding, frequency,
                                                      ql.Duration.Modified)
extreme_convexity = ql.BondFunctions.convexity(fixed_rate_bond, extreme_yield,
                                               day_count, compounding, frequency)

print(f"\nAnalytics at Extreme Yield (15%):")
print(f"DV01: ${extreme_dv01:.8f}")
print(f"Modified Duration: {extreme_modified_duration:.6f} years")
print(f"Convexity: {extreme_convexity:.6f}")

# Mark Problem 1 as complete
print("\n" + "-"*80)
print("PROBLEM 1 COMPLETE")
print("-"*80)

# ============================================================================
# PROBLEM 2: PERPETUAL BONDS
# ============================================================================
print("\n" + "="*80)
print("PROBLEM 2: PERPETUAL BONDS")
print("="*80)

# ----------------------------------------------------------------------------
# Problem 2a: Price a fixed rate perpetual bond
# ----------------------------------------------------------------------------
print("\n--- Problem 2a: Price a fixed rate perpetual bond ---")

# Define symbolic variables
T = sp.symbols('T', positive=True, real=True)
c = sp.symbols('c', positive=True, real=True)
y = sp.symbols('y', positive=True, real=True)

# Generic fixed rate bond PV formula (from homework)
bond_pv_eq = 1 + (c/2 / (sp.exp(y/2) - 1) - 1) * (1 - sp.exp(-T*y))

print("\nGeneric Fixed Rate Bond PV Formula:")
print(bond_pv_eq)

# Take limit as T approaches infinity for perpetual bond
perpetual_bond_pv = sp.limit(bond_pv_eq, T, sp.oo)
perpetual_bond_pv_simplified = sp.simplify(perpetual_bond_pv)

print("\nPerpetual Bond PV (T → ∞):")
print(perpetual_bond_pv_simplified)
print("\nSimplified formula:")
print(f"PV = c / (2*(exp(y/2) - 1))")
print("\nThis represents the present value of a perpetual semi-annual coupon stream.")

# ----------------------------------------------------------------------------
# Problem 2b: Perpetual bonds priced "at par"
# ----------------------------------------------------------------------------
print("\n--- Problem 2b: Perpetual bonds priced 'at par' ---")

# Solve for yield when bond price equals 100 (par)
# PV = c / (2*(exp(y/2) - 1)) = 100
# c / 2 = 100 * (exp(y/2) - 1)
# exp(y/2) = c/200 + 1
# y/2 = ln(c/200 + 1)
# y = 2*ln(c/200 + 1)

at_par_yield = 2 * sp.log(c/200 + 1)
print("\nYield for bond trading at par (PV = 100):")
print(f"y = {at_par_yield}")
print("\nSimplified: y = 2*ln(c/200 + 1)")
print("\nInterpretation: The bond trades at par when the yield equals")
print("twice the natural log of (1 + coupon/200).")
print("For a 5% coupon: y = 2*ln(1.025) ≈ 4.94%")

# Numerical example
c_numeric = 5  # 5% coupon
y_at_par = 2 * np.log(c_numeric/200 + 1)
print(f"\nExample: 5% coupon perpetual bond trades at par with yield = {y_at_par*100:.4f}%")

# ----------------------------------------------------------------------------
# Problem 2c: Duration and DV01 for a fixed rate perpetual bond
# ----------------------------------------------------------------------------
print("\n--- Problem 2c: Duration and DV01 for perpetual bond ---")

# Perpetual bond price formula
perpetual_pv = c / (2 * (sp.exp(y/2) - 1))

# Calculate DV01 (derivative of price with respect to yield, divided by 100)
dpv_dy = sp.diff(perpetual_pv, y)
DV01_formula = dpv_dy / 100

print("\nDV01 Formula (dPV/dy / 100):")
DV01_simplified = sp.simplify(DV01_formula)
print(DV01_simplified)

# Calculate Duration (modified duration = -(1/P) * dP/dy)
Duration_formula = -dpv_dy / perpetual_pv
Duration_simplified = sp.simplify(Duration_formula)

print("\nModified Duration Formula:")
print(Duration_simplified)
print("\nSimplified: Duration = exp(y/2) / (2*exp(y/2) - 2)")

# Numerical example
y_numeric = 0.05  # 5% yield
duration_numeric = np.exp(y_numeric/2) / (2*np.exp(y_numeric/2) - 2)
pv_numeric = c_numeric / (2 * (np.exp(y_numeric/2) - 1))
dv01_numeric = -1 * c_numeric * np.exp(y_numeric/2) / (2 * (np.exp(y_numeric/2) - 1)**2) / 100

print(f"\nExample (5% coupon, 5% yield):")
print(f"Perpetual Bond Price: ${pv_numeric:.4f}")
print(f"Modified Duration: {duration_numeric:.4f} years")
print(f"DV01: ${dv01_numeric:.6f}")

# ----------------------------------------------------------------------------
# Problem 2d: Convexity of a fixed rate perpetual bond
# ----------------------------------------------------------------------------
print("\n--- Problem 2d: Convexity of perpetual bond ---")

# Convexity = (1/P) * d²P/dy²
d2pv_dy2 = sp.diff(dpv_dy, y)
Convexity_formula = d2pv_dy2 / perpetual_pv
Convexity_simplified = sp.simplify(Convexity_formula)

print("\nConvexity Formula (1/P * d²P/dy²):")
print(Convexity_simplified)

# Numerical example
# Second derivative calculation
d2pv_dy2_func = lambda y_val, c_val: (c_val * np.exp(y_val/2) * (2*np.exp(y_val/2) - 1) +
                                      c_val * np.exp(y_val))/(4*(np.exp(y_val/2) - 1)**3)
convexity_numeric = d2pv_dy2_func(y_numeric, c_numeric) / pv_numeric

print(f"\nExample (5% coupon, 5% yield):")
print(f"Convexity: {convexity_numeric:.4f}")

# Mark Problem 2 as complete
print("\n" + "-"*80)
print("PROBLEM 2 COMPLETE")
print("-"*80)

# ============================================================================
# PROBLEM 3: US SOFR SWAP CURVE CALIBRATION
# ============================================================================
print("\n" + "="*80)
print("PROBLEM 3: US SOFR SWAP CURVE CALIBRATION AS OF 2024-04-15")
print("="*80)

# ----------------------------------------------------------------------------
# Problem 3a: Load and explore US SOFR swaps symbology and market data
# ----------------------------------------------------------------------------
print("\n--- Problem 3a: Load and explore SOFR swaps data ---")

# Load data files
data_dir = "/home/user/Credit-Markets/Assignments/PSET 3/data"
sofr_swap_symbology = pd.read_excel(os.path.join(data_dir, 'sofr_swaps_symbology.xlsx'))
sofr_swaps_market_data = pd.read_excel(os.path.join(data_dir, 'sofr_swaps_market_data_eod.xlsx'))

print("\nSOFR Swap Symbology:")
print(sofr_swap_symbology)

print("\nAvailable tenors:", sorted(sofr_swap_symbology['tenor'].unique()))

print("\nSOFR Swaps Market Data (first 5 rows):")
print(sofr_swaps_market_data.head())

print(f"\nAvailable dates: {len(sofr_swaps_market_data['date'].unique())} days")
print(f"Date range: {sofr_swaps_market_data['date'].min()} to {sofr_swaps_market_data['date'].max()}")

# Merge symbology with market data
sofr_merged = pd.merge(sofr_swaps_market_data, sofr_swap_symbology, on='figi')

# Convert date to datetime for plotting
sofr_merged['date'] = pd.to_datetime(sofr_merged['date'])

# Filter for specific tenors to plot
tenors_to_plot = [1, 2, 3, 5, 7, 10, 20, 30]
sofr_plot_data = sofr_merged[sofr_merged['tenor'].isin(tenors_to_plot)]

# Plot historical SOFR rates
fig2 = go.Figure()
for tenor in tenors_to_plot:
    tenor_data = sofr_plot_data[sofr_plot_data['tenor'] == tenor]
    fig2.add_trace(go.Scatter(
        x=tenor_data['date'],
        y=tenor_data['midRate'],
        mode='lines',
        name=f'{tenor}Y'
    ))

fig2.update_layout(
    title='Historical SOFR Swap Rates by Tenor',
    xaxis_title='Date',
    yaxis_title='SOFR Rate (%)',
    legend_title='Tenor',
    template='plotly_white',
    hovermode='x unified'
)
output_file = os.path.join(output_dir, 'problem3a_sofr_historical_rates.html')
fig2.write_html(output_file)
print(f"\nPlot saved to: {output_file}")

# ----------------------------------------------------------------------------
# Problem 3b: Calibrate the US SOFR yield curve
# ----------------------------------------------------------------------------
print("\n--- Problem 3b: Calibrate US SOFR yield curve ---")

# Filter for 2024-04-15
sofr_merged['date'] = pd.to_datetime(sofr_merged['date'])
target_date = pd.to_datetime('2024-04-15')
sofr_20240415 = sofr_merged[sofr_merged['date'] == target_date].copy()

print(f"\nSOFR rates as of 2024-04-15:")
print(sofr_20240415[['tenor', 'midRate']].sort_values('tenor'))

# Define calibration function
def calibrate_sofr_curve_from_frame(calc_date: ql.Date,
                                    sofr_details: pd.DataFrame,
                                    rate_quote_column: str):
    """Create a calibrated yield curve from a SOFR details dataframe."""
    ql.Settings.instance().evaluationDate = calc_date

    # Sort dataframe by maturity
    sorted_details_frame = sofr_details.sort_values(by='tenor')

    # SOFR OIS swap parameters
    settle_days = 2
    day_count = ql.Actual360()
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)

    sofr_helpers = []
    for index, row in sorted_details_frame.iterrows():
        sofr_quote = row[rate_quote_column]
        tenor_in_years = row['tenor']
        sofr_tenor = ql.Period(int(tenor_in_years), ql.Years)

        # Create SOFR OIS rate helper
        sofr_helper = ql.OISRateHelper(
            settle_days,
            sofr_tenor,
            ql.QuoteHandle(ql.SimpleQuote(sofr_quote/100)),
            ql.Sofr()
        )
        sofr_helpers.append(sofr_helper)

    # Bootstrap the yield curve
    sofr_yield_curve = ql.PiecewiseLinearZero(settle_days, calendar,
                                              sofr_helpers, day_count)
    sofr_yield_curve.enableExtrapolation()

    return sofr_yield_curve

# Calibrate the curve
sofr_yield_curve = calibrate_sofr_curve_from_frame(calc_date, sofr_20240415, 'midRate')
sofr_yield_curve_handle = ql.YieldTermStructureHandle(sofr_yield_curve)

print(f"\nSOFR Yield Curve Reference Date: {sofr_yield_curve.referenceDate()}")
print("Curve calibration successful!")

# ----------------------------------------------------------------------------
# Problem 3c: Display the calibrated SOFR discount curve dataframe
# ----------------------------------------------------------------------------
print("\n--- Problem 3c: Display calibrated SOFR discount curve ---")

# Get yield curve details at calibration points
sofr_yield_curve_simple_df = get_yield_curve_details_df(sofr_yield_curve)
print("\nCalibrated SOFR Yield Curve (at calibration nodes):")
print(sofr_yield_curve_simple_df.to_string())

# Get yield curve details at regular grid
grid_dates = [sofr_yield_curve.referenceDate() + ql.Period(i, ql.Years)
              for i in range(0, 31, 2)]
sofr_yield_curve_details_df = get_yield_curve_details_df(sofr_yield_curve, grid_dates)
print("\nSOFR Yield Curve (2-year grid):")
print(sofr_yield_curve_details_df.to_string())

# ----------------------------------------------------------------------------
# Problem 3d: Plot the calibrated US SOFR curves
# ----------------------------------------------------------------------------
print("\n--- Problem 3d: Plot calibrated SOFR curves ---")

# Plot Zero Rates
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=sofr_yield_curve_details_df['Date'],
    y=sofr_yield_curve_details_df['ZeroRate'],
    mode='lines+markers',
    name='Zero Rate',
    line=dict(color='blue', width=2),
    marker=dict(size=6)
))
fig3.update_layout(
    title='SOFR Curve: Zero Rates (as of 2024-04-15)',
    xaxis_title='Date',
    yaxis_title='Zero Rate (%)',
    template='plotly_white',
    hovermode='x unified'
)
output_file = os.path.join(output_dir, 'problem3d_sofr_zero_rates.html')
fig3.write_html(output_file)
print(f"\nZero rates plot saved to: {output_file}")

# Plot Discount Factors
fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=sofr_yield_curve_details_df['Date'],
    y=sofr_yield_curve_details_df['DiscountFactor'],
    mode='lines+markers',
    name='Discount Factor',
    line=dict(color='green', width=2),
    marker=dict(size=6)
))
fig4.update_layout(
    title='SOFR Curve: Discount Factors (as of 2024-04-15)',
    xaxis_title='Date',
    yaxis_title='Discount Factor',
    template='plotly_white',
    hovermode='x unified'
)
output_file = os.path.join(output_dir, 'problem3d_sofr_discount_factors.html')
fig4.write_html(output_file)
print(f"Discount factors plot saved to: {output_file}")

# Mark Problem 3 as complete
print("\n" + "-"*80)
print("PROBLEM 3 COMPLETE")
print("-"*80)

# ============================================================================
# PROBLEM 4: CDS HAZARD RATE CALIBRATION AND VALUATION
# ============================================================================
print("\n" + "="*80)
print("PROBLEM 4: CDS HAZARD RATE CALIBRATION AND VALUATION")
print("="*80)

# ----------------------------------------------------------------------------
# Problem 4a: Load and explore CDS market data
# ----------------------------------------------------------------------------
print("\n--- Problem 4a: Load and explore CDS market data ---")

# Load CDS market data
cds_market_data = pd.read_excel(os.path.join(data_dir, 'cds_market_data_eod.xlsx'))

print("\nCDS Market Data (first 5 rows):")
print(cds_market_data.head())

print(f"\nAvailable dates: {len(cds_market_data['date'].unique())} days")
print(f"Date range: {cds_market_data['date'].min()} to {cds_market_data['date'].max()}")

# Convert date to datetime
cds_market_data['date'] = pd.to_datetime(cds_market_data['date'])

# Plot historical CDS par spreads
fig5 = go.Figure()
tenors = ['1y', '2y', '3y', '5y', '7y', '10y']
colors = ['blue', 'cyan', 'green', 'orange', 'red', 'purple']

for tenor, color in zip(tenors, colors):
    col_name = f'par_spread_{tenor}'
    if col_name in cds_market_data.columns:
        fig5.add_trace(go.Scatter(
            x=cds_market_data['date'],
            y=cds_market_data[col_name],
            mode='lines',
            name=tenor.upper(),
            line=dict(color=color)
        ))

fig5.update_layout(
    title='Historical CDS Par Spreads by Tenor (IBM)',
    xaxis_title='Date',
    yaxis_title='CDS Par Spread (bps)',
    legend_title='Tenor',
    template='plotly_white',
    hovermode='x unified'
)
output_file = os.path.join(output_dir, 'problem4a_cds_historical_spreads.html')
fig5.write_html(output_file)
print(f"\nPlot saved to: {output_file}")

# ----------------------------------------------------------------------------
# Problem 4b: Calibrate IBM hazard rate curve
# ----------------------------------------------------------------------------
print("\n--- Problem 4b: Calibrate IBM hazard rate curve ---")

# Filter for 2024-04-15
target_date = pd.to_datetime('2024-04-15')
cds_20240415 = cds_market_data[cds_market_data['date'] == target_date]

if len(cds_20240415) == 0:
    print("\nWarning: No data for 2024-04-15, using latest available date")
    latest_date = cds_market_data['date'].max()
    cds_20240415 = cds_market_data[cds_market_data['date'] == latest_date]
    print(f"Using date: {latest_date}")

print("\nCDS Par Spreads as of calibration date:")
for tenor in tenors:
    col_name = f'par_spread_{tenor}'
    if col_name in cds_20240415.columns:
        value = cds_20240415[col_name].values[0]
        print(f"{tenor.upper()}: {value:.2f} bps")

# CDS calibration parameters
CDS_recovery_rate = 0.4
CDS_day_count = ql.Actual360()

# Extract CDS spreads for calibration
CDS_tenors = [ql.Period(int(t[:-1]), ql.Years) for t in tenors]
CDS_spreads = [cds_20240415[f'par_spread_{t}'].values[0] for t in tenors]

# Create CDS helpers
settle_days = 2
CDS_helpers = []
for (CDS_spread, CDS_tenor) in zip(CDS_spreads, CDS_tenors):
    cds_helper = ql.SpreadCdsHelper(
        CDS_spread / 10000.0,  # Convert bps to decimal
        CDS_tenor,
        settle_days,
        ql.TARGET(),
        ql.Quarterly,
        ql.Following,
        ql.DateGeneration.TwentiethIMM,
        CDS_day_count,
        CDS_recovery_rate,
        sofr_yield_curve_handle
    )
    CDS_helpers.append(cds_helper)

# Bootstrap hazard rate curve
hazard_rate_curve = ql.PiecewiseFlatHazardRate(calc_date, CDS_helpers, CDS_day_count)
hazard_rate_curve.enableExtrapolation()

# Extract hazard rates and survival probabilities
hazard_list = [(hr[0].to_date(),
                CDS_day_count.yearFraction(calc_date, hr[0]),
                hr[1] * 100,
                np.exp(-hr[1] * CDS_day_count.yearFraction(calc_date, hr[0])),
                hazard_rate_curve.survivalProbability(hr[0]))
               for hr in hazard_rate_curve.nodes()]

hazard_rates_df = pd.DataFrame(hazard_list,
                               columns=['Date', 'YearFrac', 'HazardRate',
                                       'SPManual', 'SurvivalProb'])

print("\nCalibrated Hazard Rate Curve:")
print(hazard_rates_df.to_string())

# ----------------------------------------------------------------------------
# Problem 4c: Plot calibrated Hazard Rates and Survival Probability curves
# ----------------------------------------------------------------------------
print("\n--- Problem 4c: Plot calibrated curves ---")

# Plot Hazard Rates
fig6 = go.Figure()
fig6.add_trace(go.Scatter(
    x=hazard_rates_df['Date'],
    y=hazard_rates_df['HazardRate'],
    mode='lines+markers',
    name='Hazard Rate',
    line=dict(color='red', width=2),
    marker=dict(size=8)
))
fig6.update_layout(
    title='IBM Hazard Rates Curve (as of 2024-04-15)',
    xaxis_title='Date',
    yaxis_title='Hazard Rate (%)',
    template='plotly_white',
    hovermode='x unified'
)
output_file = os.path.join(output_dir, 'problem4c_hazard_rates.html')
fig6.write_html(output_file)
print(f"\nHazard rates plot saved to: {output_file}")

# Plot Survival Probabilities
fig7 = go.Figure()
fig7.add_trace(go.Scatter(
    x=hazard_rates_df['Date'],
    y=hazard_rates_df['SurvivalProb'],
    mode='lines+markers',
    name='Survival Probability',
    line=dict(color='blue', width=2),
    marker=dict(size=8)
))
fig7.update_layout(
    title='IBM Survival Probability Curve (as of 2024-04-15)',
    xaxis_title='Date',
    yaxis_title='Survival Probability',
    template='plotly_white',
    hovermode='x unified'
)
output_file = os.path.join(output_dir, 'problem4c_survival_probability.html')
fig7.write_html(output_file)
print(f"Survival probability plot saved to: {output_file}")

# ----------------------------------------------------------------------------
# Problem 4d: Compute fair/par spread and PV of a CDS
# ----------------------------------------------------------------------------
print("\n--- Problem 4d: CDS valuation ---")

# CDS specifications
side = ql.Protection.Seller
face_notional = 100
contractual_spread = 100 / 10000  # 100 bps

# CDS dates
cds_start_date = calc_date
cds_maturity_date = ql.Date(20, 6, 2029)

# Create CDS schedule
cds_schedule = ql.MakeSchedule(
    cds_start_date,
    cds_maturity_date,
    ql.Period('3M'),
    ql.Quarterly,
    ql.TARGET(),
    ql.Following,
    ql.Unadjusted,
    ql.DateGeneration.TwentiethIMM
)

# Create CDS object
cds_obj = ql.CreditDefaultSwap(
    side,
    face_notional,
    contractual_spread,
    cds_schedule,
    ql.Following,
    ql.Actual360()
)

# Create pricing engine
cds_surv_prob_curve_handle = ql.DefaultProbabilityTermStructureHandle(hazard_rate_curve)
cds_pricing_engine = ql.MidPointCdsEngine(
    cds_surv_prob_curve_handle,
    CDS_recovery_rate,
    sofr_yield_curve_handle
)
cds_obj.setPricingEngine(cds_pricing_engine)

# Print CDS valuation results
print(f"\nCDS Specifications:")
print(f"Contractual Spread: {contractual_spread*10000:.0f} bps")
print(f"Maturity: {cds_maturity_date}")
print(f"Notional: ${face_notional}")
print(f"Recovery Rate: {CDS_recovery_rate*100:.0f}%")

print(f"\nCDS Valuation Results:")
print(f"Protection Start Date: {cds_obj.protectionStartDate()}")
print(f"Fair/Par Spread: {cds_obj.fairSpread()*10000:.3f} bps")
print(f"CDS PV: ${cds_obj.NPV():.4f}")
print(f"Premium Leg PV: ${cds_obj.couponLegNPV():.4f}")
print(f"Default Leg PV: ${cds_obj.defaultLegNPV():.4f}")
print(f"Survival Probability to Maturity: {hazard_rate_curve.survivalProbability(cds_maturity_date):.4f}")

# Mark Problem 4 as complete
print("\n" + "-"*80)
print("PROBLEM 4 COMPLETE")
print("-"*80)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ALL PROBLEMS COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"\nAll plots have been saved to: {output_dir}")
print("\nSummary of outputs:")
print("  - problem1c_scenario_prices.html")
print("  - problem3a_sofr_historical_rates.html")
print("  - problem3d_sofr_zero_rates.html")
print("  - problem3d_sofr_discount_factors.html")
print("  - problem4a_cds_historical_spreads.html")
print("  - problem4c_hazard_rates.html")
print("  - problem4c_survival_probability.html")
print("\n" + "="*80)
