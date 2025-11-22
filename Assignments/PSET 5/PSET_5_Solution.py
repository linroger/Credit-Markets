"""
FINM 35700 - Credit Markets
Spring 2024
Problem Set 5 - Complete Solution

Author: Solution Script
Date: 2024-11-22

This script provides complete solutions to all problems in Homework 5:
1. Fixed rate bond prices and sensitivities (True/False questions)
2. Credit Default Swaps - hazard rate model (True/False questions)
3. Pricing bonds in the Merton Structural Credit Model
4. Credit ETF analysis on HYG

Requirements:
- QuantLib for credit modeling
- pandas, numpy, scipy for numerical computations
- plotly for interactive visualizations
"""

import QuantLib as ql
import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import minimize, root_scalar
from scipy.stats import norm
import plotly.graph_objects as go
import plotly.express as px
import os

# Set display options for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("="*80)
print("FINM 35700 - Credit Markets - Spring 2024")
print("Problem Set 5 - Complete Solution")
print("="*80)
print()

# ============================================================================
# HELPER FUNCTIONS (from credit_market_tools.py)
# ============================================================================

def get_ql_date(date) -> ql.Date:
    """Convert dt.date or string to ql.Date"""
    if isinstance(date, dt.date):
        return ql.Date(date.day, date.month, date.year)
    elif isinstance(date, str):
        date = dt.datetime.strptime(date, "%Y-%m-%d").date()
        return ql.Date(date.day, date.month, date.year)
    else:
        raise ValueError(f"to_qldate, {type(date)}, {date}")

def create_schedule_from_symbology(details: dict):
    """Create a QuantLib cashflow schedule from symbology details dictionary"""
    maturity = get_ql_date(details['maturity'])
    acc_first = get_ql_date(details['acc_first'])
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    period = ql.Period(2)  # Semi-annual
    business_day_convention = ql.Unadjusted
    termination_date_convention = ql.Unadjusted
    date_generation = ql.DateGeneration.Backward

    schedule = ql.MakeSchedule(
        effectiveDate=acc_first,
        terminationDate=maturity,
        tenor=period,
        calendar=calendar,
        convention=business_day_convention,
        terminalDateConvention=termination_date_convention,
        rule=date_generation,
        endOfMonth=True,
        firstDate=ql.Date(),
        nextToLastDate=ql.Date()
    )
    return schedule

def create_bond_from_symbology(details: dict):
    """Create a US fixed rate bond object from symbology details dictionary"""
    # Day count convention
    if details['class'] == 'Corp':
        day_count = ql.Thirty360(ql.Thirty360.USA)
    elif details['class'] == 'Govt':
        day_count = ql.ActualActual(ql.ActualActual.ISMA)
    else:
        raise ValueError(f"unsupported asset class: {details['class']}")

    issue_date = get_ql_date(details['start_date'])
    days_settle = int(float(details['days_settle']))
    coupon = float(details['coupon']) / 100.0

    schedule = create_schedule_from_symbology(details)

    face_value = 100
    redemption = 100
    payment_convention = ql.Unadjusted

    fixed_rate_bond = ql.FixedRateBond(
        days_settle,
        face_value,
        schedule,
        [coupon],
        day_count,
        payment_convention,
        redemption,
        issue_date
    )

    return fixed_rate_bond

# ============================================================================
# SETUP: Calculation Date and Data Paths
# ============================================================================

# Use static calculation/valuation date of 2024-04-26
calc_date = ql.Date(26, 4, 2024)
ql.Settings.instance().evaluationDate = calc_date
as_of_date = pd.to_datetime('2024-04-26')

# Data paths
data_dir = "/home/user/Credit-Markets/Assignments/PSET 5/data"
output_dir = "/home/user/Credit-Markets/Assignments/PSET 5"

print(f"Calculation Date: {calc_date}")
print(f"Data Directory: {data_dir}")
print(f"Output Directory: {output_dir}")
print()

# ============================================================================
# PROBLEM 1: Fixed Rate Bond Prices and Sensitivities (20 points)
# ============================================================================

print("="*80)
print("PROBLEM 1: Fixed Rate Bond Prices and Sensitivities")
print("="*80)
print()

print("Bond valuation formula in flat yield model (formula [6] from Lecture 1):")
print("PV_Bond(c, T, y_sa) = 1 + (c - y_sa)/y_sa * [1 - (1 + y_sa/2)^(-2T)]")
print()

# Part a: True or False (fixed rate bond prices)
print("-" * 80)
print("Part a: True or False (Fixed Rate Bond Prices)")
print("-" * 80)
print()

answers_1a = {
    "1. Fixed rate bond price is increasing in yield":
        "FALSE - A fixed rate bond's price decreases as yields increase (inverse relationship).",

    "2. Fixed rate bond price is increasing in coupon":
        "TRUE - A higher coupon increases the cash flows, which raises the present value.",

    "3. Fixed rate bond price is increasing in bond maturity":
        "FALSE - The relationship is ambiguous. For discount bonds, price increases with maturity. "
        "For premium bonds, price decreases with maturity. It depends on the relationship between "
        "coupon and yield.",

    "4. Fixed rate callable bond prices are higher or equal to their 'bullet' (non-callable) version":
        "FALSE - Callable bonds trade at a discount due to the embedded call option that benefits "
        "the issuer. The call option reduces the value to the bondholder."
}

for question, answer in answers_1a.items():
    print(f"{question}")
    print(f"   Answer: {answer}")
    print()

# Part b: True or False (fixed rate bond yields)
print("-" * 80)
print("Part b: True or False (Fixed Rate Bond Yields)")
print("-" * 80)
print()

answers_1b = {
    "1. Fixed rate bond yield is increasing in interest rate":
        "TRUE - Higher interest rates lead to lower bond prices, which increases yields.",

    "2. Fixed rate bond yield is increasing in credit spread":
        "TRUE - A higher credit spread means the bond earns a higher yield over the risk-free rate.",

    "3. Fixed rate bond yield is increasing in coupon":
        "FALSE - Higher coupon bonds typically trade closer to par and have lower yields to maturity "
        "compared to similar maturity low-coupon bonds.",

    "4. Fixed rate bond yield is increasing in bond maturity":
        "DEPENDS - This depends on the shape of the yield curve. In a normal upward-sloping yield curve, "
        "yields increase with maturity. In an inverted curve, they decrease.",

    "5. Fixed rate callable bond yields are lower or equal to their 'bullet' (non-callable) version":
        "FALSE - Callable bonds have higher yields to compensate investors for the call risk "
        "(reinvestment risk if called early)."
}

for question, answer in answers_1b.items():
    print(f"{question}")
    print(f"   Answer: {answer}")
    print()

# Part c: True or False (fixed rate bond durations)
print("-" * 80)
print("Part c: True or False (Fixed Rate Bond Durations)")
print("-" * 80)
print()

answers_1c = {
    "1. Fixed rate bond duration is increasing with yield":
        "FALSE - Duration is inversely related to yield. Higher yields result in lower durations "
        "because future cash flows are discounted more heavily.",

    "2. Fixed rate bond duration is increasing in coupon":
        "FALSE - Higher coupon rates result in lower durations because more of the bond's value "
        "comes from earlier cash flows.",

    "3. Fixed rate bond duration is increasing with bond maturity":
        "TRUE - Longer-dated bonds have longer durations because their cash flows are further "
        "in the future.",

    "4. Fixed rate callable bond durations are higher or equal to their 'bullet' (non-callable) version":
        "FALSE - Callable bonds have lower durations than non-callable bonds because the call option "
        "effectively shortens the bond's expected life."
}

for question, answer in answers_1c.items():
    print(f"{question}")
    print(f"   Answer: {answer}")
    print()

# Part d: True or False (fixed rate bond convexities)
print("-" * 80)
print("Part d: True or False (Fixed Rate Bond Convexities)")
print("-" * 80)
print()

answers_1d = {
    "1. Fixed rate bond convexity is increasing with yield":
        "FALSE - Convexity tends to decrease as yields rise. At higher yield levels, "
        "the price-yield curve flattens, reducing convexity.",

    "2. Fixed rate bond convexity is increasing in coupon":
        "FALSE - Bonds with higher coupon rates have lower convexity because the cash flows "
        "are received earlier, reducing the compounding effect.",

    "3. Fixed rate bond convexity is increasing with bond maturity":
        "TRUE - Convexity increases with longer maturities because bonds with longer maturities "
        "are more sensitive to yield changes.",

    "4. Fixed rate callable bond convexities are higher or equal to their 'bullet' (non-callable) version":
        "FALSE - Callable bonds exhibit negative convexity when the call option is near the money, "
        "as price appreciation is limited when yields fall."
}

for question, answer in answers_1d.items():
    print(f"{question}")
    print(f"   Answer: {answer}")
    print()

# ============================================================================
# PROBLEM 2: Credit Default Swaps - Hazard Rate Model (20 points)
# ============================================================================

print("="*80)
print("PROBLEM 2: Credit Default Swaps (Hazard Rate Model)")
print("="*80)
print()

print("CDS valuation formulas in simple hazard rate model (from Lecture 3):")
print("PV_CDS_PL = (c/(r+h)) * [1 - exp(-T*(r+h))]")
print("PV_CDS_DL = ((1-R)*h/(r+h)) * [1 - exp(-T*(r+h))]")
print("PV_CDS = PV_CDS_PL - PV_CDS_DL")
print("CDS_ParSpread ≈ (1-R) * h")
print()

# Part a: True or False (CDS Premium Leg PV)
print("-" * 80)
print("Part a: True or False (CDS Premium Leg PV)")
print("-" * 80)
print()

answers_2a = {
    "1. CDS premium leg PV is increasing in CDS Par Spread":
        "TRUE - Higher spreads mean larger premium payments, increasing the PV of the premium leg.",

    "2. CDS premium leg PV is increasing in interest rate":
        "FALSE - Higher interest rates increase discounting, reducing the PV of future premium payments.",

    "3. CDS premium leg PV is increasing in hazard rate":
        "FALSE - Higher hazard rates imply shorter expected duration (higher default probability), "
        "reducing the expected number of premium payments.",

    "4. CDS premium leg PV is increasing in recovery rate":
        "FALSE - Recovery rate does not directly affect the premium leg PV, which depends on "
        "the coupon rate and survival probabilities.",

    "5. CDS premium leg PV is increasing in coupon":
        "TRUE - Higher coupon rates directly increase the premium amounts paid, raising their PV.",

    "6. CDS premium leg PV is increasing in CDS maturity":
        "TRUE - Longer maturities mean more premium payments, increasing the PV (though subject "
        "to discounting and survival probability effects)."
}

for question, answer in answers_2a.items():
    print(f"{question}")
    print(f"   Answer: {answer}")
    print()

# Part b: True or False (CDS Default Leg PV)
print("-" * 80)
print("Part b: True or False (CDS Default Leg PV)")
print("-" * 80)
print()

answers_2b = {
    "1. CDS default leg PV is increasing in CDS Par Spread":
        "FALSE - The par spread doesn't directly affect the default leg PV, which depends on "
        "default probability and loss given default.",

    "2. CDS default leg PV is increasing in interest rate":
        "FALSE - Higher interest rates increase discounting, reducing the PV of potential default payouts.",

    "3. CDS default leg PV is increasing in hazard rate":
        "TRUE - Higher hazard rates mean higher default probability, increasing the expected payout "
        "and PV of the default leg.",

    "4. CDS default leg PV is increasing in recovery rate":
        "FALSE - Higher recovery rates reduce the loss given default (1-R), decreasing the default leg PV.",

    "5. CDS default leg PV is increasing in coupon":
        "FALSE - The coupon rate affects the premium leg, not the default leg.",

    "6. CDS default leg PV is increasing in CDS maturity":
        "TRUE - Longer maturities increase the time window for default to occur, raising the "
        "expected payout (though discounting effects apply)."
}

for question, answer in answers_2b.items():
    print(f"{question}")
    print(f"   Answer: {answer}")
    print()

# Part c: True or False (CDS PV)
print("-" * 80)
print("Part c: True or False (CDS PV)")
print("-" * 80)
print()

answers_2c = {
    "1. CDS PV is increasing in CDS Par Spread":
        "TRUE - Higher spreads increase the premium leg value more than proportionally, "
        "increasing total CDS PV (for protection buyer, becomes more negative).",

    "2. CDS PV is increasing in interest rate":
        "AMBIGUOUS - Interest rates affect both premium and default legs through discounting. "
        "The net effect depends on their relative magnitudes.",

    "3. CDS PV is increasing in hazard rate":
        "AMBIGUOUS - Higher hazard rates increase default leg PV but decrease premium leg PV. "
        "The net effect depends on the par spread vs. actual hazard rate.",

    "4. CDS PV is increasing in recovery rate":
        "FALSE - Higher recovery rates decrease the default leg value, making the CDS less valuable "
        "to the protection buyer (assuming constant spread).",

    "5. CDS PV is increasing in coupon":
        "DEPENDS - Higher coupons increase the premium leg cost. If trading at par, PV = 0. "
        "If off-market, this affects the net PV.",

    "6. CDS PV is increasing in CDS maturity":
        "AMBIGUOUS - Both premium and default legs increase with maturity, but the net effect "
        "depends on the spread level and term structure."
}

for question, answer in answers_2c.items():
    print(f"{question}")
    print(f"   Answer: {answer}")
    print()

# Part d: True or False (CDS Par Spread)
print("-" * 80)
print("Part d: True or False (CDS Par Spread)")
print("-" * 80)
print()

answers_2d = {
    "1. CDS Par Spread is increasing in interest rates":
        "FALSE - Par spread is primarily determined by credit risk (hazard rate), not interest rates. "
        "Interest rates affect discounting but not the spread directly.",

    "2. CDS Par Spread is increasing in hazard rate":
        "TRUE - Par spread ≈ (1-R) * h, so it increases linearly with hazard rate.",

    "3. CDS Par Spread is increasing in recovery rate":
        "FALSE - Par spread ≈ (1-R) * h, so it decreases as recovery rate increases.",

    "4. CDS Par Spread is increasing in coupon":
        "FALSE - The par spread is independent of the coupon. Par spread is the coupon that "
        "makes the CDS worth zero at inception.",

    "5. CDS Par Spread is increasing in CDS maturity":
        "DEPENDS - This depends on the term structure of credit spreads. If hazard rates are "
        "increasing with maturity, par spreads will also increase."
}

for question, answer in answers_2d.items():
    print(f"{question}")
    print(f"   Answer: {answer}")
    print()

# ============================================================================
# PROBLEM 3: Pricing Bonds in the Merton Structural Credit Model (30 points)
# ============================================================================

print("="*80)
print("PROBLEM 3: Pricing Bonds in the Merton Structural Credit Model")
print("="*80)
print()

print("Using the Merton Model to price corporate bonds and equity as options on firm assets.")
print()

# Given parameters
A0 = 125e6      # Total assets ($125 MM)
K = 100e6       # Face value of debt ($100 MM)
T = 5           # Time to maturity (years)
sigma = 0.20    # Asset volatility (20%)
r = 0.04        # Risk-free rate (4%)

print(f"Input Parameters:")
print(f"  Assets (A0):              ${A0:,.0f}")
print(f"  Debt Face Value (K):      ${K:,.0f}")
print(f"  Maturity (T):             {T} years")
print(f"  Asset Volatility (σ):     {sigma:.2%}")
print(f"  Risk-Free Rate (r):       {r:.2%}")
print()

# Part a: Company balance sheet metrics & fair value of equity
print("-" * 80)
print("Part a: Company Balance Sheet Metrics & Fair Value of Equity")
print("-" * 80)
print()

# Calculate leverage and book value of equity
L = K / A0
BVE = A0 - K

print(f"Leverage (K/A0):          {L:.4f}")
print(f"Book Value of Equity:     ${BVE:,.2f}")
print()

# Calculate d1 and d2 for Black-Scholes formula
d1 = (np.log(A0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

print(f"Black-Scholes Parameters:")
print(f"  d1:                      {d1:.6f}")
print(f"  d2:                      {d2:.6f}")
print()

# Fair value of equity (call option on assets)
E0 = A0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

print(f"Fair Value of Equity (E0): ${E0:,.2f}")
print(f"  (Equity as a call option on firm assets with strike K)")
print()

# Part b: Risky Bond Valuation (Fair Value of Liabilities)
print("-" * 80)
print("Part b: Risky Bond Valuation (Fair Value of Liabilities)")
print("-" * 80)
print()

# Fair value of liabilities
B0 = A0 - E0

print(f"Fair Value of Liabilities (B0): ${B0:,.2f}")
print(f"  (Using balance sheet identity: B0 = A0 - E0)")
print()

# Risk-free bond value for comparison
B0_riskfree = K * np.exp(-r * T)
print(f"Risk-Free Bond Value:           ${B0_riskfree:,.2f}")
print(f"Credit Risk Discount:           ${B0_riskfree - B0:,.2f}")
print()

# Part c: Flat yield, spread and hazard rate
print("-" * 80)
print("Part c: Flat Yield, Spread and Hazard Rate")
print("-" * 80)
print()

# Distance to default
distance_to_default = d2

# Default probability
default_probability = norm.cdf(-d2)

# Bond yield (from risky bond valuation formula)
bond_yield = -(1/T) * np.log((1/L) * norm.cdf(-d1) + np.exp(-r * T) * norm.cdf(d2))

# Bond credit spread
credit_spread = bond_yield - r

# Flat hazard rate
flat_hazard_rate = -(1/T) * np.log(norm.cdf(d1))

# Expected recovery on default
expected_recovery = A0 / K

print(f"Credit Risk Metrics:")
print(f"  Distance to Default:        {distance_to_default:.6f}")
print(f"  Default Probability:        {default_probability:.4%}")
print(f"  Bond Yield:                 {bond_yield:.4%}")
print(f"  Bond Credit Spread:         {credit_spread:.4%} ({credit_spread * 10000:.2f} bps)")
print(f"  Flat Hazard Rate:           {flat_hazard_rate:.4%} ({flat_hazard_rate * 10000:.2f} bps)")
print(f"  Expected Recovery:          {expected_recovery:.4f}")
print()

# Create plots for credit spreads and expected recovery vs. initial asset values
print("Creating plots for credit spreads and expected recovery...")

# Grid of asset values from $50 MM to $200 MM in steps of $5 MM
A0_grid = np.arange(50e6, 205e6, 5e6)

# Calculate metrics for each asset value
spreads = []
recoveries = []
default_probs = []
hazard_rates = []

for A0_val in A0_grid:
    L_val = K / A0_val
    d1_val = (np.log(A0_val / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2_val = d1_val - sigma * np.sqrt(T)

    # Bond yield and spread
    yield_val = -(1/T) * np.log((1/L_val) * norm.cdf(-d1_val) + np.exp(-r * T) * norm.cdf(d2_val))
    spread_val = yield_val - r
    spreads.append(spread_val * 10000)  # in bps

    # Expected recovery
    recovery_val = A0_val / K
    recoveries.append(recovery_val)

    # Additional metrics
    default_probs.append(norm.cdf(-d2_val))
    hazard_rates.append(-(1/T) * np.log(norm.cdf(d1_val)) * 10000)  # in bps

# Plot 1: Bond Credit Spreads vs. Initial Asset Values
fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=A0_grid / 1e6,
    y=spreads,
    mode='lines',
    name='Bond Credit Spread',
    line=dict(color='red', width=2)
))
fig1.update_layout(
    title='Bond Credit Spreads vs. Initial Asset Values (Merton Model)',
    xaxis_title='Initial Asset Value ($ Millions)',
    yaxis_title='Credit Spread (bps)',
    template='plotly_white',
    hovermode='x unified'
)
output_file1 = os.path.join(output_dir, 'Problem3c_CreditSpreads.html')
fig1.write_html(output_file1)
print(f"  Saved: {output_file1}")

# Plot 2: Expected Recovery on Default vs. Initial Asset Values
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=A0_grid / 1e6,
    y=recoveries,
    mode='lines',
    name='Expected Recovery',
    line=dict(color='green', width=2)
))
fig2.add_hline(y=1.0, line_dash="dash", line_color="gray",
               annotation_text="Full Recovery (100%)")
fig2.update_layout(
    title='Expected Recovery on Default vs. Initial Asset Values (Merton Model)',
    xaxis_title='Initial Asset Value ($ Millions)',
    yaxis_title='Expected Recovery Rate',
    template='plotly_white',
    hovermode='x unified'
)
output_file2 = os.path.join(output_dir, 'Problem3c_ExpectedRecovery.html')
fig2.write_html(output_file2)
print(f"  Saved: {output_file2}")
print()

# Part d: Equity volatility
print("-" * 80)
print("Part d: Equity Volatility")
print("-" * 80)
print()

# Equity volatility using Merton model
# σ_E = (A0/E0) * N(d1) * σ_A
sigma_E = (A0 / E0) * norm.cdf(d1) * sigma

print(f"Equity Volatility (σ_E):      {sigma_E:.4f} ({sigma_E:.2%})")
print(f"Asset Volatility (σ_A):       {sigma:.4f} ({sigma:.2%})")
print(f"Leverage Effect:              {sigma_E / sigma:.2f}x")
print()

print("Creating plot for equity volatility vs. asset values...")

# Calculate equity volatility for each asset value
equity_vols = []

for A0_val in A0_grid:
    d1_val = (np.log(A0_val / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2_val = d1_val - sigma * np.sqrt(T)

    # Fair value of equity
    E0_val = A0_val * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)

    # Equity volatility
    if E0_val > 0:
        sigma_E_val = (A0_val / E0_val) * norm.cdf(d1_val) * sigma
        equity_vols.append(sigma_E_val)
    else:
        equity_vols.append(np.nan)

# Plot: Equity Volatility vs. Initial Asset Values
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=A0_grid / 1e6,
    y=equity_vols,
    mode='lines',
    name='Equity Volatility',
    line=dict(color='blue', width=2)
))
fig3.add_hline(y=sigma, line_dash="dash", line_color="gray",
               annotation_text=f"Asset Volatility = {sigma:.2%}")
fig3.update_layout(
    title='Equity Volatility vs. Initial Asset Values (Merton Model)',
    xaxis_title='Initial Asset Value ($ Millions)',
    yaxis_title='Equity Volatility',
    template='plotly_white',
    hovermode='x unified'
)
output_file3 = os.path.join(output_dir, 'Problem3d_EquityVolatility.html')
fig3.write_html(output_file3)
print(f"  Saved: {output_file3}")
print()

print("Interpretation:")
print("  - As asset value increases, equity volatility decreases")
print("  - This is due to the leverage effect: lower leverage → lower equity volatility")
print("  - When firm is near distress (low A0), equity becomes highly levered and volatile")
print()

# ============================================================================
# PROBLEM 4: Credit ETF Analysis on HYG (30 points)
# ============================================================================

print("="*80)
print("PROBLEM 4: Credit ETF Analysis on HYG")
print("="*80)
print()

# Part a: Load and explore the HYG basket composition and market data
print("-" * 80)
print("Part a: Load and Explore HYG Basket Composition and Market Data")
print("-" * 80)
print()

# Load data files
hyg_basket_file = os.path.join(data_dir, 'hyg_basket_composition.xlsx')
hyg_symbology_file = os.path.join(data_dir, 'hyg_corp_symbology.xlsx')

print(f"Loading data from:")
print(f"  {hyg_basket_file}")
print(f"  {hyg_symbology_file}")
print()

hyg_basket_composition = pd.read_excel(hyg_basket_file)
hyg_corp_symbology = pd.read_excel(hyg_symbology_file)

print(f"HYG Basket Composition Shape: {hyg_basket_composition.shape}")
print(f"HYG Corp Symbology Shape:     {hyg_corp_symbology.shape}")
print()

# Merge datasets
bond_combined = pd.merge(
    hyg_corp_symbology,
    hyg_basket_composition[['isin', 'bidYield', 'askYield', 'midYield',
                            'face_notional', 'face_notional_weight']],
    on=['isin']
).reset_index(drop=True)

print(f"Combined Dataset Shape:       {bond_combined.shape}")
print()

# Number of bonds in HYG basket
num_bonds = len(bond_combined)
print(f"Number of corporate bonds in HYG basket: {num_bonds}")
print()

# Bond face notional statistics
bond_stats = bond_combined['face_notional'].describe()
print("Bond Face Notional Statistics:")
print(f"  Mean:   ${bond_stats['mean']:,.2f}")
print(f"  Median: ${bond_stats['50%']:,.2f}")
print(f"  Std:    ${bond_stats['std']:,.2f}")
print(f"  Min:    ${bond_stats['min']:,.2f}")
print(f"  Max:    ${bond_stats['max']:,.2f}")
print()

# Ticker statistics
ticker_stats = bond_combined.groupby('ticker')['face_notional'].sum().describe()
num_tickers = len(bond_combined['ticker'].unique())
print(f"Number of unique tickers in HYG basket: {num_tickers}")
print()
print("Ticker Face Notional Statistics:")
print(f"  Mean:   ${ticker_stats['mean']:,.2f}")
print(f"  Median: ${ticker_stats['50%']:,.2f}")
print(f"  Std:    ${ticker_stats['std']:,.2f}")
print(f"  Min:    ${ticker_stats['min']:,.2f}")
print(f"  Max:    ${ticker_stats['max']:,.2f}")
print()

# Yield-to-maturity statistics
ytm_stats = bond_combined['midYield'].describe()
print("Yield-to-Maturity Statistics (%):")
print(f"  Mean:   {ytm_stats['mean']:.4f}%")
print(f"  Median: {ytm_stats['50%']:.4f}%")
print(f"  Std:    {ytm_stats['std']:.4f}%")
print(f"  Min:    {ytm_stats['min']:.4f}%")
print(f"  Max:    {ytm_stats['max']:.4f}%")
print()

# Part b: Compute NAV and intrinsic ETF price
print("-" * 80)
print("Part b: Compute NAV of HYG Basket and Intrinsic Price of One ETF Share")
print("-" * 80)
print()

print("Creating bond objects and computing dirty prices from yields...")

# Create bonds and calculate dirty prices
dirty_prices = []
for index, row in bond_combined.iterrows():
    try:
        bond_object = create_bond_from_symbology(row)
        bond_yield = row['midYield'] / 100.0
        bond_dirty_price = bond_object.dirtyPrice(
            bond_yield,
            ql.Thirty360(ql.Thirty360.USA),
            ql.Compounded,
            ql.Semiannual
        )
        dirty_prices.append(bond_dirty_price)
    except Exception as e:
        print(f"Warning: Could not price bond {row['isin']}: {e}")
        dirty_prices.append(np.nan)

bond_combined['dirtyPrice'] = dirty_prices

# Calculate NAV contribution for each bond
bond_combined['NAV'] = bond_combined['dirtyPrice'] * bond_combined['face_notional_weight']

# Total ETF NAV (on $100 face)
ETF_intrinsic_nav = bond_combined['NAV'].sum() / 100

print(f"ETF Intrinsic NAV (per $100 face): ${ETF_intrinsic_nav:.4f}")
print()

# Total face notional and market cap
total_face_notional = bond_combined['face_notional'].sum()
ETF_marketCap = ETF_intrinsic_nav * total_face_notional / 100

print(f"Total Face Notional:          ${total_face_notional:,.0f}")
print(f"ETF Intrinsic Market Cap:     ${ETF_marketCap:,.0f}")
print()

# Number of shares outstanding
num_shares_outstanding = 188_700_000
ETF_intrinsic_pricePerShare = ETF_marketCap / num_shares_outstanding

print(f"Number of Shares Outstanding: {num_shares_outstanding:,}")
print(f"ETF Intrinsic Price per Share: ${ETF_intrinsic_pricePerShare:.4f}")
print()
print(f"Reference: HYG ETF market price as of 2024-04-26 was around $76.59")
print(f"Difference: ${ETF_intrinsic_pricePerShare - 76.59:.4f}")
print()

# Part c: Compute ETF yield using ACF method
print("-" * 80)
print("Part c: Compute ETF Yield using ACF (Aggregated Cash-Flows) Method")
print("-" * 80)
print()

def calc_ETF_NAV_from_yield(etf_yield, bond_df):
    """Calculate ETF NAV for a given flat yield"""
    dirty_prices_temp = []

    for index, row in bond_df.iterrows():
        try:
            bond_object = create_bond_from_symbology(row)
            bond_dirty_price = bond_object.dirtyPrice(
                etf_yield,
                ql.Thirty360(ql.Thirty360.USA),
                ql.Compounded,
                ql.Semiannual
            )
            dirty_prices_temp.append(bond_dirty_price)
        except:
            dirty_prices_temp.append(np.nan)

    # Calculate NAV
    nav_contributions = [p * w for p, w in zip(dirty_prices_temp,
                                                bond_df['face_notional_weight'])]
    return sum(nav_contributions) / 100

def target_function(etf_yield, bond_df, target_nav):
    """Target function for root finding"""
    return calc_ETF_NAV_from_yield(etf_yield, bond_df) - target_nav

print("Solving for ETF yield using root finding...")
print(f"Target NAV: ${ETF_intrinsic_nav:.4f}")
print()

# Use root_scalar to find the yield
result = root_scalar(
    target_function,
    args=(bond_combined, ETF_intrinsic_nav),
    bracket=[0.0, 1.0],
    method='brentq'
)

optimal_yield = result.root

print(f"ETF Yield (ACF method):       {optimal_yield:.4%}")
print(f"Reference: HYG ETF market yield as of 2024-04-26 was around 8.20%")
print(f"Difference: {(optimal_yield - 0.0820) * 10000:.2f} bps")
print()

# Part d: Compute ETF DV01, Duration and Convexity
print("-" * 80)
print("Part d: Compute ETF DV01, Duration and Convexity")
print("-" * 80)
print()

print("Computing sensitivities using +/- 1 bp yield scenarios...")

# Calculate prices for base and bumped scenarios
y = optimal_yield
y_bump = 1e-4  # 1 bp

price_base = ETF_intrinsic_nav
price_up_1bp = calc_ETF_NAV_from_yield(y + y_bump, bond_combined)
price_down_1bp = calc_ETF_NAV_from_yield(y - y_bump, bond_combined)

# Calculate sensitivities
dv01 = (price_down_1bp - price_base) / y_bump / 100
duration = dv01 / price_base * 100
convexity = (price_down_1bp - 2*price_base + price_up_1bp) / (y_bump**2) / price_base

print(f"Price (base):                 ${price_base:.6f}")
print(f"Price (yield + 1bp):          ${price_up_1bp:.6f}")
print(f"Price (yield - 1bp):          ${price_down_1bp:.6f}")
print()
print(f"ETF DV01:                     {dv01:.4f}")
print(f"ETF Duration:                 {duration:.4f} years")
print(f"ETF Convexity:                {convexity:.4f}")
print()
print(f"Reference values from HYG YAS screen as of 2024-04-26:")
print(f"  DV01:      3.57")
print(f"  Duration:  3.72 years")
print(f"  Convexity: 187")
print()
print(f"Differences:")
print(f"  DV01:      {dv01 - 3.57:.4f}")
print(f"  Duration:  {duration - 3.72:.4f} years")
print(f"  Convexity: {convexity - 187:.4f}")
print()

# Create summary table
print("-" * 80)
print("SUMMARY: HYG ETF Analysis Results")
print("-" * 80)
print()

summary_df = pd.DataFrame({
    'Metric': ['Number of Bonds', 'Number of Tickers', 'Total Face Notional',
               'ETF NAV (per $100)', 'ETF Price per Share', 'ETF Yield',
               'ETF DV01', 'ETF Duration', 'ETF Convexity'],
    'Value': [f'{num_bonds}', f'{num_tickers}', f'${total_face_notional:,.0f}',
              f'${ETF_intrinsic_nav:.4f}', f'${ETF_intrinsic_pricePerShare:.4f}',
              f'{optimal_yield:.4%}', f'{dv01:.4f}', f'{duration:.4f}', f'{convexity:.4f}'],
    'Reference': ['N/A', 'N/A', 'N/A', 'N/A', '$76.59', '8.20%',
                  '3.57', '3.72', '187']
})

print(summary_df.to_string(index=False))
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("="*80)
print("SOLUTION COMPLETE")
print("="*80)
print()
print("All problems have been solved successfully!")
print()
print("Generated Output Files:")
print(f"  1. {output_file1}")
print(f"  2. {output_file2}")
print(f"  3. {output_file3}")
print()
print("Summary of Results:")
print()
print("Problem 1 & 2: True/False questions answered with detailed explanations")
print()
print("Problem 3: Merton Model Analysis")
print(f"  - Fair Value of Equity:       ${E0:,.2f}")
print(f"  - Fair Value of Liabilities:  ${B0:,.2f}")
print(f"  - Credit Spread:              {credit_spread:.4%} ({credit_spread * 10000:.2f} bps)")
print(f"  - Equity Volatility:          {sigma_E:.4f} ({sigma_E:.2%})")
print()
print("Problem 4: HYG ETF Analysis")
print(f"  - Number of Bonds:            {num_bonds}")
print(f"  - ETF Price per Share:        ${ETF_intrinsic_pricePerShare:.4f}")
print(f"  - ETF Yield:                  {optimal_yield:.4%}")
print(f"  - ETF DV01:                   {dv01:.4f}")
print(f"  - ETF Duration:               {duration:.4f} years")
print(f"  - ETF Convexity:              {convexity:.4f}")
print()
print("="*80)
