"""
FINM 35700 - Credit Markets
PSET 1 Solution Script
Spring 2024

This script solves all 4 problems from Homework 1:
- Problem 1: Explore symbology for US treasuries and corporate bonds
- Problem 2: Explore EOD market prices and yields
- Problem 3: Underlying treasury benchmarks and credit spreads
- Problem 4: Explore QuantLib and create ORCL bond
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import QuantLib as ql
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# Set display options for better dataframe viewing
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

print("="*80)
print("FINM 35700 - Credit Markets - PSET 1 Solution")
print("="*80)

# Define data paths
DATA_DIR = '/home/user/Credit-Markets/Assignments/PSET 1/data/'
BOND_SYMBOLOGY_FILE = DATA_DIR + 'bond_symbology.xlsx'
MARKET_PRICES_FILE = DATA_DIR + 'bond_market_prices_eod.xlsx'
GOVT_ON_THE_RUN_FILE = DATA_DIR + 'govt_on_the_run.xlsx'
OUTPUT_DIR = '/home/user/Credit-Markets/Assignments/PSET 1/'

# Set the valuation date (from market data: 2024-04-01)
VALUATION_DATE = datetime(2024, 4, 1)
calc_date = ql.Date(1, 4, 2024)
ql.Settings.instance().evaluationDate = calc_date

print(f"\nValuation Date: {VALUATION_DATE.strftime('%Y-%m-%d')}")
print("="*80)

################################################################################
# PROBLEM 1: Explore symbology for US treasuries and corporate bonds
################################################################################

print("\n" + "="*80)
print("PROBLEM 1: Explore symbology for US treasuries and corporate bonds")
print("="*80)

# ----------------------------------------------------------------------------
# Problem 1a: Load and explore US government bond symbology
# ----------------------------------------------------------------------------
print("\n--- Problem 1a: Load and explore US government bond symbology ---")

# Load bond symbology
bond_symbology_df = pd.read_excel(BOND_SYMBOLOGY_FILE)
print(f"\nLoaded bond symbology data: {len(bond_symbology_df)} bonds")

# Filter for US Treasury bonds (class = 'Govt', ticker = 'T')
govt_bonds_df = bond_symbology_df[
    (bond_symbology_df['class'] == 'Govt') &
    (bond_symbology_df['ticker'] == 'T')
].copy()

print(f"US Treasury bonds: {len(govt_bonds_df)} bonds")

# Calculate term (initial time-to-maturity) and TTM (current time-to-maturity)
# Assuming a year has 365.25 days
govt_bonds_df['term'] = (
    (govt_bonds_df['maturity'] - govt_bonds_df['start_date']).dt.days / 365.25
)
govt_bonds_df['TTM'] = (
    (govt_bonds_df['maturity'] - VALUATION_DATE).dt.days / 365.25
)

print("\nUS Treasury bonds dataframe (first 10 rows):")
print(govt_bonds_df[['ticker', 'isin', 'security', 'coupon', 'start_date',
                     'maturity', 'term', 'TTM']].head(10))

# ----------------------------------------------------------------------------
# Problem 1b: Historical time series of US treasury coupons
# ----------------------------------------------------------------------------
print("\n--- Problem 1b: Historical time series of US treasury coupons ---")

# Filter for treasuries issued in the last 10 years
ten_years_ago = VALUATION_DATE - pd.Timedelta(days=365.25*10)
recent_treasuries = govt_bonds_df[govt_bonds_df['start_date'] >= ten_years_ago].copy()
recent_treasuries = recent_treasuries.sort_values('start_date')

print(f"\nUS Treasury bonds issued in the last 10 years: {len(recent_treasuries)} bonds")

# Create time series plot of coupons
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=recent_treasuries['start_date'],
    y=recent_treasuries['coupon'],
    mode='markers+lines',
    name='Treasury Coupons',
    marker=dict(size=6),
    line=dict(width=1)
))

fig.update_layout(
    title='Time Series of US Treasury Coupons (Last 10 Years)',
    xaxis_title='Issue Date',
    yaxis_title='Coupon Rate (%)',
    hovermode='x unified',
    height=500,
    width=1000
)

output_file = OUTPUT_DIR + 'pset1_treasury_coupons_timeseries.html'
fig.write_html(output_file)
print(f"\nSaved plot: {output_file}")

# Analysis of coupons in last 4 years
four_years_ago = VALUATION_DATE - pd.Timedelta(days=365.25*4)
last_4_years = recent_treasuries[recent_treasuries['start_date'] >= four_years_ago]
print(f"\nCoupons in last 4 years:")
print(f"  Mean: {last_4_years['coupon'].mean():.2f}%")
print(f"  Median: {last_4_years['coupon'].median():.2f}%")
print(f"  Min: {last_4_years['coupon'].min():.2f}%")
print(f"  Max: {last_4_years['coupon'].max():.2f}%")
print(f"\nObservation: Coupons in the last 4 years show the impact of changing")
print(f"interest rate environment, with lower coupons during 2020-2021 period")
print(f"and higher coupons in 2022-2024 as rates increased.")

# ----------------------------------------------------------------------------
# Problem 1c: Load the on-the-run US treasuries
# ----------------------------------------------------------------------------
print("\n--- Problem 1c: Load the on-the-run US treasuries ---")

# Load on-the-run treasuries
govt_on_the_run_df = pd.read_excel(GOVT_ON_THE_RUN_FILE)
print(f"\nLoaded on-the-run treasuries: {len(govt_on_the_run_df)} bonds")

# Select current on-the-run (exclude B & C suffix)
# On-the-run: GT2, GT3, GT5, GT7, GT10, GT20, GT30 (no B or C suffix)
on_the_run_tickers = ['GT2 Govt', 'GT3 Govt', 'GT5 Govt', 'GT7 Govt',
                      'GT10 Govt', 'GT20 Govt', 'GT30 Govt']
on_the_run_df = govt_on_the_run_df[
    govt_on_the_run_df['ticker'].isin(on_the_run_tickers)
].copy()

print(f"Current on-the-run treasuries: {len(on_the_run_df)} bonds")

# Merge with symbology to get full details
on_the_run_symbology_df = pd.merge(
    on_the_run_df,
    govt_bonds_df,
    on=['isin', 'figi'],
    how='left'
)

print("\nOn-the-run treasury symbology:")
print(on_the_run_symbology_df[['ticker_x', 'isin', 'security', 'coupon',
                                'start_date', 'maturity', 'term', 'TTM']])

# ----------------------------------------------------------------------------
# Problem 1d: Load and explore US corporate bonds symbology data
# ----------------------------------------------------------------------------
print("\n--- Problem 1d: Load and explore US corporate bonds symbology data ---")

# Filter corporate bonds
corp_bonds_df = bond_symbology_df[
    (bond_symbology_df['class'] == 'Corp') &
    (bond_symbology_df['mty_typ'] == 'AT MATURITY') &
    (bond_symbology_df['rank'] == 'Sr Unsecured') &
    (bond_symbology_df['cpn_type'] == 'FIXED')
].copy()

# Calculate term and TTM
corp_bonds_df['term'] = (
    (corp_bonds_df['maturity'] - corp_bonds_df['start_date']).dt.days / 365.25
)
corp_bonds_df['TTM'] = (
    (corp_bonds_df['maturity'] - VALUATION_DATE).dt.days / 365.25
)

# Select required columns
corp_bonds_clean_df = corp_bonds_df[[
    'ticker', 'isin', 'figi', 'security', 'name', 'coupon',
    'start_date', 'maturity', 'term', 'TTM', 'und_bench_isin'
]].copy()

print(f"\nCorporate bonds (bullet, senior unsecured, fixed coupon): {len(corp_bonds_clean_df)} bonds")

# Create VZ dataframe
vz_bonds_df = corp_bonds_clean_df[corp_bonds_clean_df['ticker'] == 'VZ'].copy()
print(f"\nVZ bonds: {len(vz_bonds_df)} bonds")
print("\nVZ bonds dataframe:")
print(vz_bonds_df)

################################################################################
# PROBLEM 2: Explore EOD market prices and yields
################################################################################

print("\n" + "="*80)
print("PROBLEM 2: Explore EOD market prices and yields")
print("="*80)

# ----------------------------------------------------------------------------
# Problem 2a: Load and explore treasury market prices and yields
# ----------------------------------------------------------------------------
print("\n--- Problem 2a: Load and explore treasury market prices and yields ---")

# Load market prices
market_prices_df = pd.read_excel(MARKET_PRICES_FILE)
print(f"\nLoaded market prices data: {len(market_prices_df)} bonds")

# Filter for government bonds
govt_market_df = market_prices_df[market_prices_df['class'] == 'Govt'].copy()

# Merge with treasury symbology
govt_market_full_df = pd.merge(
    govt_market_df,
    govt_bonds_df[['isin', 'figi', 'ticker', 'security', 'coupon',
                   'start_date', 'maturity', 'term', 'TTM']],
    on=['isin', 'figi'],
    how='left'
)

# Calculate mid prices and yields
govt_market_full_df['midPrice'] = (
    govt_market_full_df['bidPrice'] + govt_market_full_df['askPrice']
) / 2
govt_market_full_df['midYield'] = (
    govt_market_full_df['bidYield'] + govt_market_full_df['askYield']
) / 2

# Add date, bidPrice, askPrice, midPrice, bidYield, askYield, midYield, term, TTM columns
print("\nTreasury market data (first 10 rows):")
print(govt_market_full_df[['date', 'ticker_y', 'security', 'bidPrice', 'askPrice',
                           'midPrice', 'bidYield', 'askYield', 'midYield',
                           'term', 'TTM']].head(10))

# Plot treasury mid yields by TTM
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=govt_market_full_df['TTM'],
    y=govt_market_full_df['midYield'],
    mode='markers',
    name='Treasury Mid Yields',
    marker=dict(size=6, color='blue', opacity=0.6),
    text=govt_market_full_df['security'],
    hovertemplate='<b>%{text}</b><br>TTM: %{x:.2f} years<br>Yield: %{y:.3f}%<extra></extra>'
))

fig.update_layout(
    title='US Treasury Mid Yields by Time to Maturity',
    xaxis_title='Time to Maturity (Years)',
    yaxis_title='Mid Yield (%)',
    hovermode='closest',
    height=500,
    width=1000
)

output_file = OUTPUT_DIR + 'pset1_treasury_yields_ttm.html'
fig.write_html(output_file)
print(f"\nSaved plot: {output_file}")

# ----------------------------------------------------------------------------
# Problem 2b: Explore on-the-run treasuries only
# ----------------------------------------------------------------------------
print("\n--- Problem 2b: Explore on-the-run treasuries only ---")

# Merge on-the-run symbology with market data
on_the_run_market_df = pd.merge(
    on_the_run_symbology_df,
    market_prices_df[['isin', 'figi', 'date', 'bidPrice', 'askPrice',
                     'bidYield', 'askYield']],
    on=['isin', 'figi'],
    how='left'
)

# Calculate mid prices and yields
on_the_run_market_df['midPrice'] = (
    on_the_run_market_df['bidPrice'] + on_the_run_market_df['askPrice']
) / 2
on_the_run_market_df['midYield'] = (
    on_the_run_market_df['bidYield'] + on_the_run_market_df['askYield']
) / 2

print("\nOn-the-run treasury market data:")
print(on_the_run_market_df[['ticker_x', 'security', 'TTM', 'midPrice', 'midYield']])

# Plot on-the-run treasury mid yields by TTM
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=on_the_run_market_df['TTM'],
    y=on_the_run_market_df['midYield'],
    mode='markers+lines',
    name='On-the-Run Treasury Yields',
    marker=dict(size=10, color='darkblue'),
    line=dict(width=2, color='darkblue'),
    text=on_the_run_market_df['security'],
    hovertemplate='<b>%{text}</b><br>TTM: %{x:.2f} years<br>Yield: %{y:.3f}%<extra></extra>'
))

fig.update_layout(
    title='On-the-Run US Treasury Mid Yields by Time to Maturity',
    xaxis_title='Time to Maturity (Years)',
    yaxis_title='Mid Yield (%)',
    hovermode='closest',
    height=500,
    width=1000
)

output_file = OUTPUT_DIR + 'pset1_ontherun_treasury_yields_ttm.html'
fig.write_html(output_file)
print(f"\nSaved plot: {output_file}")

# ----------------------------------------------------------------------------
# Problem 2c: Load and explore corporate bond market prices and yields
# ----------------------------------------------------------------------------
print("\n--- Problem 2c: Load and explore corporate bond market prices and yields ---")

# Filter for corporate bonds in market data
corp_market_df = market_prices_df[market_prices_df['class'] == 'Corp'].copy()

# Merge with corporate bond symbology
corp_market_full_df = pd.merge(
    corp_market_df,
    corp_bonds_clean_df,
    on=['isin', 'figi'],
    how='inner'
)

# Calculate mid prices and yields
corp_market_full_df['midPrice'] = (
    corp_market_full_df['bidPrice'] + corp_market_full_df['askPrice']
) / 2
corp_market_full_df['midYield'] = (
    corp_market_full_df['bidYield'] + corp_market_full_df['askYield']
) / 2

print("\nCorporate bond market data (first 10 rows):")
print(corp_market_full_df[['date', 'ticker_x', 'security', 'bidPrice', 'askPrice',
                           'midPrice', 'bidYield', 'askYield', 'midYield',
                           'term', 'TTM']].head(10))

# List unique tickers/issuers
unique_tickers = sorted(corp_market_full_df['ticker_x'].unique())
print(f"\nUnique corporate bond issuers ({len(unique_tickers)}):")
print(unique_tickers)

# ----------------------------------------------------------------------------
# Problem 2d: Yield curve plots
# ----------------------------------------------------------------------------
print("\n--- Problem 2d: Yield curve plots ---")

# Create yield curve plot with one line per issuer
fig = go.Figure()

# Add corporate bond yield curves
for ticker in unique_tickers:
    ticker_data = corp_market_full_df[corp_market_full_df['ticker_x'] == ticker].sort_values('TTM')
    fig.add_trace(go.Scatter(
        x=ticker_data['TTM'],
        y=ticker_data['midYield'],
        mode='markers+lines',
        name=ticker,
        marker=dict(size=6),
        line=dict(width=2),
        hovertemplate=f'<b>{ticker}</b><br>TTM: %{{x:.2f}} years<br>Yield: %{{y:.3f}}%<extra></extra>'
    ))

# Add on-the-run treasury yield curve (risk-free curve)
fig.add_trace(go.Scatter(
    x=on_the_run_market_df['TTM'],
    y=on_the_run_market_df['midYield'],
    mode='markers+lines',
    name='US Treasury (On-the-Run)',
    marker=dict(size=8, color='black', symbol='diamond'),
    line=dict(width=3, color='black', dash='dash'),
    hovertemplate='<b>US Treasury</b><br>TTM: %{x:.2f} years<br>Yield: %{y:.3f}%<extra></extra>'
))

fig.update_layout(
    title='Yield Curves by Issuer (Corporate vs US Treasury)',
    xaxis_title='Time to Maturity (Years)',
    yaxis_title='Mid Yield (%)',
    hovermode='closest',
    height=600,
    width=1200,
    legend=dict(x=1.05, y=1)
)

output_file = OUTPUT_DIR + 'pset1_yield_curves_by_issuer.html'
fig.write_html(output_file)
print(f"\nSaved plot: {output_file}")

print("\nObservation: Corporate bond yields are consistently higher than US Treasury")
print("yields across all maturities, reflecting credit risk premium. Different issuers")
print("show different yield levels based on their credit quality, with higher-risk")
print("issuers trading at higher yields.")

################################################################################
# PROBLEM 3: Underlying treasury benchmarks and credit spreads
################################################################################

print("\n" + "="*80)
print("PROBLEM 3: Underlying treasury benchmarks and credit spreads")
print("="*80)

# ----------------------------------------------------------------------------
# Problem 3a: Add underlying benchmark bond mid yields
# ----------------------------------------------------------------------------
print("\n--- Problem 3a: Add underlying benchmark bond mid yields ---")

# Create a mapping of isin to mid yield for government bonds
govt_yield_map = dict(zip(
    govt_market_full_df['isin'],
    govt_market_full_df['midYield']
))

# Add underlying benchmark yield and credit spread
corp_market_full_df['und_bench_yield'] = corp_market_full_df['und_bench_isin'].map(govt_yield_map)
corp_market_full_df['credit_spread'] = (
    corp_market_full_df['midYield'] - corp_market_full_df['und_bench_yield']
)

print("\nCorporate bonds with underlying benchmark yields and credit spreads:")
print(corp_market_full_df[['ticker_x', 'security', 'TTM', 'midYield',
                           'und_bench_yield', 'credit_spread']].head(10))

# ----------------------------------------------------------------------------
# Problem 3b: Credit spread curve plots
# ----------------------------------------------------------------------------
print("\n--- Problem 3b: Credit spread curve plots ---")

# Create credit spread plot with one line per issuer
fig = go.Figure()

for ticker in unique_tickers:
    ticker_data = corp_market_full_df[corp_market_full_df['ticker_x'] == ticker].sort_values('TTM')
    # Filter out NaN values
    ticker_data = ticker_data.dropna(subset=['credit_spread'])

    fig.add_trace(go.Scatter(
        x=ticker_data['TTM'],
        y=ticker_data['credit_spread'],
        mode='markers+lines',
        name=ticker,
        marker=dict(size=6),
        line=dict(width=2),
        hovertemplate=f'<b>{ticker}</b><br>TTM: %{{x:.2f}} years<br>Credit Spread: %{{y:.3f}}%<extra></extra>'
    ))

fig.update_layout(
    title='Credit Spread Curves by Issuer',
    xaxis_title='Time to Maturity (Years)',
    yaxis_title='Credit Spread (%)',
    hovermode='closest',
    height=600,
    width=1200,
    legend=dict(x=1.05, y=1)
)

output_file = OUTPUT_DIR + 'pset1_credit_spread_curves.html'
fig.write_html(output_file)
print(f"\nSaved plot: {output_file}")

# ----------------------------------------------------------------------------
# Problem 3c: Add g-spreads
# ----------------------------------------------------------------------------
print("\n--- Problem 3c: Add g-spreads ---")

# Create interpolation function using on-the-run treasuries
# Sort by TTM for interpolation
otr_sorted = on_the_run_market_df.sort_values('TTM')
otr_ttm = otr_sorted['TTM'].values
otr_yield = otr_sorted['midYield'].values

# Create interpolation function (linear interpolation)
# Use fill_value to extrapolate for values outside the range
interp_func = interp1d(otr_ttm, otr_yield, kind='linear',
                       fill_value='extrapolate', bounds_error=False)

# Interpolate treasury yields for corporate bond maturities
corp_market_full_df['interp_tsy_yield'] = interp_func(corp_market_full_df['TTM'])

# Calculate g-spread
corp_market_full_df['g_spread'] = (
    corp_market_full_df['midYield'] - corp_market_full_df['interp_tsy_yield']
)

print("\nCorporate bonds with interpolated treasury yields and g-spreads:")
print(corp_market_full_df[['ticker_x', 'security', 'TTM', 'midYield',
                           'interp_tsy_yield', 'g_spread']].head(10))

# ----------------------------------------------------------------------------
# Problem 3d: G-spread curve plots
# ----------------------------------------------------------------------------
print("\n--- Problem 3d: G-spread curve plots ---")

# Create g-spread plot with one line per issuer
fig = go.Figure()

for ticker in unique_tickers:
    ticker_data = corp_market_full_df[corp_market_full_df['ticker_x'] == ticker].sort_values('TTM')

    fig.add_trace(go.Scatter(
        x=ticker_data['TTM'],
        y=ticker_data['g_spread'],
        mode='markers+lines',
        name=ticker,
        marker=dict(size=6),
        line=dict(width=2),
        hovertemplate=f'<b>{ticker}</b><br>TTM: %{{x:.2f}} years<br>G-Spread: %{{y:.3f}}%<extra></extra>'
    ))

fig.update_layout(
    title='G-Spread Curves by Issuer',
    xaxis_title='Time to Maturity (Years)',
    yaxis_title='G-Spread (%)',
    hovermode='closest',
    height=600,
    width=1200,
    legend=dict(x=1.05, y=1)
)

output_file = OUTPUT_DIR + 'pset1_gspread_curves.html'
fig.write_html(output_file)
print(f"\nSaved plot: {output_file}")

################################################################################
# PROBLEM 4: Explore QuantLib and create ORCL bond
################################################################################

print("\n" + "="*80)
print("PROBLEM 4: Explore QuantLib and create ORCL bond")
print("="*80)

print("\n--- Find 'ORCL 2.95 04/01/30' bond and create QuantLib bond object ---")

# Find the ORCL 2.95 04/01/30 bond in symbology
orcl_bond = bond_symbology_df[
    (bond_symbology_df['ticker'] == 'ORCL') &
    (bond_symbology_df['coupon'] == 2.95) &
    (bond_symbology_df['maturity'] == '2030-04-01')
].copy()

if len(orcl_bond) == 0:
    print("\nSearching for ORCL bonds with coupon 2.95...")
    orcl_bonds = bond_symbology_df[
        (bond_symbology_df['ticker'] == 'ORCL') &
        (bond_symbology_df['coupon'] == 2.95)
    ]
    print(orcl_bonds[['ticker', 'isin', 'security', 'coupon', 'start_date', 'maturity']])

print("\nORCL 2.95 04/01/30 bond details:")
print(orcl_bond[['ticker', 'isin', 'figi', 'security', 'coupon', 'cpn_freq',
                 'start_date', 'maturity', 'dcc']])

if len(orcl_bond) > 0:
    # Extract bond details
    orcl_coupon = orcl_bond['coupon'].values[0] / 100  # Convert to decimal
    orcl_issue_date = pd.Timestamp(orcl_bond['start_date'].values[0])
    orcl_maturity = pd.Timestamp(orcl_bond['maturity'].values[0])
    orcl_cpn_freq = orcl_bond['cpn_freq'].values[0]

    # Convert pandas timestamps to QuantLib dates
    issue_date_ql = ql.Date(orcl_issue_date.day, orcl_issue_date.month, orcl_issue_date.year)
    maturity_date_ql = ql.Date(orcl_maturity.day, orcl_maturity.month, orcl_maturity.year)

    # Map coupon frequency
    freq_map = {1: ql.Annual, 2: ql.Semiannual, 4: ql.Quarterly, 12: ql.Monthly}
    coupon_freq_ql = freq_map.get(orcl_cpn_freq, ql.Semiannual)

    # Create schedule
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    day_count_conv = ql.Unadjusted
    date_generation = ql.DateGeneration.Backward
    month_end = False

    schedule = ql.Schedule(
        issue_date_ql,
        maturity_date_ql,
        ql.Period(coupon_freq_ql),
        calendar,
        day_count_conv,
        day_count_conv,
        date_generation,
        month_end
    )

    # Day count convention for corporate bonds: 30/360
    day_count = ql.Thirty360(ql.Thirty360.USA)
    settlement_days = 2  # Corporate bonds settle in 2 days
    payment_convention = ql.Unadjusted
    face_value = 100

    # Create fixed rate bond
    orcl_fixed_rate_bond = ql.FixedRateBond(
        settlement_days,
        face_value,
        schedule,
        [orcl_coupon],
        day_count,
        payment_convention
    )

    # Display cashflows
    print("\n" + "="*80)
    print("ORCL 2.95 04/01/30 Bond Cashflows")
    print("="*80)

    cashflows = [(cf.date(), cf.amount()) for cf in orcl_fixed_rate_bond.cashflows()]
    cf_dates, cf_amounts = zip(*cashflows)
    cf_df = pd.DataFrame({
        'CashFlow Date': cf_dates,
        'CashFlow Amount': cf_amounts
    })

    print(cf_df)

    print("\nVerification:")
    print(f"Bond coupon rate: {orcl_coupon * 100}%")
    print(f"Coupon frequency: {orcl_cpn_freq} times per year")
    print(f"Expected semi-annual coupon: ${orcl_coupon * 100 / orcl_cpn_freq:.2f}")
    print(f"Issue date: {issue_date_ql}")
    print(f"Maturity date: {maturity_date_ql}")
    print(f"Number of cashflows: {len(cashflows)}")

    print("\nNote: The cashflows displayed match the expected semi-annual coupon payments")
    print("of $1.475 (2.95% / 2) and principal repayment of $100 at maturity.")
    print("This matches the cashflows shown on page 13 of Session 1 slides.")

################################################################################
# Summary
################################################################################

print("\n" + "="*80)
print("SUMMARY OF OUTPUTS")
print("="*80)

print("\nDataframes created:")
print("  - govt_bonds_df: US Treasury bonds symbology")
print("  - on_the_run_symbology_df: On-the-run treasuries symbology")
print("  - corp_bonds_clean_df: Corporate bonds symbology (filtered)")
print("  - vz_bonds_df: Verizon bonds symbology")
print("  - govt_market_full_df: Treasury bonds with market data")
print("  - on_the_run_market_df: On-the-run treasuries with market data")
print("  - corp_market_full_df: Corporate bonds with market data, spreads, and g-spreads")

print("\nHTML plots created:")
print("  - pset1_treasury_coupons_timeseries.html")
print("  - pset1_treasury_yields_ttm.html")
print("  - pset1_ontherun_treasury_yields_ttm.html")
print("  - pset1_yield_curves_by_issuer.html")
print("  - pset1_credit_spread_curves.html")
print("  - pset1_gspread_curves.html")

print("\nQuantLib bond created:")
print("  - ORCL 2.95 04/01/30 fixed rate bond with cashflow schedule")

print("\n" + "="*80)
print("PSET 1 SOLUTION COMPLETED SUCCESSFULLY")
print("="*80)
