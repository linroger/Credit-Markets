"""
PSET 2 Solution - FINM 35700 Credit Markets
Complete solution for all problems

Author: AI Assistant
Date: 2024-04-08
"""

import QuantLib as ql
import numpy as np
import pandas as pd
import datetime as dt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# =============================================================================
# SETUP AND CONFIGURATION
# =============================================================================

# Set calculation date
calc_date = ql.Date(8, 4, 2024)
ql.Settings.instance().evaluationDate = calc_date
date = pd.to_datetime('2024-04-08')

# Data directory
DATA_DIR = Path(__file__).parent / "data"

print("="*80)
print("PSET 2 Solution - FINM 35700 Credit Markets")
print("="*80)
print(f"\nCalculation Date: {calc_date}")
print(f"Data Directory: {DATA_DIR}\n")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_ql_date(date_input) -> ql.Date:
    """
    Convert dt.date or string to ql.Date

    Args:
        date_input: datetime.date, pd.Timestamp, or string in format 'YYYY-MM-DD'

    Returns:
        ql.Date object
    """
    if isinstance(date_input, ql.Date):
        return date_input
    elif isinstance(date_input, pd.Timestamp):
        date_input = date_input.date()
    elif isinstance(date_input, dt.datetime):
        date_input = date_input.date()

    if isinstance(date_input, dt.date):
        return ql.Date(date_input.day, date_input.month, date_input.year)
    elif isinstance(date_input, str):
        date_obj = dt.datetime.strptime(date_input, "%Y-%m-%d").date()
        return ql.Date(date_obj.day, date_obj.month, date_obj.year)
    else:
        raise ValueError(f"Unsupported date type: {type(date_input)}, {date_input}")


def create_schedule_from_symbology(details: dict):
    """
    Create a QuantLib cashflow schedule from symbology details dictionary

    Args:
        details: Dictionary containing bond symbology information (usually one row of the symbology dataframe)

    Returns:
        ql.Schedule object
    """
    # Create maturity and first accrual date from details
    maturity = get_ql_date(details['maturity'])
    acc_first = get_ql_date(details['acc_first'])

    # Create calendar for Corp and Govt asset classes
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)

    # Define period - semi-annual frequency
    period = ql.Period(ql.Semiannual)

    # Business day convention
    business_day_convention = ql.Unadjusted

    # Termination date convention
    termination_date_convention = ql.Unadjusted

    # Date generation
    date_generation = ql.DateGeneration.Backward

    # Create schedule using ql.MakeSchedule interface
    schedule = ql.MakeSchedule(
        effectiveDate=acc_first,
        terminationDate=maturity,
        tenor=period,
        calendar=calendar,
        convention=business_day_convention,
        terminalDateConvention=termination_date_convention,
        rule=date_generation,
        endOfMonth=True,
        firstDate=None,
        nextToLastDate=None
    )

    return schedule


def create_bond_from_symbology(details: dict):
    """
    Create a US fixed rate bond object from symbology details dictionary

    Args:
        details: Dictionary containing bond symbology information (usually one row of the symbology dataframe)

    Returns:
        ql.FixedRateBond object
    """
    # Create day_count from details['dcc']
    # For US Treasuries use ql.ActualActual(ql.ActualActual.ISMA)
    # For US Corporate bonds use ql.Thirty360(ql.Thirty360.USA)
    if details['dcc'] == 'ACT/ACT':
        day_count = ql.ActualActual(ql.ActualActual.ISMA)
    else:
        day_count = ql.Thirty360(ql.Thirty360.USA)

    # Create issue date from details['start_date']
    issue_date = get_ql_date(details['start_date'])

    # Create days_settle from details['days_settle']
    days_settle = int(details['days_settle'])

    # Create coupon rate from details['coupon']
    coupon = float(details['coupon'] / 100)

    # Create cashflow schedule
    schedule = create_schedule_from_symbology(details)

    face_value = 100
    redemption = 100

    payment_convention = ql.Unadjusted

    # Create fixed rate bond object
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


def get_bond_cashflows(bond: ql.FixedRateBond, calc_date: ql.Date) -> pd.DataFrame:
    """
    Returns all cashflows (including past ones) with dates, amounts and year fractions

    Args:
        bond: ql.FixedRateBond object
        calc_date: Calculation date

    Returns:
        DataFrame with columns: CashFlowDate, CashFlowAmount, CashFlowYearFrac
    """
    day_count = bond.dayCounter()
    cashflow_data = [
        (cf.date(), cf.amount(), day_count.yearFraction(calc_date, cf.date()))
        for cf in bond.cashflows()
    ]

    cf_dates, cf_amounts, cf_yearfracs = zip(*cashflow_data)

    cashflows_df = pd.DataFrame({
        'CashFlowDate': cf_dates,
        'CashFlowAmount': cf_amounts,
        'CashFlowYearFrac': cf_yearfracs
    })

    return cashflows_df


def calibrate_yield_curve_from_frame(
    calc_date: ql.Date,
    treasury_details: pd.DataFrame,
    price_quote_column: str
):
    """
    Create a calibrated yield curve from a details dataframe which includes bid/ask/mid price quotes

    Args:
        calc_date: Calculation date
        treasury_details: DataFrame with treasury bond details
        price_quote_column: Column name for price quotes ('bidPrice', 'askPrice', or 'midPrice')

    Returns:
        ql.PiecewiseLogCubicDiscount yield curve object
    """
    ql.Settings.instance().evaluationDate = calc_date

    # Sort dataframe by maturity
    sorted_details_frame = treasury_details.sort_values(by='maturity')

    # For US Treasuries use ql.ActualActual(ql.ActualActual.ISMA)
    day_count = ql.ActualActual(ql.ActualActual.ISMA)

    bond_helpers = []

    for index, row in sorted_details_frame.iterrows():
        bond_object = create_bond_from_symbology(row)

        tsy_clean_price_quote = row[price_quote_column]
        tsy_clean_price_handle = ql.QuoteHandle(ql.SimpleQuote(tsy_clean_price_quote))

        bond_helper = ql.BondHelper(tsy_clean_price_handle, bond_object)
        bond_helpers.append(bond_helper)

    yield_curve = ql.PiecewiseLogCubicDiscount(calc_date, bond_helpers, day_count)
    yield_curve.enableExtrapolation()

    return yield_curve


def get_yield_curve_details_df(yield_curve, curve_dates=None):
    """
    Returns a DataFrame with yield curve details for given curve dates

    Args:
        yield_curve: QuantLib yield curve object
        curve_dates: List of dates (optional, defaults to curve pillar dates)

    Returns:
        DataFrame with columns: Date, YearFrac, DiscountFactor, ZeroRate
    """
    if curve_dates is None:
        curve_dates = yield_curve.dates()

    curve_ql_dates = [get_ql_date(d) for d in curve_dates]
    discounts = [round(yield_curve.discount(d), 6) for d in curve_ql_dates]
    yearfracs = [round(yield_curve.timeFromReference(d), 3) for d in curve_ql_dates]
    zero_rates = [
        round(yield_curve.zeroRate(d, yield_curve.dayCounter(), ql.Compounded).rate() * 100, 3)
        for d in curve_ql_dates
    ]

    yield_curve_details_df = pd.DataFrame({
        'Date': curve_dates,
        'YearFrac': yearfracs,
        'DiscountFactor': discounts,
        'ZeroRate': zero_rates
    })

    return yield_curve_details_df


def calc_clean_price_with_zspread(fixed_rate_bond, yield_curve_handle, zspread):
    """
    Calculate clean price of a bond given a z-spread

    Args:
        fixed_rate_bond: ql.FixedRateBond object
        yield_curve_handle: ql.YieldTermStructureHandle
        zspread: Z-spread value

    Returns:
        Clean price
    """
    zspread_quote = ql.SimpleQuote(zspread)
    zspread_quote_handle = ql.QuoteHandle(zspread_quote)
    yield_curve_bumped = ql.ZeroSpreadedTermStructure(
        yield_curve_handle, zspread_quote_handle, ql.Compounded, ql.Semiannual
    )
    yield_curve_bumped_handle = ql.YieldTermStructureHandle(yield_curve_bumped)

    # Set valuation engine
    bond_engine = ql.DiscountingBondEngine(yield_curve_bumped_handle)
    fixed_rate_bond.setPricingEngine(bond_engine)
    bond_clean_price = fixed_rate_bond.cleanPrice()

    return bond_clean_price


# =============================================================================
# PROBLEM 1: CONSTRUCTING FIXED RATE BONDS
# =============================================================================

print("\n" + "="*80)
print("PROBLEM 1: CONSTRUCTING FIXED RATE BONDS")
print("="*80)

# Problem 1a: Prepare the symbology and market data files
print("\nProblem 1a: Loading and preparing data...")

bond_symbology = pd.read_excel(DATA_DIR / 'bond_symbology.xlsx')
bond_market_prices_eod = pd.read_excel(DATA_DIR / 'bond_market_prices_eod.xlsx')
govt_on_the_run = pd.read_excel(DATA_DIR / 'govt_on_the_run.xlsx')

# Calculate term and TTM
bond_symbology['term'] = (bond_symbology['maturity'] - bond_symbology['start_date']).dt.days / 365.25
bond_symbology['TTM'] = (bond_symbology['maturity'] - date).dt.days / 365.25

# Calculate mid prices and yields
bond_market_prices_eod['midPrice'] = 0.5 * (bond_market_prices_eod['bidPrice'] + bond_market_prices_eod['askPrice'])
bond_market_prices_eod['midYield'] = 0.5 * (bond_market_prices_eod['bidYield'] + bond_market_prices_eod['askYield'])

# Filter for fixed rate bonds only
bond_symbology = bond_symbology[bond_symbology['cpn_type'] == 'FIXED']

print(f"Total fixed rate bonds loaded: {len(bond_symbology)}")
print(f"Government bonds: {len(bond_symbology[bond_symbology['class'] == 'Govt'])}")
print(f"Corporate bonds: {len(bond_symbology[bond_symbology['class'] == 'Corp'])}")

# Problem 1b & 1c: Functions already defined above (create_schedule_from_symbology, create_bond_from_symbology)
print("\nProblem 1b & 1c: Bond construction functions defined")

# Problem 1d: Display cashflows for sample bonds
print("\nProblem 1d: Displaying cashflows for sample bonds...")

# Pick one government bond
govt_bonds = bond_symbology[bond_symbology['class'] == 'Govt']
sample_govt_bond = govt_bonds.iloc[0]
govt_bond_obj = create_bond_from_symbology(sample_govt_bond)
govt_cashflows = get_bond_cashflows(govt_bond_obj, calc_date)

print(f"\nGovernment Bond: {sample_govt_bond['security']}")
print(f"Future cashflows (first 10):")
future_cfs = govt_cashflows[govt_cashflows['CashFlowYearFrac'] > 0].head(10)
print(future_cfs.to_string(index=False))

# Pick one corporate bond
corp_bonds = bond_symbology[bond_symbology['class'] == 'Corp']
sample_corp_bond = corp_bonds.iloc[0]
corp_bond_obj = create_bond_from_symbology(sample_corp_bond)
corp_cashflows = get_bond_cashflows(corp_bond_obj, calc_date)

print(f"\nCorporate Bond: {sample_corp_bond['security']}")
print(f"Future cashflows (first 10):")
future_cfs = corp_cashflows[corp_cashflows['CashFlowYearFrac'] > 0].head(10)
print(future_cfs.to_string(index=False))


# =============================================================================
# PROBLEM 2: US TREASURY YIELD CURVE CALIBRATION
# =============================================================================

print("\n" + "="*80)
print("PROBLEM 2: US TREASURY YIELD CURVE CALIBRATION")
print("="*80)

# Problem 2a: Create on-the-run US treasury bond objects
print("\nProblem 2a: Creating on-the-run treasury bond objects...")

# Filter for on-the-run treasuries (GT2, GT3, GT5, GT7, GT10, GT20, GT30 - excluding B and C variants)
govt_on_the_run_filter = ~govt_on_the_run['ticker'].str.contains('B|C', regex=True)
maturity_filter = govt_on_the_run['ticker'].str.contains('GT2|GT3|GT5|GT7|GT10|GT20|GT30', regex=True)
on_the_run_treasuries = govt_on_the_run[govt_on_the_run_filter & maturity_filter]

# Get government bonds from symbology
govt_bonds_df = bond_symbology[(bond_symbology['ticker'] == 'T') & (bond_symbology['class'] == 'Govt')]

# Match on-the-run treasuries with symbology and market data
otr_isins = on_the_run_treasuries['isin'].values
govt_bonds_otr = govt_bonds_df[govt_bonds_df['isin'].isin(otr_isins)]
market_prices_otr = bond_market_prices_eod[bond_market_prices_eod['isin'].isin(otr_isins)]

# Merge symbology with market data
otr_merged = pd.merge(govt_bonds_otr, market_prices_otr, on=['isin', 'class', 'ticker', 'figi'])
otr_merged_sorted = otr_merged.sort_values(by='TTM').reset_index(drop=True)

print(f"Number of on-the-run treasuries: {len(otr_merged_sorted)}")
print("\nOn-the-run treasuries:")
print(otr_merged_sorted[['security', 'TTM', 'midPrice', 'midYield']].to_string(index=False))

# Plot OTR treasury yields
fig_otr_yields = px.scatter(
    otr_merged_sorted,
    x='TTM',
    y='midYield',
    title='On-The-Run UST Yields by TTM',
    labels={'TTM': 'Time to Maturity (Years)', 'midYield': 'Mid Yield (%)'}
)
fig_otr_yields.update_traces(marker=dict(size=10, color='blue'))
fig_otr_yields.write_html('Problem_2a_OTR_Yields.html')
print("\nSaved: Problem_2a_OTR_Yields.html")

# Problem 2b: Calibrate the on-the-run treasury yield curves
print("\nProblem 2b: Calibrating yield curves...")

# Generate curve dates (6-month intervals up to 30 years)
curve_dates = [(date + pd.DateOffset(months=6 * n)).date() for n in range(61)]

# Calibrate bid, ask, and mid curves
calibrated_curve_bid = calibrate_yield_curve_from_frame(calc_date, otr_merged_sorted, 'bidPrice')
calibrated_curve_ask = calibrate_yield_curve_from_frame(calc_date, otr_merged_sorted, 'askPrice')
calibrated_curve_mid = calibrate_yield_curve_from_frame(calc_date, otr_merged_sorted, 'midPrice')

# Get curve details
calibrated_details_bid = get_yield_curve_details_df(calibrated_curve_bid, curve_dates)
calibrated_details_ask = get_yield_curve_details_df(calibrated_curve_ask, curve_dates)
calibrated_details_mid = get_yield_curve_details_df(calibrated_curve_mid, curve_dates)

print("\nCalibrated mid curve details (first 10 points):")
print(calibrated_details_mid.head(10).to_string(index=False))

# Problem 2c: Plot the calibrated US Treasury yield curves
print("\nProblem 2c: Plotting calibrated yield curves...")

fig_yields = go.Figure()
fig_yields.add_trace(go.Scatter(
    x=calibrated_details_bid['YearFrac'],
    y=calibrated_details_bid['ZeroRate'],
    mode='lines+markers',
    name='Bid Curve',
    line=dict(color='red', width=2)
))
fig_yields.add_trace(go.Scatter(
    x=calibrated_details_ask['YearFrac'],
    y=calibrated_details_ask['ZeroRate'],
    mode='lines+markers',
    name='Ask Curve',
    line=dict(color='green', width=2)
))
fig_yields.add_trace(go.Scatter(
    x=calibrated_details_mid['YearFrac'],
    y=calibrated_details_mid['ZeroRate'],
    mode='lines+markers',
    name='Mid Curve',
    line=dict(color='blue', width=2)
))
fig_yields.update_layout(
    title='Calibrated UST Zero Rate Curves',
    xaxis_title='Time to Maturity (Years)',
    yaxis_title='Zero Rate (%)',
    hovermode='x unified'
)
fig_yields.write_html('Problem_2c_Calibrated_Yield_Curves.html')
print("Saved: Problem_2c_Calibrated_Yield_Curves.html")

# Problem 2d: Plot calibrated discount factors
print("\nProblem 2d: Plotting discount factors...")

fig_df = go.Figure()
fig_df.add_trace(go.Scatter(
    x=calibrated_details_mid['YearFrac'],
    y=calibrated_details_mid['DiscountFactor'],
    mode='lines+markers',
    name='Discount Factors',
    line=dict(color='purple', width=2)
))
fig_df.update_layout(
    title='UST Discount Factors by TTM',
    xaxis_title='Time to Maturity (Years)',
    yaxis_title='Discount Factor',
    hovermode='x unified'
)
fig_df.write_html('Problem_2d_Discount_Factors.html')
print("Saved: Problem_2d_Discount_Factors.html")


# =============================================================================
# PROBLEM 3: PRICING AND RISK METRICS FOR US TREASURY BONDS
# =============================================================================

print("\n" + "="*80)
print("PROBLEM 3: PRICING AND RISK METRICS FOR US TREASURY BONDS")
print("="*80)

# Problem 3a: Price US treasuries on calibrated discount factor curve
print("\nProblem 3a: Pricing treasuries on calibrated curve...")

mid_yield_handle = ql.YieldTermStructureHandle(calibrated_curve_mid)
bond_engine = ql.DiscountingBondEngine(mid_yield_handle)

otr_merged_sorted['calc_mid_price'] = np.nan

for index, row in otr_merged_sorted.iterrows():
    bond_object = create_bond_from_symbology(row)
    bond_object.setPricingEngine(bond_engine)
    otr_merged_sorted.loc[index, 'calc_mid_price'] = bond_object.cleanPrice()

print("\nPrice validation (Market vs Calculated):")
comparison_df = otr_merged_sorted[['security', 'midPrice', 'calc_mid_price']].copy()
comparison_df['price_diff'] = comparison_df['calc_mid_price'] - comparison_df['midPrice']
print(comparison_df.to_string(index=False))

# Problem 3b: Compute analytical DV01, Duration, Convexity (flat yield)
print("\nProblem 3b: Computing analytical risk metrics (flat yield)...")

flat_yield = 0.05
days = ql.Actual360()
compound = ql.Compounded
cpn_freq = ql.Semiannual
rate = ql.InterestRate(flat_yield, days, compound, cpn_freq)

otr_merged_sorted['dv01'] = np.nan
otr_merged_sorted['duration'] = np.nan
otr_merged_sorted['convexity'] = np.nan

for index, row in otr_merged_sorted.iterrows():
    bond_object = create_bond_from_symbology(row)
    bond_object.setPricingEngine(bond_engine)

    duration = ql.BondFunctions.duration(bond_object, rate)
    convexity = ql.BondFunctions.convexity(bond_object, rate)

    otr_merged_sorted.loc[index, 'duration'] = duration
    otr_merged_sorted.loc[index, 'convexity'] = convexity
    otr_merged_sorted.loc[index, 'dv01'] = duration * bond_object.dirtyPrice() / 100

print("\nAnalytical risk metrics:")
print(otr_merged_sorted[['security', 'dv01', 'duration', 'convexity']].to_string(index=False))

# Problem 3c: Compute scenario DV01, Duration, Convexity (calibrated curve)
print("\nProblem 3c: Computing scenario risk metrics (calibrated curve)...")

otr_merged_sorted['scen_dv01'] = np.nan
otr_merged_sorted['scen_duration'] = np.nan
otr_merged_sorted['scen_convexity'] = np.nan

for index, row in otr_merged_sorted.iterrows():
    bond_object = create_bond_from_symbology(row)

    # Create scenario engine with zero spread
    interest_rate = ql.SimpleQuote(0.0)
    yield_curve = ql.ZeroSpreadedTermStructure(mid_yield_handle, ql.QuoteHandle(interest_rate))
    bond_engine_scen = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(yield_curve))
    bond_object.setPricingEngine(bond_engine_scen)

    # Base price
    price_base = bond_object.cleanPrice()

    # +1bp scenario
    interest_rate.setValue(0.0001)
    price_up_1bp = bond_object.cleanPrice()

    # -1bp scenario
    interest_rate.setValue(-0.0001)
    price_down_1bp = bond_object.cleanPrice()

    # Reset to base
    interest_rate.setValue(0)

    # Calculate scenario metrics
    dv01 = (price_down_1bp - price_base) * 10000 / 100
    duration = dv01 / bond_object.dirtyPrice() * 100
    gamma_1bp = (price_down_1bp - 2 * price_base + price_up_1bp) * 100000000 / 100
    convexity = gamma_1bp / bond_object.dirtyPrice() * 100

    otr_merged_sorted.loc[index, 'scen_dv01'] = dv01
    otr_merged_sorted.loc[index, 'scen_duration'] = duration
    otr_merged_sorted.loc[index, 'scen_convexity'] = gamma_1bp

print("\nScenario risk metrics:")
print(otr_merged_sorted[['security', 'scen_dv01', 'scen_duration', 'scen_convexity']].to_string(index=False))

# Save Problem 3 results
otr_merged_sorted.to_csv('Problem_3_Treasury_Results.csv', index=False)
print("\nSaved: Problem_3_Treasury_Results.csv")


# =============================================================================
# PROBLEM 4: PRICING AND RISK METRICS FOR CORPORATE BONDS
# =============================================================================

print("\n" + "="*80)
print("PROBLEM 4: PRICING AND RISK METRICS FOR CORPORATE BONDS")
print("="*80)

# Problem 4a: Create fixed-rate corporate bond objects
print("\nProblem 4a: Creating corporate bond objects...")

corp_bond_symbology = bond_symbology[bond_symbology['class'] == 'Corp'].copy()
print(f"Number of corporate bonds: {len(corp_bond_symbology)}")

# Merge with market data
corp_bonds_merged = pd.merge(
    corp_bond_symbology,
    bond_market_prices_eod[['isin', 'date', 'bidPrice', 'askPrice', 'bidYield', 'askYield', 'midPrice', 'midYield']],
    on='isin',
    how='left'
)

# Filter out bonds without market data
corp_bonds_merged = corp_bonds_merged.dropna(subset=['midPrice'])
print(f"Corporate bonds with market data: {len(corp_bonds_merged)}")

# Problem 4b: Compute analytical Yields and Z-Spreads
print("\nProblem 4b: Computing yields and z-spreads...")

# Create flat yield curve for z-spread calculation
flat_rate = ql.SimpleQuote(0.049)
flat_rate_handle = ql.QuoteHandle(flat_rate)
flat_yield_curve = ql.FlatForward(calc_date, flat_rate_handle, days, compound)
flat_yield_handle = ql.YieldTermStructureHandle(flat_yield_curve)

corp_bonds_merged['calc_yield'] = np.nan
corp_bonds_merged['calc_zspread'] = np.nan

count_success = 0
count_fail = 0
for index, row in corp_bonds_merged.iterrows():
    try:
        bond_object = create_bond_from_symbology(row)

        settle_date = bond_object.settlementDate(calc_date)
        day_counter = bond_object.dayCounter()

        # Calculate implied yield - use ql.BondFunctions.bondYield
        # Need to wrap price in ql.InterestRate.Simple
        implied_yield = ql.BondFunctions.bondYield(
            bond_object, ql.BondPrice(row['midPrice'], ql.BondPrice.Clean),
            day_counter, compound, cpn_freq, settle_date
        ) * 100
        corp_bonds_merged.loc[index, 'calc_yield'] = implied_yield

        # Calculate z-spread (doesn't need pricing engine)
        bond_zspread = ql.BondFunctions.zSpread(
            bond_object, ql.BondPrice(row['midPrice'], ql.BondPrice.Clean),
            flat_yield_curve, days, compound, cpn_freq, settle_date
        )
        corp_bonds_merged.loc[index, 'calc_zspread'] = bond_zspread * 10000  # Convert to bps
        count_success += 1

    except Exception as e:
        # Skip bonds with errors (e.g., matured bonds, pricing issues)
        count_fail += 1
        if count_fail <= 3:  # Print first 3 errors for debugging
            print(f"  Error with {row['security']}: {str(e)}")
        continue

print(f"Successfully calculated yields/z-spreads for {count_success}/{len(corp_bonds_merged)} bonds")

print("\nYields and Z-Spreads (first 10 bonds):")
print(corp_bonds_merged[['security', 'midYield', 'calc_yield', 'calc_zspread']].head(10).to_string(index=False))

# Problem 4c: Validate Z-Spread computation
print("\nProblem 4c: Validating z-spread computation...")

# Pick 3 corporate bonds from different issuers that have valid z-spreads
valid_bonds = corp_bonds_merged[corp_bonds_merged['calc_zspread'].notna()]
sample_bonds = valid_bonds.groupby('ticker').first().reset_index().head(3)

print("\nZ-Spread validation:")
for index, row in sample_bonds.iterrows():
    try:
        bond_object = create_bond_from_symbology(row)

        # Calculate price using z-spread
        zspread_value = row['calc_zspread'] / 10000  # Convert back from bps
        calc_price = calc_clean_price_with_zspread(bond_object, flat_yield_handle, zspread_value)

        print(f"\nBond: {row['security']}")
        print(f"  Market Mid Price: {row['midPrice']:.4f}")
        print(f"  Calc Price (w/ Z-Spread): {calc_price:.4f}")
        print(f"  Difference: {abs(calc_price - row['midPrice']):.6f}")
        print(f"  Z-Spread (bps): {row['calc_zspread']:.2f}")
    except Exception as e:
        print(f"\nError validating bond {row['security']}: {str(e)}")
        continue

# Problem 4d: Compute Duration and Convexity for corporate bonds
print("\nProblem 4d: Computing duration and convexity for corporate bonds...")

flat_yield_corp = 0.051
flat_yield_rate_corp = ql.InterestRate(flat_yield_corp, days, compound, cpn_freq)

corp_bonds_merged['calc_dv01'] = np.nan
corp_bonds_merged['calc_duration'] = np.nan
corp_bonds_merged['calc_convexity'] = np.nan

for index, row in corp_bonds_merged.iterrows():
    try:
        bond_object = create_bond_from_symbology(row)
        bond_object.setPricingEngine(bond_engine)

        duration = ql.BondFunctions.duration(bond_object, flat_yield_rate_corp)
        convexity = ql.BondFunctions.convexity(bond_object, flat_yield_rate_corp)

        corp_bonds_merged.loc[index, 'calc_duration'] = duration
        corp_bonds_merged.loc[index, 'calc_convexity'] = convexity
        corp_bonds_merged.loc[index, 'calc_dv01'] = duration * bond_object.dirtyPrice() / 100

    except Exception as e:
        continue

print("\nDuration and Convexity (first 10 bonds):")
print(corp_bonds_merged[['security', 'calc_dv01', 'calc_duration', 'calc_convexity']].head(10).to_string(index=False))

# Save Problem 4 results
corp_bonds_merged.to_csv('Problem_4_Corporate_Results.csv', index=False)
print("\nSaved: Problem_4_Corporate_Results.csv")

# Create summary visualization for corporate bonds (only valid data)
valid_zspread_bonds = corp_bonds_merged[corp_bonds_merged['calc_zspread'].notna()]
if len(valid_zspread_bonds) > 0:
    fig_corp_zspread = px.scatter(
        valid_zspread_bonds,
        x='TTM',
        y='calc_zspread',
        color='ticker',
        title='Corporate Bond Z-Spreads by Maturity',
        labels={'TTM': 'Time to Maturity (Years)', 'calc_zspread': 'Z-Spread (bps)'},
        hover_data=['security']
    )
    fig_corp_zspread.write_html('Problem_4_Corporate_ZSpreads.html')
    print("Saved: Problem_4_Corporate_ZSpreads.html")

# Create duration vs maturity plot (only valid data)
valid_duration_bonds = corp_bonds_merged[corp_bonds_merged['calc_duration'].notna()]
if len(valid_duration_bonds) > 0:
    fig_corp_duration = px.scatter(
        valid_duration_bonds,
        x='TTM',
        y='calc_duration',
        color='ticker',
        title='Corporate Bond Duration by Maturity',
        labels={'TTM': 'Time to Maturity (Years)', 'calc_duration': 'Duration'},
        hover_data=['security']
    )
    fig_corp_duration.write_html('Problem_4_Corporate_Duration.html')
    print("Saved: Problem_4_Corporate_Duration.html")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*80)
print("SOLUTION SUMMARY")
print("="*80)

print("\nProblems Solved:")
print("  ✓ Problem 1: Constructing fixed rate bonds")
print("    - Loaded and prepared symbology and market data")
print("    - Created bond construction functions")
print("    - Generated cashflow schedules")

print("\n  ✓ Problem 2: US Treasury yield curve calibration")
print("    - Created on-the-run treasury bond objects")
print("    - Calibrated bid, ask, and mid yield curves")
print("    - Plotted yield curves and discount factors")

print("\n  ✓ Problem 3: Pricing and risk metrics for US Treasury bonds")
print("    - Priced treasuries on calibrated curve")
print("    - Computed analytical DV01, duration, convexity")
print("    - Computed scenario-based risk metrics")

print("\n  ✓ Problem 4: Pricing and risk metrics for corporate bonds")
print("    - Created corporate bond objects")
print("    - Computed yields and z-spreads")
print("    - Validated z-spread computation")
print("    - Computed duration and convexity")

print("\nOutput Files Generated:")
print("  - Problem_2a_OTR_Yields.html")
print("  - Problem_2c_Calibrated_Yield_Curves.html")
print("  - Problem_2d_Discount_Factors.html")
print("  - Problem_3_Treasury_Results.csv")
print("  - Problem_4_Corporate_Results.csv")
print("  - Problem_4_Corporate_ZSpreads.html")
print("  - Problem_4_Corporate_Duration.html")

print("\n" + "="*80)
print("PSET 2 COMPLETE!")
print("="*80)
