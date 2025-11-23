import nbformat as nbf
from textwrap import dedent
from pathlib import Path

cells = []


def m(text: str):
    cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))


def c(text: str):
    cells.append(nbf.v4.new_code_cell(dedent(text).strip()))


m(
    """
    # Credit Markets Final Exam – Consolidated, Step-by-Step Solutions
    This notebook restates every exam question and solves it in order using the provided data and helper utilities.
    Data manipulations are shown as data frames, Plotly provides interactive visuals, and latex-formatted tables summarize
    key outputs.  All calculations are rerunnable from a clean environment.
    """
)

c(
    r"""import pandas as pd
import numpy as np
import sympy as sp
import QuantLib as ql
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from UChicago_FINM_35700_CreditMarkets_Spring2024_FinalExam import credit_market_tools as cmt

pd.set_option('display.float_format', lambda v: f"{v:,.4f}")
BASE = Path('UChicago_FINM_35700_CreditMarkets_Spring2024_FinalExam')
DATA = BASE / 'data'
calc_date = pd.Timestamp('2024-05-03')
ql_calc_date = ql.Date(calc_date.day, calc_date.month, calc_date.year)
ql.Settings.instance().evaluationDate = ql_calc_date

def latex_table(df):
    align = '|'.join(['c'] * len(df.columns))
    lines = [r"\begin{document}", rf"\begin{{tabular}}{{|{align}|}}", r"\hline"]
    lines.append(' & '.join(df.columns) + r" \\ \hline")
    for _, row in df.iterrows():
        lines.append(' & '.join([str(v) for v in row]) + r" \\ \hline")
    lines.extend([r"\end{tabular}", r"\end{document}"])
    return "```latex\n" + "\n".join(lines) + "\n```"
"""
)

m(
    """
    ## Problem 1 – Overall understanding of credit models (True/False)
    Ceteris paribus assumptions apply. Each table below directly mirrors the four sub-questions.
    """
)

c(
    """
problem1_answers = {
    "1a_prices": [
        ("Price vs interest rate", True),
        ("Price vs hazard rate", True),
        ("Price vs expected recovery", False),
        ("Price vs coupon", False),
        ("Price vs maturity", True),
    ],
    "1b_yields": [
        ("Yield vs interest rate", False),
        ("Yield vs hazard rate", False),
        ("Yield vs expected recovery", True),
        ("Yield vs coupon", False),
        ("Yield vs maturity", False),
    ],
    "1c_merton": [
        ("Equity value vs assets", False),
        ("Equity vol vs assets", True),
        ("Equity value vs asset vol", False),
        ("Equity value vs liabilities", True),
        ("Equity vol vs liabilities", False),
    ],
    "1d_spreads": [
        ("Yield vs liabilities", False),
        ("Expected recovery vs liabilities", True),
        ("Yield vs asset vol", False),
        ("Credit spread vs assets", True),
        ("Credit spread vs asset vol", False),
    ],
}

problem1_tables = {k: pd.DataFrame(v, columns=["Statement", "True?"]) for k, v in problem1_answers.items()}
for key, df in problem1_tables.items():
    display(df)
    print(latex_table(df))
"""
)

m(
    """
    ## Problem 2 – AAPL fixed-rate corporate bond (US037833AT77)
    *2a* build the bond object and list cash flows.  *2b* compute analytic price/DV01/duration/convexity from mid-yield.
    *2c–d* run scenario curves from 2%–10% in 0.5% steps.
    """
)

c(
    """
bond_sym = pd.read_excel(DATA / 'bond_symbology.xlsx')
bond_mkt = pd.read_excel(DATA / 'bond_market_prices_eod.xlsx')

AAPL_ISIN = 'US037833AT77'
aapl_sym = bond_sym[bond_sym['isin'] == AAPL_ISIN].iloc[0]
aapl_mkt = bond_mkt[bond_mkt['isin'] == AAPL_ISIN].iloc[0]
aapl_mid_yield = float(np.mean([aapl_mkt['bidYield'], aapl_mkt['askYield']]))

fixed_rate_bond = cmt.create_bond_from_symbology(aapl_sym)
cashflows_df = cmt.get_bond_cashflows(fixed_rate_bond, ql_calc_date)
cashflows_df
"""
)

c(
    """
bond_dc = fixed_rate_bond.dayCounter()
yield_decimal = aapl_mid_yield / 100
rate = ql.InterestRate(yield_decimal, bond_dc, ql.Compounded, ql.Semiannual)

price = ql.BondFunctions.cleanPrice(fixed_rate_bond, yield_decimal, bond_dc, ql.Compounded, ql.Semiannual)
duration = ql.BondFunctions.duration(fixed_rate_bond, rate)
convexity = ql.BondFunctions.convexity(fixed_rate_bond, yield_decimal, bond_dc, ql.Compounded, ql.Semiannual)
bond_dv01 = ql.BondFunctions.bps(fixed_rate_bond, yield_decimal, bond_dc, ql.Compounded, ql.Semiannual)

aapl_metrics = pd.DataFrame({
    'mid_yield': [aapl_mid_yield],
    'price': [price],
    'duration': [duration],
    'convexity': [convexity],
    'dv01': [bond_dv01]
})
display(aapl_metrics)
print(latex_table(aapl_metrics))
"""
)

c(
    """
scenario_yields = np.arange(0.02, 0.1001, 0.005)
scenario_prices = []
scenario_durations = []
scenario_convexities = []

for y in scenario_yields:
    price_y = ql.BondFunctions.cleanPrice(fixed_rate_bond, y, bond_dc, ql.Compounded, ql.Semiannual)
    dur_y = ql.BondFunctions.duration(fixed_rate_bond, ql.InterestRate(y, bond_dc, ql.Compounded, ql.Semiannual))
    conv_y = ql.BondFunctions.convexity(fixed_rate_bond, y, bond_dc, ql.Compounded, ql.Semiannual)
    scenario_prices.append(price_y)
    scenario_durations.append(dur_y)
    scenario_convexities.append(conv_y)

scenario_df = pd.DataFrame({
    'yield_pct': scenario_yields * 100,
    'price': scenario_prices,
    'duration': scenario_durations,
    'convexity': scenario_convexities,
})

fig_price = px.line(scenario_df, x='yield_pct', y='price', title='AAPL Price vs Yield', markers=True)
fig_duration = px.line(scenario_df, x='yield_pct', y='duration', title='AAPL Duration vs Yield', markers=True)
fig_convexity = px.line(scenario_df, x='yield_pct', y='convexity', title='AAPL Convexity vs Yield', markers=True)

scenario_df.head(), fig_price, fig_duration, fig_convexity
"""
)

m(
    """
    ## Problem 3 – Ford CDS curve calibration and valuation
    *3a* bootstrap SOFR zero/discount curves. *3b* plot historical Ford CDS par spreads. *3c* calibrate Ford hazard curve as of
    2024-05-03. *3d* value a 100 bps CDS maturing 2029-06-20.
    """
)

c(
    """
sofr_sym = pd.read_excel(DATA / 'sofr_swaps_symbology.xlsx')
sofr_mkt = pd.read_excel(DATA / 'sofr_swaps_market_data_eod.xlsx')
sofr_details = sofr_sym.merge(sofr_mkt, on='figi').sort_values('tenor').drop_duplicates(subset='tenor')
sofr_curve = cmt.calibrate_sofr_curve_from_frame(ql_calc_date, sofr_details, 'midRate')
sofr_df = cmt.get_yield_curve_details_df(sofr_curve)

fig_sofr_zero = px.line(sofr_df, x='YearFrac', y='ZeroRate', title='SOFR Zero Rates')
fig_sofr_df = px.line(sofr_df, x='YearFrac', y='DiscountFactor', title='SOFR Discount Factors')
sofr_df.head(), fig_sofr_zero, fig_sofr_df
"""
)

c(
    """
cds_df = pd.read_excel(DATA / 'cds_market_data_eod.xlsx')
ford_cds = cds_df[cds_df['ticker'] == 'F']
spread_cols = ['par_spread_1y','par_spread_2y','par_spread_3y','par_spread_5y','par_spread_7y','par_spread_10y']

ford_long = ford_cds.melt(id_vars=['date'], value_vars=spread_cols, var_name='tenor', value_name='par_spread_bps')
fig_cds_hist = px.line(ford_long, x='date', y='par_spread_bps', color='tenor', title='Ford CDS Par Spreads History')
ford_long.head(), fig_cds_hist
"""
)

c(
    """
ford_spread_today = ford_cds[ford_cds['date'] == calc_date].iloc[0]
par_spreads_vector = [ford_spread_today[c] for c in spread_cols]
sofr_handle = ql.YieldTermStructureHandle(sofr_curve)
hazard_curve = cmt.calibrate_cds_hazard_rate_curve(ql_calc_date, sofr_handle, par_spreads_vector, cds_recovery_rate=0.4)
hazard_df = cmt.get_hazard_rates_df(hazard_curve)

fig_hazard = px.line(hazard_df, x='YearFrac', y='HazardRateBps', title='Ford Hazard Rates')
fig_surv = px.line(hazard_df, x='YearFrac', y='SurvivalProb', title='Ford Survival Probability')
hazard_df.head(), fig_hazard, fig_surv
"""
)

c(
    """
maturity_date = ql.Date(20, 6, 2029)
schedule = ql.Schedule(ql_calc_date, maturity_date, ql.Period(ql.Quarterly), ql.TARGET(), ql.Following, ql.Following, ql.DateGeneration.TwentiethIMM, False)
cds = ql.CreditDefaultSwap(ql.Protection.Seller, 10_000_000, 0.01, schedule, ql.Following, ql.Actual360(), True, True)
spot_rate = sofr_curve.zeroRate(sofr_curve.referenceDate(), ql.Actual365Fixed(), ql.Compounded).rate()
cds_discount_curve = ql.FlatForward(ql_calc_date, ql.QuoteHandle(ql.SimpleQuote(spot_rate)), ql.Actual365Fixed())
cds_discount_handle = ql.YieldTermStructureHandle(cds_discount_curve)
engine = ql.IsdaCdsEngine(ql.DefaultProbabilityTermStructureHandle(hazard_curve), 0.4, cds_discount_handle)
cds.setPricingEngine(engine)

cds_metrics = {
    'cds_pv': cds.NPV(),
    'premium_leg_pv': cds.couponLegNPV(),
    'default_leg_pv': cds.defaultLegNPV(),
    'par_spread_bps': cds.fairSpread() * 1e4,
    'survival_to_maturity': hazard_curve.survivalProbability(maturity_date)
}
cds_df_metrics = pd.DataFrame([cds_metrics])
display(cds_df_metrics)
print(latex_table(cds_df_metrics))
"""
)

m(
    """
    ## Problem 4 – Sympy derivations for flat-yield risky bonds
    We re-derive analytic PV and DV01 for zero-coupon and interest-only bonds, then solve for the coupon \(c^*\) that equalizes the PVs.
    """
)

c(
    """
c_sym, y_sym, T_sym = sp.symbols('c y T', positive=True)

zero_coupon_pv = sp.exp(-y_sym * T_sym)
zero_coupon_dv01 = -sp.diff(zero_coupon_pv, y_sym) * 1e-4

interest_only_pv = (c_sym/2) * (1 - sp.exp(-y_sym * T_sym)) / (sp.exp(y_sym/2) - 1)
interest_only_dv01 = -sp.diff(interest_only_pv, y_sym) * 1e-4

zero_coupon_pv, zero_coupon_dv01, interest_only_pv, interest_only_dv01
"""
)

c(
    """
zc_pv_func = sp.lambdify((c_sym, y_sym, T_sym), zero_coupon_pv, 'numpy')
zc_dv01_func = sp.lambdify((c_sym, y_sym, T_sym), zero_coupon_dv01, 'numpy')
io_pv_func = sp.lambdify((c_sym, y_sym, T_sym), interest_only_pv, 'numpy')
io_dv01_func = sp.lambdify((c_sym, y_sym, T_sym), interest_only_dv01, 'numpy')

c_val = 0.05
y_vals = np.linspace(0.01, 0.10, 30)
T_vals = np.arange(1, 21)
Y, T = np.meshgrid(y_vals, T_vals)

zc_pv_vals = zc_pv_func(c_val, Y, T)
zc_dv_vals = zc_dv01_func(c_val, Y, T)
io_pv_vals = io_pv_func(c_val, Y, T)
io_dv_vals = io_dv01_func(c_val, Y, T)

fig_zc_pv = go.Figure(data=[go.Surface(x=Y, y=T, z=zc_pv_vals)])
fig_zc_pv.update_layout(title='Zero Coupon PV Surface', scene=dict(xaxis_title='Yield', yaxis_title='Maturity', zaxis_title='PV'))

fig_zc_dv = go.Figure(data=[go.Surface(x=Y, y=T, z=zc_dv_vals)])
fig_zc_dv.update_layout(title='Zero Coupon DV01 Surface', scene=dict(xaxis_title='Yield', yaxis_title='Maturity', zaxis_title='DV01'))

fig_io_pv = go.Figure(data=[go.Surface(x=Y, y=T, z=io_pv_vals)])
fig_io_pv.update_layout(title='Interest Only PV Surface', scene=dict(xaxis_title='Yield', yaxis_title='Maturity', zaxis_title='PV'))

fig_io_dv = go.Figure(data=[go.Surface(x=Y, y=T, z=io_dv_vals)])
fig_io_dv.update_layout(title='Interest Only DV01 Surface', scene=dict(xaxis_title='Yield', yaxis_title='Maturity', zaxis_title='DV01'))

fig_zc_pv, fig_zc_dv, fig_io_pv, fig_io_dv
"""
)

c(
    """
coupon_star = sp.solve(sp.Eq(interest_only_pv, zero_coupon_pv), c_sym)[0]
coupon_star
"""
)

m(
    """
    ## Problem 5 – LQD ETF basket DV01 analysis
    *5a* load basket and symbology; summarize counts/face and yield stats. *5b* compute each bond DV01 and contribution.
    *5c–d* bucket by underlying benchmark Treasury, aggregate metrics, and visualize exposures.
    """
)

c(
    """
lqd_basket = pd.read_excel(DATA / 'lqd_basket_composition.xlsx')
lqd_sym = pd.read_excel(DATA / 'lqd_corp_symbology.xlsx')

lqd_combined = lqd_basket.merge(lqd_sym, on='isin', suffixes=('', '_sym'))
lqd_combined = lqd_combined.dropna(subset=['midYield'])

summary_stats = {
    'bond_count': [len(lqd_combined)],
    'face_notional_mean': [lqd_combined['face_notional'].mean()],
    'face_notional_median': [lqd_combined['face_notional'].median()],
    'yield_mean': [lqd_combined['midYield'].mean()],
    'yield_std': [lqd_combined['midYield'].std()],
}
summary_df = pd.DataFrame(summary_stats)
display(summary_df)
print(latex_table(summary_df))
"""
)

c(
    """
bond_dv01_list = []
for idx, row in lqd_combined.iterrows():
    row_local = row.to_dict()
    row_local['class'] = str(row_local['class']).capitalize()
    bond_obj = cmt.create_bond_from_symbology(row_local)
    y = row['midYield'] / 100
    dc = bond_obj.dayCounter()
    dv01 = ql.BondFunctions.bps(bond_obj, y, dc, ql.Compounded, ql.Semiannual)
    bond_dv01_list.append(dv01)

lqd_combined = lqd_combined.assign(bond_DV01=bond_dv01_list)
lqd_combined['basket_DV01'] = lqd_combined['bond_DV01'] * lqd_combined['face_notional'] / 10000
lqd_combined.head()
"""
)

c(
    """
agg_cols = {
    'isin': 'count',
    'face_notional': 'sum',
    'basket_DV01': 'sum'
}
agg_df = lqd_combined.groupby('und_bench_tsy_isin').agg(agg_cols).rename(columns={'isin': 'basket_count'}).reset_index()
display(agg_df)
print(latex_table(agg_df))
"""
)

c(
    """
bond_sym = pd.read_excel(DATA / 'bond_symbology.xlsx')
bench_sym = bond_sym[bond_sym['isin'].isin(agg_df['und_bench_tsy_isin'])][['isin','security','maturity']]
bench_sym = bench_sym.assign(TTM=(bench_sym['maturity'] - calc_date).dt.days / 365)
combined_bench = agg_df.merge(bench_sym, left_on='und_bench_tsy_isin', right_on='isin', how='left')
combined_bench = combined_bench.sort_values('TTM')

bar_count = px.bar(combined_bench, x='security', y='basket_count', title='Basket Count by Benchmark TSY')
bar_face = px.bar(combined_bench, x='security', y='face_notional', title='Face Notional by Benchmark TSY')
bar_dv01 = px.bar(combined_bench, x='security', y='basket_DV01', title='DV01 by Benchmark TSY')

combined_bench, bar_count, bar_face, bar_dv01
"""
)

m(
    """
    ## Problem 6 – Nelson–Siegel smooth hazard curve for ORCL
    *6a* calibrate on-the-run Treasury curve. *6b* prepare ORCL fixed bonds (amt_out > 100). *6c* calibrate NS parameters.
    *6d–e* compute model prices/yields, edges, and plot comparisons.
    """
)

c(
    """
bond_sym = pd.read_excel(DATA / 'bond_symbology.xlsx')
bond_mkt = pd.read_excel(DATA / 'bond_market_prices_eod.xlsx')

run_sym = pd.read_excel(DATA / 'govt_on_the_run.xlsx')
run_details = run_sym.merge(bond_sym, on='isin', how='left')
run_market = bond_mkt.merge(run_sym[['isin']], on='isin')
run_market['midPrice'] = run_market[['bidPrice','askPrice']].mean(axis=1)
run_details = run_details.merge(run_market[['isin','midPrice']], on='isin')
run_details = run_details.sort_values('maturity')
run_details['class'] = 'Govt'

tsy_curve = cmt.calibrate_yield_curve_from_frame(ql_calc_date, run_details, 'midPrice')
tsy_curve_df = cmt.get_yield_curve_details_df(tsy_curve)

fig_tsy_zero = px.line(tsy_curve_df, x='YearFrac', y='ZeroRate', title='On-the-Run Treasury Zero Rates')
fig_tsy_df = px.line(tsy_curve_df, x='YearFrac', y='DiscountFactor', title='On-the-Run Treasury Discount Factors')
tsy_curve_df.head(), fig_tsy_zero, fig_tsy_df
"""
)

c(
    """
orcl_sym = bond_sym[bond_sym['ticker'] == 'ORCL']
orcl_mkt = bond_mkt[bond_mkt['ticker'] == 'ORCL']
orcl_combined = orcl_sym.merge(orcl_mkt, on=['ticker','isin','figi','class'])
orcl_combined = orcl_combined[(orcl_combined['cpn_type']=='FIXED') & (orcl_combined['amt_out']>100)]
orcl_combined['midPrice'] = orcl_combined[['bidPrice','askPrice']].mean(axis=1)
orcl_combined['midYield'] = orcl_combined[['bidYield','askYield']].mean(axis=1)
orcl_combined = orcl_combined.dropna(subset=['midPrice'])
orcl_combined = orcl_combined.sort_values('maturity')
orcl_combined['TTM'] = (orcl_combined['maturity'] - calc_date).dt.days / 365

fig_orcl_yield = px.scatter(orcl_combined, x='TTM', y='midYield', title='ORCL Market Yields by TTM')
orcl_combined.head(), fig_orcl_yield
"""
)

c(
    """
initial_params = (0.02, 0.01, -0.01, 1.0)
tsy_handle = ql.YieldTermStructureHandle(tsy_curve)
calib_results = cmt.calibrate_nelson_siegel_model(initial_params, ql_calc_date, orcl_combined, tsy_handle, bond_recovery_rate=0.4)
opt_params = calib_results.x

fixed_rate_bond_objects, calib_weights, bond_market_prices, bond_yields, bond_DV01s, bond_durations = cmt.create_bonds_and_weights(orcl_combined, tsy_handle)

bond_model_prices, bond_model_yields = cmt.calculate_nelson_siegel_model_prices_and_yields(opt_params, ql_calc_date, fixed_rate_bond_objects, tsy_handle, bond_recovery_rate=0.4)

orcl_results = orcl_combined.copy()
orcl_results['modelPrice'] = bond_model_prices
orcl_results['modelYield'] = bond_model_yields
orcl_results['edgePrice'] = orcl_results['midPrice'] - orcl_results['modelPrice']
orcl_results['edgeYield'] = orcl_results['midYield'] - orcl_results['modelYield']

params_df = pd.DataFrame({'parameter': ['theta1','theta2','theta3','lambda'], 'value': opt_params})
display(params_df)
print(latex_table(params_df))
orcl_results.head()
"""
)

c(
    """
fig_price_comp = px.scatter(orcl_results, x='maturity', y=['midPrice','modelPrice'], title='ORCL Model vs Market Prices')
fig_yield_comp = px.scatter(orcl_results, x='maturity', y=['midYield','modelYield'], title='ORCL Model vs Market Yields')
fig_edge_yield = px.bar(orcl_results, x='maturity', y='edgeYield', title='ORCL Yield Edges')

fig_price_comp, fig_yield_comp, fig_edge_yield
"""
)

nb = nbf.v4.new_notebook()
nb['cells'] = cells
nb['metadata'] = {"kernelspec": {"name": "python3", "language": "python", "display_name": "Python 3"}}

output_path = Path('UChicago_FINM_35700_CreditMarkets_Spring2024_FinalExam/final_exam_solutions.ipynb')
nbf.write(nb, output_path)
print('Wrote', output_path)
