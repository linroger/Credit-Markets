import marimo as mo
import pandas as pd
import numpy as np
import sympy as sp
import QuantLib as ql
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from UChicago_FINM_35700_CreditMarkets_Spring2024_FinalExam import credit_market_tools as cmt

app = mo.App()
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
calc_date = pd.Timestamp("2024-05-03")
ql_calc_date = ql.Date(calc_date.day, calc_date.month, calc_date.year)
ql.Settings.instance().evaluationDate = ql_calc_date


@app.cell
def _():
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
    tables = {k: pd.DataFrame(v, columns=["Dependency", "StatementTrue"]) for k, v in problem1_answers.items()}
    return tables


@app.cell
def _(tables):
    bond_sym = pd.read_excel(DATA / "bond_symbology.xlsx")
    bond_mkt = pd.read_excel(DATA / "bond_market_prices_eod.xlsx")
    aapl_sym = bond_sym[bond_sym["isin"] == "US037833AT77"].iloc[0]
    aapl_mkt = bond_mkt[bond_mkt["isin"] == "US037833AT77"].iloc[0]
    aapl_mid_yield = float(np.mean([aapl_mkt["bidYield"], aapl_mkt["askYield"]]))
    bond = cmt.create_bond_from_symbology(aapl_sym)
    cashflows = cmt.get_bond_cashflows(bond, ql_calc_date)
    dc = bond.dayCounter()
    y_dec = aapl_mid_yield / 100
    rate = ql.InterestRate(y_dec, dc, ql.Compounded, ql.Semiannual)
    price = ql.BondFunctions.cleanPrice(bond, y_dec, dc, ql.Compounded, ql.Semiannual)
    duration = ql.BondFunctions.duration(bond, rate)
    convexity = ql.BondFunctions.convexity(bond, y_dec, dc, ql.Compounded, ql.Semiannual)
    dv01 = ql.BondFunctions.bps(bond, y_dec, dc, ql.Compounded, ql.Semiannual)
    return tables, cashflows, pd.DataFrame({"mid_yield": [aapl_mid_yield], "price": [price], "duration": [duration], "convexity": [convexity], "dv01": [dv01]})


@app.cell
def _():
    sofr_sym = pd.read_excel(DATA / "sofr_swaps_symbology.xlsx")
    sofr_mkt = pd.read_excel(DATA / "sofr_swaps_market_data_eod.xlsx")
    sofr_details = sofr_sym.merge(sofr_mkt, on="figi").sort_values("tenor").drop_duplicates(subset="tenor")
    sofr_curve = cmt.calibrate_sofr_curve_from_frame(ql_calc_date, sofr_details, "midRate")
    sofr_df = cmt.get_yield_curve_details_df(sofr_curve)
    return sofr_curve, sofr_df


@app.cell
def _(sofr_curve, sofr_df):
    cds_df = pd.read_excel(DATA / "cds_market_data_eod.xlsx")
    ford = cds_df[cds_df["ticker"] == "F"]
    spread_cols = ["par_spread_1y", "par_spread_2y", "par_spread_3y", "par_spread_5y", "par_spread_7y", "par_spread_10y"]
    ford_today = ford[ford["date"] == pd.Timestamp("2024-05-03")].iloc[0]
    par_spreads = [ford_today[c] for c in spread_cols]
    hazard_curve = cmt.calibrate_cds_hazard_rate_curve(ql_calc_date, ql.YieldTermStructureHandle(sofr_curve), par_spreads, cds_recovery_rate=0.4)
    hazard_df = cmt.get_hazard_rates_df(hazard_curve)
    discount_curve = ql.FlatForward(ql_calc_date, ql.QuoteHandle(ql.SimpleQuote(sofr_curve.zeroRate(sofr_curve.referenceDate(), ql.Actual365Fixed(), ql.Compounded).rate())), ql.Actual365Fixed())
    maturity_date = ql.Date(20, 6, 2029)
    schedule = ql.Schedule(ql_calc_date, maturity_date, ql.Period(ql.Quarterly), ql.TARGET(), ql.Following, ql.Following, ql.DateGeneration.TwentiethIMM, False)
    cds = ql.CreditDefaultSwap(ql.Protection.Seller, 10_000_000, 0.01, schedule, ql.Following, ql.Actual360(), True, True)
    cds.setPricingEngine(ql.IsdaCdsEngine(ql.DefaultProbabilityTermStructureHandle(hazard_curve), 0.4, ql.YieldTermStructureHandle(discount_curve)))
    cds_metrics = {
        "cds_pv": cds.NPV(),
        "premium_leg_pv": cds.couponLegNPV(),
        "default_leg_pv": cds.defaultLegNPV(),
        "par_spread_bps": cds.fairSpread() * 1e4,
        "survival_to_maturity": hazard_curve.survivalProbability(maturity_date),
    }
    return sofr_df, hazard_df, pd.DataFrame([cds_metrics])


@app.cell
def _():
    c_sym, y_sym, T_sym = sp.symbols("c y T", positive=True)
    zero_coupon_pv = sp.exp(-y_sym * T_sym)
    zero_coupon_dv01 = -sp.diff(zero_coupon_pv, y_sym) * 1e-4
    interest_only_pv = (c_sym / 2) * (1 - sp.exp(-y_sym * T_sym)) / (sp.exp(y_sym / 2) - 1)
    interest_only_dv01 = -sp.diff(interest_only_pv, y_sym) * 1e-4
    coupon_star = sp.solve(sp.Eq(interest_only_pv, zero_coupon_pv), c_sym)[0]
    return zero_coupon_pv, zero_coupon_dv01, interest_only_pv, interest_only_dv01, coupon_star


@app.cell
def _(zero_coupon_pv, zero_coupon_dv01, interest_only_pv, interest_only_dv01):
    zc_pv_func = sp.lambdify((sp.symbols("c"), sp.symbols("y"), sp.symbols("T")), zero_coupon_pv, "numpy")
    y_vals = np.linspace(0.01, 0.1, 30)
    T_vals = np.arange(1, 21)
    Y, T = np.meshgrid(y_vals, T_vals)
    zc_pv_vals = zc_pv_func(0.05, Y, T)
    fig_zc_pv = go.Figure(data=[go.Surface(x=Y, y=T, z=zc_pv_vals)])
    fig_zc_pv.update_layout(title="Zero Coupon PV Surface")
    return fig_zc_pv


@app.cell
def _():
    lqd_basket = pd.read_excel(DATA / "lqd_basket_composition.xlsx")
    lqd_sym = pd.read_excel(DATA / "lqd_corp_symbology.xlsx")
    lqd_combined = lqd_basket.merge(lqd_sym, on="isin", suffixes=("", "_sym"))
    lqd_combined = lqd_combined.dropna(subset=["midYield"])
    dv01s = []
    for _, row in lqd_combined.iterrows():
        row_local = row.to_dict()
        row_local["class"] = str(row_local["class"]).capitalize()
        bond = cmt.create_bond_from_symbology(row_local)
        y = row["midYield"] / 100
        dc = bond.dayCounter()
        dv01s.append(ql.BondFunctions.bps(bond, y, dc, ql.Compounded, ql.Semiannual))
    lqd_combined = lqd_combined.assign(bond_DV01=dv01s)
    lqd_combined["basket_DV01"] = lqd_combined["bond_DV01"] * lqd_combined["face_notional"] / 10000
    agg_df = lqd_combined.groupby("und_bench_tsy_isin").agg({"isin": "count", "face_notional": "sum", "basket_DV01": "sum"}).rename(columns={"isin": "basket_count"}).reset_index()
    bench_sym = pd.read_excel(DATA / "bond_symbology.xlsx")
    bench_sym = bench_sym[bench_sym["isin"].isin(agg_df["und_bench_tsy_isin"])]
    bench_sym = bench_sym.assign(TTM=(bench_sym["maturity"] - calc_date).dt.days / 365)
    combined = agg_df.merge(bench_sym[["isin", "security", "TTM"]], left_on="und_bench_tsy_isin", right_on="isin", how="left")
    combined = combined.sort_values("TTM")
    return lqd_combined, combined


@app.cell
def _(lqd_combined, combined):
    run_sym = pd.read_excel(DATA / "govt_on_the_run.xlsx")
    bond_sym = pd.read_excel(DATA / "bond_symbology.xlsx")
    bond_mkt = pd.read_excel(DATA / "bond_market_prices_eod.xlsx")
    run_details = run_sym.merge(bond_sym, on="isin", how="left")
    run_market = bond_mkt.merge(run_sym[["isin"]], on="isin")
    run_market["midPrice"] = run_market[["bidPrice", "askPrice"]].mean(axis=1)
    run_details = run_details.merge(run_market[["isin", "midPrice"]], on="isin")
    run_details = run_details.dropna(subset=["midPrice"])
    run_details = run_details.sort_values("maturity")
    run_details["class"] = "Govt"
    tsy_curve = cmt.calibrate_yield_curve_from_frame(ql_calc_date, run_details, "midPrice")
    tsy_df = cmt.get_yield_curve_details_df(tsy_curve)
    orcl_sym = bond_sym[bond_sym["ticker"] == "ORCL"]
    orcl_mkt = bond_mkt[bond_mkt["ticker"] == "ORCL"]
    orcl_combined = orcl_sym.merge(orcl_mkt, on=["ticker", "isin", "figi", "class"])
    orcl_combined = orcl_combined[(orcl_combined["cpn_type"] == "FIXED") & (orcl_combined["amt_out"] > 100)]
    orcl_combined["midPrice"] = orcl_combined[["bidPrice", "askPrice"]].mean(axis=1)
    orcl_combined["midYield"] = orcl_combined[["bidYield", "askYield"]].mean(axis=1)
    orcl_combined = orcl_combined.dropna(subset=["midPrice"])
    orcl_combined = orcl_combined.sort_values("maturity")
    tsy_handle = ql.YieldTermStructureHandle(tsy_curve)
    calib = cmt.calibrate_nelson_siegel_model((0.02, 0.01, -0.01, 1.0), ql_calc_date, orcl_combined, tsy_handle, bond_recovery_rate=0.4)
    params = calib.x
    bonds, weights, market_prices, market_yields, bond_dv01s, durations = cmt.create_bonds_and_weights(orcl_combined, tsy_handle)
    model_prices, model_yields = cmt.calculate_nelson_siegel_model_prices_and_yields(params, ql_calc_date, bonds, tsy_handle, bond_recovery_rate=0.4)
    orcl_results = orcl_combined.copy()
    orcl_results["modelPrice"] = model_prices
    orcl_results["modelYield"] = model_yields
    orcl_results["edgePrice"] = orcl_results["midPrice"] - orcl_results["modelPrice"]
    orcl_results["edgeYield"] = orcl_results["midYield"] - orcl_results["modelYield"]
    return tsy_df, orcl_results


if __name__ == "__main__":
    app.run()
