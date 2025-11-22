import marimo

__generated_with = "0.1.0"
app = marimo.App()

@app.cell
def __():
    import QuantLib as ql
    import numpy as np
    import pandas as pd
    import datetime as dt
    import sympy as sp
    import plotly.graph_objects as go
    import plotly.express as px
    from scipy.optimize import minimize
    import credit_market_tools as cmt
    import matplotlib.pyplot as plt
    return cmt, dt, go, minimize, np, pd, plt, px, ql, sp


@app.cell
def __(ql, pd):
    # Setup calculation date
    calc_date = ql.Date(3, 5, 2024)
    ql.Settings.instance().evaluationDate = calc_date
    as_of_date = pd.to_datetime('2024-05-03')
    return as_of_date, calc_date


@app.cell
def __(cmt, ql):
    # Override problematic functions from credit_market_tools.py
    def create_bonds_and_weights_fixed(bond_details, tsy_yield_curve_handle):
        risk_free_bond_engine = ql.DiscountingBondEngine(tsy_yield_curve_handle)
        fixed_rate_bond_objects = []
        bond_market_prices = []
        bond_yields = []
        bond_DV01s = []
        bond_durations = []
        valid_indices = []

        for index, row in bond_details.iterrows():
            fixed_rate_bond = cmt.create_bond_from_symbology(row)
            fixed_rate_bond.setPricingEngine(risk_free_bond_engine)
            bond_price = row['midPrice']

            try:
                p = float(bond_price)
                try:
                    bond_yield = fixed_rate_bond.bondYield(p, ql.Thirty360(ql.Thirty360.USA), ql.Compounded, ql.Semiannual) * 100
                except Exception:
                    bond_yield = 5.0

                bond_yield_rate = ql.InterestRate(bond_yield/100, ql.ActualActual(ql.ActualActual.ISMA), ql.Compounded, ql.Semiannual)
                bond_duration = ql.BondFunctions.duration(fixed_rate_bond, bond_yield_rate)
                accrued = fixed_rate_bond.accruedAmount()
                dirty_price_val = p + accrued
                bond_mod_duration = ql.BondFunctions.duration(fixed_rate_bond, bond_yield_rate, ql.Duration.Modified)
                bond_DV01 = dirty_price_val * bond_mod_duration * 0.0001

                fixed_rate_bond_objects.append(fixed_rate_bond)
                bond_market_prices.append(bond_price)
                bond_yields.append(bond_yield)
                bond_DV01s.append(bond_DV01)
                bond_durations.append(bond_duration)
                valid_indices.append(index)
            except Exception as e:
                continue

        calib_weights = [1 / d if d != 0 else 0 for d in bond_DV01s]
        if not calib_weights:
            return [], [], [], [], [], [], []
        sum_calib_weights = sum(calib_weights)
        if sum_calib_weights == 0:
             calib_weights = [1.0/len(calib_weights)] * len(calib_weights)
        else:
             calib_weights = [x / sum_calib_weights for x in calib_weights]
        return(fixed_rate_bond_objects, calib_weights, bond_market_prices, bond_yields, bond_DV01s, bond_durations, valid_indices)

    def calculate_nelson_siegel_model_prices_and_yields_fixed(nelson_siegel_params, calc_date, fixed_rate_bond_objects, tsy_yield_curve_handle, bond_recovery_rate = 0.4):
        nelson_siegel_surv_prob_curve_handle = cmt.create_nelson_siegel_curve(calc_date, nelson_siegel_params)
        nelson_siegel_risky_bond_engine = ql.RiskyBondEngine(nelson_siegel_surv_prob_curve_handle, bond_recovery_rate, tsy_yield_curve_handle)
        bond_model_prices = []
        bond_model_yields = []
        for fixed_rate_bond in fixed_rate_bond_objects:
            fixed_rate_bond.setPricingEngine(nelson_siegel_risky_bond_engine)
            bond_price = 0.0
            bond_yield = 0.0
            try:
                bond_price = fixed_rate_bond.cleanPrice()
            except Exception:
                bond_price = 0.0
            if bond_price > 0:
                try:
                    bond_yield = fixed_rate_bond.bondYield(bond_price, ql.Thirty360(ql.Thirty360.USA), ql.Compounded, ql.Semiannual) * 100
                except Exception:
                    bond_yield = 0.0
            bond_model_prices.append(bond_price)
            bond_model_yields.append(bond_yield)
        return(bond_model_prices, bond_model_yields)

    def nelson_siegel_sse_fixed(nelson_siegel_params, calc_date, fixed_rate_bond_objects, market_prices, calib_weights, tsy_yield_curve_handle, bond_recovery_rate = 0.4):
        bond_model_prices, bond_model_yields = calculate_nelson_siegel_model_prices_and_yields_fixed(nelson_siegel_params, calc_date, fixed_rate_bond_objects, tsy_yield_curve_handle, bond_recovery_rate)
        sse = 0
        for i in range(len(market_prices)):
            if bond_model_prices[i] <= 0.01:
                 model_error = 100.0
            else:
                 model_error = market_prices[i] - bond_model_prices[i]
            sse += model_error * model_error * calib_weights[i]
        return(sse)

    def calibrate_nelson_siegel_model_fixed(initial_nelson_siegel_params, calc_date, bond_details, tsy_yield_curve_handle, bond_recovery_rate = 0.4):
        fixed_rate_bond_objects, calib_weights, bond_market_prices, bond_yields, bond_DV01s, bond_durations, valid_indices = create_bonds_and_weights_fixed(bond_details, tsy_yield_curve_handle)
        if not fixed_rate_bond_objects:
            raise ValueError("No valid bonds for calibration")
        param_bounds = [(1e-3, 0.1), (-0.1, 0.1), (-0.1, 0.1), (1e-3, 10)]
        calib_results = cmt.minimize(nelson_siegel_sse_fixed, initial_nelson_siegel_params, args = (calc_date, fixed_rate_bond_objects, bond_market_prices, calib_weights, tsy_yield_curve_handle, bond_recovery_rate), bounds = param_bounds)
        return calib_results, valid_indices

    def calibrate_yield_curve_from_frame_stable(calc_date, treasury_details, price_quote_column):
        ql.Settings.instance().evaluationDate = calc_date
        sorted_details_frame = treasury_details.sort_values(by='maturity')
        day_count = ql.ActualActual(ql.ActualActual.ISMA)
        bond_helpers = []
        for index, row in sorted_details_frame.iterrows():
            bond_object = cmt.create_bond_from_symbology(row)
            tsy_clean_price_quote = row[price_quote_column]
            tsy_clean_price_handle = ql.QuoteHandle(ql.SimpleQuote(tsy_clean_price_quote))
            bond_helper = ql.BondHelper(tsy_clean_price_handle, bond_object)
            bond_helpers.append(bond_helper)
        yield_curve = ql.PiecewiseFlatForward(calc_date, bond_helpers, day_count)
        yield_curve.enableExtrapolation()
        return yield_curve
    return (
        calculate_nelson_siegel_model_prices_and_yields_fixed,
        calibrate_nelson_siegel_model_fixed,
        calibrate_yield_curve_from_frame_stable,
        create_bonds_and_weights_fixed,
        nelson_siegel_sse_fixed,
    )


@app.cell
def __(calc_date, cmt, pd, ql):
    # --- Problem 2 ---
    bond_symbology = pd.read_excel('UChicago_FINM_35700_CreditMarkets_Spring2024_FinalExam/data/bond_symbology.xlsx')
    bond_market_prices_eod = pd.read_excel('UChicago_FINM_35700_CreditMarkets_Spring2024_FinalExam/data/bond_market_prices_eod.xlsx')
    bond_market_prices_eod['midPrice'] = (bond_market_prices_eod['bidPrice'] + bond_market_prices_eod['askPrice']) / 2
    bond_market_prices_eod['midYield'] = (bond_market_prices_eod['bidYield'] + bond_market_prices_eod['askYield']) / 2

    aapl_row = bond_symbology[bond_symbology['isin'] == 'US037833AT77'].iloc[0]
    aapl_bond = cmt.create_bond_from_symbology(aapl_row)
    aapl_cashflows = cmt.get_bond_cashflows(aapl_bond, calc_date)

    aapl_market = bond_market_prices_eod[bond_market_prices_eod['isin'] == 'US037833AT77'].iloc[0]
    aapl_yield = aapl_market['midYield']
    y = aapl_yield / 100.0

    flat_rate = ql.SimpleQuote(y)
    rate_handle = ql.QuoteHandle(flat_rate)
    day_count = ql.Thirty360(ql.Thirty360.USA)
    compounding = ql.Compounded
    frequency = ql.Semiannual

    yield_ts = ql.FlatForward(calc_date, rate_handle, day_count, compounding, frequency)
    engine = ql.DiscountingBondEngine(ql.YieldTermStructureHandle(yield_ts))
    aapl_bond.setPricingEngine(engine)

    price = aapl_bond.cleanPrice()
    interest_rate = ql.InterestRate(y, day_count, compounding, frequency)
    mod_duration = ql.BondFunctions.duration(aapl_bond, interest_rate, ql.Duration.Modified)
    convexity = ql.BondFunctions.convexity(aapl_bond, interest_rate)
    dv01 = mod_duration * price * 0.0001
    return (
        aapl_bond,
        aapl_cashflows,
        aapl_market,
        aapl_row,
        aapl_yield,
        bond_market_prices_eod,
        bond_symbology,
        compounding,
        convexity,
        day_count,
        dv01,
        engine,
        flat_rate,
        frequency,
        interest_rate,
        mod_duration,
        price,
        rate_handle,
        y,
        yield_ts,
    )


@app.cell
def __(calc_date, cmt, pd, ql):
    # --- Problem 3 ---
    sofr_symbology = pd.read_excel('UChicago_FINM_35700_CreditMarkets_Spring2024_FinalExam/data/sofr_swaps_symbology.xlsx')
    sofr_market = pd.read_excel('UChicago_FINM_35700_CreditMarkets_Spring2024_FinalExam/data/sofr_swaps_market_data_eod.xlsx')
    sofr_df = pd.merge(sofr_market, sofr_symbology, on='figi').drop_duplicates(subset=['tenor'])
    sofr_curve = cmt.calibrate_sofr_curve_from_frame(calc_date, sofr_df, 'midRate')
    sofr_handle = ql.YieldTermStructureHandle(sofr_curve)

    cds_market = pd.read_excel('UChicago_FINM_35700_CreditMarkets_Spring2024_FinalExam/data/cds_market_data_eod.xlsx')
    ford_cds_row = cds_market[cds_market['ticker'] == 'F'].iloc[0]
    tenors_yrs = [1, 2, 3, 5, 7, 10]
    ford_spreads = [ford_cds_row[f'par_spread_{t}y'] for t in tenors_yrs]

    ford_hazard_curve = cmt.calibrate_cds_hazard_rate_curve(calc_date, sofr_handle, ford_spreads)
    ford_hazard_handle = ql.DefaultProbabilityTermStructureHandle(ford_hazard_curve)

    cds_coupon = 100.0 / 10000.0
    cds_maturity = ql.Date(20, 6, 2029)
    notional = 10000000.0
    side = ql.Protection.Buyer
    schedule = ql.MakeSchedule(effectiveDate=calc_date, terminationDate=cds_maturity, tenor=ql.Period(3, ql.Months), calendar=ql.UnitedStates(ql.UnitedStates.GovernmentBond), convention=ql.Following, terminalDateConvention=ql.Unadjusted, rule=ql.DateGeneration.TwentiethIMM, endOfMonth=False)
    cds_obj = ql.CreditDefaultSwap(side, notional, cds_coupon, schedule, ql.Following, ql.Actual360())
    cds_engine = ql.MidPointCdsEngine(ford_hazard_handle, 0.4, sofr_handle)
    cds_obj.setPricingEngine(cds_engine)

    cds_pv = cds_obj.NPV()
    return (
        cds_coupon,
        cds_engine,
        cds_market,
        cds_maturity,
        cds_obj,
        cds_pv,
        ford_cds_row,
        ford_hazard_curve,
        ford_hazard_handle,
        ford_spreads,
        notional,
        schedule,
        side,
        sofr_curve,
        sofr_df,
        sofr_handle,
        sofr_market,
        sofr_symbology,
        tenors_yrs,
    )


@app.cell
def __(calc_date, cmt, pd, ql):
    # --- Problem 5 ---
    lqd_comp = pd.read_excel('UChicago_FINM_35700_CreditMarkets_Spring2024_FinalExam/data/lqd_basket_composition.xlsx')
    lqd_symb = pd.read_excel('UChicago_FINM_35700_CreditMarkets_Spring2024_FinalExam/data/lqd_corp_symbology.xlsx')
    lqd_merged = pd.merge(lqd_comp, lqd_symb, on='isin', how='left')

    bond_dv01s = []
    basket_dv01s = []

    for idx, row in lqd_merged.iterrows():
        try:
            bond = cmt.create_bond_from_symbology(row)
            y_5 = row['yield'] / 100.0
            dc = bond.dayCounter()
            ir = ql.InterestRate(y_5, dc, ql.Compounded, ql.Semiannual)
            price_5 = bond.cleanPrice(y_5, dc, ql.Compounded, ql.Semiannual, calc_date)
            mod_dur = ql.BondFunctions.duration(bond, ir, ql.Duration.Modified, calc_date)
            b_dv01 = price_5 * mod_dur * 0.0001
            basket_dv01 = b_dv01 * row['face_notional'] / 100.0
            bond_dv01s.append(b_dv01)
            basket_dv01s.append(basket_dv01)
        except:
            bond_dv01s.append(0)
            basket_dv01s.append(0)

    lqd_merged['bond_DV01'] = bond_dv01s
    lqd_merged['basket_DV01'] = basket_dv01s
    return (
        b_dv01,
        basket_dv01,
        basket_dv01s,
        bond,
        bond_dv01s,
        dc,
        idx,
        ir,
        lqd_comp,
        lqd_merged,
        lqd_symb,
        mod_dur,
        price_5,
        row,
        y_5,
    )


@app.cell
def __(
    as_of_date,
    bond_market_prices_eod,
    bond_symbology,
    calc_date,
    calculate_nelson_siegel_model_prices_and_yields_fixed,
    calibrate_nelson_siegel_model_fixed,
    calibrate_yield_curve_from_frame_stable,
    create_bonds_and_weights_fixed,
    pd,
    ql,
):
    # --- Problem 6 ---
    govt_symb = pd.read_excel('UChicago_FINM_35700_CreditMarkets_Spring2024_FinalExam/data/govt_on_the_run.xlsx')
    govt_symb_full = pd.merge(govt_symb[['isin']], bond_symbology, on='isin', how='left')
    govt_merged = pd.merge(govt_symb_full, bond_market_prices_eod[['isin', 'midPrice']], on='isin')
    tsy_curve = calibrate_yield_curve_from_frame_stable(calc_date, govt_merged, 'midPrice')
    tsy_handle = ql.YieldTermStructureHandle(tsy_curve)

    full_universe = pd.merge(bond_symbology, bond_market_prices_eod, on='isin')
    if 'ticker_x' in full_universe.columns:
        full_universe.rename(columns={'ticker_x': 'ticker'}, inplace=True)

    orcl_bonds = full_universe[
        (full_universe['ticker'] == 'ORCL') &
        (full_universe['cpn_type'] == 'FIXED') &
        (full_universe['amt_out'] > 100)
    ].copy()

    if 'class_x' in orcl_bonds.columns:
        orcl_bonds.rename(columns={'class_x': 'class'}, inplace=True)

    for col in ['days_settle', 'coupon', 'start_date', 'maturity', 'acc_first', 'cpn_freq']:
        if f'{col}_x' in orcl_bonds.columns:
            orcl_bonds.rename(columns={f'{col}_x': col}, inplace=True)

    orcl_bonds['maturity_dt'] = pd.to_datetime(orcl_bonds['maturity'])
    orcl_bonds['TTM'] = (orcl_bonds['maturity_dt'] - as_of_date).dt.days / 365.25
    orcl_bonds.sort_values('TTM', inplace=True)

    init_params = [0.01, 0.01, 0.01, 1.0]
    try:
        res, valid_indices = calibrate_nelson_siegel_model_fixed(init_params, calc_date, orcl_bonds, tsy_handle)
        opt_params = res.x

        orcl_bonds_valid = orcl_bonds.loc[valid_indices].copy()
        fixed_bonds, weights, mkt_prices, mkt_yields, dv01s, durs, _ = create_bonds_and_weights_fixed(orcl_bonds_valid, tsy_handle)
        mod_prices, mod_yields = calculate_nelson_siegel_model_prices_and_yields_fixed(opt_params, calc_date, fixed_bonds, tsy_handle)

        orcl_bonds_valid['modelPrice'] = mod_prices
        orcl_bonds_valid['modelYield'] = mod_yields

        print("NS Calibrated")
    except Exception as e:
        print(f"NS Calibration Error: {e}")
    return (
        col,
        durs,
        dv01s,
        fixed_bonds,
        full_universe,
        govt_merged,
        govt_symb,
        govt_symb_full,
        init_params,
        mkt_prices,
        mkt_yields,
        mod_prices,
        mod_yields,
        opt_params,
        orcl_bonds,
        orcl_bonds_valid,
        res,
        tsy_curve,
        tsy_handle,
        valid_indices,
        weights,
    )


if __name__ == "__main__":
    app.run()
