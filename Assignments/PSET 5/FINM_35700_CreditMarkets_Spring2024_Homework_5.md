# Homework 5

## FINM 35700 - Spring 2024

### UChicago Financial Mathematics

### Due Date: 2024-05-05

* Alex Popovici
* alex.popovici@uchicago.edu

This homework relies on following symbology & data files, as of 2024-04-26.

HYG ETF corporate bonds:
- the HYG bond symbology file `hyg_bond_symbology` and
- the HYG basket composition file (containing bond weights and yields) `hyg_basket_composition`.


## Scoring: Total of 100 points

| Problem | Points |
|---------|--------|
| 1       | 20     |
| 2       | 20     |
| 3       | 30     |
| 4       | 30     |


```python
import QuantLib as ql
import numpy as np
import pandas as pd
import datetime as dt
from scipy.optimize import minimize

def get_ql_date(date) -> ql.Date:
    """
    convert dt.date to ql.Date
    """
    if isinstance(date, dt.date):
        return ql.Date(date.day, date.month, date.year)
    elif isinstance(date, str):
        date = dt.datetime.strptime(date, "%Y-%m-%d").date()
        return ql.Date(date.day, date.month, date.year)
    else:
        raise ValueError(f"to_qldate, {type(date)}, {date}")
    
def create_schedule_from_symbology(details: dict):
    '''Create a QuantLib cashflow schedule from symbology details dictionary (usually one row of the symbology dataframe)
    '''
    
    # Create maturity from details['maturity']
    maturity = get_ql_date(details['maturity'])
    
    # Create acc_first from details['acc_first']
    acc_first =  get_ql_date(details['acc_first'])
    
    # Create calendar for Corp and Govt asset classes
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    
    # define period from details['cpn_freq'] ... can be hard-coded to 2 = semi-annual frequency
    period = ql.Period(2)
    
    # business_day_convention
    business_day_convention = ql.Unadjusted
    
    # termination_date_convention
    termination_date_convention = ql.Unadjusted
    
    # date_generation
    date_generation=ql.DateGeneration.Backward
    
    # Create schedule using ql.MakeSchedule interface (with keyword arguments)
    schedule = ql.MakeSchedule(effectiveDate=acc_first,  # this may not be the same as the bond's start date
                            terminationDate=maturity,
                            tenor=period,
                            calendar=calendar,
                            convention=business_day_convention,
                            terminalDateConvention=termination_date_convention,
                            rule=date_generation,
                            endOfMonth=True,
                            firstDate=ql.Date(),
                            nextToLastDate=ql.Date())
    return schedule

def create_bond_from_symbology(details: dict):
    '''Create a US fixed rate bond object from symbology details dictionary (usually one row of the symbology dataframe)
    '''
    
     # Create day_count from details['dcc']
     # For US Treasuries use ql.ActualActual(ql.ActualActual.ISMA)
     # For US Corporate bonds use ql.Thirty360(ql.Thirty360.USA)
    
    if details['class'] == 'Corp':
        day_count = ql.Thirty360(ql.Thirty360.USA)
    elif details['class'] == 'Govt':
        day_count = ql.ActualActual(ql.ActualActual.ISMA)
    else:
        raise ValueError(f"unsupported asset class, {type(details['class'])}, {details['class']}")

    
    # Create issue_date from details['start_date']
    issue_date = get_ql_date(details['start_date'])
    
    # Create days_settle from details['days_settle']
    days_settle = int(float(details['days_settle']))

    # Create coupon from details['coupon']
    coupon = float(details['coupon'])/100.


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
        issue_date)        

    return fixed_rate_bond

def get_bond_cashflows(bond: ql.FixedRateBond, calc_date=ql.Date) -> pd.DataFrame:
    '''Returns all future cashflows as of calc_date, i.e. with payment dates > calc_date.
    '''    
    day_counter = bond.dayCounter()    
    
    x = [(cf.date(), day_counter.yearFraction(calc_date, cf.date()), cf.amount()) for cf in bond.cashflows()]
    cf_date, cf_yearFrac, cf_amount = zip(*x)
    cashflows_df = pd.DataFrame(data={'CashFlowDate': cf_date, 'CashFlowYearFrac': cf_yearFrac, 'CashFlowAmount': cf_amount})

    # filter for payment dates > calc_date
    cashflows_df = cashflows_df[cashflows_df.CashFlowYearFrac > 0]
    return cashflows_df


def calibrate_yield_curve_from_frame(
        calc_date: ql.Date,
        treasury_details: pd.DataFrame,
        price_quote_column: str):
    '''Create a calibrated yield curve from a details dataframe which includes bid/ask/mid price quotes.
    '''
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
    # yield_curve = ql.PiecewiseFlatForward(calc_date, bond_helpers, day_count)
    
    yield_curve.enableExtrapolation()
    return yield_curve



def get_yield_curve_details_df(yield_curve, curve_dates=None):
    
    if(curve_dates == None):
        curve_dates = yield_curve.dates()

    dates = [d.to_date() for d in curve_dates]
    discounts = [round(yield_curve.discount(d), 3) for d in curve_dates]
    yearfracs = [round(yield_curve.timeFromReference(d), 3) for d in curve_dates]
    zeroRates = [round(yield_curve.zeroRate(d, yield_curve.dayCounter(), ql.Compounded).rate() * 100, 3) for d in curve_dates]

    yield_curve_details_df = pd.DataFrame(data={'Date': dates,
                             'YearFrac': yearfracs,
                             'DiscountFactor': discounts,
                             'ZeroRate': zeroRates})                             
    return yield_curve_details_df


def calc_clean_price_with_zspread(fixed_rate_bond, yield_curve_handle, zspread):
    zspread_quote = ql.SimpleQuote(zspread)
    zspread_quote_handle = ql.QuoteHandle(zspread_quote)
    yield_curve_bumped = ql.ZeroSpreadedTermStructure(yield_curve_handle, zspread_quote_handle, ql.Compounded, ql.Semiannual)
    yield_curve_bumped_handle = ql.YieldTermStructureHandle(yield_curve_bumped)
    
    # Set Valuation engine
    bond_engine = ql.DiscountingBondEngine(yield_curve_bumped_handle)
    fixed_rate_bond.setPricingEngine(bond_engine)
    bond_clean_price = fixed_rate_bond.cleanPrice()
    return bond_clean_price


def calibrate_sofr_curve_from_frame(
        calc_date: ql.Date,
        sofr_details: pd.DataFrame,
        rate_quote_column: str):
    '''Create a calibrated yield curve from a SOFR details dataframe which includes rate quotes.
    '''
    ql.Settings.instance().evaluationDate = calc_date

    # Sort dataframe by maturity
    sorted_details_frame = sofr_details.sort_values(by='tenor')    
    
    # settle_days
    settle_days = 2
    
    # For US SOFR OIS Swaps 
    day_count = ql.Actual360()

    # For US SOFR Swaps     
    calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
    
    sofr_helpers = []
    
    for index, row in sorted_details_frame.iterrows():
        sofr_quote = row[rate_quote_column]
        tenor_in_years = row['tenor']
        sofr_tenor = ql.Period(tenor_in_years, ql.Years)
        
        # create sofr_rate_helper
        sofr_helper = ql.OISRateHelper(settle_days, sofr_tenor, ql.QuoteHandle(ql.SimpleQuote(sofr_quote/100)), ql.Sofr())
                        
        sofr_helpers.append(sofr_helper)
        
    sofr_yield_curve = ql.PiecewiseLinearZero(settle_days, calendar, sofr_helpers, day_count)
    sofr_yield_curve.enableExtrapolation()
    
    return sofr_yield_curve


def calibrate_cds_hazard_rate_curve(calc_date, sofr_yield_curve_handle, cds_par_spreads_bps, cds_recovery_rate = 0.4):
    '''Calibrate hazard rate curve from CDS Par Spreads'''
    CDS_settle_days = 2

    CDS_day_count = ql.Actual360()

    # CDS standard tenors: 1Y, 2Y, 3Y, 5Y 7Y and 10Y
    CDS_tenors = [ql.Period(y, ql.Years) for y in [1, 2, 3, 5, 7, 10]]
              

    CDS_helpers = [ql.SpreadCdsHelper((cds_par_spread / 10000.0), CDS_tenor, CDS_settle_days, ql.TARGET(),
                                  ql.Quarterly, ql.Following, ql.DateGeneration.TwentiethIMM, CDS_day_count, cds_recovery_rate, sofr_yield_curve_handle)
               
    for (cds_par_spread, CDS_tenor) in zip(cds_par_spreads_bps, CDS_tenors)]

    # bootstrap hazard_rate_curve
    hazard_rate_curve = ql.PiecewiseFlatHazardRate(calc_date, CDS_helpers, CDS_day_count)
    hazard_rate_curve.enableExtrapolation()

    return(hazard_rate_curve)


def get_hazard_rates_df(hazard_rate_curve):
    '''Return dataframe with calibrated hazard rates and survival probabilities'''
    
    CDS_day_count = ql.Actual360()
    
    hazard_list = [(hr[0].to_date(), 
                CDS_day_count.yearFraction(calc_date, hr[0]),
                hr[1] * 1e4,
                hazard_rate_curve.survivalProbability(hr[0])) for hr in hazard_rate_curve.nodes()]

    grid_dates, year_frac, hazard_rates, surv_probs = zip(*hazard_list)

    hazard_rates_df = pd.DataFrame(data={'Date': grid_dates, 
                                     'YearFrac': year_frac,
                                     'HazardRateBps': hazard_rates,                                     
                                     'SurvivalProb': surv_probs})
    return(hazard_rates_df)


def nelson_siegel(params, maturity):
    ''' params = (theta1, theta2, theta3, lambda)'''        
    if(maturity == 0 or params[3] <= 0):
        slope_1 = 1
        curvature = 0
    else:
        slope_1 = (1 - np.exp(-maturity/params[3]))/(maturity/params[3])
        curvature = slope_1 - np.exp(-maturity/params[3])

    total_value = params[0] + params[1] * slope_1 + params[2] * curvature
    
    return total_value

def create_nelson_siegel_curve(calc_date, nelson_siegel_params):
    ''' nelson_siegel_params = (theta1, theta2, theta3, lambda)'''            
    nelson_siegel_surv_prob_dates = [calc_date + ql.Period(T , ql.Years) for T in range(31)]
    nelson_siegel_average_hazard_rates = [nelson_siegel(nelson_siegel_params, T) for T in range(31)]
    nelson_siegel_surv_prob_levels = [np.exp(-T * nelson_siegel_average_hazard_rates[T]) for T in range(31)]
    
    # cap and floor survival probs
    nelson_siegel_surv_prob_levels = [max(min(x,1),1e-8) for x in nelson_siegel_surv_prob_levels]

    # nelson_siegel_surv_prob_curve
    nelson_siegel_credit_curve = ql.SurvivalProbabilityCurve(nelson_siegel_surv_prob_dates, nelson_siegel_surv_prob_levels, ql.Actual360(), ql.TARGET())
    nelson_siegel_credit_curve.enableExtrapolation()
    nelson_siegel_credit_curve_handle = ql.DefaultProbabilityTermStructureHandle(nelson_siegel_credit_curve)
    
    return(nelson_siegel_credit_curve_handle)


def calculate_nelson_siegel_model_prices_and_yields(nelson_siegel_params, 
                      calc_date, 
                      fixed_rate_bond_objects, 
                      tsy_yield_curve_handle, 
                      bond_recovery_rate = 0.4):
    
    # nelson_siegel_surv_prob_curve_handle
    nelson_siegel_surv_prob_curve_handle = create_nelson_siegel_curve(calc_date, nelson_siegel_params)
    
    # nelson_siegel_risky_bond_engine
    nelson_siegel_risky_bond_engine = ql.RiskyBondEngine(nelson_siegel_surv_prob_curve_handle, bond_recovery_rate, tsy_yield_curve_handle)
    
    bond_model_prices = []
    bond_model_yields = []
    
    for fixed_rate_bond in fixed_rate_bond_objects:
        fixed_rate_bond.setPricingEngine(nelson_siegel_risky_bond_engine)
        
        bond_price = fixed_rate_bond.cleanPrice()                
        bond_yield = fixed_rate_bond.bondYield(bond_price, ql.Thirty360(ql.Thirty360.USA), ql.Compounded, ql.Semiannual) * 100
        
        bond_model_prices.append(bond_price)
        bond_model_yields.append(bond_yield)
    
    return(bond_model_prices, bond_model_yields)

def nelson_siegel_sse(nelson_siegel_params, 
                      calc_date, 
                      fixed_rate_bond_objects, 
                      market_prices, 
                      calib_weights,
                      tsy_yield_curve_handle, 
                      bond_recovery_rate = 0.4):
    
    # bond_model_prices
    bond_model_prices, bond_model_yields = calculate_nelson_siegel_model_prices_and_yields(nelson_siegel_params, 
                      calc_date, 
                      fixed_rate_bond_objects, 
                      tsy_yield_curve_handle, 
                      bond_recovery_rate)
    # sse    
    sse = 0
    
    for i in range(len(market_prices)):
        model_error = market_prices[i] - bond_model_prices[i]                
        sse += model_error * model_error * calib_weights[i]                        
    
    return(sse)    


def create_bonds_and_weights(bond_details, tsy_yield_curve_handle):
    
    # risk_free_bond_engine
    risk_free_bond_engine = ql.DiscountingBondEngine(tsy_yield_curve_handle)


    fixed_rate_bond_objects = []
    bond_market_prices = []    
    bond_yields = []
    bond_DV01s = []    
    bond_durations = []    
    
    for index,row in bond_details.iterrows():
        fixed_rate_bond = create_bond_from_symbology(row)
        fixed_rate_bond.setPricingEngine(risk_free_bond_engine)
        
        fixed_rate_bond_objects.append(fixed_rate_bond)
        
        bond_price = row['midPrice']                
        bond_yield = fixed_rate_bond.bondYield(bond_price, ql.Thirty360(ql.Thirty360.USA), ql.Compounded, ql.Semiannual) * 100
        bond_yield_rate = ql.InterestRate(bond_yield/100, ql.ActualActual(ql.ActualActual.ISMA), ql.Compounded, ql.Semiannual)
        bond_duration = ql.BondFunctions.duration(fixed_rate_bond, bond_yield_rate)
        bond_DV01   = fixed_rate_bond.dirtyPrice() * bond_duration
        
        bond_market_prices.append(bond_price)
        bond_yields.append(bond_yield)
        bond_DV01s.append(bond_DV01)
        bond_durations.append(bond_duration)   
             
    calib_weights = [1 / d for d in bond_DV01s]
    
    sum_calib_weights = sum(calib_weights)
    calib_weights = [x / sum_calib_weights for x in calib_weights]
    
    return(fixed_rate_bond_objects, calib_weights, bond_market_prices, bond_yields, bond_DV01s, bond_durations)



def calibrate_nelson_siegel_model(initial_nelson_siegel_params,
                                  calc_date, 
                                  bond_details, 
                                  tsy_yield_curve_handle, 
                                  bond_recovery_rate = 0.4):
    # create_bonds_and_weights
    fixed_rate_bond_objects, calib_weights, bond_market_prices, bond_yields, bond_DV01s, bond_durations = create_bonds_and_weights(bond_details, tsy_yield_curve_handle)
    
    # start calibration
    param_bounds = [(1e-3, 0.1), (-0.1, 0.1), (-0.1, 0.1), (1e-3, 10)]
            
    calib_results = minimize(nelson_siegel_sse,
                                            initial_nelson_siegel_params, 
                                            args = (calc_date, 
                                                    fixed_rate_bond_objects, 
                                                    bond_market_prices, 
                                                    calib_weights,
                                                    tsy_yield_curve_handle, 
                                                    bond_recovery_rate),
                                            bounds = param_bounds)


    return(calib_results)



```


```python

```


```python
# import tools from previous homeworks


# Use static calculation/valuation date of 2024-04-26, matching data available in the market prices EOD file
calc_date = ql.Date(26, 4, 2024)
ql.Settings.instance().evaluationDate = calc_date

# Calculation/valuation date as pd datetime
as_of_date = pd.to_datetime('2024-04-26')
```

-----------------------------------------------------------
# Problem 1: Fixed rate bond prices and sensitivities (bond yield model)

## When computing sensitivities, assume "everything else being equal" (ceteris paribus).

For a better understanding of dependencies, you can use the fixed rate bond valuation formula in the flat yield model (formula [6] in Lecture 1).

\begin{align}
PV_{Bond}\left(c,T,y_{sa} \right)=1+\frac{c-y_{sa}}{y_{sa}}\cdot\left[1-\left(1+\frac{y_{sa}}{2}\right)^{-2T}\right]
\end{align}


## a. True or False (fixed rate bond prices)

1. Fixed rate bond price is increasing in yield
   1. False, a fixed rate bond's price decreases as yields increase.
2. Fixed rate bond price is increasing in coupon
   1. True, a higher coupon increases the cash flows accruing to a bond, which raises its present value.
3. Fixed rate bond price is increasing in bond maturity
   1. True, the longer a fixed rate bond's maturity, the more coupon payments it makes, and the higher the present value of its cash flows.
4. Fixed rate callable bond prices are higher or equal to their "bullet" (non-callable) version.
   1. False, fixed rate callable bonds are sold at a discount given the risk of being called early, which would leave the buyer with fewer cash flows and expose them to early reinvestment risk. 

## b. True or False (fixed rate bond yields)

1. Fixed rate bond yield is increasing in interest rate
   1. False, all else equal, a rise in interest rates reduces the yield on the fixed rate bond.
2. Fixed rate bond yield is increasing in credit spread
   1. Yes, a higher credit spread means that the bond earns a higher yield over the risk free rate.
3. Fixed rate bond yield is increasing in coupon
   1. yes, an increase in the coupon increases the PV of the bond, which increases yield.
4. Fixed rate bond yield is increasing in bond maturity
   1. False. Investors demand a higher return for holding bonds for a longer term, which lowers the price of the bond and drives down the yield. 
5. Fixed rate callable bond yields are lower or equal to their "bullet" (non-callable) version.
   1. Because fixed rate callable bonds are more risky, they command a higher yield, since the holder is exposed to reinvestment risk if the bond is exercised early. This would 

## c. True or False (fixed rate bond durations)

1. Fixed rate bond duration is increasing with yield
   1. False. In reality, the duration of a fixed-rate bond is inversely related to yield. That means, if yield increases, the duration decreases.
2. Fixed rate bond duration is increasing in coupon
   1. False. Higher coupon rates, all else being equal, actually result in lower bond durations because more of the bond's total return is being received earlier through the higher coupon payments.
3. Fixed rate bond duration is increasing with bond maturity
   1. True. The duration of a fixed-rate bond typically increases with its maturity. Longer-dated bonds have a longer time before their final cash flows, hence their durations are longer.
4. Fixed rate callable bond durations are higher or equal to their "bullet" (non-callable) version .
   1. False. Callable bonds tend to have lower durations than their non-callable counterparts. This is because the call option shortens the expected life of the bond, thus reducing its duration.

## d. True or False (fixed rate bond convexities)

1. Fixed rate bond convexity is increasing with yield
   1. False. For fixed-rate bonds, convexity tends to decrease as yields rise. At higher yield levels, the price-yield curve flattens out, reducing the bond's convexity measure.

2. Fixed rate bond convexity is increasing in coupon
   1. False. Bonds with higher coupon rates generally have lower convexity compared to low coupon bonds. This is because the higher coupon cash flows are received earlier, diminishing the compounding effect and convexity.

3. Fixed rate bond convexity is increasing with bond maturity
   1. True. Convexity for fixed-rate bonds usually increases with longer maturities. Bonds with longer maturities are more sensitive to changes in yields, resulting in higher convexity measures.

4. Fixed rate callable bond convexities are higher or equal to their "bullet" (non-callable) version .
   1. False. Callable bonds exhibit lower convexity than non-callable bonds. The issuer's option to call the bond limits its price appreciation when yields fall, decreasing convexity.



-----------------------------------------------------------
# Problem 2: Credit Default Swaps (hazard rate model)

## When computing sensitivities, assume "everything else being equal" (ceteris paribus).

For a better understanding of dependencies, you can use the CDS valuation formulas in the simple hazard rate model (formulas[43] and [44] in Lecture 3).

\begin{align}
PV_{CDS\_PL}\left(c,r,h,R,T\right) = \frac{c}{4 \cdot \left(e^{\left(r+h\right)/4}-1 \right)} \cdot\left[1-e^{-T\cdot\left(r+h\right)}\right] \simeq \frac{c}{r+h} \cdot\left[1-e^{-T\cdot\left(r+h\right)}\right]
\end{align}

\begin{align}
PV_{CDS\_DL}\left(c,r,h,R,T\right) = \frac{\left(1-R\right)\cdot h}{r+h} \cdot\left[1-e^{-T\cdot\left(r+h\right)}\right]
\end{align}

\begin{align}
PV_{CDS} = PV_{CDS\_PL} - PV_{CDS\_DL} \simeq \frac{c - \left(1-R\right)\cdot h}{r+h} \cdot\left[1-e^{-T\cdot\left(r+h\right)}\right]
\end{align}

\begin{align}
CDS\_ParSpread = c \cdot \frac{PV_{CDS\_DL}}{PV_{CDS\_PL}} \simeq \left(1-R\right)\cdot h
\end{align}


## a. True or False (CDS Premium Leg PV)

1. CDS premium leg PV is increasing in CDS Par Spread
   1. True. For a credit default swap (CDS), the present value of premiums paid increases as the CDS spread rises. Higher spreads mean the protection buyer pays larger premiums over the life of the contract.
2. CDS premium leg PV is increasing in interest rate
   1. False. An increase in interest rates decreases the present value of CDS premium payments, since future payments are discounted more heavily.
3. CDS premium leg PV is increasing in hazard rate
   1. True. A higher hazard rate, indicating greater default risk, raises the present value of CDS premiums since larger premiums compensate for this risk.
4. CDS premium leg PV is increasing in recovery rate
   1. False. Higher recovery rates reduce the net payout on a CDS, so the buyer pays lower premiums, decreasing their present value.
5. CDS premium leg PV is increasing in coupon
   1. True. Larger coupon rates on the CDS directly increase the regular premium amounts paid, raising their present value.
6. CDS premium leg PV is increasing in CDS maturity
   1.  True. Longer CDS maturities generally increase the present value of premiums paid, since there are more premium payments over the extended duration. However, discounting effects make this relationsh

## b. True or False (CDS Default Leg PV)

1. CDS default leg PV is increasing in CDS Par Spread
   1. False. The present value of the CDS default leg is not directly impacted by the premium (spread) amount paid. It mainly depends on the default probability, loss given default, and discount rates.
2. CDS default leg PV is increasing in interest rate
   1. False. Higher interest rates reduce the present value of potential default payouts on a CDS by discounting them more heavily.
3. CDS default leg PV is increasing in hazard rate
   1. True. As the hazard/default risk increases, so does the expected payout to the CDS buyer in a default event, raising the present value of the default leg.
4. CDS default leg PV is increasing in recovery rate
   1. False. Higher recovery rates lower the net loss on a default, decreasing the expected payout and present value of the CDS default leg.
5. CDS default leg PV is increasing in coupon
   1. False. The regular coupon rate on CDS premiums does not directly impact the present value of potential default payouts, which is the default leg.
6. CDS default leg PV is increasing in CDS maturity
   1. True. Longer CDS contract maturities increase the time period over which a default can occur, raising the expected payout and present value of the default leg. But discounting effects make this relationship non-linear.

## c. True or False (CDS PV)


1. CDS PV is increasing in CDS Par Spread
   1. False. The total present value of a CDS does not necessarily increase with the premium spread paid. Higher spreads raise the premium leg value but may also signal higher default risk impacting the default leg value. The net effect depends on these offsetting factors.
2. CDS PV is increasing in interest rate
   1. False. Interest rate changes have mixed effects on the total CDS present value by impacting the premium and default leg values in opposite ways through discounting effects.

3. CDS PV is increasing in hazard rate
   1. False. Higher hazard/default risk raises both the premium leg value (more premiums paid) and default leg value (higher expected payouts). The net impact on total CDS present value depends on the relative magnitudes.
4. CDS PV is increasing in recovery rate
   1. False. Recovery rate changes affect the premium leg value (fewer premiums at higher recoveries) and default leg value (lower payouts) in opposite ways, so the net effect on total CDS present value is uncertain.
5. CDS PV is increasing in coupon
   1. False. While higher coupon rates directly increase the premium leg value, they do not impact the default leg value. So the net effect on total CDS present value is ambiguous.
6. CDS PV is increasing in CDS maturity
   1.  False. Longer maturities raise both the premium leg (more payments) and default leg (wider time window) values, but discounting effects make the net impact on total CDS present value unclear.

## d. True or False (CDS Par Spread)


1. CDS Par Spread is increasing in interest rates
   1. False. The CDS spread compensates for credit/default risk and is not directly driven by changes in interest rates.
2. CDS Par Spread is increasing in hazard rate
   1. True. As default risk increases, represented by a higher hazard rate, the CDS spread should rise to compensate for this risk.
3. CDS Par Spread is increasing in recovery rate
   1. False. Higher recovery rates signal lower losses given default, allowing for lower CDS spread levels
4. CDS Par Spread is increasing in coupon
   1. False. The CDS spread is delinked from the coupon rate on the underlying reference bond
5. CDS Par Spread is increasing in CDS maturity
   1. False. The relationship between CDS spread and maturity is complex, depending on the shape of the credit spread term structure which can slope upwards or downwards.

-----------------------------------------------------------
# Problem 3: Pricing bonds in the Merton Structural Credit Model
## Follow Lecture 5, "Structural Credit Default Models"

## a. Company balance sheet metrics & fair value of equity
- Assets of $125 MM
- Liabilities of $100 MM face value, consisting of one zero coupon bond.
- Bond maturity is at T = 5 years
- Asset volatility at 20% (log-normal annualized vol)
- Flat risk-free interest rates of 4%

Compute the company Leverage, "Book Value of Equity" and fair value of equity metrics.



```python
from scipy.stats import norm
import numpy as np

A0 = 125e6   # Total assets ($125 MM)
K = 100e6   # Face value of liabilities (debt) ($100 MM)
T = 5         # Time to maturity of the debt (years)
sigma = 0.20  # volatility (20%)
r = 0.04      # Risk-free interest rate (4%)

L = K / A0
BVE = A0 - K

d1 = (np.log(A0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)

E0 = A0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


print("Leverage:", L)
print("Book Value of Equity:", round(BVE,2))
print("Fair Value of Equity:", round(E0,2))
```

    Leverage: 0.8
    Book Value of Equity: 25000000.0
    Fair Value of Equity: 47234305.06


## b. Risky Bond Valuation (Fair Value of Liabilities)

Compute the fair value of the risky bond.


```python
B0 = A0 - E0
print("Fair Value of Liabilities:", round(B0,2))
```

    Fair Value of Liabilities: 77765694.94


## c. Flat yield, spread and hazard rate

Compute the following credit risk metrics:
- Distance to Default
- Default Probability
- Bond Yield
- Bond Credit Spread
- Flat Hazard Rate
- Expected Recovery on Default

Plot separate charts for 
- Bond Credit Spreads and 
- Expected Recovery on Defaults

as a function of initial Asset values, on a grid from $50 MM to $200 MM in steps of $5 MM.


```python
Default = d2
Default_Probability = norm.cdf(-d2)
bond_yield = -(1/T)*np.log((1/L)*norm.cdf(-d1)+np.exp(-r * T) * norm.cdf(d2))
spread = bond_yield - r
flat_hazard_rate = -(1/T)*np.log(norm.cdf(d1))
expected_recovery_default = A0/K

print("Distance to Default:",Default)
print("Default Probability:",Default_Probability)
print("Bond Yield:", bond_yield)
print("Bond Credit Spread:", spread,4)
print("Flat Hazard Rate:", flat_hazard_rate)
print("Expected Recovery on Default:", expected_recovery_default)
```

    Distance to Default: 0.7225709472292644
    Default Probability: 0.23497176139035986
    Bond Yield: 0.05029395821624366
    Bond Credit Spread: 0.01029395821624366 4
    Flat Hazard Rate: 0.025804052118242278
    Expected Recovery on Default: 1.25



```python
import plotly.express as px 
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Create a grid of initial Asset values from $50 MM to $200 MM in steps of $5 MM
A0_grid = np.arange(50e6, 205e6, 5e6)

# Calculate the credit risk metrics for each initial Asset value
Default_grid = []
Default_Probability_grid = []
bond_yield_grid = []
spread_grid = []
flat_hazard_rate_grid = []
expected_recovery_default_grid = []

for A0 in A0_grid:
    L = K / A0
    d1 = (np.log(A0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    Default_grid.append(d2)
    Default_Probability_grid.append(norm.cdf(-d2))
    bond_yield_grid.append(-(1/T)*np.log((1/L)*norm.cdf(-d1)+np.exp(-r * T) * norm.cdf(d2)))
    spread_grid.append(bond_yield_grid[-1] - r)
    flat_hazard_rate_grid.append(-(1/T)*np.log(norm.cdf(d1)))
    expected_recovery_default_grid.append(A0/K)

# Create the Bond Credit Spreads chart
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=A0_grid, y=spread_grid, mode='lines', name='Bond Credit Spreads'))
fig1.update_layout(title='Bond Credit Spreads vs. Initial Asset Values',
                   xaxis_title='Initial Asset Values',
                   yaxis_title='Bond Credit Spreads')

# Create the Expected Recovery on Defaults chart
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=A0_grid, y=expected_recovery_default_grid, mode='lines', name='Expected Recovery on Defaults'))
fig2.update_layout(title='Expected Recovery on Defaults vs. Initial Asset Values',
                   xaxis_title='Initial Asset Values',
                   yaxis_title='Expected Recovery on Defaults')

# Display the charts
fig1.show()
fig2.show()
```





## d. Equity volatility

Compute the Equity Volatility.

What happens to the equity volatility if initial Assets value goes up/down (as of time 0)?

Plot Equity Volatilities of initial Asset values, on a grid from $50 MM to $200 MM in steps of $5 MM.

 


```python
sigmaE = A0/E0*norm.cdf(d2)*sigma
print("Equity Volatility:", round(sigmaE,4))
```

    Equity Volatility: 0.8146



```python
import plotly.graph_objects as go

num = int((200-50)/5)+1
A0_grid = np.linspace(50, 200, num, endpoint=True)
d1_grid = (np.log(A0_grid / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2_grid = d1_grid - sigma * np.sqrt(T)
sigmaE_grid = A0_grid/E0*norm.cdf(d2_grid)*sigma

fig = go.Figure()
fig.add_trace(go.Scatter(x=A0_grid, y=sigmaE_grid, mode='lines', name='Equity Volatility'))
fig.update_layout(title='Equity Volatility vs. Initial Asset Values',
                  xaxis_title='Initial Asset Values',
                  yaxis_title='Equity Volatility')

fig.show()
```



-----------------------------------------------------------
# Problem 4: Credit ETF analysis on HYG

## a. Load and explore the HYG basket composition and market data

Load the `hyg_basket_composition` Excel file into a dataframe. It contains the HYG basket constituent face notionals, weights and yields-to-maturities as of 2024-04-26.

Load the `hyg_corp_symbology` Excel file into a dataframe. It contains the corporate bond details for HYG constituents.

How many corporate bonds are in the HYG basket?  What are the average and median face notionals for a bond?

How many unique tickers are in the HYG basket? What are the average and median face notionals for a ticker?

Compute mean, median and standard deviation of yields-to-maturity of bonds in the basket.


```python
# High Yield Corporate Bond ETF
hyg_corp_symbology = pd.read_excel('/Users/rogerlin/Downloads/UChicago_FINM_35700_CreditMarkets_Spring2024_Homework_5/data/hyg_corp_symbology.xlsx')
hyg_basket_composition = pd.read_excel('/Users/rogerlin/Downloads/UChicago_FINM_35700_CreditMarkets_Spring2024_Homework_5/data/hyg_basket_composition.xlsx')
hyg_corp_symbology.head()
hyg_basket_composition.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>holdings_date</th>
      <th>etf_ticker</th>
      <th>isin</th>
      <th>security</th>
      <th>issuer</th>
      <th>coupon</th>
      <th>maturity</th>
      <th>cpn_type</th>
      <th>class</th>
      <th>currency</th>
      <th>bidYield</th>
      <th>askYield</th>
      <th>midYield</th>
      <th>face_notional</th>
      <th>face_notional_weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-04-26</td>
      <td>2024-04-24</td>
      <td>HYG</td>
      <td>US25470MAG42</td>
      <td>DISH 11 3/4 11/15/27</td>
      <td>DISH NETWORK CORP 11.75 11/15/2027 144a (SECURED)</td>
      <td>1900-01-11 18:00:00</td>
      <td>2027-11-15</td>
      <td>FIXED</td>
      <td>CORP</td>
      <td>USD</td>
      <td>11.500</td>
      <td>11.027</td>
      <td>11.2635</td>
      <td>71988000</td>
      <td>0.466269</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-04-26</td>
      <td>2024-04-24</td>
      <td>HYG</td>
      <td>US62482BAA08</td>
      <td>MEDIND 3 7/8 04/01/29</td>
      <td>MEDLINE BORROWER LP 3.875 04/01/2029 144a (SEC...</td>
      <td>1900-01-03 21:00:00</td>
      <td>2029-04-01</td>
      <td>FIXED</td>
      <td>CORP</td>
      <td>USD</td>
      <td>6.379</td>
      <td>6.219</td>
      <td>6.2990</td>
      <td>68223000</td>
      <td>0.441883</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-04-26</td>
      <td>2024-04-24</td>
      <td>HYG</td>
      <td>US18912UAA07</td>
      <td>TIBX 9 09/30/29</td>
      <td>CLOUD SOFTWARE GROUP INC 9.0 09/30/2029 144a (...</td>
      <td>1900-01-09 00:00:00</td>
      <td>2029-09-30</td>
      <td>FIXED</td>
      <td>CORP</td>
      <td>USD</td>
      <td>10.117</td>
      <td>9.963</td>
      <td>10.0400</td>
      <td>60382000</td>
      <td>0.391097</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-04-26</td>
      <td>2024-04-24</td>
      <td>HYG</td>
      <td>US88632QAE35</td>
      <td>TIBX 6 1/2 03/31/29</td>
      <td>CLOUD SOFTWARE GROUP INC 6.5 03/31/2029 144a (...</td>
      <td>1900-01-06 12:00:00</td>
      <td>2029-03-31</td>
      <td>FIXED</td>
      <td>CORP</td>
      <td>USD</td>
      <td>7.901</td>
      <td>7.692</td>
      <td>7.7965</td>
      <td>61979000</td>
      <td>0.401440</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-04-26</td>
      <td>2024-04-24</td>
      <td>HYG</td>
      <td>US25461LAA08</td>
      <td>DTV 5 7/8 08/15/27</td>
      <td>DIRECTV FINANCING LLC 5.875 08/15/2027 144a (S...</td>
      <td>1900-01-05 21:00:00</td>
      <td>2027-08-15</td>
      <td>FIXED</td>
      <td>CORP</td>
      <td>USD</td>
      <td>8.380</td>
      <td>8.171</td>
      <td>8.2755</td>
      <td>56308000</td>
      <td>0.364709</td>
    </tr>
  </tbody>
</table>
</div>




```python
hyg_corp_symbology.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ticker</th>
      <th>class</th>
      <th>figi</th>
      <th>isin</th>
      <th>und_bench_isin</th>
      <th>security</th>
      <th>name</th>
      <th>type</th>
      <th>coupon</th>
      <th>cpn_type</th>
      <th>...</th>
      <th>start_date</th>
      <th>cpn_first</th>
      <th>acc_first</th>
      <th>maturity</th>
      <th>mty_typ</th>
      <th>rank</th>
      <th>amt_out</th>
      <th>country</th>
      <th>currency</th>
      <th>status</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AA</td>
      <td>Corp</td>
      <td>BBG00VYJS3B3</td>
      <td>US013822AE11</td>
      <td>US91282CKJ98</td>
      <td>AA 5 1/2 12/15/27</td>
      <td>ALCOA NEDERLAND HOLDING</td>
      <td>PRIV PLACEMENT</td>
      <td>5.500</td>
      <td>FIXED</td>
      <td>...</td>
      <td>2020-07-13</td>
      <td>2020-12-15</td>
      <td>2020-07-13</td>
      <td>2027-12-15</td>
      <td>CALLABLE</td>
      <td>Sr Unsecured</td>
      <td>750.0</td>
      <td>NE</td>
      <td>USD</td>
      <td>ACTV</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AA</td>
      <td>Corp</td>
      <td>BBG00KXCGK19</td>
      <td>US013822AC54</td>
      <td>US91282CKG59</td>
      <td>AA 6 1/8 05/15/28</td>
      <td>ALCOA NEDERLAND HOLDING</td>
      <td>PRIV PLACEMENT</td>
      <td>6.125</td>
      <td>FIXED</td>
      <td>...</td>
      <td>2018-05-17</td>
      <td>2018-11-15</td>
      <td>2018-05-17</td>
      <td>2028-05-15</td>
      <td>CALLABLE</td>
      <td>Sr Unsecured</td>
      <td>500.0</td>
      <td>NE</td>
      <td>USD</td>
      <td>ACTV</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AA</td>
      <td>Corp</td>
      <td>BBG00ZKWG886</td>
      <td>US013822AG68</td>
      <td>US91282CKG59</td>
      <td>AA 4 1/8 03/31/29</td>
      <td>ALCOA NEDERLAND HOLDING</td>
      <td>PRIV PLACEMENT</td>
      <td>4.125</td>
      <td>FIXED</td>
      <td>...</td>
      <td>2021-03-24</td>
      <td>2021-09-30</td>
      <td>2021-03-24</td>
      <td>2029-03-31</td>
      <td>CALLABLE</td>
      <td>Sr Unsecured</td>
      <td>500.0</td>
      <td>NE</td>
      <td>USD</td>
      <td>ACTV</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AA</td>
      <td>Corp</td>
      <td>BBG01LW38YD3</td>
      <td>US013822AH42</td>
      <td>US91282CKG59</td>
      <td>AA 7 1/8 03/15/31</td>
      <td>ALCOA NEDERLAND HOLDING</td>
      <td>PRIV PLACEMENT</td>
      <td>7.125</td>
      <td>FIXED</td>
      <td>...</td>
      <td>2024-03-21</td>
      <td>2024-09-15</td>
      <td>2024-03-21</td>
      <td>2031-03-15</td>
      <td>CALLABLE</td>
      <td>Sr Unsecured</td>
      <td>750.0</td>
      <td>NE</td>
      <td>USD</td>
      <td>ACTV</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AAL</td>
      <td>Corp</td>
      <td>BBG01F5TZPJ2</td>
      <td>US023771T329</td>
      <td>US91282CKJ98</td>
      <td>AAL 7 1/4 02/15/28</td>
      <td>AMERICAN AIRLINES INC</td>
      <td>PRIV PLACEMENT</td>
      <td>7.250</td>
      <td>FIXED</td>
      <td>...</td>
      <td>2023-02-15</td>
      <td>2023-08-15</td>
      <td>2023-02-15</td>
      <td>2028-02-15</td>
      <td>CALLABLE</td>
      <td>1st lien</td>
      <td>750.0</td>
      <td>US</td>
      <td>USD</td>
      <td>ACTV</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>




```python
hyg_basket_composition.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>holdings_date</th>
      <th>etf_ticker</th>
      <th>isin</th>
      <th>security</th>
      <th>issuer</th>
      <th>coupon</th>
      <th>maturity</th>
      <th>cpn_type</th>
      <th>class</th>
      <th>currency</th>
      <th>bidYield</th>
      <th>askYield</th>
      <th>midYield</th>
      <th>face_notional</th>
      <th>face_notional_weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2024-04-26</td>
      <td>2024-04-24</td>
      <td>HYG</td>
      <td>US25470MAG42</td>
      <td>DISH 11 3/4 11/15/27</td>
      <td>DISH NETWORK CORP 11.75 11/15/2027 144a (SECURED)</td>
      <td>1900-01-11 18:00:00</td>
      <td>2027-11-15</td>
      <td>FIXED</td>
      <td>CORP</td>
      <td>USD</td>
      <td>11.500</td>
      <td>11.027</td>
      <td>11.2635</td>
      <td>71988000</td>
      <td>0.466269</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2024-04-26</td>
      <td>2024-04-24</td>
      <td>HYG</td>
      <td>US62482BAA08</td>
      <td>MEDIND 3 7/8 04/01/29</td>
      <td>MEDLINE BORROWER LP 3.875 04/01/2029 144a (SEC...</td>
      <td>1900-01-03 21:00:00</td>
      <td>2029-04-01</td>
      <td>FIXED</td>
      <td>CORP</td>
      <td>USD</td>
      <td>6.379</td>
      <td>6.219</td>
      <td>6.2990</td>
      <td>68223000</td>
      <td>0.441883</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2024-04-26</td>
      <td>2024-04-24</td>
      <td>HYG</td>
      <td>US18912UAA07</td>
      <td>TIBX 9 09/30/29</td>
      <td>CLOUD SOFTWARE GROUP INC 9.0 09/30/2029 144a (...</td>
      <td>1900-01-09 00:00:00</td>
      <td>2029-09-30</td>
      <td>FIXED</td>
      <td>CORP</td>
      <td>USD</td>
      <td>10.117</td>
      <td>9.963</td>
      <td>10.0400</td>
      <td>60382000</td>
      <td>0.391097</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2024-04-26</td>
      <td>2024-04-24</td>
      <td>HYG</td>
      <td>US88632QAE35</td>
      <td>TIBX 6 1/2 03/31/29</td>
      <td>CLOUD SOFTWARE GROUP INC 6.5 03/31/2029 144a (...</td>
      <td>1900-01-06 12:00:00</td>
      <td>2029-03-31</td>
      <td>FIXED</td>
      <td>CORP</td>
      <td>USD</td>
      <td>7.901</td>
      <td>7.692</td>
      <td>7.7965</td>
      <td>61979000</td>
      <td>0.401440</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2024-04-26</td>
      <td>2024-04-24</td>
      <td>HYG</td>
      <td>US25461LAA08</td>
      <td>DTV 5 7/8 08/15/27</td>
      <td>DIRECTV FINANCING LLC 5.875 08/15/2027 144a (S...</td>
      <td>1900-01-05 21:00:00</td>
      <td>2027-08-15</td>
      <td>FIXED</td>
      <td>CORP</td>
      <td>USD</td>
      <td>8.380</td>
      <td>8.171</td>
      <td>8.2755</td>
      <td>56308000</td>
      <td>0.364709</td>
    </tr>
  </tbody>
</table>
</div>



## b. Compute the NAV of the HYG basket and the intrinsic price of one ETF share.

Create the bond objects for all constituents of HYG. Compute the dirty price for each bond (from yield-to-maturity).

Aggregate the ETF NAV value (intrisic value of bond basket) as the weighted sum of dirty prices times basket weights. Keep in mind that the resulting ETF NAV will be on a face of $100, since the basket face notional weights add up to 100 percent.

Compute the intrinisc market capitalization of the HYG ETF by scaling the ETF NAV price to the ETF total face notional. 

Divide by 188,700,000 (the number of ETF shared outstanding as of 2024-04-26) to obtain the intrinsic price of one HYG ETF share.

As a reference, the HYG ETF market price as of 2024-04-26 was around $76.59, see the HYG YAS screen below.

![alt text](HYG_Price_Yield_Duration.JPG)


```python
bond_combined = pd.merge(hyg_corp_symbology,
                         hyg_basket_composition[['isin','bidYield','askYield', 'midYield', 'face_notional', 'face_notional_weight']], 
                         on=['isin']).reset_index(drop=True)
bond_combined.sort_values('ticker')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ticker</th>
      <th>class</th>
      <th>figi</th>
      <th>isin</th>
      <th>und_bench_isin</th>
      <th>security</th>
      <th>name</th>
      <th>type</th>
      <th>coupon</th>
      <th>cpn_type</th>
      <th>...</th>
      <th>rank</th>
      <th>amt_out</th>
      <th>country</th>
      <th>currency</th>
      <th>status</th>
      <th>bidYield</th>
      <th>askYield</th>
      <th>midYield</th>
      <th>face_notional</th>
      <th>face_notional_weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>AA</td>
      <td>Corp</td>
      <td>BBG00VYJS3B3</td>
      <td>US013822AE11</td>
      <td>US91282CKJ98</td>
      <td>AA 5 1/2 12/15/27</td>
      <td>ALCOA NEDERLAND HOLDING</td>
      <td>PRIV PLACEMENT</td>
      <td>5.500</td>
      <td>FIXED</td>
      <td>...</td>
      <td>Sr Unsecured</td>
      <td>750.0</td>
      <td>NE</td>
      <td>USD</td>
      <td>ACTV</td>
      <td>6.112</td>
      <td>5.913</td>
      <td>6.0125</td>
      <td>6130000</td>
      <td>0.039704</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AA</td>
      <td>Corp</td>
      <td>BBG00KXCGK19</td>
      <td>US013822AC54</td>
      <td>US91282CKG59</td>
      <td>AA 6 1/8 05/15/28</td>
      <td>ALCOA NEDERLAND HOLDING</td>
      <td>PRIV PLACEMENT</td>
      <td>6.125</td>
      <td>FIXED</td>
      <td>...</td>
      <td>Sr Unsecured</td>
      <td>500.0</td>
      <td>NE</td>
      <td>USD</td>
      <td>ACTV</td>
      <td>6.342</td>
      <td>6.152</td>
      <td>6.2470</td>
      <td>6071000</td>
      <td>0.039322</td>
    </tr>
    <tr>
      <th>2</th>
      <td>AA</td>
      <td>Corp</td>
      <td>BBG00ZKWG886</td>
      <td>US013822AG68</td>
      <td>US91282CKG59</td>
      <td>AA 4 1/8 03/31/29</td>
      <td>ALCOA NEDERLAND HOLDING</td>
      <td>PRIV PLACEMENT</td>
      <td>4.125</td>
      <td>FIXED</td>
      <td>...</td>
      <td>Sr Unsecured</td>
      <td>500.0</td>
      <td>NE</td>
      <td>USD</td>
      <td>ACTV</td>
      <td>6.282</td>
      <td>6.116</td>
      <td>6.1990</td>
      <td>8751000</td>
      <td>0.056681</td>
    </tr>
    <tr>
      <th>3</th>
      <td>AA</td>
      <td>Corp</td>
      <td>BBG01LW38YD3</td>
      <td>US013822AH42</td>
      <td>US91282CKG59</td>
      <td>AA 7 1/8 03/15/31</td>
      <td>ALCOA NEDERLAND HOLDING</td>
      <td>PRIV PLACEMENT</td>
      <td>7.125</td>
      <td>FIXED</td>
      <td>...</td>
      <td>Sr Unsecured</td>
      <td>750.0</td>
      <td>NE</td>
      <td>USD</td>
      <td>ACTV</td>
      <td>6.889</td>
      <td>6.732</td>
      <td>6.8105</td>
      <td>12292000</td>
      <td>0.079616</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AAL</td>
      <td>Corp</td>
      <td>BBG01F5TZPJ2</td>
      <td>US023771T329</td>
      <td>US91282CKJ98</td>
      <td>AAL 7 1/4 02/15/28</td>
      <td>AMERICAN AIRLINES INC</td>
      <td>PRIV PLACEMENT</td>
      <td>7.250</td>
      <td>FIXED</td>
      <td>...</td>
      <td>1st lien</td>
      <td>750.0</td>
      <td>US</td>
      <td>USD</td>
      <td>ACTV</td>
      <td>7.072</td>
      <td>6.879</td>
      <td>6.9755</td>
      <td>11161000</td>
      <td>0.072290</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1176</th>
      <td>ZFFNGR</td>
      <td>Corp</td>
      <td>BBG01MG435M5</td>
      <td>US98877DAF24</td>
      <td>US91282CKG59</td>
      <td>ZFFNGR 6 3/4 04/23/30</td>
      <td>ZF NA CAPITAL</td>
      <td>PRIV PLACEMENT</td>
      <td>6.750</td>
      <td>FIXED</td>
      <td>...</td>
      <td>Sr Unsecured</td>
      <td>800.0</td>
      <td>US</td>
      <td>USD</td>
      <td>ACTV</td>
      <td>6.769</td>
      <td>6.618</td>
      <td>6.6935</td>
      <td>4117000</td>
      <td>0.026666</td>
    </tr>
    <tr>
      <th>1179</th>
      <td>ZIGGO</td>
      <td>Corp</td>
      <td>BBG00QL6KSS1</td>
      <td>US98955DAA81</td>
      <td>US91282CKG59</td>
      <td>ZIGGO 4 7/8 01/15/30</td>
      <td>ZIGGO BV</td>
      <td>PRIV PLACEMENT</td>
      <td>4.875</td>
      <td>FIXED</td>
      <td>...</td>
      <td>1st lien</td>
      <td>991.0</td>
      <td>NE</td>
      <td>USD</td>
      <td>ACTV</td>
      <td>7.726</td>
      <td>7.529</td>
      <td>7.6275</td>
      <td>14929000</td>
      <td>0.096696</td>
    </tr>
    <tr>
      <th>1180</th>
      <td>ZIGGO</td>
      <td>Corp</td>
      <td>BBG00RMZ9HN6</td>
      <td>US98953GAD79</td>
      <td>US91282CKG59</td>
      <td>ZIGGO 5 1/8 02/28/30</td>
      <td>ZIGGO BOND CO BV</td>
      <td>PRIV PLACEMENT</td>
      <td>5.125</td>
      <td>FIXED</td>
      <td>...</td>
      <td>Sr Unsecured</td>
      <td>500.0</td>
      <td>NE</td>
      <td>USD</td>
      <td>ACTV</td>
      <td>9.049</td>
      <td>8.872</td>
      <td>8.9605</td>
      <td>7160000</td>
      <td>0.046376</td>
    </tr>
    <tr>
      <th>1178</th>
      <td>ZIGGO</td>
      <td>Corp</td>
      <td>BBG00DSX0YW8</td>
      <td>US98954UAB98</td>
      <td>US91282CKJ98</td>
      <td>ZIGGO 6 01/15/27</td>
      <td>ZIGGO BOND CO BV</td>
      <td>PRIV PLACEMENT</td>
      <td>6.000</td>
      <td>FIXED</td>
      <td>...</td>
      <td>Sr Unsecured</td>
      <td>625.0</td>
      <td>NE</td>
      <td>USD</td>
      <td>ACTV</td>
      <td>6.848</td>
      <td>6.665</td>
      <td>6.7565</td>
      <td>9059000</td>
      <td>0.058675</td>
    </tr>
    <tr>
      <th>1181</th>
      <td>ZIGGO</td>
      <td>Corp</td>
      <td>BBG014D6JTZ0</td>
      <td>US91845AAA34</td>
      <td>US91282CJZ59</td>
      <td>ZIGGO 5 01/15/32</td>
      <td>VZ SECURED FINANCING BV</td>
      <td>PRIV PLACEMENT</td>
      <td>5.000</td>
      <td>FIXED</td>
      <td>...</td>
      <td>Secured</td>
      <td>1525.0</td>
      <td>NE</td>
      <td>USD</td>
      <td>ACTV</td>
      <td>7.942</td>
      <td>7.740</td>
      <td>7.8410</td>
      <td>23664000</td>
      <td>0.153273</td>
    </tr>
  </tbody>
</table>
<p>1182 rows × 28 columns</p>
</div>




```python
stats = bond_combined['face_notional'].describe()
print(stats['count'])
print(stats['mean'])
print(stats['50%'])
```

    1182.0
    13061891.708967851
    11010500.0



```python
stats = bond_combined.groupby('ticker')['face_notional'].sum().describe()
print(stats['count'])
print(stats['mean'])
print(stats['50%'])
```

    422.0
    36585677.72511848
    25908000.0



```python
bond_combined['midYield'].describe()
```




    count    1182.000000
    mean        8.736248
    std        13.283695
    min        -4.215500
    25%         6.493750
    50%         6.907250
    75%         8.022125
    max       414.622000
    Name: midYield, dtype: float64




```python
dirty_price_lst = []
for index, row in bond_combined.iterrows():
    bond_object = create_bond_from_symbology(row)
    bond_yield = row['midYield'] / 100
    bond_dirty_price = bond_object.dirtyPrice(bond_yield, ql.Thirty360(ql.Thirty360.USA), ql.Compounded, ql.Semiannual)
    dirty_price_lst.append(bond_dirty_price)

bond_combined['dirtyPrice'] = dirty_price_lst
bond_combined['NAV'] = bond_combined['dirtyPrice']*bond_combined['face_notional_weight']
display(bond_combined[['isin','midYield', 'face_notional', 'face_notional_weight','dirtyPrice','NAV']])

ETF_intrinsic_nav = sum(bond_combined['NAV'])/100
ETF_marketCap = ETF_intrinsic_nav*bond_combined['face_notional'].sum()/100
ETF_intrinsic_pricePerShare = ETF_marketCap/188700000
print(round(ETF_intrinsic_nav,4))
print(round(ETF_marketCap))
print(round(ETF_intrinsic_pricePerShare,4))

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>isin</th>
      <th>midYield</th>
      <th>face_notional</th>
      <th>face_notional_weight</th>
      <th>dirtyPrice</th>
      <th>NAV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>US013822AE11</td>
      <td>6.0125</td>
      <td>6130000</td>
      <td>0.039704</td>
      <td>100.407601</td>
      <td>3.986608</td>
    </tr>
    <tr>
      <th>1</th>
      <td>US013822AC54</td>
      <td>6.2470</td>
      <td>6071000</td>
      <td>0.039322</td>
      <td>102.373791</td>
      <td>4.025552</td>
    </tr>
    <tr>
      <th>2</th>
      <td>US013822AG68</td>
      <td>6.1990</td>
      <td>8751000</td>
      <td>0.056681</td>
      <td>91.664075</td>
      <td>5.195571</td>
    </tr>
    <tr>
      <th>3</th>
      <td>US013822AH42</td>
      <td>6.8105</td>
      <td>12292000</td>
      <td>0.079616</td>
      <td>102.639297</td>
      <td>8.171705</td>
    </tr>
    <tr>
      <th>4</th>
      <td>US023771T329</td>
      <td>6.9755</td>
      <td>11161000</td>
      <td>0.072290</td>
      <td>102.415730</td>
      <td>7.403656</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1177</th>
      <td>US98877DAG07</td>
      <td>6.7275</td>
      <td>4117000</td>
      <td>0.026666</td>
      <td>101.068186</td>
      <td>2.695081</td>
    </tr>
    <tr>
      <th>1178</th>
      <td>US98954UAB98</td>
      <td>6.7565</td>
      <td>9059000</td>
      <td>0.058675</td>
      <td>99.893781</td>
      <td>5.861316</td>
    </tr>
    <tr>
      <th>1179</th>
      <td>US98955DAA81</td>
      <td>7.6275</td>
      <td>14929000</td>
      <td>0.096696</td>
      <td>88.862194</td>
      <td>8.592592</td>
    </tr>
    <tr>
      <th>1180</th>
      <td>US98953GAD79</td>
      <td>8.9605</td>
      <td>7160000</td>
      <td>0.046376</td>
      <td>83.739450</td>
      <td>3.883467</td>
    </tr>
    <tr>
      <th>1181</th>
      <td>US91845AAA34</td>
      <td>7.8410</td>
      <td>23664000</td>
      <td>0.153273</td>
      <td>85.241320</td>
      <td>13.065161</td>
    </tr>
  </tbody>
</table>
<p>1182 rows × 6 columns</p>
</div>


    93.5584
    14444632328
    76.5481


## c. Compute the ETF yield using the ACF (Aggregated Cash-Flows) method

Create the bond objects for all constituents of HYG. 

Write a function that computes the ETF NAV for a given flat yield y.

Use a numerical root finder (e.g. root_scalar from scipy.optimize) to solve for 

ETF_NAV(yield) = ETF_NAV_Price 

and obtain the ETF yield.

As a reference, the HYG ETF market yield as of 2024-04-26 was around 8.20%.


```python
import scipy.optimize as opt

def calc_ETF_NAV_from_yield(etf_yield, bond_combined):
    dirty_price_lst = []
    for index, row in bond_combined.iterrows():
        bond_object = create_bond_from_symbology(row)  # Ensure this function is defined or implemented correctly
        bond_dirty_price = bond_object.dirtyPrice(etf_yield, ql.Thirty360(ql.Thirty360.USA), ql.Compounded, ql.Semiannual)        
        dirty_price_lst.append(bond_dirty_price)

    bond_combined['dirtyPrice'] = dirty_price_lst
    bond_combined['NAV'] = bond_combined['dirtyPrice'] * bond_combined['face_notional_weight']
    
    ETF_intrinsic_nav = sum(bond_combined['NAV']) / 100
    return ETF_intrinsic_nav 

def target_function(etf_yield, bond_combined, target_ETF_nav):
    return  calc_ETF_NAV_from_yield(etf_yield, bond_combined)- target_ETF_nav # Return the difference to be zeroed



# Example call to root_scalar
target_ETF_nav = ETF_intrinsic_nav
result = opt.root_scalar(target_function, args=(bond_combined, target_ETF_nav), bracket=[0, 1], method='brentq')  # Adjust bracket as needed

optimal_yield = result.root
print(f"The yield that matches the target NAV is: {optimal_yield:.4f}.")

```

    The yield that matches the target NAV is: 0.0820.


## d. Compute the ETF DV01, Duration and Convexity

Treat the ETF basket as a synthetic bond.

Use +/- 1 bp scenarios in ETF yield space to compute the ETF DV01, Duration and Convexity.

As a reference, the HYG ETF risk metrics as of 2024-04-26 are: DV01 of 3.57, Duration of 3.72 and Convexity of 187.


```python
y = optimal_yield
y_bumped = 1e-4

price_base = ETF_intrinsic_nav
price_up_1bp = calc_ETF_NAV_from_yield(y+y_bumped, bond_combined)
price_down_1bp = calc_ETF_NAV_from_yield(y-y_bumped, bond_combined)

# Compute scenario sensitivities
dv01 = (price_down_1bp - price_base) /y_bumped / 100
duration = dv01 / price_base * 100
convexity = (price_down_1bp - 2*price_base + price_up_1bp) /y_bumped**2/ price_base 
print('ETF DV01', round(dv01,2))
print('ETF Duration', round(duration,2))
print('ETF Convexity', round(convexity,2))
```

    ETF DV01 3.56
    ETF Duration 3.8
    ETF Convexity 20.06

