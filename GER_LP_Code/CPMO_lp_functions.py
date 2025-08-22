# -*- coding: utf-8 -*-
"""
Linear program functions file for CLIENT_PHARMACY_MAC_OPTIMIZATION

"""
import CPMO_parameters as p
import pandas as pd
import numpy as np
import time
import bisect
import logging
import pulp

from typing import (
    Callable,
    List,
    Tuple,
    Optional,
)

from pulp import *


def prepare_pricing(price_data: pd.DataFrame,
                    low_fac: float,
                    up_fac: float,
                    bounds_df: Optional[pd.DataFrame] = None,
                    unc_low_fac: Optional[float] = None,
                    unc_up_fac: Optional[float] = None,
                    unc_bounds_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    This function calculates the metrics required to generate price bounds
    for each row in the DataFrame. It returns a copy of the given DataFrame
    (i.e. the given DataFrame is not mutated in place).

    Parameters
    ----------
    price_data : pd.DataFrame
        This DataFrame must have the following columns:
            * "CHAIN_GROUP"
            * "PRICING_QTY_PROJ_EOY"
            * "PRICE_TIER"
            * "MAC1026_UNIT_PRICE"
            * "MAC_PRICE_UNIT_ADJ"
            * "GOODRX_UPPER_LIMIT"
            * "ZBD_UPPER_LIMIT"
            * "BEG_PERIOD_PRICE"
            * "PRICING_CLAIMS"
            * "PRICING_QTY"
            * "PRICING_PRICE_REIMB_CLAIM"

    low_fac : float
        Low factor that is the default decrease limit for any entry.

    up_fac : float
        Up factor that is the default increase limit for any entry.

    bounds_df : pd.DataFrame, optional
        This DataFrame's "upper_bound" column is used to bin the `price_data`
        table's "PRICING_PRICE_REIMB_CLAIM" field.

        `bounds_df` must have the following columns (the first two are required):
            * "upper_bound" (used to bin the data)
            * "max_percent_increase" (becomes up_fac)
            * "max_dollar_increase" (optional, becomes member_disruption_amt when given)
        
        If bounds_df is not given, or if it doesn't have all of the above columns,
        then up_fac, member_disruption_amt, and member_disruption_bound 
        will be set to their default values.

    unc_low_fac : float
        Low factor that is the default decrease limit for any U&C optimized entry.

    unc_bounds_df : pd.DataFrame, optional
        This DataFrame's "upper_bound" column is used to bin the `price_data`
        table's "PRICING_PRICE_REIMB_CLAIM" field for claims subject to U&C optimization.

        `unc_bounds_df` must have the following columns (the first two are required):
            * "upper_bound" (used to bin the data)
            * "max_percent_increase" (becomes up_fac)
            * "max_dollar_increase" (optional, becomes member_disruption_amt when given)
        
        If unc_bounds_df is not given, then
        up_fac, member_disruption_amt, and member_disruption_bound 
        will be set to their default values.

    Returns
    -------
    pd.DataFrame
        The metrics required for price bounds calculations are saved in the
        following columns:
            * "up_fac"
            * "low_fac"
            * "member_disruption_amt"
            * "wtw_upper_limit"
            * "pricing_awp_x_avg_fac"
            * "goodrx_upper_limit"
            * "zbd_upper_limit"
            * "member_disruption_bound"

    """
    import numpy as np

    # calculate up_fac, low_fac, member_disruption_amt
    df = price_data.assign(low_fac=low_fac)

    if bounds_df is not None:
        bounds_df = bounds_df.copy()
        bounds_cols = 'up_fac', 'member_disruption_amt'

        bins = bounds_df.pop('upper_bound')  # this mutates in place, hence copy
        bounds_idx = np.digitize(df.PRICING_PRICE_REIMB_CLAIM.fillna(bins.max()),
                                 bins,
                                 right=True)

        df = (df
               .assign(bidx=bounds_idx)
               .merge(bounds_df, how='left', left_on='bidx', right_index=True)
               .rename(columns=dict(zip(bounds_df.columns, bounds_cols)))
               .drop('bidx', axis=1))
    if p.UNC_OPT:
        unc_bounds_df = unc_bounds_df.copy()
        unc_bounds_cols = 'unc_up_fac', 'unc_member_disruption_amt'

        bins = unc_bounds_df.pop('upper_bound')  # this mutates in place, hence copy
        unc_bounds_idx = np.digitize(df.PRICING_PRICE_REIMB_CLAIM.fillna(bins.max()),
                                 bins,
                                 right=True)

        df = (df
               .assign(bidx=unc_bounds_idx)
               .merge(unc_bounds_df, how='left', left_on='bidx', right_index=True)
               .rename(columns=dict(zip(unc_bounds_df.columns, unc_bounds_cols)))
               .drop('bidx', axis=1))

    # set to default, incase bounds_df was None or had only a subset of these columns
    # note, `df.get` ensures that the value will not be overridden by the default
    df = df.assign(member_disruption_amt=df.get('member_disruption_amt', 20_000),
                   up_fac=df.get('up_fac', up_fac))
    if p.UNC_OPT:
        df = df.assign(unc_member_disruption_amt=df.get('unc_member_disruption_amt', 20_000),
                       up_fac=df.get('up_fac', up_fac))

    eoy_zero_conditions = [
     ((df.PRICING_QTY_PROJ_EOY == 0) & (df.PRICE_TIER != 'CONFLICT')) & p.ZERO_QTY_TIGHT_BOUNDS & p.TIERED_PRICE_LIM,
     ((df.PRICING_QTY_PROJ_EOY == 0) & (df.PRICE_TIER != 'CONFLICT')) & p.ZERO_QTY_TIGHT_BOUNDS & ~p.TIERED_PRICE_LIM,
     ((df.PRICING_QTY_PROJ_EOY == 0) & (df.PRICE_TIER != 'CONFLICT')) & ~p.ZERO_QTY_TIGHT_BOUNDS,
    ]

    up_facs = [p.PRICE_BOUNDS_DF.max_percent_increase[1], p.GPI_UP_FAC, 10]
    lo_facs = [p.GPI_LOW_FAC, p.GPI_LOW_FAC, 0.95]

    for cndtn, up, lo in zip(eoy_zero_conditions, up_facs, lo_facs):
        df.loc[cndtn, ['up_fac', 'low_fac']] = up, lo

    if p.HIGHEROF_PRICE_LIM: ## for higher of logic: assigm minimum value for dollar increases
        df.loc[df.PRICING_QTY_PROJ_EOY == 0, 'member_disruption_amt'] = 0.000001 
    else:
        df.loc[df.PRICING_QTY_PROJ_EOY == 0, 'member_disruption_amt'] = 20_000

    if p.HANDLE_CONFLICT_GPI and p.CONFLICT_GPI_AS_TIERS:
        df.loc[df.PRICE_TIER == 'CONFLICT', 'low_fac'] = p.CONFLICT_GPI_LOW_BOUND

    if p.UNC_OPT:
        unc_up_facs = [p.UNC_PRICE_BOUNDS_DF.max_percent_increase[1], p.UNC_GPI_UP_FAC, 10]
        unc_lo_facs = [p.UNC_GPI_LOW_FAC, p.UNC_GPI_LOW_FAC, 0.95]

        for cndtn, up, lo in zip(eoy_zero_conditions, unc_up_facs, unc_lo_facs):
            df.loc[cndtn, ['unc_up_fac', 'unc_low_fac']] = up, lo

        df.loc[df.PRICING_QTY_PROJ_EOY == 0, 'unc_member_disruption_amt'] = 20_000

        if p.HANDLE_CONFLICT_GPI and p.CONFLICT_GPI_AS_TIERS:
            df.loc[df.PRICE_TIER == 'CONFLICT', 'unc_low_fac'] = p.CONFLICT_GPI_LOW_BOUND
        
        df.up_fac = np.where(df.PRICE_CHANGED_UC & ~df.IS_MAINTENANCE_UC, df.unc_up_fac, df.up_fac)
        df.member_disruption_amount = np.where(df.PRICE_CHANGED_UC & ~df.IS_MAINTENANCE_UC,
                                               df.unc_member_disruption_amt, 
                                               df.member_disruption_amt)
        df.low_fac = np.where((df.PRICING_QTY_PROJ_EOY > 0) & df.PRICE_CHANGED_UC & ~df.IS_MAINTENANCE_UC, p.UNC_GPI_LOW_FAC, df.low_fac)

    # Adjust the allowable price decrease factor for claims that are kept
    # to price them below INTERCEPT_HIGH
    if p.INTERCEPTOR_OPT:
        df.low_fac = np.where((df.INTERCEPT_HIGH < df.MAC_PRICE_UNIT_ADJ * (1 - df.low_fac) + 0.001) & (df.MAC_PRICE_UNIT_ADJ > 0), 1 - ((df.INTERCEPT_HIGH - 0.0001) / df.MAC_PRICE_UNIT_ADJ) + 0.001, df.low_fac)
        if p.ALLOW_INTERCEPT_LIMIT and not ((p.CLIENT_NAME_TABLEAU.startswith('WTW') and not p.REMOVE_WTW_RESTRICTION) or (p.CLIENT_NAME_TABLEAU.startswith('AON'))):
            df.up_fac = np.where((df.INTERCEPT_LOW > df.MAC_PRICE_UNIT_ADJ * (1 + df.up_fac) + 0.001) & (df.MAC_PRICE_UNIT_ADJ > 0), (((df.INTERCEPT_LOW + 0.0001) / df.MAC_PRICE_UNIT_ADJ)-1) + 0.001, df.up_fac)
        
        # For the fictional R90 measurements created to accomodate R90OK vcmls, 
        # we assign the same increase limits as its R30 counterpart sharing the same vcmls
        if (df.loc[(~df.MAC_LIST.str.contains('OK')) &
                      (df.MEASUREMENT == 'R90'), 'CLAIMS'].sum() == 0) & (p.GUARANTEE_CATEGORY == 'Pure Vanilla'):

            groupers = ['CLIENT','REGION','BREAKOUT','MEASUREMENT','CHAIN_GROUP','CHAIN_SUBGROUP','MAC_LIST','GPI_NDC']

            r90_bound = df[(~df.MAC_LIST.str.contains('OK')) & (df.MEASUREMENT == 'R90')].groupby(groupers).agg(up_fac = ('up_fac', np.nanmin)).reset_index()
            r30_bound = df[(~df.MAC_LIST.str.contains('OK')) & (df.MEASUREMENT == 'R30')].groupby(groupers).agg(up_fac = ('up_fac', np.nanmin)).reset_index()

            r90_r30_bound = pd.merge(r90_bound, r30_bound, how = 'left', on = [x for x in groupers if x != 'MEASUREMENT'], suffixes = ('','_30'))
            r90_r30_bound['up_fac'] = r90_r30_bound['up_fac_30']
            r90_r30_bound.drop(columns = ['MEASUREMENT_30','up_fac_30'], inplace = True)
            r90_r30_bound.set_index(groupers, inplace = True)
            
            df.set_index(groupers, inplace = True)
            df.update(r90_r30_bound)
            df.reset_index(inplace = True)   
        
    flr1026 = df.MAC1026_UNIT_PRICE  # floor 1026
    mailchoice = df.CHAIN_GROUP.isin(['MAIL', 'MCHOICE'])
    macprice_uadj_1m = df.MAC_PRICE_UNIT_ADJ * 1_000_000

    # calculate goodrx upper limit
    # this first term is for older versions of CPMO_parameters.py that may
    # lack the GOODRX_OPT flag
    # if not any((p.GOODRX_OPT,
    #             p.RMS_OPT,
    #             p.APPLY_GENERAL_MULTIPLIER,
    #             p.APPLY_MAIL_MULTIPLIER,
    #             p.TRUECOST_CLIENT)):
    #     goodrx_up = macprice_uadj_1m
    # else:
        # at least bring price down to 1026 floor

    floor_non_mac_price = np.where(df.IN_NONCAP_OK_VCML & ~df.IS_MAC, df.PRICING_AVG_AWP * (1 - p.FLOOR_NON_MAC_RATE), -1)
    goodrx_up = np.where((df.GOODRX_UPPER_LIMIT < flr1026) & ~mailchoice & (floor_non_mac_price < flr1026),
                         flr1026,
                         np.where(df.GOODRX_UPPER_LIMIT < floor_non_mac_price,
                                  floor_non_mac_price + .00011,
                                  df.GOODRX_UPPER_LIMIT))

    # calculate zbd upper limit
    mask = [(df.ZBD_UPPER_LIMIT < flr1026) & ~mailchoice, df.ZBD_UPPER_LIMIT <= df.INTERCEPT_LOW]
    value = [flr1026, df.MAC_PRICE_UNIT_ADJ * 10000]
    zbd_up = np.select(mask, value, default = df.ZBD_UPPER_LIMIT)

    # calculate wtw upper limit
    if not ((p.CLIENT_NAME_TABLEAU.startswith('WTW') and not p.REMOVE_WTW_RESTRICTION) or (p.CLIENT_NAME_TABLEAU.startswith('AON'))):
        wtw_up = macprice_uadj_1m
    else:
        bpp1245 = df.BEG_PERIOD_PRICE * 1.249
        wtw_up = np.where((bpp1245 < flr1026) & ~mailchoice, flr1026, bpp1245)

    # calculate AVG FAC
    # create AVG_FAC based on mail/retail channel
    avg_fac = np.where(df.BG_FLAG == 'G',
                       np.where(mailchoice,
                                1 - p.MAIL_NON_MAC_RATE,
                                1 - p.RETAIL_NON_MAC_RATE),
                       1 - p.BRAND_NON_MAC_RATE)

    # calculate member disruption bound
    if p.HIGHEROF_PRICE_LIM: ## for higher of logic: assigm minimum value for dollar increase
        mbr_bound = np.where(df.PRICING_QTY_PROJ_EOY > 0,
                             (df.PRICING_CLAIMS / df.PRICING_QTY)
                             * (df.PRICING_PRICE_REIMB_CLAIM + df.member_disruption_amt),
                             0.00001) 
    else:
        mbr_bound = np.where(df.PRICING_QTY_PROJ_EOY > 0,
                             (df.PRICING_CLAIMS / df.PRICING_QTY)
                             * (df.PRICING_PRICE_REIMB_CLAIM + df.member_disruption_amt),
                             100_000)


    return df.assign(wtw_upper_limit=wtw_up,
                     pricing_awp_x_avg_fac=avg_fac * df.PRICING_AVG_AWP,
                     zbd_upper_limit=zbd_up,
                     goodrx_upper_limit=goodrx_up,
                     member_disruption_bound=mbr_bound)


def calculate_price_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """
    This takes a dataframe row of pricing information and returns the price bounds for that entry.
    It is meant to be used with pandas `DataFrame.apply` method.
    Please forgive me for hard coding the constraints and tiers in.  I plan on moving them to an
    input as soon as I can catch a breath.

    Required Parameters:
        MONTH - Month of the LP run
        foy_dict - a dictionary of clients and the month of their first run of the year
        AVG_FAC - the customary discount percentage (usually 75%)
        SIM - whether or not this is a simulation
    INPUT:
        df - dataframe row with 1026 price, U&C price, AWP, Region, projected quantity, projected claims,
            price reimbursed per claim, and current price
        low_fac = low factor that is the default decrease limit for any entry
        up_fac = up factor that is the default increase limit for any entry
    Output:
        A tuple with price bounds

    """
    from CPMO_shared_functions import round_to

    def diagnostic_report(lb, ub, lower_bound_lst, lower_bound_name_lst, upper_bound_lst, upper_bound_name_lst):
        lower_sorted_pairs = sorted(zip(lower_bound_lst, lower_bound_name_lst), reverse=True)
        upper_sorted_pairs = sorted(zip(upper_bound_lst, upper_bound_name_lst))
        lb_name = dict(lower_sorted_pairs).get(lb,'Error')
        ub_name = dict(upper_sorted_pairs).get(ub,'Error')
        lb = round_to(lower_bound)
        ub = round_to(upper_bound)
        return pd.Series([(lb,ub), lb_name, ub_name,lower_sorted_pairs, upper_sorted_pairs],
                         index=['Price_Bounds', 'lb_name', 'ub_name','lower_bound_ordered','upper_bound_ordered'])

    if (df.PRICE_MUTABLE == 0):
        lower_bound_lst = [df.MAC_PRICE_UNIT_ADJ]
        lower_bound_name_lst = ["Immutable"]
        upper_bound_lst = [df.MAC_PRICE_UNIT_ADJ]
        upper_bound_name_lst = ["Immutable"]
        lower_bound = df.MAC_PRICE_UNIT_ADJ
        upper_bound = df.MAC_PRICE_UNIT_ADJ
        return diagnostic_report(lower_bound, upper_bound, lower_bound_lst, lower_bound_name_lst, upper_bound_lst, upper_bound_name_lst)

    if p.PHARMACY_EXCLUSION:
        if df.CHAIN_GROUP in set(p.LIST_PHARMACY_EXCLUSION):
            lower_bound_lst = [df.MAC1026_UNIT_PRICE, df.MAC_PRICE_UNIT_ADJ]
            lower_bound_name_lst = ['floor 1026','Mac Price Unit Adj']
            upper_bound_lst = [df.MAC1026_UNIT_PRICE, df.MAC_PRICE_UNIT_ADJ]
            upper_bound_name_lst = ['floor 1026','Mac Price Unit Adj']
            lower_bound = np.nanmax([df.MAC1026_UNIT_PRICE, df.MAC_PRICE_UNIT_ADJ])
            upper_bound = np.nanmax([df.MAC1026_UNIT_PRICE, df.MAC_PRICE_UNIT_ADJ]) + 0.0001
            return diagnostic_report(lower_bound, upper_bound, lower_bound_lst, lower_bound_name_lst, upper_bound_lst,
                                     upper_bound_name_lst)

    if p.FORCE_FLOOR and (df.FLOOR_VCML or df.CHAIN_GROUP == 'MAIL' or df.CHAIN_GROUP == 'MCHOICE'):
            #Handle situations where MAC1026 floor is 0
            if df.MAC1026_UNIT_PRICE == 0:
                lower_bound_lst = [df.MAC1026_UNIT_PRICE + .00011]
                lower_bound_name_lst = ['floor 1026']
                upper_bound_lst = [df.MAC1026_UNIT_PRICE + .00011]
                upper_bound_name_lst = ['floor 1026']
                lower_bound = df.MAC1026_UNIT_PRICE + .00011
                upper_bound = df.MAC1026_UNIT_PRICE + .00011 + 0.0001
            else:
                lower_bound_lst = [df.MAC1026_UNIT_PRICE]
                lower_bound_name_lst = ['floor 1026']
                upper_bound_lst = [df.MAC1026_UNIT_PRICE]
                upper_bound_name_lst = ['floor 1026']
                lower_bound = df.MAC1026_UNIT_PRICE
                upper_bound = df.MAC1026_UNIT_PRICE + 0.0001
            return diagnostic_report(lower_bound, upper_bound, lower_bound_lst, lower_bound_name_lst, upper_bound_lst,
                                     upper_bound_name_lst)
    if p.FORCE_FLOOR and df.MATCH_CVSSP and df.CHAIN_SUBGROUP=='CVS':
            #Handle situations where MAC1026 floor is 0
            if df.MAC1026_UNIT_PRICE == 0:
                lower_bound_lst = [df.MAC1026_UNIT_PRICE + .00011, df.goodrx_upper_limit, df.wtw_upper_limit, df.zbd_upper_limit, df.INTERCEPT_LOW, df.MAC_PRICE_UNIT_ADJ*(1-df.low_fac), .0001]
                lower_bound_name_lst = ['floor 1026', 'goodrx_upper_limit','wtw_upper_limit','zbd_upper_limit','df.INTERCEPT_LOW','df.MAC_PRICE_UNIT_ADJ*(1-low_fac)', '.0001 constant']
                lower_bound = np.nanmin([df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, 
                                        np.nanmax([df.MAC1026_UNIT_PRICE + .00011, df.INTERCEPT_LOW, df.MAC_PRICE_UNIT_ADJ*(1-df.low_fac), .0001])])

                upper_bound_lst = [(df.MAC1026_UNIT_PRICE + .00011)*p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH]
                upper_bound_name_lst = ['floor 1026 times parity price collar']
            else:
                lower_bound_lst = [df.MAC1026_UNIT_PRICE, df.goodrx_upper_limit, df.wtw_upper_limit, df.zbd_upper_limit, df.INTERCEPT_LOW, df.MAC_PRICE_UNIT_ADJ*(1-df.low_fac), .0001]
                lower_bound_name_lst = ['floor 1026', 'goodrx_upper_limit','wtw_upper_limit','zbd_upper_limit','df.INTERCEPT_LOW','df.MAC_PRICE_UNIT_ADJ*(1-low_fac)', '.0001 constant']
                lower_bound = np.nanmin([df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, 
                                        np.nanmax([df.MAC1026_UNIT_PRICE, df.INTERCEPT_LOW, df.MAC_PRICE_UNIT_ADJ*(1-df.low_fac), .0001])])

                upper_bound_lst = [df.MAC1026_UNIT_PRICE*p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH]
                upper_bound_name_lst = ['floor 1026 times parity price collar']

            upper_bound = np.nanmax([lower_bound+0.0001, df.MAC1026_UNIT_PRICE*p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH])                

            return diagnostic_report(lower_bound, upper_bound, lower_bound_lst, lower_bound_name_lst, upper_bound_lst,
                                     upper_bound_name_lst)

    if pd.notna(df.UNC_OVRD_AMT):
        lower_bound_lst = [df.UNC_OVRD_AMT]
        lower_bound_name_lst = ['UNC_OVRD_AMT']
        upper_bound_lst = [df.UNC_OVRD_AMT]
        upper_bound_name_lst = ['UNC_OVRD_AMT']
        lower_bound = df.UNC_OVRD_AMT
        upper_bound = df.UNC_OVRD_AMT
        return diagnostic_report(lower_bound, upper_bound, lower_bound_lst, lower_bound_name_lst, upper_bound_lst,
                                 upper_bound_name_lst)

    if p.UNC_OPT and df.PRICE_CHANGED_UC:
        if df.RAISED_PRICE_UC and ( # either this price is < goodrx ref price, or we want U&C to override goodrx
                df.goodrx_upper_limit > df.MAC_PRICE_UNIT_ADJ or p.UNC_OVERRIDE_GOODRX):
            lower_bound_lst = [df.CURRENT_MAC_PRICE]
            lower_bound_name_lst = ['U&C Raised Price']
            upper_bound_lst = [df.CURRENT_MAC_PRICE]
            upper_bound_name_lst = ['U&C Raised Price']
            lower_bound = df.CURRENT_MAC_PRICE
            upper_bound = df.CURRENT_MAC_PRICE
            return diagnostic_report(lower_bound, upper_bound, lower_bound_lst, lower_bound_name_lst, upper_bound_lst,
                                     upper_bound_name_lst)
        # check MAC1026 here, because if *this* price is below the floor, then the U&C price will also be, and we shouldn't
        # bother doing this step. (This check is also performed in DIR, but is kept here in case the more complex floor logic
        # makes a difference.)
        if df.IS_TWOSTEP_UNC and not df.RAISED_PRICE_UC and df.MAC_PRICE_UPPER_LIMIT_UC > df.MAC1026_UNIT_PRICE and ( # either this price is < goodrx ref price, or we want U&C to override goodrx
                df.goodrx_upper_limit > df.MAC_PRICE_UPPER_LIMIT_UC or p.UNC_OVERRIDE_GOODRX):
            lower_bound_lst = [df.MAC_PRICE_UPPER_LIMIT_UC]
            lower_bound_name_lst = ['Mac Price Upper Limit UC']
            upper_bound_lst = [df.MAC_PRICE_UPPER_LIMIT_UC]
            upper_bound_name_lst = ['Mac Price Upper Limit UC']
            lower_bound = df.MAC_PRICE_UPPER_LIMIT_UC
            upper_bound = df.MAC_PRICE_UPPER_LIMIT_UC
            return diagnostic_report(lower_bound, upper_bound, lower_bound_lst, lower_bound_name_lst, upper_bound_lst,
                                     upper_bound_name_lst)
    
    if (df.low_fac == 0) & (df.up_fac == 0) & (df.MAC1026_UNIT_PRICE > df.pricing_awp_x_avg_fac):  ### AVG_AWP is incorrect
        lower_bound_lst = [df.MAC1026_UNIT_PRICE]
        lower_bound_name_lst = ['1026 floor']
        upper_bound_lst = [df.MAC1026_UNIT_PRICE, df.MAC_PRICE_UNIT_ADJ]
        upper_bound_name_lst = ['1026 floor', 'MAC Price Unit Adj']
        lower_bound = df.MAC1026_UNIT_PRICE
        upper_bound = np.nanmax([df.MAC1026_UNIT_PRICE, df.MAC_PRICE_UNIT_ADJ])
        if (p.CLIENT_NAME_TABLEAU.startswith('WTW') and not p.REMOVE_WTW_RESTRICTION) or (p.CLIENT_NAME_TABLEAU.startswith('AON')):
            lower_bound_lst.append(df.wtw_upper_limit)
            lower_bound_name_lst.append('wtw_upper_limit')
            lower_bound = np.nanmin([lower_bound, df.wtw_upper_limit])
            upper_bound_lst.append(df.wtw_upper_limit)
            upper_bound_name_lst.append('wtw_upper_limit+0.0001')
            upper_bound = np.nanmin([upper_bound, df.wtw_upper_limit])
        return diagnostic_report(lower_bound, upper_bound, lower_bound_lst, lower_bound_name_lst, upper_bound_lst,
                                 upper_bound_name_lst)
    if (df.low_fac == 0) & (df.up_fac == 0) & (df.MAC1026_UNIT_PRICE <= df.pricing_awp_x_avg_fac):
        lower_bound_lst = [df.MAC1026_UNIT_PRICE]
        lower_bound_name_lst = ['1026 floor']
        upper_bound_lst = [df.pricing_awp_x_avg_fac, df.MAC_PRICE_UNIT_ADJ]
        upper_bound_name_lst = ['PRICING_AVG_AWP * AVG_FAC', 'MAC Price Unit Adj']
        lower_bound = df.MAC1026_UNIT_PRICE
        upper_bound = np.nanmax([df.pricing_awp_x_avg_fac, df.MAC_PRICE_UNIT_ADJ])
        if (p.CLIENT_NAME_TABLEAU.startswith('WTW') and not p.REMOVE_WTW_RESTRICTION) or (p.CLIENT_NAME_TABLEAU.startswith('AON')):
            lower_bound_lst.append(df.wtw_upper_limit)
            lower_bound_name_lst.append('wtw_upper_limit')
            lower_bound = np.nanmin([lower_bound, df.wtw_upper_limit])
            upper_bound_lst.append(df.wtw_upper_limit)
            upper_bound_name_lst.append('wtw_upper_limit+0.0001')
            upper_bound = np.nanmin([upper_bound, df.wtw_upper_limit])
        return diagnostic_report(lower_bound, upper_bound, lower_bound_lst, lower_bound_name_lst, upper_bound_lst,
                                 upper_bound_name_lst)
    
    if (df.MAC1026_UNIT_PRICE == 0) & (df.low_fac == 0):
        lower_bound_lst = [df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, df.INTERCEPT_LOW, df.MAC_PRICE_UNIT_ADJ * 0.20]
        lower_bound_name_lst = ['zbd_upper_limit','goodrx_upper_limit','wtw_upper_limit', 'df.INTERCEPT_LOW','df.MAC_PRICE_UNIT_ADJ * 0.20']
        lower_bound = np.nanmin([df.zbd_upper_limit, df.goodrx_upper_limit,  df.wtw_upper_limit, np.nanmax([df.INTERCEPT_LOW, df.MAC_PRICE_UNIT_ADJ * 0.20])])### dont go below MAC1026 price
        
    elif (df.CHAIN_GROUP == 'MAIL' or df.CHAIN_GROUP == 'MCHOICE') and not p.APPLY_FLOORS_MAIL:
        # This branch will not apply MAC1026 floors to Mail prices
        lower_bound_lst = [df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, df.MAC_PRICE_UNIT_ADJ*(1-df.low_fac), .0001, df.INTERCEPT_LOW]
        lower_bound_name_lst = ['zbd_upper_limit', 'goodrx_upper_limit','wtw_upper_limit','df.MAC_PRICE_UNIT_ADJ*(1-low_fac)', '.0001 constant', 'df.INTERCEPT_LOW']
        lower_bound = np.nanmin([df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, np.nanmax([df.MAC_PRICE_UNIT_ADJ*(1-df.low_fac), .0001, df.INTERCEPT_LOW])])
        
    elif (df.CHAIN_GROUP == 'MAIL' or df.CHAIN_GROUP == 'MCHOICE') and p.APPLY_FLOORS_MAIL and not p.FULL_YEAR:
        # This branch will apply MAC1026 floors to Mail prices, but not if they violate our price increase limits or Non-Mac Rate discount.
        lower_bound_lst = [df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, df.MAC_PRICE_UNIT_ADJ*(1-df.low_fac), 
                           .0001, df.INTERCEPT_LOW, df.MAC1026_UNIT_PRICE * p.MAIL_FLOORS_FACTOR, df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac), df.pricing_awp_x_avg_fac]
        lower_bound_name_lst = ['zbd_upper_limit', 'goodrx_upper_limit','wtw_upper_limit','df.MAC_PRICE_UNIT_ADJ*(1-low_fac)', 
                                '.0001 constant', 'df.INTERCEPT_LOW','floor 1026*factor','df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac)','df.pricing_awp_x_avg_fac']
        lower_bound = np.nanmin([df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, 
                                 np.nanmax([np.nanmin([df.MAC1026_UNIT_PRICE * p.MAIL_FLOORS_FACTOR, df.pricing_awp_x_avg_fac,
                                                       df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac)]),
                                            df.MAC_PRICE_UNIT_ADJ*(1-df.low_fac), .0001, df.INTERCEPT_LOW])])
        
    elif (df.CHAIN_GROUP == 'MAIL' or df.CHAIN_GROUP == 'MCHOICE') and p.APPLY_FLOORS_MAIL and p.FULL_YEAR:
        # This branch will apply MAC1026 floors to Mail prices, but they will be allowed to violate price increase limits. Only meant to be used for FULL_YEAR runs.
        lower_bound_lst = [df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, df.MAC_PRICE_UNIT_ADJ*(1-df.low_fac), 
                           .0001, df.INTERCEPT_LOW, df.MAC1026_UNIT_PRICE * p.MAIL_FLOORS_FACTOR, df.pricing_awp_x_avg_fac]
        lower_bound_name_lst = ['zbd_upper_limit', 'goodrx_upper_limit','wtw_upper_limit','df.MAC_PRICE_UNIT_ADJ*(1-low_fac)', 
                                '.0001 constant', 'df.INTERCEPT_LOW','floor 1026*factor','df.pricing_awp_x_avg_fac']
        lower_bound = np.nanmin([df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, 
                                 np.nanmax([np.nanmin([df.MAC1026_UNIT_PRICE * p.MAIL_FLOORS_FACTOR, df.pricing_awp_x_avg_fac]), 
                                            df.MAC_PRICE_UNIT_ADJ*(1-df.low_fac), .0001, df.INTERCEPT_LOW])])
        
    elif p.UNC_OPT and (df.PRICE_CHANGED_UC) and not (df.RAISED_PRICE_UC):
        lower_bound_lst = [df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, df.INTERCEPT_LOW, df.MAC1026_UNIT_PRICE, df.MAC_PRICE_UNIT_ADJ*(1-df.low_fac), .0001]
        lower_bound_name_lst = ['zbd_upper_limit', 'goodrx_upper_limit', 'wtw_upper_limit', 'df.INTERCEPT_LOW', 'floor 1026','df.MAC_PRICE_UNIT_ADJ*(1-low_fac)', '.0001']
        lower_bound = np.nanmin([df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, np.nanmax([df.INTERCEPT_LOW, df.MAC1026_UNIT_PRICE, df.MAC_PRICE_UNIT_ADJ*(1-df.low_fac), .0001])]) #.995 is the limit to find outlier claims
        
    else:
        lower_bound_lst = [df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, df.INTERCEPT_LOW, df.MAC1026_UNIT_PRICE, df.MAC_PRICE_UNIT_ADJ*(1-df.low_fac), .0001]
        lower_bound_name_lst = ['zbd_upper_limit', 'goodrx_upper_limit', 'wtw_upper_limit', 'df.INTERCEPT_LOW', 'floor 1026','df.MAC_PRICE_UNIT_ADJ*(1-low_fac)', '.0001']
        lower_bound = np.nanmin([df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, np.nanmax([df.INTERCEPT_LOW, df.MAC1026_UNIT_PRICE, df.MAC_PRICE_UNIT_ADJ*(1-df.low_fac), .0001])]) #.995 is the limit to find outlier claims

    if (p.CLIENT_NAME_TABLEAU.startswith('WTW') and not p.REMOVE_WTW_RESTRICTION) or (p.CLIENT_NAME_TABLEAU.startswith('AON')):
        if 1.25*df.BEG_PERIOD_PRICE < df.MAC1026_UNIT_PRICE and df.CHAIN_GROUP != 'MAIL' and df.CHAIN_GROUP != 'MCHOICE':
            lower_bound_lst = [df.wtw_upper_limit]
            lower_bound_name_lst = ['wtw_upper_limit']
            lower_bound = df.wtw_upper_limit
    
    if (df.PRICING_UC_UNIT > 0) and (df.PRICING_AVG_AWP > 0) and (df.PRICING_QTY_PROJ_EOY > 0):
        if p.UNC_OPT and (df.PRICE_CHANGED_UC) and not (df.RAISED_PRICE_UC):
            upper_bound_lst = [lower_bound+0.0001, df.pricing_awp_x_avg_fac, df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac),
                                df.member_disruption_bound,df.MAC_PRICE_UPPER_LIMIT_UC, df.CLIENT_MAX_PRICE, df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit]
            upper_bound_name_lst = ['lower_bound + 0.0001','df.PRICING_AVG_AWP*AVG_FAC', 'df.MAC_PRICE_UNIT_ADJ*(1.0+up_fac)',
                                    'member_disruption_bound','df.MAC_PRICE_UPPER_LIMIT_UC', 'df.CLIENT_MAX_PRICE', 'zbd_upper_limit', 'goodrx_upper_limit','wtw_upper_limit+0.0001']
            if p.HIGHEROF_PRICE_LIM:
                upper_bound = np.nanmax([lower_bound+0.0001, np.nanmin([df.pricing_awp_x_avg_fac, np.nanmax([df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac), 
                                                      df.member_disruption_bound]), df.MAC_PRICE_UPPER_LIMIT_UC,
                                                      df.CLIENT_MAX_PRICE, df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit])])
            else:
                upper_bound = np.nanmax([lower_bound+0.0001, np.nanmin([df.pricing_awp_x_avg_fac, df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac), 
                                                      df.member_disruption_bound, df.MAC_PRICE_UPPER_LIMIT_UC,
                                                      df.CLIENT_MAX_PRICE, df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit])])
        else:
            upper_bound_lst = [lower_bound+0.0001,df.pricing_awp_x_avg_fac, df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac), 
                               df.member_disruption_bound, df.PRICING_UC_UNIT, df.CLIENT_MAX_PRICE, df.zbd_upper_limit, df.goodrx_upper_limit, 
                               df.wtw_upper_limit, df.INTERCEPT_HIGH]
            upper_bound_name_lst = ['lower_bound+0.0001', 'df.PRICING_AVG_AWP*AVG_FAC','df.MAC_PRICE_UNIT_ADJ*(1.0+up_fac)',
                                    'member_disruption_bound','df.PRICING_UC_UNIT','df.CLIENT_MAX_PRIC', 'zbd_upper_limit', 'goodrx_upper_limit', 
                                    'wtw_upper_limit+0.0001', 'df.INTERCEPT_HIGH']
            if p.HIGHEROF_PRICE_LIM:
                upper_bound = np.nanmax([lower_bound+0.0001, np.nanmin([df.pricing_awp_x_avg_fac, np.nanmax([df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac), 
                                        df.member_disruption_bound]), df.PRICING_UC_UNIT, df.CLIENT_MAX_PRICE, df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, 
                                        df.INTERCEPT_HIGH])])
            else: 
                upper_bound = np.nanmax([lower_bound+0.0001, np.nanmin([df.pricing_awp_x_avg_fac, df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac), 
                                        df.member_disruption_bound, df.PRICING_UC_UNIT, df.CLIENT_MAX_PRICE, df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, 
                                        df.INTERCEPT_HIGH])])  ### dont go above awp unit price
    elif df.PRICING_QTY_PROJ_EOY == 0:
        if p.UNC_OPT and (df.PRICE_CHANGED_UC) and (df.MAC_PRICE_UPPER_LIMIT_UC > 0) and not (df.RAISED_PRICE_UC):
                upper_bound_lst = [lower_bound+0.0001, df.pricing_awp_x_avg_fac, df.member_disruption_bound, df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac),
                                   df.CLIENT_MAX_PRICE, df.MAC_PRICE_UPPER_LIMIT_UC, df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit]
                upper_bound_name_lst = ['lower_bound+0.0001','df.PRICING_AVG_AWP*AVG_FAC', 'member_disruption_bound', 'df.MAC_PRICE_UNIT_ADJ*(1.0+up_fac)',
                                        'df.CLIENT_MAX_PRICE', 'df.MAC_PRICE_UPPER_LIMIT_UC', 'zbd_upper_limit', 'goodrx_upper_limit', 'wtw_upper_limit+0.0001']
                if  p.HIGHEROF_PRICE_LIM:
                    upper_bound = np.nanmax([lower_bound+0.0001, np.nanmin([df.pricing_awp_x_avg_fac, np.nanmax([df.member_disruption_bound, df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac)]), df.CLIENT_MAX_PRICE, df.MAC_PRICE_UPPER_LIMIT_UC, df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit])])
                else:
                    upper_bound = np.nanmax([lower_bound+0.0001, np.nanmin([df.pricing_awp_x_avg_fac, df.member_disruption_bound, df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac), 
df.CLIENT_MAX_PRICE, df.MAC_PRICE_UPPER_LIMIT_UC, df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit])])     
        else:
            upper_bound_lst = [lower_bound+0.0001, df.pricing_awp_x_avg_fac, df.member_disruption_bound, df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac),
                                df.CLIENT_MAX_PRICE, df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, df.INTERCEPT_HIGH]
            upper_bound_name_lst = ['lower_bound+0.0001','df.PRICING_AVG_AWP*AVG_FAC', 'member_disruption_bound', 'df.MAC_PRICE_UNIT_ADJ*(1.0+up_fac)',
                                    'df.CLIENT_MAX_PRICE', 'zbd_upper_limit', 'goodrx_upper_limit', 'wtw_upper_limit+0.0001', 'df.INTERCEPT_HIGH']
            if p.HIGHEROF_PRICE_LIM:
                upper_bound = np.nanmax([lower_bound+0.0001, np.nanmin([df.pricing_awp_x_avg_fac, np.nanmax([df.member_disruption_bound, df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac)]), df.CLIENT_MAX_PRICE, df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, df.INTERCEPT_HIGH])])
            else:
                upper_bound = np.nanmax([lower_bound+0.0001, np.nanmin([df.pricing_awp_x_avg_fac, df.member_disruption_bound, df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac),
                                        df.CLIENT_MAX_PRICE, df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, df.INTERCEPT_HIGH])])
    else:
        if p.UNC_OPT and (df.PRICE_CHANGED_UC) and (df.MAC_PRICE_UPPER_LIMIT_UC > 0) and not (df.RAISED_PRICE_UC):
                upper_bound_lst = [lower_bound+0.0001, df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac), df.member_disruption_bound,
                                   df.pricing_awp_x_avg_fac, df.CLIENT_MAX_PRICE, df.MAC_PRICE_UPPER_LIMIT_UC, df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit]
                upper_bound_name_lst = ['lower_bound+0.0001', 'df.MAC_PRICE_UNIT_ADJ*(1.0+up_fac)', 'member_disruption_bound',
                                        'df.PRICING_AVG_AWP*AVG_FAC, df.CLIENT_MAX_PRICE', 'df.MAC_PRICE_UPPER_LIMIT_UC',
                                        'zbd_upper_limit', 'goodrx_upper_limit', 'wtw_upper_limit+0.0001']
                if p.HIGHEROF_PRICE_LIM:
                    upper_bound = np.nanmax([lower_bound+0.0001, np.nanmin([np.nanmax([df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac), df.member_disruption_bound]), 
                                            df.pricing_awp_x_avg_fac, df.CLIENT_MAX_PRICE, df.MAC_PRICE_UPPER_LIMIT_UC, df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit])])
                else:
                    upper_bound = np.nanmax([lower_bound+0.0001, np.nanmin([df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac), df.member_disruption_bound, 
                                            df.pricing_awp_x_avg_fac, df.CLIENT_MAX_PRICE, df.MAC_PRICE_UPPER_LIMIT_UC, df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit])])

        else:
            upper_bound_lst = [lower_bound+0.0001, df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac), df.member_disruption_bound,
                                                      df.pricing_awp_x_avg_fac, df.CLIENT_MAX_PRICE, df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, df.INTERCEPT_HIGH]
            upper_bound_name_lst = ['lower_bound+0.0001', 'df.MAC_PRICE_UNIT_ADJ*(1.0+up_fac)', 'member_disruption_bound',
                                                      'df.PRICING_AVG_AWP*AVG_FAC', 'df.CLIENT_MAX_PRICE', 'zbd_upper_limit', 'goodrx_upper_limit', 'wtw_upper_limit+0.0001', 'df.INTERCEPT_HIGH']
            if p.HIGHEROF_PRICE_LIM:
                upper_bound = np.nanmax([lower_bound+0.0001, np.nanmin([np.nanmax([df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac), df.member_disruption_bound]),
                                                      df.pricing_awp_x_avg_fac, df.CLIENT_MAX_PRICE, df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, df.INTERCEPT_HIGH])])  ### dont go above awp unit price
            else:
                upper_bound = np.nanmax([lower_bound+0.0001, np.nanmin([df.MAC_PRICE_UNIT_ADJ*(1.0+df.up_fac), df.member_disruption_bound,
                                                      df.pricing_awp_x_avg_fac, df.CLIENT_MAX_PRICE, df.zbd_upper_limit, df.goodrx_upper_limit, df.wtw_upper_limit, df.INTERCEPT_HIGH])])  ### dont go above awp unit price

    if (p.CLIENT_NAME_TABLEAU.startswith('WTW') and not p.REMOVE_WTW_RESTRICTION) or (p.CLIENT_NAME_TABLEAU.startswith('AON')):
        if 1.25*df.BEG_PERIOD_PRICE < df.MAC1026_UNIT_PRICE and df.CHAIN_GROUP != 'MAIL' and df.CHAIN_GROUP != 'MCHOICE':
            upper_bound_lst = [df.wtw_upper_limit+0.0001]
            upper_bound_name_lst = ['wtw_upper_limit+0.0001']
            upper_bound = df.wtw_upper_limit+0.0001

    #incorporate client suggested upperbound
    if df.CLIENT_MIN_PRICE >= (upper_bound - 0.0001):
        if (upper_bound * .95) > lower_bound:
            lower_bound_lst.append(upper_bound * .95)
            lower_bound_name_lst.append('upper_bound * .95')
            lower_bound = upper_bound * .95
    elif df.PRICING_QTY_PROJ_EOY == 0:
        lower_bound = lower_bound
    else:
        lower_bound_lst.append(df.CLIENT_MIN_PRICE)
        lower_bound_name_lst.append('df.CLIENT_MIN_PRICE')
        lower_bound = np.nanmax([lower_bound, df.CLIENT_MIN_PRICE])

    if df.CLIENT_MAX_PRICE < lower_bound:
        upper_bound_lst.append(lower_bound + (.05 * df.MAC_PRICE_UNIT_ADJ))
        upper_bound_name_lst.append('lower_bound + (.05 * df.MAC_PRICE_UNIT_ADJ)')
        upper_bound = lower_bound + (.05 * df.MAC_PRICE_UNIT_ADJ)
        logging.debug('Error with client supplied upper bound at %s %s %s %s', df.REGION, df.CHAIN_GROUP, df.CHAIN_SUBGROUP, df.GPI_NDC)

    if (upper_bound == 0) or (lower_bound == 0):
        logging.debug('Error with bounds at %s', df.GPI_NDC)
            
    return diagnostic_report(lower_bound, upper_bound, lower_bound_lst, lower_bound_name_lst, upper_bound_lst,
                             upper_bound_name_lst)


def generate_price_bounds(lower_scale_factor: float,
                          upper_scale_factor: float,
                          price_data: pd.DataFrame,
                          ) -> pd.DataFrame:
    """
    Parameters
    ---------
    price_data : pd.DataFrame
        A DataFrame with existing_prices, mac1026 prices, awp_price. It must
        have the following columns:
            * "CLIENT"
            * "PRICE_TIER"
            * "CHAIN_GROUP"
            * "PRICING_QTY_PROJ_EOY"
            * "MAC1026_UNIT_PRICE"
            * "MAC_PRICE_UNIT_ADJ"
            * "CURRENT_MAC_PRICE"
            * "GOODRX_UPPER_LIMIT"
            * "ZBD_UPPER_LIMIT"
            * "BEG_PERIOD_PRICE"
            * "PRICING_CLAIMS"
            * "PRICING_QTY"
            * "PRICING_PRICE_REIMB_CLAIM"

    Returns
    -------
    pd.DataFrame
         Return a DataFrame with lower bound and upper bound for the price.

    Raises
    ------
    ValueError
        The price_data DataFrame's "PRICE_TIER" column must contain only one
        unique price tier. This function will throw an error if that is not
        the case.

    Output:
        Return a dataframe with lower bound and upper bound for the price

    """
    # calculate start of year (soy)
    soy_conditions = [price_data.CLIENT.isin(p.TIERED_PRICE_CLIENT)
                      & p.TIERED_PRICE_LIM,
                      p.HIGHEROF_PRICE_LIM,
                      price_data.GPI_CHANGE_EXCEPT == 1,]
    start_of_year = np.select(soy_conditions, [True, True, True], p.FULL_YEAR)

    try:
        price_tier, = price_data.PRICE_TIER.unique()
    except ValueError as e:
        raise ValueError('More than one PRICE_TIER: '
                         f'{price_data.PRICE_TIER.unique()}') from e

    # `bounds` is a NoneType object or pd.DataFrame\
    if not start_of_year.any():
        bounds = None
    elif p.HIGHEROF_PRICE_LIM:
        _data = {'upper_bound': price_data.PRICING_PRICE_REIMB_CLAIM.max(),
                 'max_percent_increase': p.GPI_UP_FAC,
                 'max_dollar_increase': p.GPI_UP_DOLLAR}
        bounds = pd.DataFrame(_data, index=range(len(_data)))

    elif p.FULL_YEAR and p.TIERED_PRICE_LIM and not p.HIGHEROF_PRICE_LIM:
        bounds_df_name = f'FULL_YEAR_LV_{p.NEW_YEAR_PRICE_LVL}_PRICE_BOUNDS_DF'
        bounds = getattr(p, bounds_df_name)  # throws AttribErr if lookup fails

    elif p.FULL_YEAR and not p.TIERED_PRICE_LIM and not p.HIGHEROF_PRICE_LIM:
        _data = {'upper_bound': price_data.PRICING_PRICE_REIMB_CLAIM.max(),
                 'max_percent_increase': p.GPI_UP_FAC,
                 'max_dollar_increase': 20_000}
        bounds = pd.DataFrame(_data, index=range(len(_data)))

    elif price_tier == '0':
        bounds = p.PRICE_BOUNDS_DF
        
    elif price_tier == 'CONFLICT':
        #since upper_bound severs as the price bracket,setting upper_bound as index to avoid it get multiplied by CONFLICT_GPI_AS_TIERS_BOUNDS
        bounds = (p.PRICE_BOUNDS_DF.set_index('upper_bound') * p.CONFLICT_GPI_AS_TIERS_BOUNDS).reset_index()

    else:
        # start_of_year AND price_data.PRICE_TIER.isin(['1', '2', '3'])
        up_facs = {'1': (0.5, 0.6, 1.2, 1.00),
                   '2': (0.25, 0.4, 0.6, 0.83333333),
                   '3': (0.15, 0.3, 0.4, 0.6666666)}
        bounds = pd.DataFrame({'upper_bound': [999_999, 100, 6, 3],
                               'max_percent_increase': up_facs.get(price_tier)})
    if price_tier=='CONFLICT': print(bounds)
    if price_tier=='0': print(bounds)

    if p.UNC_OPT:
        unc_low_fac = p.UNC_GPI_LOW_FAC
        unc_up_fac = p.UNC_GPI_UP_FAC
        unc_bounds = p.UNC_PRICE_BOUNDS_DF
    else:
        unc_low_fac = None
        unc_up_fac = None
        unc_bounds = None
        
    calc_df = prepare_pricing(price_data.assign(start_of_year=start_of_year),
                              lower_scale_factor,
                              upper_scale_factor,
                              bounds,
                              unc_low_fac,
                              unc_up_fac,
                              unc_bounds)
    priced_df = calc_df.apply(calculate_price_bounds, axis=1)
    
    return priced_df[['Price_Bounds',
                      'lb_name',
                      'ub_name',
                      'lower_bound_ordered',
                      'upper_bound_ordered']]


def create_CBMRC_price_decision(df: pd.DataFrame) -> pd.Series:
    """
    CBMRC stands for "Client, Breakout, Measure, Region, Chain".
    Creates an array of pulp decision variables for LP use.
    The calculation is done by concatenating 7 columns from the input DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame with the following columns:
            * "GPI_NDC"
            * "CLIENT"
            * "BREAKOUT"
            * "MEASUREMENT"
            * "BG_FLAG"
            * "REGION"
            * "CHAIN_GROUP"
            * "CHAIN_SUBGROUP"

    Returns
    -------
    pd.Series
        An array of pulp decision variables.
    """
    columns = [
     'GPI_NDC',
     'CLIENT',
     'BREAKOUT',
     'MEASUREMENT',
     'REGION',
     'BG_FLAG',
     'CHAIN_GROUP',
     'CHAIN_SUBGROUP',
    ]
    fields = [df[col] for col in columns]

    price_var = [f'P_{gpi_ndc}_{clnt}_{bout}_{meas}_{reg}_{bg}_{cgrp}_{csubgrp}'
                 for gpi_ndc, clnt, bout, meas, reg, bg, cgrp, csubgrp
                 in zip(*fields)]

    return pd.Series([pulp.LpVariable(pv, *pb)
                      for pv, pb in zip(price_var, df.Price_Bounds)])


def generatePricingDecisionVariables(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Generates the pricing decision variable name as well as calls
    the function to create the price decision PuLP variable.

    NOTE: the input DataFrame is mutated in place.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame with the following columns:
            * "GPI_NDC"
            * "CLIENT"
            * "BREAKOUT"
            * "MEASUREMENT"
            * "REGION"
            * "BG_FLAG"
            * "CHAIN_GROUP"
            * "CHAIN_SUBGROUP"

    Returns
    -------
    pd.DataFrame
        The input DataFrame is mutated with two columns added to it:
            * "Price_Decision_Var"
            * "Dec_Var_Name"
    '''
    df['Price_Decision_Var'] = create_CBMRC_price_decision(df)
    df['Dec_Var_Name'] = df.Price_Decision_Var.map(lambda x: x.getName())
    return df


def generateLambdaDecisionVariables_ebit(Cl_df, Ph_list):
    '''
    Input: List of PHARMACY types
    Output: Dataframe of PHARMACY type and lambda decision variable with over and under performance

    '''

    Lambda_df = pd.DataFrame(columns = ['PHARMACY_TYPE', 'Lambda_Over', 'Lambda_Under', 'Lambda_Level', 'PHARMACY'])
    row_count = 0
    ## Clients  #####
    lambda_var_over, lambda_var_under  = str('lambda_over'), str('lambda_under')

    for client in Cl_df.Combined.values:
        Lambda_df.loc[row_count,'PHARMACY_TYPE'] = 'CLIENT'

        if lambda_var_over != 'NA':
            Lambda_df.loc[row_count,'Lambda_Over'] = pulp.LpVariable(client + '_' + lambda_var_over, lowBound=0)
        if lambda_var_under != 'NA':
            Lambda_df.loc[row_count,'Lambda_Under'] = pulp.LpVariable(client + '_' + lambda_var_under, lowBound=0)

        Lambda_df.loc[row_count,'Lambda_Level'] = 'CLIENT'
        Lambda_df.loc[row_count,'PHARMACY'] = client
        row_count +=1

    ## Pharmacies #####
    for ph in Ph_list:
        non_cap_list = []
        if p.GENERIC_OPT and p.BRAND_OPT:
            non_cap_list = set(p.NON_CAPPED_PHARMACY_LIST['GNRC']).intersection(p.NON_CAPPED_PHARMACY_LIST['BRND'])
        elif p.BRAND_OPT:
            non_cap_list = p.NON_CAPPED_PHARMACY_LIST['BRND']
        elif p.GENERIC_OPT:
            non_cap_list = p.NON_CAPPED_PHARMACY_LIST['GNRC']
        if ph not in non_cap_list:
            if ph in ['CVS', 'PREF_OTH']:
                Lambda_df.loc[row_count,'PHARMACY_TYPE'] = 'Preferred'
            else:
                Lambda_df.loc[row_count,'PHARMACY_TYPE'] = 'Non_Preferred'
            lambda_var_over, lambda_var_under  = str(ph  + '_lambda_over'), str(ph  + '_lambda_under')
            Lambda_df.loc[row_count,'Lambda_Over'] = pulp.LpVariable(lambda_var_over, lowBound=0)
            Lambda_df.loc[row_count,'Lambda_Under'] = pulp.LpVariable(lambda_var_under, lowBound=0)
            if ph == 'CVS':
                Lambda_df.loc[row_count,'Lambda_Level'] = 'CLIENT'
            else:
                Lambda_df.loc[row_count,'Lambda_Level'] = 'CLIENT'
            Lambda_df.loc[row_count,'PHARMACY'] = ph
            row_count +=1

    return Lambda_df

def generateLeakageDecisionVariables(df, cat, n = 0, nm = ''):
    '''
    Generates the delta and leakage variables for the solver.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame with the following columns:
            * "GPI_NDC"
            * "CLIENT"
            * "BREAKOUT"
            * "MEASUREMENT"
            * "REGION"
            * "CHAIN_GROUP"
            * "CHAIN_SUBGROUP"
            
    cat : Type of LpVariable, Binary or Continuous
    
    n : Number used to give distinct label to each delta variable
    
    nm : String used to distinguish copay/coinsurance/complex leakage variables

    Returns
    -------
    pd.Series
        Depending on arguments, returns one of the below Series:
            * "delta1"
            * "delta2"
            * "delta3"
            * "delta4"
            * "delta5"
            * "delta6"
            * "leakage_copay"
            * "leakage_coins"
            * "leakage_complex"
    '''
    
    columns = [
        'GPI_NDC',
        'CLIENT',
        'BREAKOUT',
        'MEASUREMENT',
        'REGION',
        'CHAIN_GROUP',
        'CHAIN_SUBGROUP'
    ]
        
    fields = [df[col] for col in columns]

    if cat == 'bin':
        delta_var = [f'D{n}_{gpi_ndc}_{clnt}_{bout}_{meas}_{reg}_{cgrp}_{csubgrp}'
                     for gpi_ndc, clnt, bout, meas, reg, cgrp, csubgrp
                     in zip(*fields)]
            
        vars_out = pd.Series([pulp.LpVariable(dv, cat = 'Binary') for dv in delta_var])
            
    elif cat == 'con':
        leakage_var = [f'leakage_{nm}_{gpi_ndc}_{clnt}_{bout}_{meas}_{reg}_{cgrp}_{csubgrp}'
                       for gpi_ndc, clnt, bout, meas, reg, cgrp, csubgrp
                       in zip(*fields)]   
            
        vars_out = pd.Series([pulp.LpVariable(lv, lowBound = 0, cat = 'Continuous') for lv in leakage_var])
            
    else: 
        assert False, "Unrecognized or no category parameter provided."
        
    return vars_out 

def generateCost_new(lambda_df, gamma, over_reimb_gamma, pen):
    '''
    INPUT:
        lamdba_df is a DataFrame with Pulp Variable lambda (over and under)
        gamma: scalar
    Output:
        LPAffineExpression:
    '''    
    cost_over = lambda_df.Lambda_Over.values*gamma
    cost_under = lambda_df.Lambda_Under.values*gamma
    
    cost = ""
    for i in range(lambda_df.shape[0]):
        if(lambda_df.Lambda_Level.values[i] == 'CLIENT'):
            if lambda_df.PHARMACY_TYPE.values[i] != 'CLIENT':
                if lambda_df.PHARMACY.values[i] != 'MCHOICE':
                    cost += cost_over[i]*(p.PHARM_PERF_WEIGHT + pen)
                elif lambda_df.PHARMACY.values[i] == 'MCHOICE':
                    cost += cost_over[i]*((p.PHARM_PERF_WEIGHT + pen)*p.PHARMACY_MCHOICE_OVRPERF_PEN)
                if p.CAPPED_OPT and (lambda_df.PHARMACY.values[i] in p.OVER_REIMB_CHAINS):
                    cost+= (-1*p.PHARM_PERF_WEIGHT * over_reimb_gamma)*cost_under[i]
                else:
                    if lambda_df.PHARMACY.values[i] == 'CVS':
                        cost += (-1*p.PHARM_PERF_WEIGHT * over_reimb_gamma)*cost_under[i]
                # For COGS optimization, penalize underperformance as well as overperformance
                if lambda_df.PHARMACY.values[i] in p.COGS_PHARMACY_LIST:
                    cost += cost_under[i]*(p.PHARM_PERF_WEIGHT + pen)
            else:
                if 'M' in lambda_df.PHARMACY.values[i].split('_')[1]:
                    cost += cost_over[i] * p.CLIENT_MAIL_OVRPERF_PEN
                    cost += cost_under[i] * p.CLIENT_MAIL_UNRPERF_PEN
                elif lambda_df.PHARMACY.values[i] == 'PRICES':
                    cost += cost_over[i]
                else:
                    cost += cost_under[i] * p.CLIENT_RETAIL_UNRPERF_PEN
                    cost += cost_over[i] * p.CLIENT_RETAIL_OVRPERF_PEN
        else:
            cost += cost_over[i] - cost_under[i]
    return cost


def generate_constraint_prices(
    df: pd.DataFrame,
    lambdaframe: pd.DataFrame,
    *filters: Callable[[pd.DataFrame], pd.Series]
) -> pd.DataFrame:
    """
    Calculates the constraint by filtering the input DataFrame
    and using the input `lambdas` decision variables.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame that must contain the column:
            * "Price_Decision_Var"

    lambdaframe : A pandas DataFrame that must contain the column:
            * "Dec_Var_ID"
            * "LAMBDA_OVER" 
            * "LAMBDA_UNDER" 
    

    filters : variable arguments of Callable type
        Variable arguments, that must be of Callable type.
        The callables must all take 1 argument of type pd.DataFrame
        and return exactly one pd.Series of boolean values (a boolean index).
        These will be used to filter the input DataFrame before it is lined
        up against the lambda decision variables.

    Returns
    -------
    pd.DataFrame
       A copy of a filtered subset of the input DataFrame, with the following
       columns added:
           * "index"; the input DataFrame's index
           * "LAMBDA_OVER"; 1st pulp.LpVariable from input lambdas tuples
           * "LAMBDA_UNDER"; 2nd pulp.LpVariable from input lambdas tuples
           * "CONSTRAINT"; the calculated pulp.LpAffineExpression
        NOTE: the corresponding AWP "target_" is the existing
        "MAC_PRICE_UNIT_ADJ" in the returned DataFrame.

    Examples
    --------
    >>> x = price_vol_df
    >>> y = price_lambdas
    # one condition
    >>> generate_constraint_prices(x, y, lambda df: df.PRICE_MUTABLE == 1)
    # multiple conditions 
    >>> generate_constraint_prices(x, y,
    ...                            lambda df: df.PRICE_MUTABLE == 1,
    ...                            lambda df: (df.PRICE_MUTABLE == 1)
    ...                                       & (df.PRICE_TIER == 'CONFLICT'))
    # predefined condtions
    >>> conditions = [
    ...  lambda df: (df.PRICE_MUTABLE == 1),
    ...  lambda df: (df.PRICE_MUTABLE == 1) & (df.PRICE_TIER == 'CONFLICT')
    ... ]
    >>> generate_constraint_prices(x, y, *conditions)
    """
    
    fltrframes = []
    for i, fltr in enumerate(filters):
        subdf = df.loc[fltr]
        fltrframes.append(subdf)
    mainframe_final_df = pd.merge(pd.concat(fltrframes, ignore_index=True), lambdaframe , on = 'Dec_Var_ID', how = 'left')
        
    mainframe_final_df['CONSTRAINT'] = (mainframe_final_df.Price_Decision_Var * 1
                            + mainframe_final_df.LAMBDA_OVER
                            - mainframe_final_df.LAMBDA_UNDER)
    
    return mainframe_final_df


def generate_constraint_pharm_new(pharmacies: List[str],
                                  price_vol_df: pd.DataFrame,
                                  lambda_df: pd.DataFrame,
                                  pharma_df: pd.DataFrame,
                                  cogs_pharm_list: dict,
                                  cogs_buffer: float,
                                  full_year: bool = False) -> pd.DataFrame:
    """
    NOTE: This function will not mutate the input DataFrames in any way.

    Parameters
    ----------
    pharmacies : list of str
        Names of pharmacies.

    price_vol_df : pd.DataFrame
        DataFrame with the following columns:
            * "CLIENT"
            * "REGION"
            * "BREAKOUT"
            * "BG_FLAG"
            * "CHAIN_GROUP"
            * "PRICE_MUTABLE"
            * "PHARM_EFF_CAPPED_PRICE"
            * "PHARM_QTY_PROJ_EOY"
            * "PHARM_FULLAWP_ADJ_PROJ_EOY"
            * "PHARM_FULLAWP_ADJ_PROJ_LAG"
            * "PHARM_FULLAWP_ADJ"
            * "PRICE_REIMB_LAG"

    lambda_df : pd.DataFrame
        DataFrame of lambda decision variables. Must contain the following fields:
            * "PHARMACY"
            * "Lambda_Over"
            * "Lambda_Under"

    pharma_df : pd.DataFrame
        DataFrame that is used to calculate the guarantee using its "RATE" field.
        Must contain the following columns:
            * "CLIENT"
            * "REGION"
            * "BREAKOUT"
            * "PHARMACY"
            * "BG_FLAG"
            * "RATE"
            * "PHRM_GRTE_TYPE"

    cogs_pharm_list : list of str
        Name(s) of any pharmacy that don't have a guarantee.

    cogs_buffer : float
        The target value when running a COGS optimization.

    full_year : bool, default False
        When True, provides 1/1 prices for the start of the year.

    Returns
    -------
    pd.DataFrame
        A DataFrame with a row for each pharmacy, as well as a field for its
        constraint and another for its target.
    """

    groupers = ['CLIENT', 'REGION', 'BREAKOUT', 'BG_FLAG', 'CHAIN_GROUP']
    pharma2cg = {'PHARMACY': 'CHAIN_GROUP'}

    # target calculation methods
    def dot_price_and_qty(field: pd.Series) -> pd.Series:
        """
        Takes a grouping from price_vol_df and calculates a dot product between
        the "PHARM_EFF_CAPPED_PRICE" and "PHARM_QTY_PROJ_EOY" fields.
        """
        other = ('PHARM_EFF_CAPPED_PRICE'
                 if field.name == 'PHARM_QTY_PROJ_EOY' else 'PHARM_QTY_PROJ_EOY')
        return field.dot(price_vol_df.loc[field.index, other])

    def calc_guarantee(df: pd.DataFrame) -> pd.Series:
        """
        We will assume that any pharmacy in the COGS pharmacy list does not
        have a guarantee. In this situation, we want to use the
        *actual CVS run rate plus a COGS buffer*.

        But for that we have to know the actual CVS run rate!

        Since we already have variables for surplus/liability, the simplest
        way to get the CVS run rate is to take the CVS guarantee and
        add (surplus + liability)/full year AWP. (Since spend is
        target + surplus + liability, and guarantee is 1-target/AWP already).
        """
        in_cogs = df.loc[((df.CHAIN_GROUP.isin(cogs_pharm_list['GNRC'])) & (df.BG_FLAG == 'G')) | ((df.CHAIN_GROUP.isin(cogs_pharm_list['BRND'])) & (df.BG_FLAG == 'B')),'CHAIN_GROUP']
        if not in_cogs.any():
            return df.RATE

        cvs_df = df.loc[df.CHAIN_GROUP == 'CVS']
        awp = (cvs_df.PHARM_FULLAWP_ADJ_PROJ_EOY
               if full_year else
               cvs_df.PHARM_FULLAWP_ADJ_PROJ_EOY
               + cvs_df.PHARM_FULLAWP_ADJ_PROJ_LAG
               + cvs_df.PHARM_FULLAWP_ADJ)
        target_ingcost = (cvs_df.PHARM_TARG_INGCOST_ADJ_PROJ_EOY
               if full_year else
               cvs_df.PHARM_TARG_INGCOST_ADJ_PROJ_EOY
               + cvs_df.PHARM_TARG_INGCOST_ADJ_PROJ_LAG
               + cvs_df.PHARM_TARG_INGCOST_ADJ)

        cvs_guarantee = (1 - round((target_ingcost/awp),4)) + (cvs_df.Lambda_Over - cvs_df.Lambda_Under)/awp + cogs_buffer
        
        # reset index to match df
        cvs_guarantee.index = range(in_cogs.idxmax(),
                                    in_cogs.idxmax() + len(cvs_guarantee))
        
        return df.RATE.fillna(cvs_guarantee.to_dict())

    def calc_target1(df: pd.DataFrame) -> pd.Series:
        
        return ((1 - df.GUARANTEE)
                * df.PHARM_FULLAWP_ADJ_PROJ_EOY
                - df.PRICE_DOT_QTY)

    def calc_cogstarget(df: pd.DataFrame) -> pd.Series:
        in_cogs = df.loc[((df.CHAIN_GROUP.isin(cogs_pharm_list['GNRC'])) & (df.BG_FLAG == 'G')) | ((df.CHAIN_GROUP.isin(cogs_pharm_list['BRND'])) & (df.BG_FLAG == 'B')),'CHAIN_GROUP']
        if not in_cogs.any():
            return df.groupby(['CHAIN_GROUP']).TARGET1.cumsum()

        if full_year:
            target = df.TARGET1 + (1 - df.GUARANTEE) * df.PHARM_FULLAWP_ADJ_PROJ_LAG
        else:
            target = (df.TARGET1
                      + (1 - df.GUARANTEE)
                      * (df.PHARM_FULLAWP_ADJ_PROJ_LAG + df.PHARM_FULLAWP_ADJ)
                      - df.PRICE_REIMB_LAG)

        return (df
                .groupby('CHAIN_GROUP')
                .TARGET1
                .transform('sum')
                .mask(in_cogs, target.loc[in_cogs].sum()))

    # calculate constraints
    constraints = {}
    for pharm in pharmacies:
        const = pulp.LpAffineExpression()

        mut = price_vol_df.loc[(price_vol_df.PRICE_MUTABLE == 1)
                               & (price_vol_df.CHAIN_GROUP == pharm)]
        lambdas = lambda_df.loc[lambda_df.PHARMACY == pharm]

        if pharm in cogs_pharm_list and mut.PHARM_QTY_PROJ_EOY.sum() == 0:
            col = 'QTY_PROJ_EOY'
        else:
            col = 'PHARM_QTY_PROJ_EOY'

        zipped = zip(mut.Price_Decision_Var, mut[col])
        const += pulp.LpAffineExpression(zipped)
        const += lambdas.Lambda_Over.values[0] - lambdas.Lambda_Under.values[0]
        constraints[pharm] = const

    # create DataFrame from dict to join for final output
    # LpAffineExpression must be zipped to be inserted into DataFrame ¯\_(ツ)_/¯
    cons_df = (pd.DataFrame(zip(constraints.keys(), constraints.values()),
                            columns=['CHAIN_GROUP', 'CONSTRAINT'],
                            index=range(len(constraints)))
                .set_index('CHAIN_GROUP')
                .join(lambda_df
                       .rename(columns=pharma2cg)
                       .loc[lambda df: df.CHAIN_GROUP.isin(pharmacies)]
                       .groupby('CHAIN_GROUP')
                       .agg({'Lambda_Over': 'first', 'Lambda_Under': 'first'}),
                      how='right',
                      on='CHAIN_GROUP')
                .reset_index())

    dots = (
     price_vol_df
      .loc[lambda df: df.PRICE_MUTABLE == 0]
      .groupby(groupers)
      .agg(PRICE_DOT_QTY=('PHARM_QTY_PROJ_EOY', dot_price_and_qty)) # gives quantity 
    )

    target_aggs = {
     'RATE': 'first',
     
    'PHARM_FULLAWP_ADJ_PROJ_EOY': 'sum',
     # the remaining aggs below are for when pharmacy in cogs_pharm_list
     'PHARM_FULLAWP_ADJ_PROJ_LAG': 'sum',
     'PHARM_FULLAWP_ADJ': 'sum',   
     
    'PHARM_TARG_INGCOST_ADJ_PROJ_EOY': 'sum',
     # the remaining aggs below are for when pharmacy in cogs_pharm_list
     'PHARM_TARG_INGCOST_ADJ_PROJ_LAG': 'sum',
     'PHARM_TARG_INGCOST_ADJ': 'sum',
        
    'PRICE_REIMB_LAG': 'sum', # spend
        
    }
    
    guaranteesdf = (
     price_vol_df
      .loc[price_vol_df.CHAIN_GROUP.isin(pharmacies)]  # filter by pharmacy list
      .merge(pharma_df.rename(columns=pharma2cg),      # join pharmacy guarantees
             how='left',
             on=groupers)
      .groupby(groupers)[list(target_aggs.keys())]     # calc target aggs
      .agg(target_aggs)
      .join(dots)                                      # join Price.Quantity
      .reset_index()
      .merge(cons_df, on='CHAIN_GROUP', how='right') # join constraints
      .assign(GUARANTEE=calc_guarantee,                # calculate targets
              TARGET1=calc_target1,  # preliminary target calculation
              TARGET=lambda df: calc_cogstarget(df).fillna(0))  # actual target
      .drop_duplicates('CHAIN_GROUP', keep='last')
    )
    
    return guaranteesdf


def generate_constraint_client(price_vol_df, lambda_df, guarantee_df, cons_type):
    '''
    INPUT:
        price_vol_df: DataFrame with Pricing Decision Var, Qty
        lambda_df : DataFrame of lambda_decision variables filtered for either Client or Pharmacy level
        guarantee_dict: Scalar with either Client level Guarantee or Pharmacy level Guarantee
        cons_type: Client or Not_Client_Level
    OUTPUT:
        cons_: pulp.LPAffineExpression
        target_: corresponding target ingredient cost (scalar)
    '''
    
    if p.INTERCEPTOR_OPT or p.COSTSAVER_CLIENT:
        #For immutable drugs we need to assign EXPECTED_KEEP_SEND to CURRENT_KEEP_SEND to accurately calculate the spend
        price_vol_df.loc[price_vol_df.PRICE_MUTABLE == 0, 'EXPECTED_KEEP_SEND'] = price_vol_df.loc[price_vol_df.PRICE_MUTABLE == 0, 'CURRENT_KEEP_SEND']
        
        #We price the fraction of zbd claims that would be sent to GoodRx from immutable pricing differently.
        p_v_df_immut_keep = price_vol_df[(price_vol_df.PRICE_MUTABLE == 0) & (price_vol_df.EXPECTED_KEEP_SEND == 1.0)]
        p_v_df_immut_send = price_vol_df[(price_vol_df.PRICE_MUTABLE == 0) & (price_vol_df.EXPECTED_KEEP_SEND == 0.0)]
        
        #Re-assigns the projected quantities for send claims based on zbd/nonzbd claims
        p_v_df_immut_send_zbd = p_v_df_immut_send.assign(QTY_PROJ_EOY = p_v_df_immut_send['QTY_PROJ_EOY']*p_v_df_immut_send['QTY_ZBD_FRAC'],\
                                                 TARG_INGCOST_ADJ_PROJ_EOY = p_v_df_immut_send['TARG_INGCOST_ADJ_PROJ_EOY']*p_v_df_immut_send['TARG_INGCOST_ADJ_ZBD_FRAC'])

        p_v_df_immut_send_nzbd = p_v_df_immut_send.assign(QTY_PROJ_EOY = p_v_df_immut_send['QTY_PROJ_EOY']*(1 - p_v_df_immut_send['QTY_ZBD_FRAC']),\
                                                TARG_INGCOST_ADJ_PROJ_EOY = p_v_df_immut_send['TARG_INGCOST_ADJ_PROJ_EOY']*(1 - p_v_df_immut_send['TARG_INGCOST_ADJ_ZBD_FRAC']))
        
        
        p_v_df_immut = pd.concat([p_v_df_immut_keep, p_v_df_immut_send_nzbd])
        
        #Check that we are not dropping any quantities     
        assert round(p_v_df_immut_send.QTY_PROJ_EOY.sum(),2) == \
            round(p_v_df_immut_send_zbd.QTY_PROJ_EOY.sum() + p_v_df_immut_send_nzbd.QTY_PROJ_EOY.sum(),2), "Check that we are not dropping any quantities"

        #Check that we are not dropping any projected target ing cost
        assert round(p_v_df_immut_send.TARG_INGCOST_ADJ_PROJ_EOY.sum(),0) == \
            round(p_v_df_immut_send_zbd.TARG_INGCOST_ADJ_PROJ_EOY.sum() + p_v_df_immut_send_nzbd.TARG_INGCOST_ADJ_PROJ_EOY.sum(),0), "Check that we are not dropping any projected target ing cost"

        del p_v_df_immut_send, p_v_df_immut_keep, p_v_df_immut_send_nzbd
        
        #We exclude the fraction of zbd claims that would be sent to GoodRx from mutable pricing.
        p_v_df_mut_keep = price_vol_df[(price_vol_df.PRICE_MUTABLE == 1) & (price_vol_df.EXPECTED_KEEP_SEND == 1.0)]
        p_v_df_mut_send = price_vol_df[(price_vol_df.PRICE_MUTABLE == 1) & (price_vol_df.EXPECTED_KEEP_SEND == 0.0)]

        #Re-assigns the projected quantities for send claims based on zbd/nonzbd claims
        p_v_df_mut_send_zbd = p_v_df_mut_send.assign(QTY_PROJ_EOY = p_v_df_mut_send['QTY_PROJ_EOY']*p_v_df_mut_send['QTY_ZBD_FRAC'],\
                                                 TARG_INGCOST_ADJ_PROJ_EOY = p_v_df_mut_send['TARG_INGCOST_ADJ_PROJ_EOY']*p_v_df_mut_send['TARG_INGCOST_ADJ_ZBD_FRAC'])

        p_v_df_mut_send_nzbd = p_v_df_mut_send.assign(QTY_PROJ_EOY = p_v_df_mut_send['QTY_PROJ_EOY']*(1 - p_v_df_mut_send['QTY_ZBD_FRAC']),\
                                                TARG_INGCOST_ADJ_PROJ_EOY = p_v_df_mut_send['TARG_INGCOST_ADJ_PROJ_EOY']*(1 - p_v_df_mut_send['TARG_INGCOST_ADJ_ZBD_FRAC']))
        
        
        p_v_df_mut = pd.concat([p_v_df_mut_keep, p_v_df_mut_send_nzbd])
        
        #Check that we are not dropping any quantities     
        assert round(p_v_df_mut_send.QTY_PROJ_EOY.sum(),2) == \
            round(p_v_df_mut_send_zbd.QTY_PROJ_EOY.sum() + p_v_df_mut_send_nzbd.QTY_PROJ_EOY.sum(),2), "Check that we are not dropping any quantities"

        #Check that we are not dropping any target ing cost
        assert round(p_v_df_mut_send.TARG_INGCOST_ADJ_PROJ_EOY.sum(),0) == \
            round(p_v_df_mut_send_zbd.TARG_INGCOST_ADJ_PROJ_EOY.sum() + p_v_df_mut_send_nzbd.TARG_INGCOST_ADJ_PROJ_EOY.sum(),0), "Check that we are not dropping any projected target ing cost"

        del p_v_df_mut_send, p_v_df_mut_keep, p_v_df_mut_send_nzbd
        
        #Combine the immutable and mutable zbd send info as they will be priced at Goodrx price
        p_v_send_zbd = pd.concat([p_v_df_immut_send_zbd, p_v_df_mut_send_zbd])
        
        del p_v_df_immut_send_zbd, p_v_df_mut_send_zbd
    
    else:
        p_v_df_mut = price_vol_df[price_vol_df.PRICE_MUTABLE == 1]
        p_v_df_immut= price_vol_df[price_vol_df.PRICE_MUTABLE == 0]
    
    breakout = price_vol_df.iloc[0]['BREAKOUT']
    client = price_vol_df.iloc[0]['CLIENT']
    
    target_ = 0
    logging.debug(client + '_' + breakout)
    for region in price_vol_df.REGION.unique():
        logging.debug(region)
        # HACK: change to accommodate different measurements for different regions
        for measure in price_vol_df.loc[price_vol_df.REGION == region, 'MEASUREMENT'].unique():
            logging.debug(measure)
            for bg_flag in price_vol_df.loc[price_vol_df.REGION == region, 'BG_FLAG'].unique():
                logging.debug(bg_flag)
            
                pref_p_v_mut = p_v_df_mut.loc[(p_v_df_mut['REGION']==region) & (p_v_df_mut['MEASUREMENT']==measure) & (p_v_df_mut['BG_FLAG'] == bg_flag) & (p_v_df_mut['PHARMACY_TYPE']=='Preferred')]
                pref_p_v_immut = p_v_df_immut.loc[(p_v_df_immut['REGION']==region) & (p_v_df_immut['MEASUREMENT']==measure) & (p_v_df_immut['BG_FLAG'] == bg_flag) &(p_v_df_immut['PHARMACY_TYPE']=='Preferred')]
                npref_p_v_mut = p_v_df_mut.loc[(p_v_df_mut['REGION']==region) & (p_v_df_mut['MEASUREMENT']==measure) & (p_v_df_mut['BG_FLAG'] == bg_flag) &(p_v_df_mut['PHARMACY_TYPE']=='Non_Preferred')]
                npref_p_v_immut = p_v_df_immut.loc[(p_v_df_immut['REGION']==region) & (p_v_df_immut['MEASUREMENT']==measure) & (p_v_df_immut['BG_FLAG'] == bg_flag) &(p_v_df_immut['PHARMACY_TYPE']=='Non_Preferred')]

                if p.INTERCEPTOR_OPT or p.COSTSAVER_CLIENT:
                    pref_p_v_send_zbd = p_v_send_zbd.loc[(p_v_send_zbd['REGION']==region) & (p_v_send_zbd['MEASUREMENT']==measure) & (p_v_send_zbd['BG_FLAG'] == bg_flag) & (p_v_send_zbd['PHARMACY_TYPE']=='Preferred')]
                    npref_p_v_send_zbd = p_v_send_zbd.loc[(p_v_send_zbd['REGION']==region) & (p_v_send_zbd['MEASUREMENT']==measure) & (p_v_send_zbd['BG_FLAG'] == bg_flag) & (p_v_send_zbd['PHARMACY_TYPE']=='Non_Preferred')]

                    target_pref = (pref_p_v_mut.TARG_INGCOST_ADJ_PROJ_EOY.sum() + pref_p_v_immut.TARG_INGCOST_ADJ_PROJ_EOY.sum() + pref_p_v_send_zbd.TARG_INGCOST_ADJ_PROJ_EOY.sum())\
                            - np.dot(pref_p_v_immut.EFF_CAPPED_PRICE.values, pref_p_v_immut.QTY_PROJ_EOY.values) - np.dot(pref_p_v_send_zbd.VENDOR_PRICE.fillna(0).values, pref_p_v_send_zbd.QTY_PROJ_EOY.values)

                    target_nonpref = (npref_p_v_mut.TARG_INGCOST_ADJ_PROJ_EOY.sum() + npref_p_v_immut.TARG_INGCOST_ADJ_PROJ_EOY.sum() + npref_p_v_send_zbd.TARG_INGCOST_ADJ_PROJ_EOY.sum())\
                            - np.dot(npref_p_v_immut.EFF_CAPPED_PRICE.values, npref_p_v_immut.QTY_PROJ_EOY.values) - np.dot(npref_p_v_send_zbd.VENDOR_PRICE.fillna(0).values, npref_p_v_send_zbd.QTY_PROJ_EOY.values)

                else:
                    target_pref = (pref_p_v_mut.TARG_INGCOST_ADJ_PROJ_EOY.sum() + pref_p_v_immut.TARG_INGCOST_ADJ_PROJ_EOY.sum())\
                            - np.dot(pref_p_v_immut.EFF_CAPPED_PRICE.values, pref_p_v_immut.QTY_PROJ_EOY.values)

                    target_nonpref = (npref_p_v_mut.TARG_INGCOST_ADJ_PROJ_EOY.sum() + npref_p_v_immut.TARG_INGCOST_ADJ_PROJ_EOY.sum())\
                            - np.dot(npref_p_v_immut.EFF_CAPPED_PRICE.values, npref_p_v_immut.QTY_PROJ_EOY.values)

                logging.debug('Pref Target Ing Cost: %f', (pref_p_v_mut.TARG_INGCOST_ADJ_PROJ_EOY.sum() + pref_p_v_immut.TARG_INGCOST_ADJ_PROJ_EOY.sum() + pref_p_v_send_zbd.TARG_INGCOST_ADJ_PROJ_EOY.sum()) if (p.INTERCEPTOR_OPT or p.COSTSAVER_CLIENT) \
                                              else (pref_p_v_mut.TARG_INGCOST_ADJ_PROJ_EOY.sum() + pref_p_v_immut.TARG_INGCOST_ADJ_PROJ_EOY.sum()))
                logging.debug('Pref Uncontrolled Spend: %f', np.dot(pref_p_v_immut.EFF_CAPPED_PRICE.values, pref_p_v_immut.QTY_PROJ_EOY.values))

                logging.debug('NPref Target Ing Cost: %f', (npref_p_v_mut.TARG_INGCOST_ADJ_PROJ_EOY.sum() + npref_p_v_immut.TARG_INGCOST_ADJ_PROJ_EOY.sum() + npref_p_v_send_zbd.TARG_INGCOST_ADJ_PROJ_EOY.sum()) if (p.INTERCEPTOR_OPT or p.COSTSAVER_CLIENT) \
                              else (npref_p_v_mut.TARG_INGCOST_ADJ_PROJ_EOY.sum() + npref_p_v_immut.TARG_INGCOST_ADJ_PROJ_EOY.sum()))
                logging.debug('NPref Uncontrolled Spend: %f', np.dot(npref_p_v_immut.EFF_CAPPED_PRICE.values, npref_p_v_immut.QTY_PROJ_EOY.values))

                if p.INTERCEPTOR_OPT or p.COSTSAVER_CLIENT:
                    logging.debug('Pref Costsaver Uncontrolled Spend: %f', np.dot(pref_p_v_send_zbd.VENDOR_PRICE.values, pref_p_v_send_zbd.QTY_PROJ_EOY.values))
                    logging.debug('NPref Costsaver Uncontrolled Spend: %f', np.dot(npref_p_v_send_zbd.VENDOR_PRICE.values, npref_p_v_send_zbd.QTY_PROJ_EOY.values))

                target_ += target_pref + target_nonpref
            
    cons_ = LpAffineExpression([(p_v_df_mut.Price_Decision_Var.values[i], p_v_df_mut.QTY_PROJ_EOY.values[i]) for i in range(p_v_df_mut.shape[0])])  - lambda_df.Lambda_Under.values[0] + lambda_df.Lambda_Over.values[0]


    return cons_, target_


def generateGuaranteeConstraintEbit(price_vol_df, lambda_df, breakout_guarantees, pharmacy_guarantees, perf_dict, breakout_df, client_list, pharmacy_list, cogs_pharmacy_list, cogs_buffer, eoy_pharm, eoy_gen_launch, eoy_brand_performance, eoy_specialty_performance, eoy_disp_fee_performance, price_lambdas_df):
    '''
    Inputs:
        price_vol_df: DataFrame with Pricing Decision Var, Qty
        lambda_df : DataFrame of lambda_decision variables
        guarantee_dict: Dictionary with Guarantees
        perf_dict: Dictionary of performance for clients and pharmacies
        client_list: List of client breakouts
        pharmacy_list: List of pharmacies
    Output:
        Generate a list with the constraints
        Generate a list with Guarantee AWP
    '''
    import time
    if isinstance(eoy_brand_performance,pd.DataFrame):
        if len(eoy_brand_performance) <= 1:
            eoy_brand_performance = eoy_brand_performance.set_index('ENTITY').squeeze(axis=1).to_dict()
        else:
            eoy_brand_performance = eoy_brand_performance.set_index('ENTITY').squeeze().to_dict()
    if isinstance(eoy_specialty_performance,pd.DataFrame):
        if len(eoy_specialty_performance) <= 1:
            eoy_specialty_performance = eoy_specialty_performance.set_index('ENTITY').squeeze(axis=1).to_dict()
        else:
            eoy_specialty_performance = eoy_specialty_performance.set_index('ENTITY').squeeze().to_dict()
    if isinstance(eoy_disp_fee_performance,pd.DataFrame):
        if len(eoy_disp_fee_performance) <= 1:
            eoy_disp_fee_performance = eoy_disp_fee_performance.set_index('ENTITY').squeeze(axis=1).to_dict()
        else:
            eoy_disp_fee_performance = eoy_disp_fee_performance.set_index('ENTITY').squeeze().to_dict()

    guarantees = generate_constraint_pharm_new(pharmacy_list,
                                               price_vol_df,
                                               lambda_df,
                                               pharmacy_guarantees,
                                               cogs_pharmacy_list,
                                               cogs_buffer,
                                               p.FULL_YEAR)
    constraint, awp_g = [], []
    logging.debug("Building Pharmacy Constraints")

    start = time.time()
    for pharmacy in pharmacy_list:
        gdf = guarantees.loc[guarantees.CHAIN_GROUP == pharmacy]
        constraint.append(gdf.CONSTRAINT.iloc[0])

        awp_ = gdf.TARGET.iloc[-1]
        if pharmacy in cogs_pharmacy_list:
            # we have already taken care of EOY spend in the generate_constraint function
            t_awp_g = awp_
        elif pharmacy in eoy_pharm:
            t_awp_g = awp_ + eoy_pharm[pharmacy]
        else:
            raise RuntimeError(f"Pharmacy {pharmacy} not found in eoy_pharm_dict")
        # for perf_name in price_vol_df.loc[price_vol_df.CHAIN_GROUP == pharmacy, 'CHAIN_SUBGROUP'].unique():
        #     # We didn't do gen launches above, so if those numbers exist, fine to accommodate them here.
        #     if perf_name in perf_dict:
        #         t_awp_g += perf_dict[perf_name] + eoy_gen_launch[perf_name]
        #     elif pharmacy in cogs_pharmacy_list:
        #         # But we won't worry about missing gen launches for COGS pharmacies, since this isn't always available in our inputs.
        #         continue
        #     else:
        #         raise RuntimeError(f"Pharmacy {pharmacy} not found in eoy_gen_launch")
        awp_g.append(t_awp_g)

    end = time.time()

    logging.debug("End  Building Pharmacy Constraints")
    logging.debug("Run time: {} mins".format((end - start)/60.))

    logging.debug("Building Breakout Constraints")
    start = time.time()
    for i in range(breakout_df.shape[0]):
        breakout_row = breakout_df.iloc[i]
        breakout_lambda = lambda_df.loc[lambda_df.PHARMACY==breakout_row.Combined]
        b_price_vol_df = price_vol_df.loc[(price_vol_df.CLIENT == breakout_row.CLIENT) & (price_vol_df.BREAKOUT == breakout_row.BREAKOUT)]
        cons_, awp_ = generate_constraint_client(b_price_vol_df, breakout_lambda, breakout_guarantees, 'CLIENT')
        constraint.append(cons_)
        awp_g.append(awp_ + perf_dict[breakout_row.Combined] + eoy_gen_launch[breakout_row.Combined] + eoy_brand_performance[breakout_row.Combined] + \
                     eoy_specialty_performance[breakout_row.Combined] + eoy_disp_fee_performance[breakout_row.Combined])

    end = time.time()

    logging.debug("End Building Breakout Constraints")
    logging.debug("Run time: {} mins".format((end - start)/60.))
    
    price_conditions = [
     # no_util condition
     lambda df: (df.PRICE_MUTABLE == 1) & (df.PRICING_QTY_PROJ_EOY == 0),
     # util_conflict condition
     lambda df: (df.PRICE_MUTABLE == 1)
                & (df.PRICING_QTY_PROJ_EOY != 0)
                & (df.PRICE_TIER == 'CONFLICT'),
     # util condition
     lambda df: (df.PRICE_MUTABLE == 1)
                & (df.PRICING_QTY_PROJ_EOY != 0)
                & (df.PRICE_TIER != 'CONFLICT'),
    ]
    constprices = generate_constraint_prices(price_vol_df,
                                             price_lambdas_df,
                                             *price_conditions)

    constraint.extend(constprices.CONSTRAINT.to_list())
    awp_g.extend(constprices.SOFT_CONST_BENCHMARK_PRICE.to_list())
    
    return constraint, awp_g


def pharmacy_type_new(row, pref_pharm_list):
    '''
    Takes dataframe row and determines if the chain group is preferrred or not based on the supplied list of preferred pharmacies
    Inputs:
        Row of a dataframe that contains the CLIENT, BREAKOUT, and CHAIN_GROUP of that row
        A dataframe that has a list of preferred pharmacies based on client, breakout, & region
    Outputs:
        A single string of "Preferred" or "Non_Preferred"
    '''
    pref_pharms = pref_pharm_list.loc[(pref_pharm_list.CLIENT == row.CLIENT) &
                                      (pref_pharm_list.BREAKOUT == row.BREAKOUT) &
                                      (pref_pharm_list.REGION == row.REGION), 'PREF_PHARMS'].values[0]


    if row.CHAIN_GROUP in pref_pharms:
        return 'Preferred'
    else:
        return 'Non_Preferred'

    
def gen_launch_df_generator_ytd_lag_eoy(generic_launch_df, pref_pharm_list_df):
    '''
    Estimates the gen_launch_ytd, gen_launch_lag, gen_launch_eoy from the generic_launch_df.  The split between ytd, lag and eoy are on a day granularity.
    It considers the day of LAST_DATA part of ytd and the day of GO_LIVE part of eoy, that way no day is double counted.
    
    Inputs: 
        generic_launch_df -- df with the generic launch impact for a list of clients
        pref_pharm_list_df -- df with the prefer pharmacy for each client
        
    Outputs:
        gen_launch_ytd -- the ytd generic launch (usualy 0 or small)
        gen_launch_lag -- the lag gneric launch impact
        gen_launch_eoy -- the eoy gneric launch impact
    '''
    from calendar import monthrange
    
    if p.LAST_DATA == p.GO_LIVE:
        raise AssertionError('If equal gen_launch_lag will not be correct')
    
    if p.LAST_DATA > p.GO_LIVE:
        raise AssertionError('Dates are in the wrong order')
        
    if p.LAST_DATA.year != p.GO_LIVE.year and not p.FULL_YEAR:
        raise AssertionError('Dates have different years')

    if p.FULL_YEAR:
        gen_launch_ytd = generic_launch_df.groupby(['CLIENT', 'BREAKOUT', 'REGION',
                                                    'MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP','CHAIN_SUBGROUP'])['FULLAWP', 'ING_COST'].agg('sum').reset_index()
        gen_launch_lag = generic_launch_df.groupby(['CLIENT', 'BREAKOUT', 'REGION',
                                                    'MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP','CHAIN_SUBGROUP'])['FULLAWP', 'ING_COST'].agg('sum').reset_index()
        gen_launch_eoy = generic_launch_df.groupby(['CLIENT', 'BREAKOUT', 'REGION',
                                                    'MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP','CHAIN_SUBGROUP'])['FULLAWP', 'ING_COST'].agg('sum').reset_index()
    else:
        
        LAST_DATA_month_len = monthrange(p.LAST_DATA.year, p.LAST_DATA.month)[1]
        GO_LIVE_month_len = monthrange(p.GO_LIVE.year, p.GO_LIVE.month)[1]
        
        gen_launch_ytd = generic_launch_df.loc[generic_launch_df.MONTH <= p.LAST_DATA.month]
        temp_ytd = gen_launch_ytd.loc[gen_launch_ytd.MONTH == p.LAST_DATA.month][['FULLAWP','ING_COST']]
        gen_launch_ytd.loc[gen_launch_ytd.MONTH == p.LAST_DATA.month, ['FULLAWP','ING_COST']] = temp_ytd - ((LAST_DATA_month_len - p.LAST_DATA.day) / LAST_DATA_month_len) * temp_ytd
        gen_launch_ytd = gen_launch_ytd.groupby(['CLIENT', 'BREAKOUT', 'REGION','MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP','CHAIN_SUBGROUP'])['FULLAWP','ING_COST'].agg('sum').reset_index()
                                                                                                      
    

        gen_launch_lag = generic_launch_df.loc[(generic_launch_df.MONTH >= p.LAST_DATA.month) & (generic_launch_df.MONTH <= p.GO_LIVE.month)]        
        temp_lag_last_data = gen_launch_lag.loc[gen_launch_lag.MONTH == p.LAST_DATA.month][['FULLAWP','ING_COST']]
        temp_lag_go_live = gen_launch_lag.loc[gen_launch_lag.MONTH == p.GO_LIVE.month][['FULLAWP','ING_COST']]
        
        if p.GO_LIVE.month != p.LAST_DATA.month:
            gen_launch_lag.loc[gen_launch_lag.MONTH == p.LAST_DATA.month, ['FULLAWP','ING_COST']] = temp_lag_last_data - (p.LAST_DATA.day / LAST_DATA_month_len) * temp_lag_last_data
            gen_launch_lag.loc[gen_launch_lag.MONTH == p.GO_LIVE.month, ['FULLAWP','ING_COST']] = temp_lag_go_live - ((GO_LIVE_month_len - p.GO_LIVE.day + 1) / GO_LIVE_month_len) * temp_lag_go_live
        else:
            gen_launch_lag.loc[gen_launch_lag.MONTH == p.LAST_DATA.month, ['FULLAWP','ING_COST']] = temp_lag_last_data \
                    - ((p.LAST_DATA.day) / LAST_DATA_month_len) * temp_lag_last_data \
                    - ((GO_LIVE_month_len - p.GO_LIVE.day + 1) / GO_LIVE_month_len) * temp_lag_go_live
            
        gen_launch_lag = gen_launch_lag.groupby(['CLIENT', 'BREAKOUT', 'REGION','MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP','CHAIN_SUBGROUP'])['FULLAWP','ING_COST'].agg('sum').reset_index()
                    


        gen_launch_eoy = generic_launch_df.loc[generic_launch_df.MONTH >= p.GO_LIVE.month]
        temp_eoy = gen_launch_eoy.loc[gen_launch_eoy.MONTH == p.GO_LIVE.month][['FULLAWP','ING_COST']]
        gen_launch_eoy.loc[gen_launch_eoy.MONTH == p.GO_LIVE.month, ['FULLAWP','ING_COST']] = ((GO_LIVE_month_len - p.GO_LIVE.day + 1) / GO_LIVE_month_len) * temp_eoy
        gen_launch_eoy = gen_launch_eoy.groupby(['CLIENT', 'BREAKOUT', 'REGION','MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP','CHAIN_SUBGROUP'])['FULLAWP','ING_COST'].agg('sum').reset_index()                                                                                             
                                                                                                                                                                                                                                                                                
    
    gen_launch_ytd['PHARMACY_TYPE'] = gen_launch_ytd.apply(pharmacy_type_new, args=tuple([pref_pharm_list_df]), axis=1)
    gen_launch_lag['PHARMACY_TYPE'] = gen_launch_lag.apply(pharmacy_type_new, args=tuple([pref_pharm_list_df]), axis=1)
    gen_launch_eoy['PHARMACY_TYPE'] = gen_launch_eoy.apply(pharmacy_type_new, args=tuple([pref_pharm_list_df]), axis=1) 
        
    
    return gen_launch_ytd, gen_launch_lag, gen_launch_eoy


def price_overrider_function(price_override, lp_df):
    '''
    Updates lp_df's CURRENT_MAC_PRICE with PRICE_OVRD_AMT from the <price_override> dataframe
    Fixes overriden prices by setting 'PRICE_MUTABLE' = 0
    In:
        price_override -- df with the price override on a client, vcml and gpi level
        lp_df -- df with all of the information need for the LP
    Out:
        lp_df -- with 'CURRENT_MAC_PRICE' and 'PRICE_MUTABLE' updated
    '''

    # NOTE: below has to be rethought if price_override include any region beside 'ALL'
    #       probably, take region into account when updating prices

    
    if 'REGION' in price_override.columns:
        if len(set(price_override['REGION']) - set(['ALL'])) > 0:
            raise Exception('There are unique regions (besides "ALL") that are not properly handled')

    price_override = price_override[['CLIENT', 'GPI', 'VCML_ID', 'PRICE_OVRD_AMT', 'NDC' , 'BG_FLAG']]
    price_override['CURRENT_MAC_PRICE'] = price_override['PRICE_OVRD_AMT']
    price_override['PRICE_MUTABLE'] = 0

    # keep original size for later sanity check: updating prices should not add any new rows
    original_size = lp_df.shape

    # partition price_override based on VCML_ID
    temp_override_all = price_override[price_override['VCML_ID'] == 'ALL'].drop(['CLIENT', 'VCML_ID'], axis = 1)
    temp_override_client = price_override[price_override['VCML_ID'] != 'ALL'].drop(['CLIENT'], axis = 1)

    # override prices for all VCML_IDs
    temp_update = lp_df[['CLIENT', 'GPI', 'MAC', 'NDC','BG_FLAG']].merge(temp_override_all, 
                                                                    on = ['GPI', 'NDC', 'BG_FLAG'],
                                                                    how = 'left',
                                                                    suffixes = ('_left', ''))
    lp_df.update(temp_update)

    # override prices for specific VCML_IDs
    temp_update = lp_df[['CLIENT', 'GPI', 'MAC', 'NDC','BG_FLAG']].merge(temp_override_client,
                                                                    left_on = ['GPI', 'MAC', 'NDC','BG_FLAG'],
                                                                    right_on = ['GPI', 'VCML_ID', 'NDC','BG_FLAG'],
                                                                    how = 'left',
                                                                    suffixes = ('_left', ''))
    lp_df.update(temp_update)
    
    # Leaving NDC-level prices different will cause NDC-level outputs in our price output files
    # but we still want to go through this function to mark them immutable
    lp_df.loc[(lp_df['NDC']!='***********') & (lp_df['GPI'].isin(price_override['GPI'])), 'CURRENT_MAC_PRICE'] = lp_df.loc[
        (lp_df['NDC']!='***********') & (lp_df['GPI'].isin(price_override['GPI'])), 'OLD_MAC_PRICE']

    # sanity check
    assert original_size == lp_df.shape, 'ERROR, the function is adding un intended rows or columns'
    return lp_df