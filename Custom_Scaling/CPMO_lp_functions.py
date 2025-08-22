# -*- coding: utf-8 -*-
"""
Linear program functions file for CLIENT_PHARMACY_MAC_OPTIMIZATION

@author: JOHN WALKER
@version: 1.1.0, 01.27.2020
"""
import CPMO_parameters as p
import pandas as pd
import numpy as np
import time
import logging

from pulp import *


def priceBound(df, low_fac, up_fac, foy_dict, month):
    '''
    This takes a dataframe row of pricing information and returns the price bounds for that entry.
    This function is designed to be used with apply accross a dataframe
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

    '''

#    logging.debug("Lower Factor = {}\n, Upper Factor {}".format(low_fac, up_fac))
#
    if False:#df.CHAIN_GROUP in p.CAPPED_PHARMACY_LIST:
        floor_1026 = 0
    else:
        floor_1026 = df.MAC1026_UNIT_PRICE

#   logging.debug('%f',floor_1026)

    if p.TIERED_PRICE_LIM and (df.CLIENT in p.TIERED_PRICE_CLIENT):
        start_of_year = True
        # logging.debug('%s Start of year pricing!', df.CLIENT)
    elif df.GPI_CHANGE_EXCEPT == 1:
        start_of_year = True
    else:
       start_of_year = False

#    if (df.REGION in ['REG_PLUS', 'REG_ALLURE']) & (MONTH in foy_dict['WELLCARE']):
#        start_of_year == True

    AVG_FAC = 0.75
    member_disruption_amt = 20000
    member_disruption_bound = 100000

#
##### code for price_bounds based on cost per Rx -  Minimize Member disruption #######

#
#    if(df.PRICE_REIMB_CLAIM <= 250):
#        up_fac = 0.4
#        member_disruption_amt = 100
##

    if df.QTY_PROJ_EOY == 0:
        if p.ZERO_QTY_TIGHT_BOUNDS:
            low_fac = p.GPI_LOW_FAC + 0.05
            if p.TIERED_PRICE_LIM:
                # It shold the the max on the tier prices that we want, until is automated should be check by hand
                up_fac = 1.0

            else:
                up_fac = p.GPI_UP_FAC
        else:
            up_fac = 1000
            low_fac = .95

    #if df.QTY_PROJ_EOY == 0:
    #    up_fac = 1000
    #    low_fac = .95

    elif start_of_year:
        if False:#df.CHAIN_GROUP in ['CVS', 'PREF_OTH']:
            low_fac = 0.00
        else:
            low_fac = p.GPI_LOW_FAC #lower bound of the price change percentage

        if df.PRICE_TIER == "0":
            if(df.PRICE_REIMB_CLAIM > 100):
                # up_fac = .15
                up_fac = .1
                #member_disruption_amt = df.PRICE_REIMB_CLAIM * up_fac

            if(df.PRICE_REIMB_CLAIM <= 100):
                # up_fac = .25
                up_fac = .2
                #member_disruption_amt = up_fac*100

            if(df.PRICE_REIMB_CLAIM <= 50):
                # up_fac = .5
                up_fac = .3
                #member_disruption_amt = 6 * up_fac

            if(df.PRICE_REIMB_CLAIM <= 25):
                # up_fac = .5
                up_fac = .4
                #member_disruption_amt = 6 * up_fac

            if(df.PRICE_REIMB_CLAIM <= 10):
                # up_fac = .5
                up_fac = 1.0
                #member_disruption_amt = 6 * up_fac

            if(df.PRICE_REIMB_CLAIM <= 5):
                up_fac = 200
                # member_disruption_amt = 1.5
                member_disruption_amt = 5

        elif df.PRICE_TIER == "1":
            if(df.PRICE_REIMB_CLAIM > 100):
                up_fac = .5
                #member_disruption_amt = df.PRICE_REIMB_CLAIM * up_fac

            if(df.PRICE_REIMB_CLAIM <= 100):
                up_fac = .6
                #member_disruption_amt = up_fac*100


            if(df.PRICE_REIMB_CLAIM <= 6):
                up_fac = 1.2
                #member_disruption_amt = 6 * up_fac


            if(df.PRICE_REIMB_CLAIM <= 3):
                up_fac = 1.00
                #member_disruption_amt = 3*up_fac
        elif df.PRICE_TIER == "2":
            if(df.PRICE_REIMB_CLAIM > 100):
                up_fac = .25
                #member_disruption_amt = df.PRICE_REIMB_CLAIM * up_fac

            if(df.PRICE_REIMB_CLAIM <= 100):
                up_fac = .4
                #member_disruption_amt = up_fac*100


            if(df.PRICE_REIMB_CLAIM <= 6):
                up_fac = .6
                #member_disruption_amt = 6 * up_fac


            if(df.PRICE_REIMB_CLAIM <= 3):
                up_fac = .83333333
                #member_disruption_amt = 3*up_fac
        elif df.PRICE_TIER == "3":
            if(df.PRICE_REIMB_CLAIM > 100):
                up_fac = .15
                #member_disruption_amt = df.PRICE_REIMB_CLAIM * up_fac

            if(df.PRICE_REIMB_CLAIM <= 100):
                up_fac = .3
                #member_disruption_amt = up_fac*100


            if(df.PRICE_REIMB_CLAIM <= 6):
                up_fac = .4
                #member_disruption_amt = 6 * up_fac


            if(df.PRICE_REIMB_CLAIM <= 3):
                up_fac = .6666666
                #member_disruption_amt = 3*up_fac
#        elif df.PRICE_TIER == "4":
#            return (df.MAC_PRICE_UNIT_Adj, df.MAC_PRICE_UNIT_Adj)

#            if(df.PRICE_REIMB_CLAIM > 100):
#                up_fac = .05
#                #member_disruption_amt = df.PRICE_REIMB_CLAIM * up_fac
#
#            if(df.PRICE_REIMB_CLAIM <= 100):
#                up_fac = .1
#                #member_disruption_amt = up_fac*100
#
#
#            if(df.PRICE_REIMB_CLAIM <= 6):
#                up_fac = .2
#                #member_disruption_amt = 6 * up_fac
#
#
#            if(df.PRICE_REIMB_CLAIM <= 3):
#                up_fac = .2
#                #member_disruption_amt = 3*up_fac
        else:
            logging.debug("Pricing Tier Error")
    else:
        up_fac = p.GPI_UP_FAC
        low_fac = p.GPI_LOW_FAC
        #member_disruption_amt = df.PRICE_REIMB_CLAIM * up_fac

    #

    if df.QTY_PROJ_EOY > 0:
        if p.SIM and (month > 6):
            pass
        else:
            member_disruption_bound = (df.CLAIMS_PROJ_EOY / df.QTY_PROJ_EOY) * (
                        df.PRICE_REIMB_CLAIM + member_disruption_amt)
    # # HACK: added full year
    # if p.FULL_YEAR:
    #     member_disruption_bound = 1000000
    ##### End code for price_bounds based on cost per Rx -  Minimize Member disruption #######

    if (df.PRICE_MUTABLE == 0):
        return (df.MAC_PRICE_UNIT_ADJ, df.MAC_PRICE_UNIT_ADJ)

    if (low_fac == 0) & (up_fac == 0) & (floor_1026 > df.AVG_AWP * AVG_FAC):  ### AVG_AWP is incorrect
        return (floor_1026, max(floor_1026, df.MAC_PRICE_UNIT_ADJ))

    if (low_fac == 0) & (up_fac == 0) & (floor_1026 <= df.AVG_AWP * AVG_FAC):
        return (floor_1026, max(df.AVG_AWP * AVG_FAC, df.MAC_PRICE_UNIT_ADJ))

    if (floor_1026 == 0) & (low_fac == 0):
        lower_bound = df.MAC_PRICE_UNIT_ADJ * 0.20  ### dont go below MAC1026 price
    else:
        lower_bound = np.nanmax(
            [floor_1026, df.MAC_PRICE_UNIT_ADJ * (1 - low_fac), .0001])  # .995 is the limit to find outlier claims
    ### GIT - removed the df.breakout_awp_max*0.0075 to remove outliers
    # HACK: remove MAC1026 for Capped
    # if df.CHAIN_GROUP in p.CAPPED_PHARMACY_LIST:
    #     lower_bound = np.nanmax([df.MAC_PRICE_UNIT_ADJ*(1-low_fac), .0001])

    # HACK: For Molina
    # if df.CHAIN_GROUP in p.CAPPED_PHARMACY_LIST and df.REGION == 'MOLINA':
    #     lower_bound = np.nanmax([df.MAC_PRICE_UNIT_ADJ*(1-low_fac), .0001])

    # HACK: Clearstone -- Remove MAC1026 for Capped
    # if df.CHAIN_GROUP in p.CAPPED_PHARMACY_LIST and df.PHARMACY_TYPE == 'Preferred' and df.REGION != "EGWP":
    #     lower_bound = np.nanmax([df.MAC_PRICE_UNIT_ADJ * (1 - low_fac), .0001])
    # elif df.CHAIN_GROUP in p.CAPPED_PHARMACY_LIST and df.REGION == "EGWP":
    # elif df.CHAIN_GROUP in p.CAPPED_PHARMACY_LIST and df.REGION == "EGWP":
    #     lower_bound = np.nanmax([df.MAC_PRICE_UNIT_ADJ * (1 - low_fac), .0001])
    # else:
    #     lower_bound = lower_bound

    # HACK: TUFTS
    # if df.CHAIN_GROUP == 'MAIL':
    #     lower_bound = np.nanmax(
    #         [df.MAC_PRICE_UNIT_ADJ * (1 - 0.9), .0001])

    # incorporate the WellCare Suggestions
    if df.MEASUREMENT == 'M30':
        lower_bound = np.nanmax([df.MAC_PRICE_UNIT_ADJ * (1 - low_fac),.0001])  # We do not manage mail so mail can go as low as necessary, use 0.9 low factor for Tufts

    # elif (df.CLIENT_MIN_PRICE >= 0.0001) and (df.CLIENT_MIN_PRICE < lower_bound):
    #     lower_bound = max(floor_1026, df.CLIENT_MIN_PRICE)
    # else:
    # lower_bound = max(lower_bound, df.CLIENT_MIN_PRICE)

    #    if df.CHAIN_GROUP == 'CVS':
    #        lower_bound = 0
    ##
    #    if df.MAC_PRICE_UNIT_ADJ < lower_bound: ## if the mac1026 price is greater than the effective mac price then set lower bound to 0
    ##        lower_bound = df.MAC_PRICE_UNIT_ADJ*0.25
    #        lower_bound = 0

#    upper_bound = max(lower_bound+0.0001,min(df.AVG_AWP*AVG_FAC,df.MAC_PRICE_UNIT_ADJ*(1.0+up_fac), df.uc_unit, member_disruption_bound))  ### dont go above awp unit price
#    logging.debug(lower_bound)
#    logging.debug(min(df.AVG_AWP*AVG_FAC,df.MAC_PRICE_UNIT_ADJ*(1.0+up_fac), df.uc_unit, member_disruption_bound))
#     if df.CLIENT_MAX_PRICE < 9998:
#         client_max = df.CLIENT_MAX_PRICE
#     else:
#         if df.MAC_PRICE_UNIT_Adj >= lower_bound:
#             client_max = df.MAC_PRICE_UNIT_Adj*(1.0+up_fac)
#         else:
# #            client_max = lower_bound*(1.0+up_fac)
#             client_max = df.MAC_PRICE_UNIT_Adj*(1.0+up_fac)

    if (df.UC_UNIT > 0) and (df.AVG_AWP > 0) and (df.QTY_PROJ_EOY > 0):
        upper_bound = max(lower_bound+0.0001, min(df.AVG_AWP*AVG_FAC, df.MAC_PRICE_UNIT_ADJ*(1.0+up_fac), df.UC_UNIT, member_disruption_bound, df.CLIENT_MAX_PRICE))  ### dont go above awp unit price
    elif df.QTY_PROJ_EOY == 0:
        upper_bound = max(lower_bound+0.0001, min(member_disruption_bound, df.MAC_PRICE_UNIT_ADJ*(1.0+up_fac), df.CLIENT_MAX_PRICE))
    else:
        upper_bound = max(lower_bound+0.0001, min(df.MAC_PRICE_UNIT_ADJ*(1.0+up_fac), member_disruption_bound, df.AVG_AWP*AVG_FAC, df.CLIENT_MAX_PRICE))  ### dont go above awp unit price

#        upper_bound = max(lower_bound+0.0001,min(df.AVG_AWP*AVG_FAC, df.uc_unit, member_disruption_bound, lower_bound*1.5))  ### dont go above awp unit price
#    if upper_bound < df.MAC_PRICE_UNIT_ADJ:
#        upper_bound = df.MAC_PRICE_UNIT_ADJ*(1+up_fac)

#    logging.debug(upper_bound)

    #incorporate client suggested upperbound
    if df.CLIENT_MIN_PRICE >= (upper_bound - 0.0001):
        if (upper_bound * .95) > lower_bound:
            lower_bound = upper_bound * .95
    elif df.QTY_PROJ_EOY == 0:
        lower_bound = lower_bound
    else:
        lower_bound = np.nanmax([lower_bound, df.CLIENT_MIN_PRICE])

    if df.CLIENT_MAX_PRICE < lower_bound:
        upper_bound = lower_bound + (.05 * df.MAC_PRICE_UNIT_ADJ)
        logging.debug('Error with client supplied upper bound at %s %s %s', df.REGION, df.CHAIN_GROUP, df.GPI_NDC)

    if (upper_bound == 0) or (lower_bound == 0):
        logging.debug('Error with bounds at %s', df.GPI_NDC)

#    logging.debug(lower_bound, upper_bound)
    return (lower_bound, upper_bound)


def generatePriceBounds(lower_scale_factor,upper_scale_factor, price_data, foy_dict, month):
    '''
    Inputs:
        scale_factor - scalar: which is a factor by which prices can vary
        price_data: dataframe with existing_prices, mac1026 prices, awp_price

    Output:
        Return a dataframe with lower bound and upper bound for the price

    '''
    price_data['Price_Bounds'] = price_data.apply(priceBound, args=(lower_scale_factor,upper_scale_factor,foy_dict, month), axis=1)

    return price_data['Price_Bounds']


def lb_ub(df):
    '''
    This takes a row that contains the Price_Bounds column and checks if the lower bound
    is lower than the upper bound.  If this logic is violated then return 1.
    Inputs:
        df - dataframe row that contains the Price_Bounds column that is a tuple of lowerbound and upper bound in that order
    Output:
        An integer of 1 if the lower bound is greater than the upper bound or 0 if there is no bound violation
    '''
    lb, ub = df.Price_Bounds
    if lb > ub:
        return 1
    else:
        return 0

def current_price_conflict(df):
    '''
    This checks to make sure the upperbound is greater than the current price and the lower bound is lower than the current price
    '''
    lb, ub = df.Price_Bounds
    if df.CURRENT_MAC_PRICE > ub:
        return 1
#    elif df.CURRENT_MAC_PRICE < lb:
#        return 1
    else:
        return 0


def createPriceDecisionVarByClientBreakoutMeasureRegionChain(df):
    '''
    Creates a pulp decision varialble for use in the LP base on a row of a dataframe based on the GPI_NDC,
    Client, Breakout, Measurement, Region, & Pharmacy
    Input:
        a dataframe row from lp_vol_mv_df
    Output:
        price_decision_variable (pulp)
    '''
    price_var = str('P_'+str(df.GPI_NDC) + '_' + str(df.CLIENT) + '_' + str(df.BREAKOUT) +
                    '_' + str(df.MEASUREMENT) + '_' + str(df.REGION) +'_' + str(df.CHAIN_GROUP))

    low_b, up_b = df.Price_Bounds

    return pulp.LpVariable(price_var, lowBound=low_b, upBound=up_b)


def generatePricingDecisionVariables(data_df):
    '''
    Generates the pricing decision variable name as well as calls the function to create the price decision PuLP variable
    Inputs:
        DataFrame - Client, Breakout, Region, Pharmacy, GPI-NDC, MAC Price, Pricing Bounds
    Output:
        Add a Pandas series of decision variables to the input dataframe
    '''
    data_df['Price_Decision_Var'] = data_df.apply(createPriceDecisionVarByClientBreakoutMeasureRegionChain, axis=1)
    data_df['Dec_Var_Name'] = data_df.apply(lambda df: 'P_'+str(df.GPI_NDC) + '_' + str(df.CLIENT) + '_' + str(df.BREAKOUT) +
                                            '_' + str(df.MEASUREMENT) + '_' + str(df.REGION) +'_' + str(df.CHAIN_GROUP), axis=1)
    data_df.Dec_Var_Name = data_df.Dec_Var_Name.str.replace(' ', '_')

    return data_df


def generateLambdaDecisionVariables(Ph_list):
    '''
    Input: List of Pharmacy types
    Output: Dataframe of pharmacy type and lambda decision variable with over and under performance

    '''

    pharm_type = ['Preferred', 'Non_Preferred']

    Lambda_df = pd.DataFrame(columns = ['PHARMACY_TYPE', 'Lambda_Over', 'Lambda_Under', 'Lambda_Level', 'PHARMACY'])

    row_count = 0
    for ph in pharm_type:
            Lambda_df.loc[row_count,'PHARMACY_TYPE'] = ph
            lambda_var_over, lambda_var_under  = str(ph  + '_lambda_over'), str(ph  + '_lambda_under')
            Lambda_df.loc[row_count,'Lambda_Over'] = pulp.LpVariable(lambda_var_over, lowBound=0)
            Lambda_df.loc[row_count,'Lambda_Under'] = pulp.LpVariable(lambda_var_under, lowBound=0)
            Lambda_df.loc[row_count,'Lambda_Level'] = 'CLIENT'
            Lambda_df.loc[row_count,'PHARMACY'] = 'CLIENT_' + ph
            row_count = row_count + 1


    for ph in Ph_list:
            if ph in ['CVS', 'PREF_OTH']:
                Lambda_df.loc[row_count,'PHARMACY_TYPE'] = 'Preferred'
            else:
                Lambda_df.loc[row_count,'PHARMACY_TYPE'] = 'Non_Preferred'
            lambda_var_over, lambda_var_under  = str(ph  + '_lambda_over'), str(ph  + '_lambda_under')
            Lambda_df.loc[row_count,'Lambda_Over'] = pulp.LpVariable(lambda_var_over, lowBound=0)
            Lambda_df.loc[row_count,'Lambda_Under'] = pulp.LpVariable(lambda_var_under, lowBound=0)
            Lambda_df.loc[row_count,'Lambda_Level'] = 'PHARMACY'
            Lambda_df.loc[row_count,'PHARMACY'] = ph
            row_count = row_count + 1

    return Lambda_df


def generateLambdaDecisionVariables_ebit(Cl_df, Ph_list):
    '''
    Input: List of PHARMACY types
    Output: Dataframe of PHARMACY type and lambda decision variable with over and under performance

    '''

#    pharm_type = ['Preferred', 'Non_Preferred']

    Lambda_df = pd.DataFrame(columns = ['PHARMACY_TYPE', 'Lambda_Over', 'Lambda_Under', 'Lambda_Level', 'PHARMACY'])
    row_count = 0
    ## Client Non Preferred CAPPED #####
##
#    Lambda_df.loc[row_count,'PHARMACY_TYPE'] = 'Non_Preferred'
#    lambda_var_over, lambda_var_under  = str('lambda_over_capped'), str('lambda_under_capped')
#
#    if lambda_var_over != 'NA':
#        Lambda_df.loc[row_count,'Lambda_Over'] = pulp.LpVariable(lambda_var_over, lowBound=0)
#    if lambda_var_under != 'NA':
#        Lambda_df.loc[row_count,'Lambda_Under'] = pulp.LpVariable(lambda_var_under, lowBound=0)
#
#    Lambda_df.loc[row_count,'Lambda_Level'] = 'CLIENT'
#    Lambda_df.loc[row_count,'PHARMACY'] = 'Non_Preferred_Capped'
#    row_count +=1
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
        ### GIT: updated to NON_CAPPED_PHARMACY_LIST to incorporate more clients
        if ph not in p.NON_CAPPED_PHARMACY_LIST:
            if ph in ['CVS', 'PREF_OTH']:
                Lambda_df.loc[row_count,'PHARMACY_TYPE'] = 'Preferred'
            else:
                Lambda_df.loc[row_count,'PHARMACY_TYPE'] = 'Non_Preferred'
            lambda_var_over, lambda_var_under  = str(ph  + '_lambda_over'), str(ph  + '_lambda_under')
#            if ph != 'CVS':
            Lambda_df.loc[row_count,'Lambda_Over'] = pulp.LpVariable(lambda_var_over, lowBound=0)
            Lambda_df.loc[row_count,'Lambda_Under'] = pulp.LpVariable(lambda_var_under, lowBound=0)
            if ph == 'CVS':
                Lambda_df.loc[row_count,'Lambda_Level'] = 'CLIENT'
            else:
                Lambda_df.loc[row_count,'Lambda_Level'] = 'CLIENT'
            Lambda_df.loc[row_count,'PHARMACY'] = ph
            row_count +=1

    ##0 usage drugs
#    Lambda_df.loc[row_count,'PHARMACY_TYPE'] = 'Prices'
#    lambda_var_over, lambda_var_under  = str('PRICES' + '_lambda_over'), str('PRICES'  + '_lambda_under')
#    Lambda_df.loc[row_count,'Lambda_Over'] = pulp.LpVariable(lambda_var_over, lowBound=0)
#    Lambda_df.loc[row_count,'Lambda_Under'] = pulp.LpVariable(lambda_var_under, lowBound=0)
#    Lambda_df.loc[row_count,'Lambda_Level'] = 'CLIENT'
#    Lambda_df.loc[row_count,'PHARMACY'] = 'PRICES'

    return Lambda_df

def generateCost_new(lambda_df, gamma, over_reimb_gamma, pen):
    '''
    INPUT:
        lamdba_df is a DataFrame with Pulp Variable lambda (over and under)
        gamma: scalar
    Output:
        LPAffineExpression:
    '''

    ### GIT: Check whether MCHOICE is present in the data to decide whether minimize underperformance for Mail
    if lambda_df.PHARMACY.str.contains('MCHOICE').any():
        has_mchoice = True
    else:
        has_mchoice = False

    cost_over = lambda_df.Lambda_Over.values*gamma
    cost_under = lambda_df.Lambda_Under.values*gamma


    cost_over = lambda_df.Lambda_Over.values*gamma
    cost_under = lambda_df.Lambda_Under.values*gamma


    cost = ""
    for i in range(lambda_df.shape[0]):
        if(lambda_df.Lambda_Level.values[i] == 'CLIENT'):
            if lambda_df.PHARMACY_TYPE.values[i] != 'CLIENT':
                cost += cost_over[i]*(1+pen)
                if p.CAPPED_OPT and (lambda_df.PHARMACY.values[i] in p.OVER_REIMB_CHAINS):
                    cost+= (-1 * over_reimb_gamma)*cost_under[i]
                # else:
                    # HACK: over-reimburse
                    # if (lambda_df.PHARMACY.values[i] == 'CVS'):  # (lambda_df.PHARMACY.values[i] == 'WAG') or (lambda_df.PHARMACY.values[i] == 'RAD') or ( lambda_df.PHARMACY.values[i] == 'WMT'):
                        # cost += (-1 * over_reimb_gamma) * cost_under[i]
            else:
                if lambda_df.PHARMACY.values[i].split('_')[1] == 'MAIL':
                    cost += cost_over[i]
                    ### GIT - Minimizing underperformance for Mail if MCHOICE is present in the data
                    if has_mchoice:
                        cost += cost_under[i]
                elif lambda_df.PHARMACY.values[i] == 'PRICES':
                    cost += cost_over[i]
                else:
                    cost += cost_under[i]
                    cost += cost_over[i] * p.CLIENT_OVRPERF_PEN
        else:
            cost += cost_over[i] - cost_under[i]
    return cost


def generate_constraint_prices(price_row, lambda_pair, cons_type):
    '''
    INPUT:
        price_vol_df: DataFrame with Pricing Decision Var, Qty
        lambda_df : DataFrame of lambda_decision variables filtered for either Client or Pharmacy level
        guarantee_dict: Scalar with either Client level Guarantee or Pharmacy level Guarantee
        cons_type: Client or Not_Client_Level
    OUTPUT:
        cons_: pulp.LPAffineExpression
        target_: corresponding AWP (scalar)
    '''

    target_ = price_row.MAC_PRICE_UNIT_ADJ
    cons_ = ""
    cons_ += price_row.Price_Decision_Var*1 + lambda_pair[0] - lambda_pair[1]

    return cons_, target_


def generate_constraint_pharm(price_vol_df, lambda_df, guarantee_df, client_list, pharm_approx, cons_type):
    '''
    INPUT:
        price_vol_df: DataFrame with Pricing Decision Var, Qty
        lambda_df : DataFrame of lambda_decision variables filtered for either Client or Pharmacy level
        guarantee_dict: Scalar with either Client level Guarantee or Pharmacy level Guarantee
        cons_type: Client or Not_Client_Level
    OUTPUT:
        cons_: pulp.LPAffineExpression
        target_: corresponding AWP (scalar)
    '''
    p_v_df_mut = price_vol_df[price_vol_df.PRICE_MUTABLE == 1]
    p_v_df_immut= price_vol_df[price_vol_df.PRICE_MUTABLE == 0]

    target_ = 0
    cons_ = LpAffineExpression()
    pharmacy = lambda_df.iloc[0]['PHARMACY']
    for client in client_list:
        logging.debug(pharmacy)
        logging.debug(client)
        scale_factor = pharm_approx.loc[(pharm_approx.CLIENT==client) & (pharm_approx.CHAIN_GROUP==pharmacy), 'SLOPE'].values[0]
        intercept = pharm_approx.loc[(pharm_approx.CLIENT==client) & (pharm_approx.CHAIN_GROUP==pharmacy), 'EOY_INTERCEPT'].values[0]
        guarantee = guarantee_df.loc[(guarantee_df.PHARMACY == pharmacy) & (guarantee_df.CLIENT == client)]['RATE'].values[0]
        unscaled_target = (1-guarantee)*(p_v_df_mut.loc[p_v_df_mut['CLIENT']==client].FULLAWP_ADJ_PROJ_EOY.sum() + p_v_df_immut.loc[p_v_df_immut['CLIENT']==client].FULLAWP_ADJ_PROJ_EOY.sum())\
                    - (np.dot(p_v_df_immut.loc[p_v_df_immut['CLIENT']==client].EFF_CAPPED_PRICE.values, p_v_df_immut.loc[p_v_df_immut['CLIENT']==client].QTY_PROJ_EOY.values))
        target_ += scale_factor * unscaled_target + intercept
        logging.debug('Unscaled target: %f', unscaled_target)
        logging.debug('Scale factor: %f', scale_factor)
        logging.debug('Intercept: %f', intercept)
        logging.debug('Final target: %f', scale_factor * unscaled_target + intercept)

        cons_ += LpAffineExpression([(p_v_df_mut.loc[p_v_df_mut['CLIENT']==client].Price_Decision_Var.values[i],
                                      p_v_df_mut.loc[p_v_df_mut['CLIENT']==client].QTY_PROJ_EOY.values[i]) for i in range(p_v_df_mut.loc[p_v_df_mut['CLIENT']==client].shape[0])]) * scale_factor
    cons_ += lambda_df.Lambda_Over.values[0] - lambda_df.Lambda_Under.values[0]

    return cons_, target_

def generate_constraint_pharm_new(price_vol_df, lambda_df, guarantee_df, client_list, pharm_approx, cons_type):
    '''
    INPUT:
        price_vol_df: DataFrame with Pricing Decision Var, Qty
        lambda_df : DataFrame of lambda_decision variables filtered for either Client or Pharmacy level
        guarantee_dict: Scalar with either Client level Guarantee or Pharmacy level Guarantee
        cons_type: Client or Not_Client_Level
    OUTPUT:
        cons_: pulp.LPAffineExpression
        target_: corresponding AWP (scalar)
    '''
    p_v_df_mut = price_vol_df[price_vol_df.PRICE_MUTABLE == 1]
    p_v_df_immut= price_vol_df[price_vol_df.PRICE_MUTABLE == 0]

    target_ = 0
    cons_ = LpAffineExpression()
    pharmacy = lambda_df.iloc[0]['PHARMACY']
    for client in client_list:
        logging.debug(pharmacy)
        logging.debug(client)
        scale_factor = pharm_approx.loc[(pharm_approx.CLIENT==client) & (pharm_approx.CHAIN_GROUP==pharmacy), 'SLOPE'].values[0]
        intercept = pharm_approx.loc[(pharm_approx.CLIENT==client) & (pharm_approx.CHAIN_GROUP==pharmacy), 'EOY_INTERCEPT'].values[0]
        unscaled_target = 0

        breakout_list = price_vol_df.loc[(price_vol_df.CLIENT==client), 'BREAKOUT'].unique()

        for breakout in breakout_list:
            reg_list = price_vol_df.loc[(price_vol_df.CLIENT==client) &
                                        (price_vol_df.BREAKOUT==breakout), 'REGION'].unique()
            for reg in reg_list:

                guarantee = guarantee_df.loc[(guarantee_df.PHARMACY == pharmacy) &
                                             (guarantee_df.CLIENT == client) &
                                             (guarantee_df.BREAKOUT == breakout) &
                                             (guarantee_df.REGION == reg)]['RATE'].values[0]
                mutable_data = p_v_df_mut.loc[(p_v_df_mut['CLIENT']==client) &
                                              (p_v_df_mut.BREAKOUT==breakout) &
                                              (p_v_df_mut.REGION==reg)]
                immutable_data = p_v_df_immut.loc[(p_v_df_immut['CLIENT']==client) &
                                                  (p_v_df_immut.BREAKOUT==breakout) &
                                                  (p_v_df_immut.REGION==reg)]
                unscaled_target += (1-guarantee)*(mutable_data.FULLAWP_ADJ_PROJ_EOY.sum() + immutable_data.FULLAWP_ADJ_PROJ_EOY.sum())\
                            - (np.dot(immutable_data.EFF_CAPPED_PRICE.values, immutable_data.QTY_PROJ_EOY.values))
        target_ += scale_factor * unscaled_target + intercept
        logging.debug('Unscaled target: %f', unscaled_target)
        logging.debug('Scale factor: %f', scale_factor)
        logging.debug('Intercept: %f', intercept)
        logging.debug('Final target: %f', scale_factor * unscaled_target + intercept)

        cons_ += LpAffineExpression([(p_v_df_mut.loc[p_v_df_mut['CLIENT']==client].Price_Decision_Var.values[i],
                                      p_v_df_mut.loc[p_v_df_mut['CLIENT']==client].QTY_PROJ_EOY.values[i]) for i in range(p_v_df_mut.loc[p_v_df_mut['CLIENT']==client].shape[0])]) * scale_factor
    cons_ += lambda_df.Lambda_Over.values[0] - lambda_df.Lambda_Under.values[0]

    return cons_, target_


def generate_constraint_client(price_vol_df, lambda_df, guarantee_df, cons_type):
    '''
    INPUT:
        price_vol_df: DataFrame with Pricing Decision Var, Qty
        lambda_df : DataFrame of lambda_decision variables filtered for either Client or Pharmacy level
        guarantee_dict: Scalar with either Client level Guarantee or Pharmacy level Guarantee
        cons_type: Client or Not_Client_Level
    OUTPUT:
        cons_: pulp.LPAffineExpression
        target_: corresponding AWP (scalar)
    '''
    p_v_df_mut = price_vol_df[price_vol_df.PRICE_MUTABLE == 1]
    p_v_df_immut= price_vol_df[price_vol_df.PRICE_MUTABLE == 0]

    breakout = price_vol_df.iloc[0]['BREAKOUT']
    client = price_vol_df.iloc[0]['CLIENT']

#    if client == 'WELLCARE':
#        price_vol_df.to_csv(client + breakout + '_objtest.csv', index = False)
    target_ = 0
    logging.debug(client + '_' + breakout)
    for region in price_vol_df.REGION.unique():
        logging.debug(region)
        # HACK: change to accommodate different measurements for different regions
        for measure in price_vol_df.loc[price_vol_df.REGION == region, 'MEASUREMENT'].unique():
            logging.debug(measure)
            guarantee_pref = guarantee_df.loc[(guarantee_df.CLIENT == client) & (guarantee_df.BREAKOUT == breakout) & (
                        guarantee_df.REGION == region) &
                                              (guarantee_df.MEASUREMENT == measure) & (
                                                          guarantee_df.PHARMACY_TYPE == 'Preferred')].RATE.values[0]
            #            logging.debug('Pref guarantee rate: ', guarantee_pref)
            guarantee_npref = guarantee_df.loc[(guarantee_df.CLIENT == client) & (guarantee_df.BREAKOUT == breakout) & (
                        guarantee_df.REGION == region) &
                                               (guarantee_df.MEASUREMENT == measure) & (
                                                           guarantee_df.PHARMACY_TYPE == 'Non_Preferred')].RATE.values[
                0]
            #            logging.debug('NPref guarantee rate: ', guarantee_npref)
            pref_p_v_mut = p_v_df_mut.loc[(p_v_df_mut['REGION'] == region) & (p_v_df_mut['MEASUREMENT'] == measure) & (
                        p_v_df_mut['PHARMACY_TYPE'] == 'Preferred')]
            pref_p_v_immut = p_v_df_immut.loc[
                (p_v_df_immut['REGION'] == region) & (p_v_df_immut['MEASUREMENT'] == measure) & (
                            p_v_df_immut['PHARMACY_TYPE'] == 'Preferred')]
            npref_p_v_mut = p_v_df_mut.loc[(p_v_df_mut['REGION'] == region) & (p_v_df_mut['MEASUREMENT'] == measure) & (
                        p_v_df_mut['PHARMACY_TYPE'] == 'Non_Preferred')]
            npref_p_v_immut = p_v_df_immut.loc[(p_v_df_immut['REGION']==region) & (p_v_df_immut['MEASUREMENT']==measure) & (p_v_df_immut['PHARMACY_TYPE']=='Non_Preferred')]

            target_pref = (1-guarantee_pref)*(pref_p_v_mut.FULLAWP_ADJ_PROJ_EOY.sum() + pref_p_v_immut.FULLAWP_ADJ_PROJ_EOY.sum())\
                        - np.dot(pref_p_v_immut.EFF_CAPPED_PRICE.values, pref_p_v_immut.QTY_PROJ_EOY.values)
            logging.debug('Pref AWP: %f', (pref_p_v_mut.FULLAWP_ADJ_PROJ_EOY.sum() + pref_p_v_immut.FULLAWP_ADJ_PROJ_EOY.sum()))
            logging.debug('Pref Uncontrolled Spend: %f', np.dot(pref_p_v_immut.EFF_CAPPED_PRICE.values, pref_p_v_immut.QTY_PROJ_EOY.values))
            target_nonpref = (1-guarantee_npref)*(npref_p_v_mut.FULLAWP_ADJ_PROJ_EOY.sum() + npref_p_v_immut.FULLAWP_ADJ_PROJ_EOY.sum())\
                        - np.dot(npref_p_v_immut.EFF_CAPPED_PRICE.values, npref_p_v_immut.QTY_PROJ_EOY.values)
            logging.debug('NPref AWP: %f', npref_p_v_mut.FULLAWP_ADJ_PROJ_EOY.sum() + npref_p_v_immut.FULLAWP_ADJ_PROJ_EOY.sum())
            logging.debug('NPref Uncontrolled Spend: %f', np.dot(npref_p_v_immut.EFF_CAPPED_PRICE.values, npref_p_v_immut.QTY_PROJ_EOY.values))
            target_ += target_pref + target_nonpref


    cons_ = LpAffineExpression([(p_v_df_mut.Price_Decision_Var.values[i], p_v_df_mut.QTY_PROJ_EOY.values[i]) for i in range(p_v_df_mut.shape[0])])  - lambda_df.Lambda_Under.values[0] + lambda_df.Lambda_Over.values[0]


    return cons_, target_


def generateGuaranteeConstraintEbit(price_vol_df, lambda_df, breakout_guarantees, pharmacy_guarantees, perf_dict, breakout_df, client_list, pharmacy_list, pharm_approx, eoy_pharm, eoy_gen_launch, price_lambdas):
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

    constraint = []
    awp_g = []

#        non_preferred_list = [ 'KRG', 'RAD', 'WAG', 'WMT']

    logging.debug("Building Pharmacy Constraints")

    start = time.time()
    for pharmacy in pharmacy_list:
        cons_, awp_ = generate_constraint_pharm_new(price_vol_df[price_vol_df.CHAIN_GROUP == pharmacy], lambda_df[lambda_df.PHARMACY == pharmacy],
                                          pharmacy_guarantees, client_list, pharm_approx, 'Pharm')
        constraint.append(cons_)
        awp_g.append(awp_ + perf_dict[pharmacy] + eoy_pharm[pharmacy] + eoy_gen_launch[pharmacy])

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
        awp_g.append(awp_ + perf_dict[breakout_row.Combined] + eoy_gen_launch[breakout_row.Combined])

    end = time.time()

    logging.debug("End Building Breakout Constraints")
    logging.debug("Run time: {} mins".format((end - start)/60.))
    no_util = price_vol_df.loc[(price_vol_df.PRICE_MUTABLE==1) & (price_vol_df.QTY_PROJ_EOY==0)]
    for i in range(len(no_util)):
        cons_, awp_ = generate_constraint_prices(no_util.iloc[i], price_lambdas[i], 'CLIENT')
        constraint.append(cons_)
        awp_g.append(awp_)



    return constraint, awp_g


def pharmacy_type(row):
    '''
    Takes dataframe row and determines if the chain group is preferrred or not
    NOTE: needs to be changed to utilize new preferred dictionary
    Inputs:
        Row of a dataframe that contains the CLIENT, BREAKOUT, and CHAIN_GROUP of that row
    Outputs:
        A single string of "Preferred" or "Non_Preferred"
    '''
    #pref_pharms = pref_pharm[row.CLIENT + '_' + row.BREAKOUT]

    #if row.CHAIN_GROUP in pref_pharms:
     #   return 'Preferred'
    #else:
     #   return 'Non_Preferred'

    return 'Non_Preferred'
# =============================================================================
#     if row['CLIENT'] == 'SSI':
#         if row['CHAIN_GROUP'] in ['CVS', 'PREF_OTH']:
#             return 'Preferred'
#         else:
#             if (row['BREAKOUT'] in ['PLUS', 'ALLURE']) & (row['CHAIN_GROUP'] in ['KRG','WAG']):
#                 return 'Preferred'
#             else:
#                 return 'Non_Preferred'
#     elif row['CLIENT'] == 'WELLCARE':
#         if row['CHAIN_GROUP'] in ['CVS', 'PREF_OTH']:
#             return 'Preferred'
#         else:
#             return 'Non_Preferred'
# =============================================================================

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


def determine_effective_price(row, old_price, uc_unit='UC_UNIT', capped_only=False):
    '''
    Takes a row of a dataframe and returns the effective price based on the price
    ceiling based on min of (U&C amount, awp discount, or old price) and the price
    floor based on max of (1026 floor and price)
    Inputs:
        -row: a row of a dataframe that includes the old price, pharmacy chain (CHAIN_GROUP),
                U&C for that drug (uc_unit), average AWP for that unit (AVG_AWP),
                and MAC 1026 price (MAC1026_UNIT_PRICE) for that drug
        -old_price: a string stating the column within the row that contains the old price
        -capped_only: an option to only impose caps and not floors onto the price
    Outputs:
        -eff_price: the effective price after capping and flooring the original price

    Note: the price will be floored after being capped so if the floor is above the price cap then
            the floor will be returned.
    '''
    if np.isfinite(row[old_price]) & (row[old_price] > 0):
        if (row[uc_unit] > 0) & (row['AVG_AWP'] > 0):
            eff_price =  min([row[uc_unit], row['AVG_AWP']*.75, row[old_price]])
        elif (row[uc_unit] > 0):
            eff_price =  min([row[uc_unit], row[old_price]])
        elif (row['AVG_AWP'] > 0):
            eff_price =  min([row['AVG_AWP']*.75, row[old_price]])
        else:
            eff_price = row[old_price]

        if not capped_only:
            if (row['MAC1026_UNIT_PRICE'] > 0) & (row['CHAIN_GROUP'] in p.NON_CAPPED_PHARMACY_LIST):
                return max(row['MAC1026_UNIT_PRICE'], eff_price)
            else:
                return eff_price
        else:
            return eff_price

    else:
        return row[old_price]
