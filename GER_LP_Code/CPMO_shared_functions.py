# -*- coding: utf-8 -*-
"""
Shared functions (non-LP) file for CLIENT_PHARMACY_MAC_OPTIMIZATION

"""
import os
import CPMO_parameters as p
import pandas as pd
import numpy as np
import copy
import logging
import util_funcs as uf
import BQ


def standardize_df(df):
    '''
    This is a series of common steps to get all dataframes and data inputs into the same
    general format.
    Inputs:
        df - a dataframe that needs to be standardized
    Outputs:
        df - the standardized dataframe
    '''
    del_cols = []
    for col in df.columns:
        if 'Unnamed:' in col:
            del_cols.append(col)
    df = df.drop(columns=del_cols)

    # Add captialization to all names to standardize the input for all clients
    df.columns = map(str.upper, df.columns)
    
    #Captilaize everything in the following columns to make sure that we can use string matching
    # PIP change added CLIENT_NAME
    for column in ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'CLIENT_NAME', 'CUSTOMER_ID', 'MAC_LIST', 'MACLIST', 'VCML_ID']:
        if column in df.columns:
            df[column] = df[column].astype(str)
            df[column] = df[column].apply(lambda x: x.upper())
    
    # Different data sources use different names for Wellcare so this standarized them to "WELLCARE"
    if 'CLIENT' in df.columns:
        df.loc[df.CLIENT == 'WC', 'CLIENT'] = 'WELLCARE'
        # TODO: Is this what we want
        df.CLIENT = df.CLIENT.astype(str)
        
    
    # Ensures that no GPI loses its leading 0
    if 'GPI' in df.columns:
        if df.GPI.dtype == float:
            df.GPI = df.GPI.astype('int64')
            
        df.GPI = df.GPI.astype(str).apply(lambda x: x.split('.')[0])
        df.GPI = ('00' + df.GPI).str[-14:]
    
    # Resolves naming differences of NDC columns between different data sources
    if 'NDC11' in df.columns:
        df.rename(columns={'NDC11': 'NDC'}, inplace=True)
    
    # Ensures that NDCs do not lose leading 0
    if 'NDC' in df.columns:
        if df.NDC.dtype == float:
            df.NDC = df.NDC.astype('int64')
            
        df.NDC = df.NDC.astype(str).apply(lambda x: x.split('.')[0])
        df.NDC = ('0000' + df.NDC).str[-11:]
    
    # Creates a common column of GPI and NDC for string matching
    if ('GPI' in df.columns) & ('NDC' in df.columns):
        df['GPI_NDC'] = df.GPI + '_' + df.NDC
    
    #MARCEL Added. reading client_claim (see Pre_processing.py) from BQ somehow results in object type
    if 'SPEND' in df.columns:
        df.SPEND = df.SPEND.astype('float64')
    if 'AWP' in df.columns:
        df.AWP = df.AWP.astype('float64')
    if 'DRUG_PRICE_AT' in df.columns:
        df.DRUG_PRICE_AT = df.DRUG_PRICE_AT.astype('float64')
    if 'QTY' in df.columns:
        df.QTY = df.QTY.astype('float64')
    # for col in df.columns:
    #     if df[col].dtype == 'object':
    #         assert (df[col].apply(type).value_counts().shape[0] == 1), f'Column "{col}" has mix data types'

    return df


def dict_to_df(dictionary, column_names):
    '''
    Simple helper function that takes a dictionary and turns it into a dataframe with two columns:
    one for keys and one for values.
    Inputs:
        dictionary: the dictionary that you want to change to a dataframe
        column_names: the column names that will be used in the dataframe for the key and values, repectively
    Outputs:
        df: the dataframe of the dictionary keys and values
    '''
    df = pd.DataFrame(columns=column_names)
    for key in dictionary:
        df = df.append({column_names[0]: key,
                        column_names[1]: dictionary[key]}, ignore_index = True)
    return df


def df_to_dict(df, column_names):
    ''' 
    Simple helper function that takes a pandas dataframe and turns it into a dictionary with key values pairings for
    every row in the dataframe.  This only works if there are unique keys in the dataframe
    Inputs:
        df: the dataframe that you want to change to a dictionary
        column_names: the column names that are used in the dataframe for the key and values, repectively
    Outputs:
        temp_dict: the dictionary keys and values from each row of the dataframe
    '''
    df = df.reset_index(drop=True)
    temp_dict = dict()
    for i in range(len(df)):
        temp_dict[df[column_names[0]][i]] = df[column_names[1]][i]
    
    return temp_dict

def round_to(number, decimals=4):
    """Rounds a number down to the specified number of decimal places using ROUND_DOWN as default or ROUND_HALF_UP when specified
       Additionally, uses ROUND_HALF_UP to round up if the digit after the rounding digit is 9 to avoid differences in prices
       This function is used in ClientPharmacyMACOptimization.py to create 'Rounded_Price' 
    """ 
    
    import pandas as pd
    from decimal import Decimal, ROUND_DOWN as DECIMAL_ROUND_DOWN, ROUND_HALF_UP
    import CPMO_parameters as p

    if pd.isna(number):
        return number

    # Build quantize string based on decimal places
    quantize_str = '1.'+ '0' * decimals
    
    if p.ROUND_DOWN:
        # Check if the digit after rounding may affect result (E.g. 0.271949 might go to 0.27195)
        # Have to catch this case and round up, otherwise you will get rounding discrepancies
        if '9' in f"{number:.20f}"[decimals+2:]:
            return float(
                Decimal(str(number)).quantize(Decimal(quantize_str),rounding=ROUND_HALF_UP)
            )
        else:
            # Flooring value using ROUND_DOWN
            return float(
                Decimal(str(number)).quantize(Decimal(quantize_str),rounding=DECIMAL_ROUND_DOWN)
            )
    else:
        if '49' in f"{number:.20f}"[decimals + 2:]:
        # Add two extra decimal places for intermediate rounding
            decimals += 2
            quantize_str = '1.'+ '0' * decimals
            temp_return = Decimal(str(number)).quantize(Decimal(quantize_str),rounding=ROUND_HALF_UP)

            # Final rounding after intermediate value, using quantize_str[:-2] to undo the earlier +=2
            return float(
                Decimal(str(temp_return)).quantize(Decimal(quantize_str[:-2]),rounding=ROUND_HALF_UP)
            )   
        else:
            # Regular rounding
            return float(
                Decimal(str(number)).quantize(Decimal(quantize_str),rounding=ROUND_HALF_UP)
            )


def calculatePerformance(data_df, client_guarantees, pharmacy_guarantees, client_list, pharmacy_list, oc_pharm_perf,
                             gen_launch_perf, brand_offset_perf, specialty_offset_perf, disp_fee_offset_perf, client_reimb_column='PRICE_REIMB',
                             pharm_reimb_column='PHARM_PRICE_REIMB', client_TARG_column='FULLAWP_ADJ',
                             pharm_TARG_column='PHARM_FULLAWP_ADJ', AWP_column='', restriction='none', other=False,
                             qty_column=False, full_disp=False, subchain_df=False):
    '''
    Calculates the performance of the clients and pharmacies as a dollar surplus given all of the inputs and then outputs the client and pharmacy performances
    Input:
        data_df: DataFrame which has the data for months that we have data for all the Pharmacies and regions
    Output:
        Dictionary of performances for CVS, Non Preferred Capped Chains, SSI, & Walgreens
    '''
    if isinstance(brand_offset_perf,pd.DataFrame):
        if len(brand_offset_perf) <= 1:
            brand_offset_perf = brand_offset_perf.set_index('ENTITY').squeeze(axis=1).to_dict()
        else:
            brand_offset_perf = brand_offset_perf.set_index('ENTITY').squeeze().to_dict()
    if isinstance(specialty_offset_perf,pd.DataFrame):
        specialty_offset_perf = specialty_offset_perf.set_index('ENTITY').squeeze().to_dict()
    if isinstance(disp_fee_offset_perf,pd.DataFrame):
        disp_fee_offset_perf = disp_fee_offset_perf.set_index('ENTITY').squeeze().to_dict()
        
    if qty_column:
        disp_qty = True
    else:
        disp_qty = False

    if len(AWP_column) > 0:
        client_TARG_column = AWP_column
        pharm_TARG_column = AWP_column

    ###############
    def client_breakout_dict(input_df,
                             values):  ##LP Mod: to get the client performance values on the client+breakout level
        d = {}
        for i in range(len(input_df)):
            d[input_df['CLIENT'][i] + '_' + input_df['BREAKOUT'][i]] = input_df[values][i]
        return d

    ###############
    ##LP Mod change: made this a function to reduce the number of lines and modularize the code
    def disp_qty_calc(input_df):
        group_by_list = ['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT','BG_FLAG']
        if disp_qty:
            print('Pharm MAC Qty: ',
                  input_df[input_df['CURRENT_MAC_PRICE'] > 0].groupby(group_by_list).agg({qty_column: 'sum'}))
            print('Pharm NonMAC Qty: ',
                  input_df[input_df['CURRENT_MAC_PRICE'] <= 0].groupby(group_by_list).agg({qty_column: 'sum'}))

        print('Preferred MAC AWP and Spend: ', input_df[input_df['CURRENT_MAC_PRICE'] > 0].groupby(group_by_list).agg(
            {client_TARG_column: 'sum', client_reimb_column: 'sum'}))
        print('Preferred NonMAC AWP and Spend: ',
              input_df[input_df['CURRENT_MAC_PRICE'] <= 0].groupby(group_by_list).agg(
                  {client_TARG_column: 'sum', client_reimb_column: 'sum'}))

    ##############
    spend = 0  ## LP Mod Used for Client perf only (Change- out of the for loop now)
    target = 0  ## LP Mod Used for Client perf only (Change- out of the for loop now)

    npref_flag = 0  ## LP Mod: Addition of this flag to identify if the Pharmacy_type for a given input row is Non-preferred if this flag is 1.
    pref_flag = 0  ## LP Mod: Addition of this flag to identify if the Pharmacy_type for a given input row is preferred if this flag is 1.

    if ['Non_Preferred'] in data_df['PHARMACY_TYPE'].values:  ##LP mod addition: as explained above
        npref_flag = 1

    if ['Preferred'] in data_df['PHARMACY_TYPE'].values:  ##LP mod addition: as explained above
        pref_flag = 1

    perf_pharm_final = {}
    perf_client_final = {}

    ####### Main Client Guarantee Calcs####################
    if restriction != 'pharm' and (npref_flag + pref_flag) >= 1:
        ##LP Mod: main dataframe for client guarantee calcs at C, R, B, M, P Type grain!
        master_client_perf_df = data_df.merge(client_guarantees, how='inner',
                                              on=['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'BG_FLAG', 'PHARMACY_TYPE']) \
            .groupby(['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'BG_FLAG', 'PHARMACY_TYPE']) \
            .agg({'RATE': 'first', client_TARG_column: 'sum', client_reimb_column: 'sum'}) \
            .reset_index()

        ##LP Mod: client guarantee calcs at Client, Region, Breakout,Measurement , Pharmacy Type grain!
        master_client_perf_df['npref_pref_guarantee'] = master_client_perf_df[client_TARG_column] - master_client_perf_df[client_reimb_column]

        ##LP Mod: Aggregation of client guarantee calcs at C, B (Client and Breakout) level
        client_breakout_perf_df = master_client_perf_df.groupby(['CLIENT', 'BREAKOUT']).agg(
            {'npref_pref_guarantee': 'sum'}).reset_index()

        ##LP Mod: Client guarantee calcs at C, B (Client and Breakout) grain in a dictionary
        client_breakout_perf_dict = client_breakout_dict(client_breakout_perf_df, 'npref_pref_guarantee')

        ##LP Mod: Client guarantee calcs at C,B grain with mathematical addition of brand_offset, gen_launch performance, specialty_offset, and disp_fee dicts- dict merge below.
        ##This is the final value for client performance.
        perf_client_final_dict = {
            k: client_breakout_perf_dict.get(k, 0) + gen_launch_perf.get(k, 0) + brand_offset_perf.get(k, 0) + specialty_offset_perf.get(k, 0) + disp_fee_offset_perf.get(k, 0) for k in
            set(client_breakout_perf_dict) & set(brand_offset_perf) & set(gen_launch_perf) & set(specialty_offset_perf) & set(disp_fee_offset_perf)}

        ############Logging############################
        ##LP Mod: Logging some client perf aggregations (at Client, Region, Breakout,Measurement level)
        target += master_client_perf_df[client_TARG_column]
        spend += master_client_perf_df[client_reimb_column]
        logging.debug('Target: %f', target)
        logging.debug('Spend: %f', spend)
        logging.debug('Client Perf: %f', perf_client_final)

    ##########Value Display########################
    ## LP Mod: Display of values for full_disp and disp_quantity input flags (not much change from the original code- just flatting and reducing the if else statemnts).This part of the code has no bearing on the overall return value/output- just print statements!

    if full_disp and npref_flag == 1:
        nperf_df = data_df[data_df['PHARMACY_TYPE'] == 'Non_Preferred']
        disp_qty_calc(nperf_df)

    if full_disp and pref_flag == 1:
        perf_df = data_df[data_df['PHARMACY_TYPE'] == 'Preferred']
        disp_qty_calc(perf_df)
    ###############################################Part 2###################################################################################
    ######################Pharmacy Performance Calculations#################################################################################
    # Design change (Melanie and Diego approved- May 19th, 2023): No need of subgroup df anymore- it will always be false
    ## Also, do all calculations for pharm performance at the subchain level first and then roll them upto Chain Group Level

    if len(AWP_column) > 0:
        client_TARG_column = AWP_column
        pharm_TARG_column = AWP_column

    perf_pharm_final = {}
    gnrc_pharmacy_list = pharmacy_list['GNRC']
    brnd_pharmacy_list = pharmacy_list['BRND']
    if restriction != 'client':
        pharm_clients = copy.deepcopy(client_list)
        if other:
            pharm_clients = np.append(pharm_clients, 'OTHER')
            breakout_list = ['OTHER']
            region_list = ['OTHER']

        if 'CHAIN_SUBGROUP' in data_df.columns:
            ##no multiplier can be applied as subchain_df is not provided as the input
            ##LP Change no merge with subchain_df as subgroup column is present in data_df

            ## LP Change: Created dataframes for gen_launch and oc_pharm for easier joins and mathemcatical additions
            # change 1: june 5th- replaced pharmact_perf_name with chain_subgroup#
            gen_launch_df = pd.DataFrame(gen_launch_perf.items(), columns=['CHAIN_SUBGROUP', 'GEN_LAUNCH_VALUE'])
            oc_pharm_df = pd.DataFrame(oc_pharm_perf.items(), columns=['CHAIN_GROUP', 'OC_VALUE'])

            ##LP Mod: main dataframe for client guarantee calcs at Client, Region, Breakout, Chain Group and chain subgroup grain!
            master_pharm_perf_df = data_df.loc[((data_df['CHAIN_GROUP'].isin(gnrc_pharmacy_list)) & (data_df['BG_FLAG']=='G')) | ((data_df['CHAIN_GROUP'].isin(brnd_pharmacy_list)) & (data_df['BG_FLAG']=='B'))] \
                .merge(pharmacy_guarantees, how='inner', left_on=['CLIENT', 'REGION', 'BREAKOUT', 'CHAIN_GROUP', 'MEASUREMENT','BG_FLAG'],
                       right_on=['CLIENT', 'REGION', 'BREAKOUT', 'PHARMACY', 'MEASUREMENT','BG_FLAG'])

                ##below calcs are at Client, Region, Breakout, Chain Group and Chain Subgroupgrain
            ##LP Mod: IMportant Note: Oc_pharm_df addition to the data_df below happens at the Chain_Group Level, NOT the chain subgroup level
            ##LP mod: IMportant Note: gen_launch_df addition to the data_df below happens at the Chain Subgroup level, NOT the Chain_group level
            # new target ingredient cost will be used here
            master_pharm_perf_df['pharm_perf'] = (
                            master_pharm_perf_df[pharm_TARG_column] - master_pharm_perf_df[pharm_reimb_column])

            # Final Aggregation on CHAIN_SUBGROUP level:
            # change 2: june 5th, replaced replaced pharmact_perf_name with chain_subgroup#
            perf_pharm_final_df = master_pharm_perf_df.groupby(['CHAIN_SUBGROUP', 'CHAIN_GROUP']) \
                .agg({'pharm_perf': 'sum'}) \
                .reset_index() \
                .merge(gen_launch_df, how='inner', on='CHAIN_SUBGROUP') \
                .merge(oc_pharm_df, how='inner', on='CHAIN_GROUP')
            # Simple method: divides OC_VALUE for one CHAIN_GROUP equally between all its CHAIN_SUBGROUPs. 
            # This will return the appropriate total OC_VALUE per pharmacy, and will also cancel out of 
            # any pre-and-post pharmacy difference. We sacrifice some accuracy in per-CHAIN_SUBGROUP 
            # performance in the name of code simplicity, but the per-client per-CHAIN_SUBGROUP performance 
            # is rarely needed.
            perf_pharm_final_df['OC_VALUE'] /= perf_pharm_final_df.groupby('CHAIN_GROUP')['CHAIN_SUBGROUP'].transform('count')

            perf_pharm_final_df['pharm_perf_final'] = perf_pharm_final_df['pharm_perf'] + perf_pharm_final_df[
                'GEN_LAUNCH_VALUE'] + perf_pharm_final_df['OC_VALUE']

            ##LP Mod: Pharm performance calcs at chain subgroup grain in a dictionary. This is the final grain for Pharm Performance
            perf_pharm_final_dict = dict(zip(perf_pharm_final_df.CHAIN_SUBGROUP, perf_pharm_final_df.pharm_perf_final))
            
            assert set(perf_pharm_final_dict.keys()).issubset(disp_fee_offset_perf.keys()), "Missing pharm disp fee offset"
            for k in perf_pharm_final_dict.keys():
                    perf_pharm_final_dict[k] = perf_pharm_final_dict[k] + disp_fee_offset_perf[k]

        else:
            raise RuntimeError('Must pass a subchain_df dict to calculatePerformance if CHAIN_SUBGROUP not in data')

        ##need to find the list of pharmacies that are in the pharmacy_list but NOT in the chain_group of the data_df
        # missed_pharmacies=pharmacy_list not in data_df['CHAIN_GROUP']
        missed_pharmacies = np.setdiff1d(list(set(gnrc_pharmacy_list+brnd_pharmacy_list)), data_df.CHAIN_GROUP.values).tolist()
        missed_awp = data_df[data_df['CHAIN_GROUP'].isin(missed_pharmacies)]

        if missed_awp.empty:
            logging.warn(f"Pharmacy :{missed_pharmacies} not found in claims data to map subgroup split...ignoring $0")
        # final step- union of the pharm and client performances

    return ({**perf_client_final_dict, **perf_pharm_final_dict})


def clean_mac_1026_NDC(mac1026):
    '''
    Simple helper function that renames columns and can provide gpi level 1026 floors if first several lines are uncommented
    Inputs:
        df: the mac1026 dataframe containing the floor prices 
    Outputs:
        df: the mac1026 dataframe with updated column names for mac_cost_amt, gpi, and ndc

    '''
    #    mac1026all = mac1026.copy(deep=True)
    #    mac1026all = mac1026all.groupby(['gpi'],as_index=False)['mac_cost_amt'].max()
    #    mac1026all['ndc'] = '***********'
    #    mac1026_gpi = mac1026.loc[mac1026['ndc'] == "***********",:]
    #    mac1026all = mac1026all.loc[~mac1026all['gpi'].isin(mac1026_gpi['gpi']),:]
    #    mac1026.drop(["mac_list_id"],axis=1,inplace=True)
    #    mac1026 = pd.concat([mac1026,mac1026all]).reset_index(drop=True)
    mac1026.rename(index=str,columns = {"mac_cost_amt": "MAC1026_unit_price",
                                        "gpi":"GPI",
                                        "ndc":"NDC"}, inplace = True)
    
    return mac1026
    

def check_price_increase_decrease_initial(df, month):
    '''
    Determines if the price changes (increase or decrease) violate the price constraints/rules 
    Inputs:
        df: the final output dataframe containing price mutable drugs
        month: month that you are running the lp for generating updated prices 
    Outputs:
        rules violated: binary interger 0 or 1 that determines if the price increase or decrease violates the price contraints/rules  
    '''
    rules_violated = 0
    
    up_fac = df.New_Price / df.MAC_PRICE_UNIT_ADJ
    
    
    if p.TIERED_PRICE_LIM and (df.CLIENT in p.TIERED_PRICE_CLIENT):
        if (df.PRICE_REIMB_CLAIM <= 100) and (up_fac > 1.5):
            rules_violated = 1
    
    
        if (df.PRICE_REIMB_CLAIM <= 6) and (up_fac > 2):
            rules_violated = 1
    
    
        if (df.PRICE_REIMB_CLAIM <= 3) and (up_fac > 201):
            rules_violated = 1
            
    else:
        if up_fac > (1 + p.GPI_UP_FAC):
            rules_violated = 1
            
    #    if rules_violated == 1:
    #        logging.debug('GPI pricing error at: '+ df.Dec_Var_Name)
    #        logging.debug('Orig price: ' + str(df.MAC_PRICE_UNIT_ADJ))
    #        logging.debug('New price: ' + str(df.New_Price))

    return rules_violated

def check_agg_price_cons(df, month):
    '''
    Determines if the price changes (increase or decrease) violate the price constraints/rules
    Inputs:
        df: the final output dataframe containing price mutable drugs
        month: month that you are running the lp for generating updated prices
    Outputs:
        rules violated: binary interger 0 or 1 that determines if the price increase or decrease violates the price contraints/rules
    '''

    price_constraints_df = df.loc[(df.PRICE_MUTABLE == 1), :]
    price_constraints_df = standardize_df(price_constraints_df)

    ####variable declaration
    rules_violated = 0  # final return variable- computes the total number of rule violations in the source datatframe

    if (not p.TIERED_PRICE_LIM) or (not df.CLIENT.any() in p.TIERED_PRICE_CLIENT):  # LP Mod: same as original code
        p_v_df = price_constraints_df.assign(
            OLD_ING_COST=price_constraints_df.MAC_PRICE_UNIT_ADJ * price_constraints_df.QTY, \
            NEW_ING_COST=price_constraints_df.NEW_PRICE * price_constraints_df.QTY) \
            .groupby(['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP']) \
            .agg({'OLD_ING_COST': 'sum', 'NEW_ING_COST': 'sum'}) \
            .reset_index()

        p_v_df_bounds = p_v_df.assign(LOWER_BOUND=p_v_df.OLD_ING_COST * (1 - p.AGG_LOW_FAC) - 1, \
                                      UPPER_BOUND=p_v_df.OLD_ING_COST * (1 + p.AGG_UP_FAC) + 1, \
                                      RULES_VIOLATED_UB=lambda x: np.where(x["NEW_ING_COST"] > x["OLD_ING_COST"], 1, 0), \
                                      RULES_VIOLATED_LB=lambda x: np.where(x["NEW_ING_COST"] < x["OLD_ING_COST"], 1, 0))

    rules_violated = p_v_df_bounds.RULES_VIOLATED_UB.sum() + p_v_df_bounds.RULES_VIOLATED_LB.sum()

    if rules_violated >= 1:
        p_v_df_bounds_violation = p_v_df_bounds.loc[
            (p_v_df_bounds["RULES_VIOLATED_UB"] > 0) | (p_v_df_bounds["RULES_VIOLATED_LB"] > 0), ['CLIENT', 'BREAKOUT',
                                                                                                  'REGION',
                                                                                                  'MEASUREMENT',
                                                                                                  'CHAIN_GROUP',
                                                                                                  'LOWER_BOUND',
                                                                                                  'UPPER_BOUND',
                                                                                                  'NEW_ING_COST']]
        print(f'Agg price upper bound violation at {p_v_df_bounds_violation}')
        logging.debug(f'Agg price upper bound violation at {p_v_df_bounds_violation}')

    return rules_violated

def is_column_unique(column):
    '''
    Determines if the price changes (increase or decrease) violate the price constraints/rules 
    Inputs:
        column: the final output dataframe containing price mutable drugs
    Outputs:
        rules violated: binary interger 0 or 1 that determines if the price increase or decrease violates the price contraints/rules  
    '''
    return (column[0] == column).all() # True if all values in a dataframe volumn are the same, False if otherwise

def unc_optimization(lp_data_df, awp_discount_percent = 0.75, cvs_independent=True, pharmacy_chain_groups=None, pref_pharm_list=None, contract_date_df=None):
    '''
    Calculates the NEW drug price based on UNC profitability. This function allows for redistribution of pricing with U&C-excluding clients and pharmacies.
    Client and pharmacy performance is optimized by adjusting MAC prices to redistribute U&C and MAC claims into (or out of) client and pharmacy guarantees for performance reconciliation.
    
    Columns added to or altered in the dataframe:
        'PRICE_CHANGED_UC' indicates this row of the dataframe had a U&C-related price change.
        'RAISED_PRICE_UC' indicates whether the desired price change is an increase (True) or decrease (False).
        'IS_TWOSTEP_UNC' indicates that the U&C optimization will not happen in this price update--instead, we are setting up
                         to allow the next price update to complete the optimization. This occurs when the desired price is 
                         too far away from the current price to reach in a single iteration.
        'IS_MAINTENANCE_UC' indicates that the current price is already optimized and we are merely maintaining it. 
        'MATCH_VCML' indicates a price change made to optimize a different pharmacy on the same VCML or to avoid UC_UNIT upper limits
                     that would cause infeasibilities via cross-row constraints
        'UNC_VALUE' is the expected value for U&C-related price changes. Maintenance and two-step changes have value 0. 
        'UNC_FRAC' is the fraction of all claims we expect to adjudicate as U&C after the price change. 
        'MAC_PRICE_UPPER_LIMIT_UC' is an upper limit for U&C-related price *decreases* only.
        'CURRENT_MAC_PRICE' already exists, but is updated with price points for U&C-related price *increases* only.
        'PRE_UC_MAC_PRICE' is a copy of CURRENT_MAC_PRICE before any alterations were made.
    Input:
        lp_data_df: DataFrame which has the Client, Breakout, Region, Measurement, GPI, NDC, for a given client including the MAC1026 price, UNC percentiles, AWP, MAC price
        cvs_independent: if True, ensure CVS parity; if False, ignore
        pharmacy_chain_groups: an iterable of chain_groups to optimize for pharmacy-side U&C
        pref_pharm_list: a list of preferred pharmacies (to ensure pref-nonpref price relationships are obeyed)
    Output:
        lp_data_df: New Dateframe with lowered or raised drugs prices based on the UNC logic, dataframe used for the LP Optimization
    '''
    import datetime as dt
    
    # Set up percentile data including max/min which are labeled differently
    percentile_columns = sorted([c for c in lp_data_df.columns if 'PCT' in c and 'UCAMT_UNIT' in c])
    available_percentiles = sorted([c[3:5] for c in lp_data_df.columns if 'PCT' in c and 'UCAMT_UNIT' in c])
    lp_data_df['AVG_QTY'] = lp_data_df['QTY']/lp_data_df['CLAIMS']
    percentile_columns = ['MIN_UCAMT_QUANTITY'] + percentile_columns + ['MAX_UCAMT_QUANTITY']
    available_percentiles = ['00'] + available_percentiles + ['100']

    # We want to know how many claims have each U&C price. Since we have discretized the distribution, we actually
    # are getting the info "how many claims have >Nth price but <=N+1th price" for a price drop and 
    # "how many claims have <Nth price but >=N-1th price" for a price raise--the claims that are newly moving MAC to U&C.
    # (Sign convention weirdness: when we drop a price, it becomes MAC, so we actually need to know the U&C prices slightly GREATER
    # than the drop.) Since the = applies to a different end of the range for each direction, we have to separately measure
    # the expected quantity for price drops and price raises, hence the two different columns below.
    lp_data_df['qty_uc_slice_00_raises'] = lp_data_df['QTY_IN_CONSTRAINTS'] - lp_data_df[f'QUANTITY_GT_PCT00']
    for i in range(1, len(available_percentiles)):
        lp_data_df[f'qty_uc_slice_{available_percentiles[i]}_raises'] = (
            lp_data_df[f'QUANTITY_GT_PCT{available_percentiles[i-1]}'] - lp_data_df[f'QUANTITY_GT_PCT{available_percentiles[i]}']
        )
        
    lp_data_df['qty_uc_slice_100_drops'] = lp_data_df['QTY_IN_CONSTRAINTS'] - lp_data_df['QUANTITY_LT_PCT100']
    for i in range(len(available_percentiles)-1):
        lp_data_df[f'qty_uc_slice_{available_percentiles[i]}_drops'] = (
            lp_data_df[f'QUANTITY_LT_PCT{available_percentiles[i+1]}'] - lp_data_df[f'QUANTITY_LT_PCT{available_percentiles[i]}']
        )
    if p.UNC_PHARMACY and p.UNC_CLIENT:
        pharm_qty_col = 'PHARM_QTY'
        client_qty_col = 'QTY'
    elif p.UNC_PHARMACY:
        pharm_qty_col = 'PHARM_QTY'
        client_qty_col = 'PHARM_QTY'
    elif p.UNC_CLIENT:
        pharm_qty_col = 'QTY'
        client_qty_col = 'QTY'
    else:
        raise RuntimeError("If UNC_OPT parameter is True, then at least one of UNC_PHARMACY or UNC_CLIENT parameters must also be True")

    # The price with a surplus/liability of 0 for the client and the pharmacy side, respectively
    # 2025 change - Adding in target pharm disp fees
    lp_data_df['BREAK_EVEN_PHARM_UNIT'] = (lp_data_df['PHARM_TARG_INGCOST_ADJ'] + lp_data_df['PHARM_TARGET_DISP_FEE'])/lp_data_df[pharm_qty_col]
    lp_data_df['BREAK_EVEN_CLIENT_UNIT'] = ((1-lp_data_df['CLIENT_RATE'])*lp_data_df['FULLAWP_ADJ'] + lp_data_df['TARGET_DISP_FEE'])/lp_data_df[client_qty_col]
    
    #since we have different pharmacy guarantee type, need to use separate calculations for each of them
    conditions = [
    (lp_data_df[pharm_qty_col] == 0) & (lp_data_df['PHRM_GRTE_TYPE'] == 'AWP'),
    (lp_data_df[pharm_qty_col] == 0) & (lp_data_df['PHRM_GRTE_TYPE'] == 'NADAC'), 
    (lp_data_df[pharm_qty_col] == 0) & (lp_data_df['PHRM_GRTE_TYPE'] == 'WAC'), 
    (lp_data_df[pharm_qty_col] == 0) & (lp_data_df['PHRM_GRTE_TYPE'] == 'ACC')
    ]

    choices = [
        ((1 - lp_data_df['PHARMACY_RATE']) * lp_data_df['FULLAWP_ADJ'] + lp_data_df['PHARM_TARGET_DISP_FEE']) / lp_data_df[client_qty_col],
        ((1 - lp_data_df['PHARMACY_RATE']) * lp_data_df['PHARM_AVG_NADAC'] * lp_data_df[client_qty_col] + lp_data_df['PHARM_TARGET_DISP_FEE']) / lp_data_df[client_qty_col],
        ((1 - lp_data_df['PHARMACY_RATE']) * lp_data_df['PHARM_AVG_WAC'] * lp_data_df[client_qty_col] + lp_data_df['PHARM_TARGET_DISP_FEE']) / lp_data_df[client_qty_col],
        ((1 - lp_data_df['PHARMACY_RATE']) * lp_data_df['PHARM_AVG_ACC'] * lp_data_df[client_qty_col] + lp_data_df['PHARM_TARGET_DISP_FEE']) / lp_data_df[client_qty_col]
    ]

    lp_data_df['BREAK_EVEN_PHARM_UNIT'] = np.select(
        conditions,
        choices,
        default=lp_data_df['BREAK_EVEN_PHARM_UNIT']
    )
    # Assert statement to check if all PHRM_GRTE_TYPE values are covered
    assert set(lp_data_df['PHRM_GRTE_TYPE'].dropna().unique()) <= {'AWP', 'NADAC', 'WAC', 'ACC'}, "Not all PHRM_GRTE_TYPE values are covered"
    lp_data_df.loc[lp_data_df[client_qty_col]==0, 'BREAK_EVEN_CLIENT_UNIT'] = ((1-lp_data_df.loc[lp_data_df[client_qty_col]==0, 'CLIENT_RATE'])*lp_data_df.loc[lp_data_df[client_qty_col]==0, 'PHARM_FULLAWP_ADJ'] + lp_data_df['TARGET_DISP_FEE'])/lp_data_df.loc[lp_data_df[client_qty_col]==0, pharm_qty_col]
    lp_data_df.loc[(lp_data_df[client_qty_col]==0) & (lp_data_df[pharm_qty_col]==0), 'BREAK_EVEN_CLIENT_UNIT'] = 0
    lp_data_df.loc[(lp_data_df[client_qty_col]==0) & (lp_data_df[pharm_qty_col]==0), 'BREAK_EVEN_PHARM_UNIT'] = 0
    
    # For price drops, we subtract a buffer that is 0.010-0.012 of AWP
    # We use digits of the GPI starting at an odd digit as a multiplier. This is repeatable run to run,
    # but isn't obviously linked to numbers associated with the particular price (as GPI is an alphanumeric string
    # instead of a number) so it looks random-ish. Digits 9-13 were found in testing to have the highest dispersion 
    # of possible multipliers.
    lp_data_df['RANDOM_OFFSET_BUFFER'] = ((0.01 + 0.002*pd.to_numeric(lp_data_df['GPI'].str[9:13], errors='coerce')/10000) 
                                          * lp_data_df['AVG_AWP'])
    
    # We loop over the percentiles twice here. The reason is as follows:
    # The outer loop is over possible *TARGET VALUES* for the U&C optimization. If we have U&C percentile values of 
    # $0.90, $0.95, and $1.00, we want to understand the value of targeting (for example) $0.9001, $0.9501, and $1.0001.
    # The inner loop is over the *U&C DISTRIBUTION*. Once we have selected a target value of (for example) $0.9501,
    # we need to compute the fraction of claims that are MAC vs U&C and how much value each claim will generate
    # with the new U&C price. This has to be re-done for every target price, because every target price means
    # a different distribution of MAC vs U&C adjudication.
    for perc, col in zip(available_percentiles, percentile_columns):
        lp_data_df[f'UC_VALUE_CLIENT_{perc}'] = 0
        lp_data_df[f'UC_VALUE_PHARM_{perc}'] = 0
        lp_data_df[f'EXCESS_MOVEMENT_{perc}'] = 0
        lp_data_df[f'PRICE_RAISE_{perc}'] = lp_data_df[col]>lp_data_df['CURRENT_MAC_PRICE']
        lp_data_df[f'price_target_{perc}'] = lp_data_df[col] - lp_data_df['RANDOM_OFFSET_BUFFER']
        lp_data_df.loc[lp_data_df[f'PRICE_RAISE_{perc}'], f'price_target_{perc}'] = lp_data_df.loc[lp_data_df[f'PRICE_RAISE_{perc}'], col] + 0.0001
        for sumperc, sumcol in zip(available_percentiles, percentile_columns):
            lp_data_df['UC_VALUE_CLIENT_CURR_TEMP'] = 0
            lp_data_df['UC_VALUE_PHARM_CURR_TEMP'] = 0
            lp_data_df['UC_VALUE_CLIENT_NEW_TEMP'] = 0
            lp_data_df['UC_VALUE_PHARM_NEW_TEMP'] = 0
            lp_data_df['IS_UC_OLD'] = lp_data_df[sumcol] <= lp_data_df['CURRENT_MAC_PRICE']
            lp_data_df['IS_UC_NEW'] = lp_data_df[sumcol] <= lp_data_df[f'price_target_{perc}']

            # Excess movement
            # If both are U&C, we don't really care how far we moved it
            # Here and below, exclude any calculations where a U&C percentile is 0, since this represents a lack of data
            # or a data error.
            uc_to_mac = lp_data_df['IS_UC_OLD'] & ~lp_data_df['IS_UC_NEW'] & (lp_data_df[col]>0) & (lp_data_df[sumcol]>0)
            mac_to_uc = ~lp_data_df['IS_UC_OLD'] & lp_data_df['IS_UC_NEW'] & (lp_data_df[col]>0) & (lp_data_df[sumcol]>0)
            lp_data_df.loc[uc_to_mac, f'EXCESS_MOVEMENT_{perc}'] += (
                np.abs(lp_data_df.loc[uc_to_mac, sumcol] - lp_data_df.loc[uc_to_mac, f'price_target_{perc}'])*lp_data_df.loc[uc_to_mac, f'qty_uc_slice_{sumperc}_drops']).fillna(0)
            lp_data_df.loc[mac_to_uc, f'EXCESS_MOVEMENT_{perc}'] += (
                np.abs(lp_data_df.loc[mac_to_uc, sumcol] - lp_data_df.loc[mac_to_uc, f'price_target_{perc}'])*lp_data_df.loc[mac_to_uc, f'qty_uc_slice_{sumperc}_raises']).fillna(0)

            # Claims on MAC count for both U&C-including and U&C-excluding entities
            mac_raise = ~lp_data_df['IS_UC_OLD'] & lp_data_df[f'PRICE_RAISE_{perc}'] & (lp_data_df[col]>0) & (lp_data_df[sumcol]>0)
            # Client guarantee: positive value if the price is less than the break-even price (aka surplus, overperformance)
            lp_data_df.loc[mac_raise, 'UC_VALUE_CLIENT_CURR_TEMP'] = ((lp_data_df.loc[mac_raise,'BREAK_EVEN_CLIENT_UNIT']
                                                                         - lp_data_df.loc[mac_raise, 'CURRENT_MAC_PRICE']
                                                                        )*lp_data_df.loc[mac_raise, f'qty_uc_slice_{sumperc}_raises']).fillna(0)
            # Pharmacy guarantee: positive value if the price is greater than the break-even price (underperformance aka over-reimbursement)
            lp_data_df.loc[mac_raise, 'UC_VALUE_PHARM_CURR_TEMP'] = ((lp_data_df.loc[mac_raise, 'CURRENT_MAC_PRICE']
                                                                        - lp_data_df.loc[mac_raise,'BREAK_EVEN_PHARM_UNIT'] 
                                                                        )*lp_data_df.loc[mac_raise, f'qty_uc_slice_{sumperc}_raises']).fillna(0)
            mac_drop = ~lp_data_df['IS_UC_OLD'] & ~lp_data_df[f'PRICE_RAISE_{perc}'] & (lp_data_df[col]>0) & (lp_data_df[sumcol]>0)
            lp_data_df.loc[mac_drop, 'UC_VALUE_CLIENT_CURR_TEMP'] = ((lp_data_df.loc[mac_drop,'BREAK_EVEN_CLIENT_UNIT']
                                                                         - lp_data_df.loc[mac_drop, 'CURRENT_MAC_PRICE']
                                                                        )*lp_data_df.loc[mac_drop, f'qty_uc_slice_{sumperc}_drops']).fillna(0)
            lp_data_df.loc[mac_drop, 'UC_VALUE_PHARM_CURR_TEMP'] = ((lp_data_df.loc[mac_drop, 'CURRENT_MAC_PRICE']
                                                                        - lp_data_df.loc[mac_drop,'BREAK_EVEN_PHARM_UNIT']
                                                                        )*lp_data_df.loc[mac_drop, f'qty_uc_slice_{sumperc}_drops']).fillna(0)
            mac_raise = ~lp_data_df['IS_UC_NEW'] & lp_data_df[f'PRICE_RAISE_{perc}'] & (lp_data_df[col]>0) & (lp_data_df[sumcol]>0)
            lp_data_df.loc[mac_raise, 'UC_VALUE_CLIENT_NEW_TEMP'] = ((lp_data_df.loc[mac_raise,'BREAK_EVEN_CLIENT_UNIT']
                                                                         - lp_data_df.loc[mac_raise, f'price_target_{perc}']
                                                                    )*lp_data_df.loc[mac_raise, f'qty_uc_slice_{sumperc}_raises']).fillna(0)
            lp_data_df.loc[mac_raise, 'UC_VALUE_PHARM_NEW_TEMP'] = ((lp_data_df.loc[mac_raise, f'price_target_{perc}']
                                                                     - lp_data_df.loc[mac_raise,'BREAK_EVEN_PHARM_UNIT']
                                                                    )*lp_data_df.loc[mac_raise, f'qty_uc_slice_{sumperc}_raises']).fillna(0)

            mac_drop = ~lp_data_df['IS_UC_NEW'] & ~lp_data_df[f'PRICE_RAISE_{perc}'] & (lp_data_df[col]>0) & (lp_data_df[sumcol]>0)
            lp_data_df.loc[mac_drop, 'UC_VALUE_CLIENT_NEW_TEMP'] = ((lp_data_df.loc[mac_drop,'BREAK_EVEN_CLIENT_UNIT']
                                                                         - lp_data_df.loc[mac_drop, f'price_target_{perc}']
                                                                    )*lp_data_df.loc[mac_drop, f'qty_uc_slice_{sumperc}_drops']).fillna(0)
            lp_data_df.loc[mac_drop, 'UC_VALUE_PHARM_NEW_TEMP'] = ((lp_data_df.loc[mac_drop, f'price_target_{perc}']
                                                                   - lp_data_df.loc[mac_drop,'BREAK_EVEN_PHARM_UNIT'].fillna(0)
                                                                    )*lp_data_df.loc[mac_drop, f'qty_uc_slice_{sumperc}_drops']).fillna(0)
            # For claims that are now U&C, they only count towards:
            # - The client guarantee if the client is U&C-including (p.UNC_CLIENT=False)
            # - The pharmacy guarantee for all pharmacies if we are not considering U&C exclusions (p.UNC_PHARMACY=False)
            #   or for the pharmacy chain groups NOT in the pharmacy_chain_groups list when p.UNC_PHARMACY=True
            # Otherwise the value against the guarantee is 0.
            uc_raise = lp_data_df['IS_UC_OLD'] & lp_data_df[f'PRICE_RAISE_{perc}'] & (lp_data_df[col]>0) & (lp_data_df[sumcol]>0)
            if not p.UNC_CLIENT:
                lp_data_df.loc[uc_raise, 'UC_VALUE_CLIENT_CURR_TEMP'] = ((lp_data_df.loc[uc_raise,'BREAK_EVEN_CLIENT_UNIT']
                                                                             - lp_data_df.loc[uc_raise, sumcol]
                                                                        )*lp_data_df.loc[uc_raise, f'qty_uc_slice_{sumperc}_raises']).fillna(0)
            if p.UNC_PHARMACY:
                pharm_mask = ~lp_data_df['CHAIN_GROUP'].isin(pharmacy_chain_groups)
            else:
                pharm_mask=True

            lp_data_df.loc[uc_raise & pharm_mask, 'UC_VALUE_PHARM_CURR_TEMP'] = ((lp_data_df.loc[uc_raise & pharm_mask, sumcol]
                                                                     - lp_data_df.loc[uc_raise & pharm_mask,'BREAK_EVEN_PHARM_UNIT']
                                                                    )*lp_data_df.loc[uc_raise & pharm_mask, f'qty_uc_slice_{sumperc}_raises']).fillna(0)

            uc_drop = lp_data_df['IS_UC_OLD'] & ~lp_data_df[f'PRICE_RAISE_{perc}'] & (lp_data_df[col]>0) & (lp_data_df[sumcol]>0)
            if not p.UNC_CLIENT:
                lp_data_df.loc[uc_drop, 'UC_VALUE_CLIENT_CURR_TEMP'] = ((lp_data_df.loc[uc_drop,'BREAK_EVEN_CLIENT_UNIT']
                                                                             - lp_data_df.loc[uc_drop, sumcol]
                                                                        )*lp_data_df.loc[uc_drop, f'qty_uc_slice_{sumperc}_drops']).fillna(0)

            lp_data_df.loc[uc_drop & pharm_mask, 'UC_VALUE_PHARM_CURR_TEMP'] = ((lp_data_df.loc[uc_drop & pharm_mask, sumcol]
                                                                     - lp_data_df.loc[uc_drop & pharm_mask,'BREAK_EVEN_PHARM_UNIT']
                                                                    )*lp_data_df.loc[uc_drop & pharm_mask, f'qty_uc_slice_{sumperc}_drops']).fillna(0)
            uc_raise = lp_data_df['IS_UC_NEW'] & lp_data_df[f'PRICE_RAISE_{perc}'] & (lp_data_df[col]>0) & (lp_data_df[sumcol]>0)
            if not p.UNC_CLIENT:
                lp_data_df.loc[uc_raise, 'UC_VALUE_CLIENT_NEW_TEMP'] = ((lp_data_df.loc[uc_raise,'BREAK_EVEN_CLIENT_UNIT']
                                                                             - lp_data_df.loc[uc_raise, f'price_target_{perc}']
                                                                        )*lp_data_df.loc[uc_raise, f'qty_uc_slice_{sumperc}_raises']).fillna(0)
            lp_data_df.loc[uc_raise & pharm_mask, 'UC_VALUE_PHARM_NEW_TEMP'] = ((lp_data_df.loc[uc_raise & pharm_mask, f'price_target_{perc}']
                                                                     - lp_data_df.loc[uc_raise & pharm_mask,'BREAK_EVEN_PHARM_UNIT']
                                                                    )*lp_data_df.loc[uc_raise & pharm_mask, f'qty_uc_slice_{sumperc}_raises']).fillna(0)
            uc_drop = lp_data_df['IS_UC_NEW'] & ~lp_data_df[f'PRICE_RAISE_{perc}'] & (lp_data_df[col]>0) & (lp_data_df[sumcol]>0)
            if not p.UNC_CLIENT:
                lp_data_df.loc[uc_drop, 'UC_VALUE_CLIENT_NEW_TEMP'] = ((lp_data_df.loc[uc_drop,'BREAK_EVEN_CLIENT_UNIT']
                                                                             - lp_data_df.loc[uc_drop, f'price_target_{perc}']
                                                                        )*lp_data_df.loc[uc_drop, f'qty_uc_slice_{sumperc}_drops']).fillna(0)
            lp_data_df.loc[uc_drop & pharm_mask, 'UC_VALUE_PHARM_NEW_TEMP'] = ((lp_data_df.loc[uc_drop & pharm_mask, f'price_target_{perc}']
                                                                     - lp_data_df.loc[uc_drop & pharm_mask,'BREAK_EVEN_PHARM_UNIT']
                                                                    )*lp_data_df.loc[uc_drop & pharm_mask, f'qty_uc_slice_{sumperc}_drops']).fillna(0)

            lp_data_df[f'UC_VALUE_CLIENT_{perc}'] += lp_data_df['UC_VALUE_CLIENT_NEW_TEMP'] - lp_data_df['UC_VALUE_CLIENT_CURR_TEMP']
            lp_data_df[f'UC_VALUE_PHARM_{perc}'] += lp_data_df['UC_VALUE_PHARM_NEW_TEMP'] - lp_data_df['UC_VALUE_PHARM_CURR_TEMP']
        lp_data_df.drop(columns=['IS_UC_NEW', 'IS_UC_OLD', 'UC_VALUE_CLIENT_NEW_TEMP', 'UC_VALUE_CLIENT_CURR_TEMP',
                                 'UC_VALUE_PHARM_NEW_TEMP', 'UC_VALUE_PHARM_CURR_TEMP', 'BREAK_EVEN_CLIENT_UNIT', 'BREAK_EVEN_PHARM_UNIT'], errors='ignore')
        # We have "TRUE_UC_VALUE" as the real value (for evaluating the effect of a price change) and "SELECT_VALUE" 
        # as what we use to pick a target price. "SELECT_VALUE" includes a penalty ("EXCESS_MOVEMENT") for making big price
        # changes that only shift a few claims MAC <-> U&C. This penalty is designed to prevent us making large price
        # swings to chase outlier U&C prices caused by data errors or other issues.
        lp_data_df[f'SELECT_UC_VALUE_{perc}'] = lp_data_df[f'UC_VALUE_CLIENT_{perc}'] + lp_data_df[f'UC_VALUE_PHARM_{perc}'] - 3*lp_data_df[f'EXCESS_MOVEMENT_{perc}']
        lp_data_df[f'TRUE_UC_VALUE_{perc}'] = lp_data_df[f'UC_VALUE_CLIENT_{perc}'] + lp_data_df[f'UC_VALUE_PHARM_{perc}']

    # If we have multiple rows with the same SELECT_UC_VALUE, then we will pick the price closest to the current price
    # to minimize disruption
    for perc, col in zip(available_percentiles, percentile_columns):
        lp_data_df[f'DISTANCE_FROM_CURR_{perc}'] = np.abs(lp_data_df['CURRENT_MAC_PRICE'] - lp_data_df[f'price_target_{perc}'])

    # We need to know if price changes are allowable within normal limits for two reasons:
    # - To select only eligible price changes
    # - To check that making a change in one price will not cause an infeasibility due to cross-row price checks
    # For example, we can't raise a CVSSP price to $2.00 if the NONPREF_OTH price is limited to $1.95.
    # That would cause a parity violation.
    # But we also can't use the native priceBounds feature of the LP because we can override some of the considerations
    # in that function (e.g. we are allowed to go beyond UC_UNIT) and we also want to consider effects of 2 iterative price changes.
    # So, we read in necessary data here for the main upper and lower bounds used in the priceBounds function and check those manually.
    # The "2X" bounds are what would happen if we raised a price to its maximum this time, and then raised THAT price to
    # ITS maximum on a later run.
    if p.CLIENT_NAME_TABLEAU.startswith(('WTW','AON')):
        drug_mac_hist = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.DRUG_MAC_HIST_FILE),dtype = p.VARIABLE_TYPE_DIC)
        drug_mac_hist = standardize_df(drug_mac_hist)
        if not (p.INTERCEPTOR_OPT or p.COSTSAVER_CLIENT):
            lp_data_df = pd.merge(lp_data_df, drug_mac_hist[['GPI', 'MAC_LIST', 'BG_FLAG','BEG_Q_PRICE', 'BEG_M_PRICE']], how='left', on=['GPI', 'MAC_LIST', 'BG_FLAG'])

        #For WTW clients, we use quarterly beginning price as beginning period price 
        if p.CLIENT_NAME_TABLEAU.startswith('WTW'):
            lp_data_df['BEG_PERIOD_PRICE'] = lp_data_df['BEG_Q_PRICE'].fillna(lp_data_df['CURRENT_MAC_PRICE'])

        #For AON clients, we use monthly beginning price as beginning period price        
        if p.CLIENT_NAME_TABLEAU.startswith('AON'):
            lp_data_df['BEG_PERIOD_PRICE'] = lp_data_df['BEG_M_PRICE'].fillna(lp_data_df['CURRENT_MAC_PRICE'])
            
    # Read in existing U&C overrides so we can include them in consistency checks
    wmt_unc_override = standardize_df(pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.WMT_UNC_PRICE_OVERRIDE_FILE), dtype = p.VARIABLE_TYPE_DIC))
        
    lp_data_df = pd.merge(lp_data_df, wmt_unc_override[['CLIENT','GPI_NDC','MAC', 'BG_FLAG' 'UNC_OVRD_AMT']], \
                                how = 'left', on = ['CLIENT','GPI_NDC','MAC', 'BG_FLAG'])
    
    # Check whether any price movements cross the client/pharmacy guarantee without possible U&C prices spanning the client/pharmacy guarantee
    # Example: MOVE_CONCURRENCE would be False if the pharmacy guarantee was 88%, the U&C prices ranged from 80-83% off, 
    # and we found we could get a better result by moving prices from 90% off to 83.5% off. Sure, but that's just a normal price change!
    lp_data_df['MOVE_CONCURRENCE'] = (((lp_data_df['MIN_UCAMT_QUANTITY'] < lp_data_df['CURRENT_MAC_PRICE']) & (lp_data_df['CURRENT_MAC_PRICE'] < lp_data_df['MAX_UCAMT_QUANTITY']))
                                      | ((lp_data_df['BREAK_EVEN_CLIENT_UNIT'] < lp_data_df['CURRENT_MAC_PRICE']) & (lp_data_df['BREAK_EVEN_CLIENT_UNIT'] < lp_data_df['MAX_UCAMT_QUANTITY']) & p.UNC_CLIENT)
                                      | ((lp_data_df['BREAK_EVEN_CLIENT_UNIT'] > lp_data_df['CURRENT_MAC_PRICE']) & (lp_data_df['BREAK_EVEN_CLIENT_UNIT'] > lp_data_df['MIN_UCAMT_QUANTITY']) & p.UNC_CLIENT)
                                      | ((lp_data_df['BREAK_EVEN_PHARM_UNIT'] < lp_data_df['CURRENT_MAC_PRICE']) & (lp_data_df['BREAK_EVEN_PHARM_UNIT'] < lp_data_df['MAX_UCAMT_QUANTITY']) & p.UNC_PHARMACY)
                                      | ((lp_data_df['BREAK_EVEN_PHARM_UNIT'] > lp_data_df['CURRENT_MAC_PRICE']) & (lp_data_df['BREAK_EVEN_PHARM_UNIT'] > lp_data_df['MIN_UCAMT_QUANTITY']) & p.UNC_PHARMACY)
                                     )
    
    # wtw logic from CPMO_lp_functions
    if not p.CLIENT_NAME_TABLEAU.startswith(('WTW','AON')) or p.REMOVE_WTW_RESTRICTION:
        lp_data_df['WTW_UPPER_LIMIT'] = 1000000*lp_data_df.CURRENT_MAC_PRICE
        lp_data_df['WTW_UPPER_LIMIT_2X'] = 1000000*lp_data_df.CURRENT_MAC_PRICE
    else:
        lp_data_df['WTW_UPPER_LIMIT'] = 1.249*lp_data_df.BEG_PERIOD_PRICE
        lp_data_df.loc[(1.249*lp_data_df.BEG_PERIOD_PRICE < lp_data_df.MAC1026_UNIT_PRICE) & (lp_data_df.CHAIN_GROUP != 'MAIL') & (lp_data_df.CHAIN_GROUP != 'MCHOICE'), 'WTW_UPPER_LIMIT'] = lp_data_df.MAC1026_UNIT_PRICE
        lp_data_df['WTW_UPPER_LIMIT_2X'] = 1.249**2*lp_data_df.BEG_PERIOD_PRICE
        lp_data_df.loc[(1.249*lp_data_df.BEG_PERIOD_PRICE < lp_data_df.MAC1026_UNIT_PRICE) & (lp_data_df.CHAIN_GROUP != 'MAIL') & (lp_data_df.CHAIN_GROUP != 'MCHOICE'), 'WTW_UPPER_LIMIT_2X'] = lp_data_df.MAC1026_UNIT_PRICE*1.249

    lp_data_df['PRICE_REIMB_ADJ'] = lp_data_df['QTY'] * lp_data_df['CURRENT_MAC_PRICE'] 
    lp_data_df['PRICE_REIMB_CLAIM'] = lp_data_df['PRICE_REIMB_ADJ'] / lp_data_df['CLAIMS']
    lp_data_df['PHARM_PRICE_REIMB_ADJ'] = lp_data_df[pharm_qty_col] * lp_data_df['CURRENT_MAC_PRICE']
    lp_data_df['PHARM_PRICE_REIMB_CLAIM'] = lp_data_df['PHARM_PRICE_REIMB_ADJ'] / lp_data_df['PHARM_CLAIMS']
    lp_data_df['PRICING_PRICE_REIMB_CLAIM'] = lp_data_df[['PRICE_REIMB_CLAIM', 'PHARM_PRICE_REIMB_CLAIM']].max(axis=1, skipna=True)   
    lp_data_df['PRICING_CLAIMS'] = lp_data_df[['CLAIMS', 'PHARM_CLAIMS']].max(axis=1, skipna=True)   
    lp_data_df['PRICING_QTY'] = lp_data_df[['QTY', pharm_qty_col]].max(axis=1, skipna=True)   
    if p.TIERED_PRICE_LIM:
        upper_bound = p.UNC_PRICE_BOUNDS_DF["upper_bound"]
        member_disruption_amt = p.UNC_PRICE_BOUNDS_DF["max_dollar_increase"]
        up_fac = p.UNC_PRICE_BOUNDS_DF["max_percent_increase"]
        # check this
        lp_data_df['UNC_CHECK_TIER'] = np.digitize(lp_data_df['PRICING_PRICE_REIMB_CLAIM'], upper_bound)
        lp_data_df.loc[lp_data_df['PRICING_PRICE_REIMB_CLAIM'].isna(), 'UNC_CHECK_TIER'] = 1 # equivalent of ZERO_QTY_TIGHT_BOUNDS, shouldn't matter
        lp_data_df.loc[lp_data_df['PRICING_PRICE_REIMB_CLAIM']==np.inf, 'UNC_CHECK_TIER'] = 1 # equivalent of ZERO_QTY_TIGHT_BOUNDS, shouldn't matter
        lp_data_df['UNC_CHECK_UB_MULT'] = lp_data_df['CURRENT_MAC_PRICE']*(1+lp_data_df['UNC_CHECK_TIER'].apply(lambda x: up_fac[x]))
        lp_data_df['UNC_CHECK_UB_ADD'] = (lp_data_df['PRICING_PRICE_REIMB_CLAIM']+lp_data_df['UNC_CHECK_TIER'].apply(lambda x: member_disruption_amt[x])
                                         )*(lp_data_df['PRICING_CLAIMS']/lp_data_df['PRICING_QTY'])
        lp_data_df['UNC_CHECK_UB'] = lp_data_df[['UNC_CHECK_UB_MULT', 'UNC_CHECK_UB_ADD']].min(axis=1)
        lp_data_df['UNC_CHECK_LB'] = lp_data_df['CURRENT_MAC_PRICE']*(1-p.GPI_LOW_FAC)
        lp_data_df.drop(columns=['UNC_CHECK_TIER', 'UNC_CHECK_UB_MULT', 'UNC_CHECK_UB_ADD'], inplace=True)
        
        lp_data_df['PRICING_PRICE_REIMB_CLAIM_MAX'] = lp_data_df['PRICING_PRICE_REIMB_CLAIM']*lp_data_df['UNC_CHECK_UB']/lp_data_df['CURRENT_MAC_PRICE']
        lp_data_df['PRICING_PRICE_REIMB_CLAIM_MIN'] = lp_data_df['PRICING_PRICE_REIMB_CLAIM']*lp_data_df['UNC_CHECK_LB']/lp_data_df['CURRENT_MAC_PRICE']
        lp_data_df['UNC_CHECK_TIER_2X'] = np.digitize(lp_data_df['PRICING_PRICE_REIMB_CLAIM_MAX'], upper_bound)
        lp_data_df.loc[lp_data_df['PRICING_PRICE_REIMB_CLAIM_MAX'].isna(), 'UNC_CHECK_TIER_2X'] = 1 # equivalent of ZERO_QTY_TIGHT_BOUNDS, shouldn't matter
        lp_data_df.loc[lp_data_df['PRICING_PRICE_REIMB_CLAIM_MAX']==np.inf, 'UNC_CHECK_TIER_2X'] = 1 # equivalent of ZERO_QTY_TIGHT_BOUNDS, shouldn't matter
        lp_data_df['UNC_CHECK_UB_MULT_2X'] = lp_data_df['UNC_CHECK_UB']*(1+lp_data_df['UNC_CHECK_TIER_2X'].apply(lambda x: up_fac[x]))
        lp_data_df['UNC_CHECK_UB_ADD_2X'] = (lp_data_df['PRICING_PRICE_REIMB_CLAIM_MAX']+lp_data_df['UNC_CHECK_TIER_2X'].apply(lambda x: member_disruption_amt[x])
                                         )*(lp_data_df['PRICING_CLAIMS']/lp_data_df['PRICING_QTY'])
        lp_data_df['UNC_CHECK_UB_2X'] = lp_data_df[['UNC_CHECK_UB_MULT_2X', 'UNC_CHECK_UB_ADD_2X']].min(axis=1)
        lp_data_df['UNC_CHECK_LB_2X'] = lp_data_df['UNC_CHECK_LB']*(1-p.GPI_LOW_FAC)
        lp_data_df.drop(columns=['UNC_CHECK_TIER_2X', 'UNC_CHECK_UB_MULT_2X', 'UNC_CHECK_UB_ADD_2X', 'PRICING_PRICE_REIMB_CLAIM_MAX', 'PRICING_PRICE_REIMB_CLAIM_MIN'], inplace=True)
    else:
        lp_data_df['UNC_CHECK_UB'] = lp_data_df['CURRENT_MAC_PRICE']*(1+p.GPI_UP_FAC)
        lp_data_df['UNC_CHECK_LB'] = lp_data_df['CURRENT_MAC_PRICE']*(1-p.GPI_LOW_FAC)
        lp_data_df['UNC_CHECK_UB_2X'] = lp_data_df['UNC_CHECK_UB']*(1+p.GPI_UP_FAC)
        lp_data_df['UNC_CHECK_LB_2X'] = lp_data_df['UNC_CHECK_LB']*(1-p.GPI_LOW_FAC)

    lp_data_df['AWP_UPPER_LIMIT'] = np.where(lp_data_df.BG_FLAG == 'G', 
                                             awp_discount_percent*lp_data_df['AVG_AWP'], 
                                             (1 - p.BRAND_NON_MAC_RATE)*lp_data_df['AVG_AWP'])
    lp_data_df.loc[lp_data_df['MAC1026_UNIT_PRICE']>0, 'UNC_CHECK_LB'] = lp_data_df.loc[
        lp_data_df['MAC1026_UNIT_PRICE']>0, 
        ['UNC_CHECK_LB', 'MAC1026_UNIT_PRICE']].max(axis=1)
    lp_data_df.loc[lp_data_df['MAC1026_UNIT_PRICE']>0, 'UNC_CHECK_LB_2X'] = lp_data_df.loc[
        lp_data_df['MAC1026_UNIT_PRICE']>0, 
        ['UNC_CHECK_LB_2X', 'MAC1026_UNIT_PRICE']].max(axis=1)
    # Combine these into upper and lower bounds
    lp_data_df['UNC_CHECK_UB'] = lp_data_df[['UNC_CHECK_UB', 'WTW_UPPER_LIMIT', 'GOODRX_UPPER_LIMIT', 'AWP_UPPER_LIMIT']].min(axis=1)
    lp_data_df['UNC_CHECK_UB'] = lp_data_df[['UNC_CHECK_UB', 'UNC_CHECK_LB']].max(axis=1)
    lp_data_df['UNC_CHECK_UB_2X'] = lp_data_df[['UNC_CHECK_UB_2X', 'WTW_UPPER_LIMIT_2X', 'GOODRX_UPPER_LIMIT', 'AWP_UPPER_LIMIT']].min(axis=1)
    lp_data_df['UNC_CHECK_UB_2X'] = lp_data_df[['UNC_CHECK_UB_2X', 'UNC_CHECK_LB_2X']].max(axis=1)
    if p.INTERCEPTOR_OPT:
        # Anything that might override an INTERCEPT_HIGH or INTERCEPT_LOW would also override our U&C optimization,
        # so we don't need the full bounds logic here.
        lp_data_df['UNC_CHECK_UB'] = lp_data_df[['UNC_CHECK_UB', 'INTERCEPT_HIGH']].min(axis=1)
        lp_data_df['UNC_CHECK_UB_2X'] = lp_data_df[['UNC_CHECK_UB_2X', 'INTERCEPT_HIGH']].min(axis=1)
        lp_data_df['UNC_CHECK_LB'] = lp_data_df[['UNC_CHECK_LB', 'INTERCEPT_LOW']].max(axis=1)
        lp_data_df['UNC_CHECK_LB_2X'] = lp_data_df[['UNC_CHECK_LB_2X', 'INTERCEPT_LOW']].max(axis=1)
    
    
    lp_data_df.loc[lp_data_df['UNC_OVRD_AMT'].notna(), 'UNC_CHECK_UB'] = lp_data_df.loc[lp_data_df['UNC_OVRD_AMT'].notna(), 'UNC_OVRD_AMT']
    lp_data_df.loc[lp_data_df['UNC_OVRD_AMT'].notna(), 'UNC_CHECK_LB'] = lp_data_df.loc[lp_data_df['UNC_OVRD_AMT'].notna(), 'UNC_OVRD_AMT']
    lp_data_df.loc[lp_data_df['UNC_OVRD_AMT'].notna(), 'UNC_CHECK_UB_2X'] = lp_data_df.loc[lp_data_df['UNC_OVRD_AMT'].notna(), 'UNC_OVRD_AMT']
    lp_data_df.loc[lp_data_df['UNC_OVRD_AMT'].notna(), 'UNC_CHECK_LB_2X'] = lp_data_df.loc[lp_data_df['UNC_OVRD_AMT'].notna(), 'UNC_OVRD_AMT']
    
    # Having completed our bounds computations, we now select WHICH U&C prices we will use as a target.
    # Right now we have a horizontal dataframe 
    # (INDEX, PRICE_PCT_00, VALUE_PCT_00, PRICE_PCT_01, VALUE_PCT_01, ... PRICE_PCT_100, VALUE_PCT_100)
    # We manually pivot this so we instead have a row per index AND PERCENTILE:
    # (INDEX, 00, PRICE_PCT_00, VALUE_PCT_00)
    # (INDEX, 01, PRICE_PCT_01, VALUE_PCT_01)
    # ...
    # (INDEX, 100, PRICE_PCT_100, VALUE_PCT_100)
    # This allows us to easily vectorize computations, at the expense of having to merge the results back on to the main dataframe
    # at the end.
    lp_data_ucprice = []
    price_selection_cols = ['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GPI_NDC',
                            'CURRENT_MAC_PRICE', 'PRICING_PRICE_REIMB_CLAIM', 'PRICING_CLAIMS', 'PRICING_QTY', 
                            'AVG_AWP', 'MAC1026_UNIT_PRICE', 'UNC_CHECK_UB', 'UNC_CHECK_LB', 'UNC_CHECK_UB_2X', 'UNC_CHECK_LB_2X']
    for perc, col in zip(available_percentiles, percentile_columns):
        # remove (most) NDC-level pricing...okay to miss a few (where there is an NDC star but also non-star NDCs
        # for the same GPI), since they will be excluded in CPMO. Just saves us some computation time here to exclude
        # obviously unsuitable rows.
        tdf = lp_data_df[(lp_data_df['NDC'] == '*'*11) & (lp_data_df['CURRENT_MAC_PRICE']>0)]
        #tdf = tdf[lp_data_df['MOVE_CONCURRENCE']][price_selection_cols + [f'SELECT_UC_VALUE_{perc}', f'TRUE_UC_VALUE_{perc}', col,
        tdf = tdf[price_selection_cols + [f'SELECT_UC_VALUE_{perc}', f'TRUE_UC_VALUE_{perc}', f'price_target_{perc}',
                             f'DISTANCE_FROM_CURR_{perc}', f'PRICE_RAISE_{perc}', f'QUANTITY_LT_PCT{perc}', f'QUANTITY_GT_PCT{perc}', 
                             f'UC_VALUE_CLIENT_{perc}', f'UC_VALUE_PHARM_{perc}']].rename(
                columns={f'price_target_{perc}': 'NEW_MAC_PRICE', f'SELECT_UC_VALUE_{perc}': 'UC_VALUE', f'TRUE_UC_VALUE_{perc}': 'TRUE_UC_VALUE', 
                         f'DISTANCE_FROM_CURR_{perc}': 'DISTANCE_FROM_CURR', f'PRICE_RAISE_{perc}': 'PRICE_RAISE',
                         f'QUANTITY_LT_PCT{perc}': 'QUANTITY_LT_UNC', f'QUANTITY_GT_PCT{perc}': 'QUANTITY_GT_UNC',
                         f'UC_VALUE_CLIENT_{perc}': 'UNC_VALUE_CLIENT', f'UC_VALUE_PHARM_{perc}': 'UNC_VALUE_PHARM', 
                        }
        )
        tdf['PERCENTILE'] = perc
        lp_data_ucprice.append(tdf)
    lp_data_ucprice = pd.concat(lp_data_ucprice)
        
    lp_data_ucprice['PRICE_CHANGE'] = 'Lowered price'
    lp_data_ucprice.loc[lp_data_ucprice['PRICE_RAISE'], 'PRICE_CHANGE'] = 'Raised price'
    lp_data_ucprice.loc[lp_data_ucprice['DISTANCE_FROM_CURR'] < 1.E-5, 'PRICE_CHANGE'] = 'Original price'

    # Check that proposed price change does not exceed price change limit. if it does: check if two price changes would get there. 
    # If two price changes gets us there: move the price to the limit of allowed prices. If it does not: remove the possible price changes
    lp_data_ucprice['IS_TWOSTEP_UNC'] = False
    lp_data_ucprice['NEW_PRICING_PRICE_REIMB_CLAIM'] = (lp_data_ucprice['PRICING_PRICE_REIMB_CLAIM']) * lp_data_ucprice['NEW_MAC_PRICE']/lp_data_ucprice['CURRENT_MAC_PRICE']
    lp_data_ucprice['CLAIM_INCREASE_PCT'] = (lp_data_ucprice['NEW_PRICING_PRICE_REIMB_CLAIM']  - lp_data_ucprice['PRICING_PRICE_REIMB_CLAIM'] ) / lp_data_ucprice['PRICING_PRICE_REIMB_CLAIM'] # need div by zero check
    lp_data_ucprice['CLAIM_INCREASE'] = lp_data_ucprice['NEW_PRICING_PRICE_REIMB_CLAIM']  - lp_data_ucprice['PRICING_PRICE_REIMB_CLAIM']
    tier_bound_broken = ((lp_data_ucprice['PRICE_CHANGE']=='Raised price') & (lp_data_ucprice['NEW_MAC_PRICE'] > lp_data_ucprice['UNC_CHECK_UB'])) | ((lp_data_ucprice['PRICE_CHANGE']=='Lowered price') & (lp_data_ucprice['NEW_MAC_PRICE'] < lp_data_ucprice['UNC_CHECK_LB']))
    tier_bound_2x_broken = ((lp_data_ucprice['PRICE_CHANGE']=='Raised price') & (lp_data_ucprice['NEW_MAC_PRICE'] > lp_data_ucprice['UNC_CHECK_UB_2X'])) | ((lp_data_ucprice['PRICE_CHANGE']=='Lowered price') & (lp_data_ucprice['NEW_MAC_PRICE'] < lp_data_ucprice['UNC_CHECK_LB_2X']))
    raise_2x = tier_bound_broken & ~tier_bound_2x_broken & (lp_data_ucprice['PRICE_CHANGE']=='Raised price')
    lower_2x = tier_bound_broken & ~tier_bound_2x_broken & (lp_data_ucprice['PRICE_CHANGE']=='Lowered price')
    lp_data_ucprice['TIER_BOUND_MET'] = ~tier_bound_broken
    lp_data_ucprice['TIER_BOUND_MET_2X'] = ~tier_bound_2x_broken
    # 0.0001 offset is to avoid numerical errors w/ limits for shared VCMLs in CPMO
    lp_data_ucprice.loc[raise_2x, 'NEW_MAC_PRICE'] = lp_data_ucprice.loc[raise_2x, 'UNC_CHECK_UB']-0.0001
    lp_data_ucprice.loc[lower_2x, 'NEW_MAC_PRICE'] = lp_data_ucprice.loc[lower_2x, 'UNC_CHECK_LB']+0.0001
    lp_data_ucprice.loc[raise_2x | lower_2x, 'IS_TWOSTEP_UNC'] = True
    lp_data_ucprice = lp_data_ucprice[~tier_bound_2x_broken]
    
    lp_data_ucprice = lp_data_ucprice[lp_data_ucprice['UC_VALUE']>0]
    
    # Sort values and keep only the highest value (and closest distance if highest value has multiple rows).
    lp_data_ucprice = lp_data_ucprice.sort_values('DISTANCE_FROM_CURR', ascending=True).sort_values(['UC_VALUE'], ascending=False)
    lp_data_ucprice.drop_duplicates(subset=['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GPI_NDC'], inplace=True)
    
    lp_data_df_temp = lp_data_df.merge(lp_data_ucprice[['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT','BG_FLAG', 
                                                       'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GPI_NDC',
                                                       'PRICE_CHANGE', 'NEW_MAC_PRICE', 'IS_TWOSTEP_UNC',
                                                       'TRUE_UC_VALUE', 'UNC_VALUE_CLIENT', 'UNC_VALUE_PHARM', 'QUANTITY_LT_UNC', 'QUANTITY_GT_UNC']], 
                                       on=['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GPI_NDC'], how='left')
    # We fillna for the bool column separately, since filling with zeros was causing issues
    lp_data_df_temp['IS_TWOSTEP_UNC'] = lp_data_df_temp['IS_TWOSTEP_UNC'].fillna(False)
    lp_data_df_temp = lp_data_df_temp.fillna(0)
    assert np.abs(lp_data_df.FULLAWP_ADJ.sum() - lp_data_df_temp.FULLAWP_ADJ.sum()) < 1.E-4, "Merge error adding U&C price targets"
    lp_data_df = lp_data_df_temp
    lp_data_df.loc[lp_data_df['PRICE_CHANGE']==0, 'PRICE_CHANGE'] = 'Original price'
    lp_data_df.loc[lp_data_df['NEW_MAC_PRICE'].isna(), 'NEW_MAC_PRICE'] = lp_data_df.loc[lp_data_df['NEW_MAC_PRICE'].isna(), 'CURRENT_MAC_PRICE']
    
    # Having selected places where a price MOVEMENT benefits us, we also want to make sure we don't UNdo a price change that is
    # currently beneficial. Here, we check for prices > a 50th or greater percentile of the U&C distribution, or
    # < a 50th or lesser percentile, and set those percentiles as limits as if they were movements.
    # We detect all candidate prices here, and choose which ones to keep below.
    lp_data_df['IS_MAINTENANCE_UC'] = False
    def pick_lower_limit(row):
        # work from the highest price backwards
        for perc, col in zip(available_percentiles[::-1], percentile_columns[::-1]):
            if row[col]>0 and row['QTY_IN_CONSTRAINTS']>0 :
                awp_disc_percent_temp = np.where(row['BG_FLAG'] == 'G', 
                                                 awp_discount_percent, 
                                                 (1 - p.BRAND_NON_MAC_RATE)).tolist()
                if row['CURRENT_MAC_PRICE'] > row[col] and row['CURRENT_MAC_PRICE'] <= awp_disc_percent_temp * row['AVG_AWP'] and row['CURRENT_MAC_PRICE'] <= row['UNC_CHECK_UB'] and row['CURRENT_MAC_PRICE'] >= row['UNC_CHECK_LB']:
                    return pd.Series([row['CURRENT_MAC_PRICE'], row[f'QUANTITY_GT_PCT{perc}']])
        return pd.Series([np.nan, row.QUANTITY_GT_UNC])
    lp_data_df[['UC_MAINTENANCE_LOWER_LIMIT', 'QTY_GT_UNC']] = lp_data_df.apply(pick_lower_limit, axis=1)

    def pick_upper_limit(row):
        for perc, col in zip(available_percentiles, percentile_columns):
            if row[col]>0 and row['QTY_IN_CONSTRAINTS']>0:
                if row['CURRENT_MAC_PRICE'] < row[col] and row['CURRENT_MAC_PRICE']>=row.MAC1026_UNIT_PRICE and row['CURRENT_MAC_PRICE'] <= row['UNC_CHECK_UB'] and row['CURRENT_MAC_PRICE'] >= row['UNC_CHECK_LB']:
                    return pd.Series([row[col], row[f'QUANTITY_LT_PCT{perc}']])
        return pd.Series([np.nan, row.QUANTITY_LT_UNC])
    lp_data_df[['UC_MAINTENANCE_UPPER_LIMIT', 'QTY_LT_UNC']] = lp_data_df.apply(pick_upper_limit, axis=1)

    maintenance_limits_apply = ((p.UNC_CLIENT & ((lp_data_df['QTY_PROJ_EOY']>0) | (lp_data_df['PHARM_QTY_PROJ_EOY']>0)) )
                         | (p.UNC_PHARMACY & ((lp_data_df['QTY_PROJ_EOY']>0) | (lp_data_df['PHARM_QTY_PROJ_EOY']>0)) & (lp_data_df['CHAIN_GROUP'].isin(pharmacy_chain_groups)))
                        )
    # Keep only prices that don't already have a U&C-related price change!
    maintenance_limits_apply = maintenance_limits_apply & (lp_data_df['PRICE_CHANGE'] == 'Original price')
    lp_data_df.loc[maintenance_limits_apply & lp_data_df['UC_MAINTENANCE_LOWER_LIMIT'].notna(), 'NEW_MAC_PRICE'] = \
        lp_data_df.loc[maintenance_limits_apply & lp_data_df['UC_MAINTENANCE_LOWER_LIMIT'].notna(), 'UC_MAINTENANCE_LOWER_LIMIT']
    lp_data_df.loc[maintenance_limits_apply & lp_data_df['UC_MAINTENANCE_LOWER_LIMIT'].notna(), 'PRICE_CHANGE'] = 'Raised price'
    lp_data_df.loc[maintenance_limits_apply & lp_data_df['UC_MAINTENANCE_UPPER_LIMIT'].notna(), 'NEW_MAC_PRICE'] = \
        lp_data_df.loc[maintenance_limits_apply & lp_data_df['UC_MAINTENANCE_UPPER_LIMIT'].notna(), 'UC_MAINTENANCE_UPPER_LIMIT']
    lp_data_df.loc[maintenance_limits_apply & lp_data_df['UC_MAINTENANCE_UPPER_LIMIT'].notna(), 'PRICE_CHANGE'] = 'Lowered price'
    lp_data_df.loc[maintenance_limits_apply & (lp_data_df['UC_MAINTENANCE_LOWER_LIMIT'].notna() | lp_data_df['UC_MAINTENANCE_UPPER_LIMIT'].notna()), 'IS_MAINTENANCE_UC'] = True

    # Now, we need to check that all of the prices we've chosen for U&C optimization or U&C maintenance are actually allowable
    # prices. Most of the rest of this function is performing all those checks.
    
    #------------------------------------------------------
    # Unify price changes for chain_groups on the same VCML
    lp_data_df['MATCH_VCML'] = False
    lp_data_df['VCML_BOUND'] = 'Satisfied'
    # Pick out places where we want to make U&C price changes on more than one chain group on a shared VCML
    shared_vcmls = lp_data_df[lp_data_df['PRICE_CHANGE']!='Original price'].groupby(['MAC_LIST', 'GPI', 'NDC', 'BG_FLAG'], as_index=False)['CHAIN_GROUP'].count()
    shared_vcmls = shared_vcmls[shared_vcmls['CHAIN_GROUP']>1]
    for _, row in shared_vcmls.iterrows():
        possible_conflict_mask = ((lp_data_df['MAC_LIST']==row.MAC_LIST)
                                  & (lp_data_df['GPI']==row.GPI)
                                  & (lp_data_df['NDC']==row.NDC)
                                  & (lp_data_df['BG_FLAG']==row.BG_FLAG)
                                 )
        # Just pick the highest-value target and use that for all VCMLs -- value from carefully optimizing across all VCMLs is likely minimal, 
        # since shared VCMLs only apply to our smallest chain_groups
        # We need to apply the price to ALL chain_groups, but only ones with a selected U&C price
        # will have all the columns populated, so only consider those for selection.
        possible_choices = lp_data_df[possible_conflict_mask & (lp_data_df['PRICE_CHANGE']!='Original price')]
        # Only consider price changes eligible for ALL chain_groups
        possible_choices = possible_choices[((possible_choices['NEW_MAC_PRICE'] >= possible_choices['UNC_CHECK_LB'].max()) & (possible_choices['PRICE_CHANGE']=='Raised price')) | ((possible_choices['NEW_MAC_PRICE'] <= possible_choices['UNC_CHECK_UB'].min())  &(possible_choices['PRICE_CHANGE']=='Lowered price'))]
        if len(possible_choices)==0:
            lp_data_df.loc[possible_conflict_mask, 'VCML_BOUND'] = 'Broken'
            continue
        best_value = possible_choices['TRUE_UC_VALUE'].max()
        best_value_move = possible_choices[possible_choices['TRUE_UC_VALUE'] == best_value]['PRICE_CHANGE'].iloc[0]
        possible_choices = possible_choices[possible_choices['PRICE_CHANGE'] == best_value_move]
        if best_value_move == 'Raised price':
            # Pick the biggest raise to capture more value
            new_price = possible_choices['NEW_MAC_PRICE'].max()
        else:
            new_price = possible_choices['NEW_MAC_PRICE'].min()
        lp_data_df.loc[possible_conflict_mask, 'NEW_MAC_PRICE'] = new_price
        lp_data_df.loc[possible_conflict_mask, 'PRICE_CHANGE'] = best_value_move
        # Just in case there were some raises & some drops in the initial set, drop the U&C value to 0 for anything that got overwritten with a move in the other direction
        lp_data_df.loc[possible_conflict_mask & (lp_data_df['PRICE_CHANGE']!=best_value_move), 'TRUE_UC_VALUE'] = 0 
        lp_data_df.loc[possible_conflict_mask & (lp_data_df['TRUE_UC_VALUE']!=best_value), 'MATCH_VCML'] = True    
    vcml_ub_lb = lp_data_df.groupby(['MAC_LIST', 'GPI', 'NDC', 'BG_FLAG'], as_index=False
                                   ).agg({'UNC_CHECK_LB': np.nanmax, 'UNC_CHECK_UB': np.nanmin}
                                        ).rename(columns={'UNC_CHECK_LB': 'UNC_CHECK_LB_VCML', 'UNC_CHECK_UB': 'UNC_CHECK_UB_VCML'})
    lp_data_df_temp = lp_data_df.merge(vcml_ub_lb, on=['MAC_LIST', 'GPI', 'NDC', 'BG_FLAG'])
    assert np.abs(lp_data_df_temp.FULLAWP_ADJ.sum() - lp_data_df_temp.FULLAWP_ADJ.sum())<1.E-4, "join failure in VCML bounds check"
    lp_data_df = lp_data_df_temp
    lp_data_df.loc[(lp_data_df['NEW_MAC_PRICE']>0) & (((lp_data_df['NEW_MAC_PRICE']<lp_data_df['UNC_CHECK_LB_VCML']) & (lp_data_df['PRICE_CHANGE']=='Lowered price')) 
                   | ((lp_data_df['NEW_MAC_PRICE']>lp_data_df['UNC_CHECK_UB_VCML']) & (lp_data_df['PRICE_CHANGE']=='Raised price'))), 'VCML_BOUND'] = 'Broken'
    vcml_new_macs = lp_data_df[(lp_data_df['PRICE_CHANGE']!='Original price') & ~(lp_data_df['VCML_BOUND'].str.contains("Broken"))].groupby(['MAC_LIST', 'GPI', 'NDC','BG_FLAG'], as_index=False
                                   ).agg({'NEW_MAC_PRICE': max, 'PRICE_CHANGE': max, 'IS_MAINTENANCE_UC': max, 'IS_TWOSTEP_UNC': max} # should never have more than 1 unique value
                                        ).rename(columns={'NEW_MAC_PRICE': 'VCML_NEW_MAC_PRICE', 
                                                          'PRICE_CHANGE': 'VCML_PRICE_CHANGE',
                                                          'IS_MAINTENANCE_UC': 'VCML_IS_MAINTENANCE_UC',
                                                          'IS_TWOSTEP_UNC': 'VCML_IS_TWOSTEP_UNC'
                                                         })
    lp_data_df_temp = lp_data_df.merge(vcml_new_macs, on=['MAC_LIST', 'GPI', 'NDC', 'BG_FLAG'], how='left')
    assert np.abs(lp_data_df_temp.FULLAWP_ADJ.sum() - lp_data_df_temp.FULLAWP_ADJ.sum())<1.E-4, "join failure in VCML bounds check"
    lp_data_df = lp_data_df_temp
    update_vcml_mask = (lp_data_df['VCML_NEW_MAC_PRICE'].notna())
    lp_data_df.loc[update_vcml_mask  & (lp_data_df['PRICE_CHANGE']=='Original price'), 'MATCH_VCML'] = True
    lp_data_df.loc[update_vcml_mask, 'NEW_MAC_PRICE'] = lp_data_df.loc[update_vcml_mask, 'VCML_NEW_MAC_PRICE']
    lp_data_df.loc[update_vcml_mask, 'IS_MAINTENANCE_UC'] = lp_data_df.loc[update_vcml_mask, 'VCML_IS_MAINTENANCE_UC']
    lp_data_df.loc[update_vcml_mask, 'IS_TWOSTEP_UNC'] = lp_data_df.loc[update_vcml_mask, 'VCML_IS_TWOSTEP_UNC']
    lp_data_df.loc[update_vcml_mask, 'PRICE_CHANGE'] = lp_data_df.loc[update_vcml_mask, 'VCML_PRICE_CHANGE']
    # To avoid later cross-check problems
    lp_data_df['UNC_CHECK_UB'] = lp_data_df['UNC_CHECK_UB_VCML'] 
    lp_data_df['UNC_CHECK_LB'] = lp_data_df['UNC_CHECK_LB_VCML'] 
    print("VCMLs unified")
    
    #-------------------------------------------------------
    # Remove brands from UNC optimization
    if p.BRAND_OPT:
        erase_mask = (lp_data_df['BG_FLAG'] == 'B')
        lp_data_df.loc[erase_mask, 'NEW_MAC_PRICE'] = lp_data_df.loc[erase_mask, 'CURRENT_MAC_PRICE']
        lp_data_df.loc[erase_mask, 'PRICE_CHANGE'] = 'Original price'
        
    
    if p.TRUECOST_CLIENT or p.UCL_CLIENT:
        erase_mask = ((lp_data_df['IN_NONCAP_OK_VCML']) & ~(lp_data_df['IS_MAC']))
        lp_data_df.loc[erase_mask, 'NEW_MAC_PRICE'] = lp_data_df.loc[erase_mask, 'CURRENT_MAC_PRICE']
        lp_data_df.loc[erase_mask, 'PRICE_CHANGE'] = 'Original price'
    
    #-------------------------------------------------------
    # Check that CVS-other pharmacy parity can be maintained
    lp_data_df['PHARMACY_PARITY_BOUND'] = 'Satisfied'
    if cvs_independent:
        # First, check if CVS raises would violate any other constraints
        # Strictly speaking we could override the UC_UNIT condition below, but whether we want to
        # do that is pretty pharmacy specific: probably a bad idea at WMT, for example, even if it's
        # fine at a non-cap. We incur some opportunity cost for client-side U&C in order to avoid
        # significant increases in complexity here.
        min_ub_retail = lp_data_df[(lp_data_df['MEASUREMENT'].str[0]!='M') 
                                   & (lp_data_df['CHAIN_GROUP']!='CVS')
                                  ].groupby(['GPI_NDC','BG_FLAG'], as_index=False)[['UNC_CHECK_UB', 'UC_UNIT']].min().rename(
                                        columns={'UNC_CHECK_UB': 'GLOBAL_UB_MIN', 'UC_UNIT': 'GLOBAL_UC_UNIT'})
        lp_data_df_temp = lp_data_df.merge(min_ub_retail, on=['GPI_NDC','BG_FLAG'], how='left')
        assert(np.abs(lp_data_df_temp['FULLAWP_ADJ'].sum() - lp_data_df['FULLAWP_ADJ'].sum())<0.01), "failed merge"
        lp_data_df = lp_data_df_temp
        if 'CHAIN_SUBGROUP' in lp_data_df and "CVSSP" in lp_data_df['CHAIN_SUBGROUP'].unique():
            erase_mask_cvssp = ((lp_data_df['CHAIN_SUBGROUP'] == 'CVSSP') 
                                & ((lp_data_df['NEW_MAC_PRICE'] > lp_data_df['GLOBAL_UB_MIN']) 
                                   | (lp_data_df['NEW_MAC_PRICE'] > lp_data_df['GLOBAL_UC_UNIT']))
                                & (lp_data_df['PRICE_CHANGE'] == 'Raised price'))
            erase_mask_cvs = ((lp_data_df['CHAIN_SUBGROUP'] == 'CVS') 
                              & ((lp_data_df['NEW_MAC_PRICE']/p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH > lp_data_df['GLOBAL_UB_MIN']) 
                                 | (lp_data_df['NEW_MAC_PRICE']/p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH > lp_data_df['GLOBAL_UC_UNIT']) )
                              & (lp_data_df['PRICE_CHANGE'] == 'Raised price'))
            erase_mask = erase_mask_cvssp | erase_mask_cvs
        else:
            erase_mask = ((lp_data_df['CHAIN_GROUP'] == 'CVS') 
                          & ((lp_data_df['NEW_MAC_PRICE'] > lp_data_df['GLOBAL_UB_MIN'])
                             | (lp_data_df['NEW_MAC_PRICE'] > lp_data_df['GLOBAL_UC_UNIT']))
                          & (lp_data_df['PRICE_CHANGE'] == 'Raised price'))
        # We ERASE these price changes (instead of waiting until later, as with other checks) 
        # because the next step will change U&C prices to align,
        # which could cause infeasibilities for other CHAIN_GROUPs.
        lp_data_df.loc[erase_mask, 'NEW_MAC_PRICE'] = lp_data_df.loc[erase_mask, 'CURRENT_MAC_PRICE']
        lp_data_df.loc[erase_mask, 'PRICE_CHANGE'] = 'Original price'
        
        # Check for places where CVS/CVSSP is raised and other pharmacies are lowered and they conflict.
        # Erase based on UNC value.
        lowered_non_cvs = lp_data_df[(lp_data_df['PRICE_CHANGE']=='Lowered price') & (lp_data_df['CHAIN_GROUP']!='CVS')].groupby(["GPI_NDC","BG_FLAG"], as_index=False).agg({'NEW_MAC_PRICE': min, 'TRUE_UC_VALUE': max}).rename(columns={'NEW_MAC_PRICE': 'NEW_MAC_PRICE_OTH', 'TRUE_UC_VALUE': 'TRUE_UC_VALUE_OTH'})
        lp_data_df_temp = lp_data_df.merge(lowered_non_cvs, on=["GPI_NDC","BG_FLAG"], how='left')
        assert(np.abs(lp_data_df_temp['FULLAWP_ADJ'].sum() - lp_data_df['FULLAWP_ADJ'].sum())<0.01), "failed merge"
        lp_data_df = lp_data_df_temp
        if 'CHAIN_SUBGROUP' in lp_data_df and "CVSSP" in lp_data_df['CHAIN_SUBGROUP'].unique():
            erase_mask = ((lp_data_df['CHAIN_SUBGROUP'] == 'CVSSP')
                          & (lp_data_df['PRICE_CHANGE'] != 'Original price')
                          & (lp_data_df['NEW_MAC_PRICE_OTH'].notna())
                          & (lp_data_df['NEW_MAC_PRICE_OTH'] < lp_data_df['NEW_MAC_PRICE'])
                          & (lp_data_df['TRUE_UC_VALUE_OTH'] >= lp_data_df['TRUE_UC_VALUE'])
                         )
            lp_data_df.loc[erase_mask, 'NEW_MAC_PRICE'] = lp_data_df.loc[erase_mask, 'CURRENT_MAC_PRICE']
            lp_data_df.loc[erase_mask, 'PRICE_CHANGE'] = 'Original price'

            erase_mask = ((lp_data_df['CHAIN_SUBGROUP'] == 'CVS')
                          & (lp_data_df['PRICE_CHANGE'] != 'Original price')
                          & (lp_data_df['NEW_MAC_PRICE_OTH'].notna())
                          & (lp_data_df['NEW_MAC_PRICE_OTH'] < lp_data_df['NEW_MAC_PRICE']/p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH)
                          & (lp_data_df['TRUE_UC_VALUE_OTH'] >= lp_data_df['TRUE_UC_VALUE'])
                         )
            lp_data_df.loc[erase_mask, 'NEW_MAC_PRICE'] = lp_data_df.loc[erase_mask, 'CURRENT_MAC_PRICE']
            lp_data_df.loc[erase_mask, 'PRICE_CHANGE'] = 'Original price'
        else:
            erase_mask = ((lp_data_df['CHAIN_GROUP'] == 'CVS')
                          & (lp_data_df['PRICE_CHANGE'] != 'Original price')
                          & (lp_data_df['NEW_MAC_PRICE_OTH'].notna())
                          & (lp_data_df['NEW_MAC_PRICE_OTH'] < lp_data_df['NEW_MAC_PRICE'])
                          & (lp_data_df['TRUE_UC_VALUE_OTH'] >= lp_data_df['TRUE_UC_VALUE'])
                         )
            lp_data_df.loc[erase_mask, 'NEW_MAC_PRICE'] = lp_data_df.loc[erase_mask, 'CURRENT_MAC_PRICE']
            lp_data_df.loc[erase_mask, 'PRICE_CHANGE'] = 'Original price'
            
        # Check two things:
        # - drops at non-CVS pharmacies to check they don't violate CVS parity constraints
        # - raises at CVS parity pharmacies to check that U&C upper bounds don't cause infeasibilities
        if 'CHAIN_SUBGROUP' in lp_data_df and "CVSSP" in lp_data_df['CHAIN_SUBGROUP'].unique():
            cvssp_prices = lp_data_df[lp_data_df['CHAIN_SUBGROUP'].str.contains("CVSSP")].groupby(['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'GPI_NDC','BG_FLAG']).agg({'NEW_MAC_PRICE': max, 'PRICE_CHANGE': max, 'UNC_CHECK_LB': max, 'UNC_CHECK_UB': min, 'TRUE_UC_VALUE': sum}).rename(
                columns={'NEW_MAC_PRICE': 'NEW_MAC_PRICE_CVSSP', 'PRICE_CHANGE': 'PRICE_CHANGE_CVSSP', 'UNC_CHECK_LB': 'UNC_CHECK_LB_CVSSP', 'UNC_CHECK_UB': 'UNC_CHECK_UB_CVSSP', 'TRUE_UC_VALUE': 'TRUE_UC_VALUE_CVSSP'}).drop_duplicates()
            cvs_prices = lp_data_df[lp_data_df['CHAIN_SUBGROUP'] == 'CVS'][['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'GPI_NDC', 'BG_FLAG', 'NEW_MAC_PRICE', 'PRICE_CHANGE', 'UNC_CHECK_LB', 'UNC_CHECK_UB','TRUE_UC_VALUE']].rename(
                columns={'NEW_MAC_PRICE': 'NEW_MAC_PRICE_CVS', 'PRICE_CHANGE': 'PRICE_CHANGE_CVS', 'UNC_CHECK_LB': 'UNC_CHECK_LB_CVS', 'UNC_CHECK_UB': 'UNC_CHECK_UB_CVS', 'TRUE_UC_VALUE': 'TRUE_UC_VALUE_CVS'}).drop_duplicates()
            lp_data_df_temp = lp_data_df.merge(cvs_prices, on=['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'GPI_NDC', 'BG_FLAG'], how = 'left')
            lp_data_df_temp = lp_data_df_temp.merge(cvssp_prices, on=['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'GPI_NDC', 'BG_FLAG'], how = 'left')
            assert(np.abs(lp_data_df_temp['FULLAWP_ADJ'].sum() - lp_data_df['FULLAWP_ADJ'].sum())<0.01), "Failed merge of CVS/CVSSP prices in U&C module"
            lp_data_df = lp_data_df_temp
            
            # We don't need to check if the CVSSP price is within price bounds--
            # if it wasn't, we erased it above!
            mask = ((lp_data_df['CHAIN_GROUP'] != 'CVS')
                    & (lp_data_df['PRICE_CHANGE'] == 'Raised price')
                    & (lp_data_df['PRICE_CHANGE_CVSSP'] == 'Raised price')
                    & (lp_data_df['NEW_MAC_PRICE'] <= lp_data_df['NEW_MAC_PRICE_CVSSP'])
                   )
            lp_data_df.loc[mask, 'NEW_MAC_PRICE'] = lp_data_df.loc[mask, 'NEW_MAC_PRICE_CVSSP']

            # Check raises at non-CVS pharmacies so that if both CVS & other pharmacies are raised, they don't get
            # their prices fixed at incompatible points
            mask = ((lp_data_df['CHAIN_GROUP'] != 'CVS')
                    & (lp_data_df['PRICE_CHANGE'] == 'Raised price')
                    & (lp_data_df['PRICE_CHANGE_CVS'] == 'Raised price')
                    # The lowest a CVSSP price could go
                    & (lp_data_df['NEW_MAC_PRICE'] < lp_data_df['NEW_MAC_PRICE_CVS']/p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH)
                   )
            lp_data_df.loc[mask, 'NEW_MAC_PRICE'] = lp_data_df.loc[mask, 'NEW_MAC_PRICE_CVS']/p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH

            # Check lowered prices & erase any where the CVS price is more valuable
            mask = ((lp_data_df['PRICE_CHANGE'] == 'Lowered price')
                    & (lp_data_df['PRICE_CHANGE_CVSSP'] == 'Raised price')
                    & (lp_data_df['NEW_MAC_PRICE'] < lp_data_df['NEW_MAC_PRICE_CVSSP'])
                    & (lp_data_df['TRUE_UC_VALUE'] < lp_data_df['TRUE_UC_VALUE_CVSSP'])
                   )
            lp_data_df.loc[mask, 'PHARMACY_PARITY_BOUND'] = 'Broken - price lowered, CVSSP raised'
            mask = ((lp_data_df['PRICE_CHANGE'] == 'Lowered price')
                    & (lp_data_df['PRICE_CHANGE_CVS'] == 'Raised price')
                    & (lp_data_df['NEW_MAC_PRICE'] < lp_data_df['NEW_MAC_PRICE_CVS']/p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH)
                    & (lp_data_df['TRUE_UC_VALUE'] < lp_data_df['TRUE_UC_VALUE_CVS'])
                   )
            lp_data_df.loc[mask, 'PHARMACY_PARITY_BOUND'] = 'Broken - price lowered, CVS raised'

            # check for drops that drop too far
            mask = ((lp_data_df['CHAIN_GROUP'] != 'CVS')
                    & ((lp_data_df['NEW_MAC_PRICE']<lp_data_df['UNC_CHECK_LB_CVSSP'])
                       | (lp_data_df['NEW_MAC_PRICE']<lp_data_df['UNC_CHECK_LB_CVS']/p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH))
                    & (lp_data_df['PRICE_CHANGE'] == 'Lowered price')
                   )
            lp_data_df.loc[mask, 'PHARMACY_PARITY_BOUND'] = 'Broken - new price too low for CVS lower bounds'
            
            # Check for problematic default U&C upper limits
            mask = ((lp_data_df['CHAIN_GROUP'] != 'CVS')
                    & (lp_data_df['PRICE_CHANGE'] == 'Original price')
                    & (lp_data_df['PRICE_CHANGE_CVSSP'] == 'Raised price')
                    & (lp_data_df['NEW_MAC_PRICE_CVSSP'] > lp_data_df['UC_UNIT'])
                   )
            lp_data_df.loc[mask, 'NEW_MAC_PRICE'] = lp_data_df.loc[mask, 'NEW_MAC_PRICE_CVSSP']
            lp_data_df.loc[mask, 'PRICE_CHANGE'] = 'Raised price'
            lp_data_df.loc[mask, 'MATCH_VCML'] = True
            lp_data_df.loc[mask, 'VCML_BOUND'] = 'Satisfied' # We may still have to do this even if price changes were erased above

            mask = ((lp_data_df['CHAIN_GROUP'] != 'CVS')
                    & (lp_data_df['PRICE_CHANGE'] == 'Original price')
                    & (lp_data_df['PRICE_CHANGE_CVS'] == 'Raised price')
                    & (lp_data_df['NEW_MAC_PRICE_CVS']/p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH > lp_data_df['UC_UNIT'])
                   )
            lp_data_df.loc[mask, 'NEW_MAC_PRICE'] = lp_data_df.loc[mask, 'NEW_MAC_PRICE_CVS']/p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH
            lp_data_df.loc[mask, 'PRICE_CHANGE'] = 'Raised price'
            lp_data_df.loc[mask, 'MATCH_VCML'] = True
            lp_data_df.loc[mask, 'VCML_BOUND'] = 'Satisfied' # We may still have to do this even if price changes were erased above
        else:
            # Can have duplicates in the case of R90OK. So we're going to check:
            # - we care when CVS is raised, but not when it's dropped, so max() the price_change
            # - we care when we need to raise a price to meet the CVS price, so we want the max CVS price
            # - we care when something has dropped too far for CVS, so we also need the max of the lower bound
            cvs_prices = lp_data_df[lp_data_df['CHAIN_GROUP'] == 'CVS'].groupby(['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'GPI_NDC','BG_FLAG'], as_index=False).agg({'NEW_MAC_PRICE': max, 'PRICE_CHANGE': max, 'UNC_CHECK_LB': max, 'TRUE_UC_VALUE': max}).rename(
                columns={'NEW_MAC_PRICE': 'NEW_MAC_PRICE_CVS', 'PRICE_CHANGE': 'PRICE_CHANGE_CVS', 'UNC_CHECK_LB': 'UNC_CHECK_LB_CVS', 'TRUE_UC_VALUE': 'TRUE_UC_VALUE_CVS'})
            lp_data_df = lp_data_df.merge(cvs_prices, on=['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'GPI_NDC', 'BG_FLAG'], how = 'left')

                        # We don't need to check if the CVSSP price is within price bounds--
            # if it wasn't, we erased it above!
            mask = ((lp_data_df['CHAIN_GROUP'] != 'CVS')
                    & (lp_data_df['PRICE_CHANGE'] == 'Raised price')
                    & (lp_data_df['PRICE_CHANGE_CVS'] == 'Raised price')
                    & (lp_data_df['NEW_MAC_PRICE'] <= lp_data_df['NEW_MAC_PRICE_CVS'])
                   )
            lp_data_df.loc[mask, 'NEW_MAC_PRICE'] = lp_data_df.loc[mask, 'NEW_MAC_PRICE_CVS']

            # Check lowered prices & erase any where the CVS price is more valuable
            mask = ((lp_data_df['PRICE_CHANGE'] == 'Lowered price')
                    & (lp_data_df['PRICE_CHANGE_CVS'] == 'Raised price')
                    & (lp_data_df['NEW_MAC_PRICE'] < lp_data_df['NEW_MAC_PRICE_CVS'])
                    & (lp_data_df['TRUE_UC_VALUE'] < lp_data_df['TRUE_UC_VALUE_CVS'])
                   )
            lp_data_df.loc[mask, 'PHARMACY_PARITY_BOUND'] = 'Broken - price lowered, CVS raised'

            # check for drops that drop too far
            mask = ((lp_data_df['CHAIN_GROUP'] != 'CVS')
                    & (lp_data_df['NEW_MAC_PRICE']<lp_data_df['UNC_CHECK_LB_CVS'])
                    & (lp_data_df['PRICE_CHANGE'] == 'Lowered price')
                   )
            lp_data_df.loc[mask, 'PHARMACY_PARITY_BOUND'] = 'Broken - new price too low for CVS lower bounds'

            # Check for problematic default U&C upper limits
            mask = ((lp_data_df['CHAIN_GROUP'] != 'CVS')
                    & (lp_data_df['PRICE_CHANGE'] == 'Original price')
                    & (lp_data_df['PRICE_CHANGE_CVS'] == 'Raised price')
                    & (lp_data_df['NEW_MAC_PRICE_CVS'] > lp_data_df['UC_UNIT'])
                   )
            lp_data_df.loc[mask, 'NEW_MAC_PRICE'] = lp_data_df.loc[mask, 'NEW_MAC_PRICE_CVS']
            lp_data_df.loc[mask, 'PRICE_CHANGE'] = 'Raised price'
            lp_data_df.loc[mask, 'MATCH_VCML'] = True
            lp_data_df.loc[mask, 'VCML_BOUND'] = 'Satisfied' # We may still have to do this even if price changes were erased above
    print("Parity constraints satisfied")
        
    #--------------------------
    # Check R30-R90 constraints
    lp_data_df['R30R90_BOUND'] = 'Satisfied'
    if 'R30' in lp_data_df['MEASUREMENT'].unique() and 'R90' in lp_data_df['MEASUREMENT'].unique():
        # We don't group on breakout because non-offsetting will have different breakouts!
        r90_prices = lp_data_df[lp_data_df['MEASUREMENT'] == 'R90'][[
            'CLIENT', 
            'REGION', 
            'CHAIN_GROUP', 
            'CHAIN_SUBGROUP', 
            'GPI_NDC',
            'BG_FLAG',
            'NEW_MAC_PRICE', 
            'PRICE_CHANGE', 
            'UNC_CHECK_LB']].rename(
            columns={'NEW_MAC_PRICE': 'NEW_MAC_PRICE_R90', 
                     'PRICE_CHANGE': 'PRICE_CHANGE_R90', 
                     'UNC_CHECK_LB': 'UNC_CHECK_LB_R90'}).drop_duplicates()
        r30_prices = lp_data_df[lp_data_df['MEASUREMENT'] == 'R30'][[
            'CLIENT', 
            'REGION', 
            'CHAIN_GROUP', 
            'CHAIN_SUBGROUP', 
            'GPI_NDC',
            'BG_FLAG',
            'NEW_MAC_PRICE', 
            'PRICE_CHANGE', 
            'UNC_CHECK_UB',
            'UC_UNIT']].rename(
            columns={'NEW_MAC_PRICE': 'NEW_MAC_PRICE_R30', 
                     'PRICE_CHANGE': 'PRICE_CHANGE_R30', 
                     'UNC_CHECK_UB': 'UNC_CHECK_UB_R30',
                     'UC_UNIT': 'UC_UNIT_R30'}).drop_duplicates()
        lp_data_df_temp = lp_data_df.merge(r30_prices, on=['CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GPI_NDC', 'BG_FLAG'], how = 'left')
        lp_data_df_temp = lp_data_df_temp.merge(r90_prices, on=['CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GPI_NDC', 'BG_FLAG'], how = 'left')
        assert np.abs(lp_data_df['FULLAWP_ADJ'].sum() - lp_data_df_temp['FULLAWP_ADJ'].sum())<0.01, "Failed join in R30-R90 price checks in U&C module"
        lp_data_df = lp_data_df_temp
        # raised R30 and R90, check that R30>=R90, raise R30 if needed & allowed...
        mask = (((lp_data_df['MEASUREMENT']=='R30') & (lp_data_df['PRICE_CHANGE'] == 'Raised price'))
                & (lp_data_df['PRICE_CHANGE_R90'] == 'Raised price')
                & (lp_data_df['NEW_MAC_PRICE'] < lp_data_df['NEW_MAC_PRICE_R90'])
                & (lp_data_df['NEW_MAC_PRICE_R90'] <= lp_data_df['UNC_CHECK_UB'])
               )
        lp_data_df.loc[mask, 'NEW_MAC_PRICE'] = lp_data_df.loc[mask, 'NEW_MAC_PRICE_R90']
        # ...otherwise undo R90 change
        mask = (((lp_data_df['MEASUREMENT']=='R90') & (lp_data_df['PRICE_CHANGE'] == 'Raised price'))
                & (lp_data_df['PRICE_CHANGE_R30'] == 'Raised price')
                & (lp_data_df['NEW_MAC_PRICE'] > lp_data_df['NEW_MAC_PRICE_R30'])
                & (lp_data_df['NEW_MAC_PRICE'] > lp_data_df['UNC_CHECK_UB_R30'])
               )
        lp_data_df.loc[mask, 'R30R90_BOUND'] = 'Broken - R90 raise above R30 upper limit'

        # changed both R30 and R90, check that R30>=R90, if not undo R90 change
        mask = (((lp_data_df['MEASUREMENT']=='R90') & (lp_data_df['PRICE_CHANGE'] != 'Original price'))
                & (lp_data_df['PRICE_CHANGE_R30'] != 'Original price')
                & (lp_data_df['NEW_MAC_PRICE'] > lp_data_df['NEW_MAC_PRICE_R30'])
               )
        lp_data_df.loc[mask, 'R30R90_BOUND'] = 'Broken - R90 raise, R30 drop'

        # lowered R30, check above R90 floor
        mask = (((lp_data_df['MEASUREMENT']=='R30') & (lp_data_df['PRICE_CHANGE'] != 'Original price'))
                & (lp_data_df['PRICE_CHANGE_R90'] == 'Original price')
                & (lp_data_df['NEW_MAC_PRICE'] < lp_data_df['UNC_CHECK_LB_R90'])
               )
        lp_data_df.loc[mask, 'R30R90_BOUND'] = 'Broken - R30 drop below R90 floor'

        # raised R90, check below R30 ceiling
        mask = (((lp_data_df['MEASUREMENT']=='R90') & (lp_data_df['PRICE_CHANGE'] != 'Original price'))
                & (lp_data_df['PRICE_CHANGE_R30'] == 'Original price')
                & ((lp_data_df['NEW_MAC_PRICE'] > lp_data_df['UNC_CHECK_UB_R30'])
                   # UC Unit check -- if we'd wanted to raise it above the U&C median, we already would have!
                   | (lp_data_df['NEW_MAC_PRICE'] > lp_data_df['UC_UNIT_R30']))
               )
        lp_data_df.loc[mask, 'R30R90_BOUND'] = 'Broken - R90 raise above R30 ceiling'

    lp_data_df['PREF_BOUND'] = 'Satisfied'
    # next line changes df with rows of ['CLIENT', 'REGION', ['PHARM1', 'PHARM2']] to
    # ['CLIENT', 'REGION', 'PHARM1'],
    # ['CLIENT', 'REGION', 'PHARM2']
    if pref_pharm_list is None:
        pref_pharm_df['PREFERRED'] = 'NONPREF'
    else:
        pref_pharm_df = pref_pharm_list.explode('PREF_PHARMS', ignore_index=True).rename(columns={"PREF_PHARMS": 'CHAIN_GROUP'})
        pref_pharm_df['PREFERRED'] = 'PREF'
        lp_data_df = lp_data_df.merge(pref_pharm_df[['CLIENT', 'BREAKOUT', 'REGION', 'CHAIN_GROUP', 'PREFERRED']], 
                                      on=['CLIENT', 'BREAKOUT', 'REGION', 'CHAIN_GROUP'], 
                                      how='left')
        lp_data_df['PREFERRED'] = lp_data_df['PREFERRED'].fillna("NONPREF")
    # then we merge on, anything in the pref_pharm_list gets 'PREF' in the 'PREFERRED' column, anything else is filled with "NONPREF"
    if (lp_data_df['PREFERRED']=='PREF').any():
        pref_lim = lp_data_df[lp_data_df['PREFERRED'] == 'PREF'].groupby([
            'CLIENT', 
            'BREAKOUT', 
            'REGION', 
            'MEASUREMENT',
            'BG_FLAG',
            'GPI_NDC'], as_index=False).agg({'UNC_CHECK_LB': max}).rename(
            columns={'UNC_CHECK_LB': 'UNC_CHECK_LB_PREF'})
        pref_prices = lp_data_df[(lp_data_df['PREFERRED'] == 'PREF') & (lp_data_df['PRICE_CHANGE'] != 'Original price')].groupby([
            'CLIENT', 
            'BREAKOUT', 
            'REGION', 
            'MEASUREMENT',
            'BG_FLAG',
            'GPI_NDC'], as_index=False).agg({'NEW_MAC_PRICE': max}).rename(
            columns={'NEW_MAC_PRICE': 'NEW_MAC_PRICE_PREF'})
        nonpref_lim = lp_data_df[lp_data_df['PREFERRED'] == 'NONPREF'].groupby([
            'CLIENT', 
            'BREAKOUT', 
            'REGION', 
            'MEASUREMENT',
            'BG_FLAG',
            'GPI_NDC'], as_index=False).agg({'UNC_CHECK_UB': min}).rename(
            columns={'UNC_CHECK_UB': 'UNC_CHECK_UB_NONPREF'})
        lp_data_df = lp_data_df.merge(pref_lim, on=['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'GPI_NDC', 'BG_FLAG'], how = 'left')
        lp_data_df = lp_data_df.merge(pref_prices, on=['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'GPI_NDC', 'BG_FLAG'], how = 'left')
        lp_data_df = lp_data_df.merge(nonpref_lim, on=['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'GPI_NDC', 'BG_FLAG'], how = 'left')

        # Check PREF below NONPREF ceiling
        mask = (((lp_data_df['PREFERRED']=='PREF') & (lp_data_df['PRICE_CHANGE'] != 'Original price'))
                & (lp_data_df['NEW_MAC_PRICE'] <= lp_data_df['UNC_CHECK_UB_NONPREF'])
               )
        lp_data_df.loc[mask, 'PREF_BOUND'] = 'Broken - PREF price above NONPREF ceiling'

        # Check NONPREF above PREF floor & new PREF prices
        mask = (((lp_data_df['MEASUREMENT']=='NONPREF') & (lp_data_df['PRICE_CHANGE'] != 'Original price'))
                & (lp_data_df['NEW_MAC_PRICE'] < lp_data_df['UNC_CHECK_LB_PREF'])
               )
        lp_data_df.loc[mask, 'PREF_BOUND'] = 'Broken - R90 raise above PREF ceiling'
        mask = (((lp_data_df['MEASUREMENT']=='NONPREF') & (lp_data_df['PRICE_CHANGE'] != 'Original price'))
                & (lp_data_df['NEW_MAC_PRICE'] < lp_data_df['NEW_MAC_PRICE_PREF'])
                & ~lp_data_df['NEW_MAC_PRICE_PREF'].isna()
               )
        lp_data_df.loc[mask, 'PREF_BOUND'] = 'Broken - NONPREF U&C below PREF U&C'

            
    #------------------------------------------------------------------
    #check if price change for CVS or CVSSP causes parity collar issues
    lp_data_df['PARITY_COLLAR_BOUND'] = 'Satisfied'
    if ('CHAIN_SUBGROUP' in lp_data_df 
        and "CVSSP" in lp_data_df['CHAIN_SUBGROUP'].unique() 
        and "CVS" in lp_data_df['CHAIN_SUBGROUP'].unique()):
        
        if not cvs_independent: # if we went through the cvs_ind case above, we already have these columns
            cvs_prices = lp_data_df[lp_data_df['CHAIN_SUBGROUP'] == 'CVS'][['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'GPI_NDC', 'BG_FLAG', 'NEW_MAC_PRICE', 'PRICE_CHANGE', 'UNC_CHECK_UB', 'UNC_CHECK_LB']].rename(
                    columns={'NEW_MAC_PRICE': 'NEW_MAC_PRICE_CVS', 
                             'PRICE_CHANGE': 'PRICE_CHANGE_CVS',
                             'UNC_CHECK_UB': 'UNC_CHECK_UB_CVS',
                             'UNC_CHECK_LB': 'UNC_CHECK_LB_CVS'
                            }).drop_duplicates()
            cvssp_prices = lp_data_df[lp_data_df['CHAIN_SUBGROUP'] == 'CVS'][['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'GPI_NDC', 'BG_FLAG', 'NEW_MAC_PRICE', 'PRICE_CHANGE', 'UNC_CHECK_UB', 'UNC_CHECK_LB']].rename(
                    columns={'NEW_MAC_PRICE': 'NEW_MAC_PRICE_CVSSP', 
                             'PRICE_CHANGE': 'PRICE_CHANGE_CVSSP',
                             'UNC_CHECK_UB': 'UNC_CHECK_UB_CVSSP',
                             'UNC_CHECK_LB': 'UNC_CHECK_LB_CVSSP'
                            }).drop_duplicates()
            match_prices = cvs_prices.merge(cvssp_prices, on=['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'GPI_NDC','BG_FLAG'], how='outer')

            lp_data_df_temp = lp_data_df.merge(match_prices, on=['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'GPI_NDC','BG_FLAG'], how = 'left')
            assert(np.abs(lp_data_df_temp['FULLAWP_ADJ'].sum() - lp_data_df['FULLAWP_ADJ'].sum())<0.01), "failed merge"
            lp_data_df = lp_data_df_temp
        
        # Check: both changed, are resulting prices allowed?
        # This checks both--which is good--need to cancel both if in conflict.
        mask_cvs = ((lp_data_df['CHAIN_SUBGROUP'] == 'CVS')
                & ((lp_data_df['PRICE_CHANGE'] != 'Original price')) 
                & ((lp_data_df['PRICE_CHANGE_CVSSP'] == 'Raised price') | (lp_data_df['PRICE_CHANGE_CVSSP'] == 'Lowered price')) 
               )
        mask_cvssp = ((lp_data_df['CHAIN_SUBGROUP'] == 'CVSSP')
                & ((lp_data_df['PRICE_CHANGE'] != 'Original price')) 
                & ((lp_data_df['PRICE_CHANGE_CVS'] == 'Raised price') | (lp_data_df['PRICE_CHANGE_CVS'] == 'Lowered price')) 
               )
        lp_data_df.loc[mask_cvs 
                       & (
                           (lp_data_df['NEW_MAC_PRICE'] > p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH*lp_data_df['NEW_MAC_PRICE_CVSSP']) 
                           | (lp_data_df['NEW_MAC_PRICE'] < p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW*lp_data_df['NEW_MAC_PRICE_CVSSP']) 
                       ), 'PARITY_COLLAR_BOUND'] = 'Broken - both changed'
        lp_data_df.loc[mask_cvssp
                       & (
                           (lp_data_df['NEW_MAC_PRICE_CVS'] > p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH*lp_data_df['NEW_MAC_PRICE']) 
                           | (lp_data_df['NEW_MAC_PRICE_CVS'] < p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW*lp_data_df['NEW_MAC_PRICE']) 
                       ), 'PARITY_COLLAR_BOUND'] = 'Broken - both changed'
        
        # Check: CVS changed, are resulting prices allowed?
        mask_cvs = ((lp_data_df['CHAIN_SUBGROUP'] == 'CVS')
                & ((lp_data_df['PRICE_CHANGE'] != 'Original price')) 
                & ((lp_data_df['PRICE_CHANGE_CVSSP'] == 'Original price')) 
               )
        lp_data_df.loc[mask_cvs 
                       & (
                           (lp_data_df['NEW_MAC_PRICE'] > p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH*lp_data_df['UNC_CHECK_UB_CVSSP']) 
                           | (lp_data_df['NEW_MAC_PRICE'] < p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW*lp_data_df['UNC_CHECK_LB_CVSSP']) 
                       ), 'PARITY_COLLAR_BOUND'] = 'Broken - CVS changed'
        # Check: CVSSP changed, are resulting prices allowed?
        mask_cvssp = ((lp_data_df['CHAIN_SUBGROUP'] == 'CVSSP')
                & ((lp_data_df['PRICE_CHANGE'] != 'Original price')) 
                & ((lp_data_df['PRICE_CHANGE_CVS'] == 'Original price')) 
               )

        lp_data_df.loc[mask_cvssp
                       & (
                           (lp_data_df['UNC_CHECK_LB_CVS'] > p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH*lp_data_df['NEW_MAC_PRICE']) 
                           | (lp_data_df['UNC_CHECK_UB_CVS'] < p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW*lp_data_df['NEW_MAC_PRICE']) 
                       ), 'PARITY_COLLAR_BOUND'] = 'Broken - CVS changed'
    
    # Now, see which of these proposed U&C-related price changes we actually want to keep!
    lp_data_df['KEEP_PRICE_CHANGE'] = ((lp_data_df['PRICE_CHANGE'] != 'Original price')
                                    & (~lp_data_df['R30R90_BOUND'].str.contains("Broken"))
                                    & (~lp_data_df['PREF_BOUND'].str.contains("Broken"))
                                    & (~lp_data_df['PARITY_COLLAR_BOUND'].str.contains("Broken"))
                                    & (~lp_data_df['PHARMACY_PARITY_BOUND'].str.contains("Broken"))
                                    & (~lp_data_df['VCML_BOUND'].str.contains("Broken"))
                                    & (lp_data_df['PRICE_MUTABLE'] == 1)
                                    & (~lp_data_df['CHAIN_GROUP'].isin(['MAIL', 'MCHOICE']))
                                    & (lp_data_df['GPI_ONLY'] == 1)
                                    & (p.UNC_CLIENT | lp_data_df['CHAIN_GROUP'].isin(p.UNC_PHARMACY_CHAIN_GROUPS))
                                    )
    if not p.UNC_2X_OPTIMIZATION:
        lp_data_df.loc[lp_data_df['IS_TWOSTEP_UNC'], 'KEEP_PRICE_CHANGE'] = False
    # We changed prices above to keep VCMLs consistent. If any of those prices broke constraints, we eliminate that price change
    # for all prices on the same VCML.
    discrepant_vcmls = lp_data_df.groupby(['CLIENT','BREAKOUT','REGION','MEASUREMENT','MAC_LIST', 'GPI', 'NDC', 'BG_FLAG'], as_index=False)['KEEP_PRICE_CHANGE'].nunique()
    discrepant_vcmls = discrepant_vcmls[discrepant_vcmls['KEEP_PRICE_CHANGE']>1].rename(columns={'KEEP_PRICE_CHANGE': 'ELIMINATE_PRICE_CHANGE_VCML'})
    lp_data_df_temp = lp_data_df.merge(discrepant_vcmls, on=['CLIENT','BREAKOUT','REGION','MEASUREMENT','MAC_LIST', 'GPI', 'NDC', 'BG_FLAG'], how='left')
    assert np.abs(lp_data_df.FULLAWP_ADJ.sum() - lp_data_df_temp.FULLAWP_ADJ.sum())<1.E-4, 'Error in discrepant VCML join'
    lp_data_df = lp_data_df_temp
    lp_data_df.loc[lp_data_df['ELIMINATE_PRICE_CHANGE_VCML']>1, 'KEEP_PRICE_CHANGE'] = False

    # Unset this key for prices we rejected
    lp_data_df.loc[~lp_data_df['KEEP_PRICE_CHANGE'], 'IS_TWOSTEP_UNC'] = False
    lp_data_df.loc[~lp_data_df['KEEP_PRICE_CHANGE'], 'IS_MAINTENANCE_UC'] = False
    # If price is lowered and kept, set the new lowered price as a "price ceiling" to be imposed in LP    
    lp_data_df['MAC_PRICE_UPPER_LIMIT_UC'] = np.where((lp_data_df['KEEP_PRICE_CHANGE']) & (lp_data_df['PRICE_CHANGE'] == 'Lowered price'), 
                                                  lp_data_df['NEW_MAC_PRICE'], 1000 * lp_data_df['CURRENT_MAC_PRICE'])
    
    # Add column to track the old (pre-UC optimization) price 
    lp_data_df['PRE_UC_MAC_PRICE'] = lp_data_df['CURRENT_MAC_PRICE']
    
    # If price is raised and kept, change the current MAC price to be the new price
    lp_data_df['CURRENT_MAC_PRICE'] = np.where((lp_data_df['KEEP_PRICE_CHANGE']) & (lp_data_df['PRICE_CHANGE'] == 'Raised price'),
                                              lp_data_df['NEW_MAC_PRICE'], lp_data_df['CURRENT_MAC_PRICE'])
    
    # Make flag to keep track of raised prices - these will be set to immutable in the MacOptimization script because they are now U&C
    lp_data_df['RAISED_PRICE_UC'] = (lp_data_df['PRICE_CHANGE'] == 'Raised price') & (lp_data_df['KEEP_PRICE_CHANGE'])
    #lp_data_df.loc[(lp_data_df['PRICE_CHANGE'] == 'Raised price') & (lp_data_df['KEEP_PRICE_CHANGE']), 'PRICE_MUTABLE'] = 0

    # KEEP_PRICE_CHANGE will serve as a flag to track changed prices from UC optimization process
    lp_data_df = lp_data_df.rename(columns = {'KEEP_PRICE_CHANGE': 'PRICE_CHANGED_UC'})

    # Compute the fraction of claims that will be U&C because of these price changes
    # Note annoying sign sense: quantity_lt is where the *U&C* quantity is less, so the fraction of things
    # adjudicating as U&C is qty_lt or 1-qty_gt (depending on whether we are just above or just below the price).
    final_raised_mask = lp_data_df['PRICE_CHANGED_UC'] & lp_data_df['RAISED_PRICE_UC']
    lp_data_df.loc[final_raised_mask, 'UNC_FRAC'] = lp_data_df.loc[final_raised_mask, 'QUANTITY_GT_UNC']/lp_data_df.loc[final_raised_mask, 'QTY_IN_CONSTRAINTS']
    final_dropped_mask = lp_data_df['PRICE_CHANGED_UC'] & ~lp_data_df['RAISED_PRICE_UC']
    lp_data_df.loc[final_dropped_mask, 'UNC_FRAC'] = 1-lp_data_df.loc[final_dropped_mask, 'QUANTITY_LT_UNC']/lp_data_df.loc[final_dropped_mask, 'QTY_IN_CONSTRAINTS']
    
    # Clean up output columns
    lp_data_df['UNC_VALUE'] = 0
    lp_data_df.loc[lp_data_df['PRICE_CHANGED_UC'] 
                   & ~lp_data_df['IS_TWOSTEP_UNC'] 
                   & ~lp_data_df['IS_MAINTENANCE_UC'], 
                   'UNC_VALUE'] = lp_data_df.loc[lp_data_df['PRICE_CHANGED_UC'] 
                                                 & ~lp_data_df['IS_TWOSTEP_UNC'] 
                                                 & ~lp_data_df['IS_MAINTENANCE_UC'], 'TRUE_UC_VALUE']
    lp_data_df['MATCH_VCML'] &= lp_data_df['PRICE_CHANGED_UC']

    # Adjust for time periods. U&C claim distributions, used in value computation, are measured for last 90 days from the day
    # the data was pulled.
    # This will usually be a little shorter than 90 days, so this following function will conservatively underestimate
    # value: we should be dividing by 90 - (today - last day of available claims), not 90.
    if contract_date_df is None:
        eoy_days = 90 # don't rescale value if eoy_days is None
    elif p.FULL_YEAR:
        timediff = pd.to_datetime(contract_date_df['CONTRACT_EXPRN_DT'])[0] - contract_date_df['CONTRACT_EFF_DT'][0] 
        eoy_days = timediff.days + 1
    else:
        eoy = dt.datetime.strptime(contract_date_df['CONTRACT_EXPRN_DT'][0], '%Y-%m-%d')
        eoy_days = (eoy - p.GO_LIVE).days + 1
        
    lp_data_df['UNC_VALUE'] *= eoy_days/90
    lp_data_df['UNC_VALUE_CLIENT'] *= eoy_days/90
    lp_data_df['UNC_VALUE_PHARM'] *= eoy_days/90

    # Drop extraneous columns  
    lp_data_df = lp_data_df.drop(columns = ['MAC1026_BOUND', 'PRICE_REIMB_UNIT', 'AWP_BOUND', 'TIER_BOUND', 
                                            'CLAIM_INCREASE_PCT', 'CLAIM_INCREASE', 'NEW_MAC_PRICE', 'PRICE_CHANGE', 
                                            'NEW_MAC_PRICE_CVS', 'PRICE_CHANGE_CVS', 'unc_low', 'unc_high', 
                                            'UNC_CHECK_UB', 'UNC_CHECK_LB', 'PARITY_COLLAR_BOUND', 'PHARMACY_PARITY_BOUND', 'NEW_MAC_PRICE_CVS', 'PRICE_CHANGE_CVS', 'UNC_CHECK_UB_CVS', 'UNC_CHECK_LB_CVS',
                                            'NEW_MAC_PRICE_CVSSP', 'PRICE_CHANGE_CVSSP', 'UNC_CHECK_UB_CVSSP','UNC_CHECK_LB_CVSSP',
                                            'NEW_MAC_PRICE_R90', 'PRICE_CHANGE_R90', 'UNC_CHECK_LB_R90',
                                            'NEW_MAC_PRICE_R30', 'PRICE_CHANGE_R30', 'UNC_CHECK_UB_R30'
                                            'PREFERRED', 'NEW_MAC_PRICE_PREF', 'UNC_CHECK_LB_PREF',
                                            'NEW_MAC_PRICE_NONPREF', 'UNC_CHECK_UB_NONPREF',
                                            'WTW_UPPER_LIMIT', 'WTW_UPPER_LIMIT_2X', 'UNC_CHECK_UB_2X', 'UNC_CHECK_LB_2X',
                                            'AVG_QTY', 'BREAK_EVEN_CLIENT_UNIT', 'BREAK_EVEN_PHARM_UNIT', 
                                            'GLOBAL_UB_MIN', 'IS_UC_NEW', 'IS_UC_OLD',
                                            'MOVE_CONCURRENCE', 'PREF_BOUND', 'PREFERRED', 'price_target',
                                            'QUANTITY_GT_PCT00', 'QUANTITY_GT_PCT100', 'QUANTITY_GT_UNC',
                                            'QUANTITY_LT_PCT00', 'QUANTITY_LT_PCT100', 'QUANTITY_LT_UNC',
                                            'R30R90_BOUND', 'RANDOM_OFFSET_BUFFER', 
                                            'UC_MAINTENANCE_LOWER_LIMIT', 'UC_MAINTENANCE_UPPER_LIMIT',
                                            'UC_VALUE_CLIENT_CURR_TEMP', 'UC_VALUE_CLIENT_NEW_TEMP', 
                                            'UC_VALUE_PHARM_CURR_TEMP', 'UC_VALUE_PHARM_NEW_TEMP',
                                            'TRUE_UC_VALUE', 'PRICE_REIMB_ADJ', 'PRICE_REIMB_CLAIM',
                                            'PHARM_PRICE_REIMB_ADJ', 'PHARM_PRICE_REIMB_CLAIM',
                                            'PRICING_PRICE_REIMB_CLAIM', 'NEW_PRICING_PRICE_REIMB_CLAIM',
                                            'PRICING_QTY', 'VCML_BOUND', 'PRICING_CLAIMS', 
                                            'QTY_LT_UNC', 'QTY_IN_CONSTRAINTS', 'QTY_GT_UNC',
                                            'UNC_CHECK_LB_VCML', 'UNC_CHECK_UB_VCML',
                                            'VCML_NEW_MAC_PRICE', 'VCML_PRICE_CHANGE',
                                            'VCML_IS_MAINTENANCE_UC', 'VCML_IS_TWOSTEP_UNC',
                                            'NEW_MAC_PRICE_OTH', 'TRUE_UC_VALUE_OTH',
                                            'BEG_Q_PRICE', 'BEG_M_PRICE', 'BEG_PERIOD_PRICE',
                                            'UNC_OVRD_AMT'
                                           ]
                                           + [c for c in lp_data_df.columns if 'TRUE_UC_VALUE_' in c]
                                           + [c for c in lp_data_df.columns if 'qty_uc_slice_' in c]
                                           + [c for c in lp_data_df.columns if 'PRICE_RAISE_' in c]
                                           + [c for c in lp_data_df.columns if 'DISTANCE_FROM_CURR' in c]
                                           + [c for c in lp_data_df.columns if 'EXCESS_MOVEMENT' in c]
                                           + [c for c in lp_data_df.columns if 'SELECT_UC_VALUE' in c]
                                           + [c for c in lp_data_df.columns if 'UC_VALUE_CLIENT_' in c]
                                           + [c for c in lp_data_df.columns if 'UC_VALUE_PHARM_' in c]
                                           + [c for c in lp_data_df.columns if 'price_target_' in c]
                                 , errors='ignore')
    print(lp_data_df['PRICE_CHANGED_UC'].sum(), "U&C changes found, of which", lp_data_df['RAISED_PRICE_UC'].sum(), "are raised prices,",
          lp_data_df['IS_TWOSTEP_UNC'].sum(), "are two-step UNC changes,", lp_data_df['IS_MAINTENANCE_UC'].sum(), "are maintenance UNC, and",
          lp_data_df['MATCH_VCML'].sum(), "are VCML-matching changes"
    )
    
    return lp_data_df

def check_and_create_folder(path):
    '''
    Checks if the computer path exist if not it creates the needed directory.
    
    Input: The computer path e.g. C:/..../Input
    '''
    
    import os
    from google.cloud import storage, exceptions
    
    if 'gs://' in path:  # cloud storage path case
        bucket, blob = path[5:].split('/', 1)
        # check bucket exists and is accessible
        client = storage.Client()
        try:
            b = client.get_bucket(bucket)
        except exceptions.NotFound as e:
            err_text = 'Could not find the bucket. Check if bucket exists and is accessible in cloud storage.'
            print(err_text)
            e.args += (err_text,)
            raise e
        return
    if not os.path.exists(path):
        os.mkdir(path)


def brand_surplus_dict_generator_ytd_lag_eoy(Brand_Surplus,Brand_data_date,performance_dict, perf_dict_col):
    '''
    Inputs:
    Brand_Surplus - the Brand Generic offset dataframe from teradata.
    performance_dict - any performance dictionary, this function only uses the 'ENTITY' as a template to get the rows
    Brand_data_date - end date for the YTD Brand performance data
    Brand_Generic_Flag - True/False for using the Brand Generic or not, if False will still create a dictionary with all zeros
    
    1. projects to the end of year
    2. maps to ENTITY in performance dictionary using breakout_df as a template - should be a performance dictionary as a df
    3. changes df to dictionary
    
    Returns
    -------
    Brand surplus EOY performance Dictionary.

    '''
       
    import datetime as dt
    import os
    import CPMO_parameters as p
    import BQ
    import util_funcs as uf
    
    # convert the template performance dict to a dataframe
    breakout_df = dict_to_df(performance_dict,perf_dict_col)
    breakout_df['PERFORMANCE'] = 0

    # If the Brand Generic Offsetting is False return the same dictionary where surplus = 0
    # Note that the flag is single while we may have multiple clients. The assumption is all clients will be in the same type in 
    # terms of BraGer Offsetting. That is the same throughout the code for goodrx,U&C Flag etc. 
    if (len(Brand_Surplus) == 0) | (Brand_Surplus['SURPLUS'].sum() == 0):
        #Convert Brand Surplus df to dictionary
        Brand_surplus_eoy_df = breakout_df.copy()
        Brand_surplus_eoy_df['PERFORMANCE']=0
        brand_surplus_dict = Brand_surplus_eoy_df.set_index('ENTITY').T.to_dict('records')[0]
        return brand_surplus_dict, brand_surplus_dict, brand_surplus_dict

    # Convert ytd surplus values into other time periods
    # The surplus values are divided by ytd # of days and multiply by 365 to make a simple projection of surplus for the year.
    contract_date_df = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.CONTRACT_DATE_FILE))
    soc = dt.datetime.strptime(contract_date_df['CONTRACT_EFF_DT'][0], '%Y-%m-%d')
    eoc = dt.datetime.strptime(contract_date_df['CONTRACT_EXPRN_DT'][0], '%Y-%m-%d')
    
    Brand_Surplus['SURPLUS_DAY'] = Brand_Surplus['SURPLUS'] / (Brand_data_date - dt.datetime(Brand_data_date.year, 1, 1)).days    
    Brand_Surplus['SURPLUS_YTD'] = Brand_Surplus['SURPLUS_DAY'] * (p.LAST_DATA - soc).days
    Brand_Surplus['SURPLUS_LAG'] = Brand_Surplus['SURPLUS_DAY'] * ((p.GO_LIVE - p.LAST_DATA).days - 1)

    Brand_Surplus['SURPLUS_EOY'] = Brand_Surplus['SURPLUS_DAY'] * ((eoc - p.GO_LIVE).days + 1)
    Brand_Surplus['FULL_YEAR'] = Brand_Surplus['SURPLUS_DAY'] * (eoc-soc).days
    # The decimal points not optimal but similar dataframes in the code (ie gen_launch_eoy_dict) also uses float64. 
    
    breakout_mapping = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.BREAKOUT_MAPPING_FILE, dtype = p.VARIABLE_TYPE_DIC)
    #because this is offsetting generic performance with brand surplus, we want to make sure the breakout matches with generic breakout(although for bg_offsetting client the breakout is the same between brand and generic)
    breakout_mapping = breakout_mapping.loc[breakout_mapping['BG_FLAG']=='G'][['CUSTOMER_ID','MEASUREMENT','BREAKOUT']].rename(columns = {"CUSTOMER_ID":"CLIENT"}).reset_index(drop=True)
    client_brand_surplus = pd.merge(Brand_Surplus,breakout_mapping, how = 'left', on = ['CLIENT','MEASUREMENT'])
    client_brand_surplus['ENTITY'] = client_brand_surplus['CLIENT'] + '_' + client_brand_surplus['BREAKOUT']
    
    if p.FULL_YEAR:
            brand_surplus_dict = client_brand_surplus = client_brand_surplus.groupby('ENTITY')[['FULL_YEAR']].sum().reset_index()
            brand_surplus_dict.rename({'FULL_YEAR':'PERFORMANCE'}, inplace=True)
            return brand_surplus_dict, brand_surplus_dict, brand_surplus_dict

    # roll up to entity
    client_brand_surplus = client_brand_surplus.groupby('ENTITY')[['SURPLUS_YTD','SURPLUS_LAG','SURPLUS_EOY']].sum().reset_index()

    # Create YTD dictionary 
    YTD_df = breakout_df.copy(deep=True)    
    YTD_df = YTD_df.merge(client_brand_surplus, on='ENTITY',how='left').fillna(0)
    YTD_df['PERFORMANCE'] = YTD_df['PERFORMANCE'] + YTD_df['SURPLUS_YTD']
    YTD_df = YTD_df[['ENTITY','PERFORMANCE']]
    brand_surplus_ytd_dict = YTD_df.set_index('ENTITY').T.to_dict('records')[0]
   
    # Create Lag dictionary 
    LAG_df = breakout_df.copy(deep=True)    
    LAG_df = LAG_df.merge(client_brand_surplus, on='ENTITY',how='left').fillna(0)
    LAG_df['PERFORMANCE'] = LAG_df['PERFORMANCE'] + LAG_df['SURPLUS_LAG']
    LAG_df = LAG_df[['ENTITY','PERFORMANCE']]
    brand_surplus_lag_dict = LAG_df.set_index('ENTITY').T.to_dict('records')[0]

    # Create EOY dictionary 
    EOY_df = breakout_df.copy(deep=True)    
    EOY_df = EOY_df.merge(client_brand_surplus, on='ENTITY',how='left').fillna(0)
    EOY_df['PERFORMANCE'] = EOY_df['PERFORMANCE'] + EOY_df['SURPLUS_EOY']
    EOY_df = EOY_df[['ENTITY','PERFORMANCE']]
    brand_surplus_eoy_dict = EOY_df.set_index('ENTITY').T.to_dict('records')[0]

    return brand_surplus_ytd_dict, brand_surplus_lag_dict, brand_surplus_eoy_dict

def specialty_surplus_dict_generator_ytd_lag_eoy(spclty_surplus, contract_date_df, performance_dict, perf_dict_col):
    '''
    Inputs:
    spclty_surplus - The specialty offset dataframe from preprocessing.
    performance_dict - Any performance dictionary. Used as a template for the output dictionary
    
    1. Divides the expected specialty performance between the YTD, LAG, and EOY time periods
    2. Maps to ENTITY in a performance df
    3. Changes the performance df to a dictionary
    
    Returns:
    specialty_surplus_ytd_dict
    specialty_surplus_lag_dict
    specialty_surplus_eoy_dict
    '''
    import datetime as dt
    import CPMO_parameters as p
    import BQ
    import util_funcs as uf
    
    if p.FULL_YEAR and p.SPECIALTY_OFFSET:
        assert False, "Specialty offsetting not currently set up for FULL_YEAR runs as a safety precaution. Need to be sure we have specialty data for the next contract."
    
    # Convert the template performance dict to a dataframe
    breakout_df = dict_to_df(performance_dict, perf_dict_col)
    breakout_df['PERFORMANCE'] = 0
    
    # If Specialty Offsetting is False or N/A return the same dictionary where surplus = 0
    if (p.SPECIALTY_OFFSET == False | len(spclty_surplus) == 0) | (spclty_surplus['SURPLUS_EOY_PROJ'].sum() == 0):
        #Convert Specialty Surplus df to dictionary
        specialty_surplus_eoy_df = breakout_df.copy()
        specialty_surplus_eoy_df['PERFORMANCE'] = 0
        specialty_surplus_dict = specialty_surplus_eoy_df.set_index('ENTITY').T.to_dict('records')[0]
        return specialty_surplus_dict, specialty_surplus_dict, specialty_surplus_dict

    # Split surplus value between time periods. Assumes linear accrual of specialty performance across the contract.
    # The surplus value is divided by the number of days in the contract, then multiplied by the number of days in each period.
    soc = contract_date_df['CONTRACT_EFF_DT'][0]
    eoc = dt.datetime.strptime(contract_date_df['CONTRACT_EXPRN_DT'][0], '%Y-%m-%d')
    
    spclty_surplus['SURPLUS_DAY'] = spclty_surplus['SURPLUS_EOY_PROJ'] / ((eoc-soc).days + 1)
    spclty_surplus['SURPLUS_YTD'] = spclty_surplus['SURPLUS_DAY'] * (p.LAST_DATA - soc).days
    spclty_surplus['SURPLUS_LAG'] = spclty_surplus['SURPLUS_DAY'] * (p.GO_LIVE - p.LAST_DATA).days
    spclty_surplus['SURPLUS_EOY'] = spclty_surplus['SURPLUS_DAY'] * ((eoc - p.GO_LIVE).days + 1)
    
    # Create the Entity column to merge with the correct row in the performance dictionary
    # Pacificsource only has full retail offsetting, but coding out other potential cases in 
    # anticipation of new complexities.        
    offset_list = ['Pure Vanilla','MedD/EGWP Vanilla','Offsetting R30/R90','MedD/EGWP Offsetting R30/R90/LTC','Offsetting Complex','MedD/EGWP Offsetting Complex']
    nonoffset_list = ['NonOffsetting R30/R90','MedD/EGWP NonOffsetting R30/R90/LTC','MedD/EGWP NonOffsetting Complex','NonOffsetting Complex']
    
    # An OFFSETTING value of 'RETAIL' implies specialty offsets with all Retail measurements. Our breakouts have to be detailed enough
    # to correctly assign the offset to the right breakout depending on the contract. Currently the below can offset specialty with either
    # all retail measurements, or just R30 or just R90.
    if spclty_surplus['OFFSETTING'][0] == 'RETAIL':
        assert p.GUARANTEE_CATEGORY in (offset_list), 'Specialty offset does not match guarantee offsetting.'
        spclty_surplus['ENTITY'] = spclty_surplus['CUSTOMER_ID'][0] + '_' + spclty_surplus['CUSTOMER_ID'][0] + '_RETAIL'
    
    elif sum(spclty_surplus['OFFSETTING'].isin(['R30', 'R90'])) > 0: 
        assert p.GUARANTEE_CATEGORY in (nonoffset_list), 'Specialty offset does not match guarantee offsetting.'
        spclty_surplus['ENTITY'] = spclty_surplus['CUSTOMER_ID'] + '_' + spclty_surplus['CUSTOMER_ID'] + '_' + spclty_surplus['OFFSETTING']

    else:
        assert False, 'Do not know how to offset specialty performance. Check the offsetting column in the SPECIALTY_SURPLUS_DATA file for unhandled values.'
    
    # Create YTD dictionary 
    YTD_df = breakout_df.copy(deep=True)    
    YTD_df = YTD_df.merge(spclty_surplus[['ENTITY','SURPLUS_YTD']], on = 'ENTITY', how = 'left').fillna(0)
    YTD_df['PERFORMANCE'] = YTD_df['PERFORMANCE'] + YTD_df['SURPLUS_YTD']
    YTD_df = YTD_df[['ENTITY','PERFORMANCE']]
    specialty_surplus_ytd_dict = YTD_df.set_index('ENTITY').T.to_dict('records')[0]
    
    # Create Lag dictionary 
    YTD_df = breakout_df.copy(deep=True)    
    YTD_df = YTD_df.merge(spclty_surplus[['ENTITY','SURPLUS_LAG']], on = 'ENTITY', how = 'left').fillna(0)
    YTD_df['PERFORMANCE'] = YTD_df['PERFORMANCE'] + YTD_df['SURPLUS_LAG']
    YTD_df = YTD_df[['ENTITY','PERFORMANCE']]
    specialty_surplus_lag_dict = YTD_df.set_index('ENTITY').T.to_dict('records')[0]
    
    # Create EOY dictionary 
    YTD_df = breakout_df.copy(deep=True)    
    YTD_df = YTD_df.merge(spclty_surplus[['ENTITY','SURPLUS_EOY']], on = 'ENTITY', how = 'left').fillna(0)
    YTD_df['PERFORMANCE'] = YTD_df['PERFORMANCE'] + YTD_df['SURPLUS_EOY']
    YTD_df = YTD_df[['ENTITY','PERFORMANCE']]
    specialty_surplus_eoy_dict = YTD_df.set_index('ENTITY').T.to_dict('records')[0]
        
    return specialty_surplus_ytd_dict, specialty_surplus_lag_dict, specialty_surplus_eoy_dict

def disp_fee_surplus_dict_generator_ytd_lag_eoy(lp_vol_mv_agg_df, performance_dict, perf_dict_col):
    '''
    Inputs:
    lp_vol_mv_agg_df - claims data, will use projections and dispensing fee avgs to calculate surplus
    performance_dict - any performance dictionary, this function only uses the 'ENTITY' as a template to get the rows
    
    1. projects to the end of year
    2. maps to ENTITY in performance dictionary using breakout_df as a template - should be a performance dictionary as a df
    3. changes df to dictionary
    
    Returns
    -------
    Dispensing Fee surplus EOY performance Dictionary.

    '''

    surplus_df = lp_vol_mv_agg_df[['CLIENT','BREAKOUT','CHAIN_GROUP', 'CHAIN_SUBGROUP','BG_FLAG',  'AVG_DISP_FEE', 
                                   'AVG_TARGET_DISP_FEE', 'DISP_FEE', 'TARGET_DISP_FEE', 'CLAIMS_PROJ_EOY', 
                                   'CLAIMS_PROJ_LAG', 'PHARMACY_DISP_FEE',  'PHARM_AVG_DISP_FEE',
                                   'PHARM_AVG_TARGET_DISP_FEE', 'PHARM_CLAIMS_PROJ_EOY', 'PHARM_CLAIMS_PROJ_LAG', 
                                   'PHARM_TARGET_DISP_FEE']].copy()
    surplus_df['SURPLUS_YTD'] = surplus_df['TARGET_DISP_FEE'] - surplus_df['DISP_FEE']
    surplus_df['SURPLUS_LAG'] = surplus_df['CLAIMS_PROJ_LAG'] * (surplus_df['AVG_TARGET_DISP_FEE'] - surplus_df['AVG_DISP_FEE'])
    surplus_df['SURPLUS_EOY'] = surplus_df['CLAIMS_PROJ_EOY'] * (surplus_df['AVG_TARGET_DISP_FEE'] - surplus_df['AVG_DISP_FEE'])
    surplus_df['FULL_YEAR'] = surplus_df['SURPLUS_YTD'] + surplus_df['SURPLUS_LAG'] + surplus_df['SURPLUS_EOY']
    
    surplus_df['PHARM_SURPLUS_YTD'] = surplus_df['PHARM_TARGET_DISP_FEE'] - surplus_df['PHARMACY_DISP_FEE']
    surplus_df['PHARM_SURPLUS_LAG'] = surplus_df['PHARM_CLAIMS_PROJ_LAG'] * (surplus_df['PHARM_AVG_TARGET_DISP_FEE'] - surplus_df['PHARM_AVG_DISP_FEE'])
    surplus_df['PHARM_SURPLUS_EOY'] = surplus_df['PHARM_CLAIMS_PROJ_EOY'] * (surplus_df['PHARM_AVG_TARGET_DISP_FEE'] - surplus_df['PHARM_AVG_DISP_FEE'])
    surplus_df['PHARM_FULL_YEAR'] = surplus_df['PHARM_SURPLUS_YTD'] + surplus_df['PHARM_SURPLUS_LAG'] + surplus_df['PHARM_SURPLUS_EOY']

    channel_perf_df = surplus_df[
        ['CLIENT','BREAKOUT','SURPLUS_YTD', 'SURPLUS_LAG', 'SURPLUS_EOY', 'FULL_YEAR']
    ].groupby(['CLIENT','BREAKOUT']).sum().reset_index()
    channel_perf_df['ENTITY'] = channel_perf_df['CLIENT'] + "_" + channel_perf_df['BREAKOUT']
    
    pharm_perf_df = surplus_df[
        ['CLIENT','CHAIN_SUBGROUP','BG_FLAG','PHARM_SURPLUS_YTD', 'PHARM_SURPLUS_LAG', 'PHARM_SURPLUS_EOY', 'PHARM_FULL_YEAR']
    ].groupby(['CLIENT','CHAIN_SUBGROUP','BG_FLAG']).sum().reset_index()
    pharm_perf_df['ENTITY'] = pharm_perf_df['CHAIN_SUBGROUP']
    pharm_perf_df.rename(columns = {'PHARM_SURPLUS_YTD': 'SURPLUS_YTD', 'PHARM_SURPLUS_LAG': 'SURPLUS_LAG', 'PHARM_SURPLUS_EOY':'SURPLUS_EOY', 'PHARM_FULL_YEAR': 'FULL_YEAR'}, inplace=True)
    
    pharm_perf_df.loc[(pharm_perf_df['ENTITY'].isin(p.NON_CAPPED_PHARMACY_LIST['GNRC']+p.COGS_PHARMACY_LIST['GNRC']) & pharm_perf_df['BG_FLAG']=='G') |
                      (pharm_perf_df['ENTITY'].isin(p.NON_CAPPED_PHARMACY_LIST['BRND']+p.COGS_PHARMACY_LIST['BRND']) & pharm_perf_df['BG_FLAG']=='B'), 
                      ['SURPLUS_YTD', 'SURPLUS_LAG', 'SURPLUS_EOY', 'FULL_YEAR']] = 0
    pharm_perf_df = pharm_perf_df[['ENTITY', 'SURPLUS_YTD', 'SURPLUS_LAG', 'SURPLUS_EOY', 'FULL_YEAR']].groupby('ENTITY').sum().reset_index()
    surplus_df = pd.concat([channel_perf_df[['ENTITY', 'SURPLUS_YTD', 'SURPLUS_LAG', 'SURPLUS_EOY', 'FULL_YEAR']],
                            pharm_perf_df[['ENTITY', 'SURPLUS_YTD', 'SURPLUS_LAG', 'SURPLUS_EOY', 'FULL_YEAR']]])
    
    breakout_df = surplus_df.copy()
    
    if p.FULL_YEAR:
            FULL_YEAR_df = breakout_df.copy(deep=True)    
            FULL_YEAR_df['PERFORMANCE'] = FULL_YEAR_df['FULL_YEAR']
            FULL_YEAR_df = FULL_YEAR_df[['ENTITY','PERFORMANCE']]
            FULL_YEAR_0_df = FULL_YEAR_df.copy(deep=True)
            FULL_YEAR_0_df['PERFORMANCE'] = 0
            return FULL_YEAR_0_df, FULL_YEAR_0_df, FULL_YEAR_df

    # Create YTD dictionary 
    YTD_df = breakout_df.copy(deep=True)    
    YTD_df['PERFORMANCE'] = YTD_df['SURPLUS_YTD']
    YTD_df = YTD_df[['ENTITY','PERFORMANCE']]
    disp_fee_surplus_ytd_dict = YTD_df.set_index('ENTITY').T.to_dict('records')[0]

    # Create Lag dictionary 
    LAG_df = breakout_df.copy(deep=True)    
    LAG_df['PERFORMANCE'] = LAG_df['SURPLUS_LAG']
    LAG_df = LAG_df[['ENTITY','PERFORMANCE']]
    disp_fee_surplus_lag_dict = LAG_df.set_index('ENTITY').T.to_dict('records')[0]

    # Create EOY dictionary 
    EOY_df = breakout_df.copy(deep=True)    
    EOY_df['PERFORMANCE'] = EOY_df['SURPLUS_EOY']
    EOY_df = EOY_df[['ENTITY','PERFORMANCE']]
    disp_fee_surplus_eoy_dict = EOY_df.set_index('ENTITY').T.to_dict('records')[0]

    return disp_fee_surplus_ytd_dict, disp_fee_surplus_lag_dict, disp_fee_surplus_eoy_dict

def determine_effective_price(df: pd.DataFrame,
                              old_price: str,
                              uc_unit: str = 'UC_UNIT',
                              capped_only: bool = False,
                              ) -> pd.Series:
    '''
    For each row of the input DataFrame, returns the effective price based on
    the price ceiling. Price ceiling is based on min of U&C amount,
    awp discount, or old price. Price floor is based on max of 1026
    floor and price.
    
    NOTE: the price will be floored after being capped, so if the floor is
    above the price cap then the floor is returned.
    
    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame that includes the following columns:
            * old price (must be named explicitly)
            * "UC_UNIT" (another U&C column can be named instead, optionally)
            * "CHAIN_GROUP"
            * "AVG_AWP"
            * "MAC1026_UNIT_PRICE"
    
    old_price : str
        A string stating the column within the DataFrame that contains
        the old price.
    uc_unit: str, default "UC_UNIT"
        A string stating the column within the DataFrame that contains
        the U&C for each drug. Will be "UC_UNIT" if not specified.
    capped_only : bool, default False
        An option to only impose caps and not floors onto the price.
    
    Returns
    -------
    pd.Series
        The effective price after capping and flooring the original price.
        This returned Series object has the same length and index
        as the input DataFrame.
    
    Examples
    --------
    # can be used to create a new column directly
    >>> lp_vol_macprice_df['EFF_UNIT_PRICE'] = (
    ...  determine_effective_price(lp_vol_macprice, 'CURRENT_MAC_PRICE')
    ... )
    # can also be used with pipe
    >>> lp_vol_macprice_df['EFF_UNIT_PRICE'] = (
    ...  lp_vol_macprice_df
    ...   .pipe(determine_effective_price, 'CURRENT_MAC_PRICE')
    ... )
    # or assign
    >>> (lp_vol_macprice_df
    ...   .assign(determine_effective_price, old_price='CURRENT_MAC_PRICE')
    ...   .rename('EFF_UNIT_PRICE'))
    0   |price1   |
    1   |price2   |
    ...
    5000|price5001|
    Name: EFF_UNIT_PRCE, Length: 5001, dtype: float64
        
    '''
    from enum import IntEnum
    
    import numba
    import numpy as np
    import pandas as pd
    
    
    rpu_exists = 'RAISED_PRICE_UC' in df.columns
    unc_opt = p.UNC_OPT
    # Split the DataFrame based on BG_FLAG
    df_brnd = df[df['BG_FLAG'] == 'B']
    df_gnrc = df[df['BG_FLAG'] == 'G']

    # Define pharm_lists for each subset
    pharm_lists_brnd = p.NON_CAPPED_PHARMACY_LIST['BRND'] + p.COGS_PHARMACY_LIST['BRND']
    pharm_lists_gnrc = p.NON_CAPPED_PHARMACY_LIST['GNRC'] + p.COGS_PHARMACY_LIST['GNRC']

    def calculate_price(sub_df, pharm_lists):
        class ColMap(IntEnum):
            """
            A mapping of column names to ints. Makes indexing a numpy array easier:
            >>> columns = ['AVG_FAC', 'CURRENT_MAC_PRICE']
            >>> df[columns]
            |'AVG_FACxAWP'|'CURRENT_MAC_PRICE'|
            |-------------|-------------------|
            |4.4          |3.3                |
            |5.5          |2.2                |
            # this means very little:
            >>> df[columns].values[0][0]
            4.4
            # this is a little easier:
            >>> df[columns].values[0][ColMap.AVG_FACxAWP.value]
            4.4
            """
            AVG_FACxAWP: int = 0
            OLD_PRICE: int = 1
            AVG_AWP: int = 2
            MAC1026_UNIT_PRICE: int = 3
            CHAIN_GROUP: int = 4
            UC_UNIT: int = 5
            ISMAC: int = 6
            PHARM_GUARANTEE: int = 7
            FLOOR_FACxAWP: int = 8
            RAISED_PRICE_UC: int = 9

        # give a number to each unique CHAIN_GROUP
        CG = IntEnum('ChainGroups',
                     list(set(sub_df.CHAIN_GROUP.values) | set(pharm_lists)))
        maxgroups = tuple(CG[group].value for group in pharm_lists)

        @numba.jit(nopython=True)
        def get_price(rows: np.array) -> np.array:
            result = np.empty(len(rows), dtype=np.float64)
            for i in range(len(rows)):
                if (p.FLOOR_NON_MAC_RATE < 1) and ((not rows[i][ColMap.PHARM_GUARANTEE.value]) and (not rows[i][ColMap.ISMAC.value])):
                    if (np.isfinite(rows[i][ColMap.OLD_PRICE.value])
                        and rows[i][ColMap.OLD_PRICE.value] > 0):
                        eff_price = max([rows[i][ColMap.OLD_PRICE.value], rows[i][ColMap.FLOOR_FACxAWP.value]])
                        if rows[i][ColMap.UC_UNIT.value] > 0:
                            eff_price = min([eff_price, rows[i][ColMap.UC_UNIT.value]])
                        result[i] = eff_price
                        assert eff_price != 0, 'Incorrect MAC calculation'
                    else:
                        result[i] = rows[i][ColMap.OLD_PRICE.value]
                elif unc_opt and rpu_exists and rows[i][ColMap.RAISED_PRICE_UC.value]:

                    if (np.isfinite(rows[i][ColMap.OLD_PRICE.value])
                        and rows[i][ColMap.OLD_PRICE.value] > 0):
                        if rows[i][ColMap.AVG_AWP.value] > 0:
                            eff_price = min([
                             rows[i][ColMap.AVG_FACxAWP.value],
                             rows[i][ColMap.OLD_PRICE.value],
                            ])
                        else:
                            eff_price = rows[i][ColMap.OLD_PRICE.value]

                        if not capped_only:
                            if (rows[i][ColMap.MAC1026_UNIT_PRICE.value] > 0
                                and rows[i][ColMap.CHAIN_GROUP.value] in maxgroups):

                                result[i] = max([
                                 rows[i][ColMap.MAC1026_UNIT_PRICE.value],
                                 eff_price,
                                ])
                            else:
                                result[i] = eff_price
                        else:
                            result[i] = eff_price
                    else:
                        result[i] = rows[i][ColMap.OLD_PRICE.value]

                elif (np.isfinite(rows[i][ColMap.OLD_PRICE.value])
                      and rows[i][ColMap.OLD_PRICE.value] > 0):
                    if (rows[i][ColMap.UC_UNIT.value] > 0
                        and rows[i][ColMap.AVG_AWP.value] > 0):
                        eff_price = min([
                         rows[i][ColMap.UC_UNIT.value],
                         rows[i][ColMap.AVG_FACxAWP.value],
                         rows[i][ColMap.OLD_PRICE.value]
                        ])

                    elif rows[i][ColMap.UC_UNIT.value] > 0:

                        eff_price = min([
                         rows[i][ColMap.UC_UNIT.value],
                         rows[i][ColMap.OLD_PRICE.value],
                        ])

                    elif rows[i][ColMap.AVG_AWP.value] > 0:

                        eff_price = min([
                         rows[i][ColMap.AVG_FACxAWP.value],
                         rows[i][ColMap.OLD_PRICE.value],
                        ])

                    else:
                        eff_price = rows[i][ColMap.OLD_PRICE.value]
                
                    if not capped_only:
                        if (rows[i][ColMap.MAC1026_UNIT_PRICE.value] > 0
                            and rows[i][ColMap.CHAIN_GROUP.value] in maxgroups):

                            result[i] = max([
                             rows[i][ColMap.MAC1026_UNIT_PRICE.value],
                             eff_price,
                            ])
                        else:
                            result[i] = eff_price
                    else:
                        result[i] = eff_price
                else:
                    result[i] = rows[i][ColMap.OLD_PRICE.value]
            return result

        # declare only the columns that numba needs for this operation
        # these must be declared in the same order as the ColMap IntEnum
        columns = [
         'AVG_FACxAWP',
         old_price,
         'AVG_AWP',
         'MAC1026_UNIT_PRICE',
         'CHAIN_GROUP',
         uc_unit,
            'ISMAC',
            'PHARM_GUARANTEE',
            'FLOOR_FACxAWP',
        ]
        if rpu_exists:
            print(sub_df['RAISED_PRICE_UC'].isna().sum(), len(sub_df), "check nans")
            print(sub_df[sub_df['QTY']>0]['RAISED_PRICE_UC'].isna().sum(), (sub_df['QTY']>0).sum(), "check nans 2")
        columns += ['RAISED_PRICE_UC'] if rpu_exists else []
        avg_facs = np.where(sub_df.BG_FLAG == 'G', 
                        np.where(sub_df.CHAIN_GROUP.isin(['MAIL', 'MCHOICE']),
                            1 - p.MAIL_NON_MAC_RATE,
                            1 - p.RETAIL_NON_MAC_RATE),
                        1 - p.BRAND_NON_MAC_RATE)

        floor_fac = 1 - p.FLOOR_NON_MAC_RATE

        mac_choices = np.where(sub_df.IS_MAC,
                            1.0,
                            0.0)

        guarantee_choices = np.where(sub_df.HAS_PHARM_GUARANTEE,
                            1.0,
                            0.0)

        price = get_price(sub_df
                           # calculate AVG_FAC x AVG_AWP beforehand
                           # map CHAIN_GROUP strings to ChainGroups integer Enum
                           .assign(AVG_FACxAWP=avg_facs * sub_df.AVG_AWP,
                                   FLOOR_FACxAWP=floor_fac * sub_df.AVG_AWP,
                                   CHAIN_GROUP=sub_df.CHAIN_GROUP.map(CG.__getattr__),
                                  ISMAC=mac_choices,
                                  PHARM_GUARANTEE=guarantee_choices)
                           # boolean datatype doesn't work with numba; must cast
                           .astype({'RAISED_PRICE_UC': int} if rpu_exists else {})
                           .filter(items=columns, axis=1)
                           .values)
        return pd.Series(price, index=sub_df.index)
    price_brnd = calculate_price(df_brnd, pharm_lists_brnd) if not df_brnd.empty else pd.Series([])
    price_gnrc = calculate_price(df_gnrc, pharm_lists_gnrc) if not df_gnrc.empty else pd.Series([])
    
    # Combine results
    return pd.concat([price_brnd, price_gnrc]).sort_index()


def set_run_status():
    """
    Inserts a new row into the client_run_status using the parameters from the CPMO_Parameter.py, with unique at_run_id
    
    PARAMETERS : N/A 
    
    RETURNS : N/A   
    
    NOTE : Func:set_run_status needs to make sure the at_run_id is unique in the client_run_status table    
    """
    from google.cloud import bigquery

    if p.TRACK_RUN:
        import datetime as dt
        bq_timestamp = str(dt.datetime.strptime(p.TIMESTAMP, '%Y-%m-%d_%H%M%S%f').replace(microsecond=0))
        query = f"""
        INSERT INTO pbm-mac-lp-prod-ai.pricing_management.client_run_status (
            customer_id, client_name, at_run_id, run_type,
            program_output_path, run_timestamp, run_status, error_type, error_message, error_location,
            run_owner, client_type, goodrx_opt, goodrx_file, unc_opt, unc_adjust,
            go_live, data_start_day, last_data, tiered_price_lim, psao_treatment,
            locked_client, interceptor_opt, full_year, zero_qty_tight_bounds,batch_id)
        VALUES (
            "{p.CUSTOMER_ID[0]}", "{p.CLIENT_NAME_TABLEAU}", "{p.AT_RUN_ID}", "{p.RUN_TYPE_TABLEAU}"
            , "{p.PROGRAM_OUTPUT_PATH}", "{bq_timestamp}", "Started", "", "", ""
            , "{p.USER}", "{p.CLIENT_TYPE}", "{p.RMS_OPT}", "{p.RAW_GOODRX}", "{p.UNC_OPT}", "{p.UNC_ADJUST}"
            , "{p.GO_LIVE}", "{p.DATA_START_DAY}", "{p.LAST_DATA}", "{p.TIERED_PRICE_LIM}", "{p.PSAO_TREATMENT}"
            , "{p.LOCKED_CLIENT}", "{p.INTERCEPTOR_OPT}", "{p.FULL_YEAR}", "{p.ZERO_QTY_TIGHT_BOUNDS}","{p.BATCH_ID}"
        )
        """
        bqclient = bigquery.Client()
        bqclient.query(query)
        print("This run is being tracked with AT_RUN_ID:{} in the client_run_status table".format(p.AT_RUN_ID))

    else:
        print("This run is not being tracked at client_run_status table")

def check_run_status(run_status = 'Failed'):
    """
    Compares the run_status for a run_id in the client_run_status table with the parameter:run_status. 

    PARAMETER : run_status 
        >> Set the run_status as 'Started', 'Complete', 'Failed', or 'Complete-BypassPerformance' 
        >> Default is 'Failed'

    RETURNS: Binary outcome
        If at_run_id is not present in the client_run_status table, check_run_status func will return 0 
        If parameter:run_status == run_status in the client_run_status table, then return 1, else 0
        If TRACK_RUN is set to False, the function will return 0

    NOTE: time.sleep(2) is required. In the pre_proceesing we update set_run_status using func:set_run_status, 
          That takes about ~2 seconds to be reflected in the table, post which we run the func:check_run_status to find the at_run_id 
          and, do the comparison.

    """
    import time
    from google.cloud import bigquery
    if not p.TRACK_RUN:    
        return 0 

    time.sleep(2)
    print("Check for the RUN ID {} in the client_run_status table".format(p.AT_RUN_ID))

    query = f"""
    SELECT run_status FROM pbm-mac-lp-prod-ai.pricing_management.client_run_status
    WHERE at_run_id = '{p.AT_RUN_ID}'
    """
    bqclient = bigquery.Client()
    query_df = bqclient.query(query).to_dataframe()
        
    if query_df.shape[0] == 0:
        print("ALERT: at_run_id doesn't exist in pbm-mac-lp-prod-ai.pricing_management.client_run_status")
        return 0
    
    if query_df.loc[0, 'run_status']  != run_status:
        print("ALERT: at_run_id exists in pbm-mac-lp-prod-ai.pricing_management.client_run_status with run_status {} and not {}".format(query_df.loc[0, 'run_status'], run_status))
        return 0
    
    print("at_run_id exists in pbm-mac-lp-prod-ai.pricing_management.client_run_status with run_status: {}".format(query_df.loc[0, 'run_status']))
    return 1


def update_run_status(run_status=None, i_error_type=None, i_error_message=None, i_error_location=None, i_run_type = None, num_conflict_gpi=None):
    """
    Updates a row in the client_run_status, for a perticular at_run_id
    
    PARAMETERS : run_status (Default : None)
                    >>  Sets the run_status as 'Started', 'Complete', Complete-BypassPerformance', or 'Failed' 
                        >> If None is passed, it refers to the run_status not being updated 
                    >>  When a new LP run is initiated the run_status is 'Started'
                    >>  If LP run is successful the run_status is 'Complete'
                    >>  If LP run is fails anywhere in the code the run_status is 'Failed'
                    >>  Param : run_status updates the run_status column in the client_run_status table

                 i_error_type (Default : None)
                    >> User defines the error type in the code. 
                    >> Param : i_error_type appends to the error_type column in the client_run_status table

                 i_error_message (Default : None) 
                    >> error_message : repr(e) is passed accross LP 
                    >> Param : i_error_message appends to error_message column in the client_run_status table

                 i_error_location (Default : None) 
                    >> Name of the .py file that failed in LP
                    >> Param : i_error_location appends to error_location column in the client_run_status table

                 i_run_type (Default : None) 
                    >> run_type is the column that is reflected on PSOT. User trys to be explicit on type of run. 
                        Example: run_type with postfix "- bypass performance" means either the code has failed the performance 
                        and/or IGNORE_PERFORMANCE_FLAG is set to true by the user. 
                    >> Param : i_run_type appends to run_type column in the client_run_status table, post checking substring is not duplicated  
                update_run_status(Default : True)
                    >> Run status is updated, only when the update_run_status is set to 'True'
                num_conflict_gpi(Default : None)
                    >> Specifies the number of conflicting GPIs extracted from the conflicting GPI file generated in CPMO.py and uploads this 
                       information to the client_run_status table 
 
    RETURNS : N/A   
    """
    import util_funcs as uf
    
    error_type = i_error_type + ', ' if i_error_type and run_status != 'Complete' else ''
    error_message = i_error_message.replace("'", '"') + ', ' if i_error_message and run_status != 'Complete' else ''
    error_location = i_error_location.replace("'", '"') + ', ' if i_error_location and run_status != 'Complete' else ''
    run_type = i_run_type if i_run_type else ''

    if p.TRACK_RUN:
        ## Only when run_status is being passed then we should populate the error_type, error_message and error location 
        if run_status:
            query = f"""
                UPDATE pbm-mac-lp-prod-ai.pricing_management.client_run_status
                SET run_status = '{run_status}',
                    error_type = error_type || '{error_type}',
                    error_message = error_message || '{error_message}',
                    error_location = error_location || '{error_location}',
                    run_type = run_type || '{run_type}'
                WHERE at_run_id = '{p.AT_RUN_ID}' 
            """
            uf.write_to_bq(
                df = None,
                project_output = None,
                dataset_output = None,
                table_id = None,
                client_name_param = None,
                timestamp_param = None,
                run_id= None,
                query = query)
       
        elif isinstance(num_conflict_gpi,int):
            try:
                query = f"""
                UPDATE pbm-mac-lp-prod-ai.pricing_management.client_run_status
                SET num_conflict_gpi = {num_conflict_gpi}
                WHERE at_run_id = '{p.AT_RUN_ID}' 
                """
                uf.write_to_bq(
                    df = None,
                    project_output = None,
                    dataset_output = None,
                    table_id = None,
                    client_name_param = None,
                    timestamp_param = None,
                    run_id= None,
                    query = query)
            except:
                print("Num Conflict GPI is not updated in client_run_status")

        # We want to reduce the write to the client_run_status table to prevent the quota issue, hence have eliminated the code that gives us information on section of the code we are running. 
        ## As next steps:  We will make sure the write to BQ update that gives us information on section of the code we are running is made only once. 
        # else: 
        #     query = f"""
        #     UPDATE pbm-mac-lp-prod-ai.pricing_management.client_run_status
        #     SET error_type = error_type || '{error_type}',
        #     error_message = error_message || '{error_message}',
        #     error_location = error_location || '{error_location}',
        #     run_type = run_type || '{run_type}'
        #     WHERE at_run_id = '{p.AT_RUN_ID}' 
        #     """

def add_virtual_r90(client_guarantees = None, update_guarantees = False, return_check = False):
    """
    Check if client has a virtual R90 offsetting, where they only have an R30 guarantee, but we measure R90 claims
    See query for combined measurement mapping for additional logic used to identify these clients.
    
    Arguments:
        client_guarantees: client guarantees df that is missing R90 guarantees
        update_guarantees: Boolean, set to True if you'd like to have the guarantees df returned with R90 rows
        return_check: Boolean, set to True if you'd like the function to return a boolean indicating a client is a virtual R90 client
    
    Returns:
        client_guarantees: df with the virtual R90 rows added
        virtual_r90_flag: Boolean indicating if the client meets virtual R90 criteria
    """
    
    import pandas as pd
    import CPMO_parameters as p
    import BQ
    import util_funcs as uf
    
    # Get measurement mapping
    if p.READ_FROM_BQ:
        measurement_mapping = uf.read_BQ_data(
                BQ.ger_opt_msrmnt_map.format(_customer_id=uf.get_formatted_string(p.CUSTOMER_ID)),
                project_id=p.BQ_INPUT_PROJECT_ID,
                dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                table_id="combined_measurement_mapping" + p.WS_SUFFIX + p.CCP_SUFFIX,
                customer=', '.join(sorted(p.CUSTOMER_ID)))
    else:
        measurement_mapping = pd.read_csv(p.FILE_INPUT_PATH + p.MEASUREMENT_MAPPING, dtype = p.VARIABLE_TYPE_DIC)

    # Get guarantee category
    guarantee_category = measurement_mapping['Guarantee_Category'][0]
    
    if ((guarantee_category == 'Pure Vanilla' or guarantee_category == 'MedD/EGWP Vanilla') and 
        'R90' in measurement_mapping['MEASUREMENT_CLEAN'].unique()):
        
        # If the code reached this point, the client meets virtual R90 criteria
        if return_check == True:
            virtual_r90_flag = True
        
        # Create R90 rows and copy the R30 guarantee to them, only if R90 guarantees are not already present
        if update_guarantees == True and 'R90' not in client_guarantees['MEASUREMENT'].values:
            
            assert isinstance(client_guarantees, pd.DataFrame), "No dataframe was passed to the virtual R90 function."
            

            vR90 = {'CLIENT': [], 'REGION': [], 'BREAKOUT': [], 'MEASUREMENT': [], 'BG_FLAG': [], 'RATE': [], 'PHARMACY_TYPE': []}
            if p.GENERIC_OPT:
                vR90['CLIENT'].extend([client_guarantees['CLIENT'][0], client_guarantees['CLIENT'][0]])
                vR90['REGION'].extend([client_guarantees['REGION'][0], client_guarantees['REGION'][0]])
                vR90['BREAKOUT'].extend([client_guarantees['BREAKOUT'].loc[(client_guarantees['MEASUREMENT'] == 'R30') & (client_guarantees['BG_FLAG'] == 'G')].values[0],
                                            client_guarantees['BREAKOUT'].loc[(client_guarantees['MEASUREMENT'] == 'R30') & (client_guarantees['BG_FLAG'] == 'G')].values[0]])
                vR90['MEASUREMENT'].extend(['R90', 'R90'])
                vR90['BG_FLAG'].extend(['G', 'G'])
                vR90['RATE'].extend([client_guarantees['RATE'].loc[(client_guarantees['MEASUREMENT'] == 'R30') & (client_guarantees['BG_FLAG'] == 'G') & (client_guarantees['PHARMACY_TYPE'] == 'Preferred')].values[0],
                                        client_guarantees['RATE'].loc[(client_guarantees['MEASUREMENT'] == 'R30') & (client_guarantees['BG_FLAG'] == 'G') & (client_guarantees['PHARMACY_TYPE'] == 'Non_Preferred')].values[0]])
                vR90['PHARMACY_TYPE'].extend(['Preferred', 'Non_Preferred'])
                
            if p.BRAND_OPT:
                vR90['CLIENT'].extend([client_guarantees['CLIENT'][0], client_guarantees['CLIENT'][0]])
                vR90['REGION'].extend([client_guarantees['REGION'][0], client_guarantees['REGION'][0]])
                vR90['BREAKOUT'].extend([client_guarantees['BREAKOUT'].loc[(client_guarantees['MEASUREMENT'] == 'R30') & (client_guarantees['BG_FLAG'] == 'B')].values[0],
                                        client_guarantees['BREAKOUT'].loc[(client_guarantees['MEASUREMENT'] == 'R30') & (client_guarantees['BG_FLAG'] == 'B')].values[0]])
                vR90['MEASUREMENT'].extend(['R90', 'R90'])
                vR90['BG_FLAG'].extend(['B', 'B'])
                vR90['RATE'].extend([client_guarantees['RATE'].loc[(client_guarantees['MEASUREMENT'] == 'R30') & (client_guarantees['BG_FLAG'] == 'B') & (client_guarantees['PHARMACY_TYPE'] == 'Preferred')].values[0],
                                    client_guarantees['RATE'].loc[(client_guarantees['MEASUREMENT'] == 'R30') & (client_guarantees['BG_FLAG'] == 'B') & (client_guarantees['PHARMACY_TYPE'] == 'Non_Preferred')].values[0]])
                vR90['PHARMACY_TYPE'].extend(['Preferred', 'Non_Preferred'])

                vR90 = pd.DataFrame(vR90)
            client_guarantees = client_guarantees.append(vR90, ignore_index = True)
    else:
        if return_check == True:
            virtual_r90_flag = False

    if update_guarantees == True and return_check == True:
        return client_guarantees, virtual_r90_flag
    elif update_guarantees == True and return_check == False:
        return client_guarantees
    elif update_guarantees == False and return_check == True:
        return virtual_r90_flag
    else:
        assert False, "You aren't asking this function to return anything, double check arguments."

def add_rur_guarantees(pharmacy_guarantees, virtual_r90_flag):
    # Adds RUR guarantees to the pharmacy guarantees file if a RUR VCML is present.
    # Guarantee is based on the corresponding client guarantee, plus a buffer defined by RUR_GUARANTEE_BUFFER.

    import pandas as pd
    import CPMO_parameters as p
    
    # Get VCML Reference file
    vcml_ref = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.VCML_REFERENCE_FILE, dtype = p.VARIABLE_TYPE_DIC)

    # If the client has a RUR VCML, add the buffered client guarantees to the pharmacy guarantees file.
    if vcml_ref['VCML_ID'].str.contains(p.APPLY_VCML_PREFIX + p.CUSTOMER_ID[0] + 'R1' or
                                        p.APPLY_VCML_PREFIX + p.CUSTOMER_ID[0] + 'R3' or
                                        p.APPLY_VCML_PREFIX + p.CUSTOMER_ID[0] + 'R9', regex=True).any():
        assert p.CLIENT_TYPE != 'MEDD', "A Rural VCML was detected. This is not currently implemented for MEDD clients."

        # Read in client guarantees generated earlier in Pre_Processing.py
        # Using Effective GERs for TC clients
        if p.TRUECOST_CLIENT:
            client_guarantees = uf.read_BQ_data(
                BQ.eff_client_guarantees_TC,
                project_id=p.BQ_OUTPUT_PROJECT_ID,
                dataset_id=p.BQ_INPUT_DATASET_SANDBOX,
                table_id='TrueCost_EFF_GER',
                client = ', '.join(sorted(p.CUSTOMER_ID)))
        else:
            client_guarantees = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.CLIENT_GUARANTEE_FILE, dtype = p.VARIABLE_TYPE_DIC)

        # Gets the appropriate client guarantee for each measurement, generates a new row with the buffered rate
        # and appends it to the pharmacy guarantees file.
        rur_guarantees = {'CLIENT': [], 'REGION': [], 'BREAKOUT': [], 'MEASUREMENT': [], 'BG_FLAG': [], 'RATE': [], 'PHARMACY': [], 'PHARMACY_SUB': [],'PHRM_GRTE_TYPE': []}

        if vcml_ref['VCML_ID'].str.contains(p.APPLY_VCML_PREFIX + p.CUSTOMER_ID[0] + 'R1').any():
            if p.GENERIC_OPT:
                rur_guarantees['CLIENT'].append(pharmacy_guarantees['CLIENT'].iloc[0])
                rur_guarantees['REGION'].append(pharmacy_guarantees['REGION'].iloc[0])
                rur_guarantees['BREAKOUT'].append(pharmacy_guarantees['BREAKOUT'].loc[(pharmacy_guarantees['MEASUREMENT'] == 'R30') & 
                                                                                     (pharmacy_guarantees['BG_FLAG'] == 'G')].values[0])
                rur_guarantees['MEASUREMENT'].append('R30')
                rur_guarantees['BG_FLAG'].append('G')
                rur_guarantees['RATE'].append(client_guarantees['RATE'].loc[(client_guarantees['MEASUREMENT'] == 'R30') &
                                                                            (client_guarantees['BG_FLAG'] == 'G') &
                                                                            (client_guarantees['PHARMACY_TYPE'] == 'Non_Preferred')].values[0] + p.RUR_GUARANTEE_BUFFER)
                rur_guarantees['PHARMACY'].append('RUR')
                rur_guarantees['PHARMACY_SUB'].append('RUR')
                rur_guarantees['PHRM_GRTE_TYPE'].append('AWP')

            if p.BRAND_OPT:
                rur_guarantees['CLIENT'].append(pharmacy_guarantees['CLIENT'].iloc[0])
                rur_guarantees['REGION'].append(pharmacy_guarantees['REGION'].iloc[0])
                rur_guarantees['BREAKOUT'].append(pharmacy_guarantees['BREAKOUT'].loc[(pharmacy_guarantees['MEASUREMENT'] == 'R30') & 
                                                                                     (pharmacy_guarantees['BG_FLAG'] == 'B')].values[0])
                rur_guarantees['MEASUREMENT'].append('R30')
                rur_guarantees['BG_FLAG'].append('B')
                rur_guarantees['RATE'].append(client_guarantees['RATE'].loc[(client_guarantees['MEASUREMENT'] == 'R30') &
                                                                            (client_guarantees['BG_FLAG'] == 'B') &
                                                                            (client_guarantees['PHARMACY_TYPE'] == 'Non_Preferred')].values[0] + p.RUR_GUARANTEE_BUFFER)
                rur_guarantees['PHARMACY'].append('RUR')
                rur_guarantees['PHARMACY_SUB'].append('RUR')
                rur_guarantees['PHRM_GRTE_TYPE'].append('AWP')

        if vcml_ref['VCML_ID'].str.contains(p.APPLY_VCML_PREFIX + p.CUSTOMER_ID[0] + 'R3' or
                                            p.APPLY_VCML_PREFIX + p.CUSTOMER_ID[0] + 'R9').any() or (virtual_r90_flag == True):
            if p.GENERIC_OPT:
                rur_guarantees['CLIENT'].append(pharmacy_guarantees['CLIENT'].iloc[0])
                rur_guarantees['REGION'].append(pharmacy_guarantees['REGION'].iloc[0])
                rur_guarantees['BREAKOUT'].append(pharmacy_guarantees['BREAKOUT'].loc[(pharmacy_guarantees['MEASUREMENT'] == 'R90') & 
                                                                                     (pharmacy_guarantees['BG_FLAG'] == 'G')].values[0])
                rur_guarantees['MEASUREMENT'].append('R90')
                rur_guarantees['BG_FLAG'].append('G')
                rur_guarantees['RATE'].append(client_guarantees['RATE'].loc[(client_guarantees['MEASUREMENT'] == 'R90') &
                                                                            (client_guarantees['BG_FLAG'] == 'G') &
                                                                            (client_guarantees['PHARMACY_TYPE'] == 'Non_Preferred')].values[0] + p.RUR_GUARANTEE_BUFFER)
                rur_guarantees['PHARMACY'].append('RUR')
                rur_guarantees['PHARMACY_SUB'].append('RUR')
                rur_guarantees['PHRM_GRTE_TYPE'].append('AWP')

            if p.BRAND_OPT:
                rur_guarantees['CLIENT'].append(pharmacy_guarantees['CLIENT'].iloc[0])
                rur_guarantees['REGION'].append(pharmacy_guarantees['REGION'].iloc[0])
                rur_guarantees['BREAKOUT'].append(pharmacy_guarantees['BREAKOUT'].loc[(pharmacy_guarantees['MEASUREMENT'] == 'R90') & 
                                                                                     (pharmacy_guarantees['BG_FLAG'] == 'B')].values[0])
                rur_guarantees['MEASUREMENT'].append('R90')
                rur_guarantees['BG_FLAG'].append('B')
                rur_guarantees['RATE'].append(client_guarantees['RATE'].loc[(client_guarantees['MEASUREMENT'] == 'R90') &
                                                                            (client_guarantees['BG_FLAG'] == 'B') &
                                                                            (client_guarantees['PHARMACY_TYPE'] == 'Non_Preferred')].values[0] + p.RUR_GUARANTEE_BUFFER)
                rur_guarantees['PHARMACY'].append('RUR')
                rur_guarantees['PHARMACY_SUB'].append('RUR')
                rur_guarantees['PHRM_GRTE_TYPE'].append('AWP')

        rur_guarantees = pd.DataFrame(rur_guarantees)
        pharmacy_guarantees = pharmacy_guarantees.append(rur_guarantees, ignore_index=True)
        
        return pharmacy_guarantees      
    else:
        # If no RUR VCML is detected, just return the pharmacy guarantees file as is.
        return pharmacy_guarantees

def calc_target_ingcost(df,colname,client_rate_col='CLIENT_RATE'):
    '''
    This function is called repeatedly within add_target_ingcost() to calculate different target_ingredient_costs (YTD,LAG,EOY,LM) for (Normal,CS,ZBD).
    For YTD costs:
        target ingredient cost = (1 - Pharmacy GER) * Actual Adjudicated Values (ex - PHARM_FULLAWP_ADJ, PHARM_FULLNADAC_ADJ, PHARM_FULLACC_ADJ)
    For EOY,LAG costs:
        target ingredient cost = (1 - Pharmacy GER) * Projected Quantity * Unit Price ( ex - AVG_AWP, AVG_NADAC, AVG_ACC)
    
    input: lp_data_culled_df, column name to be calculated
    output: lp_data_culled_df with added target ingredient cost columns 
    '''
    if 'PHARM' in colname: # calculate pharmacy target ing cost based on pharmacy guarantee type
        conditions = [df.PHRM_GRTE_TYPE == 'AWP',df.PHRM_GRTE_TYPE == 'NADAC',df.PHRM_GRTE_TYPE == 'WAC',df.PHRM_GRTE_TYPE == 'ACC']
        if '_LAG' in colname or '_EOY' in colname or '_PRIOR' in colname:
            val = colname.replace('TARG_INGCOST_ADJ','QTY')
            choices = [(1 - df.PHARMACY_RATE)*df[val]*df.PHARM_AVG_AWP, 
                       (1 - df.PHARMACY_RATE)*df[val]*df.PHARM_AVG_NADAC,
                       (1 - df.PHARMACY_RATE)*df[val]*df.PHARM_AVG_WAC,
                       (1 - df.PHARMACY_RATE)*df[val]*df.PHARM_AVG_ACC]
            df[colname] = np.select(conditions,choices, default= None).astype('float')
    else: # calculate client target ing cost based on if truecost client or not
        # double replace because some columns do not include ADJ
        val = colname.replace('TARG_INGCOST','QTY').replace('_ADJ','')
        if p.TRUECOST_CLIENT:
            targ_ing_cost = (1 - df[client_rate_col])*df[val]*df.NET_COST_GUARANTEE_UNIT
        else:
            targ_ing_cost = (1 - df[client_rate_col])*df[colname.replace('TARG_INGCOST','FULLAWP')]
        df[colname] = targ_ing_cost.astype('float')
    return df    
        
def add_target_ingcost(lp_data_culled_df, client_guarantees, client_rate_col = 'CLIENT_RATE', target_cols=None):
    '''
    The purpose of this function is to define target ingredient costs for YTD, LAG & EOY, as well as ZBD
    These target ingredient costs can directly be used to calculate performance 
    
    input: lp_data_culled_df
    output: lp_data_culled_df with added target ingredient cost columns 
    
    '''
    if target_cols:
        target_ingcost_cols = target_cols.copy()
    else:
        if 'PHARM_FULLAWP_ADJ_PROJ_LAG' not in lp_data_culled_df.columns:
            target_ingcost_cols = ['PHARM_TARG_INGCOST_ADJ']
        else:
            target_ingcost_cols = ['PHARM_TARG_INGCOST_ADJ','LM_PHARM_TARG_INGCOST_ADJ','PHARM_TARG_INGCOST_ADJ_ZBD',
                                   'CS_LM_PHARM_TARG_INGCOST_ADJ','CS_LM_PHARM_TARG_INGCOST_ADJ_ZBD','PHARM_TARG_INGCOST_ADJ_PROJ_LAG',
                                   'PHARM_TARG_INGCOST_ADJ_PROJ_EOY']
        if 'FULLAWP_ADJ_PROJ_LAG' not in lp_data_culled_df.columns:
            target_ingcost_cols += ['TARG_INGCOST_ADJ']
        else:
            target_ingcost_cols += ['TARG_INGCOST_ADJ','LM_TARG_INGCOST_ADJ','TARG_INGCOST_ADJ_ZBD',
                                   'CS_LM_TARG_INGCOST_ADJ','CS_LM_TARG_INGCOST_ADJ_ZBD','TARG_INGCOST_ADJ_PROJ_LAG',
                                   'TARG_INGCOST_ADJ_PROJ_EOY']
    if (any('PHARM' not in col for col in target_ingcost_cols)) and (client_rate_col not in lp_data_culled_df.columns):
        # add client guarantees if not already in df
        lp_data_culled_df = pd.merge(
                lp_data_culled_df, client_guarantees, how ='left', 
                on = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'BG_FLAG', 'PHARMACY_TYPE'])
        for col in target_ingcost_cols:
            lp_data_culled_df = calc_target_ingcost(lp_data_culled_df,col,client_rate_col=client_rate_col)

        lp_data_culled_df.drop(columns = [client_rate_col], inplace=True)
    else:
        for col in target_ingcost_cols:
            lp_data_culled_df = calc_target_ingcost(lp_data_culled_df,col,client_rate_col=client_rate_col)
    
    return lp_data_culled_df

def write_spend_total_df(lp_data_output_df, unc_flag, comp_score, lp_vol_mv_agg_df_actual=None):
    """
    Generate and write spend total data for performance analysis and reporting.
    
    This function calculates aggregated spend totals across multiple dimensions including client spend,
    pharmacy spend, and dispensing fee totals.
    
    Parameters
    ----------
    lp_data_output_df : total output after lp has been run
    lp_vol_mv_agg_df_actual : unc spend data before lp has been run (default: None if UNC_OPT is False)
    
    Returns
    -------
    Modified lp_data_output_df with additional calculated projection columns for dispensing fees 
    (DISP_FEE_PROJ_LAG, DISP_FEE_PROJ_EOY, etc.).
    """
    if p.UNC_OPT:
        old_price_reimb_actual = (lp_vol_mv_agg_df_actual.groupby(['CLIENT','REGION', 'BREAKOUT', 'CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT',
                                                                                        'BG_FLAG','PHARMACY_TYPE','IS_MAC','IS_SPECIALTY'])[['Old_Price_Effective_Reimb_Proj_EOY'
                                                                                        ]].sum()).reset_index()
        pharm_old_price_reimb_actual = (lp_vol_mv_agg_df_actual.groupby(['CLIENT','REGION', 'BREAKOUT', 'CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT','BG_FLAG',
                                                                                            'PHARMACY_TYPE', 'IS_MAC', 'IS_SPECIALTY'])[['Pharm_Old_Price_Effective_Reimb_Proj_EOY'
                                                                                        ]].sum()).reset_index()
    if p.CROSS_CONTRACT_PROJ:
        # Use current_contract_data to get YTD FULLAWP, as lp_data_output_df has prior contract data.
        # Left joining on proj_spend_total to assure every pharmacy that has projected utilization is
        # included in awp_spend_total.
        current_contract_data = standardize_df(pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.CURRENT_CONTRACT_DATA_FILE), dtype = p.VARIABLE_TYPE_DIC))
        if p.NO_MAIL:
            current_contract_data = current_contract_data.loc[current_contract_data.MEASUREMENT != 'M30']
        
        awp_spend_total = (current_contract_data.groupby(['CLIENT','REGION', 'BREAKOUT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP',
                                                            'MEASUREMENT','BG_FLAG','PHARMACY_TYPE', 'IS_MAC', 'IS_SPECIALTY'])[['FULLAWP_ADJ','PRICE_REIMB']].sum()).reset_index()
        
        proj_spend_total = (lp_data_output_df.groupby(['CLIENT','REGION', 'BREAKOUT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP',
                                                            'MEASUREMENT','BG_FLAG','PHARMACY_TYPE', 'IS_MAC', 'IS_SPECIALTY'])[['FULLAWP_ADJ_PROJ_LAG','FULLAWP_ADJ_PROJ_EOY',
                                                                                            'LAG_REIMB','Old_Price_Effective_Reimb_Proj_EOY',
                                                                                            'Price_Effective_Reimb_Proj','TARG_INGCOST_ADJ_PROJ_EOY', 
                                                                                            'TARG_INGCOST_ADJ','TARG_INGCOST_ADJ_PROJ_LAG','DISP_FEE',
                                                                                            'DISP_FEE_PROJ_LAG','DISP_FEE_PROJ_EOY','TARGET_DISP_FEE',
                                                                                            'TARGET_DISP_FEE_PROJ_LAG','TARGET_DISP_FEE_PROJ_EOY']].sum()).reset_index()
        if p.UNC_OPT:
            del proj_spend_total['Old_Price_Effective_Reimb_Proj_EOY']
            proj_spend_total = pd.merge(proj_spend_total,old_price_reimb_actual, on = ['CLIENT','REGION', 'BREAKOUT', 'CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT','BG_FLAG','PHARMACY_TYPE', 'IS_MAC', 'IS_SPECIALTY'] )
        awp_spend_total = pd.merge(proj_spend_total, awp_spend_total, how='left', on=['CLIENT','REGION','BREAKOUT','CHAIN_GROUP','CHAIN_SUBGROUP',
                                                                                        'MEASUREMENT','PHARMACY_TYPE', 'IS_MAC', 'IS_SPECIALTY'])
        
        awp_spend_total = awp_spend_total[['CLIENT','REGION', 'BREAKOUT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP',
                                            'MEASUREMENT','BG_FLAG','PHARMACY_TYPE','FULLAWP_ADJ','FULLAWP_ADJ_PROJ_LAG',
                                            'FULLAWP_ADJ_PROJ_EOY','PRICE_REIMB','LAG_REIMB',
                                            'Old_Price_Effective_Reimb_Proj_EOY','Price_Effective_Reimb_Proj',
                                            'TARG_INGCOST_ADJ_PROJ_EOY', 'TARG_INGCOST_ADJ','TARG_INGCOST_ADJ_PROJ_LAG','DISP_FEE',
                                            'DISP_FEE_PROJ_LAG','DISP_FEE_PROJ_EOY','TARGET_DISP_FEE','TARGET_DISP_FEE_PROJ_LAG','TARGET_DISP_FEE_PROJ_EOY']].fillna(0)
        
        # pharm - awp spend
        pharm_awp_spend_total = (current_contract_data.groupby(['CLIENT','REGION', 'BREAKOUT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP',
                                                            'MEASUREMENT','BG_FLAG','PHARMACY_TYPE', 'IS_MAC', 'IS_SPECIALTY'])\
                                    [['PHARM_QTY','PHARM_FULLAWP_ADJ' # to be removed once we shift completely to target ing cost 
                                    ,'PHARM_TARG_INGCOST_ADJ','PHARM_PRICE_REIMB']].sum()).reset_index()
        
        pharm_proj_spend_total = (lp_data_output_df.groupby(['CLIENT','REGION', 'BREAKOUT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP',
                                                            'MEASUREMENT','BG_FLAG','PHARMACY_TYPE', 'IS_MAC', 'IS_SPECIALTY'])[['PHARM_QTY_PROJ_LAG','PHARM_QTY_PROJ_EOY',\
                                                                                            'PHARM_FULLAWP_ADJ_PROJ_LAG','PHARM_FULLAWP_ADJ_PROJ_EOY',\
                                                                                            'PHARM_TARG_INGCOST_ADJ_PROJ_LAG',
                                                                                            'PHARM_TARG_INGCOST_ADJ_PROJ_EOY',
                                                                                            'PHARM_LAG_REIMB','PHARM_TARGET_DISP_FEE',
                                                                                            'PHARM_TARGET_DISP_FEE_PROJ_LAG','PHARM_TARGET_DISP_FEE_PROJ_EOY','Pharm_Old_Price_Effective_Reimb_Proj_EOY',
                                                                                            'Pharm_Price_Effective_Reimb_Proj']].sum()).reset_index()                
        if p.UNC_OPT:
            del pharm_proj_spend_total['Pharm_Old_Price_Effective_Reimb_Proj_EOY']
            pharm_proj_spend_total = pd.merge(pharm_proj_spend_total,pharm_old_price_reimb_actual, on = ['CLIENT','REGION', 'BREAKOUT',
                                                                                        'CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT','BG_FLAG','PHARMACY_TYPE', 'IS_MAC', 'IS_SPECIALTY'] )
        pharm_awp_spend_total = pd.merge(pharm_proj_spend_total, pharm_awp_spend_total, how='left', on=['CLIENT','REGION','BREAKOUT',
                                                                                        'CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT','BG_FLAG','PHARMACY_TYPE', 'IS_MAC', 'IS_SPECIALTY'])
        
        pharm_awp_spend_total = pharm_awp_spend_total[['CLIENT','REGION', 'BREAKOUT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP',
                                            'MEASUREMENT','BG_FLAG','PHARMACY_TYPE','PHARM_QTY',
                                            'PHARM_QTY_PROJ_LAG','PHARM_QTY_PROJ_EOY','PHARM_FULLAWP_ADJ','PHARM_FULLAWP_ADJ_PROJ_LAG',
                                            'PHARM_FULLAWP_ADJ_PROJ_EOY','PHARM_PRICE_REIMB','PHARM_LAG_REIMB', 'PHARM_TARG_INGCOST_ADJ',
                                            'PHARM_TARG_INGCOST_ADJ_PROJ_LAG', 'PHARM_TARG_INGCOST_ADJ_PROJ_EOY','PHARM_TARGET_DISP_FEE',
                                            'PHARM_TARGET_DISP_FEE_PROJ_LAG','PHARM_TARGET_DISP_FEE_PROJ_EOY', 
                                            'Pharm_Old_Price_Effective_Reimb_Proj_EOY','Pharm_Price_Effective_Reimb_Proj']].fillna(0)
        # dispense fees spend performance
        # calculating projections
        lp_data_output_df['DISP_FEE_PROJ_LAG'] = lp_data_output_df['CLAIMS_PROJ_LAG'] * lp_data_output_df['AVG_DISP_FEE']
        lp_data_output_df['DISP_FEE_PROJ_EOY'] = lp_data_output_df['CLAIMS_PROJ_EOY'] * lp_data_output_df['AVG_DISP_FEE']
        lp_data_output_df['TARGET_DISP_FEE_PROJ_LAG'] = lp_data_output_df['CLAIMS_PROJ_LAG'] * lp_data_output_df['AVG_TARGET_DISP_FEE']
        lp_data_output_df['TARGET_DISP_FEE_PROJ_EOY'] = lp_data_output_df['CLAIMS_PROJ_EOY'] * lp_data_output_df['AVG_TARGET_DISP_FEE']
        lp_data_output_df['PHARM_DISP_FEE_PROJ_LAG'] = lp_data_output_df['PHARM_CLAIMS_PROJ_LAG'] * lp_data_output_df['PHARM_AVG_DISP_FEE']
        lp_data_output_df['PHARM_DISP_FEE_PROJ_EOY'] = lp_data_output_df['PHARM_CLAIMS_PROJ_EOY'] * lp_data_output_df['PHARM_AVG_DISP_FEE']
        lp_data_output_df['PHARM_TARGET_DISP_FEE_PROJ_LAG'] = lp_data_output_df['PHARM_CLAIMS_PROJ_LAG'] * lp_data_output_df['PHARM_AVG_TARGET_DISP_FEE']
        lp_data_output_df['PHARM_TARGET_DISP_FEE_PROJ_EOY'] = lp_data_output_df['PHARM_CLAIMS_PROJ_EOY'] * lp_data_output_df['PHARM_AVG_TARGET_DISP_FEE']
        
        disp_fee_total = (lp_data_output_df.groupby(['CLIENT','REGION', 'BREAKOUT', 'CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT',
                                                                            'BG_FLAG', 'PHARMACY_TYPE'])[['CLAIMS','CLAIMS_PROJ_LAG','CLAIMS_PROJ_EOY',
                                                                                                            'DISP_FEE','TARGET_DISP_FEE',
                                                                                                            'DISP_FEE_PROJ_LAG','TARGET_DISP_FEE_PROJ_LAG',
                                                                                                            'DISP_FEE_PROJ_EOY','TARGET_DISP_FEE_PROJ_EOY',
                                                                                                            'PHARM_CLAIMS','PHARM_CLAIMS_PROJ_LAG','PHARM_CLAIMS_PROJ_EOY',
                                                                                                            'PHARMACY_DISP_FEE','PHARM_TARGET_DISP_FEE',
                                                                                                            'PHARM_DISP_FEE_PROJ_LAG','PHARM_TARGET_DISP_FEE_PROJ_LAG',
                                                                                                            'PHARM_DISP_FEE_PROJ_EOY','PHARM_TARGET_DISP_FEE_PROJ_EOY'
                                                                                                            ]].sum()).reset_index()

        awp_spend_total = pd.merge(awp_spend_total, comp_score, how='left', on=['CLIENT','REGION','BREAKOUT','CHAIN_GROUP','CHAIN_SUBGROUP',
                                                                            'MEASUREMENT','BG_FLAG','PHARMACY_TYPE', 'IS_MAC', 'IS_SPECIALTY'])
        
    else:
        awp_spend_total = (lp_data_output_df.groupby(['CLIENT','REGION', 'BREAKOUT', 'CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT','BG_FLAG','PHARMACY_TYPE', 'IS_MAC', 'IS_SPECIALTY'])
                            [['FULLAWP_ADJ','FULLAWP_ADJ_PROJ_LAG','FULLAWP_ADJ_PROJ_EOY','PRICE_REIMB','LAG_REIMB','Old_Price_Effective_Reimb_Proj_EOY',
                            'Price_Effective_Reimb_Proj','TARG_INGCOST_ADJ_PROJ_EOY', 'TARG_INGCOST_ADJ','TARG_INGCOST_ADJ_PROJ_LAG','DISP_FEE',
                                'DISP_FEE_PROJ_LAG','DISP_FEE_PROJ_EOY','TARGET_DISP_FEE','TARGET_DISP_FEE_PROJ_LAG','TARGET_DISP_FEE_PROJ_EOY'
                                ]].sum()).reset_index()
        awp_spend_total = pd.merge(awp_spend_total, comp_score, how='left', on=['CLIENT','REGION','BREAKOUT','CHAIN_GROUP','CHAIN_SUBGROUP',
                                                                                'MEASUREMENT','BG_FLAG','PHARMACY_TYPE', 'IS_MAC', 'IS_SPECIALTY'])
        if p.UNC_OPT:
            del awp_spend_total['Old_Price_Effective_Reimb_Proj_EOY']
            awp_spend_total = pd.merge(awp_spend_total,old_price_reimb_actual, on = ['CLIENT','REGION', 'BREAKOUT',
                                                                                        'CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT', 'BG_FLAG','PHARMACY_TYPE', 'IS_MAC', 'IS_SPECIALTY'] )
            # pharmacy spend performance
        pharm_awp_spend_total = lp_data_output_df.groupby(['CLIENT','REGION', 'BREAKOUT', 'CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT',
                                                                                'BG_FLAG', 'PHARMACY_TYPE', 'IS_MAC', 'IS_SPECIALTY'], dropna=False)[['PHARM_QTY',
                                                                                            'PHARM_QTY_PROJ_LAG','PHARM_QTY_PROJ_EOY','PHARM_FULLAWP_ADJ',
                                                                                            'PHARM_FULLAWP_ADJ_PROJ_LAG','PHARM_FULLAWP_ADJ_PROJ_EOY',
                                                                                                            'PHARM_TARG_INGCOST_ADJ',
                                                                                            'PHARM_TARG_INGCOST_ADJ_PROJ_LAG','PHARM_TARG_INGCOST_ADJ_PROJ_EOY',
                                                                                                            'PHARM_PRICE_REIMB','PHARM_LAG_REIMB',
                                                                                            'PHARM_TARGET_DISP_FEE', 'PHARM_TARGET_DISP_FEE_PROJ_LAG','PHARM_TARGET_DISP_FEE_PROJ_EOY',
                                                                                                            'Pharm_Old_Price_Effective_Reimb_Proj_EOY',
                                                                                            'Pharm_Price_Effective_Reimb_Proj','PHARMACY_DISP_FEE']].sum().reset_index()
        if p.UNC_OPT:
            del pharm_awp_spend_total['Pharm_Old_Price_Effective_Reimb_Proj_EOY']
            pharm_awp_spend_total = pd.merge(pharm_awp_spend_total,pharm_old_price_reimb_actual, on = ['CLIENT','REGION', 'BREAKOUT',
                                                                                                        'CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT','BG_FLAG','PHARMACY_TYPE','IS_MAC','IS_SPECIALTY'] )
        
        # dispense fees spend performance
        # calculating projections
        lp_data_output_df['DISP_FEE_PROJ_LAG'] = lp_data_output_df['CLAIMS_PROJ_LAG'] * lp_data_output_df['AVG_DISP_FEE']
        lp_data_output_df['DISP_FEE_PROJ_EOY'] = lp_data_output_df['CLAIMS_PROJ_EOY'] * lp_data_output_df['AVG_DISP_FEE']
        lp_data_output_df['TARGET_DISP_FEE_PROJ_LAG'] = lp_data_output_df['CLAIMS_PROJ_LAG'] * lp_data_output_df['AVG_TARGET_DISP_FEE']
        lp_data_output_df['TARGET_DISP_FEE_PROJ_EOY'] = lp_data_output_df['CLAIMS_PROJ_EOY'] * lp_data_output_df['AVG_TARGET_DISP_FEE']
        lp_data_output_df['PHARM_DISP_FEE_PROJ_LAG'] = lp_data_output_df['PHARM_CLAIMS_PROJ_LAG'] * lp_data_output_df['PHARM_AVG_DISP_FEE']
        lp_data_output_df['PHARM_DISP_FEE_PROJ_EOY'] = lp_data_output_df['PHARM_CLAIMS_PROJ_EOY'] * lp_data_output_df['PHARM_AVG_DISP_FEE']
        lp_data_output_df['PHARM_TARGET_DISP_FEE_PROJ_LAG'] = lp_data_output_df['PHARM_CLAIMS_PROJ_LAG'] * lp_data_output_df['PHARM_AVG_TARGET_DISP_FEE']
        lp_data_output_df['PHARM_TARGET_DISP_FEE_PROJ_EOY'] = lp_data_output_df['PHARM_CLAIMS_PROJ_EOY'] * lp_data_output_df['PHARM_AVG_TARGET_DISP_FEE']
        
        disp_fee_total = (lp_data_output_df.groupby(['CLIENT','REGION', 'BREAKOUT', 'CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT','BG_FLAG',
                                                                                            'PHARMACY_TYPE'])[['CLAIMS','CLAIMS_PROJ_LAG','CLAIMS_PROJ_EOY',
                                                                                                            'DISP_FEE','TARGET_DISP_FEE',
                                                                                                            'DISP_FEE_PROJ_LAG','TARGET_DISP_FEE_PROJ_LAG',
                                                                                                            'DISP_FEE_PROJ_EOY','TARGET_DISP_FEE_PROJ_EOY',
                                                                                                            'PHARM_CLAIMS','PHARM_CLAIMS_PROJ_LAG','PHARM_CLAIMS_PROJ_EOY',
                                                                                                            'PHARMACY_DISP_FEE','PHARM_TARGET_DISP_FEE',
                                                                                                            'PHARM_DISP_FEE_PROJ_LAG','PHARM_TARGET_DISP_FEE_PROJ_LAG',
                                                                                                            'PHARM_DISP_FEE_PROJ_EOY','PHARM_TARGET_DISP_FEE_PROJ_EOY'
                                                                                                            ]].sum()).reset_index()
    # For Non UNC Clients: (WRITE_TO_BQ = True and unc_flag = False and UNC_ADJUST=False, hence below condition will evaluate to True
    # For UNC Clients:
    #   1st Iteration: (WRITE_TO_BQ = True) and unc_flag = False and UNC_ADJUST=True, hence below condition will evaluate to False
    #   2nd Iteration: (WRITE_TO_BQ = True) and unc_flag = True and UNC_ADJUST=True, hence below condition will evaluate to True
    if p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False):
        uf.write_to_bq(
            awp_spend_total,
            project_output = p.BQ_OUTPUT_PROJECT_ID,
            dataset_output = p.BQ_OUTPUT_DATASET,
            table_id = "awp_spend_total_medd_comm_subgroup",
            client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
            timestamp_param = p.TIMESTAMP,
            run_id = p.AT_RUN_ID,
            schema = None # TODO: create schema
        )
        uf.write_to_bq(
            pharm_awp_spend_total,
            project_output = p.BQ_OUTPUT_PROJECT_ID,
            dataset_output = p.BQ_OUTPUT_DATASET,
            table_id = "pharm_awp_spend_total_medd_comm_subgroup",
            client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
            timestamp_param = p.TIMESTAMP,
            run_id = p.AT_RUN_ID,
            schema = None # TODO: create schema
        )
        uf.write_to_bq(
            disp_fee_total,
            project_output = p.BQ_OUTPUT_PROJECT_ID,
            dataset_output = p.BQ_OUTPUT_DATASET,
            table_id = "disp_fee_total_medd_comm_subgroup",
            client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
            timestamp_param = p.TIMESTAMP,
            run_id = p.AT_RUN_ID,
            schema = None # TODO: create schema
        )
    else:
        awp_spend_total.to_csv(os.path.join(p.FILE_OUTPUT_PATH, "awp_spend_total_" + p.DATA_ID + ".csv"), index=False)
        pharm_awp_spend_total.to_csv(os.path.join(p.FILE_OUTPUT_PATH, "pharm_awp_spend_total_" + p.DATA_ID + ".csv"), index=False)
        disp_fee_total.to_csv(os.path.join(p.FILE_OUTPUT_PATH, "disp_fee_total_" + p.DATA_ID + ".csv"), index=False)
        
    return lp_data_output_df

def read_tru_mac_list_prices():
    """
    Reads TRU and MAC price lists from a BigQuery dataset and performs data manipulation and merging.
    Returns:
        trumac_prices_final (pandas.DataFrame): Unified TRU price list with merged MAC prices.
    Logic:
    1. Reads TRU and MAC price lists from a BigQuery dataset.
    2. Gets a set of MAC_LISTs present in both TRU and MAC price lists.
    3. Filters TRU and MAC price lists to include only common MAC_LISTs.
    4. Merges NDC-Level MAC pricing for NDC-Level TRU prices in common MAC_LISTs.
    5. Merges GPI-Level MAC pricing for GPI-Level TRU prices in common MAC_LISTs.
    6. Creates a unified TRU price list by merging NDC-Level and GPI-Level prices.
    7. Performs quality assurance checks to ensure the integrity of the TRU price list.
    8. Prints a warning message if there are GPIs present in the MAC price list but not in the TRU price list.
    9. Separates MAC_LISTs that only exist in the MAC price list.
    10. Separates MAC_LISTs that only exist in the TRU price list.
    11. Combines all the price lists into a final unified TRU price list.
    12. Performs a quality assurance check to ensure there are no duplicate rows in the final price list.
    13. Returns the final unified TRU price list.
    """
    
    import CPMO_parameters as p
    from itertools import product
    
    tru_list_prices = uf.read_BQ_data(
            BQ.mac_list_tru, 
            project_id=p.BQ_INPUT_PROJECT_ID,
            dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
            table_id = 'tru_price_list'+p.WS_SUFFIX, 
            customer = ', '.join(sorted(p.CUSTOMER_ID)),  
            mac = True,
            vcml_ref_table_id='vcml_reference'+p.WS_SUFFIX)

    tru_list_prices = (
        tru_list_prices.melt(
            id_vars=['MAC', 'GPI', 'NDC', 'MAC_LIST', 'IS_MAC'], 
            value_vars=['GENERIC_PRICE', 'BRAND_PRICE'], 
            var_name='BG_FLAG', 
            value_name='PRICE'
        )
        .assign(BG_FLAG=lambda x: x['BG_FLAG'].map({'GENERIC_PRICE': 'G', 'BRAND_PRICE': 'B'}))
        .query('PRICE > 0')
    )
    tru_list_prices.loc[tru_list_prices['BG_FLAG'] == 'B', 'IS_MAC'] = False
         
    mac_list_prices = uf.read_BQ_data(
            BQ.mac_list, 
            project_id=p.BQ_INPUT_PROJECT_ID,
            dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
            table_id = 'mac_list'+p.WS_SUFFIX, 
            customer = ', '.join(sorted(p.CUSTOMER_ID)),
            mac = True,
            vcml_ref_table_id='vcml_reference'+p.WS_SUFFIX)
    
    # Separate brand prices from tru_list_prices
    tru_list_prices_brand = tru_list_prices[tru_list_prices['BG_FLAG'] == 'B']
    tru_list_prices = tru_list_prices[tru_list_prices['BG_FLAG'] == 'G']
    
    # Getting a set of MAC_LISTs present in both TRU and MAC price lists
    TRUs = tru_list_prices['MAC_LIST'].unique().tolist()
    MACs = mac_list_prices['MAC_LIST'].unique().tolist()
    TRUMACs = list(set(TRUs).intersection(set(MACs)))
    tru_prices_common = tru_list_prices[tru_list_prices['MAC_LIST'].isin(TRUMACs)].sort_values(['MAC','GPI','NDC','BG_FLAG','MAC_LIST','IS_MAC']).reset_index().drop(columns= ['index'])
    mac_prices_common = mac_list_prices[mac_list_prices['MAC_LIST'].isin(TRUMACs)]


    
    # Getting NDC - Level MAC pricing for NDC - Level TRU prices in common MAC_LISTs
    tru_prices_common_ndc = tru_prices_common[tru_prices_common['NDC']!='***********']
    mac_prices_common_ndc = mac_prices_common[mac_prices_common['NDC']!='***********']
    tru_prices_common_ndc = tru_prices_common_ndc.merge(mac_prices_common_ndc[['MAC_LIST','GPI','NDC','BG_FLAG','PRICE']],
                            on = ['MAC_LIST','GPI','NDC','BG_FLAG'],
                           how = 'left',
                           suffixes = ('','_NDC_MAC'))
    tru_prices_common_ndc['PRICE'] = tru_prices_common_ndc[['PRICE','PRICE_NDC_MAC']].min(axis=1)
    tru_prices_common_ndc.drop(['PRICE_NDC_MAC'],axis=1,inplace=True)

    # Getting GPI-Level MAC pricing for GPI-Level TRU prices in common MAC_LISTs
    tru_prices_common_gpi = tru_prices_common[tru_prices_common['NDC']=='***********']

    ## transforming MAC Lists to GPI-Level pricing
    gpi_mac_list_prices = mac_prices_common[mac_prices_common['NDC'] == '***********']
    ndc_mac_list_prices = mac_prices_common[mac_prices_common['NDC'] != '***********']

    ndc_mac_list_prices = ndc_mac_list_prices.groupby(['MAC','GPI','MAC_LIST','IS_MAC','BG_FLAG'])['PRICE'].min().reset_index()
    ndc_mac_list_prices['NDC'] = '***********'

    mac_prices_common_gpi = pd.concat([gpi_mac_list_prices,ndc_mac_list_prices]).drop_duplicates(subset = gpi_mac_list_prices.columns.difference(['PRICE']).tolist(), keep='first')

    ## creating a unified TRU price list
    trumac_prices_common_gpi = tru_prices_common_gpi.merge(mac_prices_common[['MAC_LIST','GPI','NDC','BG_FLAG','PRICE']],
                                on = ['MAC_LIST','GPI','NDC','BG_FLAG'],
                               how = 'left',
                               suffixes = ('','_NDC_GPI'))
    trumac_prices_common_gpi['PRICE'] = trumac_prices_common_gpi[['PRICE','PRICE_NDC_GPI','BG_FLAG']].min(axis=1)
    trumac_prices_common_gpi.drop(['PRICE_NDC_GPI'],axis=1,inplace=True)
    cols = tru_prices_common.columns
    trumac_prices_common = pd.concat([tru_prices_common_ndc,tru_prices_common_gpi])[cols].sort_values(['MAC','GPI','NDC','BG_FLAG','MAC_LIST','IS_MAC']).reset_index().drop(columns= ['index'])

    #QA - to check if the TRU price list structure is preserved
    assert trumac_prices_common[['MAC','GPI','NDC','BG_FLAG','MAC_LIST','IS_MAC']].equals(tru_prices_common[['MAC','GPI','NDC','BG_FLAG','MAC_LIST','IS_MAC']]), 'TRU price list structure was changed. Check Code.'
    #QA - to check if any GPIs exist for which we have a MAC price but not a TRU price in the common MAC_LISTs
    mac_gpi = set((mac_prices_common['MAC_LIST']+'_'+mac_prices_common['GPI']).tolist())
    tru_gpi = set((tru_prices_common['MAC_LIST']+'_'+tru_prices_common['GPI']).tolist())
    print('WARNING: Some GPIs are not present in TRU price list but exist in MAC price list. These are - {}'.format(mac_gpi - tru_gpi))

    # Getting MAC_LISTs which only exist as MAC price list
    mac_prices_diff = mac_list_prices[~mac_list_prices['MAC_LIST'].isin(TRUMACs)]
    mac_prices_diff.loc[:]['MAC'] = p.APPLY_VCML_PREFIX + mac_prices_diff['MAC'].str[3:].copy()

    # Getting MAC_LISTs which only exist as TRU price list
    tru_prices_diff = tru_list_prices[~tru_list_prices['MAC_LIST'].isin(TRUMACs)]

    # combining all the price lists
    if p.GENERIC_OPT and p.BRAND_OPT:
        trumac_prices_final = pd.concat([trumac_prices_common,mac_prices_diff,tru_prices_diff,tru_list_prices_brand])
    elif p.GENERIC_OPT and not p.BRAND_OPT:
        trumac_prices_final = pd.concat([trumac_prices_common,mac_prices_diff,tru_prices_diff])
    elif not p.GENERIC_OPT and p.BRAND_OPT:
        trumac_prices_final = tru_list_prices_brand.copy()
        
    ##### Fill any Non-MAC/Non-TRU price points with 0 #####
    trumac_prices_final['IDX'] = trumac_prices_final['GPI'] + '_' + trumac_prices_final['NDC'] + '_' + trumac_prices_final['BG_FLAG']

    # Create all combos of GPI/NDC/BG Flag and MAC
    trumac_prices_fill = pd.DataFrame(list(product(trumac_prices_final['IDX'].unique(), trumac_prices_final['MAC'].unique())), columns=['IDX', 'MAC'])
    trumac_prices_fill[['GPI', 'NDC', 'BG_FLAG']] = trumac_prices_fill['IDX'].str.split('_', expand=True, n=3)

    # Get price if exists
    trumac_prices_fill = trumac_prices_fill.merge(trumac_prices_final[['IDX', 'MAC', 'IS_MAC', 'PRICE']], how='left', on=['IDX', 'MAC'], validate='1:1')
    trumac_prices_fill = trumac_prices_fill.merge(trumac_prices_final.groupby('IDX').agg({'IS_MAC':'max'}).reset_index().rename(columns={'IS_MAC':'IS_MAC_FILL'}), 
                          how='left', on=['IDX'], validate='m:1')

    # Determine if price already exists in MAC/TRU list
    unique_mac_gpi = trumac_prices_final.groupby(['MAC', 'GPI', 'BG_FLAG']).agg({'PRICE':'first'}).reset_index()
    unique_mac_gpi['PRICE_EXISTS'] = True
    trumac_prices_fill = trumac_prices_fill.merge(unique_mac_gpi[['MAC', 'GPI', 'BG_FLAG', 'PRICE_EXISTS']], how='left', on=['GPI', 'MAC', 'BG_FLAG'], validate='m:1')

    # Fill nulls
    trumac_prices_fill['PRICE_EXISTS'] = trumac_prices_fill['PRICE_EXISTS'].fillna(False)
    trumac_prices_fill['IS_MAC'] = trumac_prices_fill['IS_MAC'].fillna(trumac_prices_fill['IS_MAC_FILL'])
    trumac_prices_fill['MAC_LIST'] = trumac_prices_fill['MAC'].str[3:]
    trumac_prices_fill.loc[(~trumac_prices_fill['PRICE_EXISTS']), 'PRICE'] = trumac_prices_fill.loc[(~trumac_prices_fill['PRICE_EXISTS']), 'PRICE'].fillna(0)
    trumac_prices_fill = trumac_prices_fill[~trumac_prices_fill['PRICE'].isna()]
    
    trumac_prices_final = trumac_prices_fill[['MAC', 'GPI', 'NDC', 'MAC_LIST', 'IS_MAC', 'BG_FLAG', 'PRICE']].copy()
    ##### End fill #####
    
    #QA - check for duplicate rows
    assert len(trumac_prices_final) == len(trumac_prices_final.drop_duplicates(subset=trumac_prices_final.columns.difference(['PRICE']))), "Duplicate pricing rows found, check the code."
    
    return trumac_prices_final