# -*- coding: utf-8 -*-
"""CPMO_goodrx_functions

This module contains the functions that are used to create and calculate the
goodrx logic and caculations for the upper limits.

This script requires pandas and numpy to be installed within the Python
environment you are running these functions in

This module contains the following functions:
      *
"""

# Import required parameters file and packages
import CPMO_parameters as p
import pandas as pd
import numpy as np
import os

# Temporary function structure and required inputs and outputs 
"""
CPMO_goodrx_functions: Create functions required for the data gathering and preprocessing steps 
involved with the logic to create the upper limit goodrx prices per client, pharmacy, GPI level

Raw goodrx data is scraped from the wed and pulled into the LP codebase by flat file 
produced by this script and its respective functi ons

"""


def goodrx_data_preprocess(lp_data_df, cleaned_goodrx_df, merge_method='forward', qty_method='median'):
    '''
    Preprocess the scraped goodrx data that is scraped from the web and pulled from Teradata,
    Inputs:
        cleaned_goodrx_df - dataframe consisting of the raw good rx information used to create the goodrx price bounds
        lp_data_df - dataframe with pricing information: Client, Breakout, Region, Pharmacy, GPI-NDC, MAC Price, Pricing Bounds, etc.
        merge_method - method in which to merge GoodRx prices to Claims data for price selection: 
            'forward' will find exact QTY match or the next highest QTY of GoodRx, biasing towards lower GoodRx price
            'lowest' will simply use the lowest priced GoodRx price (often a high quantity bulk discount)
            'nearest' will find the nearest QTY match between GoodRx and the Claims data
        qty_average - method of choosing the 'average' QTY dispensed in claims, which is then merged given the merge_method 
            'median' uses the median QTY dispensed, and is the default
            'mean' uses the mean QTY dispensed
    Output:
        lp_data_goodrx_df - new dataframe containing the preprocessed goodrx data attached to the original daily totals dataframe
    '''
    # Clean Daily Totals input file 
    assert merge_method in ['forward', 'lowest', 'nearest'], 'Not a valid merge_method.'
    assert qty_method in ['mean', 'median'], 'Not a valid qty_average type.'
    
    lp_data_df = lp_data_df[lp_data_df.CHAIN_GROUP != 'MCHOICE']
    lp_data_df['CHAIN_GROUP'] = goodrx_pharmacy_map(lp_data_df['CHAIN_GROUP'])
    #lp_data_df['CHAIN_NAME'] = lp_data_df['CHAIN_GROUP']
    #lp_data_df.loc[~(lp_data_df['CHAIN_NAME'].isin(['CVS','WAG','WMT','RAD','KRG'])),'CHAIN_NAME'] = 'IND'
    
    # Create CLIENT BILL AMOUNT for GoodRx logic 
    lp_data_df['CLIENT_BILL_AMOUNT'] = ((lp_data_df['SPEND'] + lp_data_df['DISP_FEE'])
                                        - lp_data_df['MEMBER_COST']).apply(lambda x: max(x, 0))
    
    # Group by GPI and CHAIN GROUP and average the qty, memebr cost, spend, and disp fee 
    grouped_multiple = lp_data_df.groupby(['GPI', 'CHAIN_GROUP']).agg(
                                   TOTAL_CLAIMS = ('CLAIMS','sum'), 
                                   TOTAL_DISP_QTY = ('QTY','sum'), 
                                   TOTAL_SPEND = ('SPEND','sum'), 
                                   TOTAL_MEMBER_COST_AMOUNT = ('MEMBER_COST','sum'), 
                                   TOTAL_CLIENT_BILL_AMOUNT = ('CLIENT_BILL_AMOUNT','sum'),
                                   TOTAL_DISP_FEE = ('DISP_FEE','sum'),
                                   DSPNS_UNT_QTY_MEDIAN = ('QTY','median'))

    grouped_multiple = grouped_multiple.reset_index()

    # Create Per Claim quantities after group by 
    grouped_multiple['DSPNS_UNT_QTY_MEAN'] = grouped_multiple['TOTAL_DISP_QTY'] / grouped_multiple['TOTAL_CLAIMS']
    grouped_multiple['SPEND_PER_CLAIM'] = grouped_multiple['TOTAL_SPEND'] / grouped_multiple['TOTAL_CLAIMS']
    grouped_multiple['MBR_TOT_COST_AMT'] = grouped_multiple['TOTAL_MEMBER_COST_AMOUNT'] / grouped_multiple['TOTAL_CLAIMS']
    grouped_multiple['CLNT_BILL_AMT'] = grouped_multiple['TOTAL_CLIENT_BILL_AMOUNT'] / grouped_multiple['TOTAL_CLAIMS']
    grouped_multiple['CLNT_DSPNS_FEE_AMT'] = grouped_multiple['TOTAL_DISP_FEE'] / grouped_multiple['TOTAL_CLAIMS']
    # Create unit cost amount for goodrx logic 
    grouped_multiple['UNT_CST'] = grouped_multiple['TOTAL_SPEND'] / grouped_multiple['TOTAL_DISP_QTY']

    # This solves the situation when there are no claims.
    grouped_multiple.loc[grouped_multiple['DSPNS_UNT_QTY_MEAN'].isna(), 'DSPNS_UNT_QTY_MEAN'] = 0
    grouped_multiple.loc[grouped_multiple['SPEND_PER_CLAIM'].isna(), 'SPEND_PER_CLAIM'] = 0
    grouped_multiple.loc[grouped_multiple['MBR_TOT_COST_AMT'].isna(), 'MBR_TOT_COST_AMT'] = 0
    grouped_multiple.loc[grouped_multiple['CLNT_BILL_AMT'].isna(), 'CLNT_BILL_AMT'] = 0
    grouped_multiple.loc[grouped_multiple['CLNT_DSPNS_FEE_AMT'].isna(), 'CLNT_DSPNS_FEE_AMT'] = 0
    grouped_multiple.loc[grouped_multiple['UNT_CST'].isna(), 'UNT_CST'] = 0

    # PREPROCESS THE GOODRX PRICES ----------------
    # ---------------------------------------------
    cleaned_goodrx_df.columns = cleaned_goodrx_df.columns.str.upper()
    
    if 'CHAIN_NAME' in cleaned_goodrx_df.columns:
        cleaned_goodrx_df.rename(columns={'CHAIN_NAME':'CHAIN_GROUP'}, inplace=True)
        
    # Rename some columns. I left these columns named as they were in the pre-preprocessing function below to
    # allow future users to better track what these variables mean.
    cleaned_goodrx_df.rename(columns={'GRX_UNIT_PRICE':'GRX_SAME_PHMCY',
                                      'GRX_UNIT_PRICE_LOWEST_NONIND':'GRX_CHAIN_LOW'}, inplace=True)
    
    # Select columns needed from SJs file and clean dataframe
    cleaned_goodrx_df = cleaned_goodrx_df[['GPI_CD','CHAIN_GROUP','GRX_SAME_PHMCY',
                                           'GRX_CHAIN_LOW', 'QTY']] # removed: 'GRX_CHAIN_LOW_NAME', may want to add back for debugging, it's informative but not needed or used
    
    # Map pharmacies
    cleaned_goodrx_df['CHAIN_GROUP'] = goodrx_pharmacy_map(cleaned_goodrx_df['CHAIN_GROUP'])
    #cleaned_goodrx_df['GRX_CHAIN_LOW_NAME'] = goodrx_pharmacy_map(cleaned_goodrx_df['GRX_CHAIN_LOW_NAME'])

    # Remove duplicated GPI and CHAIN GROUPS by aggregating the Independents and selecting min goodrx prices
    cleaned_goodrx_agg = cleaned_goodrx_df.groupby(['GPI_CD', 'CHAIN_GROUP']) \
        .agg({'GRX_SAME_PHMCY': np.nanmin, 'GRX_CHAIN_LOW': np.nanmin}) \
        .reset_index()
    lp_data_goodrx_low_df = pd.merge(grouped_multiple, cleaned_goodrx_agg,
                                       how='inner',
                                       left_on=['GPI', 'CHAIN_GROUP'],
                                       right_on=['GPI_CD', 'CHAIN_GROUP'])

    # Merge the preprocessed claims with the cleaned goodrx dataframe to create the input df for the goodrx logic function
    if merge_method in ['forward', 'nearest']:
        cleaned_goodrx_agg = cleaned_goodrx_df.copy()
        qty_col = None
        if qty_method == 'mean':
            qty_col = 'DSPNS_UNT_QTY_MEAN'
        elif qty_method == 'median':
            qty_col = 'DSPNS_UNT_QTY_MEDIAN'
        grouped_multiple = grouped_multiple[pd.notna(grouped_multiple[['GPI', 'CHAIN_GROUP', qty_col]]).all(axis=1)]
            
        cleaned_goodrx_agg = cleaned_goodrx_agg[pd.notna(cleaned_goodrx_agg[['GPI_CD', 'CHAIN_GROUP', 'QTY']]).all(axis=1)]
        cleaned_goodrx_agg = cleaned_goodrx_agg.sort_values(by=['QTY'])

        lp_data_goodrx_df = pd.merge_asof(grouped_multiple.sort_values(by=[qty_col]),
                                          cleaned_goodrx_agg,
                                          left_on=qty_col,
                                          right_on='QTY',
                                          direction=merge_method,
                                          left_by=['GPI', 'CHAIN_GROUP'],
                                          right_by=['GPI_CD', 'CHAIN_GROUP'],
                                          allow_exact_matches=True)

        if merge_method == 'forward':
            # if merge_method is forward, do a second merge to catch any matches that were missed
            lp_data_goodrx_df.loc[pd.isna(lp_data_goodrx_df['QTY'])] = \
                pd.merge_asof(lp_data_goodrx_df.loc[pd.isna(lp_data_goodrx_df['QTY'])].sort_values(by=[qty_col]),
                              cleaned_goodrx_agg,
                              left_on=qty_col,
                              right_on='QTY',
                              direction='nearest',
                              left_by=['GPI', 'CHAIN_GROUP'],
                              right_by=['GPI_CD', 'CHAIN_GROUP'],
                              allow_exact_matches=True)
        lp_data_goodrx_df = lp_data_goodrx_df[pd.notna(lp_data_goodrx_df['QTY'])]

        # append any combinations from old merge_method that didn't make it through
        lp_data_goodrx_low_df = lp_data_goodrx_low_df[~lp_data_goodrx_low_df.set_index(['GPI', 'CHAIN_GROUP']).index.isin(lp_data_goodrx_df.set_index(['GPI', 'CHAIN_GROUP']).index)]
        lp_data_goodrx_df = lp_data_goodrx_df.append(lp_data_goodrx_low_df, ignore_index=True)
    else:
        lp_data_goodrx_df = lp_data_goodrx_low_df

    # Clean final dataframe drop and rename columns
    lp_data_goodrx_df = lp_data_goodrx_df.drop(['TOTAL_CLAIMS', 'TOTAL_DISP_QTY', 'TOTAL_SPEND', 
                                                'TOTAL_MEMBER_COST_AMOUNT', 'TOTAL_CLIENT_BILL_AMOUNT', 
                                                'TOTAL_DISP_FEE', 'SPEND_PER_CLAIM', 'GPI_CD'], axis=1)
    if 'QTY' in lp_data_goodrx_df.columns:
        lp_data_goodrx_df = lp_data_goodrx_df.drop(columns=['QTY'])
    
    lp_data_goodrx_df.rename(columns={'GPI': 'GPI_CD'}, inplace=True)
    
    return lp_data_goodrx_df

#Parameterized inputs to set custom rules
def goodrx_rules_setup(param_file=None, verbose=False):
    '''
    This function will simply return the pricing tiers dataframe.
    Future functionality should offer the ability to read in a .csv to set/change the parameters.
    '''
    rules = pd.DataFrame()
    rules['BENCHMARK_TYPE'] = ['SAME']*len(p.MBR_THRESHOLD_TYPE_SAME) + ['CHAIN']*len(p.MBR_THRESHOLD_TYPE_CHAIN)
    rules['LOW'] = [-np.inf] + p.MBR_PRICING_TIER_SAME + [-np.inf] + p.MBR_PRICING_TIER_CHAIN
    rules['UPPER'] = p.MBR_PRICING_TIER_SAME + [np.inf] + p.MBR_PRICING_TIER_CHAIN + [np.inf]
    rules['THRESHOLD_TYPE'] = p.MBR_THRESHOLD_TYPE_SAME + p.MBR_THRESHOLD_TYPE_CHAIN
    rules['CHANGE_THRESHOLD'] = p.MBR_CHANGE_THRESHOLD_SAME + p.MBR_CHANGE_THRESHOLD_CHAIN #Negative values
    rules['GOODRX_COMPETITIVE_MULTIPLIER'] = p.MBR_GOODRX_COMPETITIVE_MULTIPLIER_SAME + p.MBR_GOODRX_COMPETITIVE_MULTIPLIER_CHAIN
    
    if verbose:
        print(rules)
    
    return rules

def goodrx_price_limits(lp_data_goodrx_df, goodrx_tiers):
    '''
    This takes a dataframe row of pricing information and returns the goodrx price upper limit for that entry.
    This function is designed to be used with apply accross a dataframe
    Inputs:
            lp_data_df - dataframe with pricing information: Client, Breakout, Region, Pharmacy, GPI-NDC, MAC Price, Pricing Bounds, etc.
            OOP_cost - member cost sharing for each drug and pharmacy combination      
    Ouputs:
            lp_data_df - dataframe with pricing information: Client, Breakout, Region, Pharmacy, GPI-NDC, MAC Price, Pricing Bounds, etc.
                         including goodrx logic 
    '''
    grx_df = lp_data_goodrx_df.copy()
    rules = goodrx_tiers.copy()
   
    # Restructure the rules dataframe into a flat thing so it can be merged onto other rows
    exploded_rules = pd.merge(rules.assign(key=1),rules.assign(key=1),
                              on='key',
                              suffixes=['.SAME','.CHAIN'])
    exploded_rules = exploded_rules[(exploded_rules['BENCHMARK_TYPE.SAME']=='SAME')
                                    & (exploded_rules['BENCHMARK_TYPE.CHAIN']=='CHAIN')]\
                                    .drop(columns=['key',
                                                   'BENCHMARK_TYPE.SAME',
                                                   'BENCHMARK_TYPE.CHAIN'])

    # Function to add the rules to every row based on that row's MBR_TOT_COST_AMT
    def add_rules_cols(df):
        return df.append(exploded_rules[(exploded_rules['LOW.SAME'] < df['MBR_TOT_COST_AMT'])
                                        & (df['MBR_TOT_COST_AMT'] <= exploded_rules['UPPER.SAME']) 
                                        & (exploded_rules['LOW.CHAIN'] < df['MBR_TOT_COST_AMT']) 
                                        & (df['MBR_TOT_COST_AMT'] <= exploded_rules['UPPER.CHAIN'])]\
                         .reset_index(drop=True).iloc[0])
    
    # Here lies all the logic from SJs workbook. This will very likely benefit from being completely reworked, but if we just dump the right data in now, we get the same thing out that he does so that's neat.
    
    # Add the rules to the dataframe 
    grx_df = grx_df.apply(add_rules_cols,axis=1)

    grx_df['IMPLIED_COINSURANCE'] = grx_df['MBR_TOT_COST_AMT'] / (grx_df['MBR_TOT_COST_AMT'] + grx_df['CLNT_BILL_AMT'])

    grx_df['TGT_ING_COST.SAME'] = grx_df['GOODRX_COMPETITIVE_MULTIPLIER.SAME'] * grx_df['GRX_SAME_PHMCY'] / grx_df['IMPLIED_COINSURANCE'] - grx_df['CLNT_DSPNS_FEE_AMT'] / grx_df['DSPNS_UNT_QTY_MEAN']
    
    grx_df['TGT_ING_COST.CHAIN'] = grx_df['GOODRX_COMPETITIVE_MULTIPLIER.CHAIN'] * grx_df['GRX_CHAIN_LOW'] / grx_df['IMPLIED_COINSURANCE'] - grx_df['CLNT_DSPNS_FEE_AMT'] / grx_df['DSPNS_UNT_QTY_MEAN']
    
    grx_df.loc[(grx_df['TGT_ING_COST.SAME'] <= 0),'TGT_ING_COST.SAME'] = np.nan
    grx_df.loc[(grx_df['TGT_ING_COST.CHAIN'] <= 0),'TGT_ING_COST.CHAIN'] = np.nan

    same_percent_mask = (grx_df['THRESHOLD_TYPE.SAME'] == 'PERCENT')
    chain_percent_mask = (grx_df['THRESHOLD_TYPE.CHAIN'] == 'PERCENT')
    same_dollar_mask = (grx_df['THRESHOLD_TYPE.SAME'] == 'DOLLAR')
    chain_dollar_mask = (grx_df['THRESHOLD_TYPE.CHAIN'] == 'DOLLAR')

    grx_df['MBR_COST_THRESHOLD.SAME'] = np.nan
    grx_df.loc[same_percent_mask,'MBR_COST_THRESHOLD.SAME'] = grx_df[same_percent_mask]['MBR_TOT_COST_AMT'] * (1 + grx_df[same_percent_mask]['CHANGE_THRESHOLD.SAME'] )
    grx_df.loc[same_dollar_mask,'MBR_COST_THRESHOLD.SAME'] = grx_df[same_dollar_mask]['MBR_TOT_COST_AMT'] + grx_df[same_dollar_mask]['CHANGE_THRESHOLD.SAME']
    
    grx_df['MBR_COST_THRESHOLD.CHAIN'] = np.nan
    grx_df.loc[chain_percent_mask,'MBR_COST_THRESHOLD.CHAIN'] = grx_df[chain_percent_mask]['MBR_TOT_COST_AMT'] * (1 + grx_df[chain_percent_mask]['CHANGE_THRESHOLD.CHAIN'] )
    grx_df.loc[chain_dollar_mask,'MBR_COST_THRESHOLD.CHAIN'] = grx_df[chain_dollar_mask]['MBR_TOT_COST_AMT']  + grx_df[chain_dollar_mask]['CHANGE_THRESHOLD.CHAIN']
    
    # Eliminate negative values
    grx_df.loc[(grx_df['MBR_COST_THRESHOLD.SAME'] < 0),'MBR_COST_THRESHOLD.SAME'] = np.nan
    grx_df.loc[(grx_df['MBR_COST_THRESHOLD.CHAIN'] < 0),'MBR_COST_THRESHOLD.CHAIN'] = np.nan
    
    grx_df['ACTUAL_THRESHOLD.SAME'] = (grx_df['MBR_COST_THRESHOLD.SAME'] / grx_df['IMPLIED_COINSURANCE'] - grx_df['CLNT_DSPNS_FEE_AMT']) / grx_df['DSPNS_UNT_QTY_MEAN']
    grx_df['ACTUAL_THRESHOLD.CHAIN'] = (grx_df['MBR_COST_THRESHOLD.CHAIN'] / grx_df['IMPLIED_COINSURANCE'] - grx_df['CLNT_DSPNS_FEE_AMT']) / grx_df['DSPNS_UNT_QTY_MEAN']
    
    # Eliminate negative values
    grx_df.loc[(grx_df['ACTUAL_THRESHOLD.SAME'] < 0),'ACTUAL_THRESHOLD.SAME'] = np.nan
    grx_df.loc[(grx_df['ACTUAL_THRESHOLD.CHAIN'] < 0),'ACTUAL_THRESHOLD.CHAIN'] = np.nan
    
    #If the target cost is greater than threshold set the bound to target cost 
    grx_df.loc[(grx_df['TGT_ING_COST.SAME'] >= grx_df['ACTUAL_THRESHOLD.SAME']),'ADJUSTED_PRICE.SAME'] = grx_df['TGT_ING_COST.SAME']
    grx_df.loc[(grx_df['TGT_ING_COST.CHAIN'] >= grx_df['ACTUAL_THRESHOLD.CHAIN']),'ADJUSTED_PRICE.CHAIN'] = grx_df['TGT_ING_COST.CHAIN']
    
    #If the target cost is less than threshold clip the bound at threshold
    grx_df.loc[(grx_df['TGT_ING_COST.SAME'] < grx_df['ACTUAL_THRESHOLD.SAME']),'ADJUSTED_PRICE.SAME'] = grx_df['ACTUAL_THRESHOLD.SAME']
    grx_df.loc[(grx_df['TGT_ING_COST.CHAIN'] < grx_df['ACTUAL_THRESHOLD.CHAIN']),'ADJUSTED_PRICE.CHAIN'] = grx_df['ACTUAL_THRESHOLD.CHAIN']
    
    # Eliminate negative values [COMMENTED OUT FOR REDUNDANCY]
#     grx_df.loc[(grx_df['TGT_ING_COST.SAME'] < 0),'TGT_ING_COST.SAME'] = np.nan
#     grx_df.loc[(grx_df['TGT_ING_COST.CHAIN'] < 0),'TGT_ING_COST.CHAIN'] = np.nan
    
    grx_df['LOWER'] = grx_df[['ADJUSTED_PRICE.SAME','ADJUSTED_PRICE.CHAIN']]\
                        .apply(lambda x: np.max(np.min(x),0),axis=1)
    grx_df['TO_SUBMIT'] = None
    grx_df.loc[(grx_df['CHAIN_GROUP'] == 'CVS'),'TO_SUBMIT'] \
            = grx_df.loc[(grx_df['CHAIN_GROUP'] == 'CVS'),'LOWER']
    grx_df.loc[~(grx_df['CHAIN_GROUP'] == 'CVS'),'TO_SUBMIT'] \
            = grx_df.loc[~(grx_df['CHAIN_GROUP'] == 'CVS'),'ADJUSTED_PRICE.SAME']

    # Keeping this code here intentionally incase we want to replace GoodRx price with Current Price
# =============================================================================
#     grx_df['DEFAULT'] = grx_df['UNT_CST'] * 1.0 - grx_df['CLNT_DSPNS_FEE_AMT'] / grx_df['DSPNS_UNT_QTY_MEAN']
#     grx_df.loc[~(grx_df['TO_SUBMIT'] < grx_df['DEFAULT']),'TO_SUBMIT'] \
#             = grx_df.loc[~(grx_df['TO_SUBMIT'] < grx_df['DEFAULT']),'DEFAULT']
# =============================================================================
    
    # Create final output file for use in GoodRx clients in an LP run 
    outdf = grx_df[['GPI_CD','CHAIN_GROUP','TO_SUBMIT']]
      
    outdf = outdf.rename(columns={'GPI_CD': 'GPI','TO_SUBMIT':'GOODRX_UPPER_LIMIT'})
    
    return  outdf

def goodrx_grxprice_preprocess(raw_goodrx_df, pharmacy_map=None):
    '''
    This takes in the raw (cleaned) GoodRx prices, those actually scraped and cleaned from the
    wild, and processes them into the form we need to customize them for each client.
    
    Expected Input:
    raw_goodrx_df.columns -> [['gpi_cd','chain_name','CALC_UNIT_CST']]
    
    Output:
    [['GPI','CHAIN_GROUP','GRX_UNIT_PRICE','GRX_UNIT_PRICE_LOWEST_NONIND']]
    
    Where:
    GRX_UNIT_PRICE is the GoodRx price for the given GPI-PHARMACY pair listed
    GRX_UNIT_PRICE_LOWEST_NONIND is the lowest GoodRx price for that GPI found 
                                 within only Non-Independent pharmacies
    
    '''
    raw_goodrx_df.columns = raw_goodrx_df.columns.str.upper()
    
    raw_goodrx_df['CHAIN_GROUP'] = goodrx_pharmacy_map(raw_goodrx_df['CHAIN'], pharmacy_map)
    
    raw_goodrx_df = raw_goodrx_df.rename(columns={'UNIT PRICE':'GRX_UNIT_PRICE'})
    
    outdf = pd.merge(raw_goodrx_df, 
                     raw_goodrx_df[raw_goodrx_df['CHAIN_GROUP']!='IND'].\
                     groupby('GPI_CD')['GRX_UNIT_PRICE'].min(),
                     on=['GPI_CD'], 
                     suffixes=['','_LOWEST_NONIND'])
    
    outdf = outdf[['GPI_CD','CHAIN_GROUP','GRX_UNIT_PRICE','GRX_UNIT_PRICE_LOWEST_NONIND', 'QTY']] \
            .drop_duplicates(subset=['GPI_CD','CHAIN_GROUP','GRX_UNIT_PRICE_LOWEST_NONIND', 'QTY'])
    
    return outdf.sort_values(by=['GPI_CD', 'CHAIN_GROUP', 'QTY'])

def goodrx_interceptor_grxprice_preprocess(raw_goodrx_df, pharmacy_map=None):
    '''
    This takes in the raw (cleaned) GoodRx prices, those actually scraped and cleaned from the
    wild, and processes them into the form we need to customize them for each client.
    
    Expected Input:
    raw_goodrx_df.columns -> [['GPI_CD','NDC','CHAIN_GROUP','UNIT PRICE']]
    
    Output:
    [['GPI','NDC','CHAIN_GROUP','GRX_UNIT_PRICE','GRX_UNIT_PRICE_LOWEST_NONIND']]
    
    Where:
    GRX_UNIT_PRICE is the GoodRx price for the given GPI-NDC-PHARMACY pair listed
    GRX_UNIT_PRICE_LOWEST_NONIND is the lowest GoodRx price for that GPI found 
                                 within only Non-Independent pharmacies
    
    '''
    raw_goodrx_df['CHAIN_GROUP'] = goodrx_pharmacy_map(raw_goodrx_df['CHAIN'], pharmacy_map)
    
    raw_goodrx_df = raw_goodrx_df.rename(columns={'UNIT PRICE':'GRX_UNIT_PRICE','GPI_CD':'GPI'})
    
    outdf = pd.merge(raw_goodrx_df, 
                     raw_goodrx_df[raw_goodrx_df['CHAIN_GROUP']!='IND'].groupby(['GPI','NDC'])['GRX_UNIT_PRICE'].min(),
                     on=['GPI','NDC'], 
                     suffixes=['','_LOWEST_NONIND'])
    
    outdf = outdf.groupby(['GPI','NDC','CHAIN_GROUP']) \
        .agg({'GRX_UNIT_PRICE': np.nanmin, 'GRX_UNIT_PRICE_LOWEST_NONIND': np.nanmin}) \
        .reset_index()
    
    return outdf.sort_values(by=['GPI', 'NDC', 'CHAIN_GROUP'])

def interceptor_vcml_grxprice_preprocess(raw_goodrx_df, pharmacy_map=None):
    '''
    This takes in the raw (cleaned) GoodRx prices, those actually scraped and cleaned from the
    wild, and processes them into the form we need to customize them for each client at the VCML level.
    
    Incase of R30/R90 VCMLs, the median goodrx price is set as the price for R30 VCML and min goodrx price is
    set as price for R90 VCML. 

    For only R30: min goodrx price is set as the R30 price
    
    Expected Input:
    raw_goodrx_df.columns -> [['GPI_CD','CHAIN_GROUP','UNIT PRICE']]
    
    Output:
    [['GPI','MAC_LIST','VCML_GRX_PRICE']]
    
    Where:
    VCML_GRX_PRICE is the per unit GoodRx price for the given GPI-VCML pair listed
    
    '''
    raw_goodrx_df['CHAIN_GROUP'] = interceptor_pharmacy_map(raw_goodrx_df['CHAIN'], pharmacy_map)
    # Add subgroup information
    raw_goodrx_df['CHAIN_SUBGROUP'] = raw_goodrx_df['CHAIN_GROUP']
    raw_goodrx_df_subgroup = raw_goodrx_df[raw_goodrx_df['CHAIN_GROUP']=='CVS'].copy()
    raw_goodrx_df_subgroup['CHAIN_SUBGROUP'] = 'CVSSP'
    raw_goodrx_df = pd.concat([raw_goodrx_df, raw_goodrx_df_subgroup])

    
    raw_goodrx_df = raw_goodrx_df.rename(columns={'UNIT PRICE':'GRX_UNIT_PRICE','GPI_CD':'GPI'})
    
    #MAC MAPPING File from Dynamic Input Folder
    chain_region_mac_mapping = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_MAPPING_FILE)
    chain_region_mac_mapping[['CUSTOMER_ID','REGION','MAC_LIST']] = chain_region_mac_mapping[['CUSTOMER_ID','REGION','MAC_LIST']].astype('string')
    
    vcml_raw_goodrx_df = raw_goodrx_df.merge(chain_region_mac_mapping,how='left', on = 'CHAIN_SUBGROUP')

    vcml_goodrx_df = vcml_raw_goodrx_df.groupby(['GPI','MEASUREMENT','MAC_LIST'])\
                                           .agg(GRX_UNIT_PRICE_MIN = ('GRX_UNIT_PRICE', np.nanmin), 
                                                GRX_UNIT_PRICE_MEDIAN = ('GRX_UNIT_PRICE', np.nanmedian))\
                                           .reset_index()
    
    vcml_goodrx_df['VCML_GRX_PRICE'] = vcml_goodrx_df['GRX_UNIT_PRICE_MIN']
    
    if 'R90' in list(chain_region_mac_mapping.MEASUREMENT.unique()):
        vcml_goodrx_df.loc[vcml_goodrx_df.MEASUREMENT == 'R30', 'VCML_GRX_PRICE'] = vcml_goodrx_df['GRX_UNIT_PRICE_MEDIAN']

    cleaned_df = vcml_goodrx_df[['GPI', 'MAC_LIST', 'VCML_GRX_PRICE']]
    
    return cleaned_df.sort_values(by=['GPI', 'MAC_LIST'])

def goodrx_output_cleanup(output, pharmacy_map=None):
    '''
    This does the final steps of cleaning up the outputs from the GoodRx Pricing.
    Currently: [~output['CHAIN_NAME'].isin(['CVS','WALGREENS','WALMART','RITE AID','KROGER'])),'CHAIN_NAME'] = 'IND'
    '''
    # Transform the 'CHAIN_GROUP' column
    output = goodrx_pharmacy_map(output, pharmacy_map)

    # Select the minimum GPI, CHAIN_GROUP pair from duplicate rows
    output = output.groupby(['GPI','CHAIN_GROUP']).min().reset_index()
    
    return output
    
    
def goodrx_pharmacy_map(x, pharmacy_map=None, verbose=False):
    '''
    This function transforms the pharmacies in a DataFrame or Series, taking them from full text
    to their abbreviated form (e.g. RITE AID -> RAD), then groups everything not in
    [CVS, WAG, WMT, RAD, KRG] into IND.
    
    Examples:
    
    a dataframe with the column 'CHAIN_GROUP' can be fed in directly:
    > df = goodrx_pharmacy_map(df)
    
    or a column of a dataframe (say, with a different name than 'CHAIN_GROUP') can be transformed:
    > df['GRX_CHAIN_LOW_NAME'] = goodrx_pharmacy_map(df['GRX_CHAIN_LOW_NAME'])
    
    '''
    # Define the pharmacy mapping dictionary
    if pharmacy_map is None:
        pharmacy_map = {r'.*CVS.*': 'CVS',
                        r'.*WAG.*': 'WAG', r'.*WALGREENS.*': 'WAG',
                        r'.*RAD.*': 'RAD', r'.*RITE AID.*': 'RAD',
                        r'.*KRG.*': 'KRG', r'.*KROGER.*': 'KRG',
                        r'.*WMT.*': 'WMT', r'.*WALMART.*': 'WMT'
                        }

    # Handle input of DataFrame with column 'CHAIN_GROUP'
    if type(x) is pd.core.frame.DataFrame:
        assert 'CHAIN_GROUP' in x.columns, 'Column \'CHAIN_GROUP\' missing.'

        if verbose:
            print('Pharmacy map applied: ', pharmacy_map)

        # Map the main pharmacies we want to distinquish between
        x['CHAIN_GROUP'] = x['CHAIN_GROUP'].str.upper().replace(pharmacy_map, regex=True)

        # Map all the rest to IND
        if verbose:
            inds = list(x.loc[~(x['CHAIN_GROUP'].isin(['CVS']+list(pharmacy_map.values()))),
                               'CHAIN_GROUP'].unique())
            print('Pharmacies mapped to IND: ',inds)

        x.loc[~(x['CHAIN_GROUP'].isin(['CVS']+list(pharmacy_map.values()))),'CHAIN_GROUP'] = 'IND'

        return x
    
    elif type(x) is pd.core.frame.Series:
        if verbose:
            print('Pharmacy map applied: ', pharmacy_map)

        # Map the main pharmacies we want to distinquish between
        x = x.str.upper().replace(pharmacy_map, regex=True)

        # Map all the rest to IND
        if verbose:
            inds = list(x.loc[~(x.isin(['CVS']+list(pharmacy_map.values())))].unique())
            print('Pharmacies mapped to IND: ',inds)

        x.loc[~(x.isin(['CVS']+list(pharmacy_map.values())))] = 'IND'

        return x
    
def interceptor_pharmacy_map(x, pharmacy_map=None, verbose=False):
    '''
    This function transforms the pharmacies in a DataFrame or Series, taking them from full text
    to their abbreviated form (e.g. RITE AID -> RAD), then groups everything not in the pharmacy mapping 
    into NONPREF_OTH.
    
    Examples:
    
    a dataframe with the column 'CHAIN_GROUP' can be fed in directly:
    > df = goodrx_pharmacy_map(df)
    
    or a column of a dataframe (say, with a different name than 'CHAIN_GROUP') can be transformed:
    > df['GRX_CHAIN_LOW_NAME'] = goodrx_pharmacy_map(df['GRX_CHAIN_LOW_NAME'])
    
    '''
    # Define the pharmacy mapping dictionary
    if pharmacy_map is None:
        pharmacy_map = {r'.*CVS.*': 'CVS',
                        r'.*WAG.*': 'WAG', r'.*WALGREENS.*': 'WAG',
                        r'.*RAD.*': 'RAD', r'.*RITE AID.*': 'RAD',
                        r'.*KRG.*': 'KRG', r'.*KROGER.*': 'KRG',
                        r'.*WMT.*': 'WMT', r'.*WALMART.*': 'WMT',
                        r'.*GIE.*': 'GIE', r'.*GIANT EAGLE.*': 'GIE',
                        r'.*MJR.*': 'MJR', r'.*MEIJER.*': 'MJR',
                        r'.*KIN.*': 'KIN', r'.*KINNEY.*': 'KIN',
                        r'.*ABS.*': 'ABS', r'.*ALBERTSONS.*': 'ABS',
                        r'.*SAFEWAY.*': 'ABS',
                        r'.*CST.*': 'CST', r'.*COSTCO.*': 'CST',
                        r'.*AHD.*': 'AHD', r'.*AHOLD.*': 'AHD',
                        r'.*ARETE.*': 'ART',
                        r'.*ELE.*': 'ELE', r'.*ELEVATE.*': 'ELE',
                        r'.*CAR.*': 'CAR', r'.*CARDINAL.*': 'CAR',
                        r'.*PBX.*': 'PBX', r'.*PUBLIX.*': 'PBX',
                        r'.*EPC.*': 'EPC', r'.*EPIC.*': 'EPC',
                        r'.*HMA.*': 'HMA', r'.*HEALTHMART.*': 'HMA',
                        r'.*TPS.*': 'TPS', r'.*THIRD.*': 'TPS',
                        r'.*GEN.*': 'GEN', r'.*GENOA.*': 'GEN'
                        }

    # Handle input of DataFrame with column 'CHAIN_GROUP'
    if type(x) is pd.core.frame.DataFrame:
        assert 'CHAIN_GROUP' in x.columns, 'Column \'CHAIN_GROUP\' missing.'

        if verbose:
            print('Pharmacy map applied: ', pharmacy_map)

        # Map the main pharmacies we want to distinquish between
        x['CHAIN_GROUP'] = x['CHAIN_GROUP'].str.upper().replace(pharmacy_map, regex=True)

        # Map all the rest to NONPREF_OTH
        if verbose:
            nonpref_oths = list(x.loc[~(x['CHAIN_GROUP'].isin(list(pharmacy_map.values()))),
                               'CHAIN_GROUP'].unique())
            print('Pharmacies mapped to NONPREF_OTH: ',nonpref_oths)

        x.loc[~(x['CHAIN_GROUP'].isin(list(pharmacy_map.values()))),'CHAIN_GROUP'] = 'NONPREF_OTH'

        return x
    
    elif type(x) is pd.core.frame.Series:
        if verbose:
            print('Pharmacy map applied: ', pharmacy_map)

        # Map the main pharmacies we want to distinquish between
        x = x.str.upper().replace(pharmacy_map, regex=True)

        # Map all the rest to NONPREF_OTH
        if verbose:
            nonpref_oths = list(x.loc[~(x.isin(list(pharmacy_map.values())))].unique())
            print('Pharmacies mapped to NONPREF_OTH: ',nonpref_oths)

        x.loc[~(x.isin(list(pharmacy_map.values())))] = 'NONPREF_OTH'

        return x



def goodrx_interceptor_optimization(lp_data_df, check_1026=True, check_cvs_ind=True, check_vcml=True, check_retail_mail=True):
    '''
    The final Interceptor price bounds (low and high) are created. The first pass creates the default price bounds the second pass verifies
    that there are no conflicts.
    For example check to determine if the drug should be indeed push up or down following the MAC1026.
    The final claims df with the bounds is then send to the LP to determine the actual price bounds.
    Input:
        lp_data_df: DataFrame which has the Client, Breakout, Region, Measurement, GPI, NDC, for a given client including
        the MAC1026 price, AWP, MAC price
    Output:
        lp_data_df: New Data frame with the Interceptor low and high bounds.
    '''

    import util_funcs as uf
    import BQ
    from CPMO_shared_functions import standardize_df

    # The GoodRx data is read in
    if p.RAW_GOODRX:
        raw_goodrx_df = pd.read_excel(p.FILE_INPUT_PATH + p.RAW_GOODRX, dtype=p.VARIABLE_TYPE_DIC)
    else:
        if p.READ_FROM_BQ:
            raw_goodrx_df = uf.read_BQ_data(
                project_id=p.BQ_INPUT_PROJECT_ID,
                dataset_id=p.BQ_INPUT_DATASET_ENT_CNFV_PROD,
                table_id='gms_goodrx_drug_price',
                query=BQ.raw_goodrx_custom.format(_project = p.BQ_INPUT_PROJECT_ID,
                                                  _staging_dataset = p.BQ_INPUT_DATASET_ENT_CNFV_PROD),
                custom=True
            )
            raw_goodrx_df = raw_goodrx_df.rename(columns={'PRICE_UNIT_QTY': 'UNIT PRICE'})
    raw_goodrx_df = standardize_df(raw_goodrx_df)

    # Cleaned_goodrx_df has a single GRx price at the: 'MEASUREMENT', 'MAC_LIST', 'GPI' level.
    # It incorporates if there is an R90/R30 price based on the VCML structure
    
    # todo price imputation to guarantee coverage at each VCML-GPI present on the data raw data
    cleaned_goodrx_df = interceptor_vcml_grxprice_preprocess(raw_goodrx_df)
    
    # The query gives out a list of GPIs which have >80% zbd claims
    if p.READ_FROM_BQ:
        zbd_gpi = uf.read_BQ_data(
            query = BQ.zbd_gpi_custom.format(_customer_id = uf.get_formatted_string(p.CUSTOMER_ID), 
                                            _project = p.BQ_INPUT_PROJECT_ID, 
                                            _landing_dataset = p.BQ_INPUT_DATASET_ENT_ENRV_PROD),
            project_id=p.BQ_INPUT_PROJECT_ID,
            dataset_id=p.BQ_INPUT_DATASET_ENT_CNFV_PROD,
            table_id='' + p.WS_SUFFIX,
            custom = True
        )
        
    zbd_gpi = standardize_df(zbd_gpi)
    
    # Include only the ZBD drugs
    cleaned_goodrx_df = pd.merge(cleaned_goodrx_df,zbd_gpi,on = 'GPI', how = 'inner')
    
    # The goodrx prices are merge in to the full table.
    lp_data_df_temp = pd.merge(
        lp_data_df, cleaned_goodrx_df, how='left', on=['GPI', 'MAC_LIST'])
    assert len(lp_data_df_temp) == len(lp_data_df), "len(lp_data_df_temp) == len(lp_data_df)"
    lp_data_df = lp_data_df_temp
    del lp_data_df_temp

    # Flag for whether GoodRx information is available or not
    lp_data_df.loc[:, 'Goodrx_Available'] = 'Available'
    lp_data_df.loc[lp_data_df['VCML_GRX_PRICE'].isnull(), 'Goodrx_Available'] = 'Not_Available'

    # The decision making happens on if the claim should be kept or sent to GoodRx for adjudication
    lp_data_df.loc[:, 'KEEP_SEND'] = 'Keep'
    lp_data_df.loc[:, 'INTERCEPT_LOW'] = 0
    lp_data_df.loc[:, 'INTERCEPT_HIGH'] = 10000*lp_data_df['CURRENT_MAC_PRICE']
    lp_data_df.loc[:, 'GRX_CONFLICT'] = False
    
    # Import the Dispensing Fee information - min_disp_fee set to 25 percentile and max_disp_fee set to 75 percentile
    # The information will include measurement
    
    assert p.READ_FROM_BQ, 'The disp_fee_pct_cust query requires the measurement mapping to be on BQ'
    
    if p.READ_FROM_BQ:
        
        disp_fee_df = uf.read_BQ_data(query=BQ.disp_fee_pct_cust.format(
                                            _customer_id = uf.get_formatted_string(p.CUSTOMER_ID),
                                            _run_id = p.AT_RUN_ID,
                                            _project = p.BQ_INPUT_PROJECT_ID,
                                            _output_project = p.BQ_OUTPUT_PROJECT_ID,
                                            _landing_dataset = p.BQ_INPUT_DATASET_ENT_ENRV_PROD, ###CHECK?
                                            _dataset = p.BQ_INPUT_DATASET_DS_PRO_LP,
                                            _output_dataset = p.BQ_OUTPUT_DATASET,
                                            _time_lag = p.INTERCEPTOR_ZBD_TIME_LAG),
                        project_id = p.BQ_INPUT_PROJECT_ID,
                        dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
                        table_id='' + p.WS_SUFFIX,
                        custom = True)

    disp_fee_vcml_df_agg = standardize_df(disp_fee_df)
    
    assert disp_fee_vcml_df_agg['MAX_DISP_FEE_UNIT'].isna().sum()/disp_fee_vcml_df_agg.shape[0] < 0.1, 'To many null values for the "MAX_DISP_FEE_UNIT" column'
        
    #Add the min and max dispensing fee per unit to lp_data_df
    lp_data_df_temp = lp_data_df.merge(disp_fee_vcml_df_agg, how='left',on = ['GPI','MAC_LIST'])
    assert len(lp_data_df_temp) == len(lp_data_df), "len(lp_data_df_temp) == len(lp_data_df)"
    lp_data_df = lp_data_df_temp
    del lp_data_df_temp
    
    lp_data_df['MAX_DISP_FEE_UNIT'] = lp_data_df['MAX_DISP_FEE_UNIT'].fillna(0)
    lp_data_df['MIN_DISP_FEE_UNIT'] = lp_data_df['MIN_DISP_FEE_UNIT'].fillna(0)
    
    # Obtain a global VCML level pharmacy rate and AWP to compare against.
    lp_data_df_copy = lp_data_df.copy(deep=True)
    lp_data_df_copy['CURR_AWP*CLAIMS'] = lp_data_df_copy['CURR_AWP'] * lp_data_df_copy['CLAIMS']
    pharm_rate_df = lp_data_df_copy.groupby(['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'GPI',
                                       'MAC_LIST']).agg({'PHARMACY_RATE': np.nanmin,
                                                         'CLAIMS': np.nansum,
                                                         'CURR_AWP*CLAIMS': np.nansum}).reset_index()
                                                        
    del lp_data_df_copy
    pharm_rate_df.loc[:, 'VCML_CURR_AWP'] = 0
    pharm_rate_df.loc[pharm_rate_df['CLAIMS'] > 0, 'VCML_CURR_AWP'] = pharm_rate_df['CURR_AWP*CLAIMS']/pharm_rate_df['CLAIMS']
    pharm_rate_df.rename(columns={'PHARMACY_RATE': 'VCML_PHARMACY_RATE'}, inplace=True)
    pharm_rate_df.drop(columns=['CLAIMS', 'CURR_AWP*CLAIMS'], inplace=True)

    lp_data_df_temp = lp_data_df.merge(pharm_rate_df, how='left', on=['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'GPI', 'MAC_LIST'])
    assert len(lp_data_df_temp) == len(lp_data_df), "len(lp_data_df_temp) == len(lp_data_df)"
    lp_data_df = lp_data_df_temp
    del lp_data_df_temp
    lp_data_df.MAC_LIST = lp_data_df.MAC_LIST.astype(str)

#     lp_data_df.loc[(lp_data_df.MAC_LIST.str[-1:]=='1') &\
#                 (lp_data_df.MAC_LIST.str[4:].str.len()==1),'VCML_PHARMACY_RATE'] = np.nanmin([0.883,lp_data_df.loc[(lp_data_df.MAC_LIST.str[-1:]=='1') &\
#                                                                                            (lp_data_df.MAC_LIST.str[4:].str.len()==1),'VCML_PHARMACY_RATE'].values.min()])

#     lp_data_df.loc[(lp_data_df.MAC_LIST.str[-1:]=='3') &\
#                 (lp_data_df.MAC_LIST.str[4:].str.len()==1),'VCML_PHARMACY_RATE'] = np.nanmin([0.883,lp_data_df.loc[(lp_data_df.MAC_LIST.str[-1:]=='3') &\
#                                                                                            (lp_data_df.MAC_LIST.str[4:].str.len()==1),'VCML_PHARMACY_RATE'].values.min()])
                                                 
    # Identify claims to be kept/sent with regards to the Pharmacy GER
    keeper_mask = (lp_data_df['VCML_GRX_PRICE'] > lp_data_df['VCML_CURR_AWP']*(1-lp_data_df['VCML_PHARMACY_RATE'])) & \
                  pd.notna(lp_data_df['VCML_GRX_PRICE']) & (lp_data_df['PRICE_MUTABLE'] != 0)
    sender_mask = (lp_data_df['VCML_GRX_PRICE'] < lp_data_df['VCML_CURR_AWP']*(1-lp_data_df['VCML_PHARMACY_RATE'])) & \
                    pd.notna(lp_data_df['VCML_GRX_PRICE']) & (lp_data_df['PRICE_MUTABLE'] != 0)
    
    # The fee must be agregated at the VCML level such that the mapping happens correctly
    # By taking the higher/lower fee we create a pricing band that we avoid to correctly assign the 'KEEP/SEND' logic.
    # Subtract the dispensing fee to compare the ingredient costs at Caremark and GoodRx
    lp_data_df.loc[sender_mask, 'KEEP_SEND'] = 'Send'
    lp_data_df.loc[sender_mask, 'INTERCEPT_HIGH'] = 1000*lp_data_df['CURRENT_MAC_PRICE']
    lp_data_df.loc[sender_mask, 'INTERCEPT_LOW'] = lp_data_df['VCML_GRX_PRICE'] - lp_data_df['MIN_DISP_FEE_UNIT'] + 0.0001

    lp_data_df.loc[keeper_mask, 'KEEP_SEND'] = 'Keep'
    lp_data_df.loc[keeper_mask, 'INTERCEPT_HIGH'] = lp_data_df['VCML_GRX_PRICE']  - lp_data_df['MAX_DISP_FEE_UNIT']
    lp_data_df.loc[keeper_mask, 'INTERCEPT_LOW'] = 0

    # If the GRx price is lower than the MAC1026 we keep the drug in the house and adjust the bounds.
    # There is no situation in which this conditions can lead to a claims issue.
    if check_1026:
        # In this situation this GPIs are not bound to the competitiveness of GRx.  Further analyisis might be needed if something more complicated has to happen.
        mask_1026 = ((lp_data_df['MAC1026_UNIT_PRICE'] > lp_data_df['INTERCEPT_HIGH']) & (pd.notna(lp_data_df['INTERCEPT_HIGH'])) & (lp_data_df['KEEP_SEND'] == 'Keep') & (lp_data_df['PRICE_MUTABLE'] != 0))
        lp_data_df.loc[mask_1026, 'KEEP_SEND'] = 'Send'
        lp_data_df.loc[mask_1026, 'INTERCEPT_HIGH'] = 1000*lp_data_df['CURRENT_MAC_PRICE']
        lp_data_df.loc[mask_1026, 'INTERCEPT_LOW'] = lp_data_df['MAC1026_UNIT_PRICE']
        lp_data_df.loc[mask_1026, 'GRX_CONFLICT'] = True

        mask_1026 = ((lp_data_df['MAC1026_UNIT_PRICE'] > lp_data_df['INTERCEPT_HIGH']) & (pd.notna(lp_data_df['INTERCEPT_HIGH'])) & (lp_data_df['KEEP_SEND'] == 'Send') & (lp_data_df['PRICE_MUTABLE'] != 0))
        lp_data_df.loc[mask_1026, 'KEEP_SEND'] = 'Send'
        lp_data_df.loc[mask_1026, 'INTERCEPT_HIGH'] = 1000*lp_data_df['CURRENT_MAC_PRICE']
        lp_data_df.loc[mask_1026, 'INTERCEPT_LOW'] = lp_data_df['MAC1026_UNIT_PRICE']
    
    # CVS prices should be lower than the IND prices. Adjust the bounds if this is violated        
    if check_cvs_ind:
        cvs_mask = lp_data_df[lp_data_df['CHAIN_GROUP'] == 'CVS']['MAC_LIST'].values
        ind_mask = lp_data_df[lp_data_df['CHAIN_GROUP'] == 'NONPREF_OTH']['MAC_LIST'].values
        
        cvs_lb_df = lp_data_df[lp_data_df['MAC_LIST'].isin(cvs_mask)].groupby(['CLIENT', 'REGION',
                                                                               'MEASUREMENT', 'GPI']).agg({'CURRENT_MAC_PRICE': np.nanmax,
                                                                                                           'INTERCEPT_LOW': np.nanmax}).reset_index()
        ind_ub_df = lp_data_df[lp_data_df['MAC_LIST'].isin(ind_mask)].groupby(['CLIENT', 'REGION',
                                                                               'MEASUREMENT', 'GPI']).agg({'INTERCEPT_HIGH': np.nanmin}).reset_index()
        
        cvs_ind_confilct_df = cvs_lb_df.merge(ind_ub_df, how='left', on=['CLIENT', 'REGION','MEASUREMENT', 'GPI'])
        cvs_ind_confilct_df.rename(columns={'CURRENT_MAC_PRICE': 'CVS_PRICE_LB', 'INTERCEPT_LOW': 'CVS_INT_LB', 'INTERCEPT_HIGH': 'IND_UB'}, inplace=True)
        
        cvs_ind_confilct_df.loc[:,'CVS_PRICE_LB'] = (1-p.GPI_LOW_FAC)*cvs_ind_confilct_df.loc[:,'CVS_PRICE_LB']
        cvs_ind_confilct_df['CVS_LB'] = cvs_ind_confilct_df[['CVS_PRICE_LB','CVS_INT_LB']].max(axis=1)
                
        if cvs_ind_confilct_df[cvs_ind_confilct_df['CVS_LB'] > cvs_ind_confilct_df['IND_UB']].shape[0] > 0:
            print('There are some CVS <> IND conflicts to be solved')
            
            lp_data_df_temp = lp_data_df.merge(cvs_ind_confilct_df, how='left', on=['CLIENT', 'REGION','MEASUREMENT', 'GPI'])
            conflict_mask = ((lp_data_df_temp['CVS_LB'] > lp_data_df_temp['IND_UB']) &\
                            (lp_data_df_temp['PRICE_MUTABLE'] != 0) &\
                            (lp_data_df_temp['MAC_LIST'].isin(ind_mask)))
            
            print('Problematic GPI:\n', lp_data_df_temp.loc[conflict_mask, 'GPI_NDC'].unique())
            
            lp_data_df_temp.loc[conflict_mask, 'KEEP_SEND'] = 'Keep'
            lp_data_df_temp.loc[conflict_mask, 'CVS_LB'] = 1.01*lp_data_df_temp.loc[conflict_mask, 'CVS_LB'] # Leaving a 1% gap between the bounds to reduce infeasibilty
            lp_data_df_temp.loc[conflict_mask, 'INTERCEPT_HIGH'] = lp_data_df_temp.loc[conflict_mask, ['INTERCEPT_HIGH','CVS_LB']].max(axis=1) # taking the max as to not override other checks
            lp_data_df_temp.loc[conflict_mask, 'GRX_CONFLICT'] = True
            lp_data_df_temp.drop(columns=['CVS_LB','CVS_PRICE_LB', 'CVS_INT_LB', 'IND_UB'], inplace=True)
            
            assert len(lp_data_df_temp) == len(lp_data_df), "len(lp_data_df_temp) == len(lp_data_df)"
            lp_data_df = lp_data_df_temp
            del lp_data_df_temp
            
            print('The CVS <> IND conflicts have been solved')
        else:
            print('There are no CVS <> IND  conflicts')
            
    if check_retail_mail:
        
        mail_lb_df = lp_data_df[lp_data_df['MEASUREMENT'] == 'M30'].groupby(['CLIENT', 'REGION', 'GPI']).agg({'CURRENT_MAC_PRICE': np.nanmax}).reset_index()
        retail_int_high_df = lp_data_df[lp_data_df['MEASUREMENT'] != 'M30'].groupby(['CLIENT', 'MEASUREMENT', 'REGION', 'GPI']).agg({'INTERCEPT_HIGH': np.nanmin}).reset_index()
        
        mail_retail_confilct_df = retail_int_high_df.merge(mail_lb_df, how='left', on=['CLIENT', 'REGION', 'GPI'])
        mail_retail_confilct_df.rename(columns={'CURRENT_MAC_PRICE': 'MAIL_LB', 'INTERCEPT_HIGH': 'RETAIL_UB'}, inplace=True)
        mail_retail_confilct_df.loc[:,'MAIL_LB'] = (1-p.GPI_LOW_FAC)*mail_retail_confilct_df.loc[:,'MAIL_LB']
        
                
        if mail_retail_confilct_df[mail_retail_confilct_df['MAIL_LB'] > mail_retail_confilct_df['RETAIL_UB']].shape[0] > 0:
            print('There are some Mail <> Retail conflicts to be solved')
            lp_data_df_temp = lp_data_df.merge(mail_retail_confilct_df, how='left', on=['CLIENT', 'MEASUREMENT', 'REGION', 'GPI'])
            conflict_mask = ((lp_data_df_temp['MAIL_LB'] > lp_data_df_temp['RETAIL_UB']) &\
                            (lp_data_df_temp['PRICE_MUTABLE'] != 0) &\
                            (lp_data_df_temp['MEASUREMENT'] != 'M30'))
            
            print('Problematic GPI:\n', lp_data_df_temp.loc[conflict_mask, 'GPI_NDC'].unique())
            
            lp_data_df_temp.loc[conflict_mask, 'KEEP_SEND'] = 'Keep'
            lp_data_df_temp.loc[conflict_mask, 'MAIL_LB'] = 1.01*lp_data_df_temp.loc[conflict_mask, 'MAIL_LB'] # Leaving a 1% gap between the bounds to reduce infeasibilty
            lp_data_df_temp.loc[conflict_mask, 'INTERCEPT_HIGH'] = lp_data_df_temp.loc[conflict_mask, ['INTERCEPT_HIGH', 'MAIL_LB']].max(axis=1) # taking the max as to not override other checks
            lp_data_df_temp.loc[conflict_mask, 'GRX_CONFLICT'] = True
            lp_data_df_temp.drop(columns=['MAIL_LB', 'RETAIL_UB'], inplace=True)
            
            assert len(lp_data_df_temp) == len(lp_data_df), "len(lp_data_df_temp) == len(lp_data_df)"
            lp_data_df = lp_data_df_temp
            del lp_data_df_temp
            
            print('The Mail <> Retail conflicts have been solved')
        else:
            print('There are no Mail <> Retail conflicts')   
    return lp_data_df

def correct_GRx_projections(lp_data_df):
    """
    Modifies the projected pharmacy claims.  The pharmacy side is modified to account for the claims that will not have
    an impact on the pharmacy GER. The Keep/Send logic is determined on goodrx_interceptor_optimization.
    Only claims that are 'kept' should count towards the pharmacy GER.
    LM or Last month information is not modified since is accurate. Only projections should be modified!
    Default set to 0
    """
    mask = (lp_data_df['KEEP_SEND'] == 'Send') & (lp_data_df['PRICE_MUTABLE'] != 0) & (lp_data_df['GRX_CONFLICT'] == False)

    # New columns are created to easily compare results w/o interceptor.
    lp_data_df.loc[:, 'ORIG_PHARM_CLAIMS_PROJ_EOY'] = lp_data_df.loc[:, 'PHARM_CLAIMS_PROJ_EOY']
    lp_data_df.loc[:, 'ORIG_PHARM_QTY_PROJ_EOY'] = lp_data_df.loc[:, 'PHARM_QTY_PROJ_EOY']
    lp_data_df.loc[:, 'ORIG_PHARM_FULLAWP_ADJ_PROJ_EOY'] = lp_data_df.loc[:, 'PHARM_QTY_PROJ_EOY']

    lp_data_df.loc[mask, 'PHARM_CLAIMS_PROJ_EOY'] = p.INTERCEPTOR_LEAKAGE*lp_data_df.loc[:, 'PHARM_CLAIMS_PROJ_EOY']
    lp_data_df.loc[mask, 'PHARM_QTY_PROJ_EOY'] = p.INTERCEPTOR_LEAKAGE*lp_data_df.loc[:, 'PHARM_QTY_PROJ_EOY']
    lp_data_df.loc[mask, 'PHARM_FULLAWP_ADJ_PROJ_EOY'] = p.INTERCEPTOR_LEAKAGE*lp_data_df.loc[:, 'PHARM_QTY_PROJ_EOY']

    return lp_data_df


