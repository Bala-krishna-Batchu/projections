# -*- coding: utf-8 -*-
"""CPMO_rms_functions

This module contains the functions that are used for Rebalanced MAC Strategy.

This script requires pandas and numpy to be installed within the Python
environment you are running these functions in

This module contains the following functions:
      *import_grx_claims_data
      *grx_claims_mapped
      *grx_mbr_rules_setup
      *grx_clnt_rules_setup
      *generate_grx_limits
"""
# Import required parameters file and packages
import CPMO_parameters as p
import pandas as pd
import numpy as np
import os
import util_funcs as uf
import BQ
from CPMO_shared_functions import standardize_df


# Function to import the latest GoodRx Min and Max prices for each GPI-CHAIN GROUP (preprocessed from a cleaned table)
# Import the latest aggregated YTD claims data
# Import the VCML Reference/MAC_MAPPING data to map chains with VCML and get a unique list of GPIs on Client MAC

def import_grx_claims_data():
    raw_goodrx_df = uf.read_BQ_data(
        query=BQ.goodrx_raw_data,
        project_id=p.BQ_INPUT_PROJECT_ID,
        dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
        table_id='GOODRX_RAW_DATA' + p.WS_SUFFIX)

    ytd_claims_df = uf.read_BQ_data(
        query=BQ.ytd_combined_claims,
        project_id=p.BQ_INPUT_PROJECT_ID,
        dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
        table_id='YTD_COMBINED_CLAIMS' + p.WS_SUFFIX + p.CCP_SUFFIX,
        customer=', '.join(sorted(p.CUSTOMER_ID)))

    # Standardize the dataframes
    raw_goodrx_df = standardize_df(raw_goodrx_df).drop(columns=['GPI_NDC'])
    ytd_claims_df = standardize_df(ytd_claims_df).drop(columns=['GPI_NDC'])

    # Import the vcml_reference table to get the base_mac_list and pull the GPIs and NDCs from the base mac list
    vcml_reference = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.VCML_REFERENCE_FILE, dtype=p.VARIABLE_TYPE_DIC)

    # Pick the BASE MAC. For Clients with multiple BASE MACS (edge case) pick the BASE MAC corresponding to the Retail Channel
    base_mac = vcml_reference[vcml_reference.CHNL_IND.str.contains('^R.*')].BASE_MAC_LIST_ID.unique()[0]

    vcml_gpi_df = uf.read_BQ_data(
        project_id=p.BQ_INPUT_PROJECT_ID,
        dataset_id=p.BQ_INPUT_DATASET_ENT_CNFV_PROD,
        table_id='gms_ger_opt_base_mac_lists',
        query=BQ.base_mac_list \
            .format(_project=p.BQ_INPUT_PROJECT_ID,
                    _dataset=p.BQ_INPUT_DATASET_ENT_CNFV_PROD,
                    _table='gms_ger_opt_base_mac_lists',
                    _base_mac=base_mac),
        custom=True)

    vcml_gpi_df = standardize_df(vcml_gpi_df).drop(columns=['GPI_NDC'])

    # Import Costplus Data
    costplus_df = uf.read_BQ_data(
        project_id=p.BQ_INPUT_PROJECT_ID,
        dataset_id=p.BQ_INPUT_DATASET_ENT_ENRV_PROD,
        table_id='COSTPLUS_DRUG_PRICE',
        query=BQ.costplus_data \
            .format(_project=p.BQ_INPUT_PROJECT_ID,
                    _dataset=p.BQ_INPUT_DATASET_ENT_ENRV_PROD,
                    _table='COSTPLUS_DRUG_PRICE'),
        custom=True)

    costplus_df = standardize_df(costplus_df)
    costplus_df['MEASUREMENT'] = 'M30'

    return raw_goodrx_df, ytd_claims_df, vcml_gpi_df, costplus_df


def grx_claims_mapped(cleaned_goodrx_df, cleaned_ytd_claims_df):
    """
    This function computes the goodrx price grouped over all the grouping colums and joins it with the claims data based on QTY columns
    Inputs:
    cleaned_goodrx_df
    cleaned_ytd_claims_df
    p.GOODRX_FACTOR

    Outputs:
    ytd_claims_df_grx
    """
    import CPMO_parameters as p

    # Replace NDCs with *s so as to produce a single, GPI level price.
    # NDC level pricing is being dissregarded since there is no real price variablity currently on the data.
    cleaned_goodrx_df.NDC = '***********'
    cleaned_ytd_claims_df.NDC = '***********'

    cleaned_goodrx_df_grouped = cleaned_goodrx_df.groupby(['GPI', 'NDC', 'CHAIN_GROUP', 'QTY']) \
        .agg(MIN_GRX_PRC_QTY=('MIN_GRX_PRC_QTY', min),
             MAX_GRX_PRC_QTY=('MAX_GRX_PRC_QTY', min)).reset_index()

    # Calculate the GoodRx Price. Adjust the GOODRX_FACTOR to select between Min or Max price.
    # 0 sets MIN PRICE and 1 sets MAX PRICE. Default set to 0.5.
    cleaned_goodrx_df_grouped['GOODRX_UNIT_PRICE'] = p.GOODRX_FACTOR * cleaned_goodrx_df_grouped.MAX_GRX_PRC_QTY - \
                                                     (p.GOODRX_FACTOR - 1) * cleaned_goodrx_df_grouped.MIN_GRX_PRC_QTY

    cleaned_goodrx_df_agg = pd.merge(cleaned_goodrx_df_grouped,
                                     cleaned_goodrx_df_grouped[cleaned_goodrx_df_grouped['CHAIN_GROUP'] != 'OTH']. \
                                     groupby(['GPI', 'NDC', 'QTY'])['GOODRX_UNIT_PRICE'].min(),
                                     on=['GPI', 'NDC', 'QTY'],
                                     suffixes=['_SAME', '_CHAIN'])

    # Aggregate the Claims data at GPI-CHAIN_GROUP, the NDCs are all "*"
    ytd_claims_df_gpi_agg = cleaned_ytd_claims_df \
        .groupby(['CUSTOMER_ID', 'CLIENT_NAME', 'REGION', 'BREAKOUT', 'MEASUREMENT',
                  'PREFERRED', 'CHAIN_GROUP', 'GPI', 'NDC']) \
        .agg(TOTAL_CLAIMS=('TOTAL_CLAIMS', 'sum'),
             TOTAL_DISP_QTY=('TOTAL_DISP_QTY', 'sum'),
             TOTAL_SPEND=('TOTAL_SPEND', 'sum'),
             TOTAL_MEMBER_COST_AMOUNT=('TOTAL_MEMBER_COST_AMOUNT', 'sum'),
             TOTAL_CLIENT_BILL_AMOUNT=('TOTAL_CLIENT_BILL_AMOUNT', 'sum'),
             TOTAL_DISP_FEE=('TOTAL_DISP_FEE', 'sum')).reset_index()

    ytd_claims_df_gpi_agg['AVG_QTY_CLAIM'] = ytd_claims_df_gpi_agg['TOTAL_DISP_QTY'] / ytd_claims_df_gpi_agg[
        'TOTAL_CLAIMS']
    ytd_claims_df_gpi_agg['SPEND_CLAIM'] = ytd_claims_df_gpi_agg['TOTAL_SPEND'] / ytd_claims_df_gpi_agg['TOTAL_CLAIMS']
    ytd_claims_df_gpi_agg['MBR_COST_CLAIM'] = ytd_claims_df_gpi_agg['TOTAL_MEMBER_COST_AMOUNT'] / ytd_claims_df_gpi_agg[
        'TOTAL_CLAIMS']
    ytd_claims_df_gpi_agg['CLNT_COST_CLAIM'] = ytd_claims_df_gpi_agg['TOTAL_CLIENT_BILL_AMOUNT'] / \
                                               ytd_claims_df_gpi_agg['TOTAL_CLAIMS']
    ytd_claims_df_gpi_agg['DISP_FEE_CLAIM'] = ytd_claims_df_gpi_agg['TOTAL_DISP_FEE'] / ytd_claims_df_gpi_agg[
        'TOTAL_CLAIMS']
    ytd_claims_df_gpi_agg['UNT_CST'] = ytd_claims_df_gpi_agg['TOTAL_SPEND'] / ytd_claims_df_gpi_agg['TOTAL_DISP_QTY']
    ytd_claims_df_gpi_agg['DISP_FEE_CLAIM_QTY'] = ytd_claims_df_gpi_agg['DISP_FEE_CLAIM'] / ytd_claims_df_gpi_agg[
        'AVG_QTY_CLAIM']

    # Sort the dataframes on QTY for a correct join
    cleaned_goodrx_df_agg = cleaned_goodrx_df_agg.sort_values('QTY')
    ytd_claims_df_gpi_agg = ytd_claims_df_gpi_agg.sort_values('AVG_QTY_CLAIM')

    # Merge on GoodRx QTY equal to less than the avg_qty_claim
    ytd_claims_df_grx = pd.merge_asof(ytd_claims_df_gpi_agg,
                                      cleaned_goodrx_df_agg,
                                      left_on='AVG_QTY_CLAIM',
                                      right_on='QTY',
                                      direction='backward',
                                      left_by=['GPI', 'NDC', 'CHAIN_GROUP'],
                                      right_by=['GPI', 'NDC', 'CHAIN_GROUP'],
                                      allow_exact_matches=True)

    ytd_claims_df_grx_notna = ytd_claims_df_grx[ytd_claims_df_grx.QTY.notna()].reset_index(drop=True)

    # For rows that didn't have a backward match, match on the nearest qty
    ytd_claims_df_grx_isna = pd.merge_asof(
        ytd_claims_df_gpi_agg.iloc[ytd_claims_df_grx[ytd_claims_df_grx.QTY.isna()].index],
        cleaned_goodrx_df_agg,
        left_on='AVG_QTY_CLAIM',
        right_on='QTY',
        direction='nearest',
        left_by=['GPI', 'NDC', 'CHAIN_GROUP'],
        right_by=['GPI', 'NDC', 'CHAIN_GROUP'],
        allow_exact_matches=True).reset_index(drop=True)

    ytd_claims_df_grx_raw = ytd_claims_df_grx_notna.append(ytd_claims_df_grx_isna).reset_index(drop=True)

    # The final df still has some rows without a GoodRx price. These are the GPI-CHAIN Groups for which we do not have GoodRx information
    return ytd_claims_df_grx_raw


# Parameterized inputs to set custom rules based on member cost share
def grx_mbr_rules_setup(param_file=None, verbose=False):
    '''
    This function will create a pricing tier dataframe based on the member cost share.
    The inputs are paramaterized and can be changed by uncommenting the default parameters in CPMO_parameters_template.py
    '''
    rules = pd.DataFrame()
    rules['BENCHMARK_TYPE'] = ['SAME'] * len(p.MBR_THRESHOLD_TYPE_SAME) + ['CHAIN'] * len(p.MBR_THRESHOLD_TYPE_CHAIN)
    rules['MBR_LOW'] = [-np.inf] + p.MBR_PRICING_TIER_SAME + [-np.inf] + p.MBR_PRICING_TIER_CHAIN
    rules['MBR_UPPER'] = p.MBR_PRICING_TIER_SAME + [np.inf] + p.MBR_PRICING_TIER_CHAIN + [np.inf]
    rules['MBR_THRESHOLD_TYPE'] = p.MBR_THRESHOLD_TYPE_SAME + p.MBR_THRESHOLD_TYPE_CHAIN
    rules['MBR_CHANGE_THRESHOLD'] = p.MBR_CHANGE_THRESHOLD_SAME + p.MBR_CHANGE_THRESHOLD_CHAIN  # Negative values
    rules[
        'MBR_GOODRX_COMPETITIVE_MULTIPLIER'] = p.MBR_GOODRX_COMPETITIVE_MULTIPLIER_SAME + p.MBR_GOODRX_COMPETITIVE_MULTIPLIER_CHAIN
    # Additional
    if verbose:
        print(rules)

    return rules


# Parameterized inputs to set custom rules based on client cost share
def grx_clnt_rules_setup(param_file=None, verbose=False):
    '''
    This function will create a pricing tier dataframe to be used with client cost share per claim.
    The default values are still not set and I am setting it to vague numbers as of now.
    The inputs are paramaterized and can be changed by uncommenting the default parameters in CPMO_parameters_template.py
    '''
    rules = pd.DataFrame()
    rules['BENCHMARK_TYPE'] = ['SAME'] * len(p.CLNT_THRESHOLD_TYPE_SAME) + ['CHAIN'] * len(p.CLNT_THRESHOLD_TYPE_CHAIN)
    rules['CLNT_LOW'] = [-np.inf] + p.CLNT_PRICING_TIER_SAME + [-np.inf] + p.CLNT_PRICING_TIER_CHAIN
    rules['CLNT_UPPER'] = p.CLNT_PRICING_TIER_SAME + [np.inf] + p.CLNT_PRICING_TIER_CHAIN + [np.inf]
    rules['CLNT_THRESHOLD_TYPE'] = p.CLNT_THRESHOLD_TYPE_SAME + p.CLNT_THRESHOLD_TYPE_CHAIN
    rules['CLNT_CHANGE_THRESHOLD'] = p.CLNT_CHANGE_THRESHOLD_SAME + p.CLNT_CHANGE_THRESHOLD_CHAIN  # Negative values
    rules[
        'CLNT_GOODRX_COMPETITIVE_MULTIPLIER'] = p.CLNT_GOODRX_COMPETITIVE_MULTIPLIER_SAME + p.CLNT_GOODRX_COMPETITIVE_MULTIPLIER_CHAIN

    # Additional
    if verbose:
        print(rules)

    return rules

##Updated as part of LP Mod
## replaced apply with cross join for vectorized operations
def map_grx_rules(ytd_claims_df_grx_raw, mbr_grx_tier, clnt_grx_tier):
    '''
    This takes a dataframe row of pricing information and returns the goodrx price upper limit for that entry.
    This function is designed to be used with apply accross a dataframe
    Inputs:
            grx_df
            mbr_rules
            clnt_rules
    Ouputs:

    '''
    grx_df = ytd_claims_df_grx_raw.copy()
    mbr_rules = mbr_grx_tier.copy()
    clnt_rules = clnt_grx_tier.copy()

    def explode_rules(rules):
        exploded_rules = pd.merge(rules.assign(key=1), rules.assign(key=1),
                                  on='key',
                                  suffixes=['.SAME', '.CHAIN'])

        exploded_rules = exploded_rules[(exploded_rules['BENCHMARK_TYPE.SAME'] == 'SAME')
                                        & (exploded_rules['BENCHMARK_TYPE.CHAIN'] == 'CHAIN')] \
            .drop(columns=['key'])

        return exploded_rules

    mbr_exploded_rules = explode_rules(mbr_rules)
    clnt_exploded_rules = explode_rules(clnt_rules)

    mbr_clnt_exploded_rules = pd.merge(mbr_exploded_rules.assign(key=1),
                                       clnt_exploded_rules.assign(key=1),
                                       on='key',
                                       suffixes=['.MBR', '.CLNT'])

    # Function to add the rules to every row based on that row's MBR_COST_CLAIM
    # Added logic based on CLNT_COST_CLAIM

    return (grx_df
             .assign(key=1)
             .merge(mbr_clnt_exploded_rules, on='key')
             .loc[lambda df: (df['MBR_COST_CLAIM'] > df['MBR_LOW.SAME'])
                            & (df['MBR_COST_CLAIM'] <= df['MBR_UPPER.SAME'])
                            & (df['MBR_COST_CLAIM'] > df['MBR_LOW.CHAIN'])
                            & (df['MBR_COST_CLAIM'] <= df['MBR_UPPER.CHAIN'])
                            & (df['CLNT_COST_CLAIM'] > df['CLNT_LOW.SAME'])
                            & (df['CLNT_COST_CLAIM'] <= df['CLNT_UPPER.SAME'])
                            & (df['CLNT_COST_CLAIM'] > df['CLNT_LOW.CHAIN'])
                            & (df['CLNT_COST_CLAIM'] <= df['CLNT_UPPER.CHAIN'])]
             .reset_index(drop=True))


def apply_grx_logic(grx_df):
    '''
    This funtion applies the goodrx logic based on member cost share and generates goodrx price limits for
    each GROUP.
    Inputs:
        grx_df: Dataframe with claims info mapped with goodrx price and rules associated per row
    Ouputs:

    '''
    # looping over Member and Client and also over SAME and CHAIN.
    # I wanted to shorten the code and avoid repeating the same calculations
    for var in ['MBR', 'CLNT']:

        grx_df[f'{var}_COINSURANCE'] = grx_df[f'{var}_COST_CLAIM'] / (
                    grx_df['MBR_COST_CLAIM'] + grx_df['CLNT_COST_CLAIM'])

        for same_chain in ['SAME', 'CHAIN']:
            grx_df[f'{var}_TGT_ING_COST.{same_chain}'] = (grx_df[f'{var}_GOODRX_COMPETITIVE_MULTIPLIER.{same_chain}'] *
                                                          grx_df[f'GOODRX_UNIT_PRICE_{same_chain}'] / grx_df[
                                                              f'{var}_COINSURANCE']) - (
                                                                     grx_df['DISP_FEE_CLAIM'] / grx_df['AVG_QTY_CLAIM'])

            grx_df.loc[(grx_df[f'{var}_TGT_ING_COST.{same_chain}'] <= 0), f'{var}_TGT_ING_COST.{same_chain}'] = np.nan

            same_percent_mask = (grx_df[f'{var}_THRESHOLD_TYPE.{same_chain}'] == 'PERCENT')
            same_dollar_mask = (grx_df[f'{var}_THRESHOLD_TYPE.{same_chain}'] == 'DOLLAR')

            grx_df[f'{var}_COST_THRESHOLD.{same_chain}'] = np.nan
            grx_df.loc[same_percent_mask, f'{var}_COST_THRESHOLD.{same_chain}'] = grx_df[same_percent_mask][
                                                                                      f'{var}_COST_CLAIM'] * (1 +
                                                                                                              grx_df[
                                                                                                                  same_percent_mask][
                                                                                                                  f'{var}_CHANGE_THRESHOLD.{same_chain}'])
            grx_df.loc[same_dollar_mask, f'{var}_COST_THRESHOLD.{same_chain}'] = grx_df[same_dollar_mask][
                                                                                    f'{var}_COST_CLAIM'] + \
                                                                                 grx_df[same_dollar_mask][
                                                                                     f'{var}_CHANGE_THRESHOLD.{same_chain}']

            # Eliminate negative values
            grx_df.loc[
                (grx_df[f'{var}_COST_THRESHOLD.{same_chain}'] < 0), f'{var}_COST_THRESHOLD.{same_chain}'] = np.nan

            #
            grx_df[f'{var}_ACTUAL_THRESHOLD.{same_chain}'] = (grx_df[f'{var}_COST_THRESHOLD.{same_chain}'] / grx_df[
                f'{var}_COINSURANCE'] - grx_df['DISP_FEE_CLAIM']) / grx_df['AVG_QTY_CLAIM']

            # Eliminate negative values
            grx_df.loc[
                (grx_df[f'{var}_ACTUAL_THRESHOLD.{same_chain}'] < 0), f'{var}_ACTUAL_THRESHOLD.{same_chain}'] = np.nan

            # If the target cost is greater than threshold set the bound to target cost
            grx_df.loc[(grx_df[f'{var}_TGT_ING_COST.{same_chain}'] >= grx_df[
                f'{var}_ACTUAL_THRESHOLD.{same_chain}']), f'{var}_ADJUSTED_PRICE.{same_chain}'] = grx_df[
                f'{var}_TGT_ING_COST.{same_chain}']

            # If the target cost is less than threshold clip the bound at threshold
            grx_df.loc[(grx_df[f'{var}_TGT_ING_COST.{same_chain}'] < grx_df[
                f'{var}_ACTUAL_THRESHOLD.{same_chain}']), f'{var}_ADJUSTED_PRICE.{same_chain}'] = grx_df[
                f'{var}_ACTUAL_THRESHOLD.{same_chain}']

        # Get the adjusted price at same vs chain (this is applicable only for cvs)
        grx_df[f'{var}_LOWER'] = grx_df[[f'{var}_ADJUSTED_PRICE.SAME', f'{var}_ADJUSTED_PRICE.CHAIN']].apply(
            lambda x: np.max(np.min(x), 0), axis=1)

        # Calculate the price to submit separely based on mbr and clnt calculations
        grx_df[f'{var}_TO_SUBMIT'] = None
        grx_df.loc[(grx_df['CHAIN_GROUP'] == 'CVS'), f'{var}_TO_SUBMIT'] = grx_df.loc[
            (grx_df['CHAIN_GROUP'] == 'CVS'), f'{var}_LOWER']
        grx_df.loc[~(grx_df['CHAIN_GROUP'] == 'CVS'), f'{var}_TO_SUBMIT'] = grx_df.loc[
            ~(grx_df['CHAIN_GROUP'] == 'CVS'), f'{var}_ADJUSTED_PRICE.SAME']

    # Highest/Lowest among the clnt and mbr prices get applied as GoodRx upper limit. Currently set to lowest
    grx_df['GOODRX_CHAIN_PRICE'] = grx_df[['MBR_TO_SUBMIT', 'CLNT_TO_SUBMIT']].apply(lambda x: np.min(np.min(x), 0),
                                                                                     axis=1)

    if p.APPLY_GENERAL_MULTIPLIER:
        # Applies a 3x price cap on all our prices with the current script cost logic
        grx_df.loc[grx_df['GOODRX_CHAIN_PRICE'] > p.GENERAL_MULTIPLIER[0] * grx_df[
            'GOODRX_UNIT_PRICE_SAME'], 'GOODRX_CHAIN_PRICE'] = p.GENERAL_MULTIPLIER[0] * grx_df[
            'GOODRX_UNIT_PRICE_SAME']

        # Applies a 3x prices on all drugs independently on the cost of the script. It only requires that we have utilization.
        grx_df.loc[(grx_df['GOODRX_CHAIN_PRICE'].isna()) &
                   (np.isfinite(grx_df['GOODRX_UNIT_PRICE_SAME'])) &
                   (grx_df['GOODRX_UNIT_PRICE_SAME'] > 0), 'GOODRX_CHAIN_PRICE'] = p.GENERAL_MULTIPLIER[0] * grx_df[
            'GOODRX_UNIT_PRICE_SAME']

    return grx_df


def prepare_rms_output(grx_calculated):
    '''
    This function is used to prepare a final goodrx limit before the overrides are applied.
    The measurement at this point will be limited only to R30 and R90. On joining with the lp_data_culled_df
    in DIR, mail would get the min goodrx price.

    Output: At dataframe grouped on 'REGION','BREAKOUT','MEASUREMENT','PREFERRED','CHAIN_GROUP','GPI'
        with a goodrx price for chain, lowest goodrx price per BREAKOUT GPI, goodrx price at NONPREF_OTH
    '''
    group_cols = ['REGION', 'BREAKOUT', 'MEASUREMENT', 'PREFERRED', 'CHAIN_GROUP', 'GPI']

    goodrx_df = grx_calculated[grx_calculated.GOODRX_CHAIN_PRICE.notna()][group_cols + ['GOODRX_CHAIN_PRICE']]
    goodrx_df = standardize_df(goodrx_df).sort_values("GOODRX_CHAIN_PRICE", ascending=True).drop_duplicates(
        subset=group_cols, keep="first")

    goodrx_df.loc[goodrx_df["CHAIN_GROUP"] == "OTH", "CHAIN_GROUP"] = "NONPREF_OTH"
    # qa_dataframe(goodrx_df, dataset="GOODRX_FILE_AT_{}".format(os.path.basename(__file__)))

    goodrx_nonpref_df = goodrx_df[goodrx_df["CHAIN_GROUP"] == "NONPREF_OTH"].drop(columns=["CHAIN_GROUP"]).rename(
        columns={"GOODRX_CHAIN_PRICE": "GOODRX_NONPREF_PRICE"}).reset_index(drop=True)

    goodrx_limits = pd.merge(goodrx_df, goodrx_nonpref_df, how='left',
                             on=['REGION', 'BREAKOUT', 'MEASUREMENT', 'PREFERRED', 'GPI'])

    return goodrx_limits

##Updated as part of LP Mod
## replaced apply with cross join for vectorized operations
def rms_logic_implementation(grx_df):
    '''
    This funtion applies the goodrx logic based on member cost share as a benchmark to generate the goodrx price.
    There is no additional logic for CVS price matching at chains since that is apply dirrectly on the LP code.  CVS will be price match with all other price points.
    Further more, there is an override to apply a price celing to all drugs.
    At a high level there are two bands.
    On the first band is only appling the a high ceiling to all drugs
    On the second band the member cost share is used to correctly assess what is the best benckmark price.

    Inputs:
        grx_df: Dataframe with claims info mapped with goodrx price and rules associated per row
    Ouputs:
        grx_df: The same DF but now with the RMS price matching columns and the final price point to use on the LP logic.
    '''
    grx_df['MEMBER_COST_SHARE'] = grx_df['MBR_COST_CLAIM'] / (grx_df['MBR_COST_CLAIM'] + grx_df['CLNT_COST_CLAIM'])

    # This is added, to replace all zeros with 0.0001 such that when applying the rules there is no issue with the bounds.
    grx_df['MEMBER_COST_SHARE'].where(grx_df['MEMBER_COST_SHARE'] > 0, grx_df['MEMBER_COST_SHARE'] + 0.0001,
                                      inplace=True, axis=0)

    rules_same = pd.DataFrame()
    rules_same['SCRIPT_COST_LOW_SAME'] = [0] + [p.PRICING_TIER[0]] * 3
    rules_same['SCRIPT_COST_UPPER_SAME'] = [p.PRICING_TIER[0]] + [np.inf] * 3
    rules_same['MBR_SHARE_LOW_SAME'] = [0] * 2 + p.MBR_COST_SHARE_TIER
    rules_same['MBR_SHARE_UPPER_SAME'] = [1] + p.MBR_COST_SHARE_TIER + [1]
    rules_same['COMPETITIVE_MULTIPLIER_SAME'] = p.GENERAL_MULTIPLIER * 2 + p.COMPETITIVE_MULTIPLIER_SAME[::-1]

    grx_df = (grx_df
               .assign(key=1)
               .merge(rules_same.assign(key=1), on='key')
               .loc[lambda df: (df['SPEND_CLAIM'] > df['SCRIPT_COST_LOW_SAME'])
                              & (df['SPEND_CLAIM'] <= df['SCRIPT_COST_UPPER_SAME'])
                              & (df['MEMBER_COST_SHARE'] > df['MBR_SHARE_LOW_SAME'])
                              & (df['MEMBER_COST_SHARE'] <= df['MBR_SHARE_UPPER_SAME'])]
               .reset_index(drop=True)
               .astype({'SCRIPT_COST_LOW_SAME': float}))

    grx_df['GOODRX_CHAIN_PRICE'] = grx_df['COMPETITIVE_MULTIPLIER_SAME'] * grx_df['GOODRX_UNIT_PRICE_SAME']

    if p.APPLY_GENERAL_MULTIPLIER:
        # Applies a 3x price cap on all our prices with the current script cost logic
        grx_df.loc[grx_df['GOODRX_CHAIN_PRICE'] > p.GENERAL_MULTIPLIER[0] * grx_df[
            'GOODRX_UNIT_PRICE_SAME'], 'GOODRX_CHAIN_PRICE'] = p.GENERAL_MULTIPLIER[0] * grx_df[
            'GOODRX_UNIT_PRICE_SAME']

        # Applies a 3x prices on all drugs independently on the cost of the script. It only requires that we have utilization.
        grx_df.loc[(grx_df['GOODRX_CHAIN_PRICE'].isna()) &
                   (np.isfinite(grx_df['GOODRX_UNIT_PRICE_SAME'])) &
                   (grx_df['GOODRX_UNIT_PRICE_SAME'] > 0), 'GOODRX_CHAIN_PRICE'] = p.GENERAL_MULTIPLIER[0] * grx_df[
            'GOODRX_UNIT_PRICE_SAME']

    return grx_df
