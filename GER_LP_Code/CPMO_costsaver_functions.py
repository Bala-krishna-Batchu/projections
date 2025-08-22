"""CPMO_costsaver_functions

Hosts all the functions used for the Costsaver/Interceptor initiative within LP
Ideally should be triggered only when INTERCEPTOR_OPT = True
"""
import os
import datetime as dt
from enum import Enum
from functools import partial, reduce
import numpy as np
import pandas as pd

from typing import (
    Callable,
    Dict,
    List,
    NoReturn,
    Optional,
    Union,
    Literal
)

import BQ
import CPMO_parameters as p
import util_funcs as uf
from CPMO_shared_functions import standardize_df, check_run_status
from qa_checks import qa_dataframe


def get_last_month_zbd_claims(gpi_vol_awp_df_agg, agg_columns_of_interest):
    """
    input: gpi_vol_awp_df_agg: dataframe with claims aggregated on daily cadence
    output: lp_data_zbd_lm_df: dataframe with all claims and zbd claims from last month to calculate the zbd fraction

    This function adds columns with last month aggregated claims and zbd claims for Costsaver performance calculations. 
    We use last month claims to be a indicator of the rest of the year fraction of zbd claims. 
    Last month is calculated differntly for Welcome season and monthly runs.
    For Welcome Season: last month corresponds to the first 30 days of the contract year. 
    For monthly runs: last month corresponds to the last 30 days from YTD claims. 
    """    
    if p.FULL_YEAR:
        lm_start_day = gpi_vol_awp_df_agg.DOF.min()
        lm_end_day = (pd.to_datetime(gpi_vol_awp_df_agg.DOF.min()).date() + dt.timedelta(days = 30)).strftime('%Y-%m-%d')
    else:
        lm_start_day = (pd.to_datetime(gpi_vol_awp_df_agg.DOF.max()).date() - dt.timedelta(days = 30)).strftime('%Y-%m-%d')
        lm_end_day = gpi_vol_awp_df_agg.DOF.max()
    
    #calculate ZBD fractions
    agg_cols = ['CLAIMS','QTY','FULLAWP_ADJ','PHARM_CLAIMS','PHARM_QTY','PHARM_FULLAWP_ADJ','PHARM_FULLNADAC_ADJ','PHARM_FULLACC_ADJ','PHARM_TARG_INGCOST_ADJ',
               'CLAIMS_ZBD','QTY_ZBD','FULLAWP_ADJ_ZBD','PHARM_CLAIMS_ZBD','PHARM_QTY_ZBD','PHARM_FULLAWP_ADJ_ZBD',
                'PHARM_FULLNADAC_ADJ_ZBD','PHARM_FULLACC_ADJ_ZBD','PHARM_TARG_INGCOST_ADJ_ZBD']
    
    lp_data_zbd_lm_df = gpi_vol_awp_df_agg.loc[gpi_vol_awp_df_agg.DOF.between(lm_start_day, lm_end_day)]\
                                        .groupby(agg_columns_of_interest)[agg_cols].agg(sum).reset_index()

    lp_data_zbd_lm_df[agg_cols] = lp_data_zbd_lm_df[agg_cols].fillna(0.0)
    

    lp_data_zbd_lm_df.rename(columns={'CLAIMS': 'CS_LM_CLAIMS',
                                   'QTY': 'CS_LM_QTY',
                                   'FULLAWP_ADJ': 'CS_LM_FULLAWP_ADJ',
                                   'PHARM_CLAIMS': 'CS_LM_PHARM_CLAIMS',
                                   'PHARM_QTY': 'CS_LM_PHARM_QTY',
                                   'PHARM_FULLAWP_ADJ': 'CS_LM_PHARM_FULLAWP_ADJ',
                                   'PHARM_FULLNADAC_ADJ': 'CS_LM_PHARM_FULLNADAC_ADJ',
                                   'PHARM_FULLACC_ADJ': 'CS_LM_PHARM_FULLACC_ADJ',
                                   'PHARM_TARG_INGCOST_ADJ': 'CS_LM_PHARM_TARG_INGCOST_ADJ',
                                   'CLAIMS_ZBD': 'CS_LM_CLAIMS_ZBD',
                                   'QTY_ZBD': 'CS_LM_QTY_ZBD',
                                   'FULLAWP_ADJ_ZBD': 'CS_LM_FULLAWP_ADJ_ZBD',
                                   'PHARM_CLAIMS_ZBD': 'CS_LM_PHARM_CLAIMS_ZBD',
                                   'PHARM_QTY_ZBD': 'CS_LM_PHARM_QTY_ZBD',
                                   'PHARM_FULLAWP_ADJ_ZBD': 'CS_LM_PHARM_FULLAWP_ADJ_ZBD',
                                   'PHARM_FULLNADAC_ADJ_ZBD': 'CS_LM_PHARM_FULLNADAC_ADJ_ZBD',
                                   'PHARM_FULLACC_ADJ_ZBD': 'CS_LM_PHARM_FULLACC_ADJ_ZBD',
                                    'PHARM_TARG_INGCOST_ADJ_ZBD': 'CS_LM_PHARM_TARG_INGCOST_ADJ_ZBD'}, inplace=True)

    return lp_data_zbd_lm_df

def get_mchoice_gpi_flag(lp_data_culled_df):
    """
    Input: 
    lp_data_culled_df: DataFrame containing aggregated year-to-date (YTD) claims, AWP, and quantity.
    Output: 
    lp_data_culled_df: The same DataFrame with an additional:MCHOICE_GPI_FLAG column
        
    This function creates the MCHOICE_GPI_FLAG column, which serves as an identifier for maintenance GPIs associated with MCHOICE in the pharmacy.

    Background information:
    MCHOICE is a program offered by Caremark that allows maintenance drugs to be filled at retail locations respectively at the same price as mail order ( mail order prices are lower than retail prices )
    """

    if "MCHOICE" in set(lp_data_culled_df['CHAIN_GROUP']):
        mchoice_gpi = set(lp_data_culled_df.loc[((lp_data_culled_df.QTY > 0) | (lp_data_culled_df.PHARM_QTY > 0)) & (lp_data_culled_df['CHAIN_GROUP'] == 'MCHOICE'), 'GPI'])
    else:
        mchoice_gpi = []
        
    mchoice_mask = (lp_data_culled_df['MEASUREMENT'] == 'M30') & (lp_data_culled_df['GPI'].isin(mchoice_gpi))
    
    lp_data_culled_df['MCHOICE_GPI_FLAG'] = 0.0
    lp_data_culled_df.loc[mchoice_mask,'MCHOICE_GPI_FLAG'] = 1.0
    
    return lp_data_culled_df


def get_zbd_fraction(lp_data_culled_df):
    """
    input: lp_data_culled_df: dataframe with aggregated YTD claims,awp,qty 
    output: lp_data_culled_df: dataframe with zbd fraction columns added and CURRENT_KEEP_SEND column
    
    This function does two things: 
    1. CURRENT_KEEP_SEND: This column tells us based on the current MAC price if we would be keeping or sending the claim to Marketplace Vendor. We need this column to calculate Do Nothing performance
    2. EXPECTED_KEEP_SEND: This column tells us based on the costsaver logic if we would be keeping or sending the claim to Marketplace Vendor. We need this column to calculate Model performance
    3. .._ZBD_FRAC: These columns give us the fractions for zbd claims with respect to all claims based on claims adjudicated last month.  
    """
    
    #Replace this with import from preprocessing bounds module 
    #This will be the desired Keep/Send column - if pharmacy ger < grx discount then send else keep
    #The import from preprocessing will add 4 columns VENDOR_PRICE, KEEP_SEND_DESIRED, INTERCEPTOR_LOW, INTERCEPTOR_HIGH 
    ##############################################################################################
#     from random import randint
#     np.random.seed(seed=123)
#     lp_data_culled_df.loc[:,'VENDOR_PRICE'] = 1.0  #will be changed to now import actual Grx price from preprocessing module
#     #1 - KEEP 0 - SEND
#     lp_data_culled_df['EXPECTED_KEEP_SEND'] = np.random.choice([0, 1], size=len(lp_data_culled_df))
#     lp_data_culled_df['EXPECTED_KEEP_SEND'] = lp_data_culled_df['EXPECTED_KEEP_SEND'].astype('float')
#     lp_data_culled_df.loc[lp_data_culled_df.EXPECTED_KEEP_SEND == 0.0, 'INTERCEPT_LOW'] = 0.0
#     lp_data_culled_df.loc[lp_data_culled_df.EXPECTED_KEEP_SEND == 0.0, 'INTERCEPT_HIGH'] = 0.9
#     lp_data_culled_df.loc[lp_data_culled_df.EXPECTED_KEEP_SEND == 1.0, 'INTERCEPT_LOW'] = 1.1
#     lp_data_culled_df.loc[lp_data_culled_df.EXPECTED_KEEP_SEND == 1.0, 'INTERCEPT_HIGH'] = 999999.0
    ###############################################################################################

    #CURRENT_KEEP_SEND: How are the claims adjudicated based on current mac price and vendor price 
    lp_data_culled_df.loc[:,'CURRENT_KEEP_SEND'] = CostsaverLogic.KEEP
    lp_data_culled_df.loc[(lp_data_culled_df.EFF_CAPPED_PRICE > lp_data_culled_df.VENDOR_PRICE), 'CURRENT_KEEP_SEND'] = CostsaverLogic.SEND
    
    #Calculate the ZBD Fractions based on last month data 
    lp_data_culled_df['CLAIMS_ZBD_FRAC'] = np.where(lp_data_culled_df.CS_LM_CLAIMS > 0,\
                                                    round(lp_data_culled_df.CS_LM_CLAIMS_ZBD/lp_data_culled_df.CS_LM_CLAIMS,4), 0.0)
    lp_data_culled_df['QTY_ZBD_FRAC'] = np.where(lp_data_culled_df.CS_LM_QTY > 0,\
                                                 round(lp_data_culled_df.CS_LM_QTY_ZBD/lp_data_culled_df.CS_LM_QTY,4), 0.0)
    lp_data_culled_df['FULLAWP_ADJ_ZBD_FRAC'] = np.where(lp_data_culled_df.CS_LM_FULLAWP_ADJ > 0,\
                                                 round(lp_data_culled_df.CS_LM_FULLAWP_ADJ_ZBD/lp_data_culled_df.CS_LM_FULLAWP_ADJ,4), 0.0)
    lp_data_culled_df['PHARM_CLAIMS_ZBD_FRAC'] = np.where(lp_data_culled_df.CS_LM_PHARM_CLAIMS > 0,\
                                                             round(lp_data_culled_df.CS_LM_PHARM_CLAIMS_ZBD/lp_data_culled_df.CS_LM_PHARM_CLAIMS,4), 0.0)
    lp_data_culled_df['PHARM_QTY_ZBD_FRAC'] = np.where(lp_data_culled_df.CS_LM_PHARM_QTY > 0,\
                                                          round(lp_data_culled_df.CS_LM_PHARM_QTY_ZBD/lp_data_culled_df.CS_LM_PHARM_QTY,4), 0.0)
    lp_data_culled_df['PHARM_FULLAWP_ADJ_ZBD_FRAC'] = np.where(lp_data_culled_df.CS_LM_PHARM_FULLAWP_ADJ > 0,\
                                                               round(lp_data_culled_df.CS_LM_PHARM_FULLAWP_ADJ_ZBD/lp_data_culled_df.CS_LM_PHARM_FULLAWP_ADJ,4), 0.0)
    lp_data_culled_df['PHARM_FULLNADAC_ADJ_ZBD_FRAC'] = np.where(lp_data_culled_df.CS_LM_PHARM_FULLNADAC_ADJ > 0,\
                                                               round(lp_data_culled_df.CS_LM_PHARM_FULLNADAC_ADJ_ZBD/lp_data_culled_df.CS_LM_PHARM_FULLNADAC_ADJ,4), 0.0)
    lp_data_culled_df['PHARM_FULLACC_ADJ_ZBD_FRAC'] = np.where(lp_data_culled_df.CS_LM_PHARM_FULLACC_ADJ > 0,\
                                                               round(lp_data_culled_df.CS_LM_PHARM_FULLACC_ADJ_ZBD/lp_data_culled_df.CS_LM_PHARM_FULLACC_ADJ,4), 0.0)
    lp_data_culled_df['PHARM_TARG_INGCOST_ADJ_ZBD_FRAC'] = np.where(lp_data_culled_df.CS_LM_PHARM_TARG_INGCOST_ADJ > 0,\
                                                               round(lp_data_culled_df.CS_LM_PHARM_TARG_INGCOST_ADJ_ZBD/lp_data_culled_df.CS_LM_PHARM_TARG_INGCOST_ADJ,4), 0.0)
    
    frac_cols = ['CLAIMS_ZBD_FRAC','QTY_ZBD_FRAC','FULLAWP_ADJ_ZBD_FRAC',
                 'PHARM_CLAIMS_ZBD_FRAC','PHARM_QTY_ZBD_FRAC','PHARM_FULLAWP_ADJ_ZBD_FRAC'
                 ,'PHARM_FULLNADAC_ADJ_ZBD_FRAC','PHARM_FULLACC_ADJ_ZBD_FRAC','PHARM_TARG_INGCOST_ADJ_ZBD_FRAC'
                ]

    lp_data_culled_df[frac_cols] = lp_data_culled_df[frac_cols].fillna(0.0)
    
    #For Mail that's not MCHOICE set the keep_send to Keep and ZBD Frac to 0.0 to ensure that the changes in EOY qty, awp, spend is not affected by costsaver logic.
    #How do we deal with this?
    mail_mask = (lp_data_culled_df.MEASUREMENT == 'M30') & (lp_data_culled_df.CHAIN_GROUP != 'MCHOICE')
    lp_data_culled_df.loc[mail_mask, 'CURRENT_KEEP_SEND'] = CostsaverLogic.KEEP
    lp_data_culled_df.loc[mail_mask, 'EXPECTED_KEEP_SEND'] = CostsaverLogic.KEEP
    lp_data_culled_df.loc[mail_mask, frac_cols] = 0.0
    
    # keep_send=KEEP and ZBD frac=0.0 for PSAOs/Indys as well 
    # since costsaver turned off for them July 2025
    psao_indy_mask = lp_data_culled_df.CHAIN_GROUP.isin(set(p.PSAO_LIST['GNRC'] + p.PSAO_LIST['BRND']) +['NONPREF_OTH'])
    lp_data_culled_df.loc[psao_indy_mask, 'CURRENT_KEEP_SEND'] = CostsaverLogic.KEEP
    lp_data_culled_df.loc[psao_indy_mask, 'EXPECTED_KEEP_SEND'] = CostsaverLogic.KEEP
    lp_data_culled_df.loc[psao_indy_mask, frac_cols] = 0.0
    
    #For all GPIs which never had any ZBD claim, set actual/desire/current keep/send to keep
    no_zbd_gpi_df = lp_data_culled_df.groupby(['GPI_NDC']).agg(ZBD_CLAIMS = ('CLAIMS_ZBD', np.nansum)).reset_index()
    no_zbd_gpi_list = list(no_zbd_gpi_df[no_zbd_gpi_df.ZBD_CLAIMS == 0.0].GPI_NDC)
    
    lp_data_culled_df.loc[lp_data_culled_df.GPI_NDC.isin(no_zbd_gpi_list), 'CURRENT_KEEP_SEND'] = CostsaverLogic.KEEP
    lp_data_culled_df.loc[lp_data_culled_df.GPI_NDC.isin(no_zbd_gpi_list), 'DESIRE_KEEP_SEND'] = CostsaverLogic.KEEP
    lp_data_culled_df.loc[lp_data_culled_df.GPI_NDC.isin(no_zbd_gpi_list), 'EXPECTED_KEEP_SEND'] = CostsaverLogic.KEEP
    
    lp_data_culled_df.loc[lp_data_culled_df.GPI_NDC.isin(no_zbd_gpi_list), 'VENDOR_CONFLICT'] = False
    lp_data_culled_df.loc[lp_data_culled_df.GPI_NDC.isin(no_zbd_gpi_list), 'INTERCEPT_REASON'] = OverwriteReasons.NO_YTD_ZBD_CLAIMS     
    
    return lp_data_culled_df


def correct_costsaver_projections(lp_data_culled_df):
    """
    This function reassigns the ytd and lag qty, awp and claims based on zbd frac and keep/send
    Note that spend calculations are not changed here and are done in CPMO.py
    We use CURRENT_KEEP_SEND for YTD and LAG adjustments and EXPECTED_KEEP_SEND for EOY adjustments 
    """
    
    #Nothing changes for client side YTD, LAG and EOY 
    if p.FULL_YEAR:
    #For pharmacy the EOY QTY, AWP and Claims would be nonzbd fraction + zbd frac if keep_send is True else just the nonzbd claims
    #The equations have been simplified for coding purpose.
    #For readability: qty_eoy = (qty_ytd+qty_lag+qty_eoy)*(1-zbd_frac) + (qty_ytd+qty_lag+qty_eoy)*zbd_frac*EXPECTED_KEEP_SEND
    #As per John for Full Year we should recalculate ytd spend based on actual keep/send and zbd frac to kind of mimic 
    # how the spend would look like next year with new mac price and current vendor price
        lp_data_culled_df = lp_data_culled_df.assign(
                        CLAIMS_PROJ_EOY_ORIG = lp_data_culled_df.CLAIMS_PROJ_EOY,
                        QTY_PROJ_EOY_ORIG = lp_data_culled_df.QTY_PROJ_EOY,
                        FULLAWP_ADJ_PROJ_EOY_ORIG = lp_data_culled_df.FULLAWP_ADJ_PROJ_EOY,                        
                        
                        CLAIMS_PROJ_EOY = lambda x: (x.CLAIMS + x.CLAIMS_PROJ_LAG + x.CLAIMS_PROJ_EOY),
                        QTY_PROJ_EOY = lambda x: (x.QTY + x.QTY_PROJ_LAG + x.QTY_PROJ_EOY),   
                        FULLAWP_ADJ_PROJ_EOY = lambda x: (x.FULLAWP_ADJ + x.FULLAWP_ADJ_PROJ_LAG + x.FULLAWP_ADJ_PROJ_EOY),
                        
            
                        #How do we want to model full year qty, awp, claims?
            
                            
                        PHARM_CLAIMS_PROJ_EOY_ORIG = lp_data_culled_df.PHARM_CLAIMS_PROJ_EOY,
                        PHARM_QTY_PROJ_EOY_ORIG = lp_data_culled_df.PHARM_QTY_PROJ_EOY,
                        PHARM_FULLAWP_ADJ_PROJ_EOY_ORIG = lp_data_culled_df.PHARM_FULLAWP_ADJ_PROJ_EOY,
                        
                        PHARM_CLAIMS_PROJ_EOY = lambda x: (x.PHARM_CLAIMS + x.PHARM_CLAIMS_PROJ_LAG + x.PHARM_CLAIMS_PROJ_EOY) * (1 - x.PHARM_CLAIMS_ZBD_FRAC * (1 - x.EXPECTED_KEEP_SEND)),
                        PHARM_QTY_PROJ_EOY = lambda x: (x.PHARM_QTY + x.PHARM_QTY_PROJ_LAG + x.PHARM_QTY_PROJ_EOY) * (1 - x.PHARM_QTY_ZBD_FRAC * (1 - x.EXPECTED_KEEP_SEND)),   
                        PHARM_FULLAWP_ADJ_PROJ_EOY = lambda x: (x.PHARM_FULLAWP_ADJ + x.PHARM_FULLAWP_ADJ_PROJ_LAG + x.PHARM_FULLAWP_ADJ_PROJ_EOY) * (1 - x.PHARM_FULLAWP_ADJ_ZBD_FRAC * (1 - x.EXPECTED_KEEP_SEND)))

    else:
    #The equations have been simplified for coding purpose.
    #For readability: qty_eoy = qty_eoy - (qty_eoy*zbd_frac*(1-EXPECTED_KEEP_SEND))
    #This removes the fraction of qty that would be zbd and be sent to marketplace vendors (e.g goodrx)
    #We should also recalculate the LAG AWP, QTY, CLAIMS for pharmacy based on current keep_send logic
    #This simplifies the pharm_lag and pharm_eoy reimburesement as mac_price*pharm_qty_proj_
        lp_data_culled_df = lp_data_culled_df.assign(
                        PHARM_QTY_PROJ_LAG = lambda x: x.PHARM_QTY_PROJ_LAG * (1 - x.PHARM_QTY_ZBD_FRAC * (1 - x.CURRENT_KEEP_SEND)),
                        PHARM_CLAIMS_PROJ_LAG = lambda x: x.PHARM_CLAIMS_PROJ_LAG * (1 - x.PHARM_CLAIMS_ZBD_FRAC * (1 - x.CURRENT_KEEP_SEND)),
                        PHARM_FULLAWP_ADJ_PROJ_LAG = lambda x: x.PHARM_FULLAWP_ADJ_PROJ_LAG * (1 - x.PHARM_FULLAWP_ADJ_ZBD_FRAC * (1 - x.CURRENT_KEEP_SEND)),
                        PHARM_QTY_PROJ_EOY = lambda x: x.PHARM_QTY_PROJ_EOY * (1 - x.PHARM_QTY_ZBD_FRAC * (1 - x.EXPECTED_KEEP_SEND)), 
                        PHARM_CLAIMS_PROJ_EOY_ORIG = lp_data_culled_df.PHARM_CLAIMS_PROJ_EOY,
                        PHARM_CLAIMS_PROJ_EOY = lambda x: x.PHARM_CLAIMS_PROJ_EOY * (1 - x.PHARM_CLAIMS_ZBD_FRAC * (1 - x.EXPECTED_KEEP_SEND)),
                        PHARM_FULLAWP_ADJ_PROJ_EOY = lambda x: x.PHARM_FULLAWP_ADJ_PROJ_EOY * (1 - x.PHARM_FULLAWP_ADJ_ZBD_FRAC * (1 - x.EXPECTED_KEEP_SEND)))
    
    return lp_data_culled_df

class VendorPrcColEnum(Enum):
    """Each Vendor has 3 price columns, an avg price, a buffer high, and buffer low"""
    PRICE = '_PRICE'
    BUFF_H = '_BUFFER_HIGH'
    BUFF_L = '_BUFFER_LOW'
    
class MktplcVendorEnum(str, Enum):
    """Define all possible vendors, and their string values as seen in bigquery tables. 
    Goals: 
    1) Centralize the definition of vendors (instead of defining lists multiple times in different parts of the codebase)
    2) Easy renaming of vendor name string since DE sometimes accidentally changes it
    3) Marketplace logic relies on a consistent order of vendors being called, which this enum can enforce via immutable definition order
    4) Help Intellisense or Copilot function better by giving it an actual enum to work with
    """
    GOODRX = 'GOODRX'
    RXPARTNER = 'RXPARTNER'
    RXSENSE = 'RXSENSE'
    INTELLIGENTRX = 'INTELLIGENTRX'
    
def get_mkt_order_list(exclude_vendor: Union[List, None] = None) -> List[str]:
    """ get list of the vendor enum full names in their definition order.
    Since the best vendor logic relies on the order of vendors in the df being consistent
    
    example: ['GOODRX','RXPARTNER',...]
        
    Parameters
    ----------
    exclude_vendor : List[MktplcVendorEnum] | None, optional
        list of vendors to exclude, if `None` then includes all vendors, 
        by default None

    Returns
    -------
    List[str]
        vendor BQ names in proper definition order
        example return value: ['GOODRX','RXPARTNER',...]
    """
    if exclude_vendor:
        exclude_vendor = set(exclude_vendor)
        return [vendor.value for vendor in MktplcVendorEnum if vendor not in exclude_vendor]
    else:
        return [vendor.value for vendor in MktplcVendorEnum]

def get_mkt_ord_to_nm_dict(exclude_vendor: Union[List, None] = None)->Dict[int, str]:
    """ get dict that maps vendor definition order to its fullname. 
    
    Used in conjunction with nanargmin to easily determine best vendor
    
    example return value: {0:'GOODRX',1:'RXPARTNER',...}
    """
    vendor_ls = get_mkt_order_list(exclude_vendor)
    vendor_count = len(vendor_ls)
    return {i: vendor_ls[i] for i in range(vendor_count)}

def get_mkt_prc_col_list(exclude_vendor: Union[List, None] = None)->List[str]:
    """get each vendor mean price column names in definition order

    e.g ['GOODRX_PRICE','RXPARTNER_PRICE',...]
    """
    return [vend + VendorPrcColEnum.PRICE.value for vend in get_mkt_order_list(exclude_vendor)]

def get_all_specific_vendor_cols(vendor:str)->List[str]:
    """get all col names related to a vendor 
    
    e.g ['GOODRX_PRICE','GOODRX_BUFFER_HIGH','GOODRX_BUFFER_LOW', ...]"""
    return [vendor + coltype.value for coltype in VendorPrcColEnum]

def get_all_mkt_vendor_prc_cols(exclude_vendor: Union[List, None] = None)->List[str]:
    """get all vendor related columns for all vendors
    
    e.g ['GOODRX_PRICE', 'GOODRX_BUFFER_HIGH', 'GOODRX_BUFFER_LOW', ...
        'RXPARTNER_PRICE', 'GOODRX_BUFFER_HIGH', GOODRX_BUFFER_LOW, ...]"""
    return [col for vendor in get_mkt_order_list(exclude_vendor) 
                for col in get_all_specific_vendor_cols(vendor)]

def get_specific_vendor_df(vendor: str) -> pd.DataFrame:
    """get the vendor data from BQ

    Parameters
    ----------
    vendor : str
        The desired vendor to pull data for

    Returns
    -------
    pd.DataFrame
        the client_{vendor}_data_gpi table from BQ as a df
    """
    goodrx_df = uf.read_BQ_data(query=BQ.vendor_query.format(_vendor = vendor),
                                project_id= p.BQ_INPUT_PROJECT_ID,
                                dataset_id= p.BQ_INPUT_DATASET_DS_PRO_LP,
                                table_id=f'client_{vendor.lower()}_data_gpi',
                                customer=', '.join(sorted(p.CUSTOMER_ID)))
    return goodrx_df
    
def get_best_vend_col(vendor_df: pd.DataFrame, has_vendor_mask: pd.Series, 
                      prc_col_enum: VendorPrcColEnum, exclude_vendor: Union[List[MktplcVendorEnum], None] = None)-> np.ndarray:
    """Inspects masked in_df to pull the Best Vendor's price column (PRICE, BUFFER_HIGH, or BUFFER_LOW) for 
    each row. 
    
    Example
    -------
    >>> in_df = pd.DataFrame({'BEST_VENDOR': ['GOODRX','RXSENSE', 'GOODRX'], 
                              'GOODRX_BUFFER_HIGH':[20, 22, 24], 
                              'RXSENSE_BUFFER_HIGH':[10, 12, 14]})
    >>> mktplc_mask = pd.Series([True, True, False])
    >>> get_best_vend_col(in_df, mktplc_mask, VendorPrcColEnum.BUFF_H)
    np.array([20, 12])"""

    return np.select(
        # each item of list is a boolean mask of whether best_vendor equals each respective vendor
        [vendor_df.loc[has_vendor_mask,'BEST_VENDOR']==vendor 
         for vendor in get_mkt_order_list(exclude_vendor = exclude_vendor)],
        # construct a list of {vendor}_{price_col} and query those columns 
        # for np select (e.g [GOODRX_BUFFER_HIGH, RXPARTNER_BUFFER_HIGH, ...])
        [vendor_df.loc[has_vendor_mask, vendor + prc_col_enum.value] 
         for vendor in get_mkt_order_list(exclude_vendor = exclude_vendor)]  
    ) 
    
def calc_best_vendor(vendor_df:pd.DataFrame,
                     exclude_vendor: Union[List[MktplcVendorEnum], None] = None)->pd.DataFrame:
    """Logic for finding the best vendor, function responsible
    for creating VENDOR_PRICE, VENDOR_BUFFER_HIGH, ... etc. 

    Parameters
    ----------
    vendor_df : pd.DataFrame
        Df of vendor prices offered at every Client, chain group, measurement, gpi, maclist 
        that we got from merging vendor prices with mac mapping
    exclude_vendor : List[MktplcVendorEnum] | None, optional
        list of vendors to exclude, if `None` then includes all vendors, 
        by default None

    Returns
    -------
    pd.DataFrame
        vendor df with best vendor's price cols copied and labeled as
        
        VENDOR_PRICE, VENDOR_BUFFER_HIGH, ... etc.
        
        Just use vendor columns from now on for calculations, can ignore rest of mktplace 
        
    """
    # isolate only the vendor price columns e.g GOODRX_PRICE, RXPARTNER_PRICE, etc
    # since we need the columns on 0 based index to properly determine best vendor
    # via definition order in the enum
    only_ordered_prcs = vendor_df.loc[:, get_mkt_prc_col_list(exclude_vendor)]
    # only care about rows where there's at least one vendor offering a price (i.e not all NaN)
    has_vendor_mask = only_ordered_prcs.any(axis=1)
    
    # get the col idx of the vendor with the lowest price (hence why we care so much about vendor order)
    vendor_df.loc[has_vendor_mask, 'BEST_VENDOR_IDX'] = (
        np.nanargmin(only_ordered_prcs.loc[has_vendor_mask],axis=1)
    )
    
    # convert best vendor col idx into the actual vendor name via enum definition order
    vendor_df.loc[has_vendor_mask, 'BEST_VENDOR']  = (
        vendor_df.loc[has_vendor_mask,'BEST_VENDOR_IDX'].map(get_mkt_ord_to_nm_dict(exclude_vendor))
    )
    vendor_df = vendor_df.drop('BEST_VENDOR_IDX',axis=1)
    
    # use best vendor column to get respective best vendor's price columns
    vendor_df.loc[has_vendor_mask, 'VENDOR_PRICE'] = get_best_vend_col(vendor_df, has_vendor_mask, 
                                                                       VendorPrcColEnum.PRICE, exclude_vendor)
    vendor_df.loc[has_vendor_mask, 'VENDOR_BUFFER_HIGH'] = get_best_vend_col(vendor_df, has_vendor_mask, 
                                                                             VendorPrcColEnum.BUFF_H, exclude_vendor)
    vendor_df.loc[has_vendor_mask, 'VENDOR_BUFFER_LOW'] = get_best_vend_col(vendor_df, has_vendor_mask, 
                                                                            VendorPrcColEnum.BUFF_L, exclude_vendor)
    
    return vendor_df

def get_all_vendor_df(mac_mapping: pd.DataFrame, 
                      exclude_vendor: Union[List[MktplcVendorEnum],None] = None) -> pd.DataFrame:
    """creates a vendor df that contains the best (cheapest) vendor's price columns as VENDOR_X columns
     (e.g if GoodRx is the cheapest, then VENDOR_PRICE = GOODRX_PRICE)

    Parameters
    ----------
    mac_mapping : pd.DataFrame
        mac mapping df
    exclude_vendor : List[MktplcVendorEnum] | None, optional
        list of vendors to exclude, if `None` then includes all vendors, 
        by default None

    Returns
    -------
    pd.DataFrame
        creates df with cols 
          [VENDOR_PRICE, VENDOR_BUFFER_HIGH/LOW, ..., 
           GOODRX_PRICE, GOODRX_BUFFER_HIGH/LOW, ...,  
            RXPARTNER_PRICE, RXPARTNER_BUFFER_HIGH/LOW, ...]

    Raises
    ------
    AssertionError
        errors out if not allowed to query from BQ
    """
    if not p.READ_FROM_BQ:
        raise AssertionError(f'need to run bq query to pull vendor DataFrames')
    
    # pull the data for each speicifc vendor and add to list
    vendor_df_ls = [get_specific_vendor_df(vendor) for vendor in get_mkt_order_list(exclude_vendor)]
    # merge all vendor dfs into a single df, outer join since if at least 1 vendor has an offer we want to take it
    vendor_df = reduce(lambda x, y: pd.merge(x, y, on=['CLIENT','CHAIN_GROUP','MEASUREMENT','GPI', 'BG_FLAG'], how='outer'), vendor_df_ls)

    # inner join since mac mapping provides primary key, and no need for vendor rows that don't exist in mac mapping
    vendor_columns = ['CLIENT', 'REGION', 'MEASUREMENT', 'CHAIN_SUBGROUP', 'MAC_LIST', 'GPI', 'BG_FLAG'] + get_all_mkt_vendor_prc_cols(exclude_vendor)
    vendor_df = pd.merge(vendor_df,
                         mac_mapping,
                         how='inner',
                         on = ['CLIENT', 'CHAIN_GROUP', 'MEASUREMENT'])[vendor_columns]
    
    # let a more granular function handle the calculation + creation of 
    # actual best vendor price columns e.g VENDOR_PRICE, VENDOR_BUFFER_HIGH, ... etc
    vendor_df = calc_best_vendor(vendor_df, exclude_vendor)
    
    if p.UNIFORM_MAC_PRICING:
        vendor_df = (vendor_df.groupby(['CLIENT', 'REGION', 'MEASUREMENT','GPI', 'BG_FLAG'])
                     .agg({'VENDOR_PRICE':np.min, 'VENDOR_BUFFER_HIGH':np.max, 'VENDOR_BUFFER_LOW':np.max})
                     .reset_index())
    else:
        vendor_df = (vendor_df.groupby(['CLIENT', 'REGION', 'MEASUREMENT', 'CHAIN_SUBGROUP', 'MAC_LIST', 'GPI', 'BG_FLAG'])
                     .agg({'VENDOR_PRICE':np.min, 'VENDOR_BUFFER_HIGH':np.max, 'VENDOR_BUFFER_LOW':np.max})
                     .reset_index())
    
    return vendor_df

def get_mac_mapping_df():
    """Get mac_mapping df from p.FILE_DYNAMIC_INPUT_PATH + p.MAC_MAPPING_FILE 
      to serve as primary key for vendor df"""
    
    mac_mapping = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_MAPPING_FILE, 
                              dtype=p.VARIABLE_TYPE_DIC)
    mac_mapping = standardize_df(mac_mapping)
    qa_dataframe(mac_mapping, dataset = 'MAC_MAPPING_FILE_AT_{}'.format(os.path.basename(__file__)))
    mapping_dict = {'chain_sub_mapping':["GIE", "ABS", "CVS", "CVSSP", "ELE", "RAD", "NONPREF_OTH_R90OK", "ELE_R90OK", "TPS_R90OK",
                                    "CVSSP_R90OK", "HMA", "AHD", "TPS", "GEN_R90OK", "NONPREF_OTH", "PBX", "MJR", "CAR", "WAG",
                                    "EPC", "WMT", "KRG", "GEN", "KIN", "PREF_OTH", "HMA_R90OK", "WMT_R90OK", "CVS_R90OK",
                                    "WAG_R90OK", "EPC_R90OK", "CAR_R90OK", "MAIL", "MCHOICE",
                                    "ART", "AHS", "RUR", "CHD", "HYV", "THF", "AMZ",
                                    "ARX", "BGY", "BRI", "DDM", "FVW", "GUA", "MGM", "MCY", "PMA", "RYS", "RPT", "SMR", "SVU", "WFN", "WIS",
                                    "CST", "FCD", "PCD", "HEB", "IGD", "LWD", "SMC", "TPS", "WGS", "GUA_R90OK", "MCY_R90OK", "SMC_R90OK"],
            'vendor_chain_mapping':["OTH", "ABS", "CVS", "CVS", "ELE", "RAD", "OTH", "ELE", "OTH", 
                                    "CVS", "HMA", "AHD", "OTH", "OTH", "OTH", "PBX", "MJR", "CAR", "WAG",
                                    "EPC", "WMT", "KRG", "OTH", "OTH", "OTH", "HMA", "WMT", "CVS", 
                                    "WAG", "EPC", "CAR", "MAIL", "MCHOICE",
                                    "OTH", "OTH", "OTH", "OTH", "OTH", "OTH", "OTH",
                                    "OTH", "OTH", "OTH", "OTH", "OTH", "OTH", "OTH", "OTH", "OTH", "OTH", "OTH", "OTH", "OTH","OTH", "OTH",
                                    "OTH", "OTH", "OTH", "OTH", "OTH", "OTH", "OTH", "OTH", "OTH", "OTH", "OTH", "OTH"],
            'chain_grp_mapping':["GIE", "ABS", "CVS", "CVS", "ELE", "RAD", "NONPREF_OTH", "ELE", "TPS", 
                                 "CVS", "HMA", "AHD", "TPS", "GEN", "NONPREF_OTH", "PBX", "MJR", "CAR", "WAG", 
                                 "EPC", "WMT", "KRG", "GEN", "KIN", "PREF_OTH", "HMA", "WMT", "CVS",
                                 "WAG", "EPC", "CAR", "MAIL", "MCHOICE",
                                 "ART", "AHS", "RUR", "CHD", "HYV", "THF", "AMZ",
                                 "ARX", "BGY", "BRI", "DDM", "FVW", "GUA", "MGM", "MCY", "PMA", "RYS", "RPT", "SMR", "SVU", "WFN", "WIS",
                                 "CST", "FCD", "PCD", "HEB", "IGD", "LWD", "SMC", "TPS", "WGS", "GUA", "MCY", "SMC"]}
    mapping_df = pd.DataFrame(data=mapping_dict)
    mac_mapping = pd.merge(mac_mapping,
                           mapping_df,
                           how='left',
                           left_on='CHAIN_SUBGROUP',
                           right_on='chain_sub_mapping')\
                    .rename(columns = {'CUSTOMER_ID':'CLIENT','vendor_chain_mapping':'CHAIN_GROUP'})
                      
    return mac_mapping

def get_vcml_pharm_guarantees(cs_pharm_guarantees: pd.DataFrame, 
                              mac_mapping: pd.DataFrame,)->pd.DataFrame:
    """queries vcml pharm guarantees based on go live year

    Parameters
    ----------
    pharmacy_guarantees : pd.DataFrame
        precalculated pharm guarantees df
    mac_mapping : pd.DataFrame
        mac mapping df from get_mac_mapping_df()

    Returns
    -------
    pd.DataFrame pharm guarantees on the MACLIST level
        
    """
    grp_cols = ['MEASUREMENT', 'CHAIN_GROUP', 'BG_FLAG']
    pharm_cols = ['CS_PHARM_GRTE_TYPE', 'CS_PHARMACY_RATE', 'CS_TARGET_PFEE']
    nadac_coalesce_cols = ['CS_PHARM_GRTE_TYPE2', 'CS_PHARMACY_RATE2', 'CS_TARGET_PFEE2',
                           'CS_PHARM_GRTE_TYPE3', 'CS_PHARMACY_RATE3', 'CS_TARGET_PFEE3']
    # appended PHARM_GRTE_TYPE to filter columns 
    vcml_pharm_guarantees = (cs_pharm_guarantees[grp_cols + pharm_cols + nadac_coalesce_cols]
                             .merge(mac_mapping[['CLIENT', 'REGION', 'MEASUREMENT', 
                                                 'chain_grp_mapping', 'MAC_LIST']],
                                    how = 'left', 
                                    left_on = grp_cols[:-1], 
                                    right_on = ['MEASUREMENT', 'chain_grp_mapping']))
                      
    if p.GO_LIVE.year >= 2025:
        vcml_pharm_guarantees.drop(columns = ['chain_grp_mapping'], inplace = True)
        vcml_pharm_guarantees = vcml_pharm_guarantees.drop_duplicates()
    else:
        vcml_pharm_guarantees = (vcml_pharm_guarantees
                                 # There were no BNCHMK/NADAC/WAC before 2025, so drop these cols
                                 .drop(columns = nadac_coalesce_cols)
                                 .groupby(grp_cols + ['MAC_LIST'])
                                 # There were no pfees before 2025, just rates, 
                                 # so only need to agg cs_pharmacy_rate
                                 .agg(VCML_PHARMACY_RATE = ('CS_PHARMACY_RATE', np.nanmin))
                                 .reset_index())
        
    return vcml_pharm_guarantees

def merge_vendor_pharm_dfs(lp_data_df: pd.DataFrame,
                           vcml_pharm_guarantees: pd.DataFrame, 
                           vendor_df: pd.DataFrame):
    """Merges the lp_data_df with the vendor df and vcml pharm guarantees 
    based on whether uniform pricing is turned on
    """
    
    if p.UNIFORM_MAC_PRICING:
        vendor_on = ['CLIENT', 'REGION', 'MEASUREMENT', 'GPI', 'BG_FLAG']
    else:
        vendor_on = ['CLIENT', 'REGION', 'MEASUREMENT', 'GPI', 'CHAIN_SUBGROUP', 'MAC_LIST', 'BG_FLAG']
    
    vcml_merge_on = [col for col in vcml_pharm_guarantees.columns 
                     if col not in ['VCML_PHARMACY_RATE'] + vcml_pharm_guarantees.filter(regex='CS_.+').columns.to_list()]
    
    lp_data_df_merged = lp_data_df.merge(
        vcml_pharm_guarantees, 
        how = 'left', 
        on = vcml_merge_on
    )
    assert len(lp_data_df) == len(lp_data_df_merged), 'Merging vcml_pharm resulted in extra rows'
    
    lp_data_df_merged = lp_data_df_merged.merge(vendor_df, how = 'left', on = vendor_on)
    assert len(lp_data_df) == len(lp_data_df_merged), 'Merging vendor df resulted in extra rows'
    
    return lp_data_df_merged

def add_missing_retail_complement(lp_data_df_merged:pd.DataFrame)->pd.DataFrame:
    """Some clients have R30 while missing R90 vcmls, or have R90 but missing R30
    
    If so, we need to clone the existing measurement for the other measurement if it's missing
    (e.g clone R30 rows and rename its measurement col to R90 if R90 is missing)
    """

    if 'R90' in lp_data_df_merged.MEASUREMENT.unique():
        group_cols = ['CLIENT', 'REGION', 'CHAIN_GROUP','CHAIN_SUBGROUP', 'GPI','NDC', 'BG_FLAG']
        mean_vendor_prices = (lp_data_df_merged[lp_data_df_merged.MEASUREMENT.isin(['R30','R90'])]
                            .pivot(index = group_cols, columns = 'MEASUREMENT',values = 'VENDOR_PRICE').reset_index())
        
        mean_vendor_prices['R90'].fillna(mean_vendor_prices['R30'],inplace = True)
        mean_vendor_prices['R30'].fillna(mean_vendor_prices['R90'],inplace = True)
        meas_vendor_prices_melt = mean_vendor_prices.melt(id_vars = group_cols, value_vars=['R30','R90'], value_name='VENDOR_PRICE_Measurement')
        lp_data_df_merged = lp_data_df_merged.merge(meas_vendor_prices_melt, 
                                                    how = 'left',
                                                    on = group_cols + ['MEASUREMENT'])
        lp_data_df_merged['VENDOR_PRICE'].fillna(lp_data_df_merged['VENDOR_PRICE_Measurement'],inplace = True)
        lp_data_df_merged.drop(columns = 'VENDOR_PRICE_Measurement', inplace = True)
    
    return lp_data_df_merged

def read_cs_pharm_guarantees_from_bq()->pd.DataFrame:
    """Reads pharm guarantees specifically in costsaver format 
    which can take care of NADAC waterfall when GO_LIVE year >= 2025, 
    but still compatible with GO_LIVE year < 2025"""
    cs_pharm_guarantees : pd.DataFrame = uf.read_BQ_data(
        query=BQ.cs_pharm_guarantees_query,
        project_id=p.BQ_INPUT_PROJECT_ID,
        dataset_id=p.BQ_INPUT_DATASET_ENT_ENRV_PROD,
        table_id='pharm_guarantees_hist',
    )
    cs_pharm_guarantees.columns = map(str.upper, cs_pharm_guarantees.columns)
    cs_pharm_guarantees = standardize_df(cs_pharm_guarantees)
    cs_pharm_guarantees = cs_pharm_guarantees.loc[cs_pharm_guarantees['YR'] == p.GO_LIVE.year]
    cs_pharm_guarantees = cs_pharm_guarantees.drop(columns=['YR'])
    return cs_pharm_guarantees

def remove_psao_vendor_prices(lp_data_df_merged: pd.DataFrame)->pd.DataFrame:
    """Sets psao/indy vendor prices to 0 since PSAOs/Indys have costsaver turnd off

    Parameters
    ----------
    lp_data_df_merged : pd.DataFrame
        result from calling merge_vendor_pharm_dfs() on lp data 
        which adds the vendor prices 

    Returns
    -------
    pd.DataFrame
        same df but with psao/indy vendor prices removed
    """
    psao_indy_mask = lp_data_df_merged.CHAIN_GROUP.isin(set(p.PSAO_LIST['GNRC'] + p.PSAO_LIST['BRND']) + ['NONPREF_OTH'])
    lp_data_df_merged.loc[psao_indy_mask, 'VENDOR_PRICE'] = np.nan
    
    return lp_data_df_merged

def prepare_marketplace_data(lp_data_df: pd.DataFrame,
                             cs_pharm_guarantees: pd.DataFrame) -> pd.DataFrame:
    """adds costsaver marketplace columns to LP df, with the best (cheapest) vendor's columns
    named VENDOR_X (e.g if GoodRx is the cheapest, then VENDOR_PRICE = GOODRX_PRICE)
    
    NOTE: PSAO/NONPREF_OTH have vendor price set to null by remove_psao_vendor_prices()
    since cost saver is turned off for them"""
    
    assert not ((p.MARKETPLACE_CLIENT==True) and (p.COSTSAVER_CLIENT==False)),  "Marketplace param should not be TRUE if not a costsaver client"
    assert not ((p.INTERCEPTOR_OPT == True) and (p.COSTSAVER_CLIENT == False))," Interceptor OPT should not be TRUE if not a costsaver client"

    mac_mapping = get_mac_mapping_df()
    
    # If a client is not a marketplace client (e.g WTW/AON), they only get GoodRx as a valid vendor
    # A non-costsaver client should never reach this function, so we don't need to worry about them
    if p.MARKETPLACE_CLIENT:
        exclude_vendor = None
    else: 
        exclude_vendor = [vendor for vendor in MktplcVendorEnum if vendor != MktplcVendorEnum.GOODRX]
        
    vendor_df = get_all_vendor_df(mac_mapping, exclude_vendor)
    
    vcml_pharm_guarantees = get_vcml_pharm_guarantees(cs_pharm_guarantees, mac_mapping)
    
    # keep a backup of len before merge for qa purposes after merge to ensure no misjoins
    old_len = len(lp_data_df)

    lp_data_df_merged = merge_vendor_pharm_dfs(lp_data_df, vcml_pharm_guarantees, vendor_df)
    # special method that removes vendor prices for psao/indies
    lp_data_df_merged = remove_psao_vendor_prices(lp_data_df_merged)
    
    # Add in MCHOICE_GPI_FLAG col first
    lp_data_df_merged = get_mchoice_gpi_flag(lp_data_df_merged)
    
    assert old_len == len(lp_data_df_merged), 'Adding unintended rows when creating MCHOICE_GPI_FLAG. Check for duplicates'
     
    lp_data_df_merged = add_missing_retail_complement(lp_data_df_merged)
        
    assert old_len == len(lp_data_df_merged), 'Adding unintended rows. Check for duplicates'
    return lp_data_df_merged
    
def check_nulls(df: pd.DataFrame,
                field: str,
                thresh: float = 0.1) -> Union[pd.DataFrame, NoReturn]:
    nulls = df[field].isna().sum() / len(df)
    if nulls > thresh:
        raise ValueError(f'Too many nulls in "{field}" field, detected '
                         f'{nulls} and threshold is {thresh}')
    return df


def add_fee_buffer_dataset(lp_data_df: pd.DataFrame) -> pd.DataFrame:
    
    if not p.READ_FROM_BQ:
        raise AssertionError('disp_fee_pct_cust query requires BigQuery access'
                             ', please set CPMO_parameters.READ_FROM_BQ')
    
    query = (BQ
              .disp_fee_pct_cust
              .format(_customer_id=uf.get_formatted_string(p.CUSTOMER_ID),
                      _run_id=p.AT_RUN_ID,
                      _project=p.BQ_INPUT_PROJECT_ID,
                      _output_project=p.BQ_OUTPUT_PROJECT_ID,
                      _dataset=p.BQ_INPUT_DATASET_DS_PRO_LP,
                      _table_id1 = uf.get_formatted_table_id('combined_daily_totals')  + p.WS_SUFFIX, 
                      _table_id2 = uf.get_formatted_table_id('ger_opt_taxonomy_final') + p.WS_SUFFIX, 
                      _output_dataset=p.BQ_OUTPUT_DATASET,
                      _time_lag=p.INTERCEPTOR_ZBD_TIME_LAG,
                      _last_data = "'" + p.LAST_DATA.strftime('%Y-%m-%d') +"'"))
    
    disp_fee_df = (uf
                    .read_BQ_data(query=query,
                                  project_id=p.BQ_INPUT_PROJECT_ID,
                                  dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                                  table_id='disp_fee_pct_cust'+ p.WS_SUFFIX,
                                  custom=True)
                    .pipe(standardize_df)
                    .pipe(check_nulls, 'MAX_DISP_FEE_UNIT'))
    return (lp_data_df
             .merge(disp_fee_df, how='left', on=['GPI', 'MAC_LIST', 'BG_FLAG'])
             .fillna({'MIN_DISP_FEE_UNIT': 0, 'MAX_DISP_FEE_UNIT': 0}))

class CostsaverLogic(float, Enum):
    SEND = 0.0
    KEEP = 1.0
    
class OverwriteReasons(str, Enum):
    """Reasons for overwriting interceptor bounds. """
    MAC1026_GT_INTERCEPT_HI = 'PRICES MUST BE HIGHER THAN THE MAC1026'
    CVS_MUST_BE_LOWER = 'CVS MUST HAVE THE LOWEST PRICE'
    MAIL_MUST_BE_LOWER = 'MAIL MUST BE LOWER THAN RETAIL'
    R90_R30_CONFLICT = 'R90 MUST BE CHEAPER THAN R30'
    IMMUTABLE = 'FROZEN PRICES'
    PRICE_INCREASE_LIMITS = 'PRICE INCREASE LIMITS'
    PRICE_DECREASE_LIMITS = 'PRICE DECREASE LIMITS'
    UNKNOWN = 'UNKNOWN'
    CONSISTENT_PRICING = 'VCML LVL PRICING'
    NO_YTD_ZBD_CLAIMS = 'NO_YTD_ZBD_CLAIMS'
    INCPT_L_AVG_AWP_CONFLICT = 'INTERCEPT LOW HIGHER THAN AVG AWP DISCOUNT'

    
def as_enum_value(enum_field: pd.Series) -> pd.Series:
    """This utility converts the pandas DataFrame column given to it
    from a column of Enum to a column of each Enum's value. Will
    raise an error if there are non-enum elements in the column.
    """
    try:
        return enum_field.map(lambda x: x.value if not pd.isna(x) else x)
    except AttributeError as e:
        field_name = enum_field.name or "enum_field pandas Series"
        non_enums = [value for value in enum_field.unique()
                     if not isinstance(value, Enum) and not pd.isna(value)]
        raise ValueError(f'Only supply enum to {field_name}. '
                         f'These values are not enum: {non_enums}') from e

        
def calc_network_margin(bound: pd.DataFrame,
                        desire_field: str = 'DESIRE_KEEP_SEND',
                        actual_field: str = 'ACTUAL_KEEP_SEND',
                        pharm_price = 'PHARMACY_GER',
                        qty: Union[int, float, str, pd.Series] = 1) -> np.array:
    # if ACTUAL = "SEND" and DESIRE = "KEEP"
    # or if ACTUAL and DESIRE are both "SEND"
    default_opt = bound[pharm_price] - bound.VENDOR_PRICE
    
    # when Pharm_Price == 0, for cases without pharmacy GERs, it should always be 0
    # when ACTUAL = "KEEP" & DESIRE = "SEND", take VENDOR_PRICE - PHARMACY_GER
    # when both fields are "KEEP", it should always be 0
    margin_opts = [0, bound.VENDOR_PRICE - bound[pharm_price],  0]
    
    keepsend_opts = [(bound[pharm_price] == 0.0),
     (bound[actual_field] == CostsaverLogic.KEEP)
     & (bound[desire_field] == CostsaverLogic.SEND),

     (bound[actual_field] == CostsaverLogic.KEEP)
     & (bound[desire_field] == CostsaverLogic.KEEP),
    ]

    return np.select(keepsend_opts, margin_opts, default=default_opt)*bound[qty]

def calculate_ic_and_pfee(lp_vendors: pd.DataFrame) -> pd.DataFrame:
    """adds two columns to lp_vendors 
    >>> CS_PHARM_TARGET_IC_UNIT 
    >>> CS_PHARM_TARGET_DISP_FEE_CLAIM
     
    which take into account 2025 BNCHMK/NADAC coalescing to calculate 
    the accurate ICs and fees.
    """
    awp_mask : pd.Series = lp_vendors['CS_PHARM_GRTE_TYPE'] == 'AWP'
    
    bnchmk_mask : pd.Series = lp_vendors['CS_PHARM_GRTE_TYPE'] == 'BNCHMK'
    nadac_mask : pd.Series = lp_vendors['CS_PHARM_GRTE_TYPE'] == 'NADAC'
    
    bnchmk_avail_mask : pd.Series = ((lp_vendors.BNCHMK.notnull()) & (lp_vendors.BNCHMK > 0))
    nadac_avail_mask : pd.Series = ((lp_vendors.NADAC.notnull()) & (lp_vendors.NADAC > 0))
    wac_avail_mask : pd.Series = ((lp_vendors.WAC.notnull()) & (lp_vendors.WAC > 0))
    
    acc_mask : pd.Series = lp_vendors['CS_PHARM_GRTE_TYPE'] == 'ACC'
    
    lp_vendors[['BNCHMK','NADAC','WAC','PHARM_AVG_ACC']] = lp_vendors[['BNCHMK','NADAC','WAC','PHARM_AVG_ACC']].astype('float')
    pharm_grte_type : List[pd.Series]  = [
        awp_mask,
        bnchmk_mask & bnchmk_avail_mask, 
        bnchmk_mask & ~bnchmk_avail_mask & nadac_avail_mask, 
        bnchmk_mask & ~bnchmk_avail_mask & ~nadac_avail_mask & wac_avail_mask, 
        bnchmk_mask & ~bnchmk_avail_mask & ~nadac_avail_mask & ~wac_avail_mask, 
        nadac_mask & nadac_avail_mask,
        nadac_mask & ~nadac_avail_mask & wac_avail_mask,
        nadac_mask & ~nadac_avail_mask & ~wac_avail_mask,
        acc_mask
    ]
    
    pharm_target_ic = [
        (1-lp_vendors.CS_PHARMACY_RATE) * lp_vendors.CS_AVG_AWP, # AWP
        (1-lp_vendors.CS_PHARMACY_RATE) * lp_vendors.BNCHMK, # bnchmk
        (1-lp_vendors.CS_PHARMACY_RATE2) * lp_vendors.NADAC, # bnchmk use nadac
        (1-lp_vendors.CS_PHARMACY_RATE3) * lp_vendors.WAC, # bnchmk use wac
        (1-lp_vendors.CS_PHARMACY_RATE3) * lp_vendors.CS_AVG_AWP/1.2, # bnchmk wac unavail
        (1-lp_vendors.CS_PHARMACY_RATE) * lp_vendors.NADAC, # nadac use nadac
        (1-lp_vendors.CS_PHARMACY_RATE2) * lp_vendors.WAC, # nadac use wac
        (1-lp_vendors.CS_PHARMACY_RATE2) * lp_vendors.CS_AVG_AWP/1.2, # nadac wac unavail
        (1-lp_vendors.CS_PHARMACY_RATE) * lp_vendors.PHARM_AVG_ACC # acc
    ]
    
    pharm_disp_fee_target = [
        lp_vendors.CS_TARGET_PFEE, # awp
        lp_vendors.CS_TARGET_PFEE, # bnchmk
        lp_vendors.CS_TARGET_PFEE2, # bnchmk use nadac
        lp_vendors.CS_TARGET_PFEE3, # bnchmk use wac
        lp_vendors.CS_TARGET_PFEE3, #bnchmk wac unavail
        lp_vendors.CS_TARGET_PFEE, # nadac 
        lp_vendors.CS_TARGET_PFEE2, # nadac use wac
        lp_vendors.CS_TARGET_PFEE2, # nadac wac unavail
        lp_vendors.CS_TARGET_PFEE #acc
    ]

    lp_vendors['CS_PHARM_TARGET_IC_UNIT'] = np.select(pharm_grte_type, pharm_target_ic, default= np.nan)
    lp_vendors['CS_PHARM_TARGET_DISP_FEE_CLAIM'] = np.select(pharm_grte_type, pharm_disp_fee_target, default= np.nan)
    
    return lp_vendors

def calculate_target_price(lp_vendors: pd.DataFrame,
                           groupers: List[str], 
                           yr: int = 2024) -> pd.DataFrame:
    """
    Calculates the target pharmacy price/disp fee to compare the vendor prices to decide
    whether to keep/send the claims. Uses NADAC/WAC/ACC based contracts for PSAOs and CVS/MCHOICE 
    for 2025 and continues to use AWP contracts for 2024.
    
    Creates/Overrides cols: 
    >>> CS_AVG_AWP, CS_PHARM_TARGET_PRICE, 
        AVG_QTY_CLM, VCML_CS_PHARM_TARGET_PRICE,
        CS_PHARM_TARGET_IC_UNIT, CS_PHARM_TARGET_DISP_FEE_CLAIM
        
    Note: Indys/PSAOs have an override logic if GO_LIVE>=July 2025 that makes target rates null 
    since we don't optimize for them anymore
    
    """
    lp_vendors['CS_AVG_AWP'] = np.where(
        ((lp_vendors.AVG_AWP==0)|(lp_vendors.AVG_AWP.isna())),
        lp_vendors.CURR_AWP,
        lp_vendors.AVG_AWP
    )
    
    if yr >= 2025:
        lp_vendors = calculate_ic_and_pfee(lp_vendors)

        # psao/indy turn off 
        psao_indy_mask = lp_vendors.CHAIN_GROUP.isin(set(p.PSAO_LIST['GNRC'] + p.PSAO_LIST['BRND']) + ['NONPREF_OTH'])
        lp_vendors.loc[psao_indy_mask, 'CS_PHARM_TARGET_IC_UNIT'] = np.nan
        lp_vendors.loc[psao_indy_mask, 'CS_PHARM_TARGET_DISP_FEE_CLAIM'] = np.nan
        
        assert len(lp_vendors[
            lp_vendors.CS_PHARM_GRTE_TYPE.notna() 
            & lp_vendors.CS_PHARM_TARGET_IC_UNIT.isna()
            & (~psao_indy_mask)
        ]) == 0, 'Missing target ingredient cost for some rows'

        assert len(lp_vendors[
            lp_vendors['CS_PHARM_GRTE_TYPE'].notna()
            & lp_vendors.CS_PHARM_TARGET_DISP_FEE_CLAIM.isna()
            & (~psao_indy_mask)
        ]) == 0, 'Missing target dispense fee for some rows'

        if p.READ_FROM_BQ == True:
            cs_avg_qty = uf.read_BQ_data(
                BQ.costsaver_avg_claim_qty,
                project_id = p.BQ_OUTPUT_PROJECT_ID,
                dataset_id = p.BQ_OUTPUT_DATASET,
                table_id = "combined_daily_totals",
            custom = True)

            cs_avg_qty = standardize_df(cs_avg_qty)

        else:
            assert False, 'Run with READ_FROM_BQ = True'

        lp_vendors = lp_vendors.merge(cs_avg_qty, on = ['MEASUREMENT','GPI'], how = 'left')    
        
        lp_vendors['CS_PHARM_TARGET_PRICE'] = np.where(
            lp_vendors['AVG_QTY_CLM'].notna(), 
            (   # target price for rows that do have a qty is Ingredient cost + Disp fee per claim
                lp_vendors['CS_PHARM_TARGET_IC_UNIT'] 
                + lp_vendors['CS_PHARM_TARGET_DISP_FEE_CLAIM']/lp_vendors['AVG_QTY_CLM']
            ), 
            0)
        
        avg_qty_clm_na_mask = lp_vendors.AVG_QTY_CLM.isna()
        lp_vendors['AVG_QTY_CLM'] = np.where(
            avg_qty_clm_na_mask,
            lp_vendors.CS_PHARM_TARGET_IC_UNIT,
            lp_vendors.AVG_QTY_CLM
        )
        # lp_vendors['AVG_QTY_CLM'] = lp_vendors['AVG_QTY_CLM'].fillna(0)
        
        bound = (lp_vendors
         .assign(CS_PHARM_TARGET_PRICE_x_CLAIMS=lp_vendors.CS_PHARM_TARGET_PRICE * lp_vendors.CLAIMS)
         .groupby(groupers)
         .agg({'CLAIMS': np.nansum, 'CS_PHARM_TARGET_PRICE_x_CLAIMS': np.nansum})
         .reset_index()
         .assign(
          VCML_CS_PHARM_TARGET_PRICE =lambda df:
             np.where(df.CLAIMS > 0, df.CS_PHARM_TARGET_PRICE_x_CLAIMS / df.CLAIMS, 0),
         )
         .drop(columns=['CLAIMS', 'CS_PHARM_TARGET_PRICE_x_CLAIMS'])
         .merge(lp_vendors, how='left', on=groupers)
         .assign(VENDOR_AVAILABLE=lambda df: df.VENDOR_PRICE.notna(),
                 PHARMACY_GER = lambda df: df.VCML_CS_PHARM_TARGET_PRICE,
                 VENDOR_CONFLICT=False,
                 INTERCEPT_REASON=np.nan)
         .astype({'MAC_LIST': str}))
        
    else:
        bound = (lp_vendors
                 .assign(CURRAWP_x_CLAIMS=lp_vendors.CS_AVG_AWP  * lp_vendors.CLAIMS)
                 .groupby(groupers)
                 .agg({'CLAIMS': np.nansum, 'CURRAWP_x_CLAIMS': np.nansum})
                 .reset_index()
                 .assign(
                  VCML_CURR_AWP=lambda df:
                     np.where(df.CLAIMS > 0, df.CURRAWP_x_CLAIMS / df.CLAIMS, 0),
                 )
                 .drop(columns=['CLAIMS', 'CURRAWP_x_CLAIMS'])
                 .merge(lp_vendors, how='left', on=groupers)
                 .assign(VENDOR_AVAILABLE=lambda df: df.VENDOR_PRICE.notna(),
                         PHARMACY_GER=lambda df: df.VCML_CURR_AWP * (1-df.VCML_PHARMACY_RATE),
                         VENDOR_CONFLICT=False,
                         INTERCEPT_REASON=np.nan)
                 .astype({'MAC_LIST': str}))
        
        bound['VCML_CURR_AWP'].replace(0, np.nan, inplace=True)
    
    return bound

# main function for interceptor bounds logic
def add_interceptor_bounds(lp_vendors: pd.DataFrame,
                           lo_fac: float,
                           reasons: Optional[OverwriteReasons] = None,
                           ) -> pd.DataFrame:
    """Main function to add interceptor bounds.
    Expects a DataFrame with the columns:
        * "VENDOR_PRICE"
    """
    import numpy as np
    from CPMO_shared_functions import standardize_df
    
    # in case it wasn't passed, default to enum from this module
    R = reasons or OverwriteReasons

    if p.UNIFORM_MAC_PRICING: 
        groupers = ['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'GPI', 'BG_FLAG']
    else:
        groupers = ['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'GPI', 'MAC_LIST', 'BG_FLAG']
        
    # "bound" DataFrame: will have interceptor bounds
    bound = calculate_target_price(lp_vendors, groupers, p.GO_LIVE.year)
    
    #Unify CLAIMS_ZBD col for MCHOICE drugs in mail channel
    # if 'MCHOICE' in set(bound['CHAIN_GROUP']): 
    ### We do not want to optimize for mchoice in costsaver currently because of the mail over performance issues 
    ### Hence turing off the logic until the costsaver logic in the krgoer end is completed 
    if False:
        keep_cols = ['CLIENT', 'REGION', 'MEASUREMENT', 'GPI','NDC','BG_FLAG']

        #Get CLAIMS_ZBD for mchoice rows
        mchoice_claims_zbd = bound.loc[bound["CHAIN_GROUP"] == 'MCHOICE', keep_cols+["CLAIMS_ZBD"]]
        mchoice_claims_zbd['CHAIN_GROUP'] = 'MAIL'
        mchoice_claims_zbd.rename(columns={'CLAIMS_ZBD':'CLAIMS_ZBD_MCHOICE'}, inplace = True)

        #Create new col CLAIMS_ZBD_UNIFIED where it's the same as CLAIMS_ZBD for all other CHAIN_GROUP excep for MAIL. 
        #For MAIL that find a match in MCHOICE, it's CLAIMS_ZBD_UNIFIED value will be equal to the matching MCHOICE'S CLAIMS_ZBD value  
        bound = bound.merge(mchoice_claims_zbd, how="left", on = keep_cols + ["CHAIN_GROUP"])
        bound['CLAIMS_ZBD_UNIFIED'] = np.where(bound['CLAIMS_ZBD_MCHOICE'].isnull(), bound['CLAIMS_ZBD'], bound['CLAIMS_ZBD_MCHOICE'])
        bound.drop(columns=['CLAIMS_ZBD_MCHOICE'],inplace=True)
    else:
        bound['CLAIMS_ZBD_UNIFIED'] = bound['CLAIMS_ZBD']
    
    # add bounds with GER
    max_price = 10_000.0
    has_vendor = bound.VENDOR_AVAILABLE
    has_zbd_claims = bound.CLAIMS_ZBD_UNIFIED > 0
    non_mac = (~bound.IS_MAC) | (bound.BG_FLAG == 'B')

    bound['PHARMACY_GER'].replace(0, np.nan, inplace=True) #a.k.a VCML_CS_PHARM_TARGET_PRICE

    sender_mask = (bound.VENDOR_PRICE < bound.PHARMACY_GER) & has_vendor & has_zbd_claims
    keeper_mask = ((bound.VENDOR_PRICE > bound.PHARMACY_GER) & has_vendor) | (~has_zbd_claims) | (non_mac)
    
    masks = [sender_mask, keeper_mask]

    # A.K.A SHOULD_KEEP
    bound['DESIRE_KEEP_SEND'] = np.select(masks,
                                          list(CostsaverLogic),
                                          default=CostsaverLogic.KEEP)
    
    # this EXPECTED_KEEP_SEND column will be overwritten if conflict
    bound['EXPECTED_KEEP_SEND'] = bound.DESIRE_KEEP_SEND

    bound['VENDOR_BUFFER_LOW'] = bound['VENDOR_BUFFER_LOW'].fillna(0)
    hi_bounds = bound.VENDOR_PRICE * p.INTERCEPT_CEILING, bound.VENDOR_PRICE - bound.MAX_DISP_FEE_UNIT - bound.VENDOR_BUFFER_LOW
    bound['INTERCEPT_HIGH'] = np.select(masks, hi_bounds, default=max_price)
    bound.loc[has_vendor & (~has_zbd_claims),'INTERCEPT_HIGH'] = p.INTERCEPT_CEILING*bound.loc[has_vendor & (~has_zbd_claims), 'VENDOR_PRICE']#add a reason for why 3x bounds
    bound.loc[has_vendor & (~has_zbd_claims),'INTERCEPT_REASON'] = R.NO_YTD_ZBD_CLAIMS
    
    bound['INTERCEPT_HIGH'] = bound['INTERCEPT_HIGH'].fillna(max_price)
    bound['VENDOR_BUFFER_HIGH'] = bound['VENDOR_BUFFER_HIGH'].fillna(0)

    lo_bounds = bound.VENDOR_PRICE + bound.MIN_DISP_FEE_UNIT + 0.001*bound.VENDOR_PRICE + bound.VENDOR_BUFFER_HIGH, 0
    bound['INTERCEPT_LOW'] = np.select(masks, lo_bounds, default=0)
    bound['INTERCEPT_LOW'] = bound['INTERCEPT_LOW'].fillna(0.0)
    
    #Remove/comment the below lines when we want to create interceptor bounds for MCHOICE.
    #The code as is always sets MAIL to KEEP, and sets the same interceptor bounds for MAIL as MCHOICE
    #Note: We are forcing the bounds for MCHOICE/MAIL to be (0,10000), however, the DESIRE and EXPECTED KEEP_SEND flag for MCHOICE still follows the costsaver logic.
    #We are assuming that the LP would correctly price MCHOICE even without specific interceptor bounds and that we would be correctly keeping/sending all the MCHOICE claims.
    mail_mask = bound.MEASUREMENT == "M30"
    bound.loc[mail_mask,'INTERCEPT_LOW'] = 0.0
    bound.loc[mail_mask,'INTERCEPT_HIGH'] = max_price
    
    # We're removing the bounds on PSAOs/Indys as well since costsaver turned off for them July 2025
    psao_indy_mask = bound.CHAIN_GROUP.isin(set(p.PSAO_LIST['GNRC'] + p.PSAO_LIST['BRND']) + ['NONPREF_OTH'])
    bound.loc[psao_indy_mask,'INTERCEPT_LOW'] = 0.0
    bound.loc[psao_indy_mask,'INTERCEPT_HIGH'] = max_price
    
    # restricting the interceptor bounds to be uniform at vcml level to prevent unwanted conflicts with other initiatives such as U&C, especially when there are 0 claims in one of the measuremnts at a vcml level
    old_len = len(bound)

    if p.UNIFORM_MAC_PRICING:
        int_grp_df = bound.groupby(['CLIENT','REGION','GPI','NDC', 'BG_FLAG']).agg(int_low = ('INTERCEPT_LOW', np.nanmax),int_high = ('INTERCEPT_HIGH', np.nanmin)).reset_index()
        bound_int_vcml = pd.merge(int_grp_df, bound, how = 'inner', on = ['CLIENT','REGION','GPI','NDC', 'BG_FLAG'])
    else:  
        int_grp_df = bound.groupby(['CLIENT','REGION','MAC_LIST','GPI','NDC', 'BG_FLAG']).agg(int_low = ('INTERCEPT_LOW', np.nanmax),int_high = ('INTERCEPT_HIGH', np.nanmin)).reset_index()
        bound_int_vcml = pd.merge(int_grp_df, bound, how = 'inner', on = ['CLIENT','REGION','MAC_LIST','GPI','NDC', 'BG_FLAG'])
    
    bound_int_vcml[['INTERCEPT_LOW', 'INTERCEPT_HIGH']] = bound_int_vcml[['int_low', 'int_high']]
    bound_int_vcml.drop(['int_low', 'int_high'], axis=1, inplace=True)
    bound = bound_int_vcml.copy(deep=True)
    assert len(bound) == old_len, "Unintentionally adding or dropping rows"
    
    #---------------------------------------------
    #---------------------------------------------
    #check price bound limits
    #---------------------------------------------
    #---------------------------------------------

    if p.TIERED_PRICE_LIM:
        
        if p.FULL_YEAR:
            PRICE_BOUNDS_DF = getattr(p, f'FULL_YEAR_LV_{p.NEW_YEAR_PRICE_LVL}_PRICE_BOUNDS_DF')
        else:    
            PRICE_BOUNDS_DF = p.PRICE_BOUNDS_DF
        
        PRICE_BOUNDS_DF['upper_bound'] = PRICE_BOUNDS_DF['upper_bound'].astype('float64') 
        old_len = len(bound)
        old_awp = bound.AVG_AWP.sum(axis=0)
        bound['SCRIPT_PRICE'] = bound['CURRENT_MAC_PRICE']*bound['QTY']/bound['CLAIMS']
        bound['AVG_QTY'] = bound['QTY']/bound['CLAIMS']
        bound = bound.sort_values(by=['SCRIPT_PRICE'],ascending=True)

        bound_temp = pd.merge_asof(bound[bound['SCRIPT_PRICE'].notnull()],
                              PRICE_BOUNDS_DF, left_on='SCRIPT_PRICE',
                              right_on='upper_bound',direction='forward')

        bound_temp2 = pd.concat([bound_temp, bound[~bound['SCRIPT_PRICE'].notnull()]])
        assert len(bound_temp2) == old_len, "Unintentionally adding or dropping rows"
        assert np.abs(bound_temp2.AVG_AWP.sum(axis=0) - old_awp) < 0.0001, "Making sure no AWP is dropped"

        bound = bound_temp2.copy(deep=True)
        bound['max_percent_increase'] = bound['max_percent_increase'].fillna(p.PRICE_BOUNDS_DF.max_percent_increase[1])
        bound['MAX_PERCENT_INCREASE'] = bound['CURRENT_MAC_PRICE']*(1.0+bound['max_percent_increase'])
        bound['MAX_DOLLAR_INCREASE'] = (bound['SCRIPT_PRICE']+bound['max_dollar_increase'])/bound['AVG_QTY']

        bound['MAC_INCREASE'] = bound[['MAX_PERCENT_INCREASE','MAX_DOLLAR_INCREASE']].min(axis=1) #This will limit how much we can increase increase drug.

    else:    
        drug_mac_hist = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.DRUG_MAC_HIST_FILE), dtype = p.VARIABLE_TYPE_DIC)
        standardize_df(drug_mac_hist)
        conditions = [p.CLIENT_NAME_TABLEAU.startswith('WTW') and not p.REMOVE_WTW_RESTRICTION, 
                        (p.CLIENT_NAME_TABLEAU.startswith('AON')),
                        (p.CLIENT_NAME_TABLEAU.startswith('MVP'))]
        column_name = ['BEG_Q_PRICE','BEG_M_PRICE','BEG_M_PRICE']
        col = np.select(conditions, column_name, default = "CURRENT_MAC_PRICE")
        bound = pd.merge(bound, drug_mac_hist[['GPI', 'MAC_LIST', 'BG_FLAG', 'BEG_Q_PRICE', 'BEG_M_PRICE']], how='left', on=['GPI', 'MAC_LIST', 'BG_FLAG'])
        bound['MAC_INCREASE'] = bound[col]*(1.0+p.GPI_UP_FAC)
        
    
    # retail price decrease limits.  Only does it on retail since the mail prices get adjusted acordingly and modified by another parameter.
    
    # If we by default allow infinite price decrease - we should set MAC_DECREASE to MAC1026?
    bound.loc[:,'MAC_DECREASE'] = bound.loc[:,'CURRENT_MAC_PRICE']*(1.0-p.GPI_LOW_FAC) #This will limit how much we can decrease the drug.
    # bound.loc[:,'MAC_DECREASE'] = 0.0 #This will limit how much we can decrease the drug.
    
    bound = bound.drop(columns = ['MAX_PERCENT_INCREASE','MAX_DOLLAR_INCREASE', 'AVG_QTY',
                                  'SCRIPT_PRICE', 'max_percent_increase', 'max_dollar_increase', 'upper_bound'], errors='ignore')
    
    # Add a price limit check
    inc_mask = ((bound['MAC_INCREASE'] < bound['INTERCEPT_LOW']) & (bound.PRICE_MUTABLE != 0))
    dec_mask = ((bound['MAC_DECREASE'] > bound['INTERCEPT_HIGH']) & (bound.PRICE_MUTABLE != 0))
    
    if p.ALLOW_INTERCEPT_LIMIT and not ((p.CLIENT_NAME_TABLEAU.startswith('WTW') and not p.REMOVE_WTW_RESTRICTION) or (p.CLIENT_NAME_TABLEAU.startswith('AON')) or (p.CLIENT_NAME_TABLEAU.startswith('MVP'))):
        #We allow the prices to increase/decrease irrespective of the price bounds to respect the intercept limits
        if len(bound.loc[inc_mask,:]) > 0 :
            bound.loc[inc_mask,'MAC_INCREASE'] = bound.loc[inc_mask,'INTERCEPT_LOW'] + 0.001
        if len(bound.loc[dec_mask,:]) > 0 :
            bound.loc[dec_mask,'MAC_DECREASE'] = bound.loc[dec_mask,'INTERCEPT_HIGH'] - 0.001
    else:
        if len(bound.loc[inc_mask,:]) > 0 :
            bound.loc[inc_mask,'INTERCEPT_LOW'] = 0 
            bound.loc[inc_mask,'EXPECTED_KEEP_SEND'] = CostsaverLogic.KEEP
            bound.loc[inc_mask,'INTERCEPT_REASON'] = R.PRICE_INCREASE_LIMITS
            bound.loc[inc_mask,'VENDOR_CONFLICT'] = True
            bound.loc[inc_mask,'INTERCEPT_HIGH'] = max_price

        if len(bound.loc[dec_mask,:]) > 0:
            bound.loc[dec_mask,'INTERCEPT_LOW'] = 0 
            bound.loc[dec_mask,'EXPECTED_KEEP_SEND'] = CostsaverLogic.SEND
            bound.loc[dec_mask,'INTERCEPT_REASON'] = R.PRICE_DECREASE_LIMITS
            bound.loc[dec_mask,'VENDOR_CONFLICT'] = True
            bound.loc[dec_mask,'INTERCEPT_HIGH'] = max_price
    
    # For the fictional R90 measurements created to accomodate R90OK vcmls, 
    # we assign the same increase limits as its R30 counterpart sharing the same vcmls
    if (bound.loc[(~bound.MAC_LIST.str.contains('OK')) &
                  (bound.MEASUREMENT == 'R90'), 'CLAIMS'].sum() == 0) & (p.GUARANTEE_CATEGORY == 'Pure Vanilla'):
    
        groupers = ['CLIENT','REGION','BREAKOUT','MEASUREMENT','CHAIN_GROUP','CHAIN_SUBGROUP','MAC_LIST','GPI','NDC', 'BG_FLAG']

        r90_bound = bound[(~bound.MAC_LIST.str.contains('OK')) & (bound.MEASUREMENT == 'R90')].groupby(groupers).agg(MAC_INCREASE = ('MAC_INCREASE', np.nanmin)).reset_index()
        r30_bound = bound[(~bound.MAC_LIST.str.contains('OK')) & (bound.MEASUREMENT == 'R30')].groupby(groupers).agg(MAC_INCREASE = ('MAC_INCREASE', np.nanmin)).reset_index()
        
        r90_r30_bound = pd.merge(r90_bound, r30_bound, how = 'left', on = [x for x in groupers if x != 'MEASUREMENT'], suffixes = ('','_30'))
        r90_r30_bound['MAC_INCREASE'] = r90_r30_bound['MAC_INCREASE_30']
        r90_r30_bound.drop(columns = ['MEASUREMENT_30','MAC_INCREASE_30'], inplace = True)
        r90_r30_bound.set_index(groupers, inplace = True)
        
        bound.set_index(groupers, inplace = True)
        bound.update(r90_r30_bound)
        bound.reset_index(inplace = True)                                                                                                     

    #---------------------------------------------
    #---------------------------------------------            
    # enforce immutable
    #---------------------------------------------
    #---------------------------------------------
    immutable_mask = ((bound.PRICE_MUTABLE == 0)
                      & (bound.VENDOR_PRICE < bound.CURRENT_MAC_PRICE))
    
    bound.loc[(bound.PRICE_MUTABLE == 0)
              & (bound.VENDOR_PRICE >= bound.CURRENT_MAC_PRICE),
              'EXPECTED_KEEP_SEND'] = CostsaverLogic.KEEP
    
    immutable_mapping = {
        'EXPECTED_KEEP_SEND': CostsaverLogic.SEND,
        'INTERCEPT_HIGH': max_price,
        'INTERCEPT_LOW': 0,
        'VENDOR_CONFLICT': True,
        'INTERCEPT_REASON': R.IMMUTABLE,
    }
    bound.loc[immutable_mask, immutable_mapping.keys()] = immutable_mapping.values()
    
    #---------------------------------------------
    #---------------------------------------------
    #check consistent mac prices - ensures that we have same MAC price across VCML
    #---------------------------------------------
    #---------------------------------------------
    
    old_len = len(bound)
    groupers = ['CLIENT','REGION','BREAKOUT','MAC_LIST','GPI','NDC', 'BG_FLAG']

    bound['MIN_PRICE'] = bound[['MAC_DECREASE','INTERCEPT_LOW']].max(axis=1)
    bound['MAX_PRICE'] = np.where(bound['UC_UNIT'] > 0, bound[['MAC_INCREASE', 'INTERCEPT_HIGH', 'UC_UNIT']].min(axis=1),\
                                  bound[['MAC_INCREASE', 'INTERCEPT_HIGH']].min(axis=1))
    
    check_conflict = bound[(bound.PRICE_MUTABLE != 0) & (bound.NDC.str.contains("\*"))]\
                        .groupby(groupers).agg(vcml_low = ('MIN_PRICE',np.max), vcml_high = ('MAX_PRICE',np.min)).reset_index()
    
    bound = bound.merge(check_conflict, how = 'left', on = groupers)
    assert len(bound) == old_len, "Unintentionally adding or dropping rows"
    
    consistent_mask = bound.INTERCEPT_HIGH < bound.vcml_low
    if consistent_mask.any():
        bound.loc[consistent_mask, 'INTERCEPT_HIGH'] = bound.loc[consistent_mask, 'vcml_low']*1.1
        bound.loc[consistent_mask, 'INTERCEPT_REASON'] = R.CONSISTENT_PRICING
        bound.loc[consistent_mask, 'EXPECTED_KEEP_SEND'] = CostsaverLogic.SEND
        bound.loc[(consistent_mask) & (bound.EXPECTED_KEEP_SEND != bound.DESIRE_KEEP_SEND), 'VENDOR_CONFLICT'] = True
    
    consistent_mask = bound.INTERCEPT_LOW > bound.vcml_high
    if consistent_mask.any():
        bound.loc[consistent_mask, 'INTERCEPT_LOW'] = 0
        bound.loc[consistent_mask, 'INTERCEPT_REASON'] = R.CONSISTENT_PRICING
        bound.loc[consistent_mask, 'EXPECTED_KEEP_SEND'] = CostsaverLogic.KEEP
        bound.loc[(consistent_mask) & (bound.EXPECTED_KEEP_SEND != bound.DESIRE_KEEP_SEND), 'VENDOR_CONFLICT'] = True

    #---------------------------------------------
    #---------------------------------------------
    # check 1026
    #---------------------------------------------
    #---------------------------------------------
    
    #Should we change this to allow prices below MAC1026 for big-capped pharmacies?
    mask1026 = ((bound.MAC1026_UNIT_PRICE > bound.INTERCEPT_HIGH)
                & (bound.INTERCEPT_HIGH != max_price)
                & (bound.PRICE_MUTABLE == 1))
    bound.loc[mask1026, 'INTERCEPT_HIGH'] = max_price
    bound.loc[mask1026, 'VENDOR_CONFLICT'] = True
    bound.loc[mask1026, 'INTERCEPT_REASON'] = R.MAC1026_GT_INTERCEPT_HI
    bound.loc[mask1026, 'EXPECTED_KEEP_SEND'] = CostsaverLogic.SEND
    
    #---------------------------------------------
    #---------------------------------------------
    # Check intercept low vs. avg awp discount
    #---------------------------------------------
    #---------------------------------------------
    mailchoice = bound.CHAIN_GROUP.isin(['MAIL', 'MCHOICE'])
    avg_fac = np.where(bound.BG_FLAG == 'G', 
                       np.where(mailchoice,
                                1 - p.MAIL_NON_MAC_RATE,
                                1 - p.RETAIL_NON_MAC_RATE),
                       1-p.BRAND_NON_MAC_RATE)
    
    bound['AVG_AWP_DISCOUNT_UNIT'] = bound.CS_AVG_AWP  * avg_fac + bound.MAX_DISP_FEE_UNIT
    
    incpt_l_infeas_mask = (bound.INTERCEPT_LOW > bound.AVG_AWP_DISCOUNT_UNIT)
    
    bound.loc[incpt_l_infeas_mask, 'INTERCEPT_LOW'] = 0
    
    bound.loc[incpt_l_infeas_mask, 'INTERCEPT_REASON'] = R.INCPT_L_AVG_AWP_CONFLICT
    bound.loc[incpt_l_infeas_mask,'VENDOR_CONFLICT'] = True 
    bound.loc[incpt_l_infeas_mask,'EXPECTED_KEEP_SEND'] = CostsaverLogic.KEEP
    
    #------------------------COA---------------------
    #---------------------------------------------
    # Check that the retail constrint (R30 > R90)
    #---------------------------------------------
    #---------------------------------------------

    # the bounds are both limited by the Intercetpor natural bounds and by the price decrease limits
    if 'R90' in set(bound['MEASUREMENT']): 

        groupers = ['CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GPI_NDC', 'BG_FLAG']

        r90 = (bound
                    .loc[(bound.MEASUREMENT == 'R90') & (bound.PRICE_MUTABLE != 0)]
                    .groupby(groupers)
                    .agg(R90_MAC_LB=('vcml_low', np.nanmax),
                         R90_INT_LB=('INTERCEPT_LOW', np.nanmax)))
        r30 = (bound
                     .loc[(bound.MEASUREMENT == 'R30') & (bound.PRICE_MUTABLE != 0)]
                     .groupby(groupers)
                     .agg(R30_INT_UB=('INTERCEPT_HIGH', np.nanmin),
                         R30_MAC_UB=('vcml_high', np.nanmin)))

        r30_r90_df = (r30
                     .join(r90, on=groupers, how='inner')
                     .reset_index())
        
        # if r90 int low is more than r30 upper bound and r90 lower bounded by r90 int low, relax the int bounds at r90 so that r90 lower bound is set lower than r30 upper bound, avoiding r30-r90 conflict due to intercept bounds
        mask = ((r30_r90_df['R90_INT_LB'] > r30_r90_df['R30_MAC_UB']) & (r30_r90_df['R90_INT_LB'] == r30_r90_df['R90_MAC_LB']))
        issue_df = r30_r90_df.loc[mask,groupers]

        old_len = len(bound)

        if len(issue_df) > 0:
            issue_df.loc[:, 'MEASUREMENT'] = 'R90'
            issue_df.loc[:, 'INTERCEPT_HIGH'] = max_price
            issue_df.loc[:, 'INTERCEPT_LOW'] = 0.0
            issue_df.loc[:, 'VENDOR_CONFLICT'] = True
            issue_df.loc[:, 'INTERCEPT_REASON'] = R.R90_R30_CONFLICT
            issue_df.loc[:, 'EXPECTED_KEEP_SEND'] = CostsaverLogic.KEEP

            issue_df = issue_df.set_index(groupers+['MEASUREMENT'])
            bound = bound.set_index(groupers+['MEASUREMENT'])
            bound.update(issue_df)
            bound = bound.reset_index()

        assert len(bound) == old_len, "Unintentionally adding or dropping rows"      
        
        # If r90 lower bound is more than r30 upper bound and r30 upper bounded by r30 int high, relax the int bounds at r30 so that r30 upper bound is set higher than r30 int high, avoiding r30- r90 conflict due to intercept bounds
        mask_2 = ((r30_r90_df['R90_MAC_LB'] > r30_r90_df['R30_INT_UB']) & (r30_r90_df['R30_INT_UB'] == r30_r90_df['R30_MAC_UB']))
        issue_df_2 = r30_r90_df.loc[mask_2,groupers]

        old_len = len(bound)

        if len(issue_df_2) > 0:
            issue_df_2.loc[:, 'MEASUREMENT'] = 'R30'
            issue_df_2.loc[:, 'INTERCEPT_HIGH'] = max_price
            issue_df_2.loc[:, 'INTERCEPT_LOW'] = 0.0
            issue_df_2.loc[:, 'VENDOR_CONFLICT'] = True
            issue_df_2.loc[:, 'INTERCEPT_REASON'] = R.R90_R30_CONFLICT
            issue_df_2.loc[:, 'EXPECTED_KEEP_SEND'] = CostsaverLogic.SEND

            issue_df_2 = issue_df_2.set_index(groupers+['MEASUREMENT'])
            bound = bound.set_index(groupers+['MEASUREMENT'])
            bound.update(issue_df_2)
            bound = bound.reset_index()

        assert len(bound) == old_len, "Unintentionally adding or dropping rows"        
        
    #---------------------------------------------
    #---------------------------------------------
    # Check CVS Parity - CVS has to have the lowest retail price (per measurement)
    #---------------------------------------------
    #---------------------------------------------

    if 'CVS' in set(bound['CHAIN_GROUP']): 

        parity = np.nan
        if 'CVSSP' in set(bound['CHAIN_SUBGROUP']):
            parity = 'CVSSP'
        else:
            parity = 'CVS'
        
        groupers = ['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'GPI_NDC', 'BG_FLAG']

        cvs_df = (bound
                       .loc[(bound.CHAIN_SUBGROUP == parity) & (bound.PRICE_MUTABLE != 0)]
                       .groupby(groupers)
                       .agg(CVS_MAC_PRC=('CURRENT_MAC_PRICE', np.nanmax),
                            CVS_INT_LB=('INTERCEPT_LOW', np.nanmax)))


        oth_df = (bound
                       .loc[(bound.CHAIN_GROUP != 'CVS') & (bound.PRICE_MUTABLE != 0)]
                       .groupby(groupers)
                       .agg(OTH_UB=('INTERCEPT_HIGH', np.nanmin),
                            OTH_MAC_UB=('vcml_high', np.nanmin)))

        cvs_oth_df = (oth_df
                     .join(cvs_df, on=groupers, how='inner')
                     .reset_index())

        mask = (cvs_oth_df['CVS_INT_LB'] > cvs_oth_df['OTH_UB']) | (cvs_oth_df['CVS_INT_LB'] > cvs_oth_df['OTH_MAC_UB'])

        issue_df = cvs_oth_df.loc[mask,groupers]
        old_len = len(bound)
        
        if len(issue_df) > 0:
            issue_df.loc[:, 'CHAIN_SUBGROUP'] = parity
            issue_df.loc[:, 'INTERCEPT_HIGH'] = max_price
            issue_df.loc[:, 'INTERCEPT_LOW'] = 0.0
            issue_df.loc[:, 'VENDOR_CONFLICT'] = True
            issue_df.loc[:, 'INTERCEPT_REASON'] = R.CVS_MUST_BE_LOWER
            issue_df.loc[:, 'EXPECTED_KEEP_SEND'] = CostsaverLogic.KEEP

            issue_df = issue_df.set_index(groupers+['CHAIN_SUBGROUP'])
            bound = bound.set_index(groupers+['CHAIN_SUBGROUP'])
            bound.update(issue_df)
            bound = bound.reset_index()

        assert len(bound) == old_len, "Unintentionally adding or dropping rows"    

        # CVS R90 has to be lower than all retail
        if 'R90' in set(bound['MEASUREMENT']):
            groupers = ['CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG']

            cvs_df = (bound
                           .loc[((bound.CHAIN_SUBGROUP == parity) | (bound.CHAIN_SUBGROUP == 'CVSSP_R90OK') ) 
                                & (bound.PRICE_MUTABLE != 0) & (bound.MEASUREMENT == 'R90')]
                           .groupby(groupers)
                           .agg(CVS_MAC_PRC=('CURRENT_MAC_PRICE', np.nanmax),
                                CVS_INT_LB=('INTERCEPT_LOW', np.nanmax)))
            oth_df = (bound
                           .loc[(bound.CHAIN_GROUP != 'CVS') & (bound.PRICE_MUTABLE != 0) & (bound.MEASUREMENT != 'M30')]
                           .groupby(groupers)
                           .agg(OTH_UB=('INTERCEPT_HIGH', np.nanmin),
                                OTH_MAC_UB=('vcml_high', np.nanmin)))

            cvs_oth_df = (oth_df
                         .join(cvs_df, on=groupers, how='inner')
                         .reset_index())

            mask = (cvs_oth_df['CVS_INT_LB'] > cvs_oth_df['OTH_UB']) | (cvs_oth_df['CVS_INT_LB'] > cvs_oth_df['OTH_MAC_UB'])        

            issue_df = cvs_oth_df.loc[mask,groupers]
            old_len = len(bound)
            
            if len(issue_df) > 0:
                issue_df_broadcast = pd.concat([issue_df.assign(CHAIN_SUBGROUP = parity), \
                                                issue_df.assign(CHAIN_SUBGROUP = 'CVSSP_R90OK')], ignore_index = True)
                issue_df_broadcast.loc[:, 'MEASUREMENT'] = 'R90' 
                issue_df_broadcast.loc[:, 'INTERCEPT_HIGH'] = max_price
                issue_df_broadcast.loc[:, 'INTERCEPT_LOW'] = 0.0
                issue_df_broadcast.loc[:, 'VENDOR_CONFLICT'] = True
                issue_df_broadcast.loc[:, 'INTERCEPT_REASON'] = R.CVS_MUST_BE_LOWER
                issue_df_broadcast.loc[:, 'EXPECTED_KEEP_SEND'] = CostsaverLogic.KEEP

                issue_df_broadcast = issue_df_broadcast.set_index(groupers+['MEASUREMENT','CHAIN_SUBGROUP'])
                bound = bound.set_index(groupers+['MEASUREMENT','CHAIN_SUBGROUP'])
                bound.update(issue_df_broadcast)
                bound = bound.reset_index()

            assert len(bound) == old_len, "Unintentionally adding or dropping rows"     

        #CVS <= PARITY_PRICE_DIFFERENCE_COLLAR_HIGH * CVSSP
        #Even with all the above checks, we are left with cases where CVS_R90_LB > CVSSP_R30_UB. 
        #These conditions are not directly enforced but are inherently active as CVS < 1.5*CVSSP, R90 < R30       
        if parity == 'CVSSP' and 'R90' in set(bound['MEASUREMENT']):

            groupers = ['CLIENT','REGION','GPI_NDC', 'BG_FLAG']

            cvs_r90 = (bound
                           .loc[(bound.CHAIN_SUBGROUP == 'CVS') & (bound.PRICE_MUTABLE != 0) & (bound.MEASUREMENT == 'R90')]
                           .groupby(groupers)
                           .agg(CVS_INT_LB=('INTERCEPT_LOW', np.nanmax)))

            cvssp_r30 = (bound
                           .loc[(bound.CHAIN_SUBGROUP == 'CVSSP') & (bound.PRICE_MUTABLE != 0) & (bound.MEASUREMENT == 'R30')]
                           .groupby(groupers)
                           .agg(CVSSP_UB=('INTERCEPT_HIGH', np.nanmin),
                                CVSSP_MAC_UB=('vcml_high', np.nanmin)))

            cvs_cvssp_df = (cvs_r90
                         .join(cvssp_r30, on=groupers, how='inner')
                         .reset_index())

            mask = (cvs_cvssp_df['CVS_INT_LB'] > p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH*cvs_cvssp_df['CVSSP_UB']) \
                    | (cvs_cvssp_df['CVS_INT_LB'] > p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH*cvs_cvssp_df['CVSSP_MAC_UB'])        

            issue_df = cvs_cvssp_df.loc[mask,groupers]

            old_len = len(bound)

            if len(issue_df) > 0:
                issue_df.loc[:, 'MEASUREMENT'] = 'R90'
                issue_df.loc[:, 'CHAIN_SUBGROUP'] = 'CVS'
                issue_df.loc[:, 'INTERCEPT_HIGH'] = max_price
                issue_df.loc[:, 'INTERCEPT_LOW'] = 0.0
                issue_df.loc[:, 'VENDOR_CONFLICT'] = True
                issue_df.loc[:, 'INTERCEPT_REASON'] = R.CVS_MUST_BE_LOWER
                issue_df.loc[:, 'EXPECTED_KEEP_SEND'] = CostsaverLogic.KEEP

                issue_df = issue_df.set_index(groupers+['MEASUREMENT','CHAIN_SUBGROUP'])
                bound = bound.set_index(groupers+['MEASUREMENT','CHAIN_SUBGROUP'])
                bound.update(issue_df)
                bound = bound.reset_index()

            assert len(bound) == old_len, "Unintentionally adding or dropping rows"
    #---------------------------------------------
    #---------------------------------------------
    # Check Mail Retail - Mail <= All Retail
    #---------------------------------------------
    #---------------------------------------------
    
    if 'M30' in set(bound['MEASUREMENT']): 
    
        mail_groupers = ['CLIENT', 'REGION', 'GPI', 'NDC', 'BG_FLAG']
        retail_groupers = ['CLIENT', 'MEASUREMENT', 'REGION', 'GPI', 'NDC', 'BG_FLAG']
        mail = (bound
                 .loc[(bound.MEASUREMENT == 'M30') & (bound.NDC.str.contains("\*")) & (bound.PRICE_MUTABLE != 0)]
                 .groupby(mail_groupers)
                 .agg({'CURRENT_MAC_PRICE': np.nanmax,
                      'INTERCEPT_LOW':np.nanmax,
                      'MCHOICE_GPI_FLAG':np.nanmean})
                 .reset_index()
                 .merge(bound
                         .loc[(bound.MEASUREMENT != 'M30') & (bound.NDC.str.contains("\*"))]
                         .groupby(mail_groupers)
                         .agg({'INTERCEPT_HIGH': np.nanmin})
                         .reset_index(),
                        how='inner',
                        on=mail_groupers)
                 .rename(columns={'INTERCEPT_HIGH': 'RETAIL_UB',
                                 'INTERCEPT_LOW': 'MCHOICE_LB'})
               )
        
        mail['MAIL_ORIG_LB'] = mail['CURRENT_MAC_PRICE'] * (1-lo_fac) #Calculate mail bound using lo_fac
        mail['MCHOICE_LB'] = mail[['MAIL_ORIG_LB','MCHOICE_LB']].max(axis=1) #Calculate MCHOICE_LB to be the max of original mail lb and intercept_low
        
        #Assign value for MAIL_LB that will be later used to compare with RETAIL_UB
        #Notie MAIL_LB for mchoice vs. non-mchoice drug is different
        mail['MAIL_LB'] = np.where(mail.MCHOICE_GPI_FLAG == 0,mail.MAIL_ORIG_LB,mail.MCHOICE_LB) 
        
        if p.MAIL_MAC_UNRESTRICTED:
            m30_cons_cap = p.MAIL_UNRESTRICTED_CAP
        else:
            m30_cons_cap = 1.0
        assert m30_cons_cap >= 1.0, "p.MAIL_UNRESTRICTED_CAP < 1 is not recommended without business justification."
        
        mail_conflict = mail.loc[mail.MAIL_LB > mail.RETAIL_UB * m30_cons_cap].reset_index(drop=True)
        
        #For the conflicting GPIs in mail, bring it's INTERCEPT_LOW to 0 to resolve conflict
        if len(mail_conflict) > 0:
            mail_conflict.loc[:, 'MEASUREMENT'] = 'M30'
            mail_conflict.loc[:, 'INTERCEPT_LOW'] = 0.0
            mail_conflict.loc[:, 'VENDOR_CONFLICT'] = True
            mail_conflict.loc[:, 'INTERCEPT_REASON'] = R.MAIL_MUST_BE_LOWER
            mail_conflict.loc[:, 'EXPECTED_KEEP_SEND'] = CostsaverLogic.KEEP
            mail_conflict.rename(columns = {'RETAIL_UB':'INTERCEPT_HIGH'}, inplace = True)

            bound = bound.set_index(retail_groupers)
            mail_conflict = mail_conflict.set_index(retail_groupers)
            bound.update(mail_conflict)
            bound = bound.reset_index()

            assert len(lp_vendors) == len(bound), 'Unintentionally dropping rows'

    bound['PHARMACY_GER'] = bound['PHARMACY_GER'].fillna(0)
    
    if p.GO_LIVE.year != 2025:
        bound['VCML_CURR_AWP'] = bound['VCML_CURR_AWP'].fillna(0)
    
    print(bound.groupby(['CLIENT', 'REGION','MEASUREMENT','INTERCEPT_REASON']).agg({'GPI_NDC':'count'}))
              
    return bound.assign(INTERCEPT_REASON=as_enum_value(bound.INTERCEPT_REASON))
               
def prepare_interceptor_report(total_output):
    '''
    This function is called in CPMO_Reporting_to_IA only for non-interceptor runs. 
    It adds the costsaver specific columns to the total_output to generate report even for non-interceptor runs.
    '''
    
    pharmacy_guarantees = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.PHARM_GUARANTEE_FILE, 
                                                     dtype = p.VARIABLE_TYPE_DIC)).rename(columns = {'PHARMACY':'CHAIN_GROUP', 
                                                                                                     'PHARMACY_SUB':'CHAIN_SUBGROUP',
                                                                                                     'RATE':'PHARMACY_RATE'})
    qa_dataframe(pharmacy_guarantees, dataset = 'PHARM_GUARANTEE_FILE_AT_{}'.format(os.path.basename(__file__)))

    # For Cost Saver, pharmacy guarantees data must be aggregated at the CHAIN_GROUP level.
    # The current pharmacy_guarantees granularity for CS is: ['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'CHAIN_GROUP', CHAIN_SUBGROUP, 'PHRM_GRTE_TYPE']
    # pharmacy_guarantees granularity for CS needs to be: ['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'CHAIN_GROUP', 'PHRM_GRTE_TYPE']
    # From pharmacy_guarantees select rows where either:
    #   - CHAIN_GROUP == CHAIN_SUBGROUP (i.e., the subgroup is not further split and matches the group), or
    #   - CHAIN_GROUP is 'MCHOICE' (since for MCHOICE, the only relevant subgroup is 'MCHOICE_CVS').
    # Assumptions and important notes:
    #   - For all CHAIN_GROUPs except 'MCHOICE', we know that all subgroups within a CHAIN_GROUP have the same guarantee.
    #     This means it is safe to aggregate or filter at the CHAIN_GROUP level without losing guarantee information.
    #   - For 'MCHOICE', the only valid subgroup is 'MCHOICE_CVS', so we explicitly include this case.
    #   - If, in the future, guarantees differ within subgroups of a CHAIN_GROUP (other than 'MCHOICE'), this logic will need to be revisited.
    #   - This approach simplifies downstream processing by ensuring each CHAIN_GROUP has a single guarantee value per combination of the other keys.
    # If you modify the structure of CHAIN_GROUPs or introduce new subgroups with different guarantees, review this logic to ensure correctness.

    pharmacy_guarantees = pharmacy_guarantees[(pharmacy_guarantees['CHAIN_GROUP'] == pharmacy_guarantees['CHAIN_SUBGROUP']) 
                                                | (pharmacy_guarantees['CHAIN_GROUP'] == 'MCHOICE')]

    ## Cost Saver code does not need to have -- PHARMACY_SUB (CHAIN_SUBGROUP)
    pharmacy_guarantees.drop(["CHAIN_SUBGROUP"], axis=1, inplace=True)
    
    #add vendor price and pharmacy rates
    cs_pharm_guarantees = read_cs_pharm_guarantees_from_bq()
    
    add_vendor_df = prepare_marketplace_data(total_output, cs_pharm_guarantees) 
    
    groupers = ['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'GPI', 'MAC_LIST']
    add_vendor_df = calculate_target_price(add_vendor_df, groupers, p.GO_LIVE.year)
    
    #Assign Current Status based on Current MAC Price
    add_vendor_df = add_vendor_df.copy()
    add_vendor_df.loc[:,'CURRENT_KEEP_SEND'] = CostsaverLogic.KEEP
    add_vendor_df.loc[add_vendor_df.VENDOR_PRICE < add_vendor_df.EFF_CAPPED_PRICE,'CURRENT_KEEP_SEND'] = CostsaverLogic.SEND

    add_vendor_df.loc[:,'DESIRE_KEEP_SEND'] = CostsaverLogic.KEEP
    if p.GO_LIVE.year >= 2025:
        add_vendor_df.loc[add_vendor_df.CS_PHARM_TARGET_PRICE > add_vendor_df.VENDOR_PRICE, 'DESIRE_KEEP_SEND'] = CostsaverLogic.SEND
    else:
        add_vendor_df.loc[add_vendor_df.VCML_PHARMACY_RATE < 1 - add_vendor_df.VENDOR_PRICE/add_vendor_df.CS_AVG_AWP , 'DESIRE_KEEP_SEND'] = CostsaverLogic.SEND
    
    add_vendor_df.loc[:, 'EXPECTED_KEEP_SEND'] = add_vendor_df.loc[:, 'DESIRE_KEEP_SEND']
    
    add_vendor_df.loc[:,'ACTUAL_KEEP_SEND'] = CostsaverLogic.KEEP
    add_vendor_df.loc[add_vendor_df.VENDOR_PRICE < add_vendor_df.CS_EFF_CAPPED_PRICE_NEW, 'ACTUAL_KEEP_SEND'] = CostsaverLogic.SEND
    mask = (add_vendor_df['ACTUAL_KEEP_SEND'] != add_vendor_df['EXPECTED_KEEP_SEND']) & (add_vendor_df.CS_EFF_CAPPED_PRICE_NEW < add_vendor_df.NEW_PRICE)
    add_vendor_df.loc[mask, 'INTERCEPT_REASON'] = "UNC/NMR CAPPED PRICES"
    
    add_vendor_df['INTERCEPT_REASON'] = 'Non_CS_Run'
    
    return add_vendor_df

def interceptor_reporting():
    """
    This function is called in CPMO_Reporting_to_IA. 
    It creates a BQ table based on the total output columns to publish data for LP Dashboard's Costsaver 
    """
    if p.READ_FROM_BQ == False:
        total_output = uf.read_BQ_data(
            BQ.lp_total_output_df,
            project_id = p.BQ_OUTPUT_PROJECT_ID,
            dataset_id = p.BQ_OUTPUT_DATASET,
            table_id = "Total_Output",
            run_id = p.AT_RUN_ID,
            client = ', '.join(sorted(p.CUSTOMER_ID)),
            period = p.TIMESTAMP,
            output = True)
        
        total_output = standardize_df(total_output)
    else:
        total_output = standardize_df(pd.read_csv(p.FILE_OUTPUT_PATH + p.TOTAL_OUTPUT, dtype=p.VARIABLE_TYPE_DIC))
        
    if p.READ_FROM_BQ == True:
        gpi_drug_name_mapping = uf.read_BQ_data(
            BQ.gpi_drug_name_mapping,
            project_id = p.BQ_OUTPUT_PROJECT_ID,
            dataset_id = p.BQ_OUTPUT_DATASET,
            table_id = "Total_Output",
        custom = True)
        
        gpi_drug_name_mapping = standardize_df(gpi_drug_name_mapping)
    else:
        assert False, 'Run with READ_FROM_BQ = True or provide a manual gpi-drug name mapping'
        
    if check_run_status(run_status = 'Complete-BypassPerformance'): 
        if p.FULL_YEAR: 
            RUN_TYPE_TABLEAU_UPDATED = "".join([p.RUN_TYPE_TABLEAU, "-BypassPerformance_WS"])
        else:
            RUN_TYPE_TABLEAU_UPDATED = "".join([p.RUN_TYPE_TABLEAU, "-BypassPerformance"])
    else:
        RUN_TYPE_TABLEAU_UPDATED = p.RUN_TYPE_TABLEAU    
    
    
    if not (p.INTERCEPTOR_OPT or p.COSTSAVER_CLIENT):
        total_output = prepare_interceptor_report(
            total_output.drop(columns = ['PHARMACY_RATE', 'CUSTOMER_ID'], errors='ignore')
        )
    elif 'CUSTOMER_ID' in total_output.columns:
        # CostSaver + UNC tends to result in 'CUSTOMER_ID' still being in the index -- drop here to avoid later problems
        total_output.drop(columns=['CUSTOMER_ID'], inplace=True)

    mail_mask = (total_output.MEASUREMENT == 'M30') & (total_output.CHAIN_GROUP != 'MCHOICE')
    generic_mask = total_output.BG_FLAG == 'G'
    
    retail_eoy_awp = total_output.loc[((~mail_mask) & (generic_mask) & (total_output.IS_MAC)), 'FULLAWP_ADJ_PROJ_EOY'].sum()
    
    interceptor_report = total_output[(~mail_mask) & (total_output.VENDOR_AVAILABLE) & (generic_mask) & (total_output.IS_MAC)].reset_index(drop = True).copy(deep = True)

    interceptor_report = pd.merge(interceptor_report, gpi_drug_name_mapping, how = 'left', on = 'GPI')
    
    interceptor_report['INTERCEPT_REASON'] = interceptor_report['INTERCEPT_REASON'].astype(str)
    interceptor_report.rename(columns = {'CLIENT':'CUSTOMER_ID','INTERCEPT_REASON':'REASON','VENDOR_CONFLICT':'CONFLICT'}, inplace = True)
    
    interceptor_report['EFF_CAPPED_PRICE_NEW'] = interceptor_report['CS_EFF_CAPPED_PRICE_NEW']
    interceptor_report['RETAIL_AWP_EOY'] = retail_eoy_awp
    
    if p.GO_LIVE.year >= 2025:
        interceptor_report['PHARM_PRICE'] = interceptor_report.CS_PHARM_TARGET_PRICE.fillna(0.0)
    else:
        interceptor_report['PHARM_PRICE'] = ((1 - interceptor_report.VCML_PHARMACY_RATE) * interceptor_report.PRICING_AVG_AWP).fillna(0.0)
        
    interceptor_report['CLAIMS_PROJ_EOY_ZBD'] = interceptor_report.CLAIMS_PROJ_EOY * interceptor_report.CS_LM_CLAIMS_ZBD/interceptor_report.CS_LM_CLAIMS
    interceptor_report['PHARM_CLAIMS_PROJ_EOY_ZBD'] = interceptor_report.PHARM_CLAIMS_PROJ_EOY * interceptor_report.CS_LM_PHARM_CLAIMS_ZBD/interceptor_report.CS_LM_PHARM_CLAIMS
    
    pharm_claims_eoy_proj = 'PHARM_CLAIMS_PROJ_EOY'
    qty_eoy_proj = 'QTY_PROJ_EOY'
    if p.FULL_YEAR:# or p.INTERCEPTOR_OPT:
        qty_eoy_proj = 'QTY_PROJ_EOY_ORIG'
    if p.FULL_YEAR or p.INTERCEPTOR_OPT:
        pharm_claims_eoy_proj = 'PHARM_CLAIMS_PROJ_EOY_ORIG'

    interceptor_report['NETWORK_MARGIN_QTY'] = (interceptor_report[['QTY','QTY_PROJ_LAG',qty_eoy_proj]].sum(axis=1)\
                                               *interceptor_report.CS_LM_QTY_ZBD/interceptor_report.CS_LM_QTY).fillna(0.0)

    interceptor_report['NETWORK_VALUE'] = calc_network_margin(interceptor_report, 
                                                              desire_field = 'DESIRE_KEEP_SEND',
                                                              actual_field = 'ACTUAL_KEEP_SEND',
                                                              pharm_price = 'PHARM_PRICE',
                                                              qty = 'NETWORK_MARGIN_QTY')

    keep_send_dict = {CostsaverLogic.SEND:'SEND', CostsaverLogic.KEEP:'KEEP'}
    keep_send_cols = ['CURRENT_KEEP_SEND','DESIRE_KEEP_SEND', 'EXPECTED_KEEP_SEND', 'ACTUAL_KEEP_SEND']
    interceptor_report.loc[:,keep_send_cols] = interceptor_report.loc[:,keep_send_cols].replace(keep_send_dict)  

    interceptor_report.loc[:,'CLIENT'] = p.CLIENT_NAME_TABLEAU
    interceptor_report.loc[:,'GO_LIVE'] = str(p.GO_LIVE.date())
    interceptor_report.loc[:,'DATA_ID'] = p.DATA_ID
    interceptor_report.loc[:,'RUN_TYPE'] = p.RUN_TYPE_TABLEAU

    pharm_rate_col_name = 'CS_PHARMACY_RATE' if p.GO_LIVE.year >= 2025 else 'VCML_PHARMACY_RATE'
    output_cols = ['CUSTOMER_ID', 'CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'MAC_LIST','GPI', 'NDC',\
                   'CURRENT_MAC_PRICE', 'FINAL_PRICE', 'CLAIMS', 'PHARM_CLAIMS', 'CLAIMS_PROJ_EOY', 'PHARM_CLAIMS_PROJ_EOY', 'FULLAWP_ADJ', \
                   'PHARM_FULLAWP_ADJ', 'QTY', 'QTY_PROJ_EOY', 'PHARM_CLAIMS_ZBD', 'CLAIMS_ZBD', 'CLAIMS_PROJ_EOY_ZBD', 'PHARM_CLAIMS_PROJ_EOY_ZBD',\
                   'PRICE_MUTABLE', 'PHARMACY_GER', pharm_rate_col_name, 'PRICING_AVG_AWP', 'VENDOR_PRICE', 'PHARM_PRICE','CURRENT_KEEP_SEND','DESIRE_KEEP_SEND',\
                   'ACTUAL_KEEP_SEND', 'CONFLICT', 'REASON', 'NETWORK_VALUE', 'GO_LIVE', 'DATA_ID', 'RUN_TYPE', 'DRUG_NAME', 'RETAIL_AWP_EOY', \
                   'EXPECTED_KEEP_SEND', 'IMMUTABLE_REASON', 'EFF_CAPPED_PRICE','EFF_CAPPED_PRICE_NEW']

    #Mapping to breakout label
    breakout_mapping = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.BREAKOUT_MAPPING_FILE, dtype = p.VARIABLE_TYPE_DIC))
    breakout_mapping = breakout_mapping[['CUSTOMER_ID','BREAKOUT','LABEL']].drop_duplicates().reset_index(drop = True)    
    interceptor_report_upload = interceptor_report.copy()
    interceptor_report_upload = pd.merge(interceptor_report_upload, breakout_mapping, how = 'left', on = ['CUSTOMER_ID','BREAKOUT'])
    interceptor_report_upload = interceptor_report_upload.rename(columns={'BREAKOUT':'BREAKOUT_OLD','LABEL':'BREAKOUT'})
    interceptor_report_upload.drop(columns = ['BREAKOUT_OLD'], inplace = True)
                   
    if p.WRITE_TO_BQ or p.UPLOAD_TO_DASH:
        uf.write_to_bq(
            interceptor_report_upload[output_cols],
            project_output = p.BQ_OUTPUT_PROJECT_ID,
            dataset_output = p.BQ_OUTPUT_DATASET,
            table_id = "costsaver_performance_dashboard",
            client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
            timestamp_param = p.TIMESTAMP,
            run_id = p.AT_RUN_ID,
            schema = None  # TODO: create schema
        )
    else:
        interceptor_report_upload[output_cols].to_csv(p.FILE_REPORT_PATH + p.DATA_ID +'costsaver_performance_dashboard.csv', index=False)

    interceptor_report['GPI_NDC'] = interceptor_report['GPI'] + '_' + interceptor_report['NDC'] 
    ## Creates an aggregate report which is used to display CS metrics in the LP Tableau dashboard 
    interceptor_report_agg = interceptor_report.groupby(['CUSTOMER_ID','CLIENT','BREAKOUT','REGION','MEASUREMENT','CHAIN_GROUP','CHAIN_SUBGROUP','MAC_LIST',\
                                                         'PRICE_MUTABLE', pharm_rate_col_name,'CURRENT_KEEP_SEND','DESIRE_KEEP_SEND','ACTUAL_KEEP_SEND','REASON',\
                                                         'GO_LIVE','DATA_ID','RUN_TYPE','EXPECTED_KEEP_SEND','IMMUTABLE_REASON'], dropna=False)\
                                                .agg({'GPI_NDC': pd.Series.nunique,
                                                     'CLAIMS': 'sum',
                                                     'PHARM_CLAIMS': 'sum',
                                                     'CLAIMS_PROJ_EOY': 'sum',
                                                     'PHARM_CLAIMS_PROJ_EOY': 'sum',
                                                     'FULLAWP_ADJ': 'sum',
                                                     'PHARM_FULLAWP_ADJ': 'sum',
                                                     'QTY': 'sum',
                                                     'QTY_PROJ_EOY': 'sum',
                                                     'PHARM_CLAIMS_ZBD': 'sum',
                                                     'CLAIMS_ZBD': 'sum',
                                                     'CLAIMS_PROJ_EOY_ZBD': 'sum',
                                                     'PHARM_CLAIMS_PROJ_EOY_ZBD': 'sum',
                                                     'NETWORK_VALUE': 'sum',
                                                     'RETAIL_AWP_EOY': 'mean'
                                                 }).rename(columns={
                                                     'GPI_NDC': 'DISTINCT_GPI_NDC'
                                                 }).reset_index()
        
    
    if p.WRITE_TO_BQ or p.UPLOAD_TO_DASH:
        uf.write_to_bq(
            interceptor_report_agg,
            project_output = p.BQ_OUTPUT_PROJECT_ID,
            dataset_output = p.BQ_OUTPUT_DATASET,
            table_id = "costsaver_performance_dashboard_agg",
            client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
            timestamp_param = p.TIMESTAMP,
            run_id = p.AT_RUN_ID,
            schema = None  # TODO: create schema
        )
    else:
        interceptor_report_agg.to_csv(p.FILE_REPORT_PATH + p.DATA_ID +'costsaver_performance_dashboard_agg.csv', index=False)

        
########################### Stale Code ###########################################

def has_cvssp_subgroup(lp_data_df: pd.DataFrame) -> pd.Series:
    subgroup = lp_data_df.CHAIN_SUBGROUP == 'CVSSP'
    return subgroup if subgroup.any() else lp_data_df.CHAIN_GROUP == 'CVS'
        

def check_r90_r30(bound: pd.DataFrame, lo_fac: float) -> pd.DataFrame:
    """
    """
    groupers = ['CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GPI']
    
    r30_low = (bound
                .loc[bound.MEASUREMENT == 'R30']
                .groupby(groupers)
                .agg(R30_LB=('CURRENT_MAC_PRICE', np.nanmax)))
    r90_high = (bound
                 .loc[bound.MEASUREMENT == 'R90']
                 .groupby(groupers)
                 .agg(R90_UB=('INTERCEPT_HIGH', np.nanmin)))
    
    conflict = (r30_low
                 .join(r90_high, on=groupers, how='inner')
                 .assign(MEASUREMENT=lambda df: [['R30', 'R90']] * len(df))
                 .reset_index())
    
    conflict_mask = conflict.R30_LB > conflict.R90_UB
    if conflict_mask.any():
        costs = (
         conflict
          .loc[conflict_mask]
          .explode('MEASUREMENT')
          .merge(bound, on=groupers + ['MEASUREMENT'], how='left')
          .assign(TEMP_KEEP=CostsaverLogic.KEEP,
                  TEMP_SEND=CostsaverLogic.SEND,
                  SEND_COST=partial(calc_network_margin,
                                    actual_field='TEMP_SEND'),
                  KEEP_COST=partial(calc_network_margin,
                                    actual_field='TEMP_KEEP'))
          .groupby(groupers)
          .agg({'SEND_COST': 'sum', 'KEEP_COST': 'sum'})
          .assign(KEEP_SEND_NEW=
                   lambda df: (df.SEND_COST < df.KEEP_COST).astype(float),
                  MEASUREMENT=lambda df: [['R30', 'R90']] * len(df))
        )
        breakpoint()
        
        decision_cols = {'INTERCEPT_HIGH': 'UB_NEW', 'INTERCEPT_LOW': 'LB_NEW'}
        decisions = (
         costs
          .explode('MEASUREMENT')
          .merge(bound, how='left', on=groupers + ['MEASUREMENT'])
        )
        

    return bound


def check_price_bounds(vendor_bounds: pd.DataFrame,
                       tiered_price_lim: bool,
                       price_bounds_df: Optional[pd.DataFrame] = None,
                       ) -> pd.DataFrame:
    if not tiered_price_lim:
        return vendor_bounds
    
    df = vendor_bounds.assign(SCRIPT_PRICE=
                            vendor_bounds.CURRENT_MAC_PRICE
                            * (vendor_bounds.QTY / vendor_bounds.CLAIM))
    
    # assign price bounds
    bins = price_bounds_df.UPPER_BOUND.to_numpy()
    bounds_cols = 'MAX_PERCENT_INCREASE', 'MAX_DOLLAR_INCREASE'
    bounds_idx = np.digitize(df.SCRIPT_PRICE.fillna(bins.max()), bins, right=True)
    df = (df
           .assign(bidx=bounds_idx)
           .merge(price_bounds_df.filter(items=bounds_cols, axis=1),
                  how='left',
                  left_on='bidx',
                  right_index=True)
           .drop('bidx', axis=1))
    
    df['MAX_PERCENT_INCREASE'] = df.SCRIPT_PRICE * (1+df.MAX_PERCENT_INCREASE)
    df['MAX_DOLLAR_INCREASE'] = df.SCRIPT_PRICE + df.MAX_DOLLAR_INCREASE
    df['SCRIPT_INCREASE'] = df[bounds_cols].min(axis=1)
    df['UNIT_INCREASE'] = df.SCRIPT_INCREASE / df.AVG_QTY
    
    # identify conflicts
    conflict_mask = ((df.UNIT_INCREASE > df.INTERCEPT_LOW)
                     & (df.VENDOR_AVAILABLE == True)
                     & (df.PRICE_MUTABLE != 0))

def ensure_cvs_parity(bound: pd.DataFrame) -> pd.DataFrame:
    # calculate network margin column
    
    cvs_groupers = ['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'GPI']
    
    cvs_macs = bound.loc[has_cvssp_subgroup, 'MAC_LIST'].unique().tolist()
    non_macs = (bound
                 .loc[bound.CHAIN_GROUP == 'NONPREF_OTH', 'MAC_LIST']
                 .unique()
                 .tolist())
    
    metric_calcs = {
     'CLAIMS': np.nansum,
     'MAC_LIST': lambda x: list(set(x)),  # need these to be list, not np.array
     'CURRENT_MAC_PRICE': np.nanmax,
     'INTERCEPT_LOW': np.nanmax,
     'INTERCEPT_HIGH': np.nanmin,
    }
    
    # columns to aggregate and rename when doing metric calculation
    rename_cvs = {
     'CURRENT_MAC_PRICE': 'CVS_PRICE_LB',
     'INTERCEPT_LOW': 'CVS_INT_LB',
     'CLAIMS': 'CVS_CLAIMS',
     'MAC_LIST': 'CVS_MAC'
    }
    
    rename_non = {
     'INTERCEPT_HIGH': 'CVS_NONPREF_UB',
     'CLAIMS': 'NON_CLAIMS',
     'MAC_LIST': 'NON_CVS_MAC'
    }
    
    # metric calculations: get the 
    cvs_metrics = (
     bound
      .loc[bound.MAC_LIST.isin(cvs_macs)]
      .groupby(cvs_groupers)
      .agg(**{rename_cvs[col]: (col, metric_calcs[col]) for col in rename_cvs})
    )
    
    non_metrics = (
     bound
      .loc[bound.MAC_LIST.isin(non_macs)]
      .groupby(cvs_groupers)
      .agg(**{rename_non[col]: (col, metric_calcs[col]) for col in rename_non})
    )
                 
    conflict = (cvs_metrics
                 .join(non_metrics)
                 .join(cvs_metrics
                        .join(non_metrics)
                        .melt(value_vars=['CVS_MAC', 'NON_CVS_MAC'],
                              var_name='CVS_MAC_TYPE',
                              value_name='MAC_LIST',
                              ignore_index=False))
                 .reset_index())
    
    conflict['CVS_PRICE_LB'] = conflict.CVS_PRICE_LB * (1 - p.GPI_LOW_FAC)
    conflict['CVS_LB'] = conflict[['CVS_PRICE_LB', 'CVS_INT_LB']].max(axis=1)
    
    if (conflict.CVS_LB >= conflict.CVS_NONPREF_UB).any():
        costs = (
         conflict
          .explode('MAC_LIST')
          .merge(bound, on=cvs_groupers + ['MAC_LIST'], how='left')
          .assign(TEMP_KEEP=CostsaverLogic.KEEP,
                  TEMP_SEND=CostsaverLogic.SEND,
                  SEND_COST=partial(calc_network_margin,
                                    actual_field='TEMP_SEND',
                                    qty=conflict.CVS_CLAIMS.max()),
                  KEEP_COST=partial(calc_network_margin,
                                    actual_field='TEMP_KEEP',
                                    qty=conflict.NON_CLAIMS.max()))
          .groupby(cvs_groupers)
          .agg({'SEND_COST': 'sum', 'KEEP_COST': 'sum'})
          # errs on the side of "KEEP" (i.e. when equal, it will be "KEEP")
          .merge(conflict, how='left', on=cvs_groupers)
          .assign(
            KEEP_SEND_NEW=
             lambda df: (df.SEND_COST < df.KEEP_COST).astype(float),
            CONFLICT_MACS=
             lambda df: (df.CVS_MAC + df.NON_CVS_MAC).map(', '.join),
          )
        )
        
        decision_cols = {'INTERCEPT_HIGH': 'UB_NEW', 'INTERCEPT_LOW': 'LB_NEW'}
        decisions = (
         costs
          .explode('MAC_LIST')
          .merge(bound, how='left', on=cvs_groupers + ['MAC_LIST'])
          .groupby(cvs_groupers + ['DESIRE_KEEP_SEND'])
          .agg(**{
            # rename if col in `decision_cols`
            decision_cols.get(col, col): (col, 'first')
            for col in list(decision_cols) + ['CONFLICT_MACS', 'KEEP_SEND_NEW']
          })
          .reset_index()
          .loc[lambda df: df.DESIRE_KEEP_SEND == df.KEEP_SEND_NEW]
          .assign(MAC_LIST=lambda df: df.CONFLICT_MACS.str.split(', '))
          .explode('MAC_LIST')
          .drop(columns=['CONFLICT_MACS', 'DESIRE_KEEP_SEND'])
        )
        
        bound = bound.merge(decisions, how='left', on=cvs_groupers + ['MAC_LIST'])
        
        conflict_mask = bound.KEEP_SEND_NEW.notna()
        decision_cols.update({'EXPECTED_KEEP_SEND': 'KEEP_SEND_NEW'})
        for oldcol, newcol in decision_cols.items():
            bound[oldcol] = bound[oldcol].mask(conflict_mask, bound[newcol])
            
        bound['INTERCEPT_REASON'] = np.where(conflict_mask,
                                             OverwriteReasons.CVS_MUST_BE_LOWER,
                                             bound.INTERCEPT_REASON)
        bound['VENDOR_CONFLICT'] = np.where(conflict_mask,
                                            True,
                                            bound.VENDOR_CONFLICT)
            
        bound.drop(columns=decision_cols.values(), inplace=True)
        
    return bound

#Copied over from CPMO.py line 4229
def value_reporting():
    if p.INTERCEPTOR_OPT:
        lp_data_output_df['awp'] = lp_data_output_df.PHARM_FULLAWP_ADJ + lp_data_output_df.PHARM_FULLAWP_ADJ_PROJ_LAG + lp_data_output_df.PHARM_FULLAWP_ADJ_PROJ_EOY
        lp_data_output_df['spend'] = lp_data_output_df.PHARM_PRICE_REIMB + lp_data_output_df.PHARM_LAG_REIMB + lp_data_output_df.Pharm_Price_Effective_Reimb_Proj
        lp_data_output_df[
            'awp_without'] = lp_data_output_df.PHARM_FULLAWP_ADJ + lp_data_output_df.PHARM_FULLAWP_ADJ_PROJ_LAG + lp_data_output_df.ORIG_PHARM_FULLAWP_ADJ_PROJ_EOY
        lp_data_output_df['spend_without'] = lp_data_output_df.PHARM_PRICE_REIMB + lp_data_output_df.PHARM_LAG_REIMB + (
                    lp_data_output_df.ORIG_PHARM_QTY_PROJ_EOY * lp_data_output_df.EFF_CAPPED_PRICE_new)

        value_df = lp_data_output_df.groupby(['CHAIN_GROUP']).agg({'awp':'sum',
                             'spend':'sum',
                             'awp_without':'sum',
                             'spend_without':'sum',
                             'ORIG_PHARM_FULLAWP_ADJ_PROJ_EOY':'sum'}).reset_index()

        value_df['ger'] = 1 - value_df.spend / value_df.awp
        value_df['ger_without'] = 1 - value_df.spend_without / value_df.awp_without

        value_df['value'] = (value_df.ger - value_df.ger_without) * value_df.ORIG_PHARM_FULLAWP_ADJ_PROJ_EOY

        value_df.to_csv(os.path.join(p.FILE_OUTPUT_PATH, p.INTERCEPTOR_VALUE_REPORT), index=False)

    else:
        lp_data_output_df['awp'] = None
        lp_data_output_df['spend'] = None
        lp_data_output_df['awp_without'] = None
        lp_data_output_df['spend_without'] = None
        lp_data_output_df['ger'] = None
        lp_data_output_df['ger_without'] = None
        lp_data_output_df['value'] = None
        
