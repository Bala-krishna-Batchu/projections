# -*- coding: utf-8 -*-
"""
"""
import os
import pandas as pd
import numpy as np
import calendar
import datetime as dt
import CPMO_parameters as p
from CPMO_shared_functions import unc_optimization, standardize_df, check_and_create_folder, determine_effective_price, calc_target_ingcost, add_target_ingcost, read_tru_mac_list_prices
from CPMO_goodrx_functions import goodrx_interceptor_optimization, correct_GRx_projections
import CPMO_costsaver_functions as cs
import util_funcs as uf
import BQ
import os
from qa_checks import qa_dataframe, cross_contract_utilization_check

pd.options.mode.chained_assignment = 'raise'

FILE_INPUT_PATH = os.path.join(p.PROGRAM_INPUT_PATH, '') 
PROGRAM_OUTPUT_PATH = p.PROGRAM_INPUT_PATH 

check_and_create_folder(p.FILE_INPUT_PATH)
check_and_create_folder(p.FILE_OUTPUT_PATH)
check_and_create_folder(p.FILE_DYNAMIC_INPUT_PATH)
check_and_create_folder(p.FILE_LP_PATH)

#================= DEFINE SOME HELPER FUNCTIONS ==============================
def pharmacy_type_new(row, pref_pharm_list):
    '''
    Takes dataframe row and determines if the chain group is preferrred or not 
    based on the supplied list of preferred pharmacies
    Inputs:
        Row of a dataframe that contains the CLIENT, BREAKOUT, and CHAIN_GROUP of that row
        A dataframe that has a list of preferred pharmacies based on client, breakout, & region
    Outputs:
        A single string of "Preferred" or "Non_Preferred"
    '''
    pref_pharms = pref_pharm_list.loc[
        (pref_pharm_list.CLIENT == row.CLIENT) &
        (pref_pharm_list.BREAKOUT == row.BREAKOUT) &
        (pref_pharm_list.REGION == row.REGION) &
        (pref_pharm_list.BG_FLAG == row.BG_FLAG), 'PREF_PHARMS'].values[0]

    if row.CHAIN_GROUP in pref_pharms:
        return 'Preferred'
    else:
        return 'Non_Preferred'


# TODO: Is is it still needed?
def clean_mac_1026_NDC(mac1026):
    '''
    Renames columns and can provide gpi level 1026 floors 
    if first several lines are uncommented
    '''
    mac1026.rename(index=str,columns = {'mac_cost_amt': 'MAC1026_UNIT_PRICE',
                                        'gpi':'GPI',
                                        'ndc':'NDC'},inplace=True)
    return mac1026

def read_gpi_vol_awp(READ_FROM_BQ=False):
    
    """Read initial daily total data from either Big Query or from file"""
    print('Reading in daily totals.........')
    if READ_FROM_BQ:
        gpi_vol_awp_df = uf.read_BQ_data(
            BQ.daily_totals_pharm,
            project_id = p.BQ_INPUT_PROJECT_ID,
            dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
            table_id = 'combined_daily_totals' + p.WS_SUFFIX + p.CCP_SUFFIX,
            locked_fix_up = (p.LOCKED_CLIENT and p.TREAT_TRANSPARENT),
            client = ', '.join(sorted(p.CUSTOMER_ID)),
            claim_start = p.DATA_START_DAY,
            claim_date = p.LAST_DATA.strftime('%m/%d/%Y'),
        )
    else:
        gpi_vol_awp_df = pd.read_csv(p.FILE_INPUT_PATH + p.DAILY_TOTALS_FILE, dtype = p.VARIABLE_TYPE_DIC)
    gpi_vol_awp_df.drop(columns = ['MEMBER_COST'], inplace = True) # Remove plan liability columns because they're not in UNC
    gpi_vol_awp_df = standardize_df(gpi_vol_awp_df)
    
    if p.LOCKED_CLIENT and p.TREAT_TRANSPARENT:
        gpi_vol_awp_df = gpi_vol_awp_df[gpi_vol_awp_df['QTY'] > 0].copy()

    # For backwards compatibility before CHAIN_SUBGROUP was introduced
    if 'CHAIN_SUBGROUP' not in gpi_vol_awp_df.columns:
        gpi_vol_awp_df['CHAIN_SUBGROUP'] = gpi_vol_awp_df['CHAIN_GROUP']
    nanlist = list(gpi_vol_awp_df.columns.drop([
        "UCAMT_UNIT",
        "PCT25_UCAMT_UNIT",
        "PHARMACY_PCT25_UCAMT_UNIT",
        "PHARMACY_PCT50_UCAMT_UNIT",
        "CLIENT_PCT25_UCAMT_UNIT",
        "CLIENT_PCT50_UCAMT_UNIT",
        "PHARMACY_ACC","PHARMACY_ACC_ZBD","TARGET_DISP_FEE_ZBD",
        "PHARMACY_TARGET_DISP_FEE_ZBD","PHARMACY_TARGET_IC_COST",
        "PHARMACY_TARGET_IC_COST_ZBD", "PHARM_TARGET_DISP_FEE", 
        "PHARMACY_NADAC", "PHARMACY_NADAC_ZBD","PHRM_GRTE_TYPE"
        ]))
    qa_dataframe(gpi_vol_awp_df, dataset = 'DAILY_TOTALS_FILE_AT_{}'.format(os.path.basename(__file__)),nanlist = nanlist)
    ## TO DO: Remove the following checks when TC is supported on e2r
    if not gpi_vol_awp_df[gpi_vol_awp_df.PHRM_GRTE_TYPE == 'NADAC'].empty:
        assert gpi_vol_awp_df[gpi_vol_awp_df.PHRM_GRTE_TYPE == 'NADAC'][['PHARMACY_NADAC','PHARMACY_NADAC_ZBD']].isna().sum().sum() == 0, 'PHARMACY_NADAC or PHARMACY_NADAC_ZBD columns cannot contain nulls for gpi_vol_awp_df'
    if not gpi_vol_awp_df[gpi_vol_awp_df.PHRM_GRTE_TYPE == 'ACC'].empty:
        assert gpi_vol_awp_df[gpi_vol_awp_df.PHRM_GRTE_TYPE == 'ACC'][['PHARMACY_ACC','PHARMACY_ACC_ZBD']].isna().sum().sum() == 0, 'PHARMACY_ACC or PHARMACY_ACC_ZBD columns cannot contain nulls for gpi_vol_awp_df'
    ## TO DO ENDS#################
    
    gpi_vol_awp_df.drop(columns=['PHRM_GRTE_TYPE'], inplace=True)                
    return gpi_vol_awp_df

def read_gpi_vol_awp_unc(READ_FROM_BQ=False):
    
    if READ_FROM_BQ:
        gpi_vol_awp_unc_df = uf.read_BQ_data(
            BQ.ger_opt_unc_daily_total,
            project_id = p.BQ_INPUT_PROJECT_ID,
            dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
            table_id = 'ger_opt_unc_daily_total' + p.WS_SUFFIX + p.CCP_SUFFIX,
            locked_fix_up = (p.LOCKED_CLIENT and p.TREAT_TRANSPARENT),
            client = ', '.join(sorted(p.CUSTOMER_ID)),
            claim_start = p.DATA_START_DAY,
            claim_date = p.LAST_DATA.strftime('%m/%d/%Y')           
        )
    else:
        gpi_vol_awp_unc_df = pd.read_csv(p.FILE_INPUT_PATH + p.UNC_DAILY_TOTALS_FILE, dtype = p.VARIABLE_TYPE_DIC)
        
    gpi_vol_awp_unc_df.drop(columns = ['MEMBER_COST'], inplace = True)
    gpi_vol_awp_unc_df = standardize_df(gpi_vol_awp_unc_df)   
    # For backwards compatibility before CHAIN_SUBGROUP was introduced
    if 'CHAIN_SUBGROUP' not in gpi_vol_awp_unc_df.columns:
        gpi_vol_awp_unc_df['CHAIN_SUBGROUP'] = gpi_vol_awp_unc_df['CHAIN_GROUP']
    nanlist = list(gpi_vol_awp_unc_df.columns.drop([
        "UCAMT_UNIT",
        "PCT25_UCAMT_UNIT",
        "PHARMACY_PCT25_UCAMT_UNIT",
        "PHARMACY_PCT50_UCAMT_UNIT",
        "CLIENT_PCT25_UCAMT_UNIT",
        "CLIENT_PCT50_UCAMT_UNIT",
        "PHARMACY_ACC","PHARMACY_ACC_ZBD","TARGET_DISP_FEE_ZBD",
        "PHARMACY_TARGET_DISP_FEE_ZBD","PHARMACY_TARGET_IC_COST",
        "PHARMACY_TARGET_IC_COST_ZBD","PHARM_TARGET_DISP_FEE",
        "PHARMACY_NADAC","PHARMACY_NADAC_ZBD","PHRM_GRTE_TYPE"
        ]))
    qa_dataframe(gpi_vol_awp_unc_df, dataset = 'DAILY_TOTALS_FILE_AT_{}'.format(os.path.basename(__file__)),nanlist = nanlist)
    ## TO DO: Remove the following checks when TC is supported on e2r
    if not gpi_vol_awp_unc_df[gpi_vol_awp_unc_df.PHRM_GRTE_TYPE == 'NADAC'].empty:
        assert gpi_vol_awp_unc_df[gpi_vol_awp_unc_df.PHRM_GRTE_TYPE == 'NADAC'][['PHARMACY_NADAC','PHARMACY_NADAC_ZBD']].isna().sum().sum() == 0, 'PHARMACY_NADAC or PHARMACY_NADAC_ZBD columns cannot contain nulls for gpi_vol_awp_df'
    if not gpi_vol_awp_unc_df[gpi_vol_awp_unc_df.PHRM_GRTE_TYPE == 'ACC'].empty:
        assert gpi_vol_awp_unc_df[gpi_vol_awp_unc_df.PHRM_GRTE_TYPE == 'ACC'][['PHARMACY_ACC','PHARMACY_ACC_ZBD']].isna().sum().sum() == 0, 'PHARMACY_ACC or PHARMACY_ACC_ZBD columns cannot contain nulls for gpi_vol_awp_df'
    ## TO DO ENDS#################

    #If there are U&C claims, quality check the dataframe, and run through U&C Optimization. 
    #If there are no U&C claims, skipping the dataframe QA since it's empty, and the code can still run through U&C Optimization without interruption while UNC_OPT=True. 
    if len(gpi_vol_awp_unc_df) > 0:
        qa_dataframe(gpi_vol_awp_unc_df, dataset = 'UNC_DAILY_TOTALS_FILE_AT_{}'.format(os.path.basename(__file__)), nanlist = nanlist)
    else:
        print('WARNING: No UNC claims.')  
    
    gpi_vol_awp_unc_df.drop(columns=['PHRM_GRTE_TYPE'], inplace=True)                
    return gpi_vol_awp_unc_df  

def read_costvantage_price(READ_FROM_BQ=False):
    if READ_FROM_BQ:
        costvantage_df = uf.read_BQ_data(
            BQ.costvantage_price,
            project_id = p.BQ_INPUT_PROJECT_ID,
            dataset_id = p.BQ_INPUT_DATASET_ENT_CNFV_PROD,
            table_id = 'gms_gms_pgm_prm_68_cst_vantage_cost_file',
            custom=True
        )
    else:
        costvantage_df = pd.read_csv(p.FILE_INPUT_PATH + p.COSTVANTAGE_FILE, dtype = p.VARIABLE_TYPE_DIC)
    costvantage_df = costvantage_df.rename(columns={'GPI14':'GPI','PHARMACY_GROUP':'CHAIN_GROUP'})
    return costvantage_df

def read_nadac_wac_gpi_price(READ_FROM_BQ=False):
    nadac_wac_df = read_nadac_wac(READ_FROM_BQ)
    nadac_wac_df['NADAC'] = nadac_wac_df['NADAC'].astype('float')
    nadac_wac_df['WAC'] = nadac_wac_df['WAC'].astype('float')
    
    # BNCHMK only became a thing in 2025, whereas nadac is a thing since 2024
    if p.GO_LIVE.year >= 2025: 
        nadac_wac_df['BNCHMK'] = nadac_wac_df['BNCHMK'].astype('float')
    
    if p.GENERIC_OPT and p.BRAND_OPT:
        nadac_wac_df = nadac_wac_df.loc[(nadac_wac_df.NDC =='***********')]
    elif p.GENERIC_OPT and not p.BRAND_OPT:
        nadac_wac_df = nadac_wac_df.loc[(nadac_wac_df.NDC =='***********') & (nadac_wac_df.BG_FLAG =='G')]
    else:
        nadac_wac_df = nadac_wac_df.loc[(nadac_wac_df.NDC =='***********') & (nadac_wac_df.BG_FLAG =='B')]
    nadac_wac_df['PHARM_AVG_NADAC'] = nadac_wac_df['NADAC'].combine_first(nadac_wac_df['WAC'])
    nadac_wac_df = nadac_wac_df.rename(columns={'CUSTOMER_ID':'CLIENT','WAC':'PHARM_AVG_WAC'})
    
    return nadac_wac_df[['CLIENT','GPI','BG_FLAG','PHARM_AVG_NADAC','PHARM_AVG_WAC']] #to-do: Add bg_flag later

def read_eoy_proj(READ_FROM_BQ=False, table_id = 'gms_eoy_proj'):  
    '''
    Read projections from BQ tables
        Inputs: 
            Flag to either read data from BQ tables or from input path
            Table name - EOY_PROJ has claims only from combined totals + UNC. 
                         Data gets filtered for just actuals based on UNC_OPT parameter
        Outputs:
            Dataframe with claims, qty and awp projections at week level for the given client along with other columns      
    '''
    print('Reading in weekly projections data.........')
    if READ_FROM_BQ:
        eoy_proj_week_df = uf.read_BQ_data(
            BQ.eoy_proj_query,
            project_id = p.BQ_INPUT_PROJECT_ID,
            dataset_id = p.BQ_INPUT_DATASET_ENT_ENRV_PROD,
            table_id = table_id,
            unc_included = p.UNC_OPT,
            proj_filter = True,
            client = ', '.join(sorted(p.CUSTOMER_ID)))
    else:
        eoy_proj_week_df = pd.read_csv(p.FILE_INPUT_PATH + p.EOY_PROJ_FILE, dtype = p.VARIABLE_TYPE_DIC)
        
    eoy_proj_week_df = standardize_df(eoy_proj_week_df)
    qa_dataframe(eoy_proj_week_df, dataset = 'EOY_PROJ_FILE_AT_{}'.format(os.path.basename(__file__)))

    return eoy_proj_week_df

def calculate_proj(eoy_proj_df, lag_days, eoy_days):

    lag_factor = (lag_days/7)
    eoy_factor = (eoy_days/7)
    
    # Multiply projections by the duration of the lag period duration
    eoy_proj_df['CLAIMS_PROJ_LAG'] = eoy_proj_df['CLAIMS_PROJ_WEEK'] * lag_factor
    eoy_proj_df['QTY_PROJ_LAG'] = eoy_proj_df['QTY_PROJ_WEEK'] * lag_factor
    eoy_proj_df['FULLAWP_ADJ_PROJ_LAG'] = eoy_proj_df['FULLAWP_ADJ_PROJ_WEEK'] * lag_factor

    # Multiply projections by the duration of the go-live date to end of year
    eoy_proj_df['CLAIMS_PROJ_EOY'] = eoy_proj_df['CLAIMS_PROJ_WEEK'] * eoy_factor
    eoy_proj_df['QTY_PROJ_EOY'] = eoy_proj_df['QTY_PROJ_WEEK'] * eoy_factor
    eoy_proj_df['FULLAWP_ADJ_PROJ_EOY'] = eoy_proj_df['FULLAWP_ADJ_PROJ_WEEK'] * eoy_factor

    # Multiply projections by the duration of the lag period duration
    eoy_proj_df['PHARM_CLAIMS_PROJ_LAG'] = eoy_proj_df['PHARM_CLAIMS_PROJ_WEEK'] * lag_factor
    eoy_proj_df['PHARM_QTY_PROJ_LAG'] = eoy_proj_df['PHARM_QTY_PROJ_WEEK'] * lag_factor
    eoy_proj_df['PHARM_FULLAWP_ADJ_PROJ_LAG'] = eoy_proj_df['PHARM_FULLAWP_ADJ_PROJ_WEEK'] * lag_factor
    
    # Multiply projections by the duration of the go-live date to end of year
    eoy_proj_df['PHARM_CLAIMS_PROJ_EOY'] = eoy_proj_df['PHARM_CLAIMS_PROJ_WEEK'] * eoy_factor
    eoy_proj_df['PHARM_QTY_PROJ_EOY'] = eoy_proj_df['PHARM_QTY_PROJ_WEEK'] * eoy_factor
    eoy_proj_df['PHARM_FULLAWP_ADJ_PROJ_EOY'] = eoy_proj_df['PHARM_FULLAWP_ADJ_PROJ_WEEK'] * eoy_factor
    
    return eoy_proj_df

#read awp history
def read_current_awp(READ_FROM_BQ = False):

    if READ_FROM_BQ:
        
        curr_awp_df = uf.read_BQ_data(
            BQ.awp_history_table, 
            project_id = p.BQ_INPUT_PROJECT_ID, 
            dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
            table_id = 'awp_history_table' + p.WS_SUFFIX)
        
    else:
        curr_awp_df = pd.read_csv(p.FILE_INPUT_PATH + p.AWP_FILE, dtype = p.VARIABLE_TYPE_DIC)
    curr_awp_df = standardize_df(curr_awp_df)
#     qa_dataframe(curr_awp_df, dataset = 'AWP_FILE_AT_{}'.format(os.path.basename(__file__)))
    
    curr_awp_df = curr_awp_df.sort_values(
        by=['GPI_NDC','SRC_CD', 'EFF_DATE'], ascending=[True,True,False])
    curr_awp_df = curr_awp_df.drop_duplicates(subset=['GPI_NDC'], keep='first')
    assert len(curr_awp_df.loc[
        curr_awp_df.duplicated(subset=['GPI_NDC'], keep=False)]) == 0,\
        "len(curr_awp_df.loc[curr_awp_df.duplicated(subset=['GPI_NDC'], keep=False)]) == 0"
    curr_awp_df.rename(columns={'DRUG_PRICE_AT': 'CURR_AWP'}, inplace=True)    
    
    return curr_awp_df

#======read MAC List
def read_mac_lists(READ_FROM_BQ=False):

    if READ_FROM_BQ:
        
        mac_list_df = uf.read_BQ_data(
            BQ.mac_list, 
            project_id = p.BQ_INPUT_PROJECT_ID, 
            dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP, 
            table_id = 'mac_list', #+ p.WS_SUFFIX, --- We do not need the welcome season table for the mac_list table as we need the most updated drug costs & WS tables do not get updated often
            customer = ', '.join(sorted(p.CUSTOMER_ID)),  
            mac = True,
            vcml_ref_table_id='vcml_reference') 
        
    else:
        mac_list_df = pd.read_csv(p.FILE_INPUT_PATH + p.MAC_LIST_FILE, dtype = p.VARIABLE_TYPE_DIC)
    
    if p.TRUECOST_CLIENT or p.UCL_CLIENT:
        assert p.READ_FROM_BQ == True, "Use p.READ_FROM_BQ=True to read table"
        mac_list_df = read_tru_mac_list_prices()


    mac_list_df = standardize_df(mac_list_df)
    qa_dataframe(mac_list_df, dataset = 'MAC_LIST_FILE_AT_{}'.format(os.path.basename(__file__)))
    
    #Remove specialTy vcmls from mac_lists cause we are not going to price them
    sprx_vcml_suffix = ['S3','S9','SX']
    mac_list_df = mac_list_df.loc[~mac_list_df['MAC'].astype(str).str[-2:].isin(sprx_vcml_suffix)]
    
    #Some checks
    mac_list_df = mac_list_df.drop_duplicates(subset=['MAC', 'BG_FLAG', 'GPI_NDC'])
    assert len(mac_list_df.loc[mac_list_df['GPI'].str.len() == 14, 'GPI']) ==\
        len(mac_list_df.GPI), "GPIs found with length < 14" 
    assert len(mac_list_df.drop_duplicates(subset=['GPI', 'BG_FLAG', 'MAC', 'NDC'])) == \
        len(mac_list_df.GPI), "Duplicates [GPI,MAC,NDC] found in mac list"
    
    #determine which GPIs are at NDC prices or GPI prices 
    mac_list_df['NDC_Count'] = 1
    old_len = len(mac_list_df)
    mac_ndc = mac_list_df.groupby(['MAC', 'BG_FLAG', 'GPI'])['NDC_Count']\
        .agg(sum).reset_index()
    mac_list_df = pd.merge(
        mac_list_df.drop(columns=['NDC_Count']), 
        mac_ndc,how='left',on=['MAC', 'BG_FLAG', 'GPI'] )
    assert len(mac_list_df) == old_len, "len(mac_list_df) == old_len"
    
    return mac_list_df



def read_chain_region_mac_mapping(READ_FROM_BQ=False):
    
    chain_region_mac_mapping = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_MAPPING_FILE, dtype = p.VARIABLE_TYPE_DIC)  
    chain_region_mac_mapping = standardize_df(chain_region_mac_mapping)
    qa_dataframe(chain_region_mac_mapping, dataset = 'MAC_MAPPING_FILE_AT_{}'.format(os.path.basename(__file__)))
    # For backwards compatibility before CHAIN_SUBGROUP was introduced
    if 'CHAIN_SUBGROUP' not in chain_region_mac_mapping.columns:
        chain_region_mac_mapping['CHAIN_SUBGROUP'] = chain_region_mac_mapping['CHAIN_GROUP']
    
    
    return chain_region_mac_mapping


#Read in package sizes ==================== 
def read_package_size(READ_FROM_BQ=False):
    
    if READ_FROM_BQ:
        
        pkg_size = uf.read_BQ_data(
            BQ.package_size_to_ndc, 
            project_id = p.BQ_INPUT_PROJECT_ID, 
            dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP, 
            table_id = 'package_size_to_ndc' + p.WS_SUFFIX) 

    else:
        pkg_size = pd.read_csv(p.FILE_INPUT_PATH + p.PACKAGE_SIZE_FILE, dtype = p.VARIABLE_TYPE_DIC) 

    pkg_size.rename(columns={'PACK_SIZE': 'PKG_SZ','NDC11': 'NDC'}, inplace=True)
    pkg_size = pkg_size.fillna(0)
    pkg_size = standardize_df(pkg_size)
    # qa_dataframe(pkg_size, dataset = 'PACKAGE_SIZE_FILE_AT_{}'.format(os.path.basename(__file__)))
    
    return pkg_size

#read in 1026
def read_mac_1026(READ_FROM_BQ=False):

    mac1026_df = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC1026_FILE, dtype = p.VARIABLE_TYPE_DIC)
    mac1026_df = mac1026_df.rename(
        index=str,
        columns = {
            'mac_cost_amt': 'MAC1026_UNIT_PRICE',
            'gpi':'GPI',
            'ndc':'NDC'
        })
    #mac1026_df = clean_mac_1026_NDC(mac1026_df)
    mac1026_df = mac1026_df.drop_duplicates(subset=['MAC_LIST', 'BG_FLAG', 'GPI','NDC'])
    mac1026_df = standardize_df(mac1026_df)
    qa_dataframe(mac1026_df.drop(columns=['MAC_EFF_DT']), dataset='MAC1026_FILE_AT_{}'.format(os.path.basename(__file__)))
    mac_1026_gpi = mac1026_df.loc[mac1026_df.NDC == '***********'].copy(deep=True)
    mac_1026_gpi.rename(columns={'PRICE': '1026_GPI_PRICE'}, inplace=True)
    mac_1026_ndc = mac1026_df.loc[mac1026_df.NDC != '***********'].copy(deep=True)
    mac_1026_ndc.rename(columns={'PRICE': '1026_NDC_PRICE'}, inplace=True)
    
    return mac_1026_gpi, mac_1026_ndc

#Read in preferred pharmacy list 
def read_pref_pharm_list(READ_FROM_BQ = False):
   
    pref_pharm_list = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.PREFERRED_PHARM_FILE, dtype = p.VARIABLE_TYPE_DIC)
    pref_pharm_list = standardize_df(pref_pharm_list)
    qa_dataframe(pref_pharm_list, dataset = 'PREFERRED_PHARM_FILE_AT_{}'.format(os.path.basename(__file__)))
    pref_pharm_list['PREF_PHARMS'] = pref_pharm_list.PREF_PHARM.apply(lambda x: x.split(','))
    
    return pref_pharm_list


#Read unc_NDC_percentiles
def read_unc_NDC_percentiles(READ_FROM_BQ=False):
    # Read in U&C Price percentiles -- removed from daily_total files
    if READ_FROM_BQ:
        unc_NDC_percentiles = uf.read_BQ_data(
            BQ.ger_opt_unc_ndc_percentiles_constrained, 
            project_id =  p.BQ_INPUT_PROJECT_ID, 
            dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
            client = ', '.join(sorted(p.CUSTOMER_ID)),
            table_id = 'ger_opt_unc_ndc_percentiles_constrained' + p.WS_SUFFIX + p.CCP_SUFFIX
        )
    else:
        unc_NDC_percentiles = pd.read_csv(p.FILE_INPUT_PATH + p.UNC_NDC_PERCENTILES_FILE, dtype = p.VARIABLE_TYPE_DIC)
    unc_NDC_percentiles = standardize_df(unc_NDC_percentiles)\
        .rename(columns = {'PREFERRED': 'PHARMACY_TYPE', 'CLAIMS': 'UC_PERCENTILE_CLAIMS'})  
    # Remove MAIL claims, which do not have a viable PHARMACY_TYPE
    unc_NDC_percentiles = unc_NDC_percentiles[unc_NDC_percentiles.MEASUREMENT.str[0]!='M']
    qa_dataframe(unc_NDC_percentiles, dataset = 'UNC_NDC_PERCENTILES_FILE_AT_{}'.format(os.path.basename(__file__)))
    unc_NDC_percentiles['PHARMACY_TYPE'] = unc_NDC_percentiles['PHARMACY_TYPE']\
        .str.replace('Non-preferred', 'Non_Preferred')

    return unc_NDC_percentiles


def read_unc_GPI_percentiles(READ_FROM_BQ=False):
    
    if READ_FROM_BQ:
        unc_GPI_percentiles = uf.read_BQ_data(
            BQ.ger_opt_unc_gpi_percentiles_constrained, 
            project_id = p.BQ_INPUT_PROJECT_ID,
            dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
            client = ', '.join(sorted(p.CUSTOMER_ID)),
            table_id = 'ger_opt_unc_gpi_percentiles_constrained' + p.WS_SUFFIX + p.CCP_SUFFIX
        )
    else:
        unc_GPI_percentiles = pd.read_csv(p.FILE_INPUT_PATH + p.UNC_GPI_PERCENTILES_FILE, dtype = p.VARIABLE_TYPE_DIC)
    unc_GPI_percentiles = standardize_df(unc_GPI_percentiles).\
        rename(columns = {'PREFERRED': 'PHARMACY_TYPE', 'CLAIMS': 'UC_PERCENTILE_CLAIMS'})
    # Remove MAIL claims, which do not have a viable PHARMACY_TYPE
    unc_GPI_percentiles = unc_GPI_percentiles[unc_GPI_percentiles.MEASUREMENT.str[0]!='M']
    qa_dataframe(unc_GPI_percentiles, dataset = 'UNC_GPI_PERCENTILES_FILE_AT_{}'.format(os.path.basename(__file__)))

    unc_GPI_percentiles['PHARMACY_TYPE'] = unc_GPI_percentiles['PHARMACY_TYPE']\
        .str.replace('Non-preferred', 'Non_Preferred')

    return unc_GPI_percentiles



# Read in Client and Pharmacy Guarantees, and exclusion lists
def read_client_guarantees(READ_FROM_BQ=False):
    from CPMO_shared_functions import add_virtual_r90
    
    # client guarantees - used to identify VCML_ID combinations without guarantees.
    client_guarantees = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.CLIENT_GUARANTEE_FILE, dtype = p.VARIABLE_TYPE_DIC).drop_duplicates()
    client_guarantees = standardize_df(client_guarantees)
    
    if p.CLIENT_TYPE == 'MEDD' and p.FULL_YEAR and not p.EGWP:
        client_guarantees = standardize_df(pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.CLIENT_GUARANTEE_FILE), dtype = p.VARIABLE_TYPE_DIC))
    
    qa_dataframe(client_guarantees, dataset = 'CLIENT_GUARANTEE_FILE_AT_{}'.format(os.path.basename(__file__)))

    client_guarantees.columns = map(str.upper, client_guarantees.columns)
    client_guarantees['REGION'] = client_guarantees['CLIENT']
    client_guarantees = client_guarantees.rename(columns = {'RATE': 'CLIENT_RATE'})
    
    return client_guarantees


def read_pharmacy_guarantees(READ_FROM_BQ=False):
    
    pharmacy_guarantees = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.PHARM_GUARANTEE_FILE, dtype = p.VARIABLE_TYPE_DIC) 
    pharmacy_guarantees.columns = map(str.upper, pharmacy_guarantees.columns)
    pharmacy_guarantees = standardize_df(pharmacy_guarantees)\
            .rename(columns = {'PHARMACY':'CHAIN_GROUP','PHARMACY_SUB':'CHAIN_SUBGROUP','RATE':'PHARMACY_RATE'})
    qa_dataframe(pharmacy_guarantees, dataset = 'PHARM_GUARANTEE_FILE_AT_{}'.format(os.path.basename(__file__)))

    return pharmacy_guarantees


def read_exclusions(READ_FROM_BQ=False):
    
    exclusions = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.SPECIALTY_EXCLUSION_FILE, dtype = p.VARIABLE_TYPE_DIC)
    exclusions = standardize_df(exclusions)
    qa_dataframe(exclusions, dataset = 'SPECIALTY_EXCLUSION_FILE_AT_{}'.format(os.path.basename(__file__)))

    
    return exclusions

def read_unc_exclusions(READ_FROM_BQ=False):
  
    unc_exclusions = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_PRICE_OVERRIDE_FILE, dtype = p.VARIABLE_TYPE_DIC)
    unc_exclusions = standardize_df(unc_exclusions)
    if p.CLIENT_LOB != 'AETNA': 
        qa_dataframe(unc_exclusions, dataset = 'MAC_PRICE_OVERRIDE_FILE_AT_{}'.format(os.path.basename(__file__)))

    
    return unc_exclusions

def read_copay_coinsurance(READ_FROM_BQ=False, level=''):
    if READ_FROM_BQ:
        copay_coins_df = uf.read_BQ_data(
            query = BQ.full_table,
            project_id =  p.BQ_INPUT_PROJECT_ID, 
            dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
            table_id = 'plan_design_{}'.format(level) + p.WS_SUFFIX,
            customer = ', '.join(sorted(p.CUSTOMER_ID)))
    else:
        if level == 'gpi':
            copay_coins_df = pd.read_csv(p.FILE_INPUT_PATH + p.PLAN_DESIGN_FILE_GPI, dtype = p.VARIABLE_TYPE_DIC)
        if level == 'ndc':
            copay_coins_df = pd.read_csv(p.FILE_INPUT_PATH + p.PLAN_DESIGN_FILE_NDC, dtype = p.VARIABLE_TYPE_DIC)
        
    copay_coins_df = standardize_df(copay_coins_df).rename(columns = {'PREFERRED':'PHARMACY_TYPE', 'CUSTOMER_ID':'CLIENT'}) 
    
    return copay_coins_df

def read_nadac_wac(READ_FROM_BQ=False):
    """Read nadac wac data from either Big Query or from file"""
    print('Reading in nadac wac data.........')
    if READ_FROM_BQ:
        nadac_wac = uf.read_BQ_data(
            BQ.nadac_wac,
            project_id=p.BQ_INPUT_PROJECT_ID,
            dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
            table_id="nadac_wac_price" + p.WS_SUFFIX,
            customer = ', '.join(sorted(p.CUSTOMER_ID))
        )
    else:
        nadac_wac = pd.read_csv(p.FILE_INPUT_PATH + p.RAW_NADAC_WAC_FILE, dtype = p.VARIABLE_TYPE_DIC)
    
    # qa_dataframe(nadac_wac, dataset = 'NADAC_WAC_FILE_AT_{}'.format(os.path.basename(__file__)))
    nadac_wac['BRND_GNRC_CD'].replace({'GNRC':'G','BRND':'B'}, inplace=True)
    nadac_wac = nadac_wac.rename(columns={'BRND_GNRC_CD':'BG_FLAG'})
    nadac_wac = standardize_df(nadac_wac)
    return nadac_wac

def read_net_cost_guarantee(READ_FROM_BQ=False):
    """Read net cost guarantee data from only Big Query"""
    print('Reading in net cost guarantees.........')
    assert p.READ_FROM_BQ == True, "Use p.READ_FROM_BQ=True to read table"
    net_cost_guarantee_df = uf.read_BQ_data(
             BQ.net_cost_guarantee_custom.format(_customer_id = uf.get_formatted_string(p.CUSTOMER_ID),
                                          _contract_eff_date = p.GO_LIVE.date()),
             project_id = p.BQ_OUTPUT_PROJECT_ID,
             dataset_id = p.BQ_INPUT_DATASET_SANDBOX,
             table_id = 'gms_truecost_gpi_14_drug_list',
             unc_included = p.UNC_OPT,
             customer = ', '.join(sorted(p.CUSTOMER_ID)),
             custom = True
             )
    net_cost_guarantee_df = standardize_df(net_cost_guarantee_df)
    qa_dataframe(net_cost_guarantee_df, dataset = 'NET_COST_GUARANTEE_FILE_AT_{}'.format(os.path.basename(__file__)))
    net_cost_guarantee_df['PCD_IDX'] = '1'
    return net_cost_guarantee_df
        
def read_benchmark_nadac_prices(READ_FROM_BQ=False):
    """Read in benchmark NADAC prices to use for ceilings"""
    print('Reading in benchmark NADAC prices......')
    assert READ_FROM_BQ == True, "Use p.READ_FROM_BQ=True to read table"
    benchmark_nadac_prices_df = uf.read_BQ_data(
             BQ.benchmark_nadac_price,
             project_id = p.BQ_OUTPUT_PROJECT_ID,
             dataset_id = p.BQ_INPUT_DATASET_SANDBOX,
             table_id = 'benchmark_nadac_cp_v2')
    benchmark_nadac_prices_df = standardize_df(benchmark_nadac_prices_df)
    benchmark_nadac_prices_df = benchmark_nadac_prices_df.drop(columns = ['GPI','NDC'])
    benchmark_nadac_prices_df = benchmark_nadac_prices_df.loc[(benchmark_nadac_prices_df['BENCHMARK_CEILING_PRICE'] > 0)]
    qa_dataframe(benchmark_nadac_prices_df, dataset = 'benchmark_nadac_prices_df')
    return benchmark_nadac_prices_df
    
    
##############################################################################
#===== Done reading files, now do some pre-processing to consolidate data=====
def pre_process_data(gpi_vol_awp_df,dataset=""):
    
    gpi_vol_awp_df = gpi_vol_awp_df.dropna(how='all') 
    na_rows = gpi_vol_awp_df.loc[gpi_vol_awp_df.CLIENT.isna()]
    assert len(na_rows) < 5, "No more than 5 rows should ever be blank "
    gpi_vol_awp_df = gpi_vol_awp_df.drop(index = na_rows.index)
    
    # TODO: Ask what to do whit this client on all the if statements that it has.
    # Wellmark Customer Name cleaning
    # ========================================================================
    # if p.client_name == 'WELLMARK':
    #     gpi_vol_awp_df['CLIENT'] = 4843
    #     gpi_vol_awp_df['REGION'] = 4843
    # =========================================================================

    #rename and drop some of the columns
    gpi_vol_awp_df.rename(columns={'NDC11': 'NDC',
                                   'AWP': 'FULLAWP_ADJ',
                                   'PHARMACY_AWP': 'PHARM_FULLAWP_ADJ',
                                   'PHARMACY_NADAC': 'PHARM_FULLNADAC_ADJ',
                                   'PHARMACY_ACC': 'PHARM_FULLACC_ADJ',
                                   'PHARMACY_TARGET_IC_COST': 'PHARM_TARG_INGCOST_ADJ',
                                   'AWP_ZBD': 'FULLAWP_ADJ_ZBD',
                                   'PHARMACY_AWP_ZBD': 'PHARM_FULLAWP_ADJ_ZBD',
                                   'PHARMACY_NADAC_ZBD': 'PHARM_FULLNADAC_ADJ_ZBD',
                                   'PHARMACY_ACC_ZBD': 'PHARM_FULLACC_ADJ_ZBD',
                                   'PHARMACY_TARGET_IC_COST_ZBD': 'PHARM_TARG_INGCOST_ADJ_ZBD',
                                   'CLAIM_DATE': 'DOF',
                                   'SPEND': 'PRICE_REIMB',
                                   'PHARMACY_SPEND': 'PHARM_PRICE_REIMB',
                                   'SPEND_ZBD': 'PRICE_REIMB_ZBD',
                                   'PHARMACY_SPEND_ZBD': 'PHARM_PRICE_REIMB_ZBD',
                                   'UCAMT_UNIT': 'UC_UNIT',
                                   'PCT25_UCAMT_UNIT': 'UC_UNIT25',
                                   'PHARMACY_CLAIMS': 'PHARM_CLAIMS',
                                   'PHARMACY_QTY': 'PHARM_QTY',
                                   'PHARMACY_CLAIMS_ZBD': 'PHARM_CLAIMS_ZBD',
                                   'PHARMACY_QTY_ZBD': 'PHARM_QTY_ZBD'}, inplace=True)
    
    gpi_vol_awp_df.drop(columns=[
        'CUSTOMER_ID', 'MAILIND', 'GENIND', 'REC_CURR_IND', 'REC_ADD_USER',
        'REC_ADD_TS', 'REC_CHG_USER', 'REC_CHG_USER', 'UNIQUE_ROW_ID'],
                                inplace=True, errors='ignore')
    gpi_vol_awp_df = standardize_df(gpi_vol_awp_df)
    
    nanlist = list(gpi_vol_awp_df.columns.drop(["UC_UNIT","UC_UNIT25","PHARMACY_PCT25_UCAMT_UNIT","PHARMACY_PCT50_UCAMT_UNIT","CLIENT_PCT25_UCAMT_UNIT","CLIENT_PCT50_UCAMT_UNIT","PHARM_FULLACC_ADJ","PHARM_TARG_INGCOST_ADJ","PHARM_FULLACC_ADJ_ZBD","PHARM_TARG_INGCOST_ADJ_ZBD","TARGET_DISP_FEE_ZBD","PHARMACY_TARGET_DISP_FEE_ZBD", "PHARM_TARGET_DISP_FEE","PHARM_FULLNADAC_ADJ","PHARM_FULLNADAC_ADJ_ZBD"]))
    qa_dataframe(gpi_vol_awp_df, dataset = 'gpi_vol_awp_df_{}AT_{}'.format(dataset,os.path.basename(__file__)),nanlist = nanlist)
    
    # TODO: From PRO, do we stil need it?
    gpi_vol_awp_df.loc[
        (gpi_vol_awp_df['CHAIN_GROUP']=='NONPREF_OTH') & 
        (gpi_vol_awp_df['BREAKOUT'].str.contains('M')),'CHAIN_GROUP'] = 'MAIL'

    
    ## convert DOF column to datetime
    #gpi_vol_awp_df['DOF'] = pd.to_datetime(gpi_vol_awp_df.DOF, format='%Y-%m-%d')
    gpi_vol_awp_df['DOF'] = pd.to_datetime(gpi_vol_awp_df.DOF) 
    gpi_vol_awp_df['DOF_MONTH'] = gpi_vol_awp_df.DOF.dt.month
    
    #This will be used later to get average awp for awp discount
    gpi_vol_awp_df['daily_avg_awp'] = gpi_vol_awp_df.FULLAWP_ADJ / gpi_vol_awp_df.QTY
    
    # GO_LIVE breaks up data into before the GO_LIVE and after.  
    gpi_vol_awp_df['GO_LIVE'] = gpi_vol_awp_df.apply(
        lambda df: 0 if df.DOF < p.GO_LIVE else 1, axis=1)
    assert gpi_vol_awp_df['GO_LIVE'].sum() == 0, "gpi_vol_awp_df['GO_LIVE'].sum() == 0"
    
    
    if p.INCLUDE_PLAN_LIABILITY:
        min_ds_df = standardize_df(pd.read_csv(p.FILE_INPUT_PATH + p.MIN_DS_FILE, dtype = p.VARIABLE_TYPE_DIC))
        qa_dataframe(min_ds_df, dataset = 'MIN_DS_FILE_AT_{}'.format(os.path.basename(__file__)))
        min_ds_df.MIN_DAYS_SCRIPT = min_ds_df.MIN_DAYS_SCRIPT.astype(float)
        gpi_vol_awp_ds = pd.merge(gpi_vol_awp_df, min_ds_df, how='left', on=['NDC'])
        assert len(gpi_vol_awp_df) == len(gpi_vol_awp_ds), "len(gpi_vol_awp_df) == len(gpi_vol_awp_ds)"
    
        gpi_vol_awp_df = gpi_vol_awp_ds.copy()
        del gpi_vol_awp_ds
    
        gpi_vol_awp_df.MIN_DAYS_SCRIPT = gpi_vol_awp_df.MIN_DAYS_SCRIPT.fillna(30.0)
    
    return gpi_vol_awp_df


def get_begin_pull_date(end_day):

    data_start_date = dt.datetime.strptime(p.DATA_START_DAY, '%Y-%m-%d')
    if (p.LAST_DATA-data_start_date).days <= 90:  # if there are 3 months or less of data then pull from the start of the contract
        begin_pull = dt.datetime.strptime(p.DATA_START_DAY, '%Y-%m-%d')
    
    else:
        # month, year tuple for moving between dates, starts three months back
        mo_yr_list = []  # create month and year list for projections, starting from 3 months back ending at 1 month forward
        for m in range(p.LAST_DATA.month + 12 - 3, p.LAST_DATA.month + 12 + 2):
            if m <= 12:  # one month back
                mo_yr_list.append((m, p.LAST_DATA.year - 1))
            elif 12 < m < 24:  # in the same year
                mo_yr_list.append((m % 12, p.LAST_DATA.year))
            elif m == 24:  # december in same year
                mo_yr_list.append((12, p.LAST_DATA.year))
            else:  # one year ahead
                mo_yr_list.append((m % 12, p.LAST_DATA.year + 1))

        if p.LAST_DATA.day < 28: #If date is before the 28th, then go from same day three months earlier
            begin_pull = dt.datetime.strptime(
                str(mo_yr_list[0][0]) + '/' + str(p.LAST_DATA.day + 1) + '/' + str(mo_yr_list[0][1]),
                '%m/%d/%Y')  # 3 months back
        else: #first day of data will match last day of data relative to last day in month (eg 29th of May will go back to Feb 26th)

            day_diff = dt.datetime.strptime(str(p.LAST_DATA.month) + '/' + str(end_day) + '/' + str(p.LAST_DATA.year),
                                            '%m/%d/%Y') - p.LAST_DATA

            begin_pull = dt.datetime.strptime(str(mo_yr_list[1][0]) + '/1/' + str(mo_yr_list[1][1]),
                                              '%m/%d/%Y') - day_diff  # 2 months back
    
    return begin_pull


def get_days_in_month(end_day):
    
    begin_pull = get_begin_pull_date(end_day)
    days_in_months = (p.LAST_DATA - begin_pull).days + 1
    
    return days_in_months

def get_sample_awp_data(gpi_vol_awp_df, end_day):

    df = gpi_vol_awp_df.copy()
    begin_pull = get_begin_pull_date(end_day)
    data_start_date = dt.datetime.strptime(p.DATA_START_DAY, '%Y-%m-%d')
    
    proj_type = 'SES'
    if ((p.LAST_DATA-data_start_date).days <= 90):
        proj_type = 'avg'
    assert p.LAST_DATA == df.DOF.max(), "p.LAST_DATA == df.DOF.max()"
    
    cols_of_interest = [
        'GPI_NDC', 'CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'BG_FLAG',
        'DOF', 'DOF_MONTH', 'CLAIMS', 'FULLAWP_ADJ', 'QTY', 'PHARM_CLAIMS', 'PHARM_QTY', 'PHARM_FULLAWP_ADJ', 'PHARM_FULLNADAC_ADJ', 'PHARM_FULLACC_ADJ','PHARM_TARG_INGCOST_ADJ']
    if p.INCLUDE_PLAN_LIABILITY:
        cols_of_interest = cols_of_interest + ['DAYSSUP']
        
    sample_vol_awp = df.loc[(df.DOF >= begin_pull) & 
                            (df.DOF <= p.LAST_DATA), cols_of_interest]
    
    days_in_months = (p.LAST_DATA - begin_pull).days + 1
    
    #AWP and utilization to be used by projection method
    if p.LAST_DATA.day == end_day:
        sample_vol_awp['PROJ_MONTH'] = sample_vol_awp['DOF_MONTH']
        
    else:
        mo_yr_list = []  # create month and year list for projections, starting from 3 months back ending at 1 month forward
        for m in range(p.LAST_DATA.month + 12 - 3, p.LAST_DATA.month + 12 + 2):
            if m <= 12:  # one month back
                mo_yr_list.append((m, p.LAST_DATA.year - 1))
            elif 12 < m < 24:  # in the same year
                mo_yr_list.append((m % 12, p.LAST_DATA.year))
            elif m == 24:  # december in same year
                mo_yr_list.append((12, p.LAST_DATA.year))
            else:  # one year ahead
                mo_yr_list.append((m % 12, p.LAST_DATA.year + 1))

        if p.LAST_DATA.day < 28:
            break_1 = dt.datetime.strptime(
                str(mo_yr_list[2][0]) + '/' + str(p.LAST_DATA.day) + '/' + str(mo_yr_list[2][1]),
                '%m/%d/%Y')  # 1 month back
            break_2 = dt.datetime.strptime(
                str(mo_yr_list[1][0]) + '/' + str(p.LAST_DATA.day) + '/' + str(mo_yr_list[1][1]),
                '%m/%d/%Y')  # 2 months back
        else:
            day_diff = dt.datetime.strptime(str(mo_yr_list[4][0]) + '/1/' + str(mo_yr_list[4][1]),
                                            '%m/%d/%Y') - p.LAST_DATA  # 1 month forward
            break_1 = dt.datetime.strptime(str(p.LAST_DATA.month) + '/1/' + str(p.LAST_DATA.year),
                                           '%m/%d/%Y') - day_diff
            break_2 = (p.LAST_DATA.replace(day=1) - dt.timedelta(days=1)).replace(day=1) - day_diff

        start_date = dt.datetime.strptime(p.DATA_START_DAY, '%Y-%m-%d')
        start_months = [x % 12 if x > 12 else x for x in range(start_date.month, start_date.month + 4)]

        sample_vol_awp.loc[(sample_vol_awp.DOF > break_1) & 
                           (sample_vol_awp.DOF <= p.LAST_DATA), 'PROJ_MONTH'] = start_months[2]
        sample_vol_awp.loc[(sample_vol_awp.DOF > break_2) & 
                           (sample_vol_awp.DOF <= break_1), 'PROJ_MONTH'] = start_months[1]
        sample_vol_awp.loc[
            (sample_vol_awp.DOF >= begin_pull) & 
            (sample_vol_awp.DOF <= break_2), 'PROJ_MONTH'] = start_months[0]
    
    group_columns = ['GPI_NDC', 'CLIENT', 'BREAKOUT', 'BG_FLAG',
                     'REGION', 'MEASUREMENT', 'PROJ_MONTH']
    aggregate_columns = ['CLAIMS', 'QTY', 'FULLAWP_ADJ', 'PHARM_CLAIMS', 'PHARM_QTY', 'PHARM_FULLAWP_ADJ', 'PHARM_FULLNADAC_ADJ', 'PHARM_FULLACC_ADJ','PHARM_TARG_INGCOST_ADJ']
    sample_awp_month = sample_vol_awp.groupby(
        group_columns)[aggregate_columns].agg(sum)
    sample_awp_month = sample_awp_month.reset_index()
    
    #Daily average to be used to project out to rest of year
    day_group_columns = ['GPI_NDC', 'CLIENT', 'BREAKOUT', 'BG_FLAG',
                         'REGION', 'MEASUREMENT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP']
    sample_awp_day = sample_vol_awp.groupby(
        day_group_columns)[aggregate_columns].agg(sum)
    sample_awp_day = sample_awp_day.reset_index()
    sample_awp_day['CLAIMS_DAY'] = sample_awp_day['CLAIMS'] / days_in_months
    sample_awp_day['QTY_DAY'] = sample_awp_day['QTY'] / days_in_months
    sample_awp_day['FULLAWP_ADJ_DAY'] = sample_awp_day['FULLAWP_ADJ'] / days_in_months
    sample_awp_day['PHARM_CLAIMS_DAY'] = sample_awp_day['PHARM_CLAIMS'] / days_in_months
    sample_awp_day['PHARM_QTY_DAY'] = sample_awp_day['PHARM_QTY'] / days_in_months
    sample_awp_day['PHARM_FULLAWP_ADJ_DAY'] = sample_awp_day['PHARM_FULLAWP_ADJ'] / days_in_months

    return (sample_awp_month,sample_awp_day,proj_type,days_in_months)


def get_proj(sample_awp_month,sample_awp_day,days_in_months,proj_type,lag_days,eoy_days,client_list):
    from statsmodels.tsa.api import SimpleExpSmoothing
    
    #define output columns
    day_group_columns = ['GPI_NDC', 'CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'BG_FLAG']
    eoy_columns = day_group_columns + \
        ['CLAIMS_DAY', 'QTY_DAY', 'FULLAWP_ADJ_DAY',
         'CLAIMS_PROJ_DAY','QTY_PROJ_DAY', 'FULLAWP_ADJ_PROJ_DAY',
         'CLAIMS_PROJ_LAG', 'QTY_PROJ_LAG', 'FULLAWP_ADJ_PROJ_LAG',
         'CLAIMS_PROJ_EOY', 'QTY_PROJ_EOY', 'FULLAWP_ADJ_PROJ_EOY',
         'PHARM_CLAIMS_DAY', 'PHARM_QTY_DAY', 'PHARM_FULLAWP_ADJ_DAY',
         'PHARM_CLAIMS_PROJ_DAY','PHARM_QTY_PROJ_DAY', 'PHARM_FULLAWP_ADJ_PROJ_DAY',
         'PHARM_CLAIMS_PROJ_LAG', 'PHARM_QTY_PROJ_LAG', 'PHARM_FULLAWP_ADJ_PROJ_LAG',
         'PHARM_CLAIMS_PROJ_EOY', 'PHARM_QTY_PROJ_EOY', 'PHARM_FULLAWP_ADJ_PROJ_EOY'
        ]
    if p.INCLUDE_PLAN_LIABILITY:
        eoy_columns = eoy_columns + \
            ['DAYSSUP_DAY', 'DAYSSUP_PROJ_DAY', 'DAYSSUP_PROJ_LAG', 'DAYSSUP_PROJ_EOY']

    eoy_proj_df = pd.DataFrame(columns=eoy_columns)

    for client in client_list:
        breakout_list = sample_awp_month.loc[sample_awp_month.CLIENT == client, 'BREAKOUT'].unique()
        for breakout in breakout_list:
            reg_list = sample_awp_month.loc[(sample_awp_month.CLIENT == client) &
                                            (sample_awp_month.BREAKOUT == breakout), 'REGION'].unique()
            for reg in reg_list:
                bg_list = sample_awp_month.loc[(sample_awp_month.CLIENT == client) &
                                              (sample_awp_month.BREAKOUT == breakout) &
                                              (sample_awp_month.REGION == reg), 'BG_FLAG'].unique()
                for bg in bg_list:
                    mes_list = sample_awp_month.loc[(sample_awp_month.CLIENT == client) &
                                                    (sample_awp_month.BREAKOUT == breakout) &
                                                    (sample_awp_month.REGION == reg) &
                                                    (sample_awp_month.BG_FLAG == bg), 'MEASUREMENT'].unique()
                    for mes in mes_list:
                        sample_month_df = sample_awp_month.loc[(sample_awp_month.CLIENT == client) &
                                                               (sample_awp_month.BREAKOUT == breakout) &
                                                               (sample_awp_month.REGION == reg) &
                                                               (sample_awp_month.BG_FLAG == bg) &
                                                               (sample_awp_month.MEASUREMENT == mes)].copy()  
                        sample_day_df = sample_awp_day.loc[(sample_awp_day.CLIENT == client) &
                                                           (sample_awp_day.BREAKOUT == breakout) &
                                                           (sample_awp_day.REGION == reg) &
                                                           (sample_awp_day.BG_FLAG == bg) &
                                                           (sample_awp_day.MEASUREMENT == mes)].copy() 
                        awp_month = sample_month_df.groupby('PROJ_MONTH')['FULLAWP_ADJ'].agg(sum)
                        awp_month_pharm = sample_month_df.groupby('PROJ_MONTH')['PHARM_FULLAWP_ADJ'].agg(sum)
                    
                        # skipping ses when data doesn't have 3 months complete data.
                        # Anamolous awp values get flagged in E2R checks and reported to DE, halting the run at the very start

                        skip_ses = (len(awp_month[awp_month > 0]) < 3)
                        skip_ses_pharm = (len(awp_month_pharm[awp_month_pharm > 0]) < 3)
                    
                        if proj_type == 'SES' and not skip_ses: #Simple Exponential Smoothing
                            fit_client = SimpleExpSmoothing(awp_month.values).fit(smoothing_level = p.PROJ_ALPHA, optimized=False)
                            fcast_client = fit_client.forecast(1) / 30
                            awp_day = awp_month.sum() / days_in_months
                            awp_factor = (fcast_client[0] / awp_day)
                        else:
                            awp_factor = 1

                        if proj_type == 'SES' and not skip_ses_pharm:
                            fit_pharm = SimpleExpSmoothing(awp_month_pharm.values).fit(smoothing_level = p.PROJ_ALPHA, optimized=False)
                            fcast_pharm = fit_pharm.forecast(1) / 30
                            awp_day_pharm = awp_month_pharm.sum() / days_in_months
                            awp_factor_pharm = (fcast_pharm[0] / awp_day_pharm)  
                        else: #Average of past three months                  
                            awp_factor_pharm = 1

                        # Calculate projections on a day-by-day basis
                        sample_day_df['CLAIMS_PROJ_DAY'] = sample_day_df['CLAIMS_DAY'] * awp_factor
                        sample_day_df['QTY_PROJ_DAY'] = sample_day_df['QTY_DAY'] * awp_factor
                        sample_day_df['FULLAWP_ADJ_PROJ_DAY'] = sample_day_df['FULLAWP_ADJ_DAY'] * awp_factor
    
                        # Multiply projections by the duration of the lag period duration
                        sample_day_df['CLAIMS_PROJ_LAG'] = sample_day_df['CLAIMS_PROJ_DAY'] * lag_days
                        sample_day_df['QTY_PROJ_LAG'] = sample_day_df['QTY_PROJ_DAY'] * lag_days
                        sample_day_df['FULLAWP_ADJ_PROJ_LAG'] = sample_day_df['FULLAWP_ADJ_PROJ_DAY'] * lag_days
    
                        # Multiply projections by the duration of the go-live date to end of year
                        sample_day_df['CLAIMS_PROJ_EOY'] = sample_day_df['CLAIMS_PROJ_DAY'] * eoy_days
                        sample_day_df['QTY_PROJ_EOY'] = sample_day_df['QTY_PROJ_DAY'] * eoy_days
                        sample_day_df['FULLAWP_ADJ_PROJ_EOY'] = sample_day_df['FULLAWP_ADJ_PROJ_DAY'] * eoy_days

                        # Calculate projections on a day-by-day basis
                        sample_day_df['PHARM_CLAIMS_PROJ_DAY'] = sample_day_df['PHARM_CLAIMS_DAY'] * awp_factor_pharm
                        sample_day_df['PHARM_QTY_PROJ_DAY'] = sample_day_df['PHARM_QTY_DAY'] * awp_factor_pharm
                        sample_day_df['PHARM_FULLAWP_ADJ_PROJ_DAY'] = sample_day_df['PHARM_FULLAWP_ADJ_DAY'] * awp_factor_pharm
                    
                        # Multiply projections by the duration of the lag period duration
                        sample_day_df['PHARM_CLAIMS_PROJ_LAG'] = sample_day_df['PHARM_CLAIMS_PROJ_DAY'] * lag_days
                        sample_day_df['PHARM_QTY_PROJ_LAG'] = sample_day_df['PHARM_QTY_PROJ_DAY'] * lag_days
                        sample_day_df['PHARM_FULLAWP_ADJ_PROJ_LAG'] = sample_day_df['PHARM_FULLAWP_ADJ_PROJ_DAY'] * lag_days
                    
                        # Multiply projections by the duration of the go-live date to end of year
                        sample_day_df['PHARM_CLAIMS_PROJ_EOY'] = sample_day_df['PHARM_CLAIMS_PROJ_DAY'] * eoy_days
                        sample_day_df['PHARM_QTY_PROJ_EOY'] = sample_day_df['PHARM_QTY_PROJ_DAY'] * eoy_days
                        sample_day_df['PHARM_FULLAWP_ADJ_PROJ_EOY'] = sample_day_df['PHARM_FULLAWP_ADJ_PROJ_DAY'] * eoy_days
                    
                        if p.INCLUDE_PLAN_LIABILITY:
                            sample_day_df['DAYSSUP_PROJ_DAY'] = sample_day_df['DAYSSUP_DAY'] * awp_factor
                            sample_day_df['DAYSSUP_PROJ_LAG'] = sample_day_df['DAYSSUP_PROJ_DAY'] * lag_days
                            sample_day_df['DAYSSUP_PROJ_EOY'] = sample_day_df['DAYSSUP_PROJ_DAY'] * eoy_days
    
                        eoy_proj_df = pd.concat([eoy_proj_df, sample_day_df[eoy_columns]])
    
    return eoy_proj_df

def get_gpi_vol_awp_agg(gpi_vol_awp_df,agg_columns_of_interest,sum_columns, contract_date_df):
    '''
    It takes the claims data and aggregates it using 'agg_columns_of_interest' the aggregates
    are sum.
    The UNC prices are also pre-calculated to maintain an understanding of the UNC cost per unit on the aggregated df.

    Inputs:
        gpi_vol_awp_df: The claims
        agg_columns_of_interest: The columns to aggregate by.
        sum_columns: The columns that will be aggregated and sum.
    Outputs:
        gpi_vol_awp_agg: The aggregated claims
    '''    
    gpi_vol_awp_df['PRICING_QTY'] = gpi_vol_awp_df[['QTY', 'PHARM_QTY']].max(axis=1, skipna=True)
    gpi_vol_awp_df['uc_unit_agg'] = gpi_vol_awp_df['UC_UNIT'] * gpi_vol_awp_df['PRICING_QTY']
    gpi_vol_awp_df['uc_unit25_agg'] = gpi_vol_awp_df['UC_UNIT25'] * gpi_vol_awp_df['PRICING_QTY']
    if p.CROSS_CONTRACT_PROJ:
        # Get columns that aggregate data from the prior contract period only
        prior_contract_exprn = pd.Timestamp.to_pydatetime(contract_date_df['CONTRACT_EFF_DT'][0] - dt.timedelta(days=1)).strftime('%Y-%m-%d')
        gpi_vol_awp_df['CLAIMS_PRIOR'] = np.where(gpi_vol_awp_df['DOF'] <= prior_contract_exprn, gpi_vol_awp_df['CLAIMS'], 0) 
        gpi_vol_awp_df['QTY_PRIOR'] = np.where(gpi_vol_awp_df['DOF'] <= prior_contract_exprn, gpi_vol_awp_df['QTY'], 0) 
        gpi_vol_awp_df['FULLAWP_ADJ_PRIOR'] = np.where(gpi_vol_awp_df['DOF'] <= prior_contract_exprn, gpi_vol_awp_df['FULLAWP_ADJ'], 0)     
        gpi_vol_awp_df['PRICE_REIMB_PRIOR'] = np.where(gpi_vol_awp_df['DOF'] <= prior_contract_exprn, gpi_vol_awp_df['PRICE_REIMB'], 0)      
        gpi_vol_awp_df['PHARM_CLAIMS_PRIOR'] = np.where(gpi_vol_awp_df['DOF'] <= prior_contract_exprn, gpi_vol_awp_df['PHARM_CLAIMS'], 0)               
        gpi_vol_awp_df['PHARM_QTY_PRIOR'] = np.where(gpi_vol_awp_df['DOF'] <= prior_contract_exprn, gpi_vol_awp_df['PHARM_QTY'], 0)           
        gpi_vol_awp_df['PHARM_FULLAWP_ADJ_PRIOR'] = np.where(gpi_vol_awp_df['DOF'] <= prior_contract_exprn, gpi_vol_awp_df['PHARM_FULLAWP_ADJ'], 0)
        
        client_guarantees = read_client_guarantees(READ_FROM_BQ = p.READ_FROM_BQ)
        gpi_vol_awp_df = add_target_ingcost(gpi_vol_awp_df, client_guarantees, client_rate_col = 'RATE', target_cols=['TARG_INGCOST_ADJ'])
        gpi_vol_awp_df['PHARM_TARG_INGCOST_ADJ_PRIOR'] = np.where(gpi_vol_awp_df['DOF'] <= prior_contract_exprn, gpi_vol_awp_df['PHARM_TARG_INGCOST_ADJ'], 0)
        gpi_vol_awp_df['TARG_INGCOST_ADJ_PRIOR'] = np.where(gpi_vol_awp_df['DOF'] <= prior_contract_exprn, gpi_vol_awp_df['TARG_INGCOST_ADJ'], 0)
        
        gpi_vol_awp_df['PHARM_PRICE_REIMB_PRIOR'] = np.where(gpi_vol_awp_df['DOF'] <= prior_contract_exprn, gpi_vol_awp_df['PHARM_PRICE_REIMB'], 0)
    if 'UC_CLAIMS' in sum_columns:
        gpi_vol_awp_df['UC_CLAIMS'] = pd.to_numeric(gpi_vol_awp_df['UC_CLAIMS'], errors='coerce')
        gpi_vol_awp_df['UC_CLAIMS'] = gpi_vol_awp_df['UC_CLAIMS'].fillna(0) 
    if 'LM_UC_CLAIMS' in sum_columns:
        gpi_vol_awp_df['LM_UC_CLAIMS'] = pd.to_numeric(gpi_vol_awp_df['LM_UC_CLAIMS'], errors='coerce')
        gpi_vol_awp_df['LM_UC_CLAIMS'] = gpi_vol_awp_df['LM_UC_CLAIMS'].fillna(0) 
    gpi_vol_awp_agg_month = gpi_vol_awp_df.groupby(agg_columns_of_interest)
    gpi_vol_awp_agg = gpi_vol_awp_agg_month[sum_columns].agg(sum)
    gpi_vol_awp_agg = gpi_vol_awp_agg.reset_index()
    gpi_vol_awp_agg['PRICING_QTY'] = gpi_vol_awp_agg[['QTY', 'PHARM_QTY']].max(axis=1, skipna=True)
    gpi_vol_awp_agg['UC_UNIT'] = gpi_vol_awp_agg['uc_unit_agg']/gpi_vol_awp_agg['PRICING_QTY']
    gpi_vol_awp_agg['UC_UNIT25'] = gpi_vol_awp_agg['uc_unit25_agg']/gpi_vol_awp_agg['PRICING_QTY']

    gpi_vol_awp_df.drop(columns=['PRICING_QTY'], inplace=True)
    gpi_vol_awp_agg.drop(columns=['PRICING_QTY'], inplace=True)

    return gpi_vol_awp_agg   
    
def add_current_awp_to_data(gpi_vol_awp_df, contract_date_df, gpi_vol_awp_df_actual=pd.DataFrame()):
    '''
    The claims are aggregated, in total and just the last month of claims (LP_*). The aggregated claims
    now also include the current AWP.

    Inputs:
        gpi_vol_awp_df: Has all the claims independently of adjudication method (U&N or GRx)
        gpi_vol_awp_df_actual: Has all the claims adjudicated at MAC. Should be changed to "_actual" for consistency
    Outputs:
    gpi_vol_awp_agg: The aggregated claims.
    '''

    sum_columns = ['CLAIMS', 'QTY', 'FULLAWP_ADJ','PRICE_REIMB', 'PHARM_CLAIMS', 'PHARM_QTY',
                   'PHARM_FULLAWP_ADJ','PHARM_FULLNADAC_ADJ','PHARM_FULLACC_ADJ','PHARM_TARG_INGCOST_ADJ','PHARM_PRICE_REIMB']

    sum_columns_of_interest = ['CLAIMS','QTY','FULLAWP_ADJ','PRICE_REIMB', 'uc_unit_agg', 'uc_unit25_agg',
                               'PHARM_CLAIMS', 'PHARM_QTY', 'PHARM_FULLAWP_ADJ','PHARM_FULLNADAC_ADJ','PHARM_FULLACC_ADJ','PHARM_TARG_INGCOST_ADJ','PHARM_PRICE_REIMB',
                                'CLAIMS_ZBD', 'FULLAWP_ADJ_ZBD', 'QTY_ZBD', 'PRICE_REIMB_ZBD',
                                'PHARM_CLAIMS_ZBD', 'PHARM_QTY_ZBD', 'PHARM_FULLAWP_ADJ_ZBD','PHARM_FULLNADAC_ADJ_ZBD','PHARM_FULLACC_ADJ_ZBD','PHARM_TARG_INGCOST_ADJ_ZBD', 'PHARM_PRICE_REIMB_ZBD',
                                'DISP_FEE', 'TARGET_DISP_FEE', 'PHARMACY_DISP_FEE', 'PHARM_TARGET_DISP_FEE', 'NADAC_WAC_YTD']

    agg_columns_of_interest =['CLIENT','BREAKOUT', 'REGION', 'MEASUREMENT', 'BG_FLAG',
                              'GPI', 'NDC', 'GPI_NDC', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GO_LIVE']
    
    if p.CROSS_CONTRACT_PROJ:
        sum_columns_of_interest = sum_columns_of_interest + \
            ['CLAIMS_PRIOR', 'QTY_PRIOR', 'FULLAWP_ADJ_PRIOR', 'TARG_INGCOST_ADJ_PRIOR','PRICE_REIMB_PRIOR', 
             'PHARM_CLAIMS_PRIOR','PHARM_QTY_PRIOR','PHARM_FULLAWP_ADJ_PRIOR','PHARM_TARG_INGCOST_ADJ_PRIOR'
             ,'PHARM_PRICE_REIMB_PRIOR']
    
    if p.UNC_OPT:
        if not gpi_vol_awp_df_actual.empty: #When there are "actual" modifiable claims
            gpi_vol_awp_agg = get_gpi_vol_awp_agg(gpi_vol_awp_df_actual, agg_columns_of_interest,
                                                  sum_columns_of_interest, contract_date_df)

        # Then make sure to count how many claims were U&C and total (which is why we do this first)
        #gpi_vol_awp_df['UC_CLAIMS'] = (gpi_vol_awp_df['ADJUDICATED_AT']=='U&C')*gpi_vol_awp_df['CLAIMS']
        sum_columns.append('UC_CLAIMS')
        sum_columns_of_interest.append('UC_CLAIMS')
    
    if p.INCLUDE_PLAN_LIABILITY:
        gpi_vol_awp_df['NUM30DAYS'] = gpi_vol_awp_df['DAYSSUP']/30
        gpi_vol_awp_df['COPAY_AGG'] = gpi_vol_awp_df['COPAY'] * gpi_vol_awp_df['NUM30DAYS']
        # gpi_vol_awp_df['COPAY_RAW_AGG'] = gpi_vol_awp_df['COPAY_RAW'] * gpi_vol_awp_df['NUM30DAYS']
        gpi_vol_awp_df['DCS_COPAY_AGG'] = gpi_vol_awp_df['COPAY'] * gpi_vol_awp_df['DAYSSUP']
        gpi_vol_awp_df['COPAY_RATE_AGG'] = gpi_vol_awp_df['COPAY_RATE'] * gpi_vol_awp_df['DAYSSUP']
        gpi_vol_awp_df['PROP_LIS_AGG'] = gpi_vol_awp_df['PROP_LIS'] * gpi_vol_awp_df['CLAIMS']
        gpi_vol_awp_df['PROP_DEDUCT_AGG'] = gpi_vol_awp_df['PROP_DEDUCT'] * gpi_vol_awp_df['FULLAWP_ADJ']
        gpi_vol_awp_df['PROP_ICL_AGG'] = gpi_vol_awp_df['PROP_ICL'] * gpi_vol_awp_df['FULLAWP_ADJ']
        gpi_vol_awp_df['PROP_GAP_AGG'] = gpi_vol_awp_df['PROP_GAP'] * gpi_vol_awp_df['FULLAWP_ADJ']
        gpi_vol_awp_df['PROP_CAT_AGG'] = gpi_vol_awp_df['PROP_CAT'] * gpi_vol_awp_df['FULLAWP_ADJ']
        gpi_vol_awp_df['TIER_AGG'] = gpi_vol_awp_df['TIER'] * gpi_vol_awp_df['FULLAWP_ADJ']
        gpi_vol_awp_df['MIN_DAYS_AGG'] = gpi_vol_awp_df['MIN_DAYS_SCRIPT'] * gpi_vol_awp_df['FULLAWP_ADJ']
        agg_columns_of_interest = agg_columns_of_interest + \
            ['DAYSSUP', 'PSTCOPAY', 'COPAY_AGG', 'COPAY_RATE_AGG', 'PROP_LIS_AGG',
             'PROP_DEDUCT_AGG', 'PROP_ICL_AGG', 'PROP_GAP_AGG', 'PROP_CAT_AGG', 
             'TIER_AGG', 'MIN_DAYS_AGG', 'DCS_COPAY_AGG'] # 'COPAY_RAW_AGG'
    
    if gpi_vol_awp_df_actual.empty:
        gpi_vol_awp_agg = get_gpi_vol_awp_agg(gpi_vol_awp_df,agg_columns_of_interest,sum_columns_of_interest,contract_date_df)
        assert abs(gpi_vol_awp_agg.FULLAWP_ADJ.sum() - gpi_vol_awp_df.FULLAWP_ADJ.sum()) < .0001, \
            "abs(gpi_vol_awp_agg.FULLAWP_ADJ.sum() - gpi_vol_awp_df.FULLAWP_ADJ.sum()) < .0001"
    
    if p.INCLUDE_PLAN_LIABILITY:
        gpi_vol_awp_agg['COPAY'] = gpi_vol_awp_agg['COPAY_AGG'] / (gpi_vol_awp_agg['DAYSSUP']/30)
        # gpi_vol_awp_agg['COPAY_RAW'] = gpi_vol_awp_agg['COPAY_RAW_AGG'] / (gpi_vol_awp_agg['DAYSSUP']/30)
        gpi_vol_awp_agg['COPAY_DCS'] = gpi_vol_awp_agg['DCS_COPAY_AGG'] / (gpi_vol_awp_agg['DAYSSUP'])
        gpi_vol_awp_agg['COPAY_RATE'] = gpi_vol_awp_agg['COPAY_RATE_AGG' ] / gpi_vol_awp_agg['DAYSSUP']
        gpi_vol_awp_agg['PROP_LIS'] = gpi_vol_awp_agg['PROP_LIS_AGG'] / gpi_vol_awp_agg['CLAIMS']
        gpi_vol_awp_agg['PROP_DEDUCT'] = gpi_vol_awp_agg['PROP_DEDUCT_AGG'] / gpi_vol_awp_agg['FULLAWP_ADJ']
        gpi_vol_awp_agg['PROP_ICL'] = gpi_vol_awp_agg['PROP_ICL_AGG'] / gpi_vol_awp_agg['FULLAWP_ADJ']
        gpi_vol_awp_agg['PROP_GAP'] = gpi_vol_awp_agg['PROP_GAP_AGG'] / gpi_vol_awp_agg['FULLAWP_ADJ']
        gpi_vol_awp_agg['PROP_CAT'] = gpi_vol_awp_agg['PROP_CAT_AGG'] / gpi_vol_awp_agg['FULLAWP_ADJ']
        gpi_vol_awp_agg['TIER'] = gpi_vol_awp_agg['TIER_AGG'] / gpi_vol_awp_agg['FULLAWP_ADJ']
        gpi_vol_awp_agg['MIN_DAYS_SCRIPT'] = gpi_vol_awp_agg['MIN_DAYS_AGG'] / gpi_vol_awp_agg['FULLAWP_ADJ']
    
        gpi_vol_awp_agg.loc[gpi_vol_awp_agg.DCS=='N', 'COPAY'] = gpi_vol_awp_agg.loc[gpi_vol_awp_agg.DCS=='N', 'COPAY_DCS']
    
    #Get last month of claims, quantity, awp, and reimbursement for recast
    if gpi_vol_awp_df_actual.empty:
        gpi_vol_awp_df_agg = gpi_vol_awp_df.copy(deep=True)
    else:
        gpi_vol_awp_df_agg = gpi_vol_awp_df_actual.copy(deep=True)
        sum_columns = ['CLAIMS','QTY','FULLAWP_ADJ','PRICE_REIMB', 'PHARM_CLAIMS', 'PHARM_QTY', 'PHARM_FULLAWP_ADJ', 'PHARM_FULLNADAC_ADJ',
                       'PHARM_FULLACC_ADJ','PHARM_TARG_INGCOST_ADJ','PHARM_PRICE_REIMB']

    last_month_vol_awp_agg = gpi_vol_awp_df_agg.loc[gpi_vol_awp_df_agg.DOF_MONTH == p.LAST_DATA.month].groupby(agg_columns_of_interest)
    last_month_vol_awp = last_month_vol_awp_agg[sum_columns].agg(sum)
    last_month_vol_awp = last_month_vol_awp.reset_index()
    
    last_month_vol_awp.rename(columns={'CLAIMS': 'LM_CLAIMS',
                                       'QTY': 'LM_QTY',
                                       'FULLAWP_ADJ': 'LM_FULLAWP_ADJ',
                                       'PRICE_REIMB': 'LM_PRICE_REIMB',
                                       'PHARM_CLAIMS': 'LM_PHARM_CLAIMS',
                                       'PHARM_QTY': 'LM_PHARM_QTY',
                                       'PHARM_FULLAWP_ADJ': 'LM_PHARM_FULLAWP_ADJ',
                                       'PHARM_FULLNADAC_ADJ': 'LM_PHARM_FULLNADAC_ADJ',
                                       'PHARM_FULLACC_ADJ': 'LM_PHARM_FULLACC_ADJ',
                                       'PHARM_TARG_INGCOST_ADJ': 'LM_PHARM_TARG_INGCOST_ADJ',
                                       'PHARM_PRICE_REIMB': 'LM_PHARM_PRICE_REIMB'}, inplace=True)

    if p.UNC_OPT: #Intercptor does not need this but could keep it
        last_month_vol_awp.rename(columns={'UC_CLAIMS': 'LM_UC_CLAIMS'}, inplace=True)
    
    old_len = len(gpi_vol_awp_agg)
    gpi_vol_awp_agg = pd.merge(gpi_vol_awp_agg, last_month_vol_awp, how='left', on=agg_columns_of_interest)
    assert len(gpi_vol_awp_agg) == old_len, "len(gpi_vol_awp_agg) == old_len"

    lp_data_zbd_lm_df = cs.get_last_month_zbd_claims(gpi_vol_awp_df_agg, agg_columns_of_interest)
    gpi_vol_awp_agg = pd.merge(gpi_vol_awp_agg, lp_data_zbd_lm_df, how='left', on=agg_columns_of_interest)
    assert len(gpi_vol_awp_agg) == old_len, "len(gpi_vol_awp_agg) == old_len"      
    
    del gpi_vol_awp_df
    
    #Add in current AWPs
    curr_awp_df = read_current_awp(READ_FROM_BQ = p.READ_FROM_BQ)
    
    gpi_vol_awp_agg_temp = pd.merge(
        gpi_vol_awp_agg, curr_awp_df[['GPI_NDC', 'CURR_AWP']], how='left', on='GPI_NDC')
    assert len(gpi_vol_awp_agg_temp) == len(gpi_vol_awp_agg), "len(gpi_vol_awp_agg_temp) == len(gpi_vol_awp_agg)"
    gpi_vol_awp_agg = gpi_vol_awp_agg_temp
    del gpi_vol_awp_agg_temp
    gpi_vol_awp_agg['CURR_AWP_MIN'] = gpi_vol_awp_agg['CURR_AWP'].copy(deep=True)
    gpi_vol_awp_agg['CURR_AWP_MAX'] = gpi_vol_awp_agg['CURR_AWP'].copy(deep=True)
    
    return gpi_vol_awp_agg


def add_projections(gpi_vol_awp_agg, eoy_proj_df):
    
    comb_columns = ['GPI_NDC', 'CLIENT', 'BREAKOUT', 'REGION', 'BG_FLAG',
                    'MEASUREMENT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP']
    proj_columns = ['CLAIMS_PROJ_LAG', 'QTY_PROJ_LAG', 'FULLAWP_ADJ_PROJ_LAG',
                    'CLAIMS_PROJ_EOY', 'QTY_PROJ_EOY', 'FULLAWP_ADJ_PROJ_EOY',
                    'PHARM_CLAIMS_PROJ_LAG', 'PHARM_QTY_PROJ_LAG', 'PHARM_FULLAWP_ADJ_PROJ_LAG', 
                    'PHARM_CLAIMS_PROJ_EOY', 'PHARM_QTY_PROJ_EOY', 'PHARM_FULLAWP_ADJ_PROJ_EOY' 
                    ]

    if p.INCLUDE_PLAN_LIABILITY:
        proj_columns = proj_columns + ['DAYSSUP_PROJ_LAG', 'DAYSSUP_PROJ_EOY']

    gpi_vol_awp_agg_ytd = pd.merge(
        gpi_vol_awp_agg, eoy_proj_df[comb_columns + proj_columns], 
        on=comb_columns, how='left')

    assert len(gpi_vol_awp_agg_ytd) == len(gpi_vol_awp_agg),\
        "Length of dataframe changed after merge"
    
    return gpi_vol_awp_agg_ytd

def update_guarantees(gpi_vol_awp_agg_ytd,chain_region_mac_mapping):
    
    chain_region_mac_mapping = chain_region_mac_mapping.merge(gpi_vol_awp_agg_ytd[['CLIENT','REGION','BREAKOUT','BG_FLAG','MEASUREMENT']].drop_duplicates(),
                                                              how='left',
                                                              ## notice we don't include BG_FLAG in merge cal here becuase mac_mapping does not have BG_FLAG
                                                              left_on=['CUSTOMER_ID','REGION','MEASUREMENT'], 
                                                              right_on=['CLIENT','REGION','MEASUREMENT'])
    # Exception case: R90 VCMLs but no R90 claims at all (typically because VCMLs are new or going live soon)
    # need to copy BREAKOUT from R30 to R90 in this case
    r90_mask = chain_region_mac_mapping['MEASUREMENT']=='R90'
    if r90_mask.any():
        if chain_region_mac_mapping.loc[r90_mask, 'BREAKOUT'].isna().all() and not (gpi_vol_awp_agg_ytd['MEASUREMENT']=='R90').any():
            for client in chain_region_mac_mapping.CUSTOMER_ID.unique():
                for region in chain_region_mac_mapping.REGION.unique():
                    for bg in chain_region_mac_mapping.BG_FLAG.unique():
                        assert gpi_vol_awp_agg_ytd.loc[(gpi_vol_awp_agg_ytd['MEASUREMENT']=='R30') 
                                                       & (gpi_vol_awp_agg_ytd['CLIENT']==client)
                                                       & (gpi_vol_awp_agg_ytd['REGION']==region)
                                                       & (gpi_vol_awp_agg_ytd['BG_FLAG'] == bg), 'BREAKOUT'].nunique() == 1, \
                        ">1 BREAKOUT for unique CLIENT-REGION-MEASUREMENT-BG_FLAG combination"
                        chain_region_mac_mapping.loc[r90_mask 
                                                     & (chain_region_mac_mapping['CUSTOMER_ID']==client)
                                                     & (chain_region_mac_mapping['REGION']==region)
                                                     & (chain_region_mac_mapping['BG_FLAG']==bg), 'BREAKOUT'] = \
                        gpi_vol_awp_agg_ytd.loc[(gpi_vol_awp_agg_ytd['MEASUREMENT']=='R30') 
                                                & (gpi_vol_awp_agg_ytd['CLIENT']==client)
                                                & (gpi_vol_awp_agg_ytd['REGION']==region)
                                                & (gpi_vol_awp_agg_ytd['BG_FLAG']==bg)
                                                & (gpi_vol_awp_agg_ytd['BREAKOUT'].notna()), 'BREAKOUT'].iloc[0]
            
    chain_region_mac_mapping.loc[:, 'CHAIN_GROUP'] = chain_region_mac_mapping.loc[:, 'CHAIN_SUBGROUP']
    # Meant to accomodate the current only chain_subgroup.  If more excist this will need to get updated!
    chain_region_mac_mapping.loc[:, 'CHAIN_GROUP'] = chain_region_mac_mapping.loc[:, 'CHAIN_GROUP'].replace(to_replace='CVSSP', value='CVS', regex=True)
    chain_region_mac_mapping.loc[:, 'CHAIN_GROUP'] = chain_region_mac_mapping.loc[:, 'CHAIN_GROUP'].replace(to_replace='MCHOICE_KRG', value='KRG', regex=True)
    chain_region_mac_mapping.loc[:, 'CHAIN_GROUP'] = chain_region_mac_mapping.loc[:, 'CHAIN_GROUP'].replace(to_replace='MCHOICE_CVS', value='MCHOICE', regex=True)
    chain_region_mac_mapping.loc[:, 'CHAIN_GROUP'] = chain_region_mac_mapping.loc[:, 'CHAIN_GROUP'].replace(to_replace='_R90OK', value='', regex=True)
    chain_region_mac_mapping.loc[:, 'CHAIN_GROUP'] = chain_region_mac_mapping.loc[:, 'CHAIN_GROUP'].replace(to_replace='_EXTRL', value='', regex=True)

    # Guarantees exist only at the chain_group level, but we want to make sure it's duplicated across subgroups
    cols = ['CUSTOMER_ID', 'BREAKOUT', 'MEASUREMENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'BG_FLAG']
    all_guarantees = chain_region_mac_mapping[cols].drop_duplicates()
    mail_guarantees_df = all_guarantees.loc[all_guarantees.MEASUREMENT=='M30']
    
    cols = ['CUSTOMER_ID', 'BREAKOUT', 'MEASUREMENT', 'REGION', 'BG_FLAG']
    guarantees_df = chain_region_mac_mapping[cols].drop_duplicates()
    guarantees_df = guarantees_df.loc[guarantees_df.MEASUREMENT != 'M30']
    
    ### added lines below to consider clients with multiple customer IDs
    # HACK: updated to use for loop to create guarantees_df
    guar_df = pd.DataFrame()
    for client in guarantees_df.CUSTOMER_ID.unique():
        for reg in guarantees_df.loc[guarantees_df.CUSTOMER_ID == client, 'REGION'].unique():
            for mes in guarantees_df.loc[
                (guarantees_df.CUSTOMER_ID == client) & (guarantees_df.REGION == reg), 'MEASUREMENT'].unique():
                for bg in guarantees_df.loc[
                (guarantees_df.CUSTOMER_ID == client) & (guarantees_df.REGION == reg) & (guarantees_df.MEASUREMENT == mes), 'BG_FLAG'].unique():
                    for chain_group in all_guarantees.loc[(all_guarantees.CUSTOMER_ID == client) & 
                                                          (all_guarantees.REGION == reg) & 
                                                          (all_guarantees.MEASUREMENT == mes) &
                                                          (all_guarantees.BG_FLAG == bg), 'CHAIN_GROUP'].unique():
                        # we do Mail separately
                        if chain_group == 'MAIL':
                            continue
                        guar_df_temp = guarantees_df.loc[
                            (guarantees_df.CUSTOMER_ID == client) & 
                            (guarantees_df.REGION == reg) & 
                            (guarantees_df.MEASUREMENT == mes) &
                            (guarantees_df.BG_FLAG == bg)]
                        all_guar_df_temp = all_guarantees.loc[
                            (all_guarantees.CUSTOMER_ID == client) & 
                            (all_guarantees.REGION == reg) & 
                            (all_guarantees.MEASUREMENT == mes) &
                            (all_guarantees.BG_FLAG == bg) &
                            (all_guarantees.CHAIN_GROUP == chain_group)]
                        reg_arr = guar_df_temp.REGION.unique()
                        meas_arr = guar_df_temp.MEASUREMENT.unique()
                        bg_arr = guar_df_temp.BG_FLAG.unique()
                        chain_arr = all_guar_df_temp.CHAIN_GROUP.unique()
                        subchain_arr = all_guar_df_temp.CHAIN_SUBGROUP.unique()
                        if mes!='R90':
                            # removes any element with 'R90OK' in the subgroup name for non-R90 measurements
                            # necessary for R90OK to avoid R30-R90 infeasibilities due to the complexity of 
                            # mail<R30<R90 constraints
                            subchain_arr = subchain_arr[np.char.find(subchain_arr.astype(str), 'R90OK')<0]
                    
                        reg_mes_chain = pd.DataFrame(
                            np.array(np.meshgrid(reg_arr, meas_arr, bg_arr, chain_arr, subchain_arr)).reshape(5, len(reg_arr) * len(
                            meas_arr) * len(bg_arr) * len(chain_arr) * len(subchain_arr)).T, columns=['REGION', 'MEASUREMENT', 'BG_FLAG','CHAIN_GROUP', 'CHAIN_SUBGROUP'])
                        guar_df_temp = pd.merge(guar_df_temp, reg_mes_chain, how='left', on=['REGION', 'MEASUREMENT','BG_FLAG'])
                        guar_df = guar_df.append(guar_df_temp, ignore_index=False)
    
    guarantees_df = guar_df.copy()
    guarantees_df = guarantees_df.loc[
        (guarantees_df.CUSTOMER_ID == 'SSI') | (guarantees_df.CHAIN_GROUP != 'RURAL_NONPREF_OTH')]
    full_guarantee_df = pd.concat([guarantees_df, mail_guarantees_df], ignore_index=True)
    full_guarantee_df.rename(columns={'CUSTOMER_ID':'CLIENT'}, inplace=True)
    
    return full_guarantee_df

def create_lp_guarant(full_guarantee_df,mac_list_df,gpi_vol_awp_agg_ytd,chain_region_mac_mapping):
    
    """Create entries in the meshgrid for only the GPI-NDCs necessary 
        (the union of GPI-NDCs in the daily totals and the GPI-NDCs in 
        the MAC list for that particular Region and Chain Group)
    """
    lp_guarant = pd.DataFrame(columns=np.concatenate((full_guarantee_df.columns, ['GPI_NDC'])))
    for client in full_guarantee_df.CLIENT.unique():
        for breakout in full_guarantee_df.loc[full_guarantee_df.CLIENT==client, 'BREAKOUT'].unique():
            for reg in full_guarantee_df.loc[(full_guarantee_df.CLIENT == client) &
                                          (full_guarantee_df.BREAKOUT == breakout), 'REGION'].unique():
                for mes in full_guarantee_df.loc[(full_guarantee_df.CLIENT == client) &
                                                 (full_guarantee_df.BREAKOUT == breakout) &
                                                 (full_guarantee_df.REGION == reg), 'MEASUREMENT'].unique():
                    for bg in full_guarantee_df.loc[(full_guarantee_df.CLIENT == client) &
                                                    (full_guarantee_df.BREAKOUT == breakout) &
                                                    (full_guarantee_df.REGION == reg) &
                                                    (full_guarantee_df.MEASUREMENT == mes),'BG_FLAG'].unique():
                        for pharm_group in full_guarantee_df.loc[(full_guarantee_df.CLIENT == client) &
                                                                 (full_guarantee_df.BREAKOUT == breakout) &
                                                                 (full_guarantee_df.REGION == reg) &
                                                                 (full_guarantee_df.MEASUREMENT == mes) &
                                                                 (full_guarantee_df.BG_FLAG == bg), 'CHAIN_GROUP'].unique():
                            for pharm_subgroup in full_guarantee_df.loc[(full_guarantee_df.CLIENT == client) &
                                                                        (full_guarantee_df.BREAKOUT == breakout) &
                                                                        (full_guarantee_df.REGION == reg) &
                                                                        (full_guarantee_df.MEASUREMENT == mes) &
                                                                        (full_guarantee_df.BG_FLAG == bg) &
                                                                        (full_guarantee_df.CHAIN_GROUP == pharm_group), 'CHAIN_SUBGROUP'].unique():
                                print(client,breakout,reg,mes,bg,pharm_group,pharm_subgroup)
                                mac_list = chain_region_mac_mapping.loc[(chain_region_mac_mapping.REGION==reg) &
                                                                        (chain_region_mac_mapping.MEASUREMENT==mes) &
                                                                        (chain_region_mac_mapping.CHAIN_SUBGROUP==pharm_subgroup), 'MAC_LIST'].values[0]
                                mac_ndcs = mac_list_df.loc[(mac_list_df.MAC_LIST==mac_list) &
                                                           (mac_list_df.BG_FLAG==bg) , 'GPI_NDC']
                                data_ndcs = gpi_vol_awp_agg_ytd.loc[(gpi_vol_awp_agg_ytd.CLIENT == client) &
                                                                    (gpi_vol_awp_agg_ytd.BREAKOUT == breakout) &
                                                                    (gpi_vol_awp_agg_ytd.REGION == reg) &
                                                                    (gpi_vol_awp_agg_ytd.MEASUREMENT == mes) &
                                                                    (gpi_vol_awp_agg_ytd.BG_FLAG == bg) &
                                                                    (gpi_vol_awp_agg_ytd.CHAIN_GROUP == pharm_group) &
                                                                    (gpi_vol_awp_agg_ytd.CHAIN_SUBGROUP == pharm_subgroup), 'GPI_NDC']
                                gpi_ndc = pd.concat([mac_ndcs, data_ndcs], ignore_index=True).unique()
                                gpi_ndc_df = pd.DataFrame(gpi_ndc, columns=['GPI_NDC'])
                                gpi_ndc_df['REGION'] = reg
                                gpi_ndc_df['MEASUREMENT'] = mes
                                gpi_ndc_df['BG_FLAG'] = bg
                                gpi_ndc_df['CHAIN_GROUP'] = pharm_group
                                gpi_ndc_df['CHAIN_SUBGROUP'] = pharm_subgroup

                                lp_guarant_temp = pd.merge(
                                    full_guarantee_df.loc[(full_guarantee_df.CLIENT == client) &
                                                          (full_guarantee_df.BREAKOUT == breakout) &
                                                          (full_guarantee_df.REGION == reg) &
                                                          (full_guarantee_df.MEASUREMENT == mes)  &
                                                          (full_guarantee_df.BG_FLAG == bg)  &
                                                          (full_guarantee_df.CHAIN_GROUP == pharm_group) &
                                                          (full_guarantee_df.CHAIN_SUBGROUP == pharm_subgroup)], 
                                    gpi_ndc_df, how='left', on=['REGION', 'MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP', 'CHAIN_SUBGROUP'])
                                lp_guarant = pd.concat([lp_guarant, lp_guarant_temp], ignore_index=True)
    
    #Test to make sure all GPI-NDC are present
    all_gpi_ndc = pd.concat([gpi_vol_awp_agg_ytd.GPI_NDC, mac_list_df.GPI_NDC]).unique()
    assert len(lp_guarant.loc[lp_guarant.GPI_NDC.isin(all_gpi_ndc)]) == len(lp_guarant), \
        "len(lp_guarant.loc[lp_guarant.GPI_NDC.isin(all_gpi_ndc)]) == len(lp_guarant)"
    
    lp_guarant_gpi_ndc = pd.DataFrame(
        lp_guarant.GPI_NDC.str.split(pat='_', n=1).tolist(), columns=['GPI', 'NDC'], index=lp_guarant.index)
    lp_guarant_test = pd.concat([lp_guarant, lp_guarant_gpi_ndc], axis=1)
    lp_guarant_test['GPI_NDC_Test'] = lp_guarant_test.GPI + '_' + lp_guarant_test.NDC
    assert len(lp_guarant_test.loc[lp_guarant_test.GPI_NDC != lp_guarant_test.GPI_NDC_Test]) == 0, \
        "len(lp_guarant_test.loc[lp_guarant_test.GPI_NDC != lp_guarant_test.GPI_NDC_Test]) == 0"
    
    lp_guarant = lp_guarant_test.drop(columns = ['GPI_NDC_Test'])
    
    return lp_guarant

def get_lp_vol_mac_df(lp_guarant, gpi_vol_awp_agg_ytd,chain_region_mac_mapping):
    
    merge_cols = ['CLIENT', 'BREAKOUT', 'MEASUREMENT', 'BG_FLAG',
                  'GPI_NDC', 'GPI', 'NDC', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'REGION']
    lp_vol_df = pd.merge(lp_guarant, gpi_vol_awp_agg_ytd, how="left", on=merge_cols)
    if abs((lp_vol_df.FULLAWP_ADJ.sum() - gpi_vol_awp_agg_ytd.FULLAWP_ADJ.sum()) < .0001):
        awp_lp_vol_df=lp_vol_df.groupby(['CLIENT', 'BREAKOUT', 'REGION','CHAIN_SUBGROUP','MEASUREMENT', 'BG_FLAG']).sum()
        awp_gpi_vol_awp_agg_ytd=gpi_vol_awp_agg_ytd.groupby(['CLIENT', 'BREAKOUT', 'REGION','CHAIN_SUBGROUP','MEASUREMENT','BG_FLAG']).sum()
        awp_lp=awp_lp_vol_df['FULLAWP_ADJ'].to_frame()
        awp_gpi=awp_gpi_vol_awp_agg_ytd['FULLAWP_ADJ'].to_frame()
        awp_join=awp_gpi.merge(awp_lp, how ='outer', on = ['CLIENT', 'BREAKOUT', 'REGION','CHAIN_SUBGROUP','MEASUREMENT','BG_FLAG'], suffixes=('_gpi_vol_awp_agg_ytd_measurement_mapping_report','_lp_vol_df_combined_daily_totals'))
        awp_join.to_csv(p.FILE_LOG_PATH+'lp_guarant_log_{}.csv'.format(p.DATA_ID))
        assert abs(lp_vol_df.FULLAWP_ADJ.sum() - gpi_vol_awp_agg_ytd.FULLAWP_ADJ.sum()) < .0001, \
        "abs(lp_vol_df.FULLAWP_ADJ.sum() - gpi_vol_awp_agg_ytd.FULLAWP_ADJ.sum()) < .0001, Please check the lp_guarant_log_{}.csv to determine the mismatch. Check pharm_list and CCP status in clnt_params table.  Missing values in _lp_vol_df_combined_daily_totals column then move pharmacies to non_capped.".format(p.DATA_ID)
    assert len(lp_vol_df) == len(lp_guarant), "len(lp_vol_df) == len(lp_guarant)"
    
    #add in region_mac_mapping
    #Notice the merge_cols here does not  contain BG_FLAG becuase chain_region_mac_mapping does not have BG_FLAG col
    merge_cols = ['REGION', 'CHAIN_SUBGROUP', 'MEASUREMENT']
    lp_vol_mac_df = pd.merge(
        lp_vol_df, chain_region_mac_mapping, how ='left', on = merge_cols)
    
    assert len(lp_vol_mac_df.loc[lp_vol_mac_df.MAC_LIST.isna()].index) == 0, "len(lp_vol_mac_df.loc[lp_vol_mac_df.MAC_LIST.isna()].index) == 0"
    assert abs((lp_vol_df.FULLAWP_ADJ.sum() - lp_vol_mac_df.FULLAWP_ADJ.sum()) < .0001), "abs((lp_vol_df.FULLAWP_ADJ.sum() - lp_vol_mac_df.FULLAWP_ADJ.sum()) < .0001)"
    assert len(lp_vol_mac_df) == len(lp_vol_df), "len(lp_vol_mac_df) == len(lp_vol_df)"
    
    return lp_vol_mac_df

def get_mac_list_gpi_ndc(mac_list_df):
    if p.NDC_UPDATE:
        mac_list_gpi = mac_list_df.loc[mac_list_df.NDC == '***********'].copy(deep=True)
        mac_list_gpi.rename(columns={'PRICE': 'GPI_PRICE',
                                     'OLD_PRICE': 'GPI_PRICE_OLD'}, inplace=True)
        mac_list_ndc = mac_list_df.loc[mac_list_df.NDC != '***********']
        mac_list_ndc.rename(columns={'PRICE': 'NDC_PRICE',
                                     'OLD_PRICE': 'NDC_PRICE_OLD'}, inplace=True)
        assert (len(mac_list_gpi) + len(mac_list_ndc)) == len(mac_list_df), "(len(mac_list_gpi) + len(mac_list_ndc)) == len(mac_list_df)"
    else:
        mac_list_gpi = mac_list_df.loc[mac_list_df.NDC == '***********'].copy(deep=True)
        mac_list_gpi.rename(columns={'PRICE': 'GPI_PRICE'}, inplace=True)
        mac_list_ndc = mac_list_df.loc[mac_list_df.NDC != '***********'].copy(deep=True)
        mac_list_ndc.rename(columns={'PRICE': 'NDC_PRICE'}, inplace=True)
        assert (len(mac_list_gpi) + len(mac_list_ndc)) == len(mac_list_df), "(len(mac_list_gpi) + len(mac_list_ndc)) == len(mac_list_df)"

    return mac_list_gpi,mac_list_ndc

#add in MAC list
##split MAC list between starred and non-starred NDCs
def get_lp_vol_macprice_df(mac_list_gpi, mac_list_ndc,lp_vol_mac_df, NDC_UPDATE=False):
    if p.NDC_UPDATE:
        selected_cols_ndc = ['MAC_LIST', 'MAC', 'BG_FLAG', 'NDC', 'NDC_PRICE', 'NDC_PRICE_OLD']
        selected_cols_gpi = ['MAC_LIST', 'MAC', 'BG_FLAG', 'GPI', 'GPI_PRICE', 'GPI_PRICE_OLD', 'NDC_Count','IS_MAC']
    else:
        selected_cols_ndc = ['MAC_LIST', 'MAC', 'BG_FLAG', 'NDC', 'NDC_PRICE']
        selected_cols_gpi = ['MAC_LIST', 'MAC', 'BG_FLAG', 'GPI', 'GPI_PRICE', 'NDC_Count','IS_MAC']
        
    lp_vol_macprice_df = pd.merge(
        lp_vol_mac_df, mac_list_ndc[selected_cols_ndc], 
        how ='left', on = ['NDC','MAC_LIST', 'BG_FLAG'])
    
    lp_vol_macprice_df = pd.merge(
        lp_vol_macprice_df,
        mac_list_gpi[selected_cols_gpi],
        how ='left', on = ['GPI','MAC_LIST', 'BG_FLAG'],
        suffixes=('_NDC', '_GPI'))
    
    lp_vol_macprice_df['MAC'] = np.where(lp_vol_macprice_df['MAC_NDC'].isna(), lp_vol_macprice_df['MAC_GPI'], lp_vol_macprice_df['MAC_NDC'])
    lp_vol_macprice_df.drop(columns=['MAC_NDC', 'MAC_GPI'], inplace=True)
    
    lp_vol_macprice_df.loc[lp_vol_macprice_df['MAC'].isna(), 'MAC'] = p.APPLY_VCML_PREFIX + lp_vol_macprice_df.loc[lp_vol_macprice_df['MAC'].isna(), 'MAC_LIST']
    
    if p.NDC_UPDATE:
            lp_vol_macprice_df['OLD_MAC_PRICE'] = lp_vol_macprice_df.apply(
            lambda df: df.NDC_PRICE_OLD if np.isfinite(df.NDC_PRICE_OLD) else df.GPI_PRICE_OLD, axis=1)
    
    assert abs((lp_vol_mac_df.FULLAWP_ADJ.sum() - lp_vol_macprice_df.FULLAWP_ADJ.sum()) < 0.0001), \
        "abs((lp_vol_mac_df.FULLAWP_ADJ.sum() - lp_vol_macprice_df.FULLAWP_ADJ.sum()) < 0.0001)"
    assert len(lp_vol_macprice_df) == len(lp_vol_mac_df), "len(lp_vol_macprice_df) == len(lp_vol_mac_df)"

    
    #Idenfity the items to collapse into NDC stars
    lp_vol_macprice_df['CURRENT_MAC_PRICE'] = lp_vol_macprice_df.apply(
        lambda df: df.NDC_PRICE if np.isfinite(df.NDC_PRICE) else df.GPI_PRICE, axis=1)
    lp_vol_macprice_df['GPI_Collapse'] = lp_vol_macprice_df.apply(
        lambda df: 1 if (np.isnan(df.NDC_PRICE) & np.isfinite(df.GPI_PRICE)) else 0, axis=1)
    lp_vol_macprice_df['GPI_ONLY'] = lp_vol_macprice_df.apply(
        lambda df: 1 if (df.GPI_Collapse == 1) & (df.NDC_Count == 1) else 0, axis=1)
    lp_vol_macprice_df['MAC_GPI_FLAG'] = lp_vol_macprice_df.apply(
        lambda df: 1 if np.isnan(df.CURRENT_MAC_PRICE) else 0, axis=1)
    assert len(lp_vol_macprice_df.loc[
        (lp_vol_macprice_df.GPI_Collapse == 1) & 
        (lp_vol_macprice_df.QTY == 0) & 
        (lp_vol_macprice_df.GPI_NDC.isin(mac_list_ndc.GPI_NDC))]) == 0, "len[GPI_Collapse == 1, qty == 0, gpi_ndc in list] == 0"
    
    return lp_vol_macprice_df

def get_lp_vol_ytd_gpi(lp_vol_macprice_df, mac_list_gpi, mac_list_ndc, pkg_size):

    #Condense down GPI only for MAC Lists
    lp_vol_ytd_gpi = lp_vol_macprice_df.loc[lp_vol_macprice_df.GPI_Collapse == 1]
    assert (len(mac_list_gpi.loc[
        mac_list_gpi.GPI_NDC.isin(lp_vol_ytd_gpi.GPI_NDC)]) == len(mac_list_gpi)), \
        "GPI MAC list length mismatches. Client may have prices that are dated to go live at a future date. Verify and consider if the run is still appropriate."
    lp_vol_ytd_ndc = lp_vol_macprice_df.loc[lp_vol_macprice_df.GPI_Collapse != 1]
    assert (len(mac_list_ndc.loc[
        mac_list_ndc.GPI_NDC.isin(lp_vol_ytd_ndc.GPI_NDC)]) == len(mac_list_ndc)), \
        "NDC MAC list length mismatches. Client may have prices that are dated to go live at a future date. Verify and consider if the run is still appropriate."
    
    old_len = len(lp_vol_ytd_ndc)
    lp_vol_ytd_ndc = pd.merge(
        lp_vol_ytd_ndc, pkg_size[['PKG_SZ', 'GPI_NDC']], on=['GPI_NDC'], how='left')
    assert len(lp_vol_ytd_ndc) == old_len, "len(lp_vol_ytd_ndc) == old_len"
    
    return lp_vol_ytd_gpi, lp_vol_ytd_ndc

def get_lp_data_df(
    lp_vol_ytd_gpi, lp_vol_ytd_ndc, lp_vol_mac_df, lp_vol_macprice_df, 
    mac_list_df, mac_list_gpi,lp_vol_mac_df_actual = pd.DataFrame(),
    NDC_UPDATE=False, INCLUDE_PLAN_LIABILITY=False,UNC_OPT=False):
    '''
    Most of the logic revolves around plan liability.
    The shared part of the logic is an aggregation at the GPI level.


    Inputs:
        lp_vol_ytd_gpi:
        lp_vol_ytd_ndc:
        lp_vol_mac_df:
        lp_vol_macprice_df:
        mac_list_df:
        mac_list_gpi:
        lp_vol_mac_df_actual: Default empty. Is the claims for the MAC Adj. claims
        NDC_UPDATE: Parameter, default false.
        INCLUDE_PLAN_LIABILITY: Parameter, default false.
        UNC_OPT: Parameter, default false.

    Outputs:
        lp_data_culled_df
    '''
    
    lp_vol_ytd_gpi = lp_vol_ytd_gpi.copy(deep=True)
    lp_vol_ytd_ndc = lp_vol_ytd_ndc.copy(deep=True)
    if NDC_UPDATE:
        agg_columns = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'BG_FLAG', 'GPI', 
                       'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GO_LIVE', 'MAC_LIST','MAC','OLD_MAC_PRICE', 
                       'CURRENT_MAC_PRICE', 'GPI_ONLY','IS_MAC']
    else:
        agg_columns = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'BG_FLAG', 'GPI', 
                       'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GO_LIVE', 'MAC_LIST','MAC','CURRENT_MAC_PRICE', 
                       'GPI_ONLY','IS_MAC']

    sum_columns = ['CLAIMS', 'QTY', 'FULLAWP_ADJ', 'PRICE_REIMB',
                   'LM_CLAIMS', 'LM_QTY', 'LM_FULLAWP_ADJ', 'LM_PRICE_REIMB',
                   'uc_unit_agg', 'uc_unit25_agg', 'CURR_AWP_AGG',
                   'PHARM_CLAIMS', 'PHARM_QTY', 'PHARM_FULLAWP_ADJ', 'PHARM_FULLNADAC_ADJ', 'PHARM_FULLACC_ADJ', 'PHARM_TARG_INGCOST_ADJ', 'PHARM_PRICE_REIMB',
                   'LM_PHARM_CLAIMS', 'LM_PHARM_QTY', 'LM_PHARM_FULLAWP_ADJ', 'LM_PHARM_FULLNADAC_ADJ', 'LM_PHARM_FULLACC_ADJ', 'LM_PHARM_TARG_INGCOST_ADJ', 'LM_PHARM_PRICE_REIMB',
                   'CLAIMS_ZBD','QTY_ZBD','FULLAWP_ADJ_ZBD',
                   'PHARM_CLAIMS_ZBD','PHARM_QTY_ZBD','PHARM_FULLAWP_ADJ_ZBD','PHARM_FULLNADAC_ADJ_ZBD','PHARM_FULLACC_ADJ_ZBD','PHARM_TARG_INGCOST_ADJ_ZBD',
                   'CS_LM_CLAIMS', 'CS_LM_QTY', 'CS_LM_FULLAWP_ADJ', 
                   'CS_LM_PHARM_CLAIMS', 'CS_LM_PHARM_QTY', 'CS_LM_PHARM_FULLAWP_ADJ', 'CS_LM_PHARM_FULLNADAC_ADJ',  'CS_LM_PHARM_FULLACC_ADJ','CS_LM_PHARM_TARG_INGCOST_ADJ', 
                   'CS_LM_CLAIMS_ZBD', 'CS_LM_QTY_ZBD', 'CS_LM_FULLAWP_ADJ_ZBD', 
                   'CS_LM_PHARM_CLAIMS_ZBD', 'CS_LM_PHARM_QTY_ZBD', 'CS_LM_PHARM_FULLAWP_ADJ_ZBD', 'CS_LM_PHARM_FULLNADAC_ADJ_ZBD',  'CS_LM_PHARM_FULLACC_ADJ_ZBD', 'CS_LM_PHARM_TARG_INGCOST_ADJ_ZBD',
                   'DISP_FEE', 'TARGET_DISP_FEE', 'PHARMACY_DISP_FEE', 'PHARM_TARGET_DISP_FEE', 'NADAC_WAC_YTD']
    
    proj_columns = ['CLAIMS_PROJ_LAG', 'QTY_PROJ_LAG', 'FULLAWP_ADJ_PROJ_LAG',
                    'CLAIMS_PROJ_EOY', 'QTY_PROJ_EOY', 'FULLAWP_ADJ_PROJ_EOY',
                    'PHARM_CLAIMS_PROJ_LAG', 'PHARM_QTY_PROJ_LAG', 'PHARM_FULLAWP_ADJ_PROJ_LAG', 
                    'PHARM_CLAIMS_PROJ_EOY', 'PHARM_QTY_PROJ_EOY', 'PHARM_FULLAWP_ADJ_PROJ_EOY']
    
    if not UNC_OPT and p.UNC_OPT:
        sum_columns.extend(['UC_CLAIMS', 'LM_UC_CLAIMS'])
    
    if p.CROSS_CONTRACT_PROJ:
        proj_columns = proj_columns + ['CLAIMS_PRIOR', 'QTY_PRIOR', 'FULLAWP_ADJ_PRIOR', 'TARG_INGCOST_ADJ_PRIOR','PRICE_REIMB_PRIOR', 
             'PHARM_CLAIMS_PRIOR','PHARM_QTY_PRIOR','PHARM_FULLAWP_ADJ_PRIOR','PHARM_TARG_INGCOST_ADJ_PRIOR',
                                       'PHARM_PRICE_REIMB_PRIOR']

    min_columns = ['CURR_AWP_MIN']
    max_columns = ['CURR_AWP_MAX']
    
    lp_vol_ytd_gpi['PRICING_QTY'] = lp_vol_ytd_gpi[['QTY', 'PHARM_QTY']].max(axis=1, skipna=True)
    lp_vol_ytd_gpi['uc_unit_agg'] = lp_vol_ytd_gpi['UC_UNIT'] * lp_vol_ytd_gpi['PRICING_QTY']
    lp_vol_ytd_gpi['uc_unit25_agg'] = lp_vol_ytd_gpi['UC_UNIT25'] * lp_vol_ytd_gpi['PRICING_QTY']
    lp_vol_ytd_gpi['CURR_AWP_AGG'] = lp_vol_ytd_gpi['CURR_AWP'] * lp_vol_ytd_gpi['PRICING_QTY']
    lp_vol_ytd_gpi.drop(columns=['PRICING_QTY'], inplace=True)
    
    lp_vol_ytd_ndc['PRICING_QTY'] = lp_vol_ytd_ndc[['QTY', 'PHARM_QTY']].max(axis=1, skipna=True)
    lp_vol_ytd_ndc['CURR_AWP_AGG'] = lp_vol_ytd_ndc['CURR_AWP'] * lp_vol_ytd_ndc['PRICING_QTY']
    lp_vol_ytd_ndc.drop(columns=['PRICING_QTY'], inplace=True)

    if INCLUDE_PLAN_LIABILITY:
        lp_vol_ytd_gpi['NUM30DAYS'] = lp_vol_ytd_gpi['DAYSSUP']/30
        lp_vol_ytd_gpi['COPAY_AGG'] = lp_vol_ytd_gpi['COPAY'] * lp_vol_ytd_gpi['NUM30DAYS']
        # lp_vol_ytd_gpi['COPAY_RAW_AGG'] = lp_vol_ytd_gpi['COPAY_RAW'] * lp_vol_ytd_gpi['NUM30DAYS']
        lp_vol_ytd_gpi['COPAY_DCS_AGG'] = lp_vol_ytd_gpi['COPAY'] / (lp_vol_ytd_gpi['DAYSSUP'])
        lp_vol_ytd_gpi['COPAY_RATE_AGG'] = lp_vol_ytd_gpi['COPAY_RATE'] * lp_vol_ytd_gpi['DAYSSUP']
        lp_vol_ytd_gpi['PROP_LIS_AGG'] = lp_vol_ytd_gpi['PROP_LIS'] * lp_vol_ytd_gpi['CLAIMS']
        lp_vol_ytd_gpi['PROP_DEDUCT_AGG'] = lp_vol_ytd_gpi['PROP_DEDUCT'] * lp_vol_ytd_gpi['FULLAWP_ADJ']
        lp_vol_ytd_gpi['PROP_ICL_AGG'] = lp_vol_ytd_gpi['PROP_ICL'] * lp_vol_ytd_gpi['FULLAWP_ADJ']
        lp_vol_ytd_gpi['PROP_GAP_AGG'] = lp_vol_ytd_gpi['PROP_GAP'] * lp_vol_ytd_gpi['FULLAWP_ADJ']
        lp_vol_ytd_gpi['PROP_CAT_AGG'] = lp_vol_ytd_gpi['PROP_CAT'] * lp_vol_ytd_gpi['FULLAWP_ADJ']
        lp_vol_ytd_gpi['TIER_AGG'] = lp_vol_ytd_gpi['TIER'] * lp_vol_ytd_gpi['FULLAWP_ADJ']
        lp_vol_ytd_gpi['MIN_DAYS_AGG'] = lp_vol_ytd_gpi['MIN_DAYS_SCRIPT'] * lp_vol_ytd_gpi['FULLAWP_ADJ']
        agg_columns = agg_columns + ['DCS']
        sum_columns = sum_columns + ['COPAY_AGG', 'COPAY_RATE_AGG', 'DAYSSUP', 'PSTCOPAY', 'PROP_LIS_AGG',
                                     'PROP_DEDUCT_AGG', 'PROP_ICL_AGG', 'PROP_GAP_AGG', 'PROP_CAT_AGG', 'TIER_AGG',
                                     'MIN_DAYS_AGG', 'COPAY_DCS_AGG'] # 'COPAY_RAW_AGG']
        
    lp_vol_ytd_ndc.loc[:, agg_columns] = lp_vol_ytd_ndc.loc[:, agg_columns].fillna(0)
    lp_vol_ytd_gpi.loc[:, agg_columns] = lp_vol_ytd_gpi.loc[:, agg_columns].fillna(0)
    lp_vol_ytd_gpi_agg = lp_vol_ytd_gpi.groupby(agg_columns)
    lp_vol_ytd_gpi_only = lp_vol_ytd_gpi_agg[sum_columns + proj_columns].agg(sum)

    # lp_vol_ytd_gpi_only = pd.concat([lp_vol_ytd_gpi_only, lp_vol_ytd_gpi_agg[avg_columns].agg(np.nanmedian)], axis=1)

    lp_vol_ytd_gpi_only = pd.concat([lp_vol_ytd_gpi_only, lp_vol_ytd_gpi_agg[min_columns].agg(np.nanmin)], axis=1)
    lp_vol_ytd_gpi_only = pd.concat([lp_vol_ytd_gpi_only, lp_vol_ytd_gpi_agg[max_columns].agg(np.nanmax)], axis=1).reset_index()
    lp_vol_ytd_gpi_only['NDC'] = '***********'

    lp_vol_ytd_gpi_only['GPI_NDC'] = lp_vol_ytd_gpi_only['GPI'] + '_' + lp_vol_ytd_gpi_only['NDC']
    lp_vol_ytd_gpi_only['PKG_SZ'] = 0.0
    assert (len(mac_list_gpi.loc[mac_list_gpi.GPI_NDC.isin(lp_vol_ytd_gpi_only.GPI_NDC)]) == len(mac_list_gpi)), \
        "len(mac_list_gpi.loc[mac_list_gpi.GPI_NDC.isin(lp_vol_ytd_gpi_only.GPI_NDC)]) == len(mac_list_gpi)"
    
    lp_vol_ytd_gpi_only['PRICING_QTY'] = lp_vol_ytd_gpi_only[['QTY', 'PHARM_QTY']].max(axis=1, skipna=True)
    lp_vol_ytd_gpi_only['UC_UNIT'] = lp_vol_ytd_gpi_only['uc_unit_agg' ]/ lp_vol_ytd_gpi_only['PRICING_QTY']
    lp_vol_ytd_gpi_only['UC_UNIT25'] = lp_vol_ytd_gpi_only['uc_unit25_agg' ]/ lp_vol_ytd_gpi_only['PRICING_QTY']
    lp_vol_ytd_gpi_only['CURR_AWP'] = lp_vol_ytd_gpi_only['CURR_AWP_AGG' ]/ lp_vol_ytd_gpi_only['PRICING_QTY']
    lp_vol_ytd_gpi_only.drop(columns=['PRICING_QTY'], inplace=True)

    if INCLUDE_PLAN_LIABILITY:
        lp_vol_ytd_gpi_only['COPAY'] = lp_vol_ytd_gpi_only['COPAY_AGG']/(lp_vol_ytd_gpi_only['DAYSSUP']/30)
        # lp_vol_ytd_gpi_only['COPAY_RAW'] = lp_vol_ytd_gpi_only['COPAY_RAW_AGG']/(lp_vol_ytd_gpi_only['DAYSSUP']/30)
        lp_vol_ytd_gpi_only['COPAY_DCS'] = lp_vol_ytd_gpi_only['COPAY_DCS_AGG']/(lp_vol_ytd_gpi_only['DAYSSUP'])
        lp_vol_ytd_gpi_only['COPAY_RATE'] = lp_vol_ytd_gpi_only['COPAY_RATE_AGG' ]/ lp_vol_ytd_gpi_only['DAYSSUP']
        lp_vol_ytd_gpi_only['PROP_LIS'] = lp_vol_ytd_gpi_only['PROP_LIS_AGG'] / lp_vol_ytd_gpi_only['CLAIMS']
        lp_vol_ytd_gpi_only['PROP_DEDUCT'] = lp_vol_ytd_gpi_only['PROP_DEDUCT_AGG'] / lp_vol_ytd_gpi_only['FULLAWP_ADJ']
        lp_vol_ytd_gpi_only['PROP_ICL'] = lp_vol_ytd_gpi_only['PROP_ICL_AGG'] / lp_vol_ytd_gpi_only['FULLAWP_ADJ']
        lp_vol_ytd_gpi_only['PROP_GAP'] = lp_vol_ytd_gpi_only['PROP_GAP_AGG'] / lp_vol_ytd_gpi_only['FULLAWP_ADJ']
        lp_vol_ytd_gpi_only['PROP_CAT'] = lp_vol_ytd_gpi_only['PROP_CAT_AGG'] / lp_vol_ytd_gpi_only['FULLAWP_ADJ']
        lp_vol_ytd_gpi_only['TIER'] = lp_vol_ytd_gpi_only['TIER_AGG'] / lp_vol_ytd_gpi_only['FULLAWP_ADJ']
        lp_vol_ytd_gpi_only['MIN_DAYS_SCRIPT'] = lp_vol_ytd_gpi_only['MIN_DAYS_AGG'] / lp_vol_ytd_gpi_only['FULLAWP_ADJ']
    
        lp_vol_ytd_gpi_only.loc[lp_vol_ytd_gpi_only.DCS=='N', 'COPAY'] = lp_vol_ytd_gpi_only.loc[lp_vol_ytd_gpi_only.DCS=='N', 'COPAY_DCS']
    
    
        lp_vol_ytd_gpi_only['NUM30DAYS'] = lp_vol_ytd_gpi_only['DAYSSUP']/30
        lp_vol_ytd_gpi_only['COPAY_AGG'] = lp_vol_ytd_gpi_only['COPAY'] * lp_vol_ytd_gpi_only['FULLAWP_ADJ']
        lp_vol_ytd_gpi_only['COPAY_RATE_AGG'] = lp_vol_ytd_gpi_only['COPAY_RATE'] * lp_vol_ytd_gpi_only['DAYSSUP']
        lp_vol_ytd_gpi_only['PROP_LIS_AGG'] = lp_vol_ytd_gpi_only['PROP_LIS'] * lp_vol_ytd_gpi_only['CLAIMS']
        lp_vol_ytd_gpi_only['PROP_DEDUCT_AGG'] = lp_vol_ytd_gpi_only['PROP_DEDUCT'] * lp_vol_ytd_gpi_only['FULLAWP_ADJ']
        lp_vol_ytd_gpi_only['PROP_ICL_AGG'] = lp_vol_ytd_gpi_only['PROP_ICL'] * lp_vol_ytd_gpi_only['FULLAWP_ADJ']
        lp_vol_ytd_gpi_only['PROP_GAP_AGG'] = lp_vol_ytd_gpi_only['PROP_GAP'] * lp_vol_ytd_gpi_only['FULLAWP_ADJ']
        lp_vol_ytd_gpi_only['PROP_CAT_AGG'] = lp_vol_ytd_gpi_only['PROP_CAT'] * lp_vol_ytd_gpi_only['FULLAWP_ADJ']
        lp_vol_ytd_gpi_only['TIER_AGG'] = lp_vol_ytd_gpi_only['TIER'] * lp_vol_ytd_gpi_only['FULLAWP_ADJ']
        lp_vol_ytd_gpi_only['MIN_DAYS_AGG'] = lp_vol_ytd_gpi_only['MIN_DAYS_SCRIPT'] * lp_vol_ytd_gpi_only['FULLAWP_ADJ']
    
        agg_columns.remove('DCS')
        agg_columns += ['NDC', 'GPI_NDC', 'PKG_SZ']
        sum_columns.remove('COPAY_DCS_AGG')
        lp_vol_ytd_gpi_agg2 = lp_vol_ytd_gpi_only.groupby(agg_columns)
        lp_vol_ytd_gpi_only2 = lp_vol_ytd_gpi_agg2[sum_columns + proj_columns].agg(sum)
        # lp_vol_ytd_gpi_only2 = pd.concat([lp_vol_ytd_gpi_only2, lp_vol_ytd_gpi_agg2[avg_columns].agg(np.nanmedian)], axis=1)
        lp_vol_ytd_gpi_only2 = pd.concat([lp_vol_ytd_gpi_only2, lp_vol_ytd_gpi_agg2[min_columns].agg(np.nanmin)], axis=1)
        lp_vol_ytd_gpi_only2 = pd.concat([lp_vol_ytd_gpi_only2, lp_vol_ytd_gpi_agg2[max_columns].agg(np.nanmax)], axis=1).reset_index()
    
        lp_vol_ytd_gpi_only2['COPAY'] = lp_vol_ytd_gpi_only2['COPAY_AGG']/(lp_vol_ytd_gpi_only2['FULLAWP_ADJ'])
        # lp_vol_ytd_gpi_only['COPAY_RAW'] = lp_vol_ytd_gpi_only['COPAY_RAW_AGG']/(lp_vol_ytd_gpi_only['DAYSSUP']/30)
        lp_vol_ytd_gpi_only2['COPAY_RATE'] = lp_vol_ytd_gpi_only2['COPAY_RATE_AGG' ]/ lp_vol_ytd_gpi_only2['DAYSSUP']
        lp_vol_ytd_gpi_only2['PROP_LIS'] = lp_vol_ytd_gpi_only2['PROP_LIS_AGG'] / lp_vol_ytd_gpi_only2['CLAIMS']
        lp_vol_ytd_gpi_only2['PROP_DEDUCT'] = lp_vol_ytd_gpi_only2['PROP_DEDUCT_AGG'] / lp_vol_ytd_gpi_only2['FULLAWP_ADJ']
        lp_vol_ytd_gpi_only2['PROP_ICL'] = lp_vol_ytd_gpi_only2['PROP_ICL_AGG'] / lp_vol_ytd_gpi_only2['FULLAWP_ADJ']
        lp_vol_ytd_gpi_only2['PROP_GAP'] = lp_vol_ytd_gpi_only2['PROP_GAP_AGG'] / lp_vol_ytd_gpi_only2['FULLAWP_ADJ']
        lp_vol_ytd_gpi_only2['PROP_CAT'] = lp_vol_ytd_gpi_only2['PROP_CAT_AGG'] / lp_vol_ytd_gpi_only2['FULLAWP_ADJ']
        lp_vol_ytd_gpi_only2['TIER'] = lp_vol_ytd_gpi_only2['TIER_AGG'] / lp_vol_ytd_gpi_only2['FULLAWP_ADJ']
        lp_vol_ytd_gpi_only2['MIN_DAYS_SCRIPT'] = lp_vol_ytd_gpi_only2['MIN_DAYS_AGG'] / lp_vol_ytd_gpi_only2['FULLAWP_ADJ']

    lp_data_df = pd.concat([lp_vol_ytd_gpi_only, lp_vol_ytd_ndc[lp_vol_ytd_gpi_only.columns]]).sort_values(agg_columns)
    assert abs(lp_data_df.FULLAWP_ADJ.sum() - lp_vol_macprice_df.FULLAWP_ADJ.sum()) < .0001, \
        "abs(lp_data_df.FULLAWP_ADJ.sum() - lp_vol_macprice_df.FULLAWP_ADJ.sum()) < .0001"
    
    # drop all items that do not have a MAC price, client claims, or pharmacy claims
    # this should never occur because all claims should have qty > 0 or a MAC price
    lp_data_culled_df = lp_data_df.loc[(lp_data_df.CURRENT_MAC_PRICE > 0) | (lp_data_df.QTY > 0) | (lp_data_df.PHARM_QTY > 0)].copy(deep=True)

    if not UNC_OPT:
        assert abs((lp_data_culled_df.FULLAWP_ADJ.sum() - lp_vol_mac_df.FULLAWP_ADJ.sum()) < .0001), \
            "abs((lp_data_culled_df.FULLAWP_ADJ.sum() - lp_vol_mac_df.FULLAWP_ADJ.sum()) < .0001)"
        assert (len(mac_list_df.loc[mac_list_df.GPI_NDC.isin(lp_data_culled_df.GPI_NDC)]) == len(mac_list_df)), \
            "(len(mac_list_df.loc[mac_list_df.GPI_NDC.isin(lp_data_culled_df.GPI_NDC)]) == len(mac_list_df))"
        if not (p.TRUECOST_CLIENT or p.UCL_CLIENT):
            assert len(lp_data_culled_df) == len(lp_data_df), "len(lp_data_culled_df) == len(lp_data_df)"
    
    #Drop NONPREF_OTH from SELECT since NONPREF_OTH doesn't exist at SELECT
    lp_data_culled_df = lp_data_culled_df.loc[
        (lp_data_culled_df.BREAKOUT != 'SELECT') | (lp_data_culled_df.CHAIN_GROUP != 'NONPREF_OTH')].copy(deep=True)
    if UNC_OPT:
        assert abs((lp_data_culled_df.FULLAWP_ADJ.sum() - lp_vol_mac_df_actual.FULLAWP_ADJ.sum()) < .0001), \
            "abs((lp_data_culled_df.FULLAWP_ADJ.sum() - lp_vol_mac_df_actual.FULLAWP_ADJ.sum()) < .0001)"
    else:
        assert abs((lp_data_culled_df.FULLAWP_ADJ.sum() - lp_vol_mac_df.FULLAWP_ADJ.sum()) < .0001), \
            "abs((lp_data_culled_df.FULLAWP_ADJ.sum() - lp_vol_mac_df.FULLAWP_ADJ.sum()) < .0001)"
    assert (len(mac_list_df.loc[mac_list_df.GPI_NDC.isin(lp_data_culled_df.GPI_NDC)]) == len(mac_list_df)), \
        "(len(mac_list_df.loc[mac_list_df.GPI_NDC.isin(lp_data_culled_df.GPI_NDC)]) == len(mac_list_df))"
    
    lp_data_culled_df.loc[:,'AVG_AWP'] = lp_data_culled_df.FULLAWP_ADJ/lp_data_culled_df.QTY
    # lp_data_culled_df.loc[:,'AVG_NADAC'] = lp_data_culled_df.FULLAWP_ADJ/lp_data_culled_df.QTY
    # lp_data_culled_df.loc[:,'AVG_ACC'] = lp_data_culled_df.FULLAWP_ADJ/lp_data_culled_df.QTY
    
    # To be used for Target Ingredient Cost projection (LAG,EOY) calculations
    lp_data_culled_df.loc[:,'PHARM_AVG_AWP'] = lp_data_culled_df.PHARM_FULLAWP_ADJ/lp_data_culled_df.PHARM_QTY
    
    # merge median NADAC/WAC prices
    # lp_data_culled_df.loc[:,'PHARM_AVG_NADAC'] = lp_data_culled_df.PHARM_FULLNADAC_ADJ/lp_data_culled_df.PHARM_QTY
    nadac_wac_df = read_nadac_wac_gpi_price(READ_FROM_BQ = p.READ_FROM_BQ)
    lp_data_culled_df_temp = pd.merge(lp_data_culled_df,nadac_wac_df,on=['CLIENT','GPI','BG_FLAG'],how='left')
    
    assert len(lp_data_culled_df) == len(lp_data_culled_df_temp), "len(lp_data_culled_df) == len(lp_data_culled_df_temp)"
    lp_data_culled_df = lp_data_culled_df_temp
    
    # merge recent costvantage prices
    costvantage_df = read_costvantage_price(READ_FROM_BQ = p.READ_FROM_BQ)
    lp_data_culled_df_temp = pd.merge(lp_data_culled_df,costvantage_df,on=['CHAIN_GROUP','GPI','BG_FLAG'],how='left')
    # lp_data_culled_df.loc[:,'PHARM_AVG_ACC'] = lp_data_culled_df.PHARM_TARG_INGCOST_ADJ/lp_data_culled_df.PHARM_QTY
    assert len(lp_data_culled_df) == len(lp_data_culled_df_temp), "len(lp_data_culled_df) == len(lp_data_culled_df_temp)"
    lp_data_culled_df = lp_data_culled_df_temp
    
    return lp_data_culled_df


def update_new_mac_list(lp_data_culled_df):
    #Read in new year region mac mapping file
    new_chain_region_mac_mapping = standardize_df(pd.read_csv(p.FILE_INPUT_PATH + p.CUSTOM_MAC_MAPPING_FILE, dtype = p.VARIABLE_TYPE_DIC))
    new_chain_region_mac_mapping = new_chain_region_mac_mapping[new_chain_region_mac_mapping.CUSTOMER_ID == p.CUSTOMER_ID[0]]
    new_chain_region_mac_mapping.rename(columns = {'CUSTOMER_ID': 'CLIENT'}, inplace = True)
    # For backwards compatibility before CHAIN_SUBGROUP was introduced
    if 'CHAIN_SUBGROUP' not in new_chain_region_mac_mapping.columns:
        new_chain_region_mac_mapping['CHAIN_SUBGROUP'] = new_chain_region_mac_mapping['CHAIN_GROUP']
    
    new_chain_region_mac_mapping.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.CUSTOM_MAC_MAPPING_FILE)
    
    lp_data_culled_df['MAC_LIST_OLD'] = lp_data_culled_df['MAC_LIST'].copy()
    lp_data_culled_df = lp_data_culled_df.drop(columns = ['MAC_LIST'])
    
    old_len = len(lp_data_culled_df)
    #Notice BG_FLAG is not included in the merge_col here becuase new_chain_region_mac_mapping does not have BG_FLAG
    lp_data_culled_df = lp_data_culled_df.merge(new_chain_region_mac_mapping, how = 'left', on = ['CLIENT', 'REGION', 'MEASUREMENT','CHAIN_SUBGROUP'])
    
    assert len(lp_data_culled_df) == old_len, "len(mac_list_df) == old_len"
    
    assert len(lp_data_culled_df.loc[lp_data_culled_df.MAC_LIST.isna()].index) == 0, \
        "len(lp_data_culled_df.loc[lp_data_culled_df.MAC_LIST.isna()].index) == 0"
    
    return lp_data_culled_df


def add_mac_1026_to_lp_data_culled_df(lp_data_culled_df, mac_1026_gpi, mac_1026_ndc):
    
    #Get Breakout AWP MAX
    lp_breakout_awp_max = lp_data_culled_df.groupby(
        by=['BREAKOUT', 'REGION', 'GPI_NDC'])['CURR_AWP_MAX']\
        .agg(np.nanmax).reset_index()
    lp_breakout_awp_max.rename(columns={'CURR_AWP_MAX': 'BREAKOUT_AWP_MAX'}, inplace=True)
    lp_data_culled_temp = pd.merge(
        lp_data_culled_df, lp_breakout_awp_max, how='left', on=['BREAKOUT', 'REGION', 'GPI_NDC'])
    assert len(lp_data_culled_temp) == len(lp_data_culled_df), "len(lp_data_culled_temp) == len(lp_data_culled_df)"
    lp_data_culled_df = lp_data_culled_temp.copy()
    del lp_data_culled_temp
    
    
    lp_data_culled_df_temp = pd.merge(
        lp_data_culled_df, mac_1026_ndc[['NDC', 'BG_FLAG', '1026_NDC_PRICE']], how ='left', on = ['NDC','BG_FLAG'])
    lp_data_culled_df_temp = pd.merge(
        lp_data_culled_df_temp, mac_1026_gpi[['GPI', 'BG_FLAG', '1026_GPI_PRICE']], how ='left', on = ['GPI','BG_FLAG'])
    assert abs((lp_data_culled_df_temp.FULLAWP_ADJ.sum() - lp_data_culled_df.FULLAWP_ADJ.sum()) < .0001), \
        "abs((lp_data_culled_df_temp.FULLAWP_ADJ.sum() - lp_data_culled_df.FULLAWP_ADJ.sum()) < .0001)"
    assert len(lp_data_culled_df) == len(lp_data_culled_df_temp), "len(lp_data_culled_df) == len(lp_data_culled_df_temp)"
    
    lp_data_culled_df_temp['MAC1026_UNIT_PRICE'] = lp_data_culled_df_temp.apply(
        lambda df: df['1026_NDC_PRICE'] if np.isfinite(df['1026_NDC_PRICE']) else df['1026_GPI_PRICE'], axis=1)
    lp_data_culled_df_temp['MAC1026_GPI_FLAG'] = lp_data_culled_df_temp.apply(
        lambda df: 1 if np.isnan(df.MAC1026_UNIT_PRICE) else 0, axis=1)
    
    lp_data_culled_df = lp_data_culled_df_temp

    return lp_data_culled_df


def add_goodrx_to_lp_data_culled_df(lp_data_culled_df,GOODRX_OPT=False):
    
    if p.GOODRX_OPT:        
        goodrx_df = pd.read_csv(
            p.FILE_DYNAMIC_INPUT_PATH + p.GOODRX_FILE, dtype = p.VARIABLE_TYPE_DIC
        ).rename(columns={'GOODRX_UPPER_LIMIT': 'GOODRX_CHAIN_PRICE'})
        goodrx_df = goodrx_df.dropna()
        goodrx_df = standardize_df(goodrx_df)\
            .sort_values("GOODRX_CHAIN_PRICE", ascending=True)\
            .drop_duplicates(subset=['GPI', 'BG_FLAG', 'CHAIN_GROUP'], keep='first')    
        goodrx_df.loc[goodrx_df['CHAIN_GROUP']=='IND', 'CHAIN_GROUP'] = 'NONPREF_OTH'        
        qa_dataframe(goodrx_df, dataset = 'GOODRX_FILE_AT_{}'.format(os.path.basename(__file__)))
        lp_data_culled_df_temp = pd.merge(
            lp_data_culled_df, goodrx_df, how='left', on=['GPI', 'BG_FLAG', 'CHAIN_GROUP'])
        goodrx_min_df = goodrx_df.groupby(['GPI','BG_FLAG'], as_index=False)['GOODRX_CHAIN_PRICE']\
            .min().rename(columns={'GOODRX_CHAIN_PRICE': 'GOODRX_MIN_PRICE'})
        goodrx_df.rename(columns={'GOODRX_CHAIN_PRICE': 'GOODRX_GPI_PRICE'}, inplace=True)
        goodrx_df = goodrx_df[goodrx_df['CHAIN_GROUP']=='NONPREF_OTH'].drop(columns=['CHAIN_GROUP'])
        lp_data_culled_df_temp = pd.merge(lp_data_culled_df_temp, goodrx_df, how='left', on=['GPI','BG_FLAG'])
        lp_data_culled_df_temp = pd.merge(lp_data_culled_df_temp, goodrx_min_df, how='left', on=['GPI','BG_FLAG'])
        lp_data_culled_df_temp['DUMMY_GOODRX_PRICE'] = 10000*lp_data_culled_df_temp['CURRENT_MAC_PRICE']
        lp_data_culled_df_temp['GOODRX_UPPER_LIMIT'] = np.nan
        mask = lp_data_culled_df['CHAIN_GROUP'].isin(['WMT', 'WAG', 'RAD', 'KRG', 'CVS'])
        lp_data_culled_df_temp.loc[mask, 'GOODRX_UPPER_LIMIT'] = \
            lp_data_culled_df_temp.loc[mask, ['GOODRX_CHAIN_PRICE', 'DUMMY_GOODRX_PRICE']].min(axis=1)
        lp_data_culled_df_temp.loc[~mask, 'GOODRX_UPPER_LIMIT'] = \
            lp_data_culled_df_temp.loc[~mask, ['GOODRX_GPI_PRICE', 'DUMMY_GOODRX_PRICE']].min(axis=1)
        
        lp_data_culled_df_temp.loc[lp_data_culled_df_temp['BREAKOUT'].str.contains('M'), 'GOODRX_UPPER_LIMIT'] =\
            lp_data_culled_df_temp.loc[lp_data_culled_df_temp['BREAKOUT'].str.contains('M'), ['GOODRX_MIN_PRICE', 'DUMMY_GOODRX_PRICE']].min(axis=1)
        
        mask = lp_data_culled_df['CHAIN_GROUP']=='CVS'
        
        lp_data_culled_df_temp.loc[mask, 'GOODRX_UPPER_LIMIT'] = lp_data_culled_df_temp.loc[mask, ['GOODRX_UPPER_LIMIT', 'GOODRX_GPI_PRICE']].min(axis=1)
        lp_data_culled_df_temp.rename(columns={'GOODRX_UPPER_LIMIT': 'INDIVIDUAL_ROW_GOODRX_UPPER_LIMIT'}, inplace=True)
        lp_data_culled_df_by_vcml = lp_data_culled_df_temp.groupby(['GPI_NDC', 'CLIENT', 'BREAKOUT', 'BG_FLAG',
                         'REGION', 'MEASUREMENT', 'MAC_LIST'], as_index=False)['INDIVIDUAL_ROW_GOODRX_UPPER_LIMIT'].min()
        lp_data_culled_df_by_vcml.rename(columns={'INDIVIDUAL_ROW_GOODRX_UPPER_LIMIT': 'GOODRX_UPPER_LIMIT'}, inplace=True)
        lp_data_culled_df_temp = lp_data_culled_df_temp.merge(lp_data_culled_df_by_vcml, on=['GPI_NDC', 'CLIENT', 'BREAKOUT', 'BG_FLAG',
                         'REGION', 'MEASUREMENT', 'MAC_LIST'], how='left')
        lp_data_culled_df_temp.drop(
            columns=['GOODRX_CHAIN_PRICE', 'GOODRX_GPI_PRICE', 'GOODRX_MIN_PRICE', 'DUMMY_GOODRX_PRICE', 'INDIVIDUAL_ROW_GOODRX_UPPER_LIMIT'], inplace=True)

        assert abs((lp_data_culled_df_temp.FULLAWP_ADJ.sum() - lp_data_culled_df.FULLAWP_ADJ.sum()) < .0001), \
            "abs((lp_data_culled_df_temp.FULLAWP_ADJ.sum() - lp_data_culled_df.FULLAWP_ADJ.sum()) < .0001)"
        assert len(lp_data_culled_df) == len(lp_data_culled_df_temp), "len(lp_data_culled_df) == len(lp_data_culled_df_temp)"
        lp_data_culled_df = lp_data_culled_df_temp
        
    else:
        lp_data_culled_df['GOODRX_UPPER_LIMIT'] = 10000*lp_data_culled_df['CURRENT_MAC_PRICE']
    
    return lp_data_culled_df

def add_rms_to_lp_data_culled_df(lp_data_culled_df, RMS_OPT=False, APPLY_GENERAL_MULTIPLIER=False,APPLY_MAIL_MULTIPLIER = False):

    if p.RMS_OPT or p.APPLY_GENERAL_MULTIPLIER:
        
        goodrx_df = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.GOODRX_FILE, dtype=p.VARIABLE_TYPE_DIC)
        group_cols = ['REGION','BREAKOUT','MEASUREMENT','GPI','BG_FLAG']
        
        lp_data_culled_df_temp = pd.merge(lp_data_culled_df, goodrx_df[group_cols + ['CHAIN_GROUP','GOODRX_CHAIN_PRICE']],\
                                          how = "left", on = group_cols + ['CHAIN_GROUP'])

        lp_data_culled_df_temp = pd.merge(lp_data_culled_df_temp, goodrx_df[group_cols + ['GOODRX_NONPREF_PRICE']].\
                                          drop_duplicates(), how = "left", on = group_cols)

        lp_data_culled_df_temp["DUMMY_GOODRX_PRICE"] = (10000 * lp_data_culled_df_temp["CURRENT_MAC_PRICE"])
        
        lp_data_culled_df_temp["GOODRX_UPPER_LIMIT"] = np.nan

        # Assign chain upper limits to Big 5 chains and IND upper limits to NONPREF_OTH pharmacies, all others get the dummy price
        chain_mask = lp_data_culled_df_temp["CHAIN_GROUP"].isin(["WMT", "WAG", "RAD", "KRG", "CVS"])
        lp_data_culled_df_temp.loc[chain_mask, "GOODRX_UPPER_LIMIT"] = lp_data_culled_df_temp.loc[chain_mask, ["GOODRX_CHAIN_PRICE", "DUMMY_GOODRX_PRICE"]].min(axis=1)
        lp_data_culled_df_temp.loc[~chain_mask, "GOODRX_UPPER_LIMIT"] = lp_data_culled_df_temp.loc[~chain_mask, ["GOODRX_NONPREF_PRICE", "DUMMY_GOODRX_PRICE"]].min(axis=1)

        # If matching all GoodRx pricing at R90 prices is desired, the below compares R30 and R90 GOODRX_UPPER_LIMITs for a GPI_NDC-CHAIN_GROUP and chooses the minimum.
        # Most useful when the GoodRx logic matches prices correctly with quantities near 30 and 90 and produces a lower upper limit at R90
        if "R90" in lp_data_culled_df_temp.MEASUREMENT.unique():
            r30mask =  lp_data_culled_df_temp["MEASUREMENT"] == "R30"
            r90mask =  (lp_data_culled_df_temp["MEASUREMENT"] == "R90")
            r90okmask = (lp_data_culled_df_temp["CHAIN_SUBGROUP"].str.contains("R90OK"))

            lp_data_culled_df_r90 = lp_data_culled_df_temp[r90mask & ~r90okmask].merge(lp_data_culled_df_temp.loc[r30mask,\
            ["GPI_NDC", "BG_FLAG","REGION", "CHAIN_GROUP", "CHAIN_SUBGROUP", "GOODRX_UPPER_LIMIT"]], how = 'left', on = ["GPI_NDC", "BG_FLAG","REGION", "CHAIN_GROUP", "CHAIN_SUBGROUP"],\
            suffixes = ('','_R30'))
            lp_data_culled_df_r30okmatch = lp_data_culled_df_temp.loc[r30mask].copy()
            lp_data_culled_df_r30okmatch['CHAIN_SUBGROUP'] = lp_data_culled_df_r30okmatch['CHAIN_SUBGROUP'] + '_R90OK'
            lp_data_culled_df_r90ok = lp_data_culled_df_temp[r90okmask].merge(lp_data_culled_df_r30okmatch.loc[:,\
            ["GPI_NDC", "BG_FLAG","REGION", "CHAIN_GROUP", "CHAIN_SUBGROUP", "GOODRX_UPPER_LIMIT"]], how = 'left', on = ["GPI_NDC", "BG_FLAG","REGION", "CHAIN_GROUP", "CHAIN_SUBGROUP"],\
            suffixes = ('','_R30'))
            lp_data_culled_df_r90 = pd.concat([lp_data_culled_df_r90, lp_data_culled_df_r90ok])

            if p.GOODRX_R90_MATCHING:
                lp_data_culled_df_r90['GOODRX_UPPER_LIMIT'] = lp_data_culled_df_r90[["GOODRX_UPPER_LIMIT", "GOODRX_UPPER_LIMIT_R30"]].min(axis =1)
            else:
                lp_data_culled_df_r90['GOODRX_UPPER_LIMIT'] = lp_data_culled_df_r90["GOODRX_UPPER_LIMIT_R30"]
            
            lp_data_culled_df_r90.drop(columns=['GOODRX_UPPER_LIMIT_R30'],inplace = True)

            lp_data_culled_df_temp_r30_r90 = lp_data_culled_df_temp[~r90mask].append(lp_data_culled_df_r90)
            
            assert len(lp_data_culled_df_temp_r30_r90) == len(lp_data_culled_df_temp), \
            "len(lp_data_culled_df_temp_r30_r90) == len(lp_data_culled_df_temp)"

            lp_data_culled_df_temp = lp_data_culled_df_temp_r30_r90
        
        # Assign the lowest of CVS GoodRx upper limit and all the pharmacies GoodRx upper limit to CVSSP chain subgroup if it exists
        if 'CVSSP' in lp_data_culled_df_temp.CHAIN_SUBGROUP.unique():
            lp_data_culled_df_temp['GOODRX_MIN_PRICE'] = lp_data_culled_df_temp.groupby(group_cols)['GOODRX_UPPER_LIMIT'].transform('min')
            cvssp_mask = lp_data_culled_df_temp["CHAIN_SUBGROUP"] == "CVSSP"
            lp_data_culled_df_temp.loc[cvssp_mask, "GOODRX_UPPER_LIMIT"] = lp_data_culled_df_temp.loc[cvssp_mask, ["GOODRX_UPPER_LIMIT", "GOODRX_MIN_PRICE"]].min(axis=1)

            # Assure CVS chain subgroup upper limit isn't above the parity price collar
            cvs_mask = lp_data_culled_df_temp["CHAIN_SUBGROUP"] == "CVS"

            lp_data_culled_df_temp.loc[cvs_mask, "GOODRX_COLLAR_HIGH_PRICE"] = lp_data_culled_df_temp.loc[cvs_mask, 
                                                                                                          "GOODRX_MIN_PRICE"] * p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH
            lp_data_culled_df_temp.loc[cvs_mask, "GOODRX_UPPER_LIMIT"] = lp_data_culled_df_temp.loc[cvs_mask, ["GOODRX_UPPER_LIMIT", 
                                                                                                        "GOODRX_COLLAR_HIGH_PRICE"]].min(axis=1) 
        else:
            lp_data_culled_df_temp['GOODRX_MIN_PRICE'] = lp_data_culled_df_temp.groupby(group_cols)['GOODRX_UPPER_LIMIT'].transform('min')
            cvs_mask = lp_data_culled_df_temp["CHAIN_GROUP"] == "CVS"
            lp_data_culled_df_temp.loc[cvs_mask, "GOODRX_UPPER_LIMIT"] = lp_data_culled_df_temp.loc[cvs_mask, ["GOODRX_UPPER_LIMIT", "GOODRX_MIN_PRICE"]].min(axis=1)
        
        # Assign the min GoodRx limit across all retail vcmls to Mail
        lp_data_culled_df_temp['GOODRX_MAIL_PRICE'] = lp_data_culled_df_temp.groupby(["REGION","GPI"])['GOODRX_UPPER_LIMIT'].transform('min')
        mail_mask = lp_data_culled_df_temp["MEASUREMENT"] == "M30"
        lp_data_culled_df_temp.loc[mail_mask, "GOODRX_UPPER_LIMIT"] = lp_data_culled_df_temp.loc[mail_mask, ["GOODRX_MAIL_PRICE", "DUMMY_GOODRX_PRICE"]].min(axis=1)

        #Costplus integration 
        if p.APPLY_MAIL_MULTIPLIER:
            costplus_df = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.COSTPLUS_FILE, dtype=p.VARIABLE_TYPE_DIC)
            lp_data_culled_df_temp = pd.merge(lp_data_culled_df_temp, costplus_df,how="left", on=['MEASUREMENT','GPI','BG_FLAG'])
            
            #Compare the Mail Ceiling (3x of Costplus) with the Min Goodrx price to get the final limit
            mccp_mask = (lp_data_culled_df_temp.MAIL_CEILING.notna()) & (lp_data_culled_df_temp.MAIL_CEILING < lp_data_culled_df_temp.GOODRX_UPPER_LIMIT)
            lp_data_culled_df_temp.loc[mccp_mask, "GOODRX_UPPER_LIMIT"] = lp_data_culled_df_temp.loc[mccp_mask, 'MAIL_CEILING']
            lp_data_culled_df_temp.drop(columns = ["MAIL_CEILING"])
        
        # VCML level pricing
        lp_data_culled_df_temp.rename(columns={"GOODRX_UPPER_LIMIT": "INDIVIDUAL_ROW_GOODRX_UPPER_LIMIT"}, inplace=True)
        
        lp_data_culled_df_by_vcml = lp_data_culled_df_temp.groupby(["GPI_NDC", "CLIENT", "BREAKOUT", "REGION", "MEASUREMENT", "BG_FLAG", "MAC_LIST"], as_index=False)["INDIVIDUAL_ROW_GOODRX_UPPER_LIMIT"].min()
        
        lp_data_culled_df_by_vcml.rename( columns={"INDIVIDUAL_ROW_GOODRX_UPPER_LIMIT": "GOODRX_UPPER_LIMIT"},inplace=True)
        
        lp_data_culled_df_temp = lp_data_culled_df_temp.merge(lp_data_culled_df_by_vcml, how="left", on=["GPI_NDC", "CLIENT", "BREAKOUT", "REGION", "MEASUREMENT", "BG_FLAG", "MAC_LIST"])

        lp_data_culled_df_temp.drop(columns=["GOODRX_CHAIN_PRICE", "GOODRX_NONPREF_PRICE","GOODRX_MIN_PRICE","GOODRX_MAIL_PRICE", "DUMMY_GOODRX_PRICE","INDIVIDUAL_ROW_GOODRX_UPPER_LIMIT"],inplace=True)
        if "GOODRX_COLLAR_HIGH_PRICE" in lp_data_culled_df_temp:
            lp_data_culled_df_temp.drop(columns=["GOODRX_COLLAR_HIGH_PRICE"],inplace=True)
            
        assert abs((lp_data_culled_df_temp.FULLAWP_ADJ.sum() - lp_data_culled_df.FULLAWP_ADJ.sum()) < 0.0001), \
        "abs((lp_data_culled_df_temp.FULLAWP_ADJ.sum() - lp_data_culled_df.FULLAWP_ADJ.sum()) < .0001)"
        
        assert len(lp_data_culled_df) == len(lp_data_culled_df_temp), \
        "len(lp_data_culled_df) == len(lp_data_culled_df_temp)"
        
        lp_data_culled_df = lp_data_culled_df_temp
        
    elif p.TRUECOST_CLIENT or p.APPLY_BENCHMARK_CAP:
        if p.TRUECOST_CLIENT and not p.APPLY_BENCHMARK_CAP:
            lp_data_culled_df.loc[lp_data_culled_df["BG_FLAG"]=='G',"GOODRX_UPPER_LIMIT"] = lp_data_culled_df.loc[lp_data_culled_df["BG_FLAG"]=='G',"NET_COST_GUARANTEE_UNIT"] * p.TRUECOST_UPPER_MULTIPLIER_GNRC
            lp_data_culled_df.loc[lp_data_culled_df["BG_FLAG"]=='B',"GOODRX_UPPER_LIMIT"] = lp_data_culled_df.loc[lp_data_culled_df["BG_FLAG"]=='B',"NET_COST_GUARANTEE_UNIT"] * p.TRUECOST_UPPER_MULTIPLIER_BRND
        if p.APPLY_BENCHMARK_CAP and not p.TRUECOST_CLIENT:
            lp_data_culled_df["GOODRX_UPPER_LIMIT"] = lp_data_culled_df["BENCHMARK_CEILING_PRICE"] * p.BENCHMARK_CAP_MULTIPLIER
        if p.TRUECOST_CLIENT and p.APPLY_BENCHMARK_CAP:
            # Use the minimum between net cost and benchmark cap ceilings
            mask = (((lp_data_culled_df["BG_FLAG"]=='G') & 
                    (lp_data_culled_df.BENCHMARK_CEILING_PRICE.notna()) & 
                    (lp_data_culled_df.BENCHMARK_CEILING_PRICE * p.BENCHMARK_CAP_MULTIPLIER <
                     lp_data_culled_df.NET_COST_GUARANTEE_UNIT * p.TRUECOST_UPPER_MULTIPLIER_GNRC)) |
                    ((lp_data_culled_df["BG_FLAG"]=='B') & 
                    (lp_data_culled_df.BENCHMARK_CEILING_PRICE.notna()) & 
                    (lp_data_culled_df.BENCHMARK_CEILING_PRICE * p.BENCHMARK_CAP_MULTIPLIER <
                     lp_data_culled_df.NET_COST_GUARANTEE_UNIT * p.TRUECOST_UPPER_MULTIPLIER_BRND)))
            
            lp_data_culled_df.loc[lp_data_culled_df["BG_FLAG"]=='G',"GOODRX_UPPER_LIMIT"] = lp_data_culled_df["NET_COST_GUARANTEE_UNIT"] * p.TRUECOST_UPPER_MULTIPLIER_GNRC
            lp_data_culled_df.loc[lp_data_culled_df["BG_FLAG"]=='B',"GOODRX_UPPER_LIMIT"] = lp_data_culled_df["NET_COST_GUARANTEE_UNIT"] * p.TRUECOST_UPPER_MULTIPLIER_BRND
            lp_data_culled_df.loc[mask, "GOODRX_UPPER_LIMIT"] = lp_data_culled_df["BENCHMARK_CEILING_PRICE"] * p.BENCHMARK_CAP_MULTIPLIER
            
        # Prevent NULL rows from populating as 0 later in the code
        lp_data_culled_df.loc[(lp_data_culled_df.GOODRX_UPPER_LIMIT.isna()), 
                              "GOODRX_UPPER_LIMIT"] = (10000 * lp_data_culled_df["CURRENT_MAC_PRICE"])

    elif p.APPLY_MAIL_MULTIPLIER:
        costplus_df = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.COSTPLUS_FILE, dtype=p.VARIABLE_TYPE_DIC)
        lp_data_culled_df_temp = pd.merge(lp_data_culled_df, costplus_df, how="left", on=['MEASUREMENT','GPI','BG_FLAG'])
        lp_data_culled_df_temp["GOODRX_UPPER_LIMIT"] = (10000 * lp_data_culled_df_temp["CURRENT_MAC_PRICE"])

        #Compare the Mail Ceiling (3x of Costplus) with the Min Goodrx price to get the final limit
        mccp_mask = (lp_data_culled_df_temp.MAIL_CEILING.notna()) & (lp_data_culled_df_temp.MAIL_CEILING < lp_data_culled_df_temp.GOODRX_UPPER_LIMIT)
        lp_data_culled_df_temp.loc[mccp_mask, "GOODRX_UPPER_LIMIT"] = lp_data_culled_df_temp.loc[mccp_mask, 'MAIL_CEILING']
        lp_data_culled_df_temp.drop(columns = ["MAIL_CEILING"])
        lp_data_culled_df = lp_data_culled_df_temp
        
    else:
        lp_data_culled_df["GOODRX_UPPER_LIMIT"] = (10000 * lp_data_culled_df["CURRENT_MAC_PRICE"])

    return lp_data_culled_df

def add_pharma_list(lp_data_culled_df, pref_pharm_list, ignore_columns = False):
    lp_data_culled_df = lp_data_culled_df.merge(
        pref_pharm_list[['CLIENT', 'BREAKOUT', 'REGION', 'PREF_PHARMS']], 
        how = 'left', on = ['CLIENT', 'BREAKOUT', 'REGION'])
    
    lp_data_culled_df['PHARMACY_TYPE']='Non_Preferred'
    
    client_list=lp_data_culled_df['CLIENT'].unique()
    for client in client_list:
        breakout_list = lp_data_culled_df.loc[(lp_data_culled_df['CLIENT'] == client),'BREAKOUT'].unique()
        for breakout in breakout_list:
            region_list = lp_data_culled_df.loc[(lp_data_culled_df['CLIENT'] == client) & (lp_data_culled_df['BREAKOUT'] == breakout),'REGION'].unique()
            for region in region_list:
                bg_list = lp_data_culled_df.loc[(lp_data_culled_df.CLIENT==client) & (lp_data_culled_df.BREAKOUT==breakout) & (lp_data_culled_df.REGION==region),'BG_FLAG'].unique()
                for bg in bg_list:
                    pref_pharmacies=pref_pharm_list.loc[(pref_pharm_list.CLIENT==client) 
                                                        & (pref_pharm_list.BREAKOUT==breakout) 
                                                        & (pref_pharm_list.REGION==region),'PREF_PHARMS']
                    lp_data_culled_df.loc[(lp_data_culled_df.CLIENT==client) 
                                          & (lp_data_culled_df.BREAKOUT==breakout) 
                                          & (lp_data_culled_df.REGION==region) 
                                          & (lp_data_culled_df.BG_FLAG==bg)
                                          & (lp_data_culled_df.CHAIN_GROUP.isin(pref_pharmacies.values[0])),'PHARMACY_TYPE']='Preferred'

    lp_data_culled_df.drop(columns = ['PREF_PHARMS'], inplace = True)
    
    if ignore_columns == False:
        lp_data_culled_df['PRICE_MUTABLE'] = 1
        lp_data_culled_df.loc[lp_data_culled_df.CURRENT_MAC_PRICE==0, 'PRICE_MUTABLE'] = 0
        lp_data_culled_df.loc[lp_data_culled_df.CURRENT_MAC_PRICE==0, 'IMMUTABLE_REASON'] = 'NON_MAC_DRUG'
        lp_data_culled_df.UC_UNIT25 = lp_data_culled_df.UC_UNIT25 * .8
    
    return lp_data_culled_df

def add_non_mac_rate(lp_data_culled_df):
    
    non_mac_rate_data = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.NON_MAC_RATE_FILE, dtype = p.VARIABLE_TYPE_DIC)
    non_mac_rate_data = standardize_df(non_mac_rate_data)
    
    non_mac_rate_data['MAC_LIST'] = non_mac_rate_data['VCML_ID'].str[3:]
    non_mac_rate_data.drop(columns=['VCML_ID'], inplace=True)
    
    lp_data_culled_df_temp = lp_data_culled_df.merge(non_mac_rate_data, on=['MAC_LIST','BG_FLAG'], how='left')
    assert abs((lp_data_culled_df_temp.FULLAWP_ADJ.sum() - lp_data_culled_df.FULLAWP_ADJ.sum()) < 0.0001), "error in non mac rate merge"
    
    lp_data_culled_df_temp.loc[(lp_data_culled_df_temp.BG_FLAG == 'B')
                               & (lp_data_culled_df_temp['NON_MAC_RATE'].isna()), 'NON_MAC_RATE'] = p.BRAND_NON_MAC_RATE
    lp_data_culled_df_temp.loc[(lp_data_culled_df_temp.MEASUREMENT.str[0]=='M')
                               & (lp_data_culled_df_temp['NON_MAC_RATE'].isna()), 'NON_MAC_RATE'] = p.MAIL_NON_MAC_RATE
    lp_data_culled_df_temp.loc[lp_data_culled_df_temp['NON_MAC_RATE'].isna(), 'NON_MAC_RATE'] = p.RETAIL_NON_MAC_RATE
    
    return lp_data_culled_df_temp

def add_zbd_to_lp_data_culled_df(lp_data_culled_df, ZBD_OPT = False, cvs_ind_scalar = 1.2, capped_scalar = 3.6, current_price_scalar = 0.5, by_vcml=True):
    '''
    Calculate the new drug price based on the welcome season 2023 ZBD logic. 
    Be aware this is an ad-hoc fix for the problem in the beginning of the year for quick implementation to capture value asap. 
    Future dev work for better ZBD optimization logic are needed for us to capture the full value of ZBD instead of limiting our scope on those top 200 leakage drug
    identified.
    
    BG info:
    ZBD is the abbrev for zero-balance-due, which are the drugs that client pays zero dollars and members pays the full drug price. 
    This is usually due to the fact that pharmacy drug price is lower than member's copay/ or hdhp members are still in their deductible phase.
    Problem of leakage happens when member pays in IND and their cost share is greater than the full pharmacy drug price. 
    Because we do not have contract with IND, we cannot collect the differences between member's pay and pharmacy drug price, which would then be considered as a
    leakage.
    We first identified this prob through ZBD claims thus named it as ZBD_OPT
    However, the issue would exist as long as member pays more than IND pharm price, regardless of whether a claim is ZBD or not.
    
    Logic:
    This function allows us to set a "ZBD_UPPER_LIMIT" for the drugs that are identified as the top leakage drugs. 
    If the drug is dispensed in CVS/IND and non capped pharmacy, then the ZBD_UPPER_LIMIT is set to be the max between 1.2 * MAC1026, or 0.5 * current_price (both 1.2
    and 0.5 are scalars that can be modified through ZBD_CVS_IND_SCALAR and ZBD_CURRENT_PRICE_SCALAR parameters, respectively)
    If the drug is dispensed in capped pharmacy (both big and small), then the ZBD_UPPER_LIMIT is set to be 3 * ZBD_UPPER_LIMIT in CVS/IND to keep the price relativity
    '''
    lp_data_culled_df_temp = lp_data_culled_df.copy()
    if p.LOCKED_CLIENT and p.ZBD_OPT:
        gpi_list = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + 'LEAKAGE_GPI_DF_{}.csv'.format(p.DATA_ID), dtype=p.VARIABLE_TYPE_DIC)['GPI'].tolist()
        lp_data_culled_df_temp['ZBD_UPPER_LIMIT'] = (10000 * lp_data_culled_df_temp["CURRENT_MAC_PRICE"].copy())
        
        #Create column to capture zbd upper limit if we decided to set it according to current_mac_price. call the column ZBD_CURRENT_PRICE_LIMIT
        #The current logic use current_price_scalar = 0.5 to set the limit to half of the current price, but can change scalar to reflect a bigger/smaller price drop
        lp_data_culled_df_temp['ZBD_CURRENT_PRICE_LIMIT'] = ( current_price_scalar * lp_data_culled_df_temp["CURRENT_MAC_PRICE"].copy())
        
        ########################################################################################################################################################
        ##Group pharmacy into different list and apply seperate ZBD_UPPER_LIMIT logic
        ##CVS/IND list: this list should include CVS, IND as well as all other non capped pharmacies
        ##    -for the pharmacies in this list, ZBD_UPPER_LIMIT will be set to be the max between scalar limit or current_price_limit
        ##    -notice if a client is SP client, later on only CVSSP will be assign the same ZBD upper limit as IND, 
        ##    and will apply PARITY_PRICE_DIFFERENCE_COLLAR_HIGH * ZBD_UPPER_LIMIT in CVSSP/IND to assign CVS price in non-sp state
        ##Capped pharm list: this list should inlcude all small capped pharmacies plus big capped pharmacies except CVS
        ##    for the pharmacies in this list, ZBD_UPPER_LIMIT will be set to be 3x of the ZBD_UPPER_LIMIT in CVS/IND
        ##Notice: PSAO_TREATMENT param will impact the list assignment here.
        ##    When PSAO_TREATMENT = True (the defalt value), PSAO will be append to the NON_CAPPED_PHARMACY_LIST; else PSAO appends to SMALL_CAPPED_PHARMACY_LIST
        #########################################################################################################################################################
        #Create CVS/IND list.
        gnrc_cvs_ind_list = ['CVS'] + p.NON_CAPPED_PHARMACY_LIST['GNRC']
        brnd_cvs_ind_list = ['CVS'] + p.NON_CAPPED_PHARMACY_LIST['BRND']

        #Create Capped pharm list
        gnrc_big4_list = p.BIG_CAPPED_PHARMACY_LIST['GNRC'].copy()
        brnd_big4_list = p.BIG_CAPPED_PHARMACY_LIST['BRND'].copy()
        gnrc_big4_list.remove('CVS')
        brnd_big4_list.remove('CVS')
        gnrc_capped_list = gnrc_big4_list + p.SMALL_CAPPED_PHARMACY_LIST['GNRC']        
        brnd_capped_list = brnd_big4_list + p.SMALL_CAPPED_PHARMACY_LIST['BRND']        
        
        #Create column to capture zbd upper limit if we decide to set it according to mac1026. call the column ZBD_SCALAR_LIMIT
        #current logic would set cvs_ind upper limit to 1.2*mac1026, capped upper limit to 3.6*mac1026, but we could change scalar to reflect a different price bound
        cvs_ind_mask = (((lp_data_culled_df_temp['CHAIN_GROUP'].isin(gnrc_cvs_ind_list))
                             & (lp_data_culled_df_temp['GPI'].isin(gpi_list))
                             & (lp_data_culled_df_temp['GPI_ONLY'] == 1)
                             & (lp_data_culled_df_temp['BG_FLAG'] == 'G')
                            ) |
                        ((lp_data_culled_df_temp['CHAIN_GROUP'].isin(brnd_cvs_ind_list))
                             & (lp_data_culled_df_temp['GPI'].isin(gpi_list))
                             & (lp_data_culled_df_temp['GPI_ONLY'] == 1)
                             & (lp_data_culled_df_temp['BG_FLAG'] == 'B')
                            ))
        capped_mask = (((lp_data_culled_df_temp['CHAIN_SUBGROUP'].isin(gnrc_capped_list))
                          & (lp_data_culled_df_temp['GPI'].isin(gpi_list))
                          & (lp_data_culled_df_temp['GPI_ONLY'] == 1)
                          & (lp_data_culled_df_temp['BG_FLAG'] == 'G')
                         ) |
                       ((lp_data_culled_df_temp['CHAIN_SUBGROUP'].isin(brnd_capped_list))
                          & (lp_data_culled_df_temp['GPI'].isin(gpi_list))
                          & (lp_data_culled_df_temp['GPI_ONLY'] == 1)
                          & (lp_data_culled_df_temp['BG_FLAG'] == 'B')
                         ))
        
        lp_data_culled_df_temp.loc[cvs_ind_mask, 'ZBD_SCALAR_LIMIT'] = lp_data_culled_df_temp['MAC1026_UNIT_PRICE'][cvs_ind_mask] * cvs_ind_scalar
        lp_data_culled_df_temp.loc[capped_mask, 'ZBD_SCALAR_LIMIT'] = lp_data_culled_df_temp['MAC1026_UNIT_PRICE'][capped_mask] * capped_scalar
        
        ############################################################################################################################
        ##OPTION 1: relaxing the upper limit on all drugs
        #############################################################################################################################
        #Determine ZBD_UPPER_LIMIT in CVS/IND by choosing the max between the scalar limit or current_price_limit
        lp_data_culled_df_temp.loc[cvs_ind_mask, 'ZBD_UPPER_LIMIT'] = lp_data_culled_df_temp.loc[cvs_ind_mask, ['ZBD_SCALAR_LIMIT', 'ZBD_CURRENT_PRICE_LIMIT']].max(axis = 1)
        
        #Determine ZBD_UPPER_LIMIT for CVS in non parity state if the client is a state parity client
        #For CVS in non parity state, ZBD_UPPER_LIMIT will be PARITY_PRICE_DIFFERENCE_COLLAR_HIGH * ZBD_UPPER_LIMIT in CVSSP/IND, so to keep price relativity between cvs&cvssp
        cvs_sp_check = lp_data_culled_df_temp['CHAIN_SUBGROUP'].unique()
        if 'CVSSP' in cvs_sp_check:
            cvs_sp = ((lp_data_culled_df_temp['CHAIN_SUBGROUP'] == 'CVSSP')
                             & (lp_data_culled_df_temp['GPI'].isin(gpi_list))
                             & (lp_data_culled_df_temp['GPI_ONLY'] == 1)
                            )
            cvs_nonsp = ((lp_data_culled_df_temp['CHAIN_SUBGROUP'] == 'CVS')
                             & (lp_data_culled_df_temp['GPI'].isin(gpi_list))
                             & (lp_data_culled_df_temp['GPI_ONLY'] == 1)
                            )
            #Get the min ZBD_UPPER_LIMIT in cvssp
            lp_data_culled_df_cvssp= lp_data_culled_df_temp[cvs_sp].groupby(["GPI_NDC", "CLIENT", "REGION", "BG_FLAG"], as_index=False)['ZBD_UPPER_LIMIT'].min()
            lp_data_culled_df_cvssp.rename(columns={"ZBD_UPPER_LIMIT":"ZBD_UPPER_LIMIT_CVSSP"}, inplace = True)
            lp_data_culled_df_temp = lp_data_culled_df_temp.merge(lp_data_culled_df_cvssp, how = "left", on = ["GPI_NDC", "CLIENT", "REGION", "BG_FLAG"])
            
            #Apply PARITY_PRICE_DIFFERENCE_COLLAR_HIGH to get ZBD_CVSNONSP_LIMIT, which will furtuer be assigned as ZBD_UPPER_LIMIT for CVS in non parity states
            lp_data_culled_df_temp.loc[cvs_nonsp & (lp_data_culled_df_temp['ZBD_UPPER_LIMIT_CVSSP'].notnull()), 'ZBD_CVSNONSP_LIMIT'] = lp_data_culled_df_temp[cvs_nonsp & (lp_data_culled_df_temp['ZBD_UPPER_LIMIT_CVSSP'].notnull())]['ZBD_UPPER_LIMIT_CVSSP'] * p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH
            lp_data_culled_df_temp.loc[cvs_nonsp & (lp_data_culled_df_temp['ZBD_UPPER_LIMIT_CVSSP'].notnull()), 'ZBD_UPPER_LIMIT'] = lp_data_culled_df_temp[cvs_nonsp & (lp_data_culled_df_temp['ZBD_UPPER_LIMIT_CVSSP'].notnull())]['ZBD_CVSNONSP_LIMIT']
            
            assert abs((lp_data_culled_df_temp.FULLAWP_ADJ.sum() - lp_data_culled_df.FULLAWP_ADJ.sum()) < 0.0001), \
            "abs((lp_data_culled_df_temp.FULLAWP_ADJ.sum() - lp_data_culled_df.FULLAWP_ADJ.sum()) < .0001)"
            
            assert len(lp_data_culled_df) == len(lp_data_culled_df_temp), \
            "len(lp_data_culled_df) == len(lp_data_culled_df_temp)" 
        
        #Determine ZBD_UPPER_LIMIT in capped pharmacy by setting it to 3x the min of ZBD_UPPER_LIMIT in CVS/IND
        lp_data_culled_df_cvsind= lp_data_culled_df_temp[cvs_ind_mask].groupby(["GPI_NDC", "CLIENT", "REGION", "BG_FLAG"], as_index=False)['ZBD_UPPER_LIMIT'].min()
        lp_data_culled_df_cvsind.rename(columns={"ZBD_UPPER_LIMIT":"ZBD_UPPER_LIMIT_CVSIND"}, inplace = True)
        lp_data_culled_df_temp = lp_data_culled_df_temp.merge(lp_data_culled_df_cvsind, how = "left", on = ["GPI_NDC", "CLIENT", "REGION", "BG_FLAG"])
        lp_data_culled_df_temp.loc[capped_mask, 'ZBD_CAPPED_3X_LIMIT'] = lp_data_culled_df_temp['ZBD_UPPER_LIMIT_CVSIND'][capped_mask] * 3
        lp_data_culled_df_temp.loc[capped_mask, 'ZBD_UPPER_LIMIT'] = lp_data_culled_df_temp['ZBD_CAPPED_3X_LIMIT'][capped_mask]
        #lp_data_culled_df_temp.loc[capped_mask, 'ZBD_UPPER_LIMIT'] = lp_data_culled_df_temp.loc[capped_mask, ['ZBD_CAPPED_3X_LIMIT', 'ZBD_CURRENT_PRICE_LIMIT']].max(axis = 1)
        
        #GOODRX vs. ZBD parity -- choose if GOODRX_UPPER_LIMIT is smaller than ZBD_UPPER_LIMIT, then use GOODRX_UPPER_LIMIT as ZBD_UPPER_LIMIT
        # lp_data_culled_df_temp['ZBD_UPPER_LIMIT'] = lp_data_culled_df_temp[['ZBD_UPPER_LIMIT','GOODRX_UPPER_LIMIT']].min(axis=1)          
                       
        # Assign the min ZBD limit across all retail vcmls to Mail
        lp_data_culled_df_temp['ZBD_MAIL_PRICE'] = lp_data_culled_df_temp.groupby(["REGION","GPI", "BG_FLAG"])['ZBD_UPPER_LIMIT'].transform('min')
        mail_mask = lp_data_culled_df_temp["MEASUREMENT"].str.contains('M') #Wildcard match to catch both M30 and M90
        lp_data_culled_df_temp.loc[mail_mask, "ZBD_UPPER_LIMIT"] = lp_data_culled_df_temp.loc[mail_mask, ["ZBD_MAIL_PRICE", "ZBD_UPPER_LIMIT"]].min(axis=1) 
        
        #MAIL vs. Retail -- Costplus integration
        costplus_df = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.COSTPLUS_FILE, dtype=p.VARIABLE_TYPE_DIC)
        costplus_df.rename(columns={"MAIL_CEILING": "MAIL_CEILING_ZBD"}, inplace=True)
        lp_data_culled_df_temp = pd.merge(lp_data_culled_df_temp, costplus_df, how="left", on=['MEASUREMENT','GPI','BG_FLAG'])
        
        #Compare the Mail Ceiling (3x of Costplus) with the Min RMS price to get the final limit
        mccp_mask = (lp_data_culled_df_temp.MAIL_CEILING_ZBD.notna()) & (lp_data_culled_df_temp.MAIL_CEILING_ZBD < lp_data_culled_df_temp.ZBD_UPPER_LIMIT)
        lp_data_culled_df_temp.loc[mccp_mask, "ZBD_UPPER_LIMIT"] = lp_data_culled_df_temp.loc[mccp_mask, 'MAIL_CEILING_ZBD']
        lp_data_culled_df_temp.drop(columns = ["MAIL_CEILING_ZBD"], inplace = True)
        
        assert abs((lp_data_culled_df_temp.FULLAWP_ADJ.sum() - lp_data_culled_df.FULLAWP_ADJ.sum()) < 0.0001), \
        "abs((lp_data_culled_df_temp.FULLAWP_ADJ.sum() - lp_data_culled_df.FULLAWP_ADJ.sum()) < .0001)"
            
        assert len(lp_data_culled_df) == len(lp_data_culled_df_temp), \
        "len(lp_data_culled_df) == len(lp_data_culled_df_temp)"
        
        #VCML level pricing1 - finalize r30 price
        if by_vcml:
            lp_data_culled_df_temp.rename(columns={"ZBD_UPPER_LIMIT": "INDIVIDUAL_ROW_ZBD_UPPER_LIMIT"}, inplace=True)
            lp_data_culled_df_by_vcml = lp_data_culled_df_temp.groupby(["GPI_NDC", "CLIENT", "BREAKOUT", "REGION", "MEASUREMENT", "MAC_LIST", "BG_FLAG"], as_index=False)["INDIVIDUAL_ROW_ZBD_UPPER_LIMIT"].min()
            lp_data_culled_df_by_vcml.rename(columns={"INDIVIDUAL_ROW_ZBD_UPPER_LIMIT": "ZBD_UPPER_LIMIT"},inplace=True)
            lp_data_culled_df_temp = lp_data_culled_df_temp.merge(lp_data_culled_df_by_vcml, how = "left", on = ["GPI_NDC", "CLIENT", "BREAKOUT", "REGION", "MEASUREMENT", "MAC_LIST", "BG_FLAG"])
            lp_data_culled_df_temp.drop(columns = ["INDIVIDUAL_ROW_ZBD_UPPER_LIMIT"], inplace = True)
            
            assert abs((lp_data_culled_df_temp.FULLAWP_ADJ.sum() - lp_data_culled_df.FULLAWP_ADJ.sum()) < 0.0001), \
            "abs((lp_data_culled_df_temp.FULLAWP_ADJ.sum() - lp_data_culled_df.FULLAWP_ADJ.sum()) < .0001)"
            
            assert len(lp_data_culled_df) == len(lp_data_culled_df_temp), \
            "len(lp_data_culled_df) == len(lp_data_culled_df_temp)"                               
                
        # Assign the min r30 ZBD limit to R90
        lp_data_culled_df_temp['ZBD_R90_PRICE'] = lp_data_culled_df_temp.groupby(["GPI_NDC","REGION", "CHAIN_SUBGROUP", "BG_FLAG"])['ZBD_UPPER_LIMIT'].transform('min')
        R90_mask = lp_data_culled_df_temp["MEASUREMENT"] == "R90"
        lp_data_culled_df_temp.loc[R90_mask, "ZBD_UPPER_LIMIT"] = lp_data_culled_df_temp.loc[R90_mask, ["ZBD_R90_PRICE", "ZBD_UPPER_LIMIT"]].min(axis=1)
        # assign the min r30 ZBD limit to R90OK
        lp_data_culled_df_temp['ZBD_R90OK_PRICE'] = lp_data_culled_df_temp.groupby(["GPI_NDC","REGION", "CHAIN_GROUP", "BG_FLAG"])['ZBD_UPPER_LIMIT'].transform('min')
        R90ok_mask = lp_data_culled_df_temp["CHAIN_SUBGROUP"].str.contains('R90OK')
        if R90ok_mask.any():
            lp_data_culled_df_temp.loc[R90ok_mask, "ZBD_UPPER_LIMIT"] = lp_data_culled_df_temp.loc[R90ok_mask, ["ZBD_R90OK_PRICE", "ZBD_UPPER_LIMIT"]].min(axis=1)
        
       #VCML level pricing -- finalize r90 price
        if by_vcml:
            lp_data_culled_df_temp.rename(columns={"ZBD_UPPER_LIMIT": "INDIVIDUAL_ROW_ZBD_UPPER_LIMIT"}, inplace=True)
            lp_data_culled_df_by_vcml = lp_data_culled_df_temp.groupby(["GPI_NDC", "CLIENT", "BREAKOUT", "REGION", "MEASUREMENT", "MAC_LIST", "BG_FLAG"], as_index=False)["INDIVIDUAL_ROW_ZBD_UPPER_LIMIT"].min()
            lp_data_culled_df_by_vcml.rename(columns={"INDIVIDUAL_ROW_ZBD_UPPER_LIMIT": "ZBD_UPPER_LIMIT"},inplace=True)
            lp_data_culled_df_temp = lp_data_culled_df_temp.merge(lp_data_culled_df_by_vcml, how = "left", on = ["GPI_NDC", "CLIENT", "BREAKOUT", "REGION", "MEASUREMENT", "MAC_LIST", "BG_FLAG"])
            lp_data_culled_df_temp.drop(columns = ["INDIVIDUAL_ROW_ZBD_UPPER_LIMIT"], inplace = True)
            
            assert abs((lp_data_culled_df_temp.FULLAWP_ADJ.sum() - lp_data_culled_df.FULLAWP_ADJ.sum()) < 0.0001), \
            "abs((lp_data_culled_df_temp.FULLAWP_ADJ.sum() - lp_data_culled_df.FULLAWP_ADJ.sum()) < .0001)"
            
            assert len(lp_data_culled_df) == len(lp_data_culled_df_temp), \
            "len(lp_data_culled_df) == len(lp_data_culled_df_temp)"                  
        
        
        lp_data_culled_df = lp_data_culled_df_temp   
    
    else:
        lp_data_culled_df['ZBD_UPPER_LIMIT'] = (10000 * lp_data_culled_df["CURRENT_MAC_PRICE"])
        if p.LOCKED_CLIENT == False and p.ZBD_OPT == True:
            print('WARNING: Trying to do ZBD optimization on transparent client, dummy ZBD_UPPER_LIMIT was used. ZBD optimization should only be applied to locked clients')
                  
    return lp_data_culled_df



##    ADD RULE BASED LOGIC HERE      ##
def add_rule_base_logic(
        lp_data_culled_df, unc_percentiles, exclusions, unc_exclusions,
        client_guarantees, pharmacy_guarantees, pref_pharm_list, contract_date_df):
    
    #add client guarantees
    lp_data_culled_df = pd.merge(
        lp_data_culled_df, client_guarantees, how ='left', 
        on = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'BG_FLAG', 'PHARMACY_TYPE'])
    
    # add in UNC percentiles
    if p.UNC_OPT:
        lp_data_culled_df = pd.merge(
            lp_data_culled_df, unc_percentiles, how = 'left',
            on = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'BG_FLAG',
                  'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'PHARMACY_TYPE', 'GPI_NDC', 'GPI', 'NDC'])
    # make excluded GPIs immutable
    # decided for now the immutable is being assigned at GPI level, regardless of B/G
    # further investigation are needed to understand 
    if p.HANDLE_INFEASIBLE:
        infeasible_exclusions = standardize_df(pd.read_csv(os.path.join(p.FILE_INPUT_PATH, p.INFEASIBLE_EXCLUSION_FILE), dtype = p.VARIABLE_TYPE_DIC))
        exclusions = pd.concat([exclusions, unc_exclusions, infeasible_exclusions])
        exclusions = exclusions.drop_duplicates(
            subset=['CLIENT', 'REGION','GPI'], keep='first')
    else:
        exclusions = pd.concat([exclusions, unc_exclusions])
        exclusions = exclusions.drop_duplicates(
            subset=['CLIENT', 'REGION','GPI'], keep='first')
    for client in exclusions.CLIENT.unique():
        if client == 'ALL':
            gpis_to_exclude = exclusions.loc[exclusions.CLIENT == client, 'GPI'].values
            lp_data_culled_df.loc[
                lp_data_culled_df.GPI.isin(gpis_to_exclude), 'PRICE_MUTABLE'] = 0
        else:
            for region in exclusions.loc[exclusions.CLIENT == client, 'REGION'].unique():
                if region == 'ALL':
                    gpis_to_exclude = exclusions.loc[
                        (exclusions.CLIENT == client) & 
                        (exclusions.REGION == 'ALL'), 'GPI'].values
                    lp_data_culled_df.loc[
                        (lp_data_culled_df.CLIENT == client) & 
                        (lp_data_culled_df.GPI.isin(gpis_to_exclude)), 
                        'PRICE_MUTABLE'] = 0
                else:
                    gpis_to_exclude = exclusions.loc[
                        (exclusions.CLIENT == client) & 
                        (exclusions.REGION == region), 'GPI'].values
                    lp_data_culled_df.loc[(lp_data_culled_df.CLIENT == client) &
                                          (lp_data_culled_df.REGION == region) &
                                          (lp_data_culled_df.GPI.isin(gpis_to_exclude)), 
                                          'PRICE_MUTABLE'] = 0
    
    # make optimization MAC price adjustments, either U&C or GoodRx
    # Spoke to Melanie who said UNC opt never makes MAIL price changes, which is hard coded in so only retail non-MAC rate is needed for awp_discount_percent
    if p.UNC_OPT:
        if (p.CLIENT_TYPE == 'COMMERCIAL' or p.CLIENT_TYPE == 'MEDICAID'):
            lp_data_culled_df = unc_optimization(
                lp_data_culled_df, awp_discount_percent = (1 - p.RETAIL_NON_MAC_RATE),
                pharmacy_chain_groups = p.UNC_PHARMACY_CHAIN_GROUPS,
                pref_pharm_list = pref_pharm_list,
                contract_date_df = contract_date_df
            )
        if p.CLIENT_TYPE == 'MEDD':
            lp_data_culled_df = unc_optimization(
                lp_data_culled_df, awp_discount_percent = (1 - p.RETAIL_NON_MAC_RATE),
                pharmacy_chain_groups = p.UNC_PHARMACY_CHAIN_GROUPS,
                pref_pharm_list = pref_pharm_list,
                contract_date_df = contract_date_df)
    
    return lp_data_culled_df

def add_copay_coins_data(lp_data_culled_df):
    merge_col = ['CLIENT','REGION','BREAKOUT','MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP','CHAIN_SUBGROUP','GPI','NDC']
    copay_coins_col = ['COPAY_ONLY_QTY_WT','COINS_ONLY_QTY_WT','COPAY_COINS_QTY_WT','NO_COPAYCOINS_QTY_WT',
                       'COPAY_ONLY_AWP_WT','COINS_ONLY_AWP_WT','COPAY_COINS_AWP_WT','NO_COPAYCOINS_AWP_WT', 
                       'AVG_COPAY_UNIT','MIN_COPAY_UNIT','MAX_COPAY_UNIT','MEDIAN_COPAY_UNIT',
                       'AVG_COINS','MIN_COINS','MAX_COINS','MEDIAN_COINS',
                       'AVG_COMPLEX_COPAY_UNIT','MIN_COMPLEX_COPAY_UNIT','MAX_COMPLEX_COPAY_UNIT','MEDIAN_COMPLEX_COPAY_UNIT',
                       'AVG_COMPLEX_COINS','MIN_COMPLEX_COINS','MAX_COMPLEX_COINS','MEDIAN_COMPLEX_COINS']
    all_col = merge_col + copay_coins_col
        
    # Get copay/coinsurance data
    copay_coins_gpi = read_copay_coinsurance(READ_FROM_BQ = p.READ_FROM_BQ, level = 'gpi')
    copay_coins_gpi = standardize_df(copay_coins_gpi)
        
    copay_coins_ndc = read_copay_coinsurance(READ_FROM_BQ = p.READ_FROM_BQ, level = 'ndc')
    copay_coins_ndc = standardize_df(copay_coins_ndc)
        
    # Bring in GPI level copay/coins for GPI_ONLY drugs
    lp_data_culled_df_temp = lp_data_culled_df.copy()
    lp_data_culled_df_gpi = lp_data_culled_df_temp.loc[lp_data_culled_df['GPI_ONLY'] == 1]
    lp_data_culled_df_gpi = pd.merge(lp_data_culled_df_gpi, copay_coins_gpi[all_col], on = merge_col, how = 'left')

    # Bring in GPI level copay/coins for GPI_ONLY drugs
    lp_data_culled_df_ndc = lp_data_culled_df_temp.loc[lp_data_culled_df['GPI_ONLY'] == 0]
    lp_data_culled_df_ndc = pd.merge(lp_data_culled_df_ndc, copay_coins_ndc[all_col], on = merge_col, how = 'left')
        
    lp_data_culled_df_temp = pd.concat([lp_data_culled_df_gpi, lp_data_culled_df_ndc], axis=0, ignore_index=True)
    assert len(lp_data_culled_df_temp) == len(lp_data_culled_df), "Number of rows are not the same after adding in copay/coins data"
        
    lp_data_culled_df = lp_data_culled_df_temp
    
    return lp_data_culled_df


def merge_nadac_wac_netcostguarantee(lp_data_culled_df):
    '''
    The purpose of this function is load nadac_wac 
    Then merge with lp_data_culled_df
    If the parameter TRUECOST_CLIENT is set to True then also load net_cost_guarantees table
    Then merge with lp_data_culled_df
    This is done to set soft and hard price contraints further in the code
    
    input: lp_data_culled_df
    output: lp_data_culled_df with added colume called BG_FLAG, NADAC, WAC, NET_COST_GUARANTEE_UNIT (if TRUECOST clinet)
    '''    
    nadac_wac = read_nadac_wac(p.READ_FROM_BQ)
    nadac_wac_price = nadac_wac.copy()
    nadac_wac_price['GPI_NDC'] = nadac_wac_price['GPI'] + '_' + nadac_wac_price['NDC']
    nadac_wac_price.rename(columns = {'CUSTOMER_ID':'CLIENT'},inplace=True) 
    nadac_wac_price = nadac_wac_price.drop(['GPI','NDC'], axis=1).drop_duplicates()
    
    assert len(nadac_wac_price) == len(nadac_wac), "len(nadac_wac_price) == len(nadac_wac), Check for duplicaitons in the original nadac,wac dataset"
    
    #Prepare and merge nadac/wac price with lp data
    merge_col = ['CLIENT', 'GPI_NDC', 'BG_FLAG']
    lp_data_culled_df_temp = lp_data_culled_df.copy()
    lp_data_culled_df_temp = pd.merge(lp_data_culled_df_temp, nadac_wac_price, on = merge_col, how = 'left')
    
    assert len(lp_data_culled_df_temp) == len(lp_data_culled_df), "len(lp_data_culled_df_temp) == len(lp_data_culled_df), check for duplication after merge"
    
    if p.TRUECOST_CLIENT:
        net_cost_guarantee = read_net_cost_guarantee()
        assert net_cost_guarantee['NET_COST_GUARANTEE_UNIT'].notna().all(), "Error: net cost guarantees contain NA values"
        
        gpi_awp_df = uf.read_BQ_data(
                                    project_id=p.BQ_INPUT_PROJECT_ID,
                                    dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                                    table_id='awp_history_table',
                                    query=BQ.gpi_awp_table_custom.format(
                                        _project_id=p.BQ_INPUT_PROJECT_ID,
                                        _dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                                        _table_id='awp_history_table',
                                    ),
                                    custom=True)

        lp_data_culled_df_temp = pd.merge(lp_data_culled_df_temp, gpi_awp_df, on = ['GPI','BG_FLAG'], how = 'left')
        
        #Prepare and merge net guarantee price with lp data  
        merge_col = ['CLIENT', 'GPI', 'BG_FLAG']
        lp_data_culled_df_temp = pd.merge(lp_data_culled_df_temp, net_cost_guarantee, on = merge_col,  how = 'left')
        
        if len(lp_data_culled_df_temp[lp_data_culled_df_temp['NET_COST_GUARANTEE_UNIT'].isna()]) > 0:
            print(f"MISSING NET COST GUARANTEES FOR {len(lp_data_culled_df_temp.loc[lp_data_culled_df_temp['NET_COST_GUARANTEE_UNIT'].isna(), 'GPI'].unique())} GPIs. FILLING WITH NADAC, WAC, or AWP")
            lp_data_culled_df_temp['AWP_LIST_GPI'] = lp_data_culled_df_temp['AWP_LIST_GPI'].astype(float)
            net_cost_fill = lp_data_culled_df_temp.groupby(['CLIENT', 'GPI_NDC', 'BG_FLAG']).agg({'FULLAWP_ADJ':'sum', 'QTY':'sum', 'NADAC':'mean','WAC':'mean', 'AWP_LIST_GPI':'mean'}).reset_index()
            net_cost_fill['AVG_AWP_temp'] = (net_cost_fill['FULLAWP_ADJ']/net_cost_fill['QTY']) * (1-.25)
            net_cost_fill['NADAC'] = net_cost_fill['NADAC'] * (1+.02)
            net_cost_fill['WAC'] = net_cost_fill['WAC'] * (1+.02)
            net_cost_fill['AWP_LIST_GPI'] = np.where(net_cost_fill['BG_FLAG'] == 'G',
                                                         net_cost_fill['AWP_LIST_GPI'] * (1-.25), 
                                                         net_cost_fill['AWP_LIST_GPI'] * (1-.15))
            net_cost_fill['NET_COST_GUARANTEE_UNIT_temp'] = net_cost_fill[['NADAC','WAC', 'AVG_AWP_temp','AWP_LIST_GPI']].min(axis=1)
            lp_data_culled_df_temp = lp_data_culled_df_temp.merge(net_cost_fill[['CLIENT', 'GPI_NDC', 'BG_FLAG', 'NET_COST_GUARANTEE_UNIT_temp']], 
                                                                  on=['CLIENT', 'GPI_NDC', 'BG_FLAG'])
            lp_data_culled_df_temp.loc[lp_data_culled_df_temp['NET_COST_GUARANTEE_UNIT'].isna(), 'NET_COST_GUARANTEE_UNIT'] = lp_data_culled_df_temp.loc[lp_data_culled_df_temp['NET_COST_GUARANTEE_UNIT'].isna(), 'NET_COST_GUARANTEE_UNIT_temp'].copy()
            lp_data_culled_df_temp.drop(columns=['NET_COST_GUARANTEE_UNIT_temp'], inplace=True)
            
            #adding 1$ as NET_COST_GUARANTEE_UNIT for GPIs with no Guarantee and 0 utilization
            lp_data_culled_df_temp.loc[(lp_data_culled_df_temp['NET_COST_GUARANTEE_UNIT'].isna()) & (lp_data_culled_df_temp['QTY'] == 0.0), 'NET_COST_GUARANTEE_UNIT'] = 1.0
            assert lp_data_culled_df_temp['NET_COST_GUARANTEE_UNIT'].notna().all(), \
            f"Error: net cost guarantees contain NA values for GPIs with utilization after trying nadac/wac/awp. GPIs missing these values are {lp_data_culled_df_temp.loc[lp_data_culled_df_temp['NET_COST_GUARANTEE_UNIT'].isna(), 'GPI'].unique()}"
        assert lp_data_culled_df_temp['NET_COST_GUARANTEE_UNIT'].notna().all(), "Error: net cost guarantees contain NA values for GPIs with utilization"
        assert len(lp_data_culled_df_temp) == len(lp_data_culled_df), "len(lp_data_culled_df_temp) == len(lp_data_culled_df), check for duplication after merge"
        
    
    # Reassign created dataset to lp_data_culled_df
    lp_data_culled_df = lp_data_culled_df_temp.copy()
    
    return lp_data_culled_df
    
    

def add_benchmark_price(lp_data_culled_df):
    '''
    Adds a benchmark price column to the input DataFrame based on business rules. 
    This column will be used by the LP function to penalize unnecessary price movement away from the benchmark price.

    The benchmark price is determined as follows:
    - For TrueCost clients, use NET_COST_GUARANTEE_UNIT.
    - For Non TrueCost clients and BENCHMARK_CEILING_PRICE = True, fill missing values in SOFT_CONST_BENCHMARK_PRICE with BENCHMARK_CEILING_PRICE,
      and if still missing, with MAC_PRICE_UNIT_ADJ (current MAC price).
    - Otherwise for Non TrueCost clients, set the benchmark price to the current MAC price (MAC_PRICE_UNIT_ADJ).

    Args:
        lp_data_culled_df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with an added column called SOFT_CONST_BENCHMARK_PRICE.
    '''   

    lp_data_culled_df_temp = lp_data_culled_df.copy()
    
    #Create SOFT_CONST_BENCHMARK_PRICE column
    lp_data_culled_df_temp['SOFT_CONST_BENCHMARK_PRICE'] = np.nan
    
    if p.TRUECOST_CLIENT:
        lp_data_culled_df_temp['SOFT_CONST_BENCHMARK_PRICE'] = lp_data_culled_df_temp.SOFT_CONST_BENCHMARK_PRICE.fillna(lp_data_culled_df_temp.NET_COST_GUARANTEE_UNIT)
  
    elif p.USE_BENCHMARK_SC: 
        # Fill missing values in 'SOFT_CONST_BENCHMARK_PRICE' first with 'BENCHMARK_CEILING_PRICE', 
        # and if still missing, then with 'MAC_PRICE_UNIT_ADJ' (current_mac_price)
        lp_data_culled_df_temp['SOFT_CONST_BENCHMARK_PRICE'] = (lp_data_culled_df_temp['SOFT_CONST_BENCHMARK_PRICE']
                                                                    .fillna(lp_data_culled_df_temp['BENCHMARK_CEILING_PRICE'])        
                                                                    .fillna(lp_data_culled_df_temp['MAC_PRICE_UNIT_ADJ']))
                                                                            
    else: #If we do not want to use business rule defined benchmark price, benchmark price is set to be current_mac_price
        lp_data_culled_df_temp['SOFT_CONST_BENCHMARK_PRICE'] =lp_data_culled_df_temp.MAC_PRICE_UNIT_ADJ
    
    # Reassign created dataset to lp_data_culled_df
    lp_data_culled_df = lp_data_culled_df_temp

    return lp_data_culled_df


def add_disp_fee_avg(lp_data_culled_df):
    '''
    Calculate dispensing fee projections for both actual and target values. Projects Lag
    and EOY.
    '''
    lp_data_culled_df['AVG_DISP_FEE'] = 0
    lp_data_culled_df.loc[lp_data_culled_df['CLAIMS'] != 0, 'AVG_DISP_FEE'] = \
        lp_data_culled_df['DISP_FEE'] / lp_data_culled_df['CLAIMS']
    
    lp_data_culled_df['AVG_TARGET_DISP_FEE'] = 0
    lp_data_culled_df.loc[lp_data_culled_df['CLAIMS'] != 0, 'AVG_TARGET_DISP_FEE'] = \
        lp_data_culled_df['TARGET_DISP_FEE'] / lp_data_culled_df['CLAIMS']
    
    lp_data_culled_df['PHARM_AVG_DISP_FEE'] = 0
    lp_data_culled_df.loc[lp_data_culled_df['PHARM_CLAIMS'] != 0, 'PHARM_AVG_DISP_FEE'] = \
        lp_data_culled_df['PHARMACY_DISP_FEE'] / lp_data_culled_df['PHARM_CLAIMS']
    
    lp_data_culled_df['PHARM_AVG_TARGET_DISP_FEE'] = 0
    lp_data_culled_df.loc[lp_data_culled_df['PHARM_CLAIMS'] != 0, 'PHARM_AVG_TARGET_DISP_FEE'] = \
        lp_data_culled_df['PHARM_TARGET_DISP_FEE'] / lp_data_culled_df['PHARM_CLAIMS']
    
    return lp_data_culled_df


def get_pharms_vcml_for_nmr(lp_data_culled_df):
    '''
    Get the list of pharmacies that do not have a guarantee and 
    list of R30 VCMLs that have prices for chain_groups that have OK chain_subgroups
    Then creates 2 new columns:
    HAS_PHARM_GUARANTEE, IN_NONCAP_OK_VCML
    '''
    
    # getting pharmacies that do not have a pharm guarantee
    no_guar_pharms = lp_data_culled_df.loc[(lp_data_culled_df['PHARMACY_RATE'].isna()) 
                                           & (lp_data_culled_df['CHAIN_GROUP'] != 'MAIL'), 'CHAIN_GROUP'].unique()
    lp_data_culled_df['HAS_PHARM_GUARANTEE'] = ~lp_data_culled_df['CHAIN_GROUP'].isin(no_guar_pharms)

    # getting unique vcmls of pharmacies that do not have a pharm guarantee
    unique_non_cap_vcml = lp_data_culled_df.loc[(lp_data_culled_df['PHARMACY_RATE'].isna()) 
                                                   & (lp_data_culled_df['CHAIN_GROUP'] != 'MAIL'), 'MAC_LIST'].unique()

    # getting unique R30 vcmls of chain_groups that have OK chain_subgroups
    ok_subgroups = lp_data_culled_df.loc[lp_data_culled_df['CHAIN_SUBGROUP'].str.contains('OK'), 'CHAIN_GROUP'].unique()
    unique_ok_vcml = lp_data_culled_df.loc[(lp_data_culled_df['CHAIN_SUBGROUP'].isin(ok_subgroups) 
                           & (lp_data_culled_df['MEASUREMENT'] == 'R30')), 'MAC_LIST'].unique()

    # unique_non_cap_ok_vcmls = set(list(unique_non_cap_vcml) + list(unique_ok_vcml)) #adhoc change - remove this once VCML mapp
    unique_non_cap_ok_vcmls = set()
    
    lp_data_culled_df['IN_NONCAP_OK_VCML'] = lp_data_culled_df['MAC_LIST'].isin(unique_non_cap_ok_vcmls)
    
    return lp_data_culled_df


def compute_effective_prices(lp_data_culled_df):
    #lp_data_culled_df = lp_data_culled_df.copy()
    if p.NDC_UPDATE:
        
        lp_data_culled_df['EFF_UNIT_PRICE_OLD'] = determine_effective_price(lp_data_culled_df,
                                                                            old_price='OLD_MAC_PRICE')
        lp_data_culled_df['EFF_UNIT_PRICE_OLD'] = lp_data_culled_df.apply(
            lambda df: df['EFF_UNIT_PRICE_OLD'] if df['EFF_UNIT_PRICE_OLD'] > 0 else df['PRICE_REIMB_UNIT'], axis=1)
        assert(len(lp_data_culled_df.loc[(lp_data_culled_df.EFF_UNIT_PRICE_OLD.isna())]) == 0), \
            "len(lp_data_culled_df.loc[(lp_data_culled_df.EFF_UNIT_PRICE_OLD.isna())]) == 0"
        
        lp_data_culled_df['PHARM_EFF_UNIT_PRICE_OLD'] = determine_effective_price(lp_data_culled_df,
                                                                                  old_price='OLD_MAC_PRICE')
        lp_data_culled_df['PHARM_EFF_UNIT_PRICE_OLD'] = lp_data_culled_df.apply(
            lambda df: df['PHARM_EFF_UNIT_PRICE_OLD'] if df['PHARM_EFF_UNIT_PRICE'] > 0 else df['PHARM_PRICE_REIMB_UNIT'], axis=1)
        assert(len(lp_data_culled_df.loc[(lp_data_culled_df.PHARM_EFF_UNIT_PRICE.isna())]) == 0), \
            "len(lp_data_culled_df.loc[(lp_data_culled_df.PHARM_EFF_UNIT_PRICE.isna())]) == 0"

    lp_data_culled_df['EFF_UNIT_PRICE'] = determine_effective_price(lp_data_culled_df,
                                                                    old_price='CURRENT_MAC_PRICE')
    lp_data_culled_df['EFF_UNIT_PRICE'] = lp_data_culled_df.apply(
        lambda df: df['EFF_UNIT_PRICE'] if df['EFF_UNIT_PRICE'] > 0 else df['PRICE_REIMB_UNIT'], axis=1)
    assert(len(lp_data_culled_df.loc[(lp_data_culled_df.EFF_UNIT_PRICE.isna())]) == 0), \
        "len(lp_data_culled_df.loc[(lp_data_culled_df.EFF_UNIT_PRICE.isna())]) == 0"

    lp_data_culled_df['PHARM_EFF_UNIT_PRICE'] = determine_effective_price(lp_data_culled_df,
                                                                          old_price='CURRENT_MAC_PRICE')
    lp_data_culled_df['PHARM_EFF_UNIT_PRICE'] = lp_data_culled_df.apply(
        lambda df: df['PHARM_EFF_UNIT_PRICE'] if df['PHARM_EFF_UNIT_PRICE'] > 0 else df['PHARM_PRICE_REIMB_UNIT'], axis=1)
    assert(len(lp_data_culled_df.loc[(lp_data_culled_df.PHARM_EFF_UNIT_PRICE.isna())]) == 0), \
        "len(lp_data_culled_df.loc[(lp_data_culled_df.PHARM_EFF_UNIT_PRICE.isna())]) == 0"

    #Create Adjusted mac price with raw mac prices for the definition of the LP
    lp_data_culled_df['MAC_PRICE_UNIT_ADJ'] = lp_data_culled_df.apply(
        lambda df: df['CURRENT_MAC_PRICE'] if df['CURRENT_MAC_PRICE']>0 else df['PRICE_REIMB_UNIT'], axis=1)
    assert(len(lp_data_culled_df.loc[(lp_data_culled_df.MAC_PRICE_UNIT_ADJ.isna())]) == 0), \
        "len(lp_data_culled_df.loc[(lp_data_culled_df.MAC_PRICE_UNIT_ADJ.isna())]) == 0"
    
    lp_data_culled_df['PHARM_MAC_PRICE_UNIT_ADJ'] = lp_data_culled_df.apply(
        lambda df: df['CURRENT_MAC_PRICE'] if df['CURRENT_MAC_PRICE']>0 else df['PHARM_PRICE_REIMB_UNIT'], axis=1)
    assert(len(lp_data_culled_df.loc[(lp_data_culled_df.MAC_PRICE_UNIT_ADJ.isna())]) == 0), \
        "len(lp_data_culled_df.loc[(lp_data_culled_df.MAC_PRICE_UNIT_ADJ.isna())]) == 0"

    if p.NDC_UPDATE:
        lp_data_culled_df['EFF_CAPPED_PRICE_OLD'] = determine_effective_price(lp_data_culled_df,
                                                                              old_price='OLD_MAC_PRICE',
                                                                              uc_unit='UC_UNIT25',
                                                                              capped_only=True)
        lp_data_culled_df['EFF_CAPPED_PRICE_OLD'] = lp_data_culled_df.apply(
            lambda df: df['EFF_CAPPED_PRICE_OLD'] if df['EFF_CAPPED_PRICE_OLD'] > 0 else df['PRICE_REIMB_UNIT'], axis=1)
        assert(len(lp_data_culled_df.loc[(lp_data_culled_df.EFF_CAPPED_PRICE_OLD.isna())]) == 0), \
            "len(lp_data_culled_df.loc[(lp_data_culled_df.EFF_CAPPED_PRICE_OLD.isna())]) == 0"

        lp_data_culled_df['PHARM_EFF_CAPPED_PRICE_OLD'] = determine_effective_price(lp_data_culled_df,
                                                                                    old_price='OLD_MAC_PRICE',
                                                                                    uc_unit='UC_UNIT25',
                                                                                    capped_only=True)
        lp_data_culled_df['PHARM_EFF_CAPPED_PRICE_OLD'] = lp_data_culled_df.apply(
            lambda df: df['PHARM_EFF_CAPPED_PRICE_OLD'] if df['PHARM_EFF_CAPPED_PRICE_OLD'] > 0 else df['PHARM_PRICE_REIMB_UNIT'], axis=1)
        assert(len(lp_data_culled_df.loc[(lp_data_culled_df.PHARM_EFF_CAPPED_PRICE_OLD.isna())]) == 0), \
            "len(lp_data_culled_df.loc[(lp_data_culled_df.PHARM_EFF_CAPPED_PRICE_OLD.isna())]) == 0"

    #To be used in LP
    lp_data_culled_df['EFF_CAPPED_PRICE'] = determine_effective_price(lp_data_culled_df,
                                                                      old_price='CURRENT_MAC_PRICE',
                                                                      uc_unit='UC_UNIT25',
                                                                      capped_only=True)
    lp_data_culled_df['EFF_CAPPED_PRICE'] = lp_data_culled_df.apply(
        lambda df: df['EFF_CAPPED_PRICE'] if df['EFF_CAPPED_PRICE'] > 0 else df['PRICE_REIMB_UNIT'], axis=1)
    assert(len(lp_data_culled_df.loc[(lp_data_culled_df.EFF_CAPPED_PRICE.isna())]) == 0), \
        "len(lp_data_culled_df.loc[(lp_data_culled_df.EFF_CAPPED_PRICE.isna())]) == 0"

    lp_data_culled_df['PHARM_EFF_CAPPED_PRICE'] = determine_effective_price(lp_data_culled_df,
                                                                            old_price='CURRENT_MAC_PRICE',
                                                                            uc_unit='UC_UNIT25',
                                                                            capped_only=True)
    lp_data_culled_df['PHARM_EFF_CAPPED_PRICE'] = lp_data_culled_df.apply(
        lambda df: df['PHARM_EFF_CAPPED_PRICE'] if df['PHARM_EFF_CAPPED_PRICE'] > 0 else df['PHARM_PRICE_REIMB_UNIT'], axis=1)
    assert(len(lp_data_culled_df.loc[(lp_data_culled_df.PHARM_EFF_CAPPED_PRICE.isna())]) == 0), \
        "len(lp_data_culled_df.loc[(lp_data_culled_df.PHARM_EFF_CAPPED_PRICE.isna())]) == 0"


    ##### Checks ####
    lp_data_culled_df[(lp_data_culled_df.CLAIMS.isna())].shape[0]
    lp_data_culled_df[(lp_data_culled_df.QTY.isna())].shape[0]
    lp_data_culled_df[(~lp_data_culled_df.QTY.isna()) & (lp_data_culled_df.QTY == 0)].shape[0]
    lp_data_culled_df.isna().sum()
    lp_data_culled_df.loc[:,lp_data_culled_df.columns != 'PHRM_GRTE_TYPE'] = lp_data_culled_df.loc[:,lp_data_culled_df.columns != 'PHRM_GRTE_TYPE'].fillna(0)
    
    #added to allow use of some older functions when there was a disctiction between
    #the two of these.  Should be removed when those functions are updated
    lp_data_culled_df['PRICE_REIMB_ADJ'] = lp_data_culled_df['PRICE_REIMB']
    
    if p.INCLUDE_PLAN_LIABILITY:
        lp_data_culled_df.loc[lp_data_culled_df.CURRENT_MAC_PRICE == 0.0000, 'COPAY'] =\
            30 * lp_data_culled_df.loc[lp_data_culled_df.CURRENT_MAC_PRICE == 0.0000, 'PSTCOPAY'] /\
                lp_data_culled_df.loc[lp_data_culled_df.CURRENT_MAC_PRICE == 0.0000, 'DAYSSUP']
        
        lp_data_culled_df['DISP_FEE'] = 0.50
        lp_data_culled_df.loc[lp_data_culled_df.Pharmacy_Type == 'Preferred', 'DISP_FEE'] = 0.40
        lp_data_culled_df.loc[lp_data_culled_df.MEASUREMENT == 'M30', 'DISP_FEE'] = 0.00

    return lp_data_culled_df

    

def prepare_output_data(lp_data_culled_df, lp_data_culled_df_actual):
    '''
    The purpose of this function is to recalculate claims, qty and awp_adj incase UNC adjustments are required. We calculate UNC_FRAC to determine
    number of claims which require UNC adjustment. This fraction when multiplied by each variable is used to repopulate the claims,qty and awp columns
    for respective time period.
    
    input: lp_data_culled_df
    output: lp_data_culled_df with recalculated claims, qty and awp columns 
    '''
    lp_data_culled_df_temp = lp_data_culled_df.merge(
        lp_data_culled_df_actual[['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'BG_FLAG', 'GPI_NDC', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'QTY']].rename(
            columns={'QTY': 'QTY_ACTUAL'}),
        on=['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'BG_FLAG', 'GPI_NDC', 'CHAIN_GROUP', 'CHAIN_SUBGROUP'],
        how='left'
    )
    assert np.abs(lp_data_culled_df.FULLAWP_ADJ.sum() - lp_data_culled_df_temp.FULLAWP_ADJ.sum()) < 1.E-4, "Merge error adding actual QTYs"
    lp_data_culled_df = lp_data_culled_df_temp
    lp_data_culled_df['UNC_FRAC_OLD'] = 1 - lp_data_culled_df['QTY_ACTUAL']/lp_data_culled_df['QTY']
    lp_data_culled_df.loc[(~lp_data_culled_df['PRICE_CHANGED_UC']) | (lp_data_culled_df['IS_MAINTENANCE_UC']), 
                          'UNC_FRAC'] = lp_data_culled_df.loc[(~lp_data_culled_df['PRICE_CHANGED_UC']) 
                                                              | (lp_data_culled_df['IS_MAINTENANCE_UC']), 
                                                              'UNC_FRAC_OLD']
    lp_data_culled_df.drop(columns=['QTY_ACTUAL'], inplace=True)
    
    if p.UNC_CLIENT:
        lp_data_culled_df['CLAIMS_PROJ_EOY_OLDUNC'] = lp_data_culled_df['CLAIMS_PROJ_EOY']*(1-lp_data_culled_df['UNC_FRAC_OLD'])
        lp_data_culled_df['QTY_PROJ_EOY_OLDUNC'] = lp_data_culled_df['QTY_PROJ_EOY']*(1-lp_data_culled_df['UNC_FRAC_OLD'])
        lp_data_culled_df['FULLAWP_ADJ_PROJ_EOY_OLDUNC'] = lp_data_culled_df['FULLAWP_ADJ_PROJ_EOY']*(1-lp_data_culled_df['UNC_FRAC_OLD'])
        lp_data_culled_df['CLAIMS_PROJ_EOY'] *= (1-lp_data_culled_df['UNC_FRAC'])
        lp_data_culled_df['QTY_PROJ_EOY'] *= (1-lp_data_culled_df['UNC_FRAC'])
        lp_data_culled_df['FULLAWP_ADJ_PROJ_EOY'] *= (1-lp_data_culled_df['UNC_FRAC'])
        lp_data_culled_df['LM_CLAIMS_OLDUNC'] = lp_data_culled_df['LM_CLAIMS']*(1-lp_data_culled_df['UNC_FRAC_OLD'])
        lp_data_culled_df['LM_QTY_OLDUNC'] = lp_data_culled_df['LM_QTY']*(1-lp_data_culled_df['UNC_FRAC_OLD'])
        lp_data_culled_df['LM_FULLAWP_ADJ_OLDUNC'] = lp_data_culled_df['LM_FULLAWP_ADJ']*(1-lp_data_culled_df['UNC_FRAC_OLD'])
        lp_data_culled_df['LM_CLAIMS'] *= (1-lp_data_culled_df['UNC_FRAC'])
        lp_data_culled_df['LM_QTY'] *= (1-lp_data_culled_df['UNC_FRAC'])
        lp_data_culled_df['LM_FULLAWP_ADJ'] *= (1-lp_data_culled_df['UNC_FRAC'])
    if p.UNC_PHARMACY:
        lp_data_culled_df['PHARM_CLAIMS_PROJ_EOY_OLDUNC'] = lp_data_culled_df['PHARM_CLAIMS_PROJ_EOY']
        lp_data_culled_df['PHARM_QTY_PROJ_EOY_OLDUNC'] = lp_data_culled_df['PHARM_QTY_PROJ_EOY']
        lp_data_culled_df['PHARM_FULLAWP_ADJ_PROJ_EOY_OLDUNC'] = lp_data_culled_df['PHARM_FULLAWP_ADJ_PROJ_EOY']
        
        pharm_mask = lp_data_culled_df['CHAIN_GROUP'].isin(p.UNC_PHARMACY_CHAIN_GROUPS)
        lp_data_culled_df.loc[pharm_mask, 'PHARM_CLAIMS_PROJ_EOY_OLDUNC'] = lp_data_culled_df.loc[pharm_mask, 'PHARM_CLAIMS_PROJ_EOY']*(1-lp_data_culled_df.loc[pharm_mask, 'UNC_FRAC_OLD'])
        lp_data_culled_df.loc[pharm_mask, 'PHARM_QTY_PROJ_EOY_OLDUNC'] = lp_data_culled_df.loc[pharm_mask, 'PHARM_QTY_PROJ_EOY']*(1-lp_data_culled_df.loc[pharm_mask, 'UNC_FRAC_OLD'])
        lp_data_culled_df.loc[pharm_mask, 'PHARM_FULLAWP_ADJ_PROJ_EOY_OLDUNC'] = lp_data_culled_df.loc[pharm_mask, 'PHARM_FULLAWP_ADJ_PROJ_EOY']*(1-lp_data_culled_df.loc[pharm_mask, 'UNC_FRAC_OLD'])
        lp_data_culled_df.loc[pharm_mask, 'PHARM_CLAIMS_PROJ_EOY'] *= (1-lp_data_culled_df.loc[pharm_mask, 'UNC_FRAC'])
        lp_data_culled_df.loc[pharm_mask, 'PHARM_QTY_PROJ_EOY'] *= (1-lp_data_culled_df.loc[pharm_mask, 'UNC_FRAC'])
        lp_data_culled_df.loc[pharm_mask, 'PHARM_FULLAWP_ADJ_PROJ_EOY'] *= (1-lp_data_culled_df.loc[pharm_mask, 'UNC_FRAC'])
    lp_data_culled_df = lp_data_culled_df.fillna(0)
    
    
    return lp_data_culled_df

def add_pharmacy_perf_names(lp_data_culled_df):
    chain_subgroup_df = lp_data_culled_df[['CLIENT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP']].drop_duplicates()
    chain_group_df = chain_subgroup_df.groupby(['CLIENT', 'CHAIN_GROUP'], as_index=False).agg(COUNT=('CHAIN_SUBGROUP', 'count'))
    chain_subgroup_df = chain_subgroup_df.merge(chain_group_df, on=['CLIENT', 'CHAIN_GROUP'])
    #June 5 change
    #chain_subgroup_df['PHARMACY_PERF_NAME'] = chain_subgroup_df['CHAIN_GROUP']+'_'+chain_subgroup_df['CHAIN_SUBGROUP']
    chain_subgroup_df['PHARMACY_PERF_NAME'] = chain_subgroup_df['CHAIN_SUBGROUP']
    chain_subgroup_identity_mask = (chain_subgroup_df['COUNT']==1) & (chain_subgroup_df['CHAIN_GROUP']==chain_subgroup_df['CHAIN_SUBGROUP'])
    chain_subgroup_df.loc[chain_subgroup_identity_mask, 'PHARMACY_PERF_NAME'] = chain_subgroup_df.loc[chain_subgroup_identity_mask, 'CHAIN_GROUP']
    lp_data_culled_df_temp = lp_data_culled_df.merge(chain_subgroup_df[['CLIENT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'PHARMACY_PERF_NAME']],
                                                     on=['CLIENT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP'])
    assert np.abs(lp_data_culled_df_temp.FULLAWP_ADJ.sum() - lp_data_culled_df.FULLAWP_ADJ.sum())<0.0001, "failed join on chain subgroup names"
    return lp_data_culled_df_temp

#################Done combining data##########################################
#===== Create Output=====
def save_data(
    lp_data_culled_df, mac_list_df, chain_region_mac_mapping, lp_data_culled_df_actual = pd.DataFrame(), WRITE_TO_BQ=False):
    
    if WRITE_TO_BQ:
        lp_data_culled_df = lp_data_culled_df.rename(
            columns = {
                '1026_NDC_PRICE':'num1026_NDC_PRICE', 
                "1026_GPI_PRICE": "num1026_GPI_PRICE"})
        uf.write_to_bq(
            lp_data_culled_df,
            project_output = p.BQ_OUTPUT_PROJECT_ID,
            dataset_output = p.BQ_OUTPUT_DATASET,
            table_id = "lp_data",
            client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
            timestamp_param = p.TIMESTAMP,
            run_id = p.AT_RUN_ID,
            schema = BQ.lp_data_schema
        )
    else:
        fname = p.FILE_DYNAMIC_INPUT_PATH + 'lp_data_' + p.DATA_ID + '.csv'
        lp_data_culled_df.to_csv(fname, index=False)

    if not lp_data_culled_df_actual.empty:
        if WRITE_TO_BQ:
            lp_data_culled_df_actual = lp_data_culled_df_actual.rename(
                columns = {
                    '1026_NDC_PRICE':'num1026_NDC_PRICE',
                    "1026_GPI_PRICE": "num1026_GPI_PRICE"})
            uf.write_to_bq(
                lp_data_culled_df_actual,
                project_output = p.BQ_OUTPUT_PROJECT_ID,
                dataset_output = p.BQ_OUTPUT_DATASET,
                table_id = "lp_data_actual",
                client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                timestamp_param = p.TIMESTAMP,
                run_id = p.AT_RUN_ID,
                schema = BQ.lp_data_schema
            )
        else:

            fname = p.FILE_DYNAMIC_INPUT_PATH + 'lp_data_actual_' + p.DATA_ID + '.csv'
            lp_data_culled_df_actual.to_csv(fname, index=False)

    if WRITE_TO_BQ:
        uf.write_to_bq(
            mac_list_df,
            project_output = p.BQ_OUTPUT_PROJECT_ID,
            dataset_output = p.BQ_OUTPUT_DATASET,
            table_id = "mac_lists",
            client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
            timestamp_param = p.TIMESTAMP,
            run_id = p.AT_RUN_ID,
            schema = BQ.mac_lists_schema
        )
        
    else:
        fname = p.FILE_DYNAMIC_INPUT_PATH + 'mac_lists_' + p.DATA_ID + '.csv'
        mac_list_df.to_csv(fname, index=False)
    
    #save mac mapping
    if WRITE_TO_BQ:
        uf.write_to_bq(
            chain_region_mac_mapping,
            project_output = p.BQ_OUTPUT_PROJECT_ID,
            dataset_output = p.BQ_OUTPUT_DATASET,
            table_id = "chain_region_mac_mapping",
            client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
            timestamp_param = p.TIMESTAMP,
            run_id = p.AT_RUN_ID,
            schema = BQ.chain_region_mac_mapping_schema
        )
    else:
        fname = p.FILE_DYNAMIC_INPUT_PATH + 'mac_mapping_' + p.DATA_ID + '.csv'
        chain_region_mac_mapping.to_csv(fname, index=False)


def main():
    
    #================= DEFINE SOME TEMP VARIABLE ===========================
    #day counts for days in data, lag, and until end of year with a one day
    #compensation because go live date is included in the projections

    contract_date_df = pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.CONTRACT_DATE_FILE))
    contract_date_df['CONTRACT_EFF_DT'] = pd.to_datetime(contract_date_df['CONTRACT_EFF_DT'])
    eoy = dt.datetime.strptime(contract_date_df['CONTRACT_EXPRN_DT'][0], '%Y-%m-%d')
    lag_days = (p.GO_LIVE-p.LAST_DATA).days - 1
    eoy_days = (eoy - p.GO_LIVE).days + 1
    
    # Determine start of data based on end of data
    end_day = calendar.monthrange(p.LAST_DATA.year, p.LAST_DATA.month)[1] 
    assert p.GO_LIVE >= p.LAST_DATA, "p.GO_LIVE >= p.LAST_DATA"
    
    mac_list_df = read_mac_lists(READ_FROM_BQ = p.READ_FROM_BQ)  
    mac_list_df = mac_list_df.rename(columns={'NEW_MAC_LIST':'MAC_LIST'}) #MARCEL ADDED to run MEDD client
    mac_list_df['MAC_LIST'] = mac_list_df['MAC_LIST'].astype(str)  #MARCEL ADDED to run MEDD client
    gpi_vol_awp_df = read_gpi_vol_awp(READ_FROM_BQ = p.READ_FROM_BQ)
    chain_region_mac_mapping = read_chain_region_mac_mapping(READ_FROM_BQ = p.READ_FROM_BQ)
    
    pkg_size = read_package_size(READ_FROM_BQ = p.READ_FROM_BQ)
    mac_1026_gpi, mac_1026_ndc = read_mac_1026(READ_FROM_BQ = p.READ_FROM_BQ)
    pref_pharm_list = read_pref_pharm_list(READ_FROM_BQ = p.READ_FROM_BQ)
    if p.APPLY_BENCHMARK_CAP:
        benchmark_nadac_prices = read_benchmark_nadac_prices(READ_FROM_BQ = p.READ_FROM_BQ)
    # gpi_vol_awp_df: Has all the claims independently of adjudication method (U&C or GRx)
    # gpi_vol_awp_df_actual: Has all the claims adjudicated at MAC.
    #TODO: Change to _actual and "".This way all of the pre-processing optimization will have the same structure.
    
    if p.UNC_OPT:
        unc_NDC_percentiles = read_unc_NDC_percentiles(READ_FROM_BQ = p.READ_FROM_BQ)
        unc_GPI_percentiles = read_unc_GPI_percentiles(READ_FROM_BQ = p.READ_FROM_BQ)
        gpi_vol_awp_df_actual = gpi_vol_awp_df.copy()
        gpi_vol_awp_unc_df = read_gpi_vol_awp_unc(READ_FROM_BQ = p.READ_FROM_BQ)
        #gpi_vol_awp_unc_df['ADJUDICATED_AT'] = 'U&C' #Removed as this info can be gotten from daily_totals
        #gpi_vol_awp_df['ADJUDICATED_AT'] = 'MAC'
        gpi_vol_awp_df = pd.concat(
            [gpi_vol_awp_df,gpi_vol_awp_unc_df],axis=0,ignore_index=True)
        
    gpi_vol_awp_df = pre_process_data(gpi_vol_awp_df)
    
    if p.UNC_OPT:
        gpi_vol_awp_df_actual = pre_process_data(gpi_vol_awp_df_actual)

    # If a market check occurred, split pre- and post-check time periods for YTD performance calculations
    if p.MARKET_CHECK:
        pharmacy_guarantees = read_pharmacy_guarantees(READ_FROM_BQ = p.READ_FROM_BQ)
        client_guarantees_premc = standardize_df(pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.CLIENT_GUARANTEE_PREMC_FILE), dtype = p.VARIABLE_TYPE_DIC))
        if p.UNC_OPT:
            gpi_vol_awp_override_pre = gpi_vol_awp_df_actual[['CLIENT','CLIENT_NAME','BREAKOUT','REGION',
                                                              'MEASUREMENT','BG_FLAG', 'PREFERRED','CHAIN_GROUP','CHAIN_SUBGROUP','GPI',
                                                              'NDC','DOF','FULLAWP_ADJ','PHARM_PRICE_REIMB','PHARM_FULLAWP_ADJ',
                                                              'PHARM_FULLNADAC_ADJ','PHARM_FULLACC_ADJ','PHARM_TARG_INGCOST_ADJ',
                                                              'PRICE_REIMB']].loc[(gpi_vol_awp_df_actual['DOF'] >=
                                                                                   contract_date_df['CONTRACT_EFF_DT'].astype('str').values[0]) & 
                                                                                   (gpi_vol_awp_df_actual['DOF'] < 
                                                                                    contract_date_df['MARKET_CHECK_DT'].astype('str').values[0])]
            gpi_vol_awp_override_pre = pd.merge(
                    gpi_vol_awp_override_pre, pharmacy_guarantees, how ='left', 
            on = ['CLIENT', 'BREAKOUT','REGION','BG_FLAG','CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT'])
            gpi_vol_awp_override_pre = add_pharma_list(gpi_vol_awp_override_pre, pref_pharm_list, ignore_columns = True)
            gpi_vol_awp_override_pre = add_pharmacy_perf_names(gpi_vol_awp_override_pre)
            gpi_vol_awp_override_pre = add_target_ingcost(gpi_vol_awp_override_pre, client_guarantees_premc, client_rate_col='RATE') 
            gpi_vol_awp_override_pre.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.PRE_MC_DATA_FILE, index = False)

            gpi_vol_awp_override_post = gpi_vol_awp_df_actual[['CLIENT','CLIENT_NAME','BREAKOUT','REGION',
                                                               'MEASUREMENT','BG_FLAG','PREFERRED','CHAIN_GROUP','CHAIN_SUBGROUP','GPI',
                                                               'NDC','DOF','FULLAWP_ADJ','PHARM_PRICE_REIMB','PHARM_FULLAWP_ADJ',
                                                               'PHARM_FULLNADAC_ADJ','PHARM_FULLACC_ADJ','PHARM_TARG_INGCOST_ADJ',
                                                               'PRICE_REIMB']].loc[(gpi_vol_awp_df_actual['DOF'] >=
                                                                                    contract_date_df['MARKET_CHECK_DT'].astype('str').values[0]) & 
                                                                                   (gpi_vol_awp_df_actual['DOF'] <= p.LAST_DATA)]
            gpi_vol_awp_override_post = pd.merge(
                    gpi_vol_awp_override_post, pharmacy_guarantees, how ='left', 
            on = ['CLIENT', 'BREAKOUT','REGION','CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT','BG_FLAG'])
            gpi_vol_awp_override_post = add_pharma_list(gpi_vol_awp_override_post, pref_pharm_list, ignore_columns = True)
            gpi_vol_awp_override_post = add_pharmacy_perf_names(gpi_vol_awp_override_post)
            gpi_vol_awp_override_post = add_target_ingcost(gpi_vol_awp_override_post, client_guarantees_premc, client_rate_col='RATE') 
            gpi_vol_awp_override_post.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.POST_MC_DATA_FILE, index = False)
        else:
            gpi_vol_awp_override_pre = gpi_vol_awp_df[['CLIENT','CLIENT_NAME','BREAKOUT','REGION',
                                                       'MEASUREMENT','BG_FLAG','PREFERRED','CHAIN_GROUP','CHAIN_SUBGROUP','GPI',
                                                       'NDC','DOF','FULLAWP_ADJ','PHARM_PRICE_REIMB','PHARM_FULLAWP_ADJ',
                                                       'PHARM_FULLNADAC_ADJ','PHARM_FULLACC_ADJ','PHARM_TARG_INGCOST_ADJ',
                                                       'PRICE_REIMB']].loc[(gpi_vol_awp_df['DOF'] >=
                                                                            contract_date_df['CONTRACT_EFF_DT'].astype('str').values[0]) & 
                                                                            (gpi_vol_awp_df['DOF'] <
                                                                             contract_date_df['MARKET_CHECK_DT'].astype('str').values[0])]
            gpi_vol_awp_override_pre = pd.merge(
                    gpi_vol_awp_override_pre, pharmacy_guarantees, how ='left', 
            on = ['CLIENT', 'BREAKOUT','REGION','CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT','BG_FLAG'])
            gpi_vol_awp_override_pre = add_pharma_list(gpi_vol_awp_override_pre, pref_pharm_list, ignore_columns = True)
            gpi_vol_awp_override_pre = add_pharmacy_perf_names(gpi_vol_awp_override_pre)
            gpi_vol_awp_override_pre = add_target_ingcost(gpi_vol_awp_override_pre, client_guarantees_premc, client_rate_col='RATE') 
            gpi_vol_awp_override_pre.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.PRE_MC_DATA_FILE, index = False)

            gpi_vol_awp_override_post = gpi_vol_awp_df[['CLIENT','CLIENT_NAME','BREAKOUT','REGION',
                                                        'MEASUREMENT','BG_FLAG','PREFERRED','CHAIN_GROUP','CHAIN_SUBGROUP','GPI',
                                                        'NDC','DOF','FULLAWP_ADJ','PHARM_PRICE_REIMB','PHARM_FULLAWP_ADJ',
                                                        'PHARM_FULLNADAC_ADJ','PHARM_FULLACC_ADJ','PHARM_TARG_INGCOST_ADJ',
                                                        'PRICE_REIMB']].loc[(gpi_vol_awp_df['DOF'] >=
                                                                            contract_date_df['MARKET_CHECK_DT'].astype('str').values[0]) & 
                                                                            (gpi_vol_awp_df['DOF'] <= p.LAST_DATA)]
            gpi_vol_awp_override_post = pd.merge(
                    gpi_vol_awp_override_post, pharmacy_guarantees, how ='left', 
            on = ['CLIENT', 'BREAKOUT','REGION','CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT','BG_FLAG'])
            gpi_vol_awp_override_post = add_pharma_list(gpi_vol_awp_override_post, pref_pharm_list, ignore_columns = True)
            gpi_vol_awp_override_post = add_pharmacy_perf_names(gpi_vol_awp_override_post)
            gpi_vol_awp_override_post = add_target_ingcost(gpi_vol_awp_override_post, client_guarantees_premc, client_rate_col='RATE') 
            gpi_vol_awp_override_post.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.POST_MC_DATA_FILE, index = False)
            
    # Get current contract YTD data for the cross contract YTD performance calculation
    if p.CROSS_CONTRACT_PROJ:
        if p.UNC_OPT:
            # Get current contract data
            gpi_vol_awp_override = gpi_vol_awp_df_actual[['CLIENT','CLIENT_NAME','BREAKOUT','REGION',
                                                          'MEASUREMENT','BG_FLAG','PREFERRED','CHAIN_GROUP','CHAIN_SUBGROUP',
                                                          'GPI','NDC','DOF','FULLAWP_ADJ','PHARM_QTY','PHARM_PRICE_REIMB',
                                                          'PHARM_FULLAWP_ADJ','PHARM_FULLNADAC_ADJ','PHARM_FULLACC_ADJ',
                                                          'PHARM_TARG_INGCOST_ADJ','PRICE_REIMB']].loc[(gpi_vol_awp_df_actual['DOF'] >=
                                                                               contract_date_df['CONTRACT_EFF_DT'].astype('str').values[0]) & 
                                                                                                        (gpi_vol_awp_df_actual['DOF'] <= p.LAST_DATA)]
            pharmacy_guarantees = read_pharmacy_guarantees(READ_FROM_BQ = p.READ_FROM_BQ)
            gpi_vol_awp_override = pd.merge(
                    gpi_vol_awp_override, pharmacy_guarantees, how ='left', 
                    on = ['CLIENT', 'BREAKOUT','REGION','CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT','BG_FLAG'])
            gpi_vol_awp_override = add_pharma_list(gpi_vol_awp_override, pref_pharm_list, ignore_columns = True)
            gpi_vol_awp_override = add_pharmacy_perf_names(gpi_vol_awp_override)
            gpi_vol_awp_override = add_target_ingcost(gpi_vol_awp_override, client_guarantees) 
            gpi_vol_awp_override.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.CURRENT_CONTRACT_DATA_FILE, index = False)
#             cross_contract_utilization_check(gpi_vol_awp_df_actual, contract_date_df)
        else:
            # Get current contract data
            gpi_vol_awp_override = gpi_vol_awp_df[['CLIENT','CLIENT_NAME','BREAKOUT','REGION',
                                                   'MEASUREMENT','BG_FLAG','PREFERRED','CHAIN_GROUP','CHAIN_SUBGROUP',
                                                   'GPI','NDC','DOF','FULLAWP_ADJ','PHARM_QTY','PHARM_PRICE_REIMB',
                                                   'PHARM_FULLAWP_ADJ','PHARM_FULLNADAC_ADJ','PHARM_FULLACC_ADJ','PHARM_TARG_INGCOST_ADJ',
                                                   'PRICE_REIMB']].loc[(gpi_vol_awp_df['DOF'] >= contract_date_df['CONTRACT_EFF_DT'].astype('str').values[0]) & 
                                                                                                        (gpi_vol_awp_df['DOF'] <= p.LAST_DATA)]
            pharmacy_guarantees = read_pharmacy_guarantees(READ_FROM_BQ = p.READ_FROM_BQ)
            gpi_vol_awp_override = pd.merge(
                    gpi_vol_awp_override, pharmacy_guarantees, how ='left', 
                    on = ['CLIENT', 'BREAKOUT','REGION','CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT','BG_FLAG'])
            gpi_vol_awp_override = add_pharma_list(gpi_vol_awp_override, pref_pharm_list, ignore_columns = True)
            gpi_vol_awp_override = add_pharmacy_perf_names(gpi_vol_awp_override)
            gpi_vol_awp_override = add_target_ingcost(gpi_vol_awp_override, client_guarantees) 
            gpi_vol_awp_override.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.CURRENT_CONTRACT_DATA_FILE, index = False) 
#             cross_contract_utilization_check(gpi_vol_awp_df, contract_date_df)
    
    # setting flag to ensure projections are read and calculated correctly
    PROJ_READ = False
    
    if p.USE_PROJECTIONS_BQ and not p.FULL_YEAR and not p.CROSS_CONTRACT_PROJ and not p.INCLUDE_PLAN_LIABILITY:
        
        try:
            print("Inside BQ projections block...")

            if p.UNC_OPT:
                print("Running WITH UNC OPT...")
                # Client has UNC, reading both actuals and UC + Actuals in separate dataframes
                eoy_proj_df_full = read_eoy_proj(READ_FROM_BQ = p.READ_FROM_BQ)
                eoy_proj_df = eoy_proj_df_full[eoy_proj_df_full['UNC_INCLUDED'] == 1].copy()
                
                assert len(eoy_proj_df) > 0, " No rows loaded in eoy_proj_df "
                eoy_proj_df = calculate_proj(eoy_proj_df, lag_days, eoy_days)
                
                eoy_proj_df_actual = eoy_proj_df_full[eoy_proj_df_full['UNC_INCLUDED'] == 0].copy()
                
                assert len(eoy_proj_df_actual) > 0, " No rows loaded in eoy_proj_df_actual "
                eoy_proj_df_actual = calculate_proj(eoy_proj_df_actual, lag_days, eoy_days)
            else:
                print("Running WITHOUT UNC OPT...")
                eoy_proj_df = read_eoy_proj(READ_FROM_BQ = p.READ_FROM_BQ) 
                assert len(eoy_proj_df) > 0, " No rows loaded in eoy_proj_df"
                eoy_proj_df = calculate_proj(eoy_proj_df, lag_days, eoy_days)
                
            PROJ_READ = True
            print("Projections data load and transformation done...")

        except Exception as e:
            print(f"Using Projections from BQ tables failed with error - {e}")
                      
    # falls back to legacy method of calculating projections incase projections parameter is false or the new method fails to fetch projections correctly
    if not PROJ_READ:  
        print("Inside projections legacy block...")
            
        (sample_awp_month,sample_awp_day,proj_type,days_in_months) = \
            get_sample_awp_data(gpi_vol_awp_df, end_day)

        if p.UNC_OPT:
            sample_awp_month_actual, sample_awp_day_actual,_,_ = get_sample_awp_data(gpi_vol_awp_df_actual, end_day)

        # Using this client list ensures coverage of all clients even when some clients do not appear in the no_unc dataframe.
        client_list = sample_awp_month.CLIENT.unique()
        eoy_proj_df = get_proj(sample_awp_month,sample_awp_day,days_in_months,proj_type,lag_days,eoy_days,client_list)

        if p.UNC_OPT:
            eoy_proj_df_actual = get_proj(
                sample_awp_month_actual,
                sample_awp_day_actual,days_in_months,proj_type,lag_days,eoy_days,client_list)
    
    if p.CROSS_CONTRACT_PROJ:
        gpi_vol_awp_df = merge_nadac_wac_netcostguarantee(gpi_vol_awp_df)
        if p.UNC_OPT:
            gpi_vol_awp_df_actual = merge_nadac_wac_netcostguarantee(gpi_vol_awp_df_actual)
            
    gpi_vol_awp_agg = add_current_awp_to_data(gpi_vol_awp_df,contract_date_df)

    if p.UNC_OPT:
        gpi_vol_awp_agg_actual = add_current_awp_to_data(
            gpi_vol_awp_df, contract_date_df, gpi_vol_awp_df_actual)
   
    gpi_vol_awp_agg_ytd = add_projections(gpi_vol_awp_agg, eoy_proj_df)

    if p.UNC_OPT:
        gpi_vol_awp_agg_ytd_actual = add_projections(
            gpi_vol_awp_agg_actual, eoy_proj_df_actual)
    
    full_guarantee_df = update_guarantees(gpi_vol_awp_agg_ytd, chain_region_mac_mapping) 
    
    lp_guarant = create_lp_guarant(
        full_guarantee_df, mac_list_df, gpi_vol_awp_agg_ytd,chain_region_mac_mapping) 
    
    if p.BRAND_OPT and p.TRUECOST_CLIENT:
        mac_list_df = mac_list_df[mac_list_df['PRICE']>0]
   
    if p.NO_MAIL:
        gpi_vol_awp_agg_ytd = gpi_vol_awp_agg_ytd[gpi_vol_awp_agg_ytd.MEASUREMENT != "M30"]
    lp_vol_mac_df = get_lp_vol_mac_df(
        lp_guarant, gpi_vol_awp_agg_ytd,chain_region_mac_mapping)

    if p.UNC_OPT:
        if p.NO_MAIL:
            gpi_vol_awp_agg_ytd_actual = gpi_vol_awp_agg_ytd_actual[gpi_vol_awp_agg_ytd_actual.MEASUREMENT != "M30"]
        
        lp_vol_mac_df_actual = get_lp_vol_mac_df(
            lp_guarant, gpi_vol_awp_agg_ytd_actual,chain_region_mac_mapping)

    mac_list_gpi, mac_list_ndc = get_mac_list_gpi_ndc(mac_list_df)
    lp_vol_macprice_df = get_lp_vol_macprice_df(
        mac_list_gpi, mac_list_ndc,lp_vol_mac_df, NDC_UPDATE=p.NDC_UPDATE)

    if p.UNC_OPT:
        lp_vol_macprice_df_actual = get_lp_vol_macprice_df(
        mac_list_gpi, mac_list_ndc, lp_vol_mac_df_actual, NDC_UPDATE=p.NDC_UPDATE)

    lp_vol_ytd_gpi, lp_vol_ytd_ndc = get_lp_vol_ytd_gpi(
        lp_vol_macprice_df, mac_list_gpi, mac_list_ndc, pkg_size)

    if p.UNC_OPT:
        lp_vol_ytd_gpi_actual, lp_vol_ytd_ndc_actual = get_lp_vol_ytd_gpi(
            lp_vol_macprice_df_actual, mac_list_gpi, mac_list_ndc, pkg_size)

    lp_data_culled_df = get_lp_data_df(
        lp_vol_ytd_gpi, lp_vol_ytd_ndc, lp_vol_mac_df, lp_vol_macprice_df, 
        mac_list_df, mac_list_gpi,
        NDC_UPDATE=p.NDC_UPDATE, INCLUDE_PLAN_LIABILITY=p.INCLUDE_PLAN_LIABILITY)

    if p.UNC_OPT:
        lp_data_culled_df_actual = get_lp_data_df(
        lp_vol_ytd_gpi_actual, lp_vol_ytd_ndc_actual, lp_vol_mac_df, 
        lp_vol_macprice_df_actual, mac_list_df, mac_list_gpi, lp_vol_mac_df_actual,
        NDC_UPDATE=p.NDC_UPDATE, INCLUDE_PLAN_LIABILITY=p.INCLUDE_PLAN_LIABILITY,UNC_OPT=True)

    # For MEDD new year pricing: load in current year pricing to new year vcmls
    if p.CLIENT_TYPE == 'MEDD' and p.FULL_YEAR and not p.EGWP:
        lp_data_culled_df = update_new_mac_list(lp_data_culled_df)  
    
    lp_data_culled_df = add_mac_1026_to_lp_data_culled_df(
        lp_data_culled_df, mac_1026_gpi, mac_1026_ndc)
    if p.UNC_OPT:
        lp_data_culled_df_actual = add_mac_1026_to_lp_data_culled_df(
            lp_data_culled_df_actual, mac_1026_gpi, mac_1026_ndc)

    # lp_data_culled_df = add_goodrx_to_lp_data_culled_df(
    #     lp_data_culled_df,GOODRX_OPT=p.GOODRX_OPT) 
    
    lp_data_culled_df = merge_nadac_wac_netcostguarantee(lp_data_culled_df)
    if p.UNC_OPT:
        lp_data_culled_df_actual = merge_nadac_wac_netcostguarantee(lp_data_culled_df_actual)
    
    # Add prices used for benchmark price ceiling
    if p.APPLY_BENCHMARK_CAP:
        lp_data_culled_df_temp = pd.merge(lp_data_culled_df, benchmark_nadac_prices, how ='left', on = ['GPI_NDC', 'BG_FLAG'])
        assert len(lp_data_culled_df) == len(lp_data_culled_df_temp), "Duplicates added when merging benchmark price ceiling prices."
        lp_data_culled_df = lp_data_culled_df_temp
        del lp_data_culled_df_temp
    
    lp_data_culled_df = add_rms_to_lp_data_culled_df(
        lp_data_culled_df,RMS_OPT=p.RMS_OPT, APPLY_GENERAL_MULTIPLIER=p.APPLY_GENERAL_MULTIPLIER, APPLY_MAIL_MULTIPLIER=p.APPLY_MAIL_MULTIPLIER)

    lp_data_culled_df = add_zbd_to_lp_data_culled_df(
        lp_data_culled_df,ZBD_OPT=p.ZBD_OPT, cvs_ind_scalar = p.ZBD_CVS_IND_SCALAR, capped_scalar = p.ZBD_CAPPED_SCALAR, current_price_scalar = p.ZBD_CURRENT_PRICE_SCALAR)
    
    lp_data_culled_df = add_pharma_list(lp_data_culled_df, pref_pharm_list)
    
    if p.UNC_OPT:
        lp_data_culled_df_actual = add_pharma_list(
            lp_data_culled_df_actual, pref_pharm_list)

    lp_data_culled_df = add_non_mac_rate(lp_data_culled_df)

    if p.UNC_OPT:
        lp_data_culled_df_actual = add_non_mac_rate(lp_data_culled_df_actual)
        
    lp_data_culled_df = add_copay_coins_data(lp_data_culled_df)
    
    client_guarantees = read_client_guarantees(READ_FROM_BQ = p.READ_FROM_BQ)
    
    if p.UNC_OPT or p.INTERCEPTOR_OPT or p.COSTSAVER_CLIENT:        
        exclusions = read_exclusions(READ_FROM_BQ = p.READ_FROM_BQ)
        unc_exclusions = read_unc_exclusions(READ_FROM_BQ = p.READ_FROM_BQ)
    
    #Create effective mac_price with floors and caps for adjudication
    lp_data_culled_df['PRICE_REIMB_UNIT'] = \
        lp_data_culled_df.PRICE_REIMB / lp_data_culled_df.QTY
    lp_data_culled_df.loc[lp_data_culled_df['QTY'] == 0, 'PRICE_REIMB_UNIT'] = 0
    lp_data_culled_df['PHARM_PRICE_REIMB_UNIT'] = \
        lp_data_culled_df.PHARM_PRICE_REIMB / lp_data_culled_df.PHARM_QTY
    lp_data_culled_df.loc[lp_data_culled_df['PHARM_QTY'] == 0, 'PHARM_PRICE_REIMB_UNIT'] = 0
    if p.UNC_OPT:
        lp_data_culled_df_actual['PRICE_REIMB_UNIT'] = \
            lp_data_culled_df_actual.PRICE_REIMB / lp_data_culled_df_actual.QTY
        lp_data_culled_df_actual.loc[lp_data_culled_df_actual['QTY'] == 0, 'PRICE_REIMB_UNIT'] = 0
        lp_data_culled_df_actual['PHARM_PRICE_REIMB_UNIT'] = \
            lp_data_culled_df_actual.PHARM_PRICE_REIMB / lp_data_culled_df_actual.PHARM_QTY
        lp_data_culled_df_actual.loc[lp_data_culled_df_actual['PHARM_QTY'] == 0, 'PHARM_PRICE_REIMB_UNIT'] = 0

    # with new target ingredient cost calculations, pharmacy guarantees will be merged irrespective of UNC_OPT, INTERCEPTOR_OPT or COSTSAVER_CLIENT parameters
    # with new target ingredient cost calculations, 
    # pharmacy guarantees will be merged irrespective of UNC_OPT, INTERCEPTOR_OPT or COSTSAVER_CLIENT parameters
    pharmacy_guarantees = read_pharmacy_guarantees(READ_FROM_BQ = p.READ_FROM_BQ)
    
    lp_data_culled_df = pd.merge(
            lp_data_culled_df, pharmacy_guarantees, how ='left', 
            on = ['CLIENT', 'BREAKOUT','REGION','CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT','BG_FLAG'])
    if p.UNC_OPT:
        lp_data_culled_df_actual = pd.merge(
            lp_data_culled_df_actual, pharmacy_guarantees, how ='left', 
            on = ['CLIENT', 'BREAKOUT','REGION','CHAIN_GROUP','CHAIN_SUBGROUP','MEASUREMENT','BG_FLAG'])

    lp_data_culled_df = get_pharms_vcml_for_nmr(lp_data_culled_df)
    if p.UNC_OPT:
        lp_data_culled_df_actual = get_pharms_vcml_for_nmr(lp_data_culled_df_actual)

    lp_data_culled_df = compute_effective_prices(lp_data_culled_df)
    if p.UNC_OPT:
        lp_data_culled_df_actual = compute_effective_prices(lp_data_culled_df_actual)

    lp_data_culled_df = add_benchmark_price(lp_data_culled_df)
    
    if p.INTERCEPTOR_OPT or p.COSTSAVER_CLIENT:
        
        cs_pharm_guarantees = cs.read_cs_pharm_guarantees_from_bq()
        
        lp_data_culled_df = (lp_data_culled_df
                             .pipe(cs.prepare_marketplace_data, cs_pharm_guarantees)
                             .pipe(cs.add_fee_buffer_dataset)
                             .pipe(cs.add_interceptor_bounds, p.GPI_LOW_FAC))    
        old_len = len(lp_data_culled_df)
        
        # to be modified - this function requires CS_LM_PHARM_TARG_INGCOST_ADJ, which gets created later
        lp_data_culled_df = cs.get_zbd_fraction(lp_data_culled_df)
        assert len(lp_data_culled_df) == old_len, "len(lp_data_culled_df) == old_len"
        
        #This function recalculates the actual QTY, Claims, AWP based on keep/send logic
        lp_data_culled_df = cs.correct_costsaver_projections(lp_data_culled_df)
        
        if p.APPLY_BENCHMARK_CAP: 
            mask = (lp_data_culled_df.GOODRX_UPPER_LIMIT <= lp_data_culled_df.INTERCEPT_LOW)
            lp_data_culled_df.loc[mask, 'GOODRX_UPPER_LIMIT'] = (2 * lp_data_culled_df.loc[mask, "INTERCEPT_LOW"])
            
        if p.ZBD_OPT:
            mask = (lp_data_culled_df.ZBD_UPPER_LIMIT < lp_data_culled_df.INTERCEPT_LOW)
            lp_data_culled_df.loc[mask, 'ZBD_UPPER_LIMIT'] = (10000 * lp_data_culled_df.loc[mask, "CURRENT_MAC_PRICE"])    

    # For GPIs with no NADAC/WAC price, setting them temporarily to -> AVG_AWP/1.2
    lp_data_culled_df.loc[lp_data_culled_df['PHARM_AVG_NADAC'].isna(),'PHARM_AVG_NADAC'] = lp_data_culled_df['PHARM_AVG_AWP']/1.2
    assert len(lp_data_culled_df[lp_data_culled_df['PHARM_AVG_NADAC'].isna()])==0,"GPIs observed for which WAC price is 0"
    lp_data_culled_df.loc[lp_data_culled_df['PHARM_AVG_WAC'].isna(),'PHARM_AVG_WAC'] = lp_data_culled_df['PHARM_AVG_AWP']/1.2
    assert len(lp_data_culled_df[lp_data_culled_df['PHARM_AVG_WAC'].isna()])==0,"GPIs observed for which WAC price is 0"
    
    if p.UNC_OPT:  
        unc_percentiles = pd.concat([unc_GPI_percentiles, unc_NDC_percentiles])
        lp_data_culled_df = add_rule_base_logic(
            lp_data_culled_df, unc_percentiles, exclusions, unc_exclusions,
            client_guarantees, pharmacy_guarantees, pref_pharm_list, contract_date_df)

        lp_data_culled_df = prepare_output_data(lp_data_culled_df, lp_data_culled_df_actual)
        len(lp_data_culled_df[lp_data_culled_df.duplicated()])
    
    lp_data_culled_df = add_pharmacy_perf_names(lp_data_culled_df)
    if p.UNC_OPT:
        lp_data_culled_df_actual = add_pharmacy_perf_names(lp_data_culled_df_actual)
    
    # calculate target ingredient cost
    lp_data_culled_df = add_target_ingcost(lp_data_culled_df, client_guarantees)   
    lp_data_culled_df = add_disp_fee_avg(lp_data_culled_df)
    if p.UNC_OPT:
        lp_data_culled_df_actual = add_target_ingcost(lp_data_culled_df_actual, client_guarantees)
        lp_data_culled_df_actual = add_disp_fee_avg(lp_data_culled_df_actual)
        
    # check that CLIENT target and actual should be equal for non-truecost clients
    if not p.TRUECOST_CLIENT:
        assert (lp_data_culled_df['DISP_FEE'] == lp_data_culled_df['TARGET_DISP_FEE']).all(), 'Non-truecost clients target vs. actual disp fees need to be equal'
        if p.UNC_OPT:
            assert (lp_data_culled_df_actual['DISP_FEE'] == lp_data_culled_df_actual['TARGET_DISP_FEE']).all(), 'Non-truecost clients (UNC) target vs. actual disp fees need to be equal'
    
    # make price freeze
    if p.BRND_PRICE_FREEZE:
        lp_data_culled_df.loc[(lp_data_culled_df.BG_FLAG == 'B'),'PRICE_MUTABLE'] = 0
    if p.GNRC_PRICE_FREEZE:
        lp_data_culled_df.loc[(lp_data_culled_df.BG_FLAG == 'G'),'PRICE_MUTABLE'] = 0
            

            

    ########### save Outputs

    if p.UNC_OPT:
        save_data(lp_data_culled_df, mac_list_df, chain_region_mac_mapping, lp_data_culled_df_actual, WRITE_TO_BQ=False)
    else:
        save_data(lp_data_culled_df, mac_list_df, chain_region_mac_mapping, WRITE_TO_BQ=False)
    
    print('Total Used GPI-NDC Count: ', 
          lp_data_culled_df.loc[
              lp_data_culled_df['FULLAWP_ADJ']>0, 'GPI_NDC'].count())
    print('Real Mac Price Count: ', 
          lp_data_culled_df.loc[
              lp_data_culled_df['CURRENT_MAC_PRICE']>0, 'GPI_NDC'].count())
    print('Proportion Real Price: ', 
          lp_data_culled_df.loc[
              lp_data_culled_df['CURRENT_MAC_PRICE']>0, 'GPI_NDC'].count()/len(lp_data_culled_df))
    print('--------------------')
    

if __name__=="__main__":
    from CPMO_shared_functions import update_run_status
    try:
        update_run_status(i_error_type='Started Daily Input Read') 
        main()
    except Exception as e:
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Daily Input Read', repr(e), error_loc)
        raise e
