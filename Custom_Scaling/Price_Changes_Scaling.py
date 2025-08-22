# -*- coding: utf-8 -*-
"""
Created on Fri Mar 01 14:00:00 2024

@author: C732567
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datetime as dt
import pandas as pd
import numpy as np
import calendar
import copy
import time
import logging
from GER_LP_Code.CPMO_lp_functions import *
from GER_LP_Code.CPMO_shared_functions import *
from google.cloud import bigquery
import GER_LP_Code.BQ
import util_funcs as uf
import re
import openpyxl
import datetime as dt

# CHANGE WRITE_TO_BQ TO True only when ready to upload
WRITE_TO_BQ = False

# CHANGE THIS
# If client is in SPLIT list code will split the price changes into increases 
# and decreases and submit to BQ with two different run ids for each customer in the list
SPLIT = []



output_files = True
output_folder = 'Outputs/'
# CHANGE THIS
output_run_id = '_March_ScalingRuns' 

# CHANGE THIS 
# CHANGE THIS
customer_id_list= [
'C4','3063'
]
# Add cid if customer is MEDD
MEDD_client = ['3063']
# maximum ratio by which CVS prices can exceed Indy prices in case of MEDD clients
medd_cvs_over_indy_ratio = {
    '3063': 1.4
}



EFFDATE = '20240408' #pd.to_datetime('today').strftime("%Y%m%d")
TERMDATE = '20391231'



# Upper bound factor to prevent the super high increases .... adjust multiplier
upper_bound_factor = 1.5
# Lower bound factor to prevent the super high DECREASES  .... adjust multiplier
lower_bound_factor = 0.7

# Scale factor for Mail, R30, R90 provide the factor for each customer
# factor values to be greater than 0 and should be between upper_bound_factor and lower_bound_factor
# for instance factor of 1 means no change and 0.9 means 10% decrease
pcf_mail={ 
    "C4" : 1 + 0.33,
    "3063" : 1 + 0.2
            }

pcf_r30={
   "C4" : 1 + 0.00,
   "3063" : 1
        }

pcf_r90={
   "C4" : 1 + 0.00,
   "3063" : 1 + 0.2
        }



# List indicate if R90 VCML exists for this customer_ids
R90_exists=[
'3063','3768','C4'
]

# List of customer_ids that have state parity VCMLs
state_parity=[

]

# The state parity constraint parameters
# DIFFERENCE AS A MULTIPLIER--the default says CVS prices can be up to 50% higher or %35 lower than State parity prices.
PARITY_PRICE_DIFFERENCE_COLLAR_HIGH = 1.5
PARITY_PRICE_DIFFERENCE_COLLAR_LOW = 0.65


# List of excluded VCMLs that should not be changed and included in the price change file
excluded_VCML_list = ['10','12','13','SX','SX1','S3','S9', 'E23', 'N23','E92', '92', '24', 'E77', 'P77', '77', 
                      '78','88','A1','A2','A3','A4','A5','L1','L2','L3','L4','L5','LD','80','E80','M78']


# List of clients with restericted price increases
RESTRICTED_CLIENT_LIST_25_PERCENT_CAP = ["4951","4887","3266","5505","4195","C29","4093","4869","4158", "5072", "3217", "4164"]
# the multiplying percentage.... here it is 1.25 for restruicted clients
restricted_upper_bound_factor = 1.2499

MAIL_MAC_UNRESTRICTED = {
    '3063': False,
    '3768': False,
    'C4': False,
}
# If customer id is in this list and MAIL_MAC_UNRESTRICTED is False, code will raise retail prices to comply with mail-retail. 
# Otherwise and agian if MAIL_MAC_UNRESTRICTED is False it will drop drop mail prices to comply with mail-retail 
raise_retail_to_comply_Mail_vs_Retail = ['3063','3768']

# If customer id is in this list code will drop R90 to maintain R30-R90. Otherwise it will raise R30 to maintain R30-R90
drop_R90_to_comply_R30_vs_R90 = ['3063','3768']

## price change should usually be within lower_bound_factor and upper_bound_factor of the currentMAC price
dictionaries = [pcf_r30,pcf_r90,pcf_mail]


# Leave uncommented during most requests to avoid manual errors
for dictionary in dictionaries:
    for k,v in dictionary.items():
        if not (lower_bound_factor <= v <= upper_bound_factor):
            raise ValueError(f"Value out of bounds for key {k} : {v}")




bqclient = bigquery.Client()

def customer_id_finder(x,customer_id_list):

    for cid in customer_id_list: 
        if x.startswith(cid):
            return str(cid)
    return None

def generate_r30_r90_MAC_combo(customer_id, combined_pairs, OK_vcml = False):
    r30_r90_combo_macs = [['MAC' + customer_id + pair[0], 'MAC' + customer_id + pair[1]] for pair in combined_pairs]
    if OK_vcml:
        r30_r90_combo_OK = [['MAC' + customer_id + pair[0], 'MAC' + customer_id + 'OK'] for pair in combined_pairs]
        r30_r90_combo_macs += r30_r90_combo_OK
        
    return r30_r90_combo_macs

def get_latest_run_id():
        """
        Generates a unique run ID based on the current timestamp and a random integer.
        The run ID is composed of a prefix 'CSF', followed by the current date and time in the format
        'YYYYMMDDHHMMSSffffff' (Eastern Time), and a random three-digit integer.
        Returns:
            str: A unique run ID string.
        """
        import datetime as dt
        import random
        from pytz import timezone
        
        
        # Example: 'CSF20210615181014049595690'
        timestamp = dt.datetime.now(timezone('US/Eastern')).strftime('%Y%m%d%H%M%S%f')
      
        AT_RUN_ID = 'CSF' + timestamp + str(random.randint(100,999))
        
        return AT_RUN_ID

# Generate a unique run ID for each customer
AT_RUN_ID_DICT = {customer_id: get_latest_run_id() for customer_id in customer_id_list}

mac_list_file_og = bqclient.query(
    f"""select * from `anbc-prod.fdl_gdp_ae_ds_pro_lp_share_ent_prod.mac_list`
        where mac in (select vcml_id from `anbc-prod.fdl_gdp_ae_ds_pro_lp_share_ent_prod.vcml_reference` 
                      where customer_id in ("{'", "'.join(customer_id_list)}"))
     """
).to_dataframe()

mac_list_file_og=standardize_df(mac_list_file_og)
mac_list_file_og['MAC_LIST']=mac_list_file_og['MAC_LIST'].astype(str)
mac_list_file_og['CUSTOMER_ID']=mac_list_file_og.MAC_LIST.apply(lambda x:customer_id_finder(x,customer_id_list))




mac_list_file_og = mac_list_file_og[~mac_list_file_og['MAC'].isin(['MAC'+ c_id + s
                                                                     for c_id in customer_id_list 
                                                                     for s in excluded_VCML_list])]

                                                     
################################ MAC 1026 FILE ############################################
mac1026 = bqclient.query(
    f"""select *
from `anbc-prod.fdl_gdp_ae_ds_pro_lp_share_ent_prod.mac_1026`
     """
).to_dataframe()

mac1026=standardize_df(mac1026)
mac1026['GPI_NDC'] = mac1026['GPI']+'_'+mac1026['NDC']
mac1026['MAC'] = mac1026['MAC_LIST']
mac1026['MAC_LIST'] = mac1026['MAC_LIST'].str[3:]


# ### 5930 left ##########################

price_change_combined=pd.DataFrame(data=[],columns=['MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE',
                    'CurrentMAC', 'client_name','1026_PRICE','PRICE_MUTABLE','PRICE_CHANGE_REASON', 'MACPRC', 'AT_RUN_ID'])


indy_cvs={}
r90_r30={}
retail_mail={}
indy_1026={}
maxx_dict={}
minx_dict={}

R30_R90_Pairs = [
['16','45'] ,# AHD
['15','43'] ,# ALB
['15','45'] ,# ALB
['15','E5'] ,# ALB
['P5','E5'] ,# ALB
['P5','45'] ,# ALB
['P5','43'] ,# ALB
['C4','C9'] ,# CHD
['33','30'] ,# CRD
['P24','E24'] ,# CSC
['P24','47'] ,# CSC
['17','47'] ,# CSC
['17','E24'] ,# CSC
['M78','E78'] ,# CVS
['4','34'] ,# CVS
['4','91'] ,# CVS
['P78','E78'] ,# CVS
['P78','34'] ,# CVS
['P78','91'] ,# CVS
['41','91'] ,# CVS
['41','34'] ,# CVS
['41','E78'] ,# CVS
['4','E78'] ,# CVS
['M78','34'] ,# CVS
['M78','91'] ,# CVS 
['24','E78'] ,# CVS
['24','34'] ,# CVS
['24','91'] ,# CVS
['M78', '91SP'], #CVS
['24', '91SP'], #CVS
['P78', '91SP'], #CVS
['41', '91SP'], #CVS
['4', '91SP'], #CVS
['44','40'] ,# ELV
['55','50'] ,# EPC
['90','E90'] ,# GIA
['P93','E93'] ,# HAD
['11','20'] ,# HMA
['61','E61'] ,# HRT
['P48','E48'] ,# HVD
['H1','H3'] ,# HYV
['H1','E53'] ,# HYV
['H1','E96'] ,# HYV
['53','H3'] ,# HYV
['53','E53'] ,# HYV
['53','E96'] ,# HYV
['P96','H3'] ,# HYV
['P96','E53'] ,# HYV
['P96','E96'] ,# HYV
['65','E65'] ,# ING
['P40','38'] ,# KGR
['8','E40'] ,# KGR
['8','38'] ,# KGR
['P40','E40'] ,# KGR
['P58','E58'] ,# KIN
['54','E54'] ,# LWD
['54','E63'] ,# LWD
['P63','E54'] ,# LWD
['P63','E63'] ,# LWD
['177','377'] ,# MDP
['18','46'] ,# MJR
['18','48'] ,# MJR
['81','E81'] ,# MPB
['81','E94'] ,# MPB
['P95','E95'] ,# MPB
['P95','E94'] ,# MPB
['P94','E81'] ,# MPB
['P94','E95'] ,# MPB
['P94','E94'] ,# MPB
['P95','E81'] ,# MPB
['81','E95'] ,# MPB
['P38','E38'] ,# MPF
['P55','E55'] ,# MPI (IGD)
['28','E28'] ,# MPM
['P11','E11'] ,# MPW
['1','9'] ,# NAT
['1','3'] ,# NAT
['P88','E88'] ,# PCD
['P89','42'] ,# PUB
['P89','E67'] ,# PUB
['P89','E89'] ,# PUB
['67','E67'] ,# PUB
['14','42'] ,# PUB
['14','E67'] ,# PUB
['14','E89'] ,# PUB
['67','E89'] ,# PUB
['67','42'] ,# PUB
['62','E62'] ,# PUR
['6','36'] ,# RAD
['16','E91'] ,# RBS
['16','46'] ,# RBS
['P91','E91'] ,# RBS
['P91','46'] ,# RBS
['R1','R3'] ,# RUR
['XR','XT'] ,# STR
['XA','XT'] ,# STR
['59','E59'] ,# THF
['66','60'] ,# TPS
['29','E29'] ,# WAD
['5','35'] ,# WAG
['7','37'] ,# WMT
['P37','37'] ,# WMT
['7','E37'] ,# WMT
['P37','E37'] ,# WMT
['P57','E57'] ,# KSD: Still active on vcml reference table
['P49','E49'] ,# HBD: Still active on vcml reference table
['19','39'],# LEWIS : Still active on vcml reference table
['64', 'E64'], # AMZ
['227','322'], # ART
['229','322'], # ART
['22','3227'], # ART
['227','3227'], # ART
['229','3227'], # ART
['22','3229'], # ART
['227','3229'], # ART
['229','3229'], # ART
]
unique_vcmls_w_mapping = list(set([vcml for pair in R30_R90_Pairs for vcml in pair]))

for customer_id in customer_id_list:
    print(customer_id)
    unique_vcml_suffix = list(set(mac_list_file_og[mac_list_file_og['CUSTOMER_ID']==customer_id]['MAC_LIST'].str.replace(f"{customer_id}","").unique()))
    OK_vcml = 'OK' in unique_vcml_suffix
    R30_R90_combo = generate_r30_r90_MAC_combo(customer_id, R30_R90_Pairs, OK_vcml)
    
    missing_vcml_mapping = [vcml for vcml in unique_vcml_suffix if (vcml!='2' and vcml!='OK' and vcml not in unique_vcmls_w_mapping)]
    assert not missing_vcml_mapping, f"There are {len(missing_vcml_mapping)} VCMLs for {customer_id} that do not have a mapping in R30_R90_Pairs. Missing VCMLs: {missing_vcml_mapping}"

    

    ##################################################################### Price change file creation ################################

    mac_list_file=mac_list_file_og.loc[mac_list_file_og.CUSTOMER_ID==customer_id]
    price_changes_ori=mac_list_file.loc[mac_list_file.CUSTOMER_ID==customer_id]
    
    price_changes1=mac_list_file[:]
    price_changes1['CUSTOMER_ID']=price_changes1.MAC_LIST.apply(lambda x:customer_id_finder(x,customer_id_list))
    price_changes1['PRICE_MUTABLE']= 1
    price_changes1['PRICE_CHANGE_REASON']= ""
    
    price_changes1.rename(columns={'MAC':'MAC ID','PRICE':'CURRENT MAC'},inplace=True)
    price_changes1 = price_changes1[~(price_changes1['MAC ID']=='MAC'+customer_id+'12')]
    
    factor_r30=pcf_r30[customer_id]
    factor_r90=pcf_r90[customer_id]
    factor_mail=pcf_mail[customer_id]
    
    price_changes1['PROPOSED MAC']=price_changes1['CURRENT MAC']
    price_changes1.loc[price_changes1['MAC ID'].isin([r30mac for r30mac, r90mac in R30_R90_combo]),'PROPOSED MAC']=price_changes1.loc[price_changes1['MAC ID'].isin([r30mac for r30mac, r90mac in R30_R90_combo]), 'CURRENT MAC']*factor_r30
    price_changes1.loc[price_changes1['MAC ID'].isin([r30mac for r30mac, r90mac in R30_R90_combo]),'PRICE_CHANGE_REASON'] +=f"R30:{factor_r30}X,"
    
    price_changes1.loc[price_changes1['MAC ID'].isin([r90mac for r30mac, r90mac in R30_R90_combo]),'PROPOSED MAC']=price_changes1.loc[price_changes1['MAC ID'].isin([r90mac for r30mac, r90mac in R30_R90_combo]), 'CURRENT MAC']*factor_r90
    price_changes1.loc[price_changes1['MAC ID'].isin([r90mac for r30mac, r90mac in R30_R90_combo]),'PRICE_CHANGE_REASON']=f"R90:{factor_r90}X,"
    
    price_changes1.loc[price_changes1['MAC ID']=='MAC'+customer_id+'2','PROPOSED MAC']=price_changes1.loc[price_changes1['MAC ID']=='MAC'+customer_id+'2','CURRENT MAC']*factor_mail
    price_changes1.loc[price_changes1['MAC ID']=='MAC'+customer_id+'2','PRICE_CHANGE_REASON']=f"MAIL:{factor_mail}X,"

     
    price_changes=pd.concat([price_changes1]).reset_index(drop=True)

    non_ndc_gpis = price_changes[price_changes.NDC!='***********']['GPI'].unique()
    
    price_changes=price_changes.loc[~price_changes.GPI.isin(non_ndc_gpis)]
    price_changes.drop(columns=['MAC_LIST'],inplace=True)
    
    
    ######## Price changes shouldn't be affecting specialty GPIs #########################
    
    unique_gpi_len1=price_changes.GPI.nunique()
    wmt_price_overrides = uf.read_BQ_data(
        BQ.wmt_unc_override_custom.format(_customer_id = uf.get_formatted_string([customer_id]),
                                          _project = "pbm-mac-lp-prod-ai",
                                          _dataset = 'ds_sandbox',
                                          _table = "wmt_unc_override"),
        project_id =  "pbm-mac-lp-prod-ai",
        dataset_id = 'ds_sandbox',
        table_id = "wmt_unc_override",
        custom = True
    )
    mac_price_overrides = uf.read_BQ_data(
        BQ.ger_opt_mac_price_override_custom.format(_customer_id = uf.get_formatted_string([customer_id]),
                                                    _project = 'anbc-prod',
                                                    _landing_dataset = 'fdl_gdp_ae_ds_pro_lp_share_ent_prod',
                                                    _table_id = "ger_opt_mac_price_override"),
        project_id='anbc-prod',
        dataset_id='fdl_gdp_ae_ds_pro_lp_share_ent_prod',
        table_id="ger_opt_mac_price_override",
        custom = True
    )
    gpi_exclusions = uf.read_BQ_data(
        BQ.gpi_change_exclusion_ndc,
        project_id='anbc-prod',
        dataset_id='fdl_gdp_ae_ds_pro_lp_share_ent_prod',
        table_id='gpi_change_exclusion_ndc'
    )
    wmt_price_overrides = standardize_df(wmt_price_overrides)
    mac_price_overrides = standardize_df(mac_price_overrides)
    gpi_exclusions = standardize_df(gpi_exclusions).rename(columns={'GPI_CD': 'GPI'})

    # Removing specialty exclusion GPIs as we are not changing their prices
    price_changes = price_changes[~price_changes['GPI'].isin(gpi_exclusions['GPI'].unique())]
    
    price_override = pd.concat([mac_price_overrides, wmt_price_overrides])
    price_override['CLIENT'] = price_override['CLIENT'].astype(str)
    price_changes['CLIENT'] = price_changes['CUSTOMER_ID'].astype(str)
    price_changes.rename(columns={'MAC ID':'MAC','PROPOSED MAC':'CURRENT_MAC_PRICE'},inplace=True)
    price_changes['OLD_MAC_PRICE'] = price_changes['CURRENT MAC']

    price_changes.reset_index(drop=True, inplace=True)
    price_override.reset_index(drop=True, inplace=True)
    
    price_changes = price_overrider_function(price_override, price_changes)
    price_changes.rename(columns={'MAC':'MAC ID'},inplace=True)
    # price_overrider_function operates on the 'CURRENT_MAC_PRICE' column so the column need to be renamed to 'PROPOSED MAC'
    price_changes.rename(columns={'CURRENT_MAC_PRICE':'PROPOSED MAC'},inplace=True)
    price_changes['PROPOSED MAC'] = price_changes['PROPOSED MAC'].astype(float)
    price_changes.loc[price_changes['PRICE_MUTABLE']==0 ,'PRICE_CHANGE_REASON'] = 'Price Override'

    ################################ Place to check MAC1026 and other price relativity constraints and modify proposed price ############################
    
    ############## 1026 PRICE
    mac_1026_req=mac1026[['GPI','NDC','PRICE']].copy()
    mac_1026_req.rename(columns={'PRICE':'1026_PRICE'},inplace=True)
    price_changes=pd.merge(price_changes,mac_1026_req, how='left', on=['GPI','NDC'])

    price_changes.loc[(price_changes['MAC ID'].isin(['MAC'+customer_id+'2'])),'1026_PRICE'] = 0
    violations_1026 = price_changes.loc[((price_changes['PRICE_MUTABLE']==1))\
                                        & (price_changes['PROPOSED MAC'] < price_changes['1026_PRICE'])].reset_index(drop=True)
    
    if output_files:
        violations_1026.to_excel(output_folder + '1026_Violations_'+customer_id+'.xlsx', index=False)
    
    if os.path.exists(output_folder + '1026_Violations.xlsx'):
        with pd.ExcelWriter(output_folder + '1026_Violations.xlsx', engine="openpyxl",
                        mode='a', if_sheet_exists='replace') as writer:  
            violations_1026.to_excel(writer, sheet_name=customer_id)
    else:
        with pd.ExcelWriter(output_folder + '1026_Violations.xlsx', engine="openpyxl",
                        ) as writer:  
            violations_1026.to_excel(writer, sheet_name=customer_id)
    
    # Drop any price changes that would go below MAC1026
    # price_changes=price_changes.loc[(price_changes['PROPOSED MAC']>=price_changes['1026_PRICE'])].reset_index(drop=True)
    
    # ## We apply our mac1026 floor in the following line
    #price_changes['PROPOSED MAC']=price_changes[['PROPOSED MAC','1026_PRICE']].max(axis=1) 
    
    # Drop prices, but only to MAC1026
    # CHANGE THIS --- COMMENT OUT WHEN DOING A HARD CODING CHANGE   -- OTHERWISE LEAVE AS IS 
    #price_changes = price_changes.loc[~(price_changes['CURRENT MAC']<price_changes['1026_PRICE'])].reset_index(drop=True)
    
    price_changes.loc[(price_changes['PRICE_MUTABLE']==1) \
                      & (price_changes['PROPOSED MAC'] < price_changes['1026_PRICE'])\
                      ,'PRICE_CHANGE_REASON'] += 'MAC1026,'
    
    price_changes.loc[price_changes['PRICE_MUTABLE']==1, 'PROPOSED MAC']=price_changes[price_changes['PRICE_MUTABLE']==1][['PROPOSED MAC','1026_PRICE']].max(axis=1) 
    
    price_changes.groupby(['MAC ID']).count()
    
    
    ############## MAIL - RETAIL
    # CHANGE THIS adjust raise_retail_to_comply_Mail_vs_Retail list to choose how to fix mail-retail
    if not MAIL_MAC_UNRESTRICTED[customer_id]:
        if customer_id in raise_retail_to_comply_Mail_vs_Retail:
            # DO THIS BRANCH to raise retail prices to comply with mail-retail
            mail_prices=price_changes[:]
            mail_prices['CUSTOMER_ID']=mail_prices['MAC ID'].apply(lambda x:customer_id_finder(x[3:],customer_id_list))
            mail_prices=mail_prices.loc[(mail_prices['MAC ID']=='MAC'+customer_id+'2')]


            mail_prices_req=mail_prices[['CUSTOMER_ID','GPI','NDC','PROPOSED MAC']]
            mail_prices_req.rename(columns={'PROPOSED MAC':'MAIL_PRICE'},inplace=True)
            price_changes=pd.merge(price_changes,mail_prices_req, how='left', on=['CUSTOMER_ID','GPI','NDC'])

            ## the mail-retail constraint is mail prices have to be lower than 2.5 times retail price. This translates to retail prices have to be greater than 0.4 times mail price.
            ## for commercial this factor is 1
            price_changes['MAIL_PRICE']=1*price_changes['MAIL_PRICE']
            price_changes.fillna(0,inplace=True)

            violations_retail_over_mail = price_changes.loc[(price_changes['PRICE_MUTABLE']==1)&(price_changes['PROPOSED MAC']<price_changes['MAIL_PRICE'])].reset_index(drop=True)

            if output_files:
                violations_retail_over_mail.to_excel(output_folder + 'Retail_over_mail_Violations'+customer_id+'.xlsx')

            if os.path.exists(output_folder + 'Retail_over_mail_Violations.xlsx'):    
                with pd.ExcelWriter(output_folder + 'Retail_over_mail_Violations.xlsx', engine="openpyxl",
                                mode='a', if_sheet_exists='replace') as writer:  
                    violations_retail_over_mail.to_excel(writer, sheet_name=customer_id)
            else:
                with pd.ExcelWriter(output_folder + 'Retail_over_mail_Violations.xlsx', engine="openpyxl",
                                ) as writer:  
                    violations_retail_over_mail.to_excel(writer, sheet_name=customer_id)

            price_changes.loc[(price_changes['PRICE_MUTABLE']==1) \
                      & (price_changes['PROPOSED MAC']<price_changes['MAIL_PRICE'])\
                      ,'PRICE_CHANGE_REASON'] += 'MAIL_RETAIL,'
            price_changes.loc[price_changes['PRICE_MUTABLE']==1,'PROPOSED MAC']=price_changes.loc[price_changes['PRICE_MUTABLE']==1, ['PROPOSED MAC','MAIL_PRICE']].max(axis=1)
            # price_changes.groupby(['MAC ID']).count()

            # price_changes['PROPOSED MAC']=price_changes[['PROPOSED MAC','MAIL_PRICE']].max(axis=1) 
        
        else:
            # DO THIS BRANCH to drop mail prices to comply with mail-retail
            ########### MAIL - RETAIL Check if MAIL Increases are below RETAIL
            retail_min_prices=price_changes[:]
            retail_min_prices['CUSTOMER_ID']=retail_min_prices['MAC ID'].apply(lambda x:customer_id_finder(x[3:],customer_id_list))
            retail_min_prices=retail_min_prices.loc[~(retail_min_prices['MAC ID']=='MAC'+customer_id+'2')]

            retail_min_prices_req=retail_min_prices[['CUSTOMER_ID','GPI','NDC','PROPOSED MAC']].groupby(['CUSTOMER_ID','GPI','NDC']).min().reset_index()
            retail_min_prices_req.rename(columns={'PROPOSED MAC':'retail_min_PRICE'},inplace=True)
            price_changes=pd.merge(price_changes,retail_min_prices_req, how='left', on=['CUSTOMER_ID','GPI','NDC'])

            ## the mail-retail constraint is mail prices have to be lower than retail price. 
            price_changes['retail_min_PRICE']=1.0*price_changes['retail_min_PRICE']
            price_changes.loc[~(price_changes['MAC ID']=='MAC'+customer_id+'2'),'retail_min_PRICE']=10000000.0
            price_changes['retail_min_PRICE'].fillna(1000000000.0,inplace=True)
            price_changes['retail_min_PRICE']=price_changes['retail_min_PRICE'].astype(float)
            violations_retail_over_mail = price_changes.loc[(price_changes['PRICE_MUTABLE']==1)&(price_changes['PROPOSED MAC']>price_changes['retail_min_PRICE']+0.0001)].reset_index(drop=True)

            if output_files:
                violations_retail_over_mail.to_excel(output_folder + 'Retail_over_mail_Violations'+customer_id+'.xlsx')

            if os.path.exists(output_folder + 'Retail_over_mail_Violations.xlsx'):    
                with pd.ExcelWriter(output_folder + 'Retail_over_mail_Violations.xlsx', engine="openpyxl",
                                mode='a', if_sheet_exists='replace') as writer:  
                    violations_retail_over_mail.to_excel(writer, sheet_name=customer_id)
            else:
                with pd.ExcelWriter(output_folder + 'Retail_over_mail_Violations.xlsx', engine="openpyxl",
                                ) as writer:  
                    violations_retail_over_mail.to_excel(writer, sheet_name=customer_id)

            price_changes.loc[(price_changes['PRICE_MUTABLE']==1) \
                      & (price_changes['PROPOSED MAC']>price_changes['retail_min_PRICE']+0.0001)\
                      ,'PRICE_CHANGE_REASON'] += 'MAIL_RETAIL,'
            ##price_changes=price_changes.loc[(price_changes['PROPOSED MAC']<=price_changes['retail_min_PRICE'])].reset_index(drop=True)
            price_changes.loc[price_changes['PRICE_MUTABLE']==1,'PROPOSED MAC']=price_changes.loc[price_changes['PRICE_MUTABLE']==1 , ['PROPOSED MAC','retail_min_PRICE']].min(axis=1)
            print(price_changes[price_changes['GPI']=='57100010007505'][['MAC ID', 'PROPOSED MAC']], "post-mail-retail")
    else:
        print(f"{customer_id} is MAIL_MAC_UNRESTRICTED")

    ############################ Increase R30 price to match R90 price #############
    
    violations_R90aboveR30=pd.DataFrame(data=[])
    
    if customer_id in R90_exists:
        
        for j in R30_R90_combo:
            a=j[0]
            b=j[1]
            
            # CHANGE THIS Adjust drop_R90_to_comply_R30_vs_R90 list to pick how to fix r30-r90   
            if customer_id in drop_R90_to_comply_R30_vs_R90:
                # DO THIS BRANCH to drop R90 to maintain R30-R90
                a_prices=price_changes.loc[price_changes['MAC ID']==a,:].copy()
                if b in price_changes['MAC ID'].unique():
                    print("Used default")
                    a_prices.loc[:,'MAC ID'] = b
                elif a[-1] in ['5', '6', '7', '8'] and 'MAC'+customer_id+'9' in price_changes['MAC ID'].unique(): #R90CHAIN
                    print("Used R90CH")                
                    a_prices.loc[:,'MAC ID']='MAC'+customer_id+'9'
                else:
                    print("Used R90")
                    a_prices.loc[:,'MAC ID']='MAC'+customer_id+'3'  #R90

               

                a_prices.rename(columns={'PROPOSED MAC':'R30price'}, inplace=True)

                R30prices=pd.concat([a_prices]).reset_index(drop=True)



                price_changes=pd.merge(price_changes,R30prices[['MAC ID','R30price','GPI','NDC']],how='left',on=['MAC ID','GPI','NDC'])

                price_changes['R30price']=price_changes['R30price'].fillna(price_changes['PROPOSED MAC'])

                violations_R90aboveR30=pd.concat([violations_R90aboveR30,price_changes.loc[(price_changes['PRICE_MUTABLE']==1)&(price_changes['PROPOSED MAC']>price_changes['R30price'])]]).reset_index(drop=True)

                price_changes.loc[(price_changes['PRICE_MUTABLE']==1) \
                      & (price_changes['PROPOSED MAC']>price_changes['R30price'])\
                      ,'PRICE_CHANGE_REASON'] += 'R30_R90,'
                
                price_changes.loc[price_changes['PRICE_MUTABLE']==1,'PROPOSED MAC']=price_changes.loc[price_changes['PRICE_MUTABLE']==1,['PROPOSED MAC','R30price']].min(axis=1) 

                price_changes.drop(columns=['R30price'],inplace=True)
                
            else:
                # DO THIS BRANCH to raise R30 to maintain R30-R90
                # this logic probably needs to be updated for the full waterfall of r30/r90 prices
                # TODO: VCML suffix 9 is deprecated and each pharmacy has it's assiciated R30 and R90 
                # need to confirm if indy vcmls 3 should be used in else
                b_prices=price_changes.loc[price_changes['MAC ID']==b,:]
                if len(b_prices) == 0 and len(price_changes.loc[price_changes['MAC ID']==a,:])>0:
                    if re.match('.*'+customer_id+'?([0-9]+)$', a) \
                        and re.match('.*'+customer_id+'?([0-9]+)$', a).group(1) in ['5', '6', '7', '8'] \
                            and 'MAC'+customer_id+'9' in price_changes['MAC ID'].unique():
                        b_prices=price_changes.loc[price_changes['MAC ID']=='MAC'+customer_id+'9',:]
                    else:
                        b_prices=price_changes.loc[price_changes['MAC ID']=='MAC'+customer_id+'3',:]
                b_prices['MAC ID']=a
                b_prices.rename(columns={'PROPOSED MAC':'R90price'}, inplace=True)

                R90prices=pd.concat([b_prices]).reset_index(drop=True)


                price_changes=pd.merge(price_changes,b_prices[['MAC ID','R90price','GPI','NDC']],how='left',on=['MAC ID','GPI','NDC'])

                price_changes['R90price']=price_changes['R90price'].fillna(price_changes['PROPOSED MAC'])

                violations_R90aboveR30=pd.concat([violations_R90aboveR30,price_changes.loc[(price_changes['PRICE_MUTABLE']==1)&(price_changes['PROPOSED MAC']<price_changes['R90price'])]]).reset_index(drop=True)

                price_changes.loc[(price_changes['PRICE_MUTABLE']==1) \
                      & (price_changes['PROPOSED MAC']<price_changes['R90price'])\
                      ,'PRICE_CHANGE_REASON'] += 'R30_R90,'
                
                price_changes.loc[price_changes['PRICE_MUTABLE']==1,'PROPOSED MAC']=price_changes.loc[price_changes['PRICE_MUTABLE']==1,['PROPOSED MAC','R90price']].max(axis=1) 

                price_changes.drop(columns=['R90price'],inplace=True)

        
        if output_files:
            violations_R90aboveR30.to_excel(output_folder + 'R90aboveR30_'+customer_id+'.xlsx')
        
        if os.path.exists(output_folder + 'R90aboveR30_Violation.xlsx'):     
            with pd.ExcelWriter(output_folder + 'R90aboveR30_Violation.xlsx', engine="openpyxl",
                        mode='a', if_sheet_exists='replace') as writer:  
                violations_R90aboveR30.to_excel(writer, sheet_name=customer_id)
        else:
            with pd.ExcelWriter(output_folder + 'R90aboveR30_Violation.xlsx', engine="openpyxl",
                        ) as writer:  
                violations_R90aboveR30.to_excel(writer, sheet_name=customer_id)
    print(price_changes[price_changes['GPI']=='57100010007505'][['MAC ID', 'PROPOSED MAC']], "post-r30r90")



    # Fix to the super high increase in macprc .... adjust multiplier
    # price_changes['PRICE_CHANGE_REASON'] = price_changes['PRICE_CHANGE_REASON'].fillna('')
    price_changes.loc[(price_changes['PRICE_MUTABLE']==1) & (price_changes['PROPOSED MAC']> price_changes["CURRENT MAC"] * upper_bound_factor)\
                      ,'PRICE_CHANGE_REASON'] += 'upper_bound_factor Cap,'
    price_changes.loc[price_changes['PRICE_MUTABLE'] == 1, "PROPOSED MAC"] = np.minimum(price_changes.loc[price_changes['PRICE_MUTABLE'] == 1, "PROPOSED MAC"],
                                                                                            price_changes.loc[price_changes['PRICE_MUTABLE'] == 1, "CURRENT MAC"] * upper_bound_factor
                                                                                        )
    
    price_changes.loc[(price_changes['PRICE_MUTABLE']==1) & (price_changes['PROPOSED MAC']< price_changes["CURRENT MAC"] * lower_bound_factor)\
                      ,'PRICE_CHANGE_REASON'] += 'lower_bound_factor Cap,'
    # Fix to the super high DECREASES in macprc .... adjust multiplier
    price_changes.loc[price_changes['PRICE_MUTABLE'] == 1, "PROPOSED MAC"] = np.maximum(price_changes.loc[price_changes['PRICE_MUTABLE'] == 1, "PROPOSED MAC"],
                                                                                            price_changes.loc[price_changes['PRICE_MUTABLE'] == 1, "CURRENT MAC"] * lower_bound_factor
                                                                                    )


    ############## CVS Indepedents (check designed to cap CVS price increases.)
    
    # Ensure parity compliance on r30
    price_match_prices = price_changes[:]
    if customer_id in MEDD_client:
        # vcmls suffixes to be excluded from parity chekc, parity will be done against indys + preffered 
        capped_pharm_30_suffix = ['2','C4','15','16','90','61','H1','53','P96','P55','P40','8','54','P63','18','14','6','59','5','7','P37']

        price_match_prices = price_match_prices.loc[(~price_match_prices['MAC ID'].isin([r90 for r30, r90 in R30_R90_combo])) & 
                                                    (~price_match_prices['MAC ID'].isin(['MAC'+ customer_id + suffix for suffix in capped_pharm_30_suffix]))]
    else:
        price_match_prices = price_match_prices.loc[(~price_match_prices['MAC ID'].isin([r90 for r30, r90 in R30_R90_combo])) 
                                                    & (price_match_prices['MAC ID']!='MAC'+customer_id+'2') 
                                                    & (price_match_prices['MAC ID']!='MAC'+customer_id+'XR') 
                                                    & (price_match_prices['MAC ID']!='MAC'+customer_id+'XA') 
                                                   ]


    price_match_prices = price_match_prices.loc[~price_match_prices['MAC ID'].isin(['MAC'+customer_id+'41', 'MAC'+customer_id+'4'])]
    price_match_prices = price_match_prices.groupby(['CUSTOMER_ID', 'GPI', 'NDC'], as_index=False)['PROPOSED MAC'].min().rename(columns={'PROPOSED MAC': 'PROPOSED MAC_MINRETAIL'})
    price_changes = price_changes.merge(price_match_prices, how='left', on=['CUSTOMER_ID', 'GPI', 'NDC'])

    if customer_id in state_parity:
        price_changes.loc[price_changes['MAC ID']!='MAC'+customer_id+'41', 'PROPOSED MAC_MINRETAIL'] = np.inf
    else:
        price_changes.loc[~price_changes['MAC ID'].isin(['MAC'+customer_id+s for s in ['4','P78','M78']])
                                        , 'PROPOSED MAC_MINRETAIL'] = np.inf


    if customer_id in MEDD_client:
        # TODO: Could this violate R90 vs R30 min that happens above?
        price_changes.loc[:, 'PROPOSED MAC_MINRETAIL'] *= medd_cvs_over_indy_ratio[customer_id]  # increasing upper bound of pharmacy to be (medd_cvs_over_indy_ratio ) times above independents

    if customer_id in R90_exists:
        price_match_prices = price_changes[:]

        if customer_id in MEDD_client:
            capped_pharm_90_suffix = ['2','C9','43','45','E5','E90','E61','H3','E53','E96','E55','38','E40','E54','E63','46','48','42','36','E59','35','37','E37']
            price_match_prices = price_match_prices.loc[price_match_prices['MAC ID'].isin([r90 for r30, r90 in R30_R90_combo])& 
                                                        (~price_match_prices['MAC ID'].isin(['MAC'+ customer_id + suffix for suffix in capped_pharm_90_suffix]))]
        else:
            price_match_prices = price_match_prices.loc[(price_match_prices['MAC ID'].isin([r90 for r30, r90 in R30_R90_combo]))
                                                         & (price_match_prices['MAC ID']!='MAC'+customer_id+'XT')]

        price_match_prices = price_match_prices.groupby(['CUSTOMER_ID', 'GPI', 'NDC'], as_index=False)['PROPOSED MAC'].min().rename(columns={'PROPOSED MAC': 'PROPOSED MAC_MINRETAIL_90'})
        
        price_changes = price_changes.merge(price_match_prices, how='left', on=['CUSTOMER_ID', 'GPI', 'NDC'])
        if customer_id not in state_parity:
            if 'MAC'+customer_id+'34' in price_changes['MAC ID'].unique() or 'MAC'+customer_id+'E78' in price_changes['MAC ID'].unique():
                price_changes.loc[~price_changes['MAC ID'].isin(['MAC'+customer_id+s for s in ['34','E78']])
                                        , 'PROPOSED MAC_MINRETAIL_90'] = np.inf
            else:
                #TODO: Code will price vcmls ending at 3 using the medd_cvs_over_indy_ratio  * MIN(Indys + PREF)
                price_changes.loc[price_changes['MAC ID']!='MAC'+customer_id+'3', 'PROPOSED MAC_MINRETAIL_90'] = np.inf
        else:
            if 'MAC'+customer_id+'91' in price_changes['MAC ID'].unique():
                price_changes.loc[price_changes['MAC ID']!='MAC'+customer_id+'91', 'PROPOSED MAC_MINRETAIL_90'] = np.inf
            elif 'MAC'+customer_id+'34' in price_changes['MAC ID'].unique():
                price_changes.loc[price_changes['MAC ID']!='MAC'+customer_id+'34', 'PROPOSED MAC_MINRETAIL_90'] = np.inf
            else:
                price_changes.loc[price_changes['MAC ID']!='MAC'+customer_id+'3', 'PROPOSED MAC_MINRETAIL_90'] = np.inf
        if customer_id in MEDD_client:
            price_changes.loc[:, 'PROPOSED MAC_MINRETAIL_90'] *= medd_cvs_over_indy_ratio[customer_id]  # increasing upper bound of pharmacy to be (medd_cvs_over_indy_ratio ) times above independents

        price_changes.loc[:, 'PROPOSED MAC_MINRETAIL'] = price_changes[['PROPOSED MAC_MINRETAIL', 'PROPOSED MAC_MINRETAIL_90']].min(axis=1)
        price_changes.loc[price_changes['PROPOSED MAC_MINRETAIL'].isna(), 'PROPOSED MAC_MINRETAIL'] = np.inf
        price_changes.loc[price_changes['PROPOSED MAC_MINRETAIL']==0, 'PROPOSED MAC_MINRETAIL'] = np.inf
    

   
    violations_retail_over_CVS = price_changes.loc[(price_changes['PRICE_MUTABLE']==1)\
        & (price_changes['PROPOSED MAC']>price_changes['PROPOSED MAC_MINRETAIL'])].reset_index(drop=True)
    
    price_changes.loc[(price_changes['PRICE_MUTABLE']==1)\
                        & (price_changes['PROPOSED MAC']>price_changes['PROPOSED MAC_MINRETAIL'])\
                      ,'PRICE_CHANGE_REASON'] += 'CVS_INDY,'
    
    price_changes.loc[price_changes['PRICE_MUTABLE']==1,'PROPOSED MAC'] = price_changes.loc[price_changes['PRICE_MUTABLE']==1, ['PROPOSED MAC', 'PROPOSED MAC_MINRETAIL']].min(axis=1) 

    price_changes.groupby(['MAC ID']).count()

    if output_files:
        violations_retail_over_CVS.to_excel(output_folder + 'Retail_over_CVS_Violations'+customer_id+'.xlsx')

    if os.path.exists(output_folder + 'Independents_over_CVS_Violations.xlsx'):        
        with pd.ExcelWriter(output_folder + 'Independents_over_CVS_Violations.xlsx', engine="openpyxl",
                        mode='a', if_sheet_exists='replace') as writer:  
            violations_retail_over_CVS.to_excel(writer, sheet_name=customer_id)
    else:
        with pd.ExcelWriter(output_folder + 'Independents_over_CVS_Violations.xlsx', engine="openpyxl",
                        ) as writer:  
            violations_retail_over_CVS.to_excel(writer, sheet_name=customer_id)
    
    
             
    ############## CVS vs CVSSP Price Collars

    if customer_id in state_parity:
        # Excluding CVS vcmls ending in 41, 4, 34, 91 (N)
        non_cvs_price_changes = price_changes.loc[~price_changes['MAC ID'].isin(['MAC'+customer_id+s for s in ['4','34','41','91','E78','P78']])]
        cvs_price_changes = price_changes.loc[price_changes['MAC ID'].isin(['MAC'+customer_id+s for s in ['4','34','41','91','E78','P78']])]

        # Create a mapping for the merge conditions
        merge_conditions = {
            '4': '41',
            'P78': '41',
            '34': '91',
            'E78': '91'
        }

        # Function to map MAC ID endings based on the conditions
        def map_mac_id(mac_id):
            for key, value in merge_conditions.items():
                if mac_id.endswith(key) and mac_id[:-len(key)].endswith(customer_id):
                    return mac_id[:-len(key)] + value
            return mac_id

        # Create a new column for the mapped MAC IDs
        cvs_price_changes['CVSSP_MAC_ID'] = cvs_price_changes['MAC ID'].apply(map_mac_id)

        # Perform the self-merge
        cvs_price_changes = cvs_price_changes.merge(
            cvs_price_changes[['MAC ID', 'GPI', 'NDC','GPI_NDC','PROPOSED MAC']],
            left_on=['CVSSP_MAC_ID','GPI', 'NDC','GPI_NDC'],
            right_on=['MAC ID', 'GPI', 'NDC','GPI_NDC'],
            suffixes=('', '_cvssp')
        )

        assert price_changes.shape[0] == non_cvs_price_changes.shape[0] + cvs_price_changes.shape[0] , "Merge Error"

        cvs_price_changes['cvs_cvssp_bound_low'] = PARITY_PRICE_DIFFERENCE_COLLAR_LOW * cvs_price_changes['PROPOSED MAC_cvssp']
        cvs_price_changes['cvs_cvssp_bound_high'] = PARITY_PRICE_DIFFERENCE_COLLAR_HIGH * cvs_price_changes['PROPOSED MAC_cvssp']

        cvs_price_changes.loc[(cvs_price_changes['PRICE_MUTABLE']==1)\
                        & (cvs_price_changes['PROPOSED MAC']> cvs_price_changes['cvs_cvssp_bound_high'])\
                        & ~np.isclose(cvs_price_changes['PROPOSED MAC'], cvs_price_changes['cvs_cvssp_bound_high'], atol=1e-10)
                      ,'PRICE_CHANGE_REASON'] += 'cvs_cvssp_bound_high,'
        cvs_price_changes.loc[cvs_price_changes['PRICE_MUTABLE']==1,'PROPOSED MAC'] = cvs_price_changes.loc[cvs_price_changes['PRICE_MUTABLE']==1, ['PROPOSED MAC', 'cvs_cvssp_bound_high']].min(axis=1)
        cvs_price_changes.loc[(cvs_price_changes['PRICE_MUTABLE']==1)\
                                & (cvs_price_changes['PROPOSED MAC']<cvs_price_changes['cvs_cvssp_bound_low'])\
                                & ~np.isclose(cvs_price_changes['PROPOSED MAC'], cvs_price_changes['cvs_cvssp_bound_high'], atol=1e-10)
                            ,'PRICE_CHANGE_REASON'] += 'cvs_cvssp_bound_low,' 
        cvs_price_changes.loc[cvs_price_changes['PRICE_MUTABLE']==1,'PROPOSED MAC'] = cvs_price_changes.loc[cvs_price_changes['PRICE_MUTABLE']==1, ['PROPOSED MAC', 'cvs_cvssp_bound_low']].max(axis=1) 

        cvs_price_changes = cvs_price_changes[price_changes.columns]
        
        price_changes = pd.concat([non_cvs_price_changes, cvs_price_changes]).reset_index(drop=True)           
    ########################################################################################################################                

    price_changes.CUSTOMER_ID=price_changes['MAC ID'].astype(str).apply(lambda x:customer_id_finder(x[3:],customer_id_list))
    
    price_changes['GPPC']='********'
    price_changes['NAME']=''
    price_changes['EFFDATE']= EFFDATE
    price_changes['TERMDATE']= TERMDATE
    price_changes.rename(columns={'NDC':'NDC11','PROPOSED MAC':'MACPRC','CURRENT MAC':'CurrentMAC','MAC ID':'MACLIST','CUSTOMER_ID':'client_name'}, inplace=True)
    # CHANGE THIS
    price_changes.loc[:,'AT_RUN_ID'] = AT_RUN_ID_DICT[customer_id]
    
    price_change_combined=pd.concat([price_change_combined,price_changes[['MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE',
                    'CurrentMAC', 'client_name','1026_PRICE', 'PRICE_MUTABLE','PRICE_CHANGE_REASON', 'MACPRC', 'AT_RUN_ID']]]).reset_index(drop=True)
    
    
    
    
    
    
    
    print(len(violations_retail_over_CVS),"Independent Prices over CVS Violation for",customer_id)
    print(len(violations_R90aboveR30),"R90 Prices over R30 Prices Violation for",customer_id)
    if not MAIL_MAC_UNRESTRICTED[customer_id]:
        print(len(violations_retail_over_mail),"Retail Prices over M30 Prices Violation for",customer_id)
    print(len(violations_1026),"Independent Prices below 1026 Prices Violation for",customer_id)
    
    
    indy_cvs[customer_id]=len(violations_retail_over_CVS)
    r90_r30[customer_id]=len(violations_R90aboveR30)
    if not MAIL_MAC_UNRESTRICTED[customer_id]:
        retail_mail[customer_id]=len(violations_retail_over_mail)
    indy_1026[customer_id]=len(violations_1026)
    
    
    
    price_changes['PRICE_DIFFERENCE']=(price_changes['MACPRC']-price_changes['CurrentMAC'])/price_changes['CurrentMAC']
    minx=price_changes[['PRICE_DIFFERENCE','client_name']].groupby(['client_name']).min()
    maxx=price_changes[['PRICE_DIFFERENCE','client_name']].groupby(['client_name']).max()
    
    maxx_dict[customer_id]=maxx.PRICE_DIFFERENCE[0]
    minx_dict[customer_id]=minx.PRICE_DIFFERENCE[0]
    
    print(minx,"Minimum Price Decrease Percentage")
    print(maxx,"Maximum Price Increase Percentage")
    print(price_changes[price_changes['GPI']=='57100010007505'][['MACLIST', 'MACPRC']], "Post all checks")
    


price_changes_ok=price_change_combined.loc[~price_change_combined.client_name.isna() &  ~price_change_combined.GPI.isna() & (price_change_combined['MACPRC']<10000000000.0) & (price_change_combined['MACLIST'].str[-2:]!='10')].reset_index(drop=True)



# CHANGE THIS - This filter is always ON Unless you need to do hardcoding changes using Ashim_Custom_Update. In that case,
# comment out this block. In normal cases, leave it un-commented out.
# price_changes_ok = price_changes_ok[price_changes_ok['MACPRC']!=price_changes_ok['CurrentMAC']]



# Fix to the super high increase in macprc for Clients with WTW and AON in their name .... adjust multiplier
# This check might violate the CVS vs Indy parity check (Future enhancement to adjust CVS parity check to account for this)

price_changes_ok.loc[(price_changes_ok['PRICE_MUTABLE']==1)\
                    & (price_changes_ok.client_name.isin(RESTRICTED_CLIENT_LIST_25_PERCENT_CAP))\
                        & (price_changes_ok['MACPRC'] > price_changes_ok["CurrentMAC"] * restricted_upper_bound_factor)\
                      ,'PRICE_CHANGE_REASON'] += 'restricted_upper_bound_factor CAP,'

price_changes_ok.loc[(price_changes_ok['PRICE_MUTABLE']==1) & (price_changes_ok.client_name.isin(RESTRICTED_CLIENT_LIST_25_PERCENT_CAP)), "MACPRC"] = np.minimum(
                    price_changes_ok.loc[(price_changes_ok['PRICE_MUTABLE']==1) & (price_changes_ok.client_name.isin(RESTRICTED_CLIENT_LIST_25_PERCENT_CAP)), "MACPRC"],
                    price_changes_ok.loc[(price_changes_ok['PRICE_MUTABLE']==1) & (price_changes_ok.client_name.isin(RESTRICTED_CLIENT_LIST_25_PERCENT_CAP)), "CurrentMAC"] * restricted_upper_bound_factor
                )

price_changes_ok.loc[price_changes_ok['PRICE_MUTABLE']==1,'MACPRC'] = price_changes_ok.loc[price_changes_ok['PRICE_MUTABLE']==1,'MACPRC'].round(4)

#CHANGE THIS 

exclude_list = ["SX", "S3", "S9"]
price_changes_ok = price_changes_ok[~price_changes_ok["MACLIST"].str.endswith(tuple(exclude_list))]

price_changes_ok=price_changes_ok[price_changes_ok['CurrentMAC']!=price_changes_ok['MACPRC']]


####################### confirm prices are above 1026_PRICE #######################

price_changes_ok[(price_changes_ok['PRICE_MUTABLE']==1)\
                 & (price_changes_ok['MACPRC'] < price_changes_ok['1026_PRICE'])].to_csv(output_folder + 'Fixed - MACPRC<1026_PRICE.csv', index = False)

price_changes_ok.loc[(price_changes_ok['PRICE_MUTABLE']==1) \
                      & (price_changes_ok['MACPRC'] < price_changes_ok['1026_PRICE'])\
                      ,'PRICE_CHANGE_REASON'] += 'MAC1026,'
price_changes_ok.loc[price_changes_ok['PRICE_MUTABLE']==1,'MACPRC']=price_changes_ok.loc[price_changes_ok['PRICE_MUTABLE']==1,['MACPRC','1026_PRICE']].max(axis=1) 
price_changes_ok = price_changes_ok[['MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE', 'CurrentMAC', 'client_name','1026_PRICE','PRICE_MUTABLE','PRICE_CHANGE_REASON', 'MACPRC', 'AT_RUN_ID']]


####################### Save the Data #######################

for customer_id in customer_id_list: 
    

    if customer_id not in SPLIT:
        # print(f"AT_RUN_ID_{customer_id} : {price_changes_ok.loc[price_changes_ok['client_name'] == customer_id,'AT_RUN_ID'][0]}")
        print(f"AT_RUN_ID_{customer_id} : {price_changes_ok.loc[price_changes_ok['client_name'] == customer_id,'AT_RUN_ID'].iloc[0]}")

    else:
        price_changes_ok_inc = price_changes_ok.loc[(price_changes_ok['client_name'] == customer_id)&(price_changes_ok['MACPRC'] > price_changes_ok['CurrentMAC'])].copy()
        price_changes_ok_dec = price_changes_ok.loc[(price_changes_ok['client_name'] == customer_id)& (price_changes_ok['MACPRC'] <= price_changes_ok['CurrentMAC'])].copy()

        # CHANGE THIS
        # AT_RUN_ID
        increase_run_id = AT_RUN_ID_DICT[customer_id][:11] + 'INC' + AT_RUN_ID_DICT[customer_id][14:]
        decrease_run_id = AT_RUN_ID_DICT[customer_id][:11] + 'DEC' + AT_RUN_ID_DICT[customer_id][14:]
        
        if len(price_changes_ok_inc) > 0:
            price_changes_ok.loc[(price_changes_ok['client_name'] == customer_id)&(price_changes_ok['MACPRC'] > price_changes_ok['CurrentMAC']), 'AT_RUN_ID'] = increase_run_id
            price_changes_ok_inc.loc[:, 'AT_RUN_ID'] = increase_run_id
            price_changes_ok_inc.to_csv(f"{output_folder}price_changes_ok_inc_{customer_id}.csv", index = False)
            print(f"Increase_AT_RUN_ID_{customer_id} : {increase_run_id}")


        if len(price_changes_ok_dec) > 0:
            price_changes_ok.loc[(price_changes_ok['client_name'] == customer_id)&(price_changes_ok['MACPRC'] <= price_changes_ok['CurrentMAC']), 'AT_RUN_ID'] = decrease_run_id
            price_changes_ok_dec.loc[:, 'AT_RUN_ID'] = decrease_run_id
            price_changes_ok_dec.to_csv(f"{output_folder}price_changes_ok_dec_{customer_id}.csv", index = False)
            print(f"Decreases_AT_RUN_ID_{customer_id}: {decrease_run_id}")


# CHANGE THIS FILENAME
price_changes_ok[['MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE',
                        'CurrentMAC', 'client_name','1026_PRICE','PRICE_MUTABLE', 'PRICE_CHANGE_REASON', 'MACPRC', 'AT_RUN_ID']].to_csv(output_folder + 'Price_Changes_Scaling_'+pd.to_datetime('today').strftime("%Y%m%d") +'.csv', index=False)
           
price_changes_ok.to_csv(output_folder + "price_changes_ok.csv", index = False)
if WRITE_TO_BQ:
    uf.write_to_bq(price_changes_ok[['MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE',
                        'CurrentMAC', 'client_name', 'MACPRC', 'AT_RUN_ID']], 
                   project_output= 'pbm-mac-lp-prod-ai',
                    dataset_output= 'ds_production_lp',
                    table_id= 'LP_Price_Recomendations_Custom',
                    timestamp_param= dt.datetime.now().strftime('%Y-%m-%d_%H%M%S%f'),
                    run_id= None)
    print('LP_Price_Recomendations_Custom is updated')
else: 
    print('WARNING: LP_Price_Recomendations_Custom is not updated')
