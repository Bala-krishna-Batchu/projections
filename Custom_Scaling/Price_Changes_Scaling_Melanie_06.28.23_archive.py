# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 15:47:42 2023

@author: C247920
"""

import datetime as dt
import pandas as pd
import numpy as np
import calendar
import copy
import time
import logging
from CPMO_lp_functions import *
from CPMO_shared_functions import *
from google.cloud import bigquery
import BQ
import util_funcs as uf
import os

output_files=False
output_folder = 'Outputs/'
# CHANGE THIS
output_run_id = '_August_BSC'

# CHANGE THIS
# June 29 customer_id_list=['4589', '4459', '5971', '4568']
customer_id_list=['2704']
bqclient = bigquery.Client()

def customer_id_finder(x):
    # CHANGE THIS if len(customer_id)<4
    return x[:4]

mac_list_file_og = bqclient.query(
    f"""select * from `pbm-mac-lp-prod-de.ds_pro_lp.mac_list`
        where mac in (select vcml_id from `pbm-mac-lp-prod-de.ds_pro_lp.vcml_reference` 
                      where customer_id in ("{'", "'.join(customer_id_list)}"))
     """
).to_dataframe()

mac_list_file_og=standardize_df(mac_list_file_og)
mac_list_file_og['MAC_LIST']=mac_list_file_og['MAC_LIST'].astype(str)
mac_list_file_og['CUSTOMER_ID']=mac_list_file_og.MAC_LIST.apply(lambda x:customer_id_finder(x))
################################ MAC 1026 FILE ############################################
mac1026 = bqclient.query(
    f"""select *
from `pbm-mac-lp-prod-de.ds_pro_lp.mac_1026`
     """
).to_dataframe()

mac1026=standardize_df(mac1026)
mac1026['GPI_NDC'] = mac1026['GPI']+'_'+mac1026['NDC']
mac1026['MAC'] = mac1026['MAC_LIST']
mac1026['MAC_LIST'] = mac1026['MAC_LIST'].str[3:]

#### 5930 left ##########################

# CHANGE THIS
pcf_r30={'2704':1-0.245}
pcf_r90={'2704':1-0.245}

# CHANGE THIS
pcf_mail={'2704':1-0.245}

# CHANGE THIS
R90_exists=['2704']

# CHANGE THIS
state_parity=[]

price_change_combined=pd.DataFrame(data=[],columns=['MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE',
                    'CurrentMAC', 'client_name', 'MACPRC', 'AT_RUN_ID'])




indy_cvs={}
r90_r30={}
retail_mail={}
indy_1026={}
maxx_dict={}
minx_dict={}

for customer_id in customer_id_list:
    print(customer_id)
    R30_R90_combo=[['MAC'+customer_id+'1','MAC'+customer_id+'3'],
                   ['MAC'+customer_id+'4','MAC'+customer_id+'34'],
                   ['MAC'+customer_id+'5','MAC'+customer_id+'35'],
                   ['MAC'+customer_id+'6','MAC'+customer_id+'36'],
                   ['MAC'+customer_id+'7','MAC'+customer_id+'37'],
                   ['MAC'+customer_id+'8','MAC'+customer_id+'38'],
                   ['MAC'+customer_id+'4','MAC'+customer_id+'9'],
                   ['MAC'+customer_id+'8','MAC'+customer_id+'38'],
                   ['MAC'+customer_id+'16','MAC'+customer_id+'45'],
                   ['MAC'+customer_id+'15','MAC'+customer_id+'43'],
                   ['MAC'+customer_id+'33','MAC'+customer_id+'30'],
                   ['MAC'+customer_id+'41','MAC'+customer_id+'91'],
                   ['MAC'+customer_id+'44','MAC'+customer_id+'40'],
                   ['MAC'+customer_id+'41','MAC'+customer_id+'91'],
                   ['MAC'+customer_id+'55','MAC'+customer_id+'50'],
                   ['MAC'+customer_id+'11','MAC'+customer_id+'20'],
                   ['MAC'+customer_id+'18','MAC'+customer_id+'46'],
                   ['MAC'+customer_id+'14','MAC'+customer_id+'42'],
                   ['MAC'+customer_id+'66','MAC'+customer_id+'60'],
                   ['MAC'+customer_id+'177','MAC'+customer_id+'737'],
                   ['MAC'+customer_id+'199','MAC'+customer_id+'399'],
                   ['MAC'+customer_id+'47','MAC'+customer_id+'347'],
                   ['MAC'+customer_id+'59','MAC'+customer_id+'359'],
                   ['MAC'+customer_id+'69','MAC'+customer_id+'369'],
                   ['MAC'+customer_id+'77','MAC'+customer_id+'377'],
                   ['MAC'+customer_id+'87','MAC'+customer_id+'387'],
                   ['MAC'+customer_id+'977','MAC'+customer_id+'397'],
                   ['MAC'+customer_id+'97','MAC'+customer_id+'39']]
    ##################################################################### Price change file creation ################################
    
    # mac_list_file=mac_list_file.loc[~mac_list_file.MAC_LIST.isin(['214510'])]
    mac_list_file=mac_list_file_og.loc[mac_list_file_og.CUSTOMER_ID==customer_id]
    price_changes_ori=mac_list_file.loc[mac_list_file.CUSTOMER_ID==customer_id]
    
    #price_changes_ori=price_changes_ori.loc[(price_changes_ori.MAC_LIST=='1LN28') | (price_changes_ori.MAC_LIST=='1LP28') 
     #                                       | (price_changes_ori.MAC_LIST=='1LP112') | (price_changes_ori.MAC_LIST=='1LN112') \
     #                                       | (price_changes_ori.MAC_LIST=='1LP113') | (price_changes_ori.MAC_LIST=='1LN113') 
     #                                       | (price_changes_ori.MAC_LIST=='1LP91') | (price_changes_ori.MAC_LIST=='1LN91') 
     #                                       | (price_changes_ori.MAC_LIST=='1L2') ]
    
    
    price_changes1=mac_list_file[:]
    price_changes1['CUSTOMER_ID']=price_changes1.MAC_LIST.apply(lambda x:customer_id_finder(x))
    
    price_changes1.rename(columns={'MAC':'MAC ID','PRICE':'CURRENT MAC'},inplace=True)
    # price_changes1['PROPOSED MAC']=price_changes1['CURRENT MAC']*1.0
    factor_r30=pcf_r30[customer_id]
    factor_r90=pcf_r90[customer_id]
    factor_mail=pcf_mail[customer_id]
    
    price_changes1['PROPOSED MAC']=price_changes1['CURRENT MAC']
    price_changes1.loc[price_changes1['MAC ID'].isin([r30mac for r30mac, r90mac in R30_R90_combo]),'PROPOSED MAC']=price_changes1.loc[price_changes1['MAC ID'].isin([r30mac for r30mac, r90mac in R30_R90_combo]), 'CURRENT MAC']*factor_r30
    price_changes1.loc[price_changes1['MAC ID'].isin([r90mac for r30mac, r90mac in R30_R90_combo]),'PROPOSED MAC']=price_changes1.loc[price_changes1['MAC ID'].isin([r90mac for r30mac, r90mac in R30_R90_combo]), 'CURRENT MAC']*factor_r90
    price_changes1.loc[price_changes1['MAC ID']=='MAC'+customer_id+'2','PROPOSED MAC']=price_changes1.loc[price_changes1['MAC ID']=='MAC'+customer_id+'2','CURRENT MAC']*factor_mail
    # price_changes2=mac_list_file[:]
    # price_changes2['CUSTOMER_ID']=price_changes2.MAC_LIST.apply(lambda x:customer_id_finder(x))
    # price_changes2=price_changes2.loc[(price_changes2.MAC_LIST=='1LP112') | (price_changes2.MAC_LIST=='1LN112')]
    
    # price_changes2.rename(columns={'MAC':'MAC ID','PRICE':'CURRENT MAC'},inplace=True)
    # price_changes2['PROPOSED MAC']=price_changes2['CURRENT MAC']*1.15
    
    # price_changes3=mac_list_file[:]
    # price_changes3['CUSTOMER_ID']=price_changes3.MAC_LIST.apply(lambda x:customer_id_finder(x))
    # price_changes3=price_changes3.loc[(price_changes3.MAC_LIST=='1L2')]
                                       
    # price_changes3.rename(columns={'MAC':'MAC ID','PRICE':'CURRENT MAC'},inplace=True)
    # price_changes3['PROPOSED MAC']=price_changes3['CURRENT MAC']*1.15
    
    
    price_changes=pd.concat([price_changes1]).reset_index(drop=True)

    # Leaving out Arete VCML     
    # price_changes=price_changes.loc[~price_changes['MAC ID'].isin(['MAC'+customer_id+'22'])].reset_index(drop=True)
    
    
    # price_changes.rename(columns={'MAC':'MAC ID','PRICE':'CURRENT MAC'},inplace=True)
    # price_changes['PROPOSED MAC']=price_changes['CURRENT MAC']*1.10
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
                                                    _project = 'pbm-mac-lp-prod-de',
                                                    _landing_dataset = 'ds_pro_lp',
                                                    _table_id = "ger_opt_mac_price_override"),
        project_id='pbm-mac-lp-prod-de',
        dataset_id='ds_pro_lp',
        table_id="ger_opt_mac_price_override",
        custom = True
    )
    gpi_exclusions = uf.read_BQ_data(
        BQ.gpi_change_exclusion_ndc,
        project_id='pbm-mac-lp-prod-de',
        dataset_id='ds_pro_lp',
        table_id='gpi_change_exclusion_ndc'
    )

    wmt_price_overrides = standardize_df(wmt_price_overrides)
    mac_price_overrides = standardize_df(mac_price_overrides)
    gpi_exclusions = standardize_df(gpi_exclusions).rename(columns={'GPI_CD': 'GPI'})
    exclude_gpi = pd.concat([mac_price_overrides, gpi_exclusions])
    exclude_gpi['CLIENT'] = exclude_gpi['CLIENT'].astype(str)
    exclude_gpi=exclude_gpi.loc[(exclude_gpi.CLIENT==str(customer_id))|(exclude_gpi.CLIENT=='ALL')|exclude_gpi.CLIENT.isna()]
    temp1=exclude_gpi.GPI.unique()
    temp2=price_changes.GPI.unique()
    not_excluded=[a for a in temp2 if a in temp1]
    problem_gpis=pd.DataFrame(data=not_excluded,columns=['GPI'])
    price_changes=price_changes.loc[~price_changes.GPI.isin(not_excluded)] 
    price_changes=price_changes.loc[(price_changes['MAC ID']!='MAC'+customer_id+'7') |  ~price_changes.GPI.isin(wmt_price_overrides[wmt_price_overrides['VCML_ID']=='MAC'+customer_id+'7']['GPI'].unique())] 
    price_changes=price_changes.loc[(price_changes['MAC ID']!='MAC'+customer_id+'5') |  ~price_changes.GPI.isin(wmt_price_overrides[wmt_price_overrides['VCML_ID']=='MAC'+customer_id+'5']['GPI'].unique())] 
    
    ################################ Place to check MAC1026 and other price relativity constraints and modify proposed price ############################
    
    ############## 1026 PRICE
    mac_1026_req=mac1026[['GPI','NDC','PRICE']]
    mac_1026_req.rename(columns={'PRICE':'1026_PRICE'},inplace=True)
    price_changes=pd.merge(price_changes,mac_1026_req, how='left', on=['GPI','NDC'])
    #price_changes.fillna(0,inplace=True)
    
    #price_changes.loc[(~price_changes['MAC ID'].isin(['MAC'+customer_id+'1'])),'1026_PRICE'] = 0
    price_changes.loc[(price_changes['MAC ID'].isin(['MAC'+customer_id+'2'])),'1026_PRICE'] = 0
    violations_1026 = price_changes.loc[(price_changes['PROPOSED MAC'] < price_changes['1026_PRICE'])].reset_index(drop=True)
    
    if output_files:
        violations_1026.to_excel(output_folder + '1026_Violations'+customer_id+'.xlsx', index=False)
    
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
    price_changes = price_changes.loc[~(price_changes['CURRENT MAC']<price_changes['1026_PRICE'])].reset_index(drop=True)
    price_changes['PROPOSED MAC']=price_changes[['PROPOSED MAC','1026_PRICE']].max(axis=1) 
    
    price_changes.groupby(['MAC ID']).count()
    
    ############## MAIL - RETAIL 
    # CHANGE THIS to choose how to fix mail-retail
    if customer_id in []:
        # DO THIS BRANCH to raise retail prices to comply with mail-retail
        mail_prices=price_changes[:]
        mail_prices['CUSTOMER_ID']=mail_prices['MAC ID'].apply(lambda x:customer_id_finder(x[3:]))
        mail_prices=mail_prices.loc[(mail_prices['MAC ID']=='MAC'+customer_id+'2')]


        mail_prices_req=mail_prices[['CUSTOMER_ID','GPI','NDC','PROPOSED MAC']]
        mail_prices_req.rename(columns={'PROPOSED MAC':'MAIL_PRICE'},inplace=True)
        price_changes=pd.merge(price_changes,mail_prices_req, how='left', on=['CUSTOMER_ID','GPI','NDC'])

        ## the mail-retail constraint is mail prices have to be lower than 2.5 times retail price. This translates to retail prices have to be greater than 0.4 times mail price.
        ## for commercial this factor is 1
        price_changes['MAIL_PRICE']=1*price_changes['MAIL_PRICE']
        price_changes.fillna(0,inplace=True)

        violations_retail_over_mail = price_changes.loc[(price_changes['PROPOSED MAC']<price_changes['MAIL_PRICE'])].reset_index(drop=True)

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

    
        price_changes['PROPOSED MAC']=price_changes[['PROPOSED MAC','MAIL_PRICE']].max(axis=1)
        # price_changes.groupby(['MAC ID']).count()

        # price_changes['PROPOSED MAC']=price_changes[['PROPOSED MAC','MAIL_PRICE']].max(axis=1) 
    
    else:
        # DO THIS BRANCH to drop mail prices to comply with mail-retail
        ########### MAIL - RETAIL Check if MAIL Increases are below RETAIL
        retail_min_prices=price_changes[:]
        retail_min_prices['CUSTOMER_ID']=retail_min_prices['MAC ID'].apply(lambda x:customer_id_finder(x[3:]))
        retail_min_prices=retail_min_prices.loc[~(retail_min_prices['MAC ID']=='MAC'+customer_id+'2')]

        retail_min_prices_req=retail_min_prices[['CUSTOMER_ID','GPI','NDC','PROPOSED MAC']].groupby(['CUSTOMER_ID','GPI','NDC']).min().reset_index()
        retail_min_prices_req.rename(columns={'PROPOSED MAC':'retail_min_PRICE'},inplace=True)
        price_changes=pd.merge(price_changes,retail_min_prices_req, how='left', on=['CUSTOMER_ID','GPI','NDC'])

        ## the mail-retail constraint is mail prices have to be lower than retail price. 
        price_changes['retail_min_PRICE']=1*price_changes['retail_min_PRICE']
        price_changes.loc[~(price_changes['MAC ID']=='MAC'+customer_id+'2'),'retail_min_PRICE']=10000000
        price_changes.fillna(1000000000,inplace=True)

        violations_retail_over_mail = price_changes.loc[(price_changes['PROPOSED MAC']>price_changes['retail_min_PRICE']+0.0001)].reset_index(drop=True)

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


        ##price_changes=price_changes.loc[(price_changes['PROPOSED MAC']<=price_changes['retail_min_PRICE'])].reset_index(drop=True)
        price_changes['PROPOSED MAC']=price_changes[['PROPOSED MAC','retail_min_PRICE']].min(axis=1)
    
    # ############## CVS Indepedents (check designed to cap CVS price increases.)
    
    # Ensure parity compliance on r30
    price_match_prices = price_changes[:]
    price_match_prices = price_match_prices.loc[price_match_prices['MAC ID'].isin([r30 for r30, r90 in R30_R90_combo])]
    price_match_prices = price_match_prices.loc[~price_match_prices['MAC ID'].isin(['MAC'+customer_id+'41', 'MAC'+customer_id+'4'])]
    price_match_prices = price_match_prices.groupby(['CUSTOMER_ID', 'GPI', 'NDC'], as_index=False)['PROPOSED MAC'].min().rename(columns={'PROPOSED MAC': 'PROPOSED MAC_MINRETAIL'})
    price_changes = price_changes.merge(price_match_prices, how='left', on=['CUSTOMER_ID', 'GPI', 'NDC'])
    if customer_id not in state_parity:
        price_changes.loc[price_changes['MAC ID']!='MAC'+customer_id+'41', 'PROPOSED MAC_MINRETAIL'] = np.inf
    else:
        price_changes.loc[price_changes['MAC ID']!='MAC'+customer_id+'4', 'PROPOSED MAC_MINRETAIL'] = np.inf
    if customer_id in R90_exists:
        price_match_prices = price_changes[:]
        price_match_prices = price_match_prices.loc[price_match_prices['MAC ID'].isin([r90 for r30, r90 in R30_R90_combo])]
#        price_match_prices = price_match_prices.loc[~price_match_prices['MAC ID'].isin(['MAC'+customer_id+'34', 'MAC'+customer_id+'91'])]
        price_match_prices = price_match_prices.groupby(['CUSTOMER_ID', 'GPI', 'NDC'], as_index=False)['PROPOSED MAC'].min().rename(columns={'PROPOSED MAC': 'PROPOSED MAC_MINRETAIL_90'})
        price_changes = price_changes.merge(price_match_prices, how='left', on=['CUSTOMER_ID', 'GPI', 'NDC'])
        if customer_id not in state_parity:
            if 'MAC'+customer_id+'34' in price_changes['MAC ID'].unique():
                price_changes.loc[price_changes['MAC ID']!='MAC'+customer_id+'34', 'PROPOSED MAC_MINRETAIL_90'] = np.inf
            else:
                price_changes.loc[price_changes['MAC ID']!='MAC'+customer_id+'3', 'PROPOSED MAC_MINRETAIL_90'] = np.inf
        else:
            if 'MAC'+customer_id+'91' in price_changes['MAC ID'].unique():
                price_changes.loc[price_changes['MAC ID']!='MAC'+customer_id+'91', 'PROPOSED MAC_MINRETAIL_90'] = np.inf
            elif 'MAC'+customer_id+'34' in price_changes['MAC ID'].unique():
                price_changes.loc[price_changes['MAC ID']!='MAC'+customer_id+'34', 'PROPOSED MAC_MINRETAIL_90'] = np.inf
            else:
                price_changes.loc[price_changes['MAC ID']!='MAC'+customer_id+'3', 'PROPOSED MAC_MINRETAIL_90'] = np.inf
        price_changes.loc['PROPOSED MAC_MINRETAIL'] = price_changes[['PROPOSED MAC_MINRETAIL', 'PROPOSED MAC_MINRETAIL_90']].min(axis=1)
        price_changes.loc[price_changes['PROPOSED MAC_MINRETAIL'].isna(), 'PROPOSED MAC_MINRETAIL'] = np.inf
        price_changes.loc[price_changes['PROPOSED MAC_MINRETAIL']==0, 'PROPOSED MAC_MINRETAIL'] = np.inf
        

        violations_retail_over_CVS = price_changes.loc[
            (price_changes['PROPOSED MAC']>price_changes['PROPOSED MAC_MINRETAIL'])].reset_index(drop=True)
        price_changes['PROPOSED MAC'] = price_changes[['PROPOSED MAC', 'PROPOSED MAC_MINRETAIL']].min(axis=1) 
        
        #price_changes=price_changes.loc[(price_changes['CURRENT MAC']>=price_changes['CVS_PRICE'])|(price_changes['MAC ID']=='MAC21452')].reset_index(drop=True)
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
    
        
    ############################ Increase R30 price to match R90 price #############
    
    violations_R90aboveR30=pd.DataFrame(data=[])
    
    if customer_id in R90_exists:
        
        for j in R30_R90_combo:
            a=j[0]
            b=j[1]
            
        #     b_prices=price_changes.loc[price_changes['MAC ID']==b,:]
        #     b_prices['MAC ID']=a
        #     b_prices.rename(columns={'PROPOSED MAC':'R90price'}, inplace=True)
            
        #     R90prices=pd.concat([b_prices]).reset_index(drop=True)
            
            
        #     price_changes=pd.merge(price_changes,b_prices[['MAC ID','R90price','GPI','NDC']],how='left',on=['MAC ID','GPI','NDC'])
            
        #     price_changes['R90price']=price_changes['R90price'].fillna(price_changes['PROPOSED MAC'])
            
        #     violations_R90aboveR30=pd.concat([violations_R90aboveR30,price_changes.loc[price_changes['PROPOSED MAC']<price_changes['R90price']]]).reset_index(drop=True)
           
        #     price_changes['PROPOSED MAC']=price_changes[['PROPOSED MAC','R90price']].max(axis=1) 
            
        #     price_changes.drop(columns=['R90price'],inplace=True)
        
        # if output_files:
        #     violations_R90aboveR30.to_excel('R90aboveR30_'+customer_id+'.xlsx')
        
        # if os.path.exists('R90aboveR30_Violation.xlsx'):     
        #     with pd.ExcelWriter('R90aboveR30_Violation.xlsx', engine="openpyxl",
        #                 mode='a', if_sheet_exists='replace') as writer:  
        #         violations_R90aboveR30.to_excel(writer, sheet_name=customer_id)
        # else:
        #     with pd.ExcelWriter('R90aboveR30_Violation.xlsx', engine="openpyxl",
        #                 ) as writer:  
        #         violations_R90aboveR30.to_excel(writer, sheet_name=customer_id)
            # CHANGE THIS to pick how to fix r30-r90   
            if customer_id in ['2704']:
                # DO THIS BRANCH to drop R90 to maintain R30-R90
                a_prices=price_changes.loc[price_changes['MAC ID']==a,:]
                if b in price_changes['MAC ID'].unique():
                    print("Used default")
                    a_prices['MAC ID']=b
                elif a[-1] in ['5', '6', '7', '8'] and 'MAC'+customer_id+'9' in price_changes['MAC ID'].unique():
                    print("Used R90CH")                
                    a_prices['MAC ID']='MAC'+customer_id+'9'
                else:
                    print("Used R90")
                    a_prices['MAC ID']='MAC'+customer_id+'3'
                a_prices.rename(columns={'PROPOSED MAC':'R30price'}, inplace=True)

                R30prices=pd.concat([a_prices]).reset_index(drop=True)


                price_changes=pd.merge(price_changes,a_prices[['MAC ID','R30price','GPI','NDC']],how='left',on=['MAC ID','GPI','NDC'])

                price_changes['R30price']=price_changes['R30price'].fillna(price_changes['PROPOSED MAC'])

                violations_R90aboveR30=pd.concat([violations_R90aboveR30,price_changes.loc[price_changes['PROPOSED MAC']>price_changes['R30price']]]).reset_index(drop=True)

                price_changes['PROPOSED MAC']=price_changes[['PROPOSED MAC','R30price']].min(axis=1) 

                price_changes.drop(columns=['R30price'],inplace=True)
            else:
                # DO THIS BRANCH to raise R30 to maintain R30-R90
                # this logic probably needs to be updated for the full waterfall of r30/r90 prices
                b_prices=price_changes.loc[price_changes['MAC ID']==b,:]
                b_prices['MAC ID']=a
                b_prices.rename(columns={'PROPOSED MAC':'R90price'}, inplace=True)

                R90prices=pd.concat([b_prices]).reset_index(drop=True)


                price_changes=pd.merge(price_changes,b_prices[['MAC ID','R90price','GPI','NDC']],how='left',on=['MAC ID','GPI','NDC'])

                price_changes['R90price']=price_changes['R90price'].fillna(price_changes['PROPOSED MAC'])

                violations_R90aboveR30=pd.concat([violations_R90aboveR30,price_changes.loc[price_changes['PROPOSED MAC']<price_changes['R90price']]]).reset_index(drop=True)

                price_changes['PROPOSED MAC']=price_changes[['PROPOSED MAC','R90price']].max(axis=1) 

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
    
    ########################################################################################################################
    
    # price_changes = pd.read_excel('Centene_MAPD_Proposed_Price_Changes_March.xlsx')
    # floors_used = pd.read_csv('lp_floors_overrides_used_CenteneMAPD_MEDD_032023.txt',sep='|')
    # floors_used.rename(columns={'#MACLIST':'MACLIST'}, inplace=True)
    
    
    # # price_changes = pd.merge(price_changes, floors_used[['MACLIST','GPI','Floor_Price']], on=['MACLIST','GPI'], how='left')
    
    # price_changes.isnull().sum()
    
    # #price_changes=price_changes.loc[price_changes.Floor_Price.isnull()].reset_index(drop=True)
    price_changes.CUSTOMER_ID=price_changes['MAC ID'].astype(str).apply(lambda x:customer_id_finder(x[3:]))
    
    price_changes['GPPC']='********'
    price_changes['NAME']=''
    price_changes['EFFDATE']='20232922'
    price_changes['TERMDATE']='20391231'
    price_changes.rename(columns={'NDC':'NDC11','PROPOSED MAC':'MACPRC','CURRENT MAC':'CurrentMAC','MAC ID':'MACLIST','CUSTOMER_ID':'client_name'}, inplace=True)
    # CHANGE THIS
    price_changes['AT_RUN_ID']='CS2023080200000000000'+customer_id
    
    
    price_change_combined=pd.concat([price_change_combined,price_changes[['MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE',
                    'CurrentMAC', 'client_name', 'MACPRC', 'AT_RUN_ID']]]).reset_index(drop=True)
    
    
    print(len(violations_retail_over_CVS),"Independent Prices over CVS Violation for",customer_id)
    print(len(violations_R90aboveR30),"R90 Prices over R30 Prices Violation for",customer_id)
    print(len(violations_retail_over_mail),"Retail Prices over M30 Prices Violation for",customer_id)
    print(len(violations_1026),"Independent Prices below 1026 Prices Violation for",customer_id)
    
    
    indy_cvs[customer_id]=len(violations_retail_over_CVS)
    r90_r30[customer_id]=len(violations_R90aboveR30)
    retail_mail[customer_id]=len(violations_retail_over_mail)
    indy_1026[customer_id]=len(violations_1026)
    
    
    
    price_changes['PRICE_DIFFERENCE']=(price_changes['MACPRC']-price_changes['CurrentMAC'])/price_changes['CurrentMAC']
    minx=price_changes[['PRICE_DIFFERENCE','client_name']].groupby(['client_name']).min()
    maxx=price_changes[['PRICE_DIFFERENCE','client_name']].groupby(['client_name']).max()
    
    maxx_dict[customer_id]=maxx.PRICE_DIFFERENCE[0]
    minx_dict[customer_id]=minx.PRICE_DIFFERENCE[0]
    
    print(minx,"Minimum Price Decrease Percentage")
    print(maxx,"Maximum Price Increase Percentage")
    
    # price_changes[['MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE',
    #                 'CurrentMAC', 'client_name', 'MACPRC', 'AT_RUN_ID']].to_excel('Price_Change_Gateway.xlsx')
    
    # price_changes[['MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE',
    #                 'CurrentMAC', 'client_name', 'MACPRC', 'AT_RUN_ID']].to_csv('Price_Change_Gateway.csv', index=False)
    
    # # # price_changes[['MACLIST','GPI','GPPC','NDC11','NAME','EFFDATE','TERMDATE','MACPRC','Current MAC']].to_excel('Centene_MAPD_Proposed_Price_Changes_March_floor_GPI_removed.xlsx',index=False)
    
    # # price_changes[['MACLIST','GPI','GPPC','NDC11','NAME','EFFDATE','TERMDATE','MACPRC','Current MAC']].to_csv('HF_MA_Price_Changes_20percent_increase.csv')
    
    
# price_change_combined=price_change_combined.loc[price_change_combined.MACLIST.isin(['MAC42956',
# 'MAC42957',
# 'MAC42955',
# 'MAC42951',
# 'MAC42954',
# 'MAC429555',
# 'MAC429566',
# 'MAC429533',
# 'MAC429511',
# 'MAC42952',
# 'MAC42958',
# 'MAC429544'])].reset_index(drop=True)

# price_change_combined=price_change_combined.loc[~price_change_combined.GPI.isin(['61400020100480','61400020100470'])]

# price_change_combined[['MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE',
#                     'CurrentMAC', 'client_name', 'MACPRC', 'AT_RUN_ID']].to_csv('Price_Changes_Scaling_June.csv', index=False)

# price_change_combined[['MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE',
#                     'CurrentMAC', 'client_name', 'MACPRC', 'AT_RUN_ID']].to_excel('Price_Changes_Scaling_June.xlsx', index=False)

# price_change_combined['PRICE_DIFFERENCE']=(price_change_combined['MACPRC']-price_change_combined['CurrentMAC'])/price_change_combined['CurrentMAC']
# minx=price_change_combined[['PRICE_DIFFERENCE','client_name']].groupby(['client_name']).min()
# maxx=price_change_combined[['PRICE_DIFFERENCE','client_name']].groupby(['client_name']).max()
    


# price_changes_4517=price_change_combined.loc[price_change_combined.AT_RUN_ID=='CS20230614000000000004517']
# price_changes_4517['PRICE_DIFFERENCE']=(price_changes_4517['MACPRC']-price_changes_4517['CurrentMAC'])/price_changes_4517['CurrentMAC']

# price_changes_4531=price_change_combined.loc[price_change_combined.AT_RUN_ID=='CS20230614000000000004531']
# price_changes_4531['PRICE_DIFFERENCE']=(price_changes_4531['MACPRC']-price_changes_4531['CurrentMAC'])/price_changes_4531['CurrentMAC']



price_changes_ok=price_change_combined.loc[~price_change_combined.client_name.isna() &  ~price_change_combined.GPI.isna() & (price_change_combined['MACPRC']<10000000000.0) & (price_change_combined['MACLIST'].str[-2:]!='10')].reset_index(drop=True)


price_changes_ok['MACPRC'] = price_changes_ok['MACPRC'].round(4)
price_changes_ok = price_changes_ok[price_changes_ok['MACPRC']!=price_changes_ok['CurrentMAC']]
# CHANGE THIS (filename)
price_changes_ok[['MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE',
                    'CurrentMAC', 'client_name', 'MACPRC', 'AT_RUN_ID']].to_csv(output_folder + 'Price_Changes_Scaling_August_02.csv', index=False)
import datetime as dt
# CHANGE THIS only when ready to upload
if False:
    uf.write_to_bq(price_changes_ok[['MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE',
                        'CurrentMAC', 'client_name', 'MACPRC', 'AT_RUN_ID']], 
                   project_output= 'pbm-mac-lp-prod-ai',
                    dataset_output= 'ds_production_lp',
                    table_id= 'LP_Price_Recomendations_Custom',
                    timestamp_param= dt.datetime.now().strftime('%Y-%m-%d_%H%M%S%f'),
                    run_id= None)



# price_changes_4531=price_change_combined.loc[price_change_combined.AT_RUN_ID=='CS20230614000000000004531']
# price_changes_4531['PRICE_DIFFERENCE']=(price_changes_4531['MACPRC']-price_changes_4531['CurrentMAC'])/price_changes_4531['CurrentMAC']

# price_changes_4517=price_change_combined.loc[price_change_combined.AT_RUN_ID=='CS20230614000000000004517']
# price_changes_4517['PRICE_DIFFERENCE']=(price_changes_4517['MACPRC']-price_changes_4517['CurrentMAC'])/price_changes_4517['CurrentMAC']

# price_changes_4568=price_change_combined.loc[price_change_combined.AT_RUN_ID=='CS20230614000000000004568']
# price_changes_4568['PRICE_DIFFERENCE']=(price_changes_4568['MACPRC']-price_changes_4568['CurrentMAC'])/price_changes_4568['CurrentMAC']
