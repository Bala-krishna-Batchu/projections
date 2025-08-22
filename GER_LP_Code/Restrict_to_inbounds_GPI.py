# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 09:15:38 2022

@author: C247920
"""

import CPMO_parameters as p

import datetime as dt
import pandas as pd
import numpy as np
import calendar
import copy
import time
import logging

from pulp import *
from CPMO_lp_functions import *
from CPMO_shared_functions import *
from CPMO_plan_liability import *

################ Outline of what the script does ##############################
# 1. Loads in output from the run with unsatisfactory results.
# 2. Checks for GPIs with constraint violations at the current MAC price and GPIs whose current prices are out of the LP bounds.
# 3. Loads in the specialty exclusion file, appends the list with problem GPIs so that they are set as immutable in the rerun.
# 4. Saves the utilization information for the problem GPIs into the output folder.

# Tip: To use this file to resolve infeasibility, save the lp dataframe before the optimizer starts as the total output and run this script.

###############################################################################


# 1.

lp_data_output_df=pd.read_csv(p.FILE_OUTPUT_PATH+p.TOTAL_OUTPUT)

lp_data_output_df.columns = map(str.upper, lp_data_output_df.columns)
lp_data_output_df.loc[lp_data_output_df.PRICE_MUTABLE==1].reset_index(drop=True)
colname='CURRENT_MAC_PRICE'


# 2.

# GPIs where current retail price is lower than the corresponding mail price
def price_compare(x):

    if x.MEASUREMENT.nunique() != 1:
        if ((x.MEASUREMENT == 'M30').any()) & (x.MEASUREMENT.str.match(pat = 'R').any()):
            if float(max(x[x.MEASUREMENT =='M30'][colname])) - float(min(x[x.MEASUREMENT !='M30'][colname])) >= 0.0001:
                return True
        else:
            return False
    else:
        return False

check_2 = lp_data_output_df.loc[lp_data_output_df.PRICE_MUTABLE==1].groupby(['CLIENT', 'REGION','GPI_NDC' ]).filter(lambda x: price_compare(x))
check_2 = check_2.reset_index(drop=True)
print(check_2.size,'Number of rows with Mail-Retail conflict')


# GPIs where current retail30 price is lower than the corresponding retail90 price
def mes_compare(x):

    if x.MEASUREMENT.nunique() != 1:
        if ((x.MEASUREMENT == 'R30').any()& (x.MEASUREMENT == 'R90').any()):
            if float(max(x[x.MEASUREMENT =='R90'][colname])) - float(min(x[x.MEASUREMENT =='R30'][colname])) >= 0.0001:
                return True
        else:
            return False
    else:
        return False

check_5 = lp_data_output_df.groupby(['CLIENT','REGION', 'CHAIN_GROUP', 'GPI_NDC' ]).filter(lambda x: mes_compare(x)) # Modify based on client
check_5 = check_5.reset_index()
print(check_5.size,'Number of rows with R30-R90 conflict')

# GPIs where current ind price is lower than the corresponding cvs price
def pharm_comp(x):
    ind_factor = 1
    if x.CHAIN_GROUP.nunique() != 1:
        if ((x.CHAIN_GROUP == 'CVS').any() & (x.CHAIN_GROUP == 'NONPREF_OTH').any()) :
            if (float(max(x[x.CHAIN_GROUP =='CVS'][colname]))*ind_factor - float(min(x[x.CHAIN_GROUP =='NONPREF_OTH'][colname])) >= 0.0001):
                return True
        if ((x.CHAIN_GROUP == 'CVS').any() & (x.CHAIN_GROUP == 'PREF_OTH').any()):
            if (float(max(x[x.CHAIN_GROUP =='CVS'][colname]))*ind_factor - float(min(x[x.CHAIN_GROUP =='PREF_OTH'][colname])) >= 0.0001):
                return True
    else:
        return False

check_3 = lp_data_output_df.loc[lp_data_output_df.PRICE_MUTABLE==1].groupby(['CLIENT', 'REGION', 'MEASUREMENT','GPI_NDC' ]).filter(lambda x: pharm_comp(x))
check_3.reset_index()
print(check_3.size,'Number of rows with CVS-Ind conflict')

# GPIs where current price at non-preferred pharamcies is lower than the corresponding preferred pharmacy price
INPUT_PATH = p.FILE_DYNAMIC_INPUT_PATH+p.PREFERRED_PHARM_FILE

pref_pharm = pd.read_csv(INPUT_PATH)

for column in ['BREAKOUT', 'MEASUREMENT', 'CHAIN_GROUP']:
    if column in pref_pharm.columns:
        pref_pharm[column] = pref_pharm[column].apply(lambda x: x.upper())
pref_pharm['PREF_PHARM'] = pref_pharm.PREF_PHARM.apply(lambda x: x.split(','))

def pref_nonpref(x, pref):
    if x.CHAIN_GROUP.nunique() != 1:
        x.flag = x.CHAIN_GROUP.apply(lambda x: 'PREF' if x in pref else 'NONPREF')
        #print((x.flag == 'PREF').any())
        if((x.flag == 'PREF').any() & (x.flag == 'NONPREF').any()):
            if float(max(x[x.flag == 'PREF'][colname])) - float(min(x[x.flag == 'NONPREF'][colname]))>= 0.0001:
                return True
        else:
            return False
    else:
        return False  

check_4 = pd.DataFrame()
for client in lp_data_output_df.CLIENT.unique():
    for region in lp_data_output_df.loc[(lp_data_output_df.CLIENT == client)& (lp_data_output_df.MEASUREMENT != 'M30'), 'REGION'].unique():
        preferred_chains_temp = pref_pharm.loc[(pref_pharm.CLIENT==client) & (pref_pharm.REGION==region), 'PREF_PHARM'].values
        preferred_chains = []
        for item in preferred_chains_temp:
            if item[0] not in ['none', 'None', 'NONE']:
                preferred_chains += list(item)
        print(preferred_chains, client, region)
        check4_temp = lp_data_output_df.loc[(lp_data_output_df.CLIENT == client) & (lp_data_output_df.REGION == region) & (lp_data_output_df.MEASUREMENT != 'M30'),:].groupby(['CLIENT', 'MEASUREMENT','GPI_NDC']).filter(lambda x: pref_nonpref(x, preferred_chains))
        check_4 = check_4.append(check4_temp, ignore_index = False)
print(check_4.size,'Number of rows with Pref-Npref conflict')


# Check to see if current prices for pharmacies on the same VCML have the same output price
vary_across_vcml = lp_data_output_df.groupby(['MAC_LIST', 'GPI_NDC'])[colname].nunique() != 1

num_prices_table = lp_data_output_df.groupby(['MAC_LIST', 'GPI_NDC'], as_index = False).agg({colname: 'nunique'}).rename(columns = {colname: 'Num_Prices'})
num_prices_table = lp_data_output_df.merge(num_prices_table, how = 'left', on = ['MAC_LIST', 'GPI_NDC'])
num_prices_table2 = num_prices_table.loc[(num_prices_table.Num_Prices > 1), :]
print(num_prices_table2.size,'Number of rows with VCML inconsistencies')

# GPIs where current price is out of LP bounds
if ( (~lp_data_output_df[colname].between(lp_data_output_df['LB']-0.0001,lp_data_output_df['UB']+0.0001)).sum() > 0):
    outofbounds = (~lp_data_output_df[colname].between(lp_data_output_df['LB']-0.0001,lp_data_output_df['UB']+0.0001)).sum()
    miss = lp_data_output_df[(~lp_data_output_df[colname].between(lp_data_output_df['LB']-0.0001,lp_data_output_df['UB']+0.0001))]
print(num_prices_table2.size,'Number of rows with current prices out of bounds')

all_errors=pd.concat([check_2,check_3,check_5,check_4,num_prices_table2,miss]).reset_index(drop=True)


# 3.
lp_mismatch=all_errors[all_errors.PRICE_MUTABLE==1].GPI.unique()

exclusion_file=pd.DataFrame(data=lp_mismatch,columns=['GPI'])
exclusion_file['CLIENT']='ALL'
exclusion_file['REGION']='ALL'

og_exclusion=pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH+p.SPECIALTY_EXCLUSION_FILE)
floor_gpi=pd.read_csv(p.FILE_INPUT_PATH+p.FLOOR_GPI_LIST)
exclusion_file=exclusion_file.loc[~exclusion_file.GPI.isin(floor_gpi.GPI.unique())].reset_index(drop=True)

# 4.
combined_exclusion=pd.concat([og_exclusion,exclusion_file]).reset_index(drop=True)
combined_exclusion.to_csv(p.FILE_DYNAMIC_INPUT_PATH+p.SPECIALTY_EXCLUSION_FILE)

Utilization_info=lp_data_output_df.loc[lp_data_output_df.GPI.isin(all_errors.GPI.unique())]
Utilization_info.to_csv(p.FILE_OUTPUT_PATH+'Problem_GPI_utilization_info.csv')
