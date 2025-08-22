import os
import pandas as pd
import datetime as dt
import socket

# The time label given to all output files and log files for the algorithm run
# This is also used as and ID for BigQuery runs (keep the format if writing to BQ)
TIMESTAMP = {{ TIMESTAMP | default("dt.datetime.now().strftime('%Y-%m-%d_%H%M%S%f')") }}
USER = {{ USER | default("socket.gethostname()") }}
# List of the customer ID's, this indicates what client(s) will be run through the LP algorithm
# ONLY CLIENTS WITH THE SAME STRUCTURE CAN BE RUN AT THE SAME TIME!
# STRUCTURE INCUDES (PHARMACY LISTS, TYPE OF CLIENT, CODE FLAGS, DATES, ...)
CUSTOMER_ID = {{ CUSTOMER_ID | default(False) }}
CLIENT_LOB = "{{ CLIENT_LOB | default('CMK') }}"

BQ_INPUT_PROJECT_ID = "{{ BQ_INPUT_PROJECT_ID | default('anbc-prod') }}"
BQ_INPUT_DATASET_ENT_CNFV_PROD = "{{ BQ_INPUT_DATASET_ENT_CNFV_PROD | default('fdl_ent_cnfv_prod') }}"
BQ_INPUT_DATASET_ENT_ENRV_PROD = "{{ BQ_INPUT_DATASET_ENT_ENRV_PROD | default('fdl_gdp_ae_ent_enrv_prod') }}"
if CLIENT_LOB == 'AETNA':
    BQ_INPUT_DATASET_DS_PRO_LP = "{{ BQ_INPUT_DATASET_DS_PRO_LP | default('fdl_gdp_ae_ent_enrv_prod') }}"
    BQ_INPUT_DATASET_SANDBOX = "{{ BQ_INPUT_DATASET_SANDBOX | default('fdl_gdp_ae_ent_enrv_prod') }}"
else:
    BQ_INPUT_DATASET_DS_PRO_LP = "{{ BQ_INPUT_DATASET_DS_PRO_LP | default('fdl_gdp_ae_ds_pro_lp_share_ent_prod') }}"
    BQ_INPUT_DATASET_SANDBOX = "{{ BQ_INPUT_DATASET_SANDBOX | default('ds_sandbox') }}"

# project output data are stored in (managed by EA)
BQ_OUTPUT_PROJECT_ID = "{{ BQ_OUTPUT_PROJECT_ID | default('pbm-mac-lp-prod-ai') }}" 
# should be changed to ds_production_lp for production runs
BQ_OUTPUT_DATASET = "{{ BQ_OUTPUT_DATASET | default('ds_development_lp') }}" 

# Identifiers for Aetna Datasets and table_ids
if CLIENT_LOB == 'AETNA':
    AETNA_DATASET_OUTPUT_SUFFIX = "{{ AETNA_DATASET_OUTPUT_SUFFIX | default('') }}"
    AETNA_TABLE_ID_PREFIX = "{{ AETNA_TABLE_ID_PREFIX | default('gms_aetna_') }}"
else:
    AETNA_DATASET_OUTPUT_SUFFIX = "{{ AETNA_DATASET_OUTPUT_SUFFIX | default('') }}"
    AETNA_TABLE_ID_PREFIX = "{{ AETNA_TABLE_ID_PREFIX | default('') }}"

WRITE_TO_BQ = {{ WRITE_TO_BQ | default(False) }}
READ_FROM_BQ = {{ READ_FROM_BQ | default(False) }}

# set to True to track this run in the client run tracking dashboard
TRACK_RUN = {{ TRACK_RUN | default(False) }}

# Audit Trail
AT_RUN_ID = "{{ AT_RUN_ID | default('') }}" # audit trail ID

#Batch Run ID
BATCH_ID = "{{ BATCH_ID | default('') }}"

# Set to True to get the client YTD projections directly read from BQ tables.
USE_PROJECTIONS_BQ = {{ USE_PROJECTIONS_BQ | default(True) }}

#================================================================
#================================================================
# MODIFY EACH RUN
#================================================================

# Name only used on the Tableau report.  Right now is a manual parameter but will need to be automated to read in multiple clients.
# Becasue the parameter is to be human readable it does not need to match exactly what one finds on the taxonomy tables. 
# For example 'State of GA'
CLIENT_NAME_TABLEAU = "{{ CLIENT_NAME_TABLEAU | default(False) }}"
# Scenario label that will be displayed, e.g., flat 24% increase
RUN_TYPE_TABLEAU = "{{ RUN_TYPE_TABLEAU | default('Default') }}"

# The identifier for all files that the LP code produces
# Only when saved to cloud storage or your laptop
DATA_ID = {{ DATA_ID | default("'test_{}_{}_{}'.format(USER, CUSTOMER_ID[0], TIMESTAMP)")}}

# The path to the different code inputs, outputs, and logs whether running on-prem or in the cloud
# Has to be changed to the personal path for each respective Data Scientist running the algorithm
PROGRAM_INPUT_PATH = "{{ PROGRAM_INPUT_PATH | default('C:\\Users') }}"
PROGRAM_OUTPUT_PATH = "{{ PROGRAM_OUTPUT_PATH | default('C:\\Users') }}"

# Path where user's credentials.py file is located (only needed if running on-prem)
CREDENTIALS_PATH = "{{ CREDENTIALS_PATH | default('C:/Users') }}"

# The type of client(s). It can either be COMMERCIAL or MEDD or MEDICAID
# Only one type of client can be run, however multiple clients of the same type can be run simultaneously
CLIENT_TYPE = "{{ CLIENT_TYPE | default(False) }}"

#Guarantee Category from client taxonomy
GUARANTEE_CATEGORY = "{{GUARANTEE_CATEGORY | default(False) }}"

# Parameter that tells us if client is EGWP
EGWP = {{ EGWP | default(False) }}

# The date of the last claims data pull is a 'str'
# With format 'yyyy-mm-dd'
DATA_START_DAY = "{{ DATA_START_DAY | default(False) }}"
LAST_DATA = {{ LAST_DATA | default(False) }}

# The date of when the price changes will go live. To be set in close coordination with the business side (IA Team)
GO_LIVE = {{ GO_LIVE | default(False) }}

# pharmacy claims and gen-launch table names in Teradata, get these table names from Aadil Patel
if DATA_START_DAY[:4] == '2020':
    PHARM_PERF_YTD_TABLE_NAME = 'SB_FINANCE_G2_STAGING.GER_OPT_PMCY_Perf_YTD_NEW'
    GEN_LAUNCH_TABLE_NAME = 'SB_FINANCE_G2_STAGING.GER_OPT_NTM_GEN_30NOV20'
if DATA_START_DAY[:4] == '2021':
    PHARM_PERF_YTD_TABLE_NAME = 'SB_FINANCE_G2_STAGING.LP_PMCY_PERF_2021_YTD'
    GEN_LAUNCH_TABLE_NAME = 'SB_FINANCE_G2_STAGING.LP_NTM_2021_GEN'
    TABLE_PHARAMCY_CHAIN_MAPPING = 'GER_OPT_VW_Pharmacy_Chain_Mapping'

# Parameter needed to access the SQL server, only needed since not all the data is on Teradata
# TODO: Once all information is on Teradata this parameter will not be used anymore
SQL_SERVER_ACCESS = {{ SQL_SERVER_ACCESS | default(True) }}

#=================================================================
# Parameter that reads generic data thorugh BQ, and LP will optimize generic price points, default is True
GENERIC_OPT = {{GENERIC_OPT | default(True)}}

#=================================================================
# Parameter that reads brands data thorugh BQ, and LP will optimize brand price points, default is False
BRAND_OPT = {{BRAND_OPT | default(False)}}

# List of capped, non capped, and PSAO pharmacies. Depending on type of client you are running through the LP,
# you will need to confirm with the business (IA Team and Pharmacy Team) and gain approval on
# which chain groups belong to which pharmacy list. Always check with the business when onboarding new clients,
# running prexisting clients during new contract year, and type of client you are running (Med D vs Commercial vs Medicaid)

# THE LIST OF PHARMACIES SHOULD INClUDE ALL THE PHARMACIES ON MEASUREMENT MAPPING AND SHOULD BE UPDATED ACCORDINGLY 
if not (GENERIC_OPT or BRAND_OPT):
    raise Exception(f'Either one of GENERIC_OPT or BRAND_OPT needs to be true to run the LP.')
    
BIG_CAPPED_PHARMACY_LIST = {{ BIG_CAPPED_PHARMACY_LIST | default({}) }}
SMALL_CAPPED_PHARMACY_LIST = {{ SMALL_CAPPED_PHARMACY_LIST | default({}) }}
NON_CAPPED_PHARMACY_LIST = {{ NON_CAPPED_PHARMACY_LIST | default({}) }}
COGS_PHARMACY_LIST = {{ COGS_PHARMACY_LIST | default({}) }}
PSAO_LIST = {{ PSAO_LIST | default({}) }}
    
if CLIENT_TYPE == 'MEDD':
    if BRAND_OPT:
        raise Exception(f'Brand Optimization cannot be run on MEDD currently')
    else: 
        BIG_CAPPED_PHARMACY_LIST = { 
            'GNRC': [] if not GENERIC_OPT else BIG_CAPPED_PHARMACY_LIST['GNRC'] if 'GNRC' in BIG_CAPPED_PHARMACY_LIST else ['CVS', 'RAD', 'WAG', 'KRG', 'WMT'] , 
            'BRND': [] if not BRAND_OPT else BIG_CAPPED_PHARMACY_LIST['BRND'] if 'BRND' in BIG_CAPPED_PHARMACY_LIST else ['CVS', 'RAD', 'WAG', 'KRG', 'WMT']
        }
        SMALL_CAPPED_PHARMACY_LIST = { 
            'GNRC': [] if not GENERIC_OPT else SMALL_CAPPED_PHARMACY_LIST['GNRC'] if 'GNRC' in SMALL_CAPPED_PHARMACY_LIST else ['MCHOICE'] , 
            'BRND': [] if not BRAND_OPT else SMALL_CAPPED_PHARMACY_LIST['BRND'] if 'BRND' in SMALL_CAPPED_PHARMACY_LIST else ['MCHOICE']
        }
        NON_CAPPED_PHARMACY_LIST = { 
            'GNRC': [] if not GENERIC_OPT else NON_CAPPED_PHARMACY_LIST['GNRC'] if 'GNRC' in NON_CAPPED_PHARMACY_LIST else ['NONPREF_OTH', 'PREF_OTH', 'ART', 'ABS', 'HYV', 'MJR', 'CST'] , 
            'BRND': [] if not BRAND_OPT else NON_CAPPED_PHARMACY_LIST['BRND'] if 'BRND' in NON_CAPPED_PHARMACY_LIST else ['NONPREF_OTH', 'PREF_OTH', 'ART', 'ABS', 'HYV', 'MJR', 'CST']
        }
        COGS_PHARMACY_LIST = { 
            'GNRC': [] if not GENERIC_OPT else COGS_PHARMACY_LIST['GNRC'] if 'GNRC' in COGS_PHARMACY_LIST else [] , 
            'BRND': [] if not BRAND_OPT else COGS_PHARMACY_LIST['BRND'] if 'BRND' in COGS_PHARMACY_LIST else []
        }
        PSAO_LIST = { 
            'GNRC': [] if not GENERIC_OPT else PSAO_LIST['GNRC'] if 'GNRC' in PSAO_LIST else ['ELE'] , 
            'BRND': [] if not BRAND_OPT else PSAO_LIST['BRND'] if 'BRND' in PSAO_LIST else ['ELE']
        }
elif (CLIENT_TYPE == 'COMMERCIAL' or CLIENT_TYPE == 'MEDICAID'):
    if CLIENT_TYPE == 'MEDICAID' and BRAND_OPT:
        raise Exception(f'Brand Optimization cannot be run on MEDICAID currently')
    else:
        BIG_CAPPED_PHARMACY_LIST = { 
            'GNRC': [] if not GENERIC_OPT else BIG_CAPPED_PHARMACY_LIST['GNRC'] if 'GNRC' in BIG_CAPPED_PHARMACY_LIST else ['CVS', 'RAD', 'WAG', 'KRG', 'WMT'], 
            'BRND': [] if not BRAND_OPT else BIG_CAPPED_PHARMACY_LIST['BRND'] if 'BRND' in BIG_CAPPED_PHARMACY_LIST else ['CVS', 'RAD', 'WAG', 'KRG', 'WMT']
        }
        SMALL_CAPPED_PHARMACY_LIST = { 
            'GNRC': [] if not GENERIC_OPT else SMALL_CAPPED_PHARMACY_LIST['GNRC'] if 'GNRC' in SMALL_CAPPED_PHARMACY_LIST else ['MCHOICE', 'THF', 'AMZ', 'HYV', 'KIN', 'ABS', 'PBX', 'AHD', 'GIE', 'MJR', 'GEN', 'TPS'] , 
            'BRND': [] if not BRAND_OPT else SMALL_CAPPED_PHARMACY_LIST['BRND'] if 'BRND' in SMALL_CAPPED_PHARMACY_LIST else ['MCHOICE', 'ABS', 'AHD', 'PBX', 'MJR', 'WGS', 'SMC', 'CST', 'KIN', 'GIE', 'HYV', 'TPM', 'SMR', 'ARX', 'WIS', 'GEN', 'BGY', 'DDM', 'MCY', 'MGM', 'PMA', 'GUA', 'FVW', 'BRI', 'AMZ', 'THF', 'TPS']
        }
        NON_CAPPED_PHARMACY_LIST = { 
            'GNRC': [] if not GENERIC_OPT else NON_CAPPED_PHARMACY_LIST['GNRC'] if 'GNRC' in NON_CAPPED_PHARMACY_LIST else ['WGS', 'TPM', 'RUR', 'SMC', 'GUA', 'SMR', 'CHD', 'MGM', 'DDM', 'BGY', 'WIS', 'CST', 'MCY', 'ARX', 'FVW', 'NONPREF_OTH', 'BRI', 'PMA'], 
            'BRND': [] if not BRAND_OPT else NON_CAPPED_PHARMACY_LIST['BRND'] if 'BRND' in NON_CAPPED_PHARMACY_LIST else ['CHD', 'NONPREF_OTH', 'RUR']
        }
        COGS_PHARMACY_LIST = { 
            'GNRC': [] if not GENERIC_OPT else COGS_PHARMACY_LIST['GNRC'] if 'GNRC' in COGS_PHARMACY_LIST else [] , 
            'BRND': [] if not BRAND_OPT else COGS_PHARMACY_LIST['BRND'] if 'BRND' in COGS_PHARMACY_LIST else [] 
        }
        PSAO_LIST = { 
            'GNRC': [] if not GENERIC_OPT else PSAO_LIST['GNRC'] if 'GNRC' in PSAO_LIST else ['ELE', 'EPC', 'HMA', 'CAR'], 
            'BRND': [] if not BRAND_OPT else PSAO_LIST['BRND'] if 'BRND' in PSAO_LIST else ['ELE', 'EPC', 'HMA', 'CAR']
        }
else:
    raise Exception(f'CLIENT_TYPE {CLIENT_TYPE} not supported in CPMO_parameters yet.')   

if not isinstance(BIG_CAPPED_PHARMACY_LIST, dict) or not isinstance(SMALL_CAPPED_PHARMACY_LIST, dict) or not isinstance(NON_CAPPED_PHARMACY_LIST, dict) or not isinstance(COGS_PHARMACY_LIST, dict) or not isinstance(PSAO_LIST, dict):
    raise Exception(f'All pharmacy lists must be split by GNRC and BRAND as a dictionary.')

required_keys = {'GNRC', 'BRND'}

for list_name, pharmacy_list in {
    'BIG_CAPPED_PHARMACY_LIST': BIG_CAPPED_PHARMACY_LIST,
    'SMALL_CAPPED_PHARMACY_LIST': SMALL_CAPPED_PHARMACY_LIST,
    'NON_CAPPED_PHARMACY_LIST': NON_CAPPED_PHARMACY_LIST,
    'COGS_PHARMACY_LIST': COGS_PHARMACY_LIST,
    'PSAO_LIST': PSAO_LIST
}.items():
    if not required_keys.issubset(pharmacy_list.keys()):
        raise Exception(f'{list_name} must contain both GNRC and BRND keys.')

# Pharmacies which we have agreements with
AGREEMENT_PHARMACY_LIST = {}
for key in BIG_CAPPED_PHARMACY_LIST.keys():
    AGREEMENT_PHARMACY_LIST[key] = BIG_CAPPED_PHARMACY_LIST[key] + SMALL_CAPPED_PHARMACY_LIST[key]
                           
# Treats PSAOs as NON_CAPPED_PHARMCY if true. Depends on the business what this parameter should be. 
# Notice: this parameter treats PSAOs as NON_CAPPED_PHARAMCYs if true in the LP and in Leakage and ZBD optimization
PSAO_TREATMENT = {{ PSAO_TREATMENT | default(True) }}
PHARMACY_LIST = {}
for key in BIG_CAPPED_PHARMACY_LIST.keys():
    if PSAO_TREATMENT:
        NON_CAPPED_PHARMACY_LIST[key] += PSAO_LIST[key]
    else:
        SMALL_CAPPED_PHARMACY_LIST[key]+= PSAO_LIST[key]
        AGREEMENT_PHARMACY_LIST[key] += PSAO_LIST[key]
    PHARMACY_LIST[key] = AGREEMENT_PHARMACY_LIST[key] + NON_CAPPED_PHARMACY_LIST[key] + COGS_PHARMACY_LIST[key]

# MCHOICE claims refer to mail claims by CVS Pharmacy the retailer
# This is compared to MAIL claims, which are mail claims by CVS Caremark the PBM  
# Code below indicates whether there are MCHOICE claims
HAS_MCHOICE = {{ HAS_MCHOICE | default(True) }}

# MEDD MCHOICE Target Rate
MEDD_MCHOICE_TARGET_RATE = {{ MEDD_MCHOICE_TARGET_RATE | default(0.8905) }}

# Parameter used to in the function round_to to round LP outputs down rather than up
ROUND_DOWN = {{ ROUND_DOWN | default(False) }}

# Set to True if run was infeasible. If set to True, the infeasible exclusion file should be placed in the Input file path.
HANDLE_INFEASIBLE = {{ HANDLE_INFEASIBLE | default(False) }}

# Set to True if run had confilct gpi. If set to True, the conflict gpi file should be placed in the Input file path.
HANDLE_CONFLICT_GPI = {{ HANDLE_CONFLICT_GPI | default(False) }}
# The flag modifies if the drugs are mutable or not.
if any(ext in CLIENT_NAME_TABLEAU.upper() for ext in ['AON', 'WTW','MVP','HEALTHFIRST']) or any(cid == CUSTOMER_ID for cid in ['183C','185C']):
    CONFLICT_GPI_AS_TIERS = False
else:   
    CONFLICT_GPI_AS_TIERS = {{ CONFLICT_GPI_AS_TIERS | default(False) }}
    
# The multiplier applied to the upper price bound for conflicting GPIs in the price tier DataFrame.
CONFLICT_GPI_AS_TIERS_BOUNDS = {{ CONFLICT_GPI_AS_TIERS_BOUNDS | default(1) }}

# the lower bound for the conflict GPI price tier
CONFLICT_GPI_LOW_BOUND = {{ CONFLICT_GPI_LOW_BOUND | default(0.8) }}

# When we run COGS optimization, this is our target (CVS performance - COGS pharmacy guarantee)
COGS_BUFFER = {{ COGS_BUFFER | default(-0.005) }}
FORCE_FLOOR = {{ FORCE_FLOOR | default(False) }}
FORCE_FLOOR_PHARMACY_SUBGROUP_LIST = {{ FORCE_FLOOR_PHARMACY_SUBGROUP_LIST | default([]) }}

# The state parity constraint parameters
# NOTE: move to business rules section and set defaults once pilot is completed
# difference AS A MULTIPLIER--the default says non-parity prices can be up to 25% higher or lower than parity prices.
PARITY_PRICE_DIFFERENCE_COLLAR_HIGH = {{ PARITY_PRICE_DIFFERENCE_COLLAR_HIGH | default(1.25) }} 
PARITY_PRICE_DIFFERENCE_COLLAR_LOW = {{ PARITY_PRICE_DIFFERENCE_COLLAR_LOW | default(0.75) }} 

#=================================================================
# Parameter to allow for TrueCost pricing. Includes, but not limited to, setting targets and benchmark price to a
# net cost guarantee instead of discount off of awp, and offsetting performance based on client dispensing fee targets
TRUECOST_CLIENT = {{TRUECOST_CLIENT | default(False)}}


#=================================================================
# Parameter to allow for UCL client pricing.
UCL_CLIENT = {{UCL_CLIENT | default(False)}}

#================================================================
#================================================================
# REGULARLY MODIFIED PARAMETERS AT THE REQUEST OF BUSINESS PARTNERS
#================================================================

# If True, the code will read an extra price override file. It will set the GPI
# to the price specified in the file and prevent further price modifications.
PRICE_OVERRIDE = {{ PRICE_OVERRIDE | default(False) }}
# output from pre-processing for price overrides
PRICE_OVERRIDE_FILE = {{ PRICE_OVERRIDE_FILE | default("'Price_Overrides_{}.csv'.format(DATA_ID)") }}  

# If True, the code will read FLOOR_GPI_LIST. A csv file with a list of GPIs that will get set to the MAC 1026 floor price
if TRUECOST_CLIENT:
    FLOOR_PRICE = {{ FLOOR_PRICE | default(False) }}
else:
    FLOOR_PRICE = {{ FLOOR_PRICE | default(True) }}
FLOOR_GPI_LIST = {{ FLOOR_GPI_LIST | default("'20201209_Floor_GPIs.csv'")}}

# If True, the client is a locked client and the coefficients will be set to 1 and 0
LOCKED_CLIENT = {{ LOCKED_CLIENT | default(False) }}

# If True, LP uses client side columns for pharmacy calculations for the Locked clients
TREAT_TRANSPARENT = {{ TREAT_TRANSPARENT | default(True) }}

# If True, adds an adjustment from the (UC_ADJUSTMENT) file to account for known overperformance due to U&C, can apply to any client(s)
# The file is derived form the model and lamda performance dictionnaries form the first client(s) run
UNC_ADJUST = {{ UNC_ADJUST | default(True) }}
UNC_ADJUSTMENT = {{ UNC_ADJUSTMENT | default("'UC_Adjustment_{}.csv'.format(DATA_ID)") }}

# If True, adds unc optimization preprocessing logic to the LP algorithm run, only for the unc initiative.
# Optimizes the re-allocation of MAC claims to UNC and vice versa.
UNC_OPT = {{ UNC_OPT | default(False) }}

# If True, optimize for exclusions for the client
UNC_CLIENT = {{ UNC_CLIENT | default(False) }}
# If True, optimize for exclusions for the pharmacy (can be simultaneous with the previous parameter)
UNC_PHARMACY = {{ UNC_PHARMACY | default(False) }}
if CLIENT_TYPE == 'MEDD':
    UNC_PHARMACY_CHAIN_GROUPS = {{ UNC_PHARMACY_CHAIN_GROUPS | default(['WMT'])}}
elif (CLIENT_TYPE == 'COMMERCIAL' or CLIENT_TYPE == 'MEDICAID'):
    UNC_PHARMACY_CHAIN_GROUPS = {{ UNC_PHARMACY_CHAIN_GROUPS | default(['WAG','WMT'])}}
# If True (the default), detects situations where the NEXT price change would be able to optimize for U&C--
# even if the current one will not
UNC_2X_OPTIMIZATION = {{ UNC_2X_OPTIMIZATION | default(True) }} 
    
# Penalize outlier U&C prices when selecting price targets. Higher numbers penalize outliers more.
UNC_OUTLIER_MULTIPLIER = {{ UNC_OUTLIER_MULTIPLIER | default(3) }}

# If True, UNC logic overrides GoodRx logic.  The is used by the price tier rules
UNC_OVERRIDE_GOODRX = {{ UNC_OVERRIDE_GOODRX | default(False) }}

# If True, implements the GoodRx Project Interceptor logic.
# The logic works with UNC_OPT and LEAKAGE_OPT but not with RMS_OPT and APPLY_GENERAL_MULTIPLIER
INTERCEPTOR_OPT = {{ INTERCEPTOR_OPT | default(False) }}

#If True, indicates a client on costsaver program
COSTSAVER_CLIENT = {{ COSTSAVER_CLIENT | default(False) }}

# Set level to select how mail prices are handled due to conflicts with retail interceptor logic. 
# Level 1: Drop Mail Prices to accommodate all the retail intercept_high and maintain the keep/send logic
# Level 2: Push the intercept_high bounds for conflicting retail to min non conflicting retail_high
# Level 3: Push retail intercept_high to inf for all conflicting gpis

HANDLE_MAIL_CONFLICT_LEVEL = {{ HANDLE_MAIL_CONFLICT_LEVEL | default("'1'") }}

INTERCEPT_CEILING = {{ INTERCEPT_CEILING | default(3) }}

#If True, indicates that a client on Cost Saver marketplace 
MARKETPLACE_CLIENT = {{ MARKETPLACE_CLIENT | default(False) }}

# If True, adds GoodRx optimization preprocessing logic to the LP algorithm run. Currently only works for vanilla commercial clients.
# RAW_GOODRX file is the cleaned GoodRx prices from the Product Team (SJ's excel "pricing tool"). Only needed if GOODRX_OPT = True
# GOODRX_FILE is the file that has the new prices that will be used to do GoodRx optimization. Only needed if GOODRX_OPT = True
GOODRX_OPT = {{ GOODRX_OPT | default(False) }}
RAW_GOODRX = {{ RAW_GOODRX | default("''") }}
GOODRX_FILE = {{ GOODRX_FILE | default("'GoodRx_Limits_{}.csv'.format(DATA_ID)") }}
COSTPLUS_FILE = {{ COSTPLUS_FILE | default("'Costplus_Limits_{}.csv'.format(DATA_ID)") }}
GOODRX_MATCH_METHOD = {{ GOODRX_MATCH_METHOD | default("'forward'") }}
GOODRX_QTY_METHOD = {{ GOODRX_QTY_METHOD | default("'median'") }}

RMS_OPT = {{ RMS_OPT | default(False) }}
GOODRX_FACTOR = {{ GOODRX_FACTOR | default(0.5) }}


# Brand Surplus Data for Brand-Generic Offsets
# The code checks the taxonomy table for the existance of a Brand-Generic offset
# Mark True if want to read the brand_surplus_{}.csv file from INPUT folder.
# to override BG offset performance adjustment set BRAND_SURPLUS_READ_CSV = True and create an empty CSV file with
# the three columns=['CLIENT', 'MEASUREMENT', 'SURPLUS']
BRAND_SURPLUS_READ_CSV = {{ BRAND_SURPLUS_READ_CSV | default(False) }}
# this file is saved into INPUT file folder
BRAND_SURPLUS_FILE = {{ BRAND_SURPLUS_FILE | default("'brand_surplus_{}.csv'.format(DATA_ID)") }}
# This is the brand specialty surplus file to be joined with the generic table for the client performance report later
# Currently using for Aetna
BRAND_SURPLUS_REPORT = {{ BRAND_SURPLUS_REPORT | default("'brand_surplus_report_{}.csv'.format(DATA_ID)") }}


# Specialty Offsetting
# Reads in from CSV file projected specialty performance at the customer_id level
# for offsetting with generic performance in the LP.
# Currently only implemented for Pacificsource and it's associated customer_ids.
SPECIALTY_OFFSET = {{ SPECIALTY_OFFSET | default(True) }}
# Contains customer IDs and projected specialty surplus for the whole contract period.
SPECIALTY_SURPLUS_DATA = {{ SPECIALTY_SURPLUS_DATA | default("'spclty_surplus_20250109.csv'") }}

# Cross Contract Projections
# To be used for early-in-contract runs where less than three months of YTD data have been accumulated.
# Uses utilization from the prior contract as projections for the current contract EOY period. The runner must
# set the DATA_START_DAY in such a way so that the number of months pulled in from the prior contract are equal 
# to the EOY period of the current contract. So in an ideal situation, if you have a 1/1 to 12/31 contract, and 
# your GO_LIVE is 3/1/2023, set the DATA_START_DAY to 3/1/2022.
CROSS_CONTRACT_PROJ = {{ CROSS_CONTRACT_PROJ | default(False) }}

# Points code to the cross contract projections pipeline when doing CROSS_CONTRACT_PROJ runs. Assumes 
# maintaining a <table_name>_CCP_<last year>_<this_year> naming convention.
if CROSS_CONTRACT_PROJ == True:
    CCP_SUFFIX = {{ CCP_SUFFIX | default("'_CCP' + '_{}_{}'.format(str(dt.date.today().year - 1), str(dt.date.today().year))") }}
else:
    CCP_SUFFIX = {{ CCP_SUFFIX | default("''") }}

# Optimize prices with the objective function for pharmacies in the NON_CAPPED_PHARMACY_LIST to minimize leakage. 
# Leakage is the amount a member pays (not the client) to a non-capped pharmacy above the pharmacy/MAC1026 price. 
# Currently only intended to be run on locked clients.
# If using LEAKAGE_OPT, using ZBD_OPT should be unnecessary.
LEAKAGE_OPT = {{ LEAKAGE_OPT | default(False) }}   

# Select top leakage GPIs for Leakage optimization and ZBD optimization, default is 200, but can be increased or decreased depending on 
# how difficult it is to acheive improved client or solver performance (longer list will increase solve time). Not applicible if
# LEAKAGE_LIST is set to 'All'
LEAKAGE_RANK = {{ LEAKAGE_RANK | default(200) }}

# Set 'Legacy' to use leakage rankings based on 2022 overall data, set 'Client' to use client specific rankings based
# on current contract data, or set to 'All' to not use the rankings list table. 'All' option only works with LEAKAGE_OPT, all others 
# work with both ZBD_OPT and LEAKAGE_OPT. The Legacy function should be phased out as these mature.
LEAKAGE_LIST = {{ LEAKAGE_LIST | default("'All'") }}

# Weighting used to control how much leakage should penalize the overall objective function. A value of 1.0 is doller-per-dollar
# weighting, a value of 0.5 halves the weight, and a value of 2.0 doubles the weight, and so on.
LEAKAGE_PENALTY = {{ LEAKAGE_PENALTY | default(5.0) }}

# Used to set which column is used for the maximum copay breakpoint of the leakage function for simple copay plan designs. 
# Acceptable values are: 'AVG_COPAY_UNIT', 'MEDIAN_COPAY_UNIT', 'MIN_COPAY_UNIT', and 'MAX_COPAY_UNIT'. 
# Setting this incorrectly will cause a key error.
LEAKAGE_COPAY_SIMPLE_BREAKPOINT = {{ LEAKAGE_COPAY_SIMPLE_BREAKPOINT | default("'AVG_COPAY_UNIT'") }}

# Used to set which column is used for the maximum copay breakpoint of the leakage function for complex plan designs. 
# Acceptable values are: 'AVG_COMPLEX_COPAY_UNIT', 'MEDIAN_COMPLEX_COPAY_UNIT', 'MIN_COMPLEX_COPAY_UNIT', and 'MAX_COMPLEX_COPAY_UNIT'. 
# Setting this incorrectly will cause a key error.
LEAKAGE_COPAY_COMPLEX_BREAKPOINT = {{ LEAKAGE_COPAY_COMPLEX_BREAKPOINT | default("'AVG_COMPLEX_COPAY_UNIT'") }}

# Used to set which column is used as the coinsurance factor for the leakage function in simple plan designs. 
# Acceptable values are: 'AVG_COINS', 'MEDIAN_COINS', 'MIN_COINS', and 'MAX_COINS'. 
# Setting this incorrectly will cause a key error.
LEAKAGE_COINSURANCE_SIMPLE = {{ LEAKAGE_COINSURANCE_SIMPLE | default("'AVG_COINS'") }}

# Used to set which column is used as the coinsurance factor for the leakage function in complex plan designs. 
# Acceptable values are: 'AVG_COMPLEX_COINS', 'MEDIAN_COMPLEX_COINS', 'MIN_COMPLEX_COINS', and 'MAX_COMPLEX_COINS'. 
# Setting this incorrectly will cause a key error.
LEAKAGE_COINSURANCE_COMPLEX = {{ LEAKAGE_COINSURANCE_COMPLEX | default("'AVG_COMPLEX_COINS'") }}

# If True, adds preprocessing logic that puts an additional ceiling on prices at pharmacies on the NON_CAPPED_PHARMACY_LIST
# in order to reduce leakage. Currently only intended to be run on locked clients.
# If using LEAKAGE_OPT, using ZBD_OPT should be unnecessary.
ZBD_OPT = {{ ZBD_OPT | default(False) }}

# Weighting used to control how much leakage should penalize the overall objective function. A value of 1.0 is doller-per-dollar
# weighting, a value of 0.5 halves the weight, and a value of 2.0 doubles the weight, and so on.
LEAKAGE_PENALTY = {{ LEAKAGE_PENALTY | default(5.0) }}

# Set CVS_IND scalar for ZBD optimization
ZBD_CVS_IND_SCALAR = {{ ZBD_CVS_IND_SCALAR | default(1.2) }}

# Set capped pharmacy scalar for ZBD optimization
ZBD_CAPPED_SCALAR = {{ ZBD_CAPPED_SCALAR | default(3.6) }}

# Set current price scalar for ZBD optimization
ZBD_CURRENT_PRICE_SCALAR = {{ ZBD_CURRENT_PRICE_SCALAR | default(0.5) }}

# For clients that have R90 claims adjudicating on the Mail VCML -- such as State of Florida -- set to True
R90_AS_MAIL = {{ R90_AS_MAIL | default(False) }}

# Require Mail prices to adhere to the MAC1026 floors.
# Adjust factor to move floor--set to 0.80 to be 80% of MAC1026 price, 1.20 to be 120% of floor, etc.
APPLY_FLOORS_MAIL = {{ APPLY_FLOORS_MAIL | default(True) }}
MAIL_FLOORS_FACTOR = {{ MAIL_FLOORS_FACTOR | default(1.0) }}

#================================================================
#================================================================
# If True, the code reads the WC_SUGGESTED_GUARDRAILS and changes the intrinsic code guard rails
# TODO: change the file/parameter name
CLIENT_GR = {{ CLIENT_GR | default(False) }}  #the client supplies guard rails set this to true
WC_SUGGESTED_GUARDRAILS = {{ WC_SUGGESTED_GUARDRAILS | default("'20200220_WCGuardRailsLax.csv'") }}
# If True, it loosens up the supplied scale factor only works if CLIENT_GR = True
GR_SCALE = {{ GR_SCALE | default(False) }}
# This is how much the scale factor will be loosened both up and down (max 1)
GR_SF = {{ GR_SF | default(1.0) }}
# If True, it implements certain tiers of guard rails given by TIER_LIST
LIM_TIER = {{ LIM_TIER | default(False) }}
# These are the only guard rails that will be implemented if LIM_TIER = True
TIER_LIST = {{ TIER_LIST | default(['1', '2']) }}
# If True, it modifies the GR on the NON_CAPPED_PHARMACY_LIST
CAPPED_ONLY = {{ CAPPED_ONLY | default(False) }}
#================================================================

#================================================================
#================================================================
# NO NEED TO MODIFY -- CAN BE CHANGED BY DS IF NEEDED WITHOUT BUSINESS APPROVAL
#================================================================

# Path structures for the different parts of the code
# On the cloud the paths will not longer exsit
FILE_INPUT_PATH = {{ FILE_INPUT_PATH | default("os.path.join(PROGRAM_INPUT_PATH, 'Input/')") }}
FILE_OUTPUT_PATH = {{ FILE_OUTPUT_PATH | default("os.path.join(PROGRAM_OUTPUT_PATH, 'Output/')") }}
FILE_DYNAMIC_INPUT_PATH = {{ FILE_DYNAMIC_INPUT_PATH | default("os.path.join(PROGRAM_OUTPUT_PATH, 'Dynamic_Input/')") }}
FILE_LOG_PATH = {{ FILE_LOG_PATH | default("os.path.join(PROGRAM_OUTPUT_PATH, 'Logs/')") }}
FILE_LP_PATH = {{ FILE_LP_PATH | default("os.path.join(PROGRAM_OUTPUT_PATH, 'LP/')") }}
FILE_REPORT_PATH = {{ FILE_REPORT_PATH | default("os.path.join(PROGRAM_OUTPUT_PATH, 'Report/')") }}

#================================================================
# The list of the input files that are created from Teradata.

# Client metadata: CLIENT, CUSTOMER_ID, Guarantee_Category, No_of_VCMLs
CLIENT_NAME_MAPPING_FILE = {{ CLIENT_NAME_MAPPING_FILE | default("'Client_Name_ID_Mapping_{}.csv'.format(DATA_ID)") }}

# List of MAC prices for selected client(s)
MAC_LIST_FILE = {{ MAC_LIST_FILE | default("'Mac_List_{}.csv'.format(DATA_ID)") }}

# List of the minimum prices (MAC 1026 floor prices) for each GPI-NDC
RAW_MAC1026_FILE = {{ RAW_MAC1026_FILE | default("'raw_MAC1026_{}.csv'.format(DATA_ID)") }}
MAC1026_FILE = {{ MAC1026_FILE | default("'MAC1026_{}.csv'.format(DATA_ID)") }}

# List of Beg_Q_Price and Beg_M_Price for WTW/AON clients
DRUG_MAC_HIST_FILE = {{ DRUG_MAC_HIST_FILE | default("'Drug_MAC_History_{}.csv'.format(DATA_ID)") }}

# List of the NADAC AND WAC prices for each GPI-NDC
RAW_NADAC_WAC_FILE = {{ RAW_NADAC_WAC_FILE| default("'raw_NADAC_WAC_{}.csv'.format(DATA_ID)") }}
NADAC_WAC_FILE = {{ NADAC_WAC_FILE | default("'NADAC_WAC_{}.csv'.format(DATA_ID)") }}

# BREAKOUT MAPPING FILE FOR THE CLIENT
RAW_BREAKOUT_MAPPING_FILE = {{RAW_BREAKOUT_MAPPING_FILE |default("'RAW_BREAKOUT_MAPPING_FILE_{}.csv'.format(DATA_ID)") }}
BREAKOUT_MAPPING_FILE = {{BREAKOUT_MAPPING_FILE |default("'BREAKOUT_MAPPING_FILE_{}.csv'.format(DATA_ID)") }}

# The client(s) guarantees for the different breakouts
CLIENT_GUARANTEE_FILE = {{ CLIENT_GUARANTEE_FILE | default("'Client_Guarantees_{}.csv'.format(DATA_ID)") }}

# Client guarantees from before a market check
CLIENT_GUARANTEE_PREMC_FILE = {{ CLIENT_GUARANTEE_PREMC_FILE | default("'Client_Guarantees_Premc_{}.csv'.format(DATA_ID)") }}

# List of the commerical pharmacy guarantees for all commercial clients
COMMERCIAL_PHARM_GUARANTEES_FILE = {{ COMMERCIAL_PHARM_GUARANTEES_FILE | default("'Pharm_Guarantees_commercial_{}.csv'.format(DATA_ID)") }}
# List of the medicaid pharmacy guarantees for all medicaid clients
MEDICAID_PHARM_GUARANTEES_FILE = {{ MEDICAID_PHARM_GUARANTEES_FILE | default("'Pharm_Guarantees_medicaid_{}.csv'.format(DATA_ID)") }}

# Pharmacy guarantees with client information after pre-processing
PHARM_GUARANTEE_FILE = {{ PHARM_GUARANTEE_FILE | default("'Pharm_Guarantees_{}.csv'.format(DATA_ID)") }}

# The VCML information for a specific client(s)
# read into pre-processing
VCML_REFERENCE_FILE = {{ VCML_REFERENCE_FILE | default("'VCML_Reference_{}.csv'.format(DATA_ID)") }}
# after pre-processing
MAC_MAPPING_FILE = {{ MAC_MAPPING_FILE | default("'Mac_Mapping_{}.csv'.format(DATA_ID)") }}

#Customized new year region map mapping file for MEDD new year pricing
CUSTOM_MAC_MAPPING_FILE = {{ CUSTOM_MAC_MAPPING_FILE | default("'Custom_Mac_Mapping_2024.csv'") }}

# Pharmacy claims at the month level (aggregated)
# read into pre-processing
PHARMACY_CLAIM_FILE = {{ PHARMACY_CLAIM_FILE | default("'Pharmacy_Raw_Claims_{}.csv'.format(DATA_ID)") }}
# Year to date pharmacy performance by breakout, measurement, and chain group
# after pre-processing
PHARMACY_YTD = {{ PHARMACY_YTD | default("'YTD_Pharmacy_Performance_{}.csv'.format(DATA_ID)") }}

# Client and pharmacy claims table claim level
CLIENT_PHARMACY_CLAIM_FILE = {{ CLIENT_PHARMACY_CLAIM_FILE | default("''") }}

# Non-mac rates per VCML
NON_MAC_RATE_FILE = {{ NON_MAC_RATE_FILE | default("'Non_Mac_Rate_{}.csv'.format(DATA_ID)") }}

# The generic launch file before being modify for proper code use (format)
# read into pre-processing
RAW_GEN_LAUNCH_FILE = {{ RAW_GEN_LAUNCH_FILE | default("'gen_launch_{}.csv'.format(DATA_ID)") }}
# The generic launch file after being modified for proper code use (format)
# after pre-processing
GENERIC_LAUNCH_FILE = {{ GENERIC_LAUNCH_FILE | default("'generic_launch_{}.csv'.format(DATA_ID)") }}

# List of GPIs and prices that will override current prices from the MAC_PRICE_OVERRIDE table; 
# the algorithm is prevented from modifying them as they will always adjudicate at the override price
# prior to pre-processing
RAW_MAC_PRICE_OVERRIDES = {{ RAW_MAC_PRICE_OVERRIDES | default("'RAW_GER_OPT_MAC_Price_Overrides_{}.csv'.format(DATA_ID)") }}
# after pre-processing
MAC_PRICE_OVERRIDE_FILE = {{ MAC_PRICE_OVERRIDE_FILE | default("'MAC_Price_Overrides_{}.csv'.format(DATA_ID)") }}
# Parameter to treat MAC PRICE OVERRIDE GPIs as Overrides (True) or Exclusions(False)
MAC_PRICE_OVERRIDE = {{ MAC_PRICE_OVERRIDE | default(True)}}
# WMT UNC OVERRIDE FILE NAME
WMT_UNC_PRICE_OVERRIDE_FILE = {{ WMT_UNC_PRICE_OVERRIDE_FILE | default("'WMT_UNC_Price_Overrides_{}.csv'.format(DATA_ID)") }}

# Information about a package size for a GPI-NDC combination
PACKAGE_SIZE_FILE = {{ PACKAGE_SIZE_FILE | default("'package_size_to_ndc_{}.csv'.format(DATA_ID)") }}

# List of backout GPIs
BACKOUT_GEN = {{ BACKOUT_GEN | default("'Gen_Launch_Backout_{}.csv'.format(DATA_ID)") }}

# File that maps the raw claims to their respective breakouts, regions, measurements, and chain groups (preferred and non preferred)
MEASUREMENT_MAPPING = {{ MEASUREMENT_MAPPING | default("'Measurement_mapping_{}.csv'.format(DATA_ID)") }}

# The mapping of VCMLs to guarantee equal prices on a VCML level
MAC_CONSTRAINT_FILE = {{ MAC_CONSTRAINT_FILE | default("'Mac_Constraints_{}.csv'.format(DATA_ID)") }}

# Client(s) claims at the GPI level
DAILY_TOTALS_FILE = {{ DAILY_TOTALS_FILE | default("'Daily_Total_{}.csv'.format(DATA_ID)") }}

# Projections data at day level
EOY_PROJ_FILE = {{ EOY_PROJ_FILE | default("'EOY_PROJ_DAY_{}.csv'.format(DATA_ID)")}}

# Scale factor values for the pharmacy performance approximation
PHARMACY_APPROXIMATIONS = {{ PHARMACY_APPROXIMATIONS | default("'Pharmacy_approx_coef_{}.csv'.format(DATA_ID)") }}
PHARMACY_APPROXIMATIONS_XY_DATA = {{ PHARMACY_APPROXIMATIONS_XY_DATA | default("'Pharmacy_approx_XYdata_{}.csv'.format(DATA_ID)") }}

# List of GPIs to be excluded from the algorithm because they are specialty drugs
RAW_SPECIALTY_EXCLUSION_FILE = {{ RAW_SPECIALTY_EXCLUSION_FILE | default("'Raw_GPI_change_exclusion_NDC_{}.csv'.format(DATA_ID)")}}

# List of GPIs to be excluded from the algorithm because of MONY changes
RAW_MONY_EXCLUSION_FILE = {{ RAW_MONY_EXCLUSION_FILE | default("''")}}

SPECIALTY_EXCLUSION_FILE = {{ SPECIALTY_EXCLUSION_FILE | default("'GPI_change_exclusion_NDC_{}.csv'.format(DATA_ID)") }}

#List of GPIs to be excluded from the algorithm because they cause an infeasibility
INFEASIBLE_EXCLUSION_FILE = {{ INFEASIBLE_EXCLUSION_FILE | default("'infeasible_exclusion_gpis_{}.csv'.format(CUSTOMER_ID[0])") }}

#List of GPIs to be excluded from the algorithm because they have conflicting R30/R90/M30 prices
CONFLICT_GPI_LIST_FILE = {{ CONFLICT_GPI_LIST_FILE | default("'conflict_exclusion_gpis_{}_{}_{}.csv'.format(CUSTOMER_ID[0], dt.date.today().strftime('%B'), RUN_TYPE_TABLEAU)") }}
#Conflicting GPIs generated this round
CONFLICT_GPI_LIST_FILE_THIS_RUN = {{ CONFLICT_GPI_LIST_FILE_THIS_RUN | default("'conflict_exclusion_gpis_this_run_{}_{}_{}.csv'.format(CUSTOMER_ID[0], dt.date.today().strftime('%B'),RUN_TYPE_TABLEAU)") }}

# The Other client pharmacy performance is only used for Med-D clients
OC_PHARM_PERF_FILE = {{ OC_PHARM_PERF_FILE | default("'all_other_medd_client_perf_{}.csv'.format(DATA_ID)") }}

# History of GPI-NDC AWP prices
AWP_FILE = {{ AWP_FILE | default("'AWP_HISTORY_TABLE_{}.csv'.format(DATA_ID)") }}

# GPI Classes file
GPI_CLASSES = {{ GPI_CLASSES | default("'GPI & CLASSES.xlsx'")}}

# List of preferred pharmacies (if any otherwise none) for each client
# The current process for generating this file for Med D clients consists of reaching out to the Networks team (Dana Jones)
# via email with a list of Med D clients to onboard Dana then responds with the list of clients and their respective CARRIER
# and NETWORKS that are preferred, then manually creating the input file for the algo run
PREFERRED_PHARM_FILE = {{ PREFERRED_PHARM_FILE | default("'Pref_Pharm_List_{}.csv'.format(DATA_ID)") }}

# UNC Daily totals and Percentiles files
UNC_DAILY_TOTALS_FILE = {{ UNC_DAILY_TOTALS_FILE | default("'UNC_Daily_Total_{}.csv'.format(DATA_ID)") }}
UNC_GPI_PERCENTILES_FILE = {{ UNC_GPI_PERCENTILES_FILE | default("'UNC_GPI_percentiles_constrained_{}.csv'.format(DATA_ID)") }}  # UCAMT percentiles for all claims at GPI level
UNC_NDC_PERCENTILES_FILE = {{ UNC_NDC_PERCENTILES_FILE | default("'UNC_NDC_percentiles_constrained_{}.csv'.format(DATA_ID)") }}  # UCAMT percentiles for all claims at GPI-NDC level

# Daily Totals Report
FILE_DAILY_TOTALS_REPORT = {{ FILE_DAILY_TOTALS_REPORT | default("'daily_totals_{}.sql'.format(DATA_ID)") }}

# Value Report due to INTERCEPTOR_OPT = True
INTERCEPTOR_VALUE_REPORT = {{ INTERCEPTOR_VALUE_REPORT | default("'Interceptor_value_{}.CSV'.format(DATA_ID)") }}
# The leakage parametrizes the claim movlume that will be miss labeled
INTERCEPTOR_LEAKAGE = {{ INTERCEPTOR_LEAKAGE | default(0.05) }}
# Determines the lenght of the timeperiod to determine if a drug is zbd or not
INTERCEPTOR_ZBD_TIME_LAG = {{ INTERCEPTOR_ZBD_TIME_LAG | default(1) }}

# U&C EBIT report
UNC_EBIT_REPORT = {{ UNC_EBIT_REPORT | default("'unc_ebit_report_{}.csv'.format(DATA_ID)") }}

# Current contract data file
CURRENT_CONTRACT_DATA_FILE = {{ CURRENT_CONTRACT_DATA_FILE | default("'current_contract_data_{}.csv'.format(DATA_ID)") }}

# Aggregated plan design files
PLAN_DESIGN_FILE_GPI = {{ PLAN_DESIGN_FILE_GPI | default("'plan_design_file_gpi_{}.csv'.format(DATA_ID)") }}
PLAN_DESIGN_FILE_NDC = {{ PLAN_DESIGN_FILE_NDC | default("'plan_design_file_ndc_{}.csv'.format(DATA_ID)") }}

# Specialty Surplus file
SPECIALTY_SURPLUS_FILE = {{ SPECIALTY_SURPLUS_FILE | default("'spclty_surplus_file_{}.csv'.format(DATA_ID)") }}

# Market check data files
PRE_MC_DATA_FILE = {{ PRE_MC_DATA_FILE | default("'pre_mc_data_{}.csv'.format(DATA_ID)") }}
POST_MC_DATA_FILE = {{ POST_MC_DATA_FILE | default("'post_mc_data_{}.csv'.format(DATA_ID)") }}

#================================================================
# The list of the output files that are created by the LP and use on QA.

# Is produced by the LP code and is used to validate the QA
PHARMACY_OUTPUT = {{ PHARMACY_OUTPUT | default("'LP_Algorithm_Pharmacy_Output_Month_{}.csv'.format(DATA_ID)") }}

PRICE_CHECK_OUTPUT = {{ PRICE_CHECK_OUTPUT | default("'Price_Check_Output_{}.csv'.format(DATA_ID)") }}

TOTAL_OUTPUT = {{ TOTAL_OUTPUT | default("'Total_Output_{}.csv'.format(DATA_ID)") }}

PRE_EXISTING_PERFORMANCE_OUTPUT = {{ PRE_EXISTING_PERFORMANCE_OUTPUT | default("'Pre_existing_Performance_{}_{}.csv'.format(str(GO_LIVE.month), DATA_ID)") }}

MODEL_PERFORMANCE_OUTPUT = {{ MODEL_PERFORMANCE_OUTPUT | default("'Model_Performance_{}_{}.csv'.format(str(GO_LIVE.month), DATA_ID)") }}

MODEL_02_PERFORMANCE_OUTPUT = {{ MODEL_02_PERFORMANCE_OUTPUT | default("'Model_02_Performance_{}_{}.csv'.format(str(GO_LIVE.month), DATA_ID)") }}

LAMBDA_PERFORMANCE_OUTPUT = {{ LAMBDA_PERFORMANCE_OUTPUT | default("'Lambda_Performance_{}_{}.csv'.format(str(GO_LIVE.month), DATA_ID)") }}

PRICE_CHANGE_FILE = {{ PRICE_CHANGE_FILE | default("'{}_{}_{}.txt'.format(CLIENT_NAME_TABLEAU, CLIENT_TYPE, GO_LIVE.strftime('%m%d%y'))") }}

DIAGNOSTIC_REPORT = {{ DIAGNOSTIC_REPORT | default("'Diagnostic_Report_{}.csv'.format(DATA_ID)") }}

CONTRACT_DATE_FILE = {{ CONTRACT_DATE_FILE | default("'Contract_Dates_{}.csv'.format(DATA_ID)") }}

PRICE_CHANGES_OUTPUT_FILE = {{ PRICE_CHANGES_OUTPUT_FILE | default("'{}_Price_Changes_{}_{}.xlsx'.format(CUSTOMER_ID[0], TIMESTAMP, str(GO_LIVE.month))") }}

###### Parameters used to create Daily Totals in Teradata #############################
# These are names of temporary tables that will be created in the SB_FINANCE_G2_GER_OPT database
# all of these are automatically removed once the queries are complete except for
#   TABLE_UNC_MEASUREMENT_MAPPING
#   TABLE_AUTOMATION_DAILY_TOTALS
#   TABLE_AUTOMATION_UNC_DAILY_TOTALS

#Turning this flag to True would clean all tables after the "csv" files are created 
DROP_TABLES = {{ DROP_TABLES | default(True) }}

GER_UNC_PREFIX = {{ GER_UNC_PREFIX | default("'GER_DA_UNC_'") }}
GER_PREFIX = {{ GER_PREFIX | default("'GER_DA_'") }}

# list of temporary tables created during daily_totals queries.
# Name of the tables should be less than 15 characters or the table can't be dropped by the program
TABLE_RAW_CLAIMS = {{ TABLE_RAW_CLAIMS | default("GER_PREFIX + USER + '_' + 'raw_clm3'") }}
if CLIENT_TYPE == 'MEDD':
    TABLE_MEASUREMENT_MAPPING = {{ TABLE_MEASUREMENT_MAPPING | default("GER_PREFIX + DATA_START_DAY[:4] +  CLIENT_NAME_TABLEAU + '_' + 'msrmnt_map3'") }}
elif (CLIENT_TYPE == 'COMMERCIAL' or CLIENT_TYPE == 'MEDICAID'):
    TABLE_MEASUREMENT_MAPPING = {{ TABLE_MEASUREMENT_MAPPING | default("GER_PREFIX + USER + '_' +'msrmnt_map3'") }}
TABLE_MAPPED_CLAIMS = {{ TABLE_MAPPED_CLAIMS | default("GER_PREFIX + USER + '_' + 'all_mac_and_uc3'") }}
TABLE_MEDIAN_AND_25TH = {{ TABLE_MEDIAN_AND_25TH | default("GER_PREFIX + USER + '_'+ 'percentile3'") }}
TABLE_AUTOMATION_DAILY_TOTALS = {{ TABLE_AUTOMATION_DAILY_TOTALS | default("GER_PREFIX + USER + '_'+ 'daily_totals3'") }}

TABLE_UNC_AUTOMATION_DAILY_TOTALS = {{ TABLE_UNC_AUTOMATION_DAILY_TOTALS | default("GER_UNC_PREFIX+ USER + '_' + 'daily_total'") }}
TABLE_UNC_RAW_CLAIMS = {{ TABLE_UNC_RAW_CLAIMS | default("GER_UNC_PREFIX+ USER + '_' + 'raw_clm3'") }}
TABLE_UNC_MAC_1026 = {{ TABLE_UNC_MAC_1026 | default("GER_UNC_PREFIX+ USER + '_' +'mac_1026'") }}
TABLE_UNC_ALL_MAC_AND_UC_CLAIMS = {{ TABLE_UNC_ALL_MAC_AND_UC_CLAIMS | default("GER_UNC_PREFIX + USER + '_' + 'claims3'") }}
TABLE_UNC_ALL_MAC_AND_UC_CLAIMS_MAC = {{ TABLE_UNC_ALL_MAC_AND_UC_CLAIMS_MAC | default("GER_UNC_PREFIX + USER + '_' + 'MAC3'") }}

# tables used to create the percentiles for the U&C algorithm
TABLE_UNC_FILTERED_MAC_AND_UNC_PERCENTILES_NDC = {{ TABLE_UNC_FILTERED_MAC_AND_UNC_PERCENTILES_NDC | default("GER_UNC_PREFIX + USER + '_' + 'NDC_prcntl'") }}
TABLE_UNC_PERCENTILES_NDC_CONSTRAINTS_MAC_NDC = {{ TABLE_UNC_PERCENTILES_NDC_CONSTRAINTS_MAC_NDC | default("GER_UNC_PREFIX + USER + '_' + 'NDC_cnstrn'") }}
TABLE_UNC_FILTERED_MAC_AND_UNC_PERCENTILES_GPI = {{ TABLE_UNC_FILTERED_MAC_AND_UNC_PERCENTILES_GPI | default("GER_UNC_PREFIX + USER + '_' + 'GPI_prcntl'") }}
TABLE_UNC_PERCENTILES_GPI_CONSTRAINTS_MAC_GPI = {{ TABLE_UNC_PERCENTILES_GPI_CONSTRAINTS_MAC_GPI | default("GER_UNC_PREFIX + USER + '_' + 'GPI_cnstrn'") }}

#================================================================
#================================================================
# 1/1 Prices
#================================================================

# If True, sets the code to provide 1/1 prices for the start of a new year
FULL_YEAR = {{ FULL_YEAR | default(False) }}

# If True the client has price tiers, default is True.  If True it only applies to the clients on the TIERED_PRICE_CLIENT list
if any(ext in CLIENT_NAME_TABLEAU.upper() for ext in ['AON', 'WTW','MVP','HEALTHFIRST']) or any(cid == CUSTOMER_ID for cid in ['183C','185C']):
    if FULL_YEAR:
        TIERED_PRICE_LIM = {{ TIERED_PRICE_LIM | default(False) }}
    else:
        TIERED_PRICE_LIM = False 
else:
    TIERED_PRICE_LIM = {{ TIERED_PRICE_LIM | default(True) }}

TIERED_PRICE_CLIENT = {{ TIERED_PRICE_CLIENT | default('CUSTOMER_ID') }}

# Higher of logic
# If True the client has higher of price change limit where prices can go to the higher of GPI_UP_FAC or GPI_UP_DOLLAR regardless of current script cost.
HIGHEROF_PRICE_LIM = {{ HIGHEROF_PRICE_LIM | default(False) }}
assert not(HIGHEROF_PRICE_LIM and TIERED_PRICE_LIM), "higher of logic does not work with tiered pricing"

#Points code to the Welcome Season 1/1 pipeline when doing FULL_YEAR runs. Assumes DE maintains <table_name>_WS_<next_year> naming convention.
#Set to '' if using default pipeline data is desired for a FULL_YEAR run.
if FULL_YEAR == True:
    WS_SUFFIX = {{ WS_SUFFIX | default("'_ws'") }}
else:
    WS_SUFFIX = {{ WS_SUFFIX | default("''") }}

# It sets the pricing rules that will be used for 1/1 prices. Goes from 1 to 3 with 1 being the least aggressive.
# 1 is also the default setting. It requires TIERED_PRICE_LIM = True to work and prices have tiers.
NEW_YEAR_PRICE_LVL = {{ NEW_YEAR_PRICE_LVL | default(1) }}

# Performance buffer on the client side, number derived from the business
CLIENT_TARGET_BUFFER = {{ CLIENT_TARGET_BUFFER | default(-0.000) }}

# Performance buffer on the pharmacy side, number derived from the business
PHARMACY_TARGET_BUFFER = {{ PHARMACY_TARGET_BUFFER | default(-0.000) }}

# Performance buffer on mail, number derived from the business (Default set at zero, since this buffer is designed to be used during a non 1/1 pricing run as well)
MAIL_TARGET_BUFFER = {{ MAIL_TARGET_BUFFER | default(-0.000) }}

# Performance buffer on retail, number derived from the business
RETAIL_TARGET_BUFFER = {{ RETAIL_TARGET_BUFFER | default(-0.000) }}

#Price Bounds for FULL_YEAR: Level 1
FULL_YEAR_LV_1_UPPER_BOUND = {{ FULL_YEAR_LV_1_UPPER_BOUND | default([8, 25, 50, 100, 999999]) }}
FULL_YEAR_LV_1_MAX_PERCENT_INCREASE = {{ FULL_YEAR_LV_1_MAX_PERCENT_INCREASE | default([20000, 0.6, 0.35, 0.25, 0.15]) }}
FULL_YEAR_LV_1_MAX_DOLLAR_INCREASE = {{ FULL_YEAR_LV_1_MAX_DOLLAR_INCREASE | default([8, 999999, 999999, 999999, 999999]) }}
FULL_YEAR_LV_1_PRICE_BOUNDS = {
    'upper_bound': FULL_YEAR_LV_1_UPPER_BOUND,
    'max_percent_increase': FULL_YEAR_LV_1_MAX_PERCENT_INCREASE,
    'max_dollar_increase': FULL_YEAR_LV_1_MAX_DOLLAR_INCREASE
}
FULL_YEAR_LV_1_PRICE_BOUNDS_DF = pd.DataFrame(FULL_YEAR_LV_1_PRICE_BOUNDS)

#Price Bounds for FULL_YEAR: Level 2
FULL_YEAR_LV_2_UPPER_BOUND = {{ FULL_YEAR_LV_2_UPPER_BOUND | default([8, 25, 50, 100, 999999]) }}
FULL_YEAR_LV_2_MAX_PERCENT_INCREASE = {{ FULL_YEAR_LV_2_MAX_PERCENT_INCREASE | default([20000, 1, 0.75, 0.35, 0.25]) }}
FULL_YEAR_LV_2_MAX_DOLLAR_INCREASE = {{ FULL_YEAR_LV_2_MAX_DOLLAR_INCREASE | default([10, 999999, 999999, 999999, 999999]) }}
FULL_YEAR_LV_2_PRICE_BOUNDS = {
    'upper_bound': FULL_YEAR_LV_2_UPPER_BOUND,
    'max_percent_increase': FULL_YEAR_LV_2_MAX_PERCENT_INCREASE,
    'max_dollar_increase': FULL_YEAR_LV_2_MAX_DOLLAR_INCREASE
}
FULL_YEAR_LV_2_PRICE_BOUNDS_DF = pd.DataFrame(FULL_YEAR_LV_2_PRICE_BOUNDS)

#Price Bounds for FULL_YEAR: Level 3
FULL_YEAR_LV_3_UPPER_BOUND = {{ FULL_YEAR_LV_3_UPPER_BOUND | default([8, 25, 50, 100, 999999]) }}
FULL_YEAR_LV_3_MAX_PERCENT_INCREASE = {{ FULL_YEAR_LV_3_MAX_PERCENT_INCREASE | default([20000, 1.5, 1.0, 0.5, 0.35]) }}
FULL_YEAR_LV_3_MAX_DOLLAR_INCREASE = {{ FULL_YEAR_LV_3_MAX_DOLLAR_INCREASE | default([12, 999999, 999999, 999999, 999999]) }}
FULL_YEAR_LV_3_PRICE_BOUNDS = {
    'upper_bound': FULL_YEAR_LV_3_UPPER_BOUND,
    'max_percent_increase': FULL_YEAR_LV_3_MAX_PERCENT_INCREASE,
    'max_dollar_increase': FULL_YEAR_LV_3_MAX_DOLLAR_INCREASE
}
FULL_YEAR_LV_3_PRICE_BOUNDS_DF = pd.DataFrame(FULL_YEAR_LV_3_PRICE_BOUNDS)


#Price Bounds for FULL_YEAR: Level 4
FULL_YEAR_LV_4_UPPER_BOUND = {{ FULL_YEAR_LV_4_UPPER_BOUND | default([3, 6, 999999]) }}
FULL_YEAR_LV_4_MAX_PERCENT_INCREASE = {{ FULL_YEAR_LV_4_MAX_PERCENT_INCREASE | default([20000, 20000, 3]) }}
FULL_YEAR_LV_4_MAX_DOLLAR_INCREASE = {{ FULL_YEAR_LV_4_MAX_DOLLAR_INCREASE | default([20, 30, 999999]) }}
FULL_YEAR_LV_4_PRICE_BOUNDS = {
    'upper_bound': FULL_YEAR_LV_4_UPPER_BOUND,
    'max_percent_increase': FULL_YEAR_LV_4_MAX_PERCENT_INCREASE,
    'max_dollar_increase': FULL_YEAR_LV_4_MAX_DOLLAR_INCREASE
}
FULL_YEAR_LV_4_PRICE_BOUNDS_DF = pd.DataFrame(FULL_YEAR_LV_4_PRICE_BOUNDS)

#================================================================
#================================================================
# NEW SIMULATION MODE PARAMETERS
#================================================================

#Parameter to decide if optimization process will skip directly to running the optimizer rather than running preprocessing first
#If set to True, will skip Preprocessing, qa_checks, and Daily_Input_Read, going straight from params_op to opt_prep_op.
SKIP_TO_OPT = {{ SKIP_TO_OPT | default(False) }}

#================================================================
#================================================================
# DO NOT MODIFY! -- CHANGE ONLY AT THE DIRECTION OF A BUSINESS PARTNER
#================================================================

# Determines the type for the read in variables. It is used on pd.read_csv.  It should be updated to incude all the variables types needed such that pandas
# does not do any undesired transformation.

# TODO: Change output from speciality_exclusion in pre_processing/sql to match naming convention 
VARIABLE_TYPE_DIC = {'GPI':str, 'NDC':str, 'GPI_CD':str, 'DRUG_ID':str, 'NDC11':str,
                     'CLIENT':str,'REGION':str, 'BREAKOUT':str, 'MEASUREMENT':str, 'CHAIN_GROUP':str, 'CHAIN_SUBGROUP':str,
                     'CLIENT_NAME':str, 'CUSTOMER_ID':str, 'MAC_LIST':str, 'MACLIST':str,
                     'VCML_ID':str}

# RUN_TIME is used to track the times of running the algorithem. it is not meant to be a parameter to be tuned 
RUN_TIME = 1

# This modifies the freedom that zero utilization drugs have. If true, then the price limits an all zero qty drugs are the same as they are for the rest of the
# drugs and are regulated by PRICE_BOUNDS_DF or GPI_UP_FAC. If False, zero qty drugs have more freedom to move up and down.  
# If true, it limits the feasible space and therefore the LP is more likely to have infeasible solutions.
# Should be True during Q1/Q2 and for WTW clients.
ZERO_QTY_TIGHT_BOUNDS = {{ ZERO_QTY_TIGHT_BOUNDS | default(True) }}

# The weight of the soft constraint on zero utilization drugs.
ZERO_QTY_WEIGHT = {{ ZERO_QTY_WEIGHT | default(10) }}

# The list of netowrks that define MCHOICE
MCHOICE_NETWORKS = ['MCHCE','CVSNPP']

# They should always be the same and correspond to the GO_LIVE month unless you are doing 1/1 prices
# in that case they should be 12 rather than 1.
if FULL_YEAR:
    LP_RUN = [12]
elif not FULL_YEAR:
    LP_RUN = [GO_LIVE.month]

#New Flag for actual simulation mode uploading to dashboard. If this is True, the simulation iterations will be written to BQ and uploaded to the dashboard even if Write_to_BQ is False. 
#This is so that simulation results can be displayed even if the simulation was done through csv files.
UPLOAD_TO_DASH = {{ UPLOAD_TO_DASH | default(False) }}

# Sets the alpha on the exponential smoothing
PROJ_ALPHA = 0.7

# If True, each flag modifies the client performance (in different ways) using the same file name LAG_YTD_Override_File
LAG_YTD_OVERRIDE = {{ LAG_YTD_OVERRIDE | default(False) }}
YTD_OVERRIDE = {{ YTD_OVERRIDE | default(False) }}
LAG_YTD_Override_File = {{ LAG_YTD_Override_File | default("''") }}

# If True, this allows us to have different pricing for the LAG period and Implementation period
NDC_UPDATE = False

# If True, this allows a list of GPI12s to have tiered pricing changes.
# This was originally developed to allow some of the prices to get into strength order
# TODO: Update information and create file name parameter.
STRENGTH_PRICE_CHANGE_EXCEPTION = False

# If True, it removes mail from being taken into account
NO_MAIL = {{ NO_MAIL | default(False) }}

# If True, limits breakouts considered only the ones in the BO_LIST
# TODO: Better parameter name
LIMITED_BO = False
BO_LIST = ['MEDD','REGIONS']

# If True, the standard input for pharmacy ytd is daily totals. If you want to only provide monthly totals, set this to true.
# It determines how the PHARMACY_YTD_FILE is treated.
MONTHLY_PHARM_YTD = True

# If True, this allows the pricing of drugs that are projected to have no utilization.
# This allows all prices on MAC list to be in line.
PRICE_ZERO_PROJ_QTY = True

# If True, the output of the LP code is written.
WRITE_OUTPUT = True

# If True, the full mac is written as the output. If False only the ones that have price changes.
OUTPUT_FULL_MAC = {{ OUTPUT_FULL_MAC | default(False) }}

# If True, the differnet '.lp' outpts are written.
WRITE_LP_FILES = True

# If True, it changes the over_reimb_gamma in objective function to over reimburse all capped pharmacies in OVER_REIMB_CHAINS list
CAPPED_OPT = True

# This parameter determines the weight of the pharmacies to the objective function.
# If 0 there is no contribution.  If 1 it has the same weight as the client, the default is 1.
PHARM_PERF_WEIGHT = 1.0

# If True, it assigns new pricing tiers for the GPI on the CLIENT_SUGGESTED_TIERS
CLIENT_TIERS = False
CLIENT_SUGGESTED_TIERS = ''

# If True, multiplies 'KRG', 'WMT' and 'NONPREF_OTH' U&Cs by 500, effectively eliminating them of the calculation
REMOVE_KRG_WMT_UC = True

# If True, multiplies U&Cs at small capped chains, PSAOS and noncapped chains by 500, effectively eliminating the U&C cap at these pharmacies
REMOVE_SMALL_CAPPED_UC = {{ REMOVE_SMALL_CAPPED_UC | default(False) }}

# If True, this excludes the pharmacies on the LIST_PHARMACY_EXCLUSION from getting their prices modified.
# Is a more severe way to enforce guard rails.  Most likely only IND and PSAO will be on the list.
PHARMACY_EXCLUSION = False
LIST_PHARMACY_EXCLUSION = []

# Penalty for client RETAIL over/underperformance.
if LOCKED_CLIENT:
    CLIENT_RETAIL_OVRPERF_PEN = {{ CLIENT_RETAIL_OVRPERF_PEN | default(1.0) }}
    CLIENT_RETAIL_UNRPERF_PEN = {{ CLIENT_RETAIL_UNRPERF_PEN | default(0.5) }}
else:
    CLIENT_RETAIL_OVRPERF_PEN = {{ CLIENT_RETAIL_OVRPERF_PEN | default(0.9) }}
    CLIENT_RETAIL_UNRPERF_PEN = {{ CLIENT_RETAIL_UNRPERF_PEN | default(1.0) }} 

# Penalty for client MAIL under-performance. To remove the Mail UP term
# from the objective function, set this to 0.
CLIENT_MAIL_UNRPERF_PEN = {{ CLIENT_MAIL_UNRPERF_PEN | default(0.8) }}

# Penalty for client MAIL over-performance
CLIENT_MAIL_OVRPERF_PEN = {{ CLIENT_MAIL_OVRPERF_PEN | default(1.0) }}

# Penalty for MChoice over-performance (under-reimbursement)
PHARMACY_MCHOICE_OVRPERF_PEN = {{ PHARMACY_MCHOICE_OVRPERF_PEN | default(0.3) }}

# If True, reads the NEW_MAC_FILE and reads in a set of MACs that are different from the pre-processing
READ_IN_NEW_MACS = False
NEW_MAC_FILE = ''

# When NDC_MAC_LISTS are not empty it reads both TMAC files, they are output mapping files
NDC_MAC_LISTS = []
TMAC_DRUG_FILE = '20200619_TMAC_Drug_Info.csv'
TMAC_MAC_MAP_FILE = '20200619_TMAC_MAC_Mapping.csv'

# Dollar amount that the current price can increase at a single GPI-NDC in a single MAC list
GPI_UP_DOLLAR = {{ GPI_UP_DOLLAR | default(3.0) }}

# Proportion that the current price can increase at a single GPI-NDC in a single MAC list
GPI_UP_FAC = {{ GPI_UP_FAC | default(0.245) }}
UNC_GPI_UP_FAC = {{ UNC_GPI_UP_FAC | default('GPI_UP_FAC') }}

# Proportion that the current price can decrease at a single GPI-NDC in a single MAC list
GPI_LOW_FAC = {{ GPI_LOW_FAC | default(0.60) }}
UNC_GPI_LOW_FAC = {{ UNC_GPI_LOW_FAC | default('GPI_LOW_FAC') }}

# Amount that an entire MAC list can increase.  This is weighted by utilization. -1 will turn this off
AGG_UP_FAC = -1

# Amount that an entire MAC list can decrease.  This is weighted by utilization. -1 will turn this off
AGG_LOW_FAC = -1

# Setting to True will allow us to violate the up and low fac
ALLOW_INTERCEPT_LIMIT = {{ ALLOW_INTERCEPT_LIMIT | default(False) }}

# They modify the weight of the Lambda variables on the objective function.
OVER_REIMB_GAMMA = 0.1
if CLIENT_TYPE == 'MEDD':
    OVER_REIMB_CHAINS = {'GNRC': {{ OVER_REIMB_CHAINS | default(['CVS','WAG','WMT'])}},
                         'BRND': {{ OVER_REIMB_CHAINS | default(['CVS','WAG','WMT']) }}}
elif (CLIENT_TYPE == 'COMMERCIAL' or CLIENT_TYPE == 'MEDICAID'):
    OVER_REIMB_CHAINS = {'GNRC': {{ OVER_REIMB_CHAINS | default(['CVS', 'WAG', 'KRG', 'WMT'])}},
                         'BRND': {{ OVER_REIMB_CHAINS | default(['CVS', 'WAG', 'KRG', 'WMT']) }}}
else:
    raise Exception(f'CLIENT_TYPE {CLIENT_TYPE} not supported in CPMO_parameters yet.')
    
COST_GAMMA = 1.0

# Factor of CVS pricing that independents cannot go below
if CLIENT_TYPE == 'MEDD':
    PREF_OTHER_FACTOR = {{ PREF_OTHER_FACTOR | default(0.7) }}
elif (CLIENT_TYPE == 'COMMERCIAL' or CLIENT_TYPE == 'MEDICAID'):
    PREF_OTHER_FACTOR = 1

# Mail Non-MAC rate: discount off AWP for non-MAC pricing
MAIL_NON_MAC_RATE  = {{ MAIL_NON_MAC_RATE | default(0.35) }}

# Retail Non-MAC rate: discount off AWP for non-MAC pricing
RETAIL_NON_MAC_RATE  = {{ RETAIL_NON_MAC_RATE | default(0.35) }}

# Brand Non-MAC rate: discount off AWP for non-MAC pricing
BRAND_NON_MAC_RATE  = {{ BRAND_NON_MAC_RATE | default(0.15) }}

# Lower limit of Non-MAC discount for Non-MAC GPIs at Non-Capped Pharmacies
FLOOR_NON_MAC_RATE = {{ FLOOR_NON_MAC_RATE | default(1.0) }}

# Allows mail prices to go above retail prices.
# Only set to true if the client contract allows it!
# The mail unrestricted cap controls how much above the minimum of R90 or R30 we allow the mail price to go.
# A value of 1.0 caps the mail price to the minimum of the R90 or R30 price. A value of 2.0 allows double the
# minimum of R90/R30. Values below 1.0 should be unnecessary.
MAIL_MAC_UNRESTRICTED = {{ MAIL_MAC_UNRESTRICTED | default(False) }}
MAIL_UNRESTRICTED_CAP = {{ MAIL_UNRESTRICTED_CAP | default(2.5) }} 

# Allows generic prices to go above brand prices.
# The generic unrestricted cap controls how much above the brand we allow the generic price to go.
# A value of 1.0 caps the generic price to the brand price. A value of 2.0 allows double the
# brand price. Values below 1.0 should be unnecessary.
BRAND_GENERIC_UNRESTRICTED = {{ BRAND_GENERIC_UNRESTRICTED | default(False) }}
GENERIC_UNRESTRICTED_CAP = {{ GENERIC_UNRESTRICTED_CAP | default(1.5) }}

# Set a high cap on prices to prevent extreme outliers that deviate from the market price
APPLY_BENCHMARK_CAP = {{ APPLY_BENCHMARK_CAP | default(True) }}
BENCHMARK_CAP_MULTIPLIER = {{ BENCHMARK_CAP_MULTIPLIER | default(5.0) }}

# Offset added to the client guarantee (R30/R90 as applicable) to be used as the RUR pharmacy guarantee.
# Intention is to allow for a higher reimbursement to rural pharmacies above reimbursement
# normally allowed to non-capped pharmacies. To use, assure RUR is in the SMALL_CAPPED_PHARMACY_LIST.
RUR_GUARANTEE_BUFFER = {{ RUR_GUARANTEE_BUFFER | default(0.005) }}

#================================================================
#================================================================
# Plan Liability parameters. Highly experimental and not fully tested
# If true, incoprorates the plan liability module, the client(s) plan information and focusses on minimizing the client(s) plan liability
INCLUDE_PLAN_LIABILITY = {{INCLUDE_PLAN_LIABILITY | default(False)}}
# List if client(s) to run with plan liability module
PLAN_LIAB_CLIENTS = ['WELLCARE']
# The plan liability weighting refers to how much importance the algorithm assigns to each dollar of client(s) plan liability
PLAN_LIAB_WEIGHT = 0.15
# Incorporates the plan liability weighting factor into the module
READ_IN_PLAN_LIAB_WIEGHTS = False
#
PLAN_LIAB_WEIGHT_FILE = '20210118_WC21_PL_CUST_WEIGHT_Full.csv'
# Factor used for the GAP Cost calculation in the plan liability module
GAP_FACTOR = 0.75
# Factor used for the CAT Cost calculation in the plan liability module
CAT_FACTOR = 0.15
#
CUST_MOVEMENT_WEIGHTS = False
#
CUST_MOVEMENT_FILE = '20210118_WC_Cust_Price_Move_Weights.csv'
# 
MIN_DS_FILE = ''
#================================================================

# Price Bounds
# Last element of each list is an upper bound value. DO NOT DELETE
# If adding additional buckets, make sure to keep the upper bound list ordered
# The index of each element is linked.
# For example, the bucket with an upper bound of $5 has a percent increase limit of 20 and dollar increase limit of 5
UPPER_BOUND = {{ UPPER_BOUND | default([5, 10, 25, 50, 100, 999999]) }}
MAX_PERCENT_INCREASE = {{ MAX_PERCENT_INCREASE | default([20, 1, 0.4, 0.3, 0.2, 0.1]) }}
MAX_DOLLAR_INCREASE = {{ MAX_DOLLAR_INCREASE | default([5, 10, 10, 15, 20, 20000]) }}
PRICE_BOUNDS = {
    'upper_bound': UPPER_BOUND,
    'max_percent_increase': MAX_PERCENT_INCREASE,
    'max_dollar_increase': MAX_DOLLAR_INCREASE
}
PRICE_BOUNDS_DF = pd.DataFrame(PRICE_BOUNDS)

UNC_UPPER_BOUND = {{ UNC_UPPER_BOUND | default('None') }}
UNC_MAX_PERCENT_INCREASE = {{ UNC_MAX_PERCENT_INCREASE | default('None') }}
UNC_MAX_DOLLAR_INCREASE = {{ UNC_MAX_DOLLAR_INCREASE | default('None') }}
# If these were NOT set, use the price tiers we'll be using in the rest of the code
if UNC_UPPER_BOUND is None:
    if FULL_YEAR:
        UNC_UPPER_BOUND = FULL_YEAR_LV_{{ NEW_YEAR_PRICE_LVL | default(1) }}_UPPER_BOUND
    else:
        UNC_UPPER_BOUND = UPPER_BOUND
if UNC_MAX_PERCENT_INCREASE is None:
    if FULL_YEAR:
        UNC_MAX_PERCENT_INCREASE = FULL_YEAR_LV_{{ NEW_YEAR_PRICE_LVL | default(1) }}_MAX_PERCENT_INCREASE
    else:
        UNC_MAX_PERCENT_INCREASE = MAX_PERCENT_INCREASE
if UNC_MAX_DOLLAR_INCREASE is None:
    if FULL_YEAR:
        UNC_MAX_DOLLAR_INCREASE = FULL_YEAR_LV_{{ NEW_YEAR_PRICE_LVL | default(1) }}_MAX_DOLLAR_INCREASE
    else:
        UNC_MAX_DOLLAR_INCREASE = MAX_DOLLAR_INCREASE
UNC_PRICE_BOUNDS = {
    'upper_bound': UNC_UPPER_BOUND,
    'max_percent_increase': UNC_MAX_PERCENT_INCREASE,
    'max_dollar_increase': UNC_MAX_DOLLAR_INCREASE
}
UNC_PRICE_BOUNDS_DF = pd.DataFrame(UNC_PRICE_BOUNDS)
#================================================================
#PARAMETERIZATION OF GOODRX_RULES_SETUP
#If adding additional pricing tiers, make sure to add corresponding inputs to rest of the parameters
#Default values are hard-coded. To use parameterized inputs, comment out the parameter template and comment the hard-coded variables.

#PRICING_TIER_SAME: Enter only the intermediate values for pricing buckets. The lower and upper limits are set to -inf and inf respectively.For eg. [5,10] gives three tiers -inf to $5, $5 to $10, $10 to inf. 
#THRESHOLD_TYPE: Specify the unit of threshold value. Should be either "PERCENT" or "DOLLAR". 
#CHANGE_THRESHOLD: Should be negative to reflect the max allowable decrease in the price. Interpreted as the percentage/dollar by which the price can be decreased. Inputs should be negative values(eg. -0.25 reads as the price can be decreased by a maximum of 25% of the current price)
#GOODRX_COMPETITIVE_MULTIPLIER: GoodRx multiplier to calculate the GoodRx limit using GoodRx logic. For eg. 1.1 means that the SAME Pharmacy price should be within 110% of GoodRx price.

#GOODRX_R90_MATCHING = {{GOODRX_R90_MATCHING | default([False]) }}
GOODRX_R90_MATCHING = False

# Selects if to use the old (2021) logic with client cost share added or move to a simpler version of it.
SIMPLE_RMS  = {{SIMPLE_RMS | default(False) }}

# Determines if the GENERAL_MULTIPLIER will be applyed or not on all drugs independently of script cost or member cost share.
APPLY_GENERAL_MULTIPLIER = {{APPLY_GENERAL_MULTIPLIER | default(False) }}
# The GoodRx muliplier for all drugs. Used for both SIMPLE_RMS
GENERAL_MULTIPLIER = {{GENERAL_MULTIPLIER | default([3]) }} 

# Determines if the MAIL_MULTIPLIER will be applied or not based on Costplus prices.
if RMS_OPT:
    APPLY_MAIL_MULTIPLIER = {{APPLY_MAIL_MULTIPLIER | default(True) }}
else:
    APPLY_MAIL_MULTIPLIER = False
    
# The GoodRx muliplier for all drugs. Used for both SIMPLE_RMS
MAIL_MULTIPLIER = {{MAIL_MULTIPLIER | default([3]) }} 

#------------------------------------
# For SIMPLE_RMS = True
#------------------------------------
# Is the price point (script cost) at which the member cost share logic kicks in. Only used on the SIMPLE_RMS = True
PRICING_TIER = {{PRICING_TIER | default([5]) }} 

# The member cost share benchmarks. They are decimals and represent how much the member most pay before moving to the next group. Only used on the SIMPLE_RMS = True
MBR_COST_SHARE_TIER = {{MBR_COST_SHARE_TIER | default([0.25, 0.75]) }} 

# The GoodRx muliplier, is used in hand with the MBR_COST_SHARE_TIER. Only used on the SIMPLE_RMS = True
COMPETITIVE_MULTIPLIER_SAME = {{COMPETITIVE_MULTIPLIER_SAME | default([1.1, 1.5]) }} 

#======================================
#Based on MEMBER COST SHARE PER CLAIM
#======================================
#MBR_PRICING_TIER_SAME = {{MBR_PRICING_TIER_SAME | default([5, 10]) }} 
MBR_PRICING_TIER_SAME = [5,10]

#MBR_THRESHOLD_TYPE_SAME = {{MBR_THRESHOLD_TYPE_SAME | default([None,"PERCENT","PERCENT"])}}
MBR_THRESHOLD_TYPE_SAME = [None,"PERCENT","PERCENT"]

#MBR_CHANGE_THRESHOLD_SAME = {{MBR_CHANGE_THRESHOLD_SAME| default([None, -0.5, -0.25])}} 
MBR_CHANGE_THRESHOLD_SAME = [None, -0.5, -0.25]

MBR_GOODRX_COMPETITIVE_MULTIPLIER_SAME = {{MBR_GOODRX_COMPETITIVE_MULTIPLIER_SAME| default([None,1.1,1.1])}} 

#MBR_PRICING_TIER_CHAIN = {{MBR_PRICING_TIER_CHAIN | default([25, 50]) }}
MBR_PRICING_TIER_CHAIN = [25, 50]

#MBR_THRESHOLD_TYPE_CHAIN  = {{MBR_THRESHOLD_TYPE_CHAIN | default([None,"PERCENT","DOLLAR"])}}
MBR_THRESHOLD_TYPE_CHAIN = [None,"PERCENT","DOLLAR"]

#MBR_CHANGE_THRESHOLD_CHAIN  = {{MBR_CHANGE_THRESHOLD_CHAIN | default([None, -0.5, -25])}} 
MBR_CHANGE_THRESHOLD_CHAIN = [None, -0.5, -25]

MBR_GOODRX_COMPETITIVE_MULTIPLIER_CHAIN  = {{MBR_GOODRX_COMPETITIVE_MULTIPLIER_CHAIN | default([None,1.2,1.2])}}

#======================================
#Based on CLIENT COST SHARE PER CLAIM
#======================================

#CLNT_PRICING_TIER_SAME = {{CLNT_PRICING_TIER_SAME | default([5000000,10000000]) }}
CLNT_PRICING_TIER_SAME = [5000000,10000000]

#CLNT_THRESHOLD_TYPE_SAME = {{CLNT_THRESHOLD_TYPE_SAME | default([None,"PERCENT","PERCENT"])}}
CLNT_THRESHOLD_TYPE_SAME = [None,"PERCENT","PERCENT"]

#CLNT_CHANGE_THRESHOLD_SAME = {{CLNT_CHANGE_THRESHOLD_SAME| default([None, -0.5, -0.25])}}
CLNT_CHANGE_THRESHOLD_SAME = [None, -0.5, -0.25]

#CLNT_GOODRX_COMPETITIVE_MULTIPLIER_SAME = {{CLNT_GOODRX_COMPETITIVE_MULTIPLIER_SAME| default([None,2.0,2.0])}}
CLNT_GOODRX_COMPETITIVE_MULTIPLIER_SAME = [None,2.0,2.0]

#MBR_PRICING_TIER_CHAIN = {{CLNT_PRICING_TIER_CHAIN | default([25000000,50000000]) }} 
CLNT_PRICING_TIER_CHAIN = [25000000,50000000]

#MBR_THRESHOLD_TYPE_CHAIN  = {{CLNT_THRESHOLD_TYPE_CHAIN | default([None,"PERCENT","DOLLAR"])}}
CLNT_THRESHOLD_TYPE_CHAIN = [None,"PERCENT","DOLLAR"]

#MBR_CHANGE_THRESHOLD_CHAIN  = {{CLNT_CHANGE_THRESHOLD_CHAIN | default([None, -0.5, -25])}}
CLNT_CHANGE_THRESHOLD_CHAIN = [None, -0.5, -250000]

#CLNT_GOODRX_COMPETITIVE_MULTIPLIER_CHAIN  = {{CLNT_GOODRX_COMPETITIVE_MULTIPLIER_CHAIN | default([None,2.2,2.2])}}
CLNT_GOODRX_COMPETITIVE_MULTIPLIER_CHAIN = [None,2.2,2.2]

#====================================================================================================
# Next year pharm guarantees
PHARM_GUARANTEE_COMM_NY= {{PHARM_GUARANTEE_COMM_NY | default("'Pharmacy_Rate_Commercial_Next_Year_2025.csv'") }}
PHARM_GUARANTEE_MEDD_NY= {{PHARM_GUARANTEE_MEDD_NY | default("'Pharmacy_Rate_MEDD_Next_Year_2025.csv'") }}
PHARM_GUARANTEE_MEDICAID_NY= {{PHARM_GUARANTEE_MEDICAID_NY | default("'Pharmacy_Rate_Medicaid_Next_Year_2025.csv'") }}
CLIENT_GUARANTEE_MEDD_NY= {{CLIENT_GUARANTEE_MEDD_NY | default("'Client_Rate_MEDD_Next_Year_2025.csv'") }}
CUSTOMER_ID_MAPPING_MEDD_NY= {{CUSTOMER_ID_MAPPING_MEDD_NY | default("'CUSTOMER_ID_MAPPING_MEDD_NY_2025.csv'") }}
VCML_CROSSWALK_MEDD_2024 = {{VCML_CROSSWALK_MEDD_2024 | default("'VCML_CROSSWALK_MEDD_2024.csv'")}}
#=====================================================================================================

# Remove WTW Restriction
REMOVE_WTW_RESTRICTION= {{REMOVE_WTW_RESTRICTION | default(False) }}

# Parameter that restricts discrepancy between mail and retail prices
MAIL_RETAIL_BOUND = {{MAIL_RETAIL_BOUND | default(0.0) }} # imposes the constraint that mail price > MAIL_RETAIL_BOUND * retail price

# Parameter that restricts discrepancy between chains
RETAIL_RETAIL_BOUND = {{RETAIL_RETAIL_BOUND | default(0.0) }} # imposes the constraint that retail pharmacy x price > MAIL_RETAIL_BOUND * retail pharmacy y price, for all pharamcy x, pharmacy y \in p.PHARAMCY_LIST -{MCHOICE,MAIL}

#Set to True and add customer_id to the IGNORE_PERFORMANCE_CHECK_LIST in json gen
IGNORE_PERFORMANCE_CHECK = {{IGNORE_PERFORMANCE_CHECK | default(False)}}

# Parameter to ensure pharmacy guaranty rate with highest total AWP is selected for a specific CHAIN_GROUP and MEASUREMENT_CLEAN
PHARM_TARGET_RATE_RATIO = {{ PHARM_TARGET_RATE_RATIO | default(0.95) }}

# Parameter to ignore the mentioned VCMLS in QA.py and qa_checks.py 
IGNORED_VCMLS = {{ IGNORED_VCMLS | default(['10','12','13','SX','SX1','S3','S9', 'E23', 'N23','E92', '92', '24', 'E77', 'P77', '77', 'E78', '78','88','A1','A2','A3','A4','A5','L1','L2','L3','L4','L5','LD','M78','69','E69','70','E70','57','86','E86']) }}

# List of clients with both XA and XR EXTRL vcmls in vcml reference table but only either XA or XR in measurement mapping. This parameter will be removed once the vcml reference table is cleaned up with only active vcmls

XA_XR_LIST = {{XA_XR_LIST | default(['4667','3211','3645','4134','4139','4443', '4450', '5746','265C', '4683', '4047', '156C','295C', '176C' ]) }}

# Parameter to set Pref_Pharm=None for the file generated in preprocessing.py
PREF_PHARM_BYPASS= {{PREF_PHARM_BYPASS | default(False)}}

# If a client had a market check performed during the current contract, allows code to correctly calculate pre-market check performance
MARKET_CHECK = {{ MARKET_CHECK | default(False) }}

# Set cutoff below which code will automatically turn on CONFLICT_GPI_AS_TIERS
CONFLICT_GPI_CUTOFF = {{ CONFLICT_GPI_CUTOFF | default(500) }}

#================================================================
# Parameter to allow for benchmark price to be used in LP soft constraint when it comes to price movement penalization. 
# Default is USE_BENCHMARK_SC = False, where it will use current_mac in LP soft constraint
USE_BENCHMARK_SC = {{USE_BENCHMARK_SC | default(False)}}

#=================================================================
# Parameter to allow for benchmark price to be set as net cost guarantee in LP soft constraint when it comes to truecost clients
# Default is TRUECOST_CLIENT = False and UCL Client = False, where it will use current_mac in LP soft constraint
if TRUECOST_CLIENT:
    APPLY_VCML_PREFIX = {{APPLY_VCML_PREFIX | default("'TRU'") }}
elif UCL_CLIENT:
    APPLY_VCML_PREFIX = {{APPLY_VCML_PREFIX | default("'UCL'") }}
else:
    APPLY_VCML_PREFIX = {{APPLY_VCML_PREFIX | default("'MAC'") }}

#=================================================================
# Parameter that takes a value to be mulltiplied with net cost guarantee to get upper price bound when it comes to truecost clients
TRUECOST_UPPER_MULTIPLIER_GNRC = {{TRUECOST_UPPER_MULTIPLIER_GNRC | default(1.249)}}
TRUECOST_UPPER_MULTIPLIER_BRND = {{TRUECOST_UPPER_MULTIPLIER_BRND | default(1.149)}}

#=================================================================
# List of clients that will get same prcing for all vcmls. These clients are not adjudicating as expected based on the VCML setup.
UNIFORM_MAC_PRICING = {{UNIFORM_MAC_PRICING | default(False)}}

#=================================================================
# Parameter used to relax same therapeutic class drug constraints
# If SM_THERA_COLLAR is set to true, 
#    instead of using new_drug1/new_drug2 = ratio as constraints to maintain a fixed ratio
#    the constraints will be  (1-SM_THERA_COLLAR_LOW) * ratio <= new_drug1/new_drug2 <= (1+SM_THERA_COLLAR_HIGH) * ratio
# With new constraint setup to only maintaine position, COLLAR_LOW and COLLAR_HIGH lost it's bounding purposes.
# Given above, setting the default values for COLLAR_LOW and COLLAR_HIGH to extreme 
SM_THERA_COLLAR = {{SM_THERA_COLLAR | default(True)}}
if SM_THERA_COLLAR:
    SM_THERA_COLLAR_LOW = {{SM_THERA_COLLAR_LOW | default(1)}}
    SM_THERA_COLLAR_HIGH ={{SM_THERA_COLLAR_HIGH | default(100000)}}
    
#=================================================================
# Freeze the price when want LP constraints to be built based on current price but not wish LP to change futuer price
BRND_PRICE_FREEZE = {{BRND_PRICE_FREEZE | default(False)}}
GNRC_PRICE_FREEZE = {{GNRC_PRICE_FREEZE | default(False)}}