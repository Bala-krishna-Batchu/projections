#Imports
from google.cloud import bigquery, storage
import json
import socket
import datetime as dt
from batch_run_sql import query_data_quality, query_data_quality_ws, query_client_tracking
from google.cloud import storage
import argparse
from dateutil.relativedelta import relativedelta
from dateutil import parser as dt_parser

# Initializing current date variable and vm_name to be used in batch id and output path
year = dt.datetime.now().strftime('%Y')
month = dt.datetime.now().strftime('%B')
vm_name = f'{socket.gethostname()}'
month_short = dt.datetime.now().strftime('%b').upper()

# Initializing bq_client object used to query BQ
bq_client = bigquery.Client()

# Parse arguments fed from shell script
parser = argparse.ArgumentParser(description='execute a query with parameters')
parser.add_argument('--ODP_Event_Key', type=str, default='', help='Event key to identify the on demand pricing runs')
parser.add_argument('--CMPAS', type=str, default='False', help='key to identify the cmpas runs')
args = parser.parse_args()

# Query used in case the run is an On Demand Pricing run
if args.ODP_Event_Key != '':
    odp_query = f"""SELECT 
                        JSON_VALUE(ODP_Param_Values[0]['clientId']) AS ODP_Event_Customer_ID, 
                        JSON_VALUE(ODP_Param_Values[0]['targetGoLiveDate']) AS ODP_Target_Go_Live_Date,
                        COALESCE(JSON_VALUE(ODP_Param_Values[0]['scenario']),'DEFAULT_TEST') AS ODP_Run_Type, 
                        COALESCE(JSON_VALUE(ODP_Param_Values[0]['BQ_OUTPUT_DATASET']),'ds_development_lp') AS ODP_BQ_OUTPUT_DATASET,
                        JSON_VALUE(ODP_Param_Values[0]['maxPerDecreasePerGpi']) AS ODP_GPI_LOW_FAC, #pilot phase 1 ODP enabled
                        JSON_VALUE(ODP_Param_Values[0]['macUnrestrictedMail']) AS ODP_MAIL_MAC_UNRESTRICTED,
                        JSON_VALUE(ODP_Param_Values[0]['costSaver']) AS ODP_COSTSAVER_CLIENT,
                        JSON_VALUE(ODP_Param_Values[0]['leakageOptimization']) AS ODP_LEAKAGE_OPT,
                        
                        JSON_VALUE(ODP_Param_Values[0]['applyBenchmarkCap']) AS ODP_APPLY_BENCHMARK_CAP,  #pilot phase 1 ODP new
                        JSON_VALUE(ODP_Param_Values[0]['benchmarkCapMultiplier']) AS ODP_BENCHMARK_CAP_MULTIPLIER,
                        JSON_VALUE(ODP_Param_Values[0]['zeroQtyTightBounds']) AS ODP_ZERO_QTY_TIGHT_BOUNDS,
                        JSON_VALUE(ODP_Param_Values[0]['zeroQtyWeight']) AS ODP_ZERO_QTY_WEIGHT,
                        JSON_VALUE(ODP_Param_Values[0]['conflictGpiAsTiers']) AS ODP_CONFLICT_GPI_AS_TIERS,
                        JSON_VALUE(ODP_Param_Values[0]['conflictGpiCutoff']) AS ODP_CONFLICT_GPI_CUTOFF,
                        JSON_VALUE(ODP_Param_Values[0]['allowInterceptLimit']) AS ODP_ALLOW_INTERCEPT_LIMIT,
                        JSON_VALUE(ODP_Param_Values[0]['ignorePerformanceCheck']) AS ODP_IGNORE_PERFORMANCE_CHECK
                    FROM pbm-mac-lp-prod-ai.psot.odp_param_intake_tbl 
                    WHERE ODP_Event_Key = '{args.ODP_Event_Key}'"""

# Client IDs to run using LP. All customer IDs needing a run should be in the below list. 
# Subsequent lists can contain subsets of these IDs, added only when parameters need to be modified.
customer_ids = [ # IDs are alphanumeric with 2 to 5 characters

]

# Clients who need a price override file should be added to this list
price_override_list = [

]

# Clients with infeasibility errors should be added in the list 
handle_infeasible_list = [

]

#--------------------------------DE ERRORS ---------------------------------
# Ad hoc parameter to be used when clients are failing with error where client last_data does not equal to the max DOF 
# (seen when using custom tables that are not updated with the same frequency as other DE tables)
max_dof_dict = {
# '152EH': "dt.datetime.strptime('06/28/2024', '%m/%d/%Y')",
}

# client with duplicate chain group to be added to the list
duplicate_chain_group_lst = [
]
# --------------------------------DE ERRORS ---------------------------------

#---------------------------MISC ADJUSTMENTS -------------------------------
# Default Price levels are as follows: lvl 0 for non-CS clients and lvl 1 for CS clients; 
# If price levels need to be adjusted, add customer ids to these list to change price level of a run
price_level_1 = [

]

price_level_2 = [
    
]

price_level_3 = [
     
]
# ---------------------------MISC ADJUSTMENTS -------------------------------

# ---------------------------MAIL OP AND SPECIAL PARAMETERS ---------------------------
# Use manually coded Costsaver/Non-Costsaver lists when clnt_params is not accurate
# e.g. during Welcome Season if another directive has been given
"""
ncs_clients = [

]

cs_clients = [

]
"""
gpi_low_fac_01 = [

]

gpi_low_fac_02 = [

]

gpi_low_fac_04 = [

]

gpi_low_fac_05 = [

]

gpi_up_fac_025 = [

]

mail_floors_factor_01 = [

]

client_unrperf_pen_02 = [

]

client_unrperf_pen_05 = [

]

mail_unrestricted_cap_35 = [

]

tiered_price_lim_false = [

]
# ---------------------------MAIL OP AND SPECIAL PARAMETERS ---------------------------

# ---------------------------SET BATCH PREFIX ----------------------------------------
# Automate changing iteration 
storage_client = storage.Client()
blobs = storage_client.list_blobs(r'pbm-mac-lp-prod-ai-bucket', prefix=f'Output/{month}{year}/{vm_name}/', delimiter='/')
iterate = [b for b in blobs]
num_list =[]
new_list = [prefix for prefix in blobs.prefixes if month in prefix]
if len(new_list) == 0:
    new_iter= 1.0
else:
    for prefix in new_list:
        prefix = prefix.split(f'Output/{month}{year}/{vm_name}/')[1]
        for i in prefix.split():
            newStr = i.replace(".","")
            newStr = newStr.replace("/","")
            if newStr.isnumeric():
                num_list.append(i.replace("/",""))
        new_iter = round((float(max(num_list))+0.01), 2)

# Set batch prefix
batch_prefix = f'{month_short}{year}_Runs_{new_iter}'
# ---------------------------SET BATCH PREFIX ----------------------------------------

# ---------------------------BASE PARAMETERS FOR ALL CLIENTS ----------------------------------------
TIMESTAMP = dt.datetime.now().strftime('%Y-%m-%d_%H%M%S%f')
BATCH_ID = batch_prefix #+ TIMESTAMP
USER = "'" + vm_name + "'"
USER = USER.replace('-', '_')
PROGRAM_INPUT_PATH = 'gs://pbm-mac-lp-prod-ai-bucket/shared_input'
READ_FROM_BQ = True
WRITE_TO_BQ = True
TRACK_RUN = False    #Change to True for run to show up in batch run tracker
BQ_OUTPUT_DATASET = 'ds_development_lp' #Change to production when needed
REMOVE_BAD_DQ_CLIENTS = True 
# ---------------------------BASE PARAMETERS FOR ALL CLIENTS ----------------------------------------
    
# --------------------------- Parameters for which confirmation was provided --------------------------------
# The parameter list can change run to run and initiative to initiative so validation is mandatory
# Not all of the parameters that can be used are listed below, for full list of parameters see the 
# CPMO_parameters_TEMPLATE.py file on the GER_LP_Code folder.

GO_LIVE = "dt.datetime.strptime('01/01/2025', '%m/%d/%Y')"
CONFLICT_GPI_AS_TIERS = True
FULL_YEAR = False
CLIENT_LOB = 'CMK'
# Price bounds
TIERED_PRICE_LIM = True   # For WTW clients, values are set below. Value is TRUE for Non WTW clients.
GPI_LOW_FAC = 0.70
# Zero QTY
ZERO_QTY_TIGHT_BOUNDS = True
ZERO_QTY_WEIGHT = 10
# Mail Pricing 
APPLY_FLOORS_MAIL = True
MAIL_FLOORS_FACTOR = 0.7
# Target guarantee buffers 
CLIENT_TARGET_BUFFER = 0.0
CLIENT_MAIL_BUFFER = 0.0
PHARMACY_TARGET_BUFFER = 0.0
#Cost saver parameters
HANDLE_MAIL_CONFLICT_LEVEL = "'1'"
ALLOW_INTERCEPT_LIMIT = True
if args.ODP_Event_Key != '':
    #Pilot 1 ODP Project 
    APPLY_BENCHMARK_CAP = True
    BENCHMARK_CAP_MULTIPLIER = 5.0
    CONFLICT_GPI_AS_TIERS = False
    CONFLICT_GPI_CUTOFF = 500
    GPI_LOW_FAC = 0.60
    IGNORE_PERFORMANCE_CHECK=False
    #Pilot 1 ODP - should not populate
    MAIL_MAC_UNRESTRICTED = False 
    COSTSAVER_CLIENT = False
    LEAKAGE_OPT = False

# If an On Demand Pricing run, fill in parameters sent from the ODP UI stored the ODP Intake Table
if args.ODP_Event_Key != '':
    odp_query_job = bq_client.query(odp_query)
    odp_param_records = [dict(row) for row in odp_query_job]
    for i in range(len(odp_param_records)):
        customer_ids.append(odp_param_records[i]['ODP_Event_Customer_ID'])
        GO_LIVE = f"""dt.datetime.strptime('{odp_param_records[i]['ODP_Target_Go_Live_Date']}', '%Y-%m-%d')"""
        RUN_TYPE_TABLEAU = odp_param_records[i]['ODP_Run_Type']
        BQ_OUTPUT_DATASET = odp_param_records[i]['ODP_BQ_OUTPUT_DATASET']
        #pilot ODP phase 1
        GPI_LOW_FAC = odp_param_records[i]['ODP_GPI_LOW_FAC']
        MAIL_MAC_UNRESTRICTED = odp_param_records[i]['ODP_MAIL_MAC_UNRESTRICTED']
        COSTSAVER_CLIENT = odp_param_records[i]['ODP_COSTSAVER_CLIENT']
        LEAKAGE_OPT = odp_param_records[i]['ODP_LEAKAGE_OPT']
        APPLY_BENCHMARK_CAP = odp_param_records[i]['ODP_APPLY_BENCHMARK_CAP']
        BENCHMARK_CAP_MULTIPLIER = odp_param_records[i]['ODP_BENCHMARK_CAP_MULTIPLIER']
        ZERO_QTY_TIGHT_BOUNDS = odp_param_records[i]['ODP_ZERO_QTY_TIGHT_BOUNDS']
        ZERO_QTY_WEIGHT = odp_param_records[i]['ODP_ZERO_QTY_WEIGHT']
        CONFLICT_GPI_AS_TIERS = odp_param_records[i]['ODP_CONFLICT_GPI_AS_TIERS']
        CONFLICT_GPI_CUTOFF = odp_param_records[i]['ODP_CONFLICT_GPI_CUTOFF']
        ALLOW_INTERCEPT_LIMIT = odp_param_records[i]['ODP_ALLOW_INTERCEPT_LIMIT']
        IGNORE_PERFORMANCE_CHECK = odp_param_records[i]['ODP_IGNORE_PERFORMANCE_CHECK']
# --------------------------- Parameters for which confirmation was provided --------------------------------

# --------------------------------------------- QUERY CLIENTS -------------------------------------------------------
# Middle term below is to add a "'" to the end of 4-digit customer IDs (or "']" to the end of 3-digit) to match the 5-character substring in clnt_params
customer_id_str = "'"+ "','".join(map(str, customer_ids)) + "'" 
if CLIENT_LOB == 'AETNA':
    # WS clnt_params/taxonomy for AETNA is the same as monthly taxonomy
    query = f"""SELECT * EXCEPT(CUSTOMER_ID), REPLACE(REPLACE(CUSTOMER_ID, "['", ""), "']", '') AS CUSTOMER_ID 
    FROM `anbc-prod.fdl_gdp_ae_ent_enrv_prod.gms_aetna_clnt_params`
    WHERE REPLACE(REPLACE(CUSTOMER_ID, "['", ""), "']", '') IN ({customer_id_str})"""
    taxonomy_query = f"""SELECT customer_id, client_name
    FROM `anbc-prod.fdl_gdp_ae_ent_enrv_prod.gms_aetna_ger_opt_taxonomy_final`
    WHERE customer_id IN ({customer_id_str})"""
elif CLIENT_LOB == 'CMK':
    if FULL_YEAR:
        query = f"""SELECT * EXCEPT(CUSTOMER_ID), REPLACE(REPLACE(CUSTOMER_ID, "['", ""), "']", '') AS CUSTOMER_ID 
        FROM `pbm-mac-lp-prod-ai.pricing_management.clnt_params_ws`
        WHERE REPLACE(REPLACE(CUSTOMER_ID, "['", ""), "']", '') IN ({customer_id_str})
        """
        taxonomy_query = f"""SELECT customer_id, client_name, CASE WHEN UPPER(client_name) LIKE '%MEDICAID%' THEN 'MEDICAID'
        WHEN medd = 0 AND egwp = 0 THEN 'COMMERCIAL' 
        ELSE 'MEDD' END AS client_type,  
        FROM `anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_TAXONOMY_FINAL_ws`
        WHERE customer_id IN ({customer_id_str})"""
    else: 
        query = f"""SELECT * EXCEPT(CUSTOMER_ID), REPLACE(REPLACE(CUSTOMER_ID, "['", ""), "']", '') AS CUSTOMER_ID
        FROM `pbm-mac-lp-prod-ai.pricing_management.clnt_params`
        WHERE REPLACE(REPLACE(CUSTOMER_ID, "['", ""), "']", '') IN ({customer_id_str})
        """
        taxonomy_query = f"""SELECT customer_id, client_name, CASE WHEN UPPER(client_name) LIKE '%MEDICAID%' THEN 'MEDICAID'
        WHEN medd = 0 AND egwp = 0 THEN 'COMMERCIAL' 
        ELSE 'MEDD' END AS client_type, 
        FROM `anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_TAXONOMY_FINAL`
        WHERE customer_id IN ({customer_id_str})"""

query_job = bq_client.query(query)
tax_query_job = bq_client.query(taxonomy_query)
# --------------------------------------------- QUERY CLIENTS -------------------------------------------------------

# ------------------------------ INITIATING QUERIES AND CUSTOMER_ID FOR LOOP -----------------------------------------------------
records = [dict(row) for row in query_job]
records_tax = [dict(row) for row in tax_query_job]

client_list = [rec['CUSTOMER_ID'] for rec in records]
cs_list = [rec['CUSTOMER_ID'] for rec in records if rec['COSTSAVER_CLIENT']]
locked_list = [rec['CUSTOMER_ID'] for rec in records if rec['LOCKED_CLIENT']]

if len(set(customer_ids) - set(client_list)) > 0:
    print(f"Clients {set(customer_ids) - set(client_list)} contained in the client_params table as a speciality client or is not in client_params and therefore cannot be run.")
    tax_list = [f'''{rec['customer_id']} : {rec['client_name']}''' for rec in records_tax if rec['customer_id'] in  set(customer_ids) - set(client_list)]
    print(f"Clients {tax_list} is/are not able to be run but are in taxonomy_final.")
    for i in range(len(records_tax)):
        dq_rows = []
        curr_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dq_rows.append(f"('{records_tax[i]['customer_id']}', '{records_tax[i]['client_name']}','{records_tax[i]['client_type']}', '---', '---', '---', '{curr_time}', 'Failed', 'Data Quality', 'Not in clnt_params', '{BATCH_ID}')")
    dq_values = ',\n'.join(dq_rows)
    bq_client.query(query_client_tracking.format(dq_values=dq_values))


# remove clients with data quality issues
dq_clients = dict()
clients_to_remove = []
if REMOVE_BAD_DQ_CLIENTS:
    # Make sure the full_year is set to true and GO_LIVE is equal to current year plus 1 
    if FULL_YEAR and (dt_parser.parse(GO_LIVE, fuzzy=True).year == dt.datetime.now().year+1):
        query_job_dq = bq_client.query(query_data_quality_ws.format(customer_id_str="'"+','.join(list(map(str,customer_ids))).replace(",", "','")+"'"))
    else:
        query_job_dq = bq_client.query(query_data_quality.format(customer_id_str="'"+','.join(list(map(str,customer_ids))).replace(",", "','")+"'"))
    records_dq = [dict(row) for row in query_job_dq]
    for d in records_dq:
        dq_clients[d['customer_id']] = dict(error_message=d['error_message'])


        
# For loop that goes through the client id : params dictionary created to generate params.json that we send to kubeflow for runs on kubernetes
for i in range(len(records)):
    records[i]['TIMESTAMP'] = TIMESTAMP
    records[i]['USER'] = USER
    #Removing outer quotes in CLIENT NAME TABLEAU
    records[i]['CLIENT_NAME_TABLEAU'] = records[i]['CLIENT_NAME_TABLEAU'][1:-1]
    CUSTOMER_ID = records[i]["CUSTOMER_ID"]
    records[i]["CUSTOMER_ID"] = "['"+ records[i]["CUSTOMER_ID"] + "']"
    cleaned_TIMESTAMP = TIMESTAMP.replace("-","")
    DATA_ID = f"'{month}_{USER[1:-1]}_{CUSTOMER_ID}_{cleaned_TIMESTAMP}'"
    records[i]['DATA_ID'] = DATA_ID    
    records[i]['BATCH_ID'] = BATCH_ID
    records[i]['PROGRAM_INPUT_PATH'] = PROGRAM_INPUT_PATH
    records[i]['PROGRAM_OUTPUT_PATH'] = f"""gs://pbm-mac-lp-prod-ai-bucket/Output/{month}{year}/{vm_name}/{batch_prefix}/{CUSTOMER_ID+'_'+records[i]["CLIENT_NAME_TABLEAU"].replace(' - ','_').replace(' ','_')}"""
    records[i]['READ_FROM_BQ'] = READ_FROM_BQ
    records[i]['WRITE_TO_BQ'] = WRITE_TO_BQ
    records[i]['TRACK_RUN'] = TRACK_RUN
    records[i]['BQ_OUTPUT_DATASET'] = BQ_OUTPUT_DATASET
    records[i]['CONFLICT_GPI_AS_TIERS'] = CONFLICT_GPI_AS_TIERS
    records[i]['CLIENT_LOB'] = CLIENT_LOB
    
    # Dates
    records[i]['GO_LIVE'] = GO_LIVE
    records[i]['FULL_YEAR'] = FULL_YEAR

    # Price bounds
    records[i]['TIERED_PRICE_LIM'] = TIERED_PRICE_LIM
    records[i]['GPI_LOW_FAC'] = GPI_LOW_FAC

    # Zero QTY
    records[i]['ZERO_QTY_TIGHT_BOUNDS'] = ZERO_QTY_TIGHT_BOUNDS
    records[i]['ZERO_QTY_WEIGHT'] = ZERO_QTY_WEIGHT
    
    # Mail Pricing 
    records[i]['APPLY_FLOORS_MAIL'] = APPLY_FLOORS_MAIL
    records[i]['MAIL_FLOORS_FACTOR'] = MAIL_FLOORS_FACTOR
    
    # Target guarantee buffers 
    records[i]['CLIENT_TARGET_BUFFER'] = CLIENT_TARGET_BUFFER
    records[i]['CLIENT_MAIL_BUFFER'] = CLIENT_MAIL_BUFFER
    records[i]['PHARMACY_TARGET_BUFFER'] = PHARMACY_TARGET_BUFFER
    
    # CostSaver Parameters
    records[i]['HANDLE_MAIL_CONFLICT_LEVEL'] = HANDLE_MAIL_CONFLICT_LEVEL
    records[i]['ALLOW_INTERCEPT_LIMIT'] = ALLOW_INTERCEPT_LIMIT
    
    records[i]['DATA_START_DAY'] = records[i]['DATA_START_DAY'][1:-1]   
    
    if args.ODP_Event_Key != '':
        #pilot phase 1
        records[i]['APPLY_BENCHMARK_CAP'] = APPLY_BENCHMARK_CAP
        records[i]['BENCHMARK_CAP_MULTIPLIER'] = BENCHMARK_CAP_MULTIPLIER
        records[i]['CONFLICT_GPI_CUTOFF'] = CONFLICT_GPI_CUTOFF
        records[i]['MAIL_MAC_UNRESTRICTED'] = MAIL_MAC_UNRESTRICTED
        records[i]['COSTSAVER_CLIENT'] = COSTSAVER_CLIENT
        records[i]['LEAKAGE_OPT'] = LEAKAGE_OPT
        records[i]['IGNORE_PERFORMANCE_CHECK'] = IGNORE_PERFORMANCE_CHECK
# ------------------------------ INITIATING QUERIES AND CUSTOMER_ID FOR LOOP -----------------------------------------------------

#--------------------------------COST SAVER -----------------------------    
    if CUSTOMER_ID in cs_list:
        cost_saver = 'CS'
        price_level = '1'
        records[i]['UPPER_BOUND'] = [8,25,50,100,999999]
        records[i]['MAX_PERCENT_INCREASE'] = [999999, 0.6, 0.35, 0.25, 0.15]
        records[i]['MAX_DOLLAR_INCREASE'] = [8, 999999, 999999, 999999, 999999]
        if CUSTOMER_ID in price_level_2:
            price_level = '2'
            records[i]['UPPER_BOUND'] = [8,25,50,100,999999]
            records[i]['MAX_PERCENT_INCREASE'] = [999999, 1, 0.75, 0.35, 0.25]
            records[i]['MAX_DOLLAR_INCREASE'] = [10, 999999, 999999, 999999, 999999]
        if CUSTOMER_ID in price_level_3:
            price_level='3'
            records[i]['UPPER_BOUND'] = [3, 6, 999999]
            records[i]['MAX_PERCENT_INCREASE'] = [999999, 999999, 3]
            records[i]['MAX_DOLLAR_INCREASE'] = [20, 30, 999999]
    elif CUSTOMER_ID not in cs_list:
        price_level='0'
        records[i]['UPPER_BOUND'] = [5, 10, 25, 50, 100, 999999]
        records[i]['MAX_PERCENT_INCREASE'] = [999999, 1.0, 0.40, 0.30, 0.20, 0.10]
        records[i]['MAX_DOLLAR_INCREASE'] = [5, 999999, 999999, 999999, 999999, 999999]
        if CUSTOMER_ID in price_level_1:
            price_level='1'
            records[i]['UPPER_BOUND'] = [8,25,50,100,999999]
            records[i]['MAX_PERCENT_INCREASE'] = [999999, 0.6, 0.35, 0.25, 0.15]
            records[i]['MAX_DOLLAR_INCREASE'] = [8, 999999, 999999, 999999, 999999]
        if CUSTOMER_ID in price_level_2:
            price_level='2'
            records[i]['UPPER_BOUND'] = [8,25,50,100,999999]
            records[i]['MAX_PERCENT_INCREASE'] = [999999, 1, 0.75, 0.35, 0.25]
            records[i]['MAX_DOLLAR_INCREASE'] = [10, 999999, 999999, 999999, 999999]
        if CUSTOMER_ID in price_level_3:
            price_level='3'
            records[i]['UPPER_BOUND'] = [3, 6, 999999]
            records[i]['MAX_PERCENT_INCREASE'] = [999999, 999999, 3]
            records[i]['MAX_DOLLAR_INCREASE'] = [20, 30, 999999]
        cost_saver = 'NCS'
#--------------------------------COST SAVER -----------------------------  

# --------------------------------- EXCEPTION CLIENTS ---------------------------
    # Increase price levels for AON, WTW, or IBM 
    if (('WTW' in records[i]['CLIENT_NAME_TABLEAU']) or ('AON' in records[i]['CLIENT_NAME_TABLEAU']) or CUSTOMER_ID == '4514'):
        records[i]['TIERED_PRICE_LIM'] = False
        records[i]['ALLOW_INTERCEPT_LIMIT'] = False
        records[i]['GPI_UP_FAC'] = 0.24
        records[i]['GPI_LOW_FAC'] = 0.25

    # Increase price levels for MVP and Healthfirst 
    if (('MVP' in records[i]['CLIENT_NAME_TABLEAU']) or ('HEALTHFIRST' in records[i]['CLIENT_NAME_TABLEAU'].upper()) or CUSTOMER_ID == '185C' or CUSTOMER_ID == '183C'):
        records[i]['TIERED_PRICE_LIM'] = False
        records[i]['ALLOW_INTERCEPT_LIMIT'] = False
        records[i]['GPI_UP_FAC'] = 0.10
        records[i]['GPI_LOW_FAC'] = 0.25
        
    # CCP Clients 
    if records[i]['CROSS_CONTRACT_PROJ']:
        GO_LIVE_dt = GO_LIVE.strip("dt.datetime.strptime(GO_LIVE, '")
        GO_LIVE_dt = GO_LIVE_dt.strip("', '%m/%d/%Y')")
        GO_LIVE_dt = dt.datetime.strptime(GO_LIVE_dt, '%m/%d/%Y')
        DATA_START_DAY = GO_LIVE_dt - relativedelta(years=1)
        DATA_START_DAY = DATA_START_DAY.strftime("%Y-%m-%d")
        records[i]['DATA_START_DAY'] = DATA_START_DAY
# --------------------------------- EXCEPTION CLIENTS ---------------------------

# --------------------------------------MAIL OP AND SPECIAL PARAMETERS -----------------------------------------------
    if CUSTOMER_ID not in cs_list:
        records[i]['INTERCEPTOR_OPT'] = False
        cost_saver = 'NCS'
      
    if CUSTOMER_ID in cs_list:
        records[i]['INTERCEPTOR_OPT'] = True
        cost_saver = 'CS'

    if CUSTOMER_ID in client_unrperf_pen_02:
        records[i]['CLIENT_RETAIL_UNRPERF_PEN'] = 0.2
        
    if CUSTOMER_ID in client_unrperf_pen_05:
        records[i]['CLIENT_RETAIL_UNRPERF_PEN'] = 0.5
        
    if CUSTOMER_ID in gpi_low_fac_01:
        records[i]['GPI_LOW_FAC'] = 0.1
        
    if CUSTOMER_ID in gpi_low_fac_02:
        records[i]['GPI_LOW_FAC'] = 0.2
        
    if CUSTOMER_ID in gpi_low_fac_04:
        records[i]['GPI_LOW_FAC'] = 0.4
    
    if CUSTOMER_ID in gpi_low_fac_05:
        records[i]['GPI_LOW_FAC'] = 0.5
        
    if CUSTOMER_ID in gpi_up_fac_025:
        records[i]['GPI_UP_FAC'] = 0.25
    
    if CUSTOMER_ID in mail_floors_factor_01:
        records[i]['MAIL_FLOORS_FACTOR'] = 1.0
        
    if CUSTOMER_ID in mail_unrestricted_cap_35:
        records[i]['MAIL_UNRESTRICTED_CAP'] = 3.5
        
    if CUSTOMER_ID in tiered_price_lim_false:
        records[i]['TIERED_PRICE_LIM'] = False
# --------------------------------------MAIL OP AND SPECIAL PARAMETERS -----------------------------------------------

# ------------------------------------ JOIN ERROR CLIENTS ------------------------------------------
# Example of how to assign pharmacies to clients that fail with no AWP on certain small capped pharmacies which need to be moved to non-capped list
# (Error thrown will be lp_vol_df error in DIR)

    # if CUSTOMER_ID == '3883':
    #     records[i]['NON_CAPPED_PHARMACY_LIST'] = "['ART','HYV','CST','AHS', 'TPS']"
    #     records[i]['PSAO_LIST'] = "[ 'HMA' , 'ELE' , 'EPC' , 'CAR']"
    #     records[i]['SMALL_CAPPED_PHARMACY_LIST'] = "['MCHOICE', 'PBX' , 'ABS' , 'GEN' , 'AHD' , 'MJR']"
    #     records[i]['COGS_PHARMACY_LIST'] = "['NONPREF_OTH']"
# ------------------------------------ JOIN ERROR CLIENTS ------------------------------------------

#---------------------------------------------COMMON ERRORS -----------------------------------------
    if CUSTOMER_ID in handle_infeasible_list: 
        records[i]['HANDLE_INFEASIBLE'] = True

    if CUSTOMER_ID in price_override_list:
        records[i]['PRICE_OVERRIDE'] = True
        records[i]['PRICE_OVERRIDE_FILE'] = f"'Price_Overrides_{CUSTOMER_ID}_{month}.csv'"
        
    # if CUSTOMER_ID in max_dof_dict.keys():
    #         records[i]['LAST_DATA'] = max_dof_dict[CUSTOMER_ID]

    # Duplicate chain groups 
    if CUSTOMER_ID in duplicate_chain_group_lst:
        records[i]['COGS_PHARMACY_LIST'] = '[]'
    
    # save info for clients with data quality issues
    if REMOVE_BAD_DQ_CLIENTS and CUSTOMER_ID in dq_clients:
        dq_clients[CUSTOMER_ID]['CLIENT_NAME_TABLEAU'] = records[i]['CLIENT_NAME_TABLEAU']
        dq_clients[CUSTOMER_ID]['CLIENT_TYPE'] = records[i]['CLIENT_TYPE']
        clients_to_remove.append(i)
# ---------------------------------------------COMMON ERRORS -----------------------------------------

# ---------------------------------------------RUN_TYPE_TABLEAU -----------------------------------------
    # Logic to automate RUN_TYPE_TABLEAU so that rerun parameters that stray from monthly run parameters will be noted in run_type
if args.ODP_Event_Key != '':
    records[i]['RUN_TYPE_TABLEAU'] = RUN_TYPE_TABLEAU
else:
    RUN_TYPE_TABLEAU = f'{month_short}{year} {cost_saver} lvl{price_level}'
    records[i]['RUN_TYPE_TABLEAU'] = RUN_TYPE_TABLEAU
    if (records[i]['INTERCEPTOR_OPT'] is False) & (cost_saver == 'CS'):
        RUN_TYPE_TABLEAU = RUN_TYPE_TABLEAU + ' NCS'
        records[i]['RUN_TYPE_TABLEAU'] = RUN_TYPE_TABLEAU
    if CUSTOMER_ID not in locked_list:
        RUN_TYPE_TABLEAU = RUN_TYPE_TABLEAU + ' SP'
        records[i]['RUN_TYPE_TABLEAU'] = RUN_TYPE_TABLEAU
    else:
        RUN_TYPE_TABLEAU = RUN_TYPE_TABLEAU + ' ZBD'
        records[i]['RUN_TYPE_TABLEAU'] = RUN_TYPE_TABLEAU
    try:
        if (('WTW' in records[i]['CLIENT_NAME_TABLEAU']) or ('AON' in records[i]['CLIENT_NAME_TABLEAU']) or CUSTOMER_ID == '4514') & (records[i]['GPI_UP_FAC'] == 0.24):
            RUN_TYPE_TABLEAU = RUN_TYPE_TABLEAU + '- Flat 24%'
            records[i]['RUN_TYPE_TABLEAU'] =  RUN_TYPE_TABLEAU
        if (('MVP' in records[i]['CLIENT_NAME_TABLEAU']) or ('HEALTHFIRST' in records[i]['CLIENT_NAME_TABLEAU'].upper()) or CUSTOMER_ID == '185C' or CUSTOMER_ID == '183C') & (records[i]['GPI_UP_FAC'] == 0.10):
            RUN_TYPE_TABLEAU = RUN_TYPE_TABLEAU + '- Flat 10%'
            records[i]['RUN_TYPE_TABLEAU'] =  RUN_TYPE_TABLEAU
    except:
        pass
    if records[i]['TIERED_PRICE_LIM'] != True:
        tpl = records[i]['TIERED_PRICE_LIM']
        RUN_TYPE_TABLEAU = RUN_TYPE_TABLEAU + f' TIERED_PRICE_LIM = {tpl}'
        records[i]['RUN_TYPE_TABLEAU'] = RUN_TYPE_TABLEAU
        
    print(f"""For customer {records[i]["CUSTOMER_ID"]} run_type is {records[i]['RUN_TYPE_TABLEAU']}.""")
# ---------------------------------------------RUN_TYPE_TABLEAU -----------------------------------------

# -----------------------------------------------------------------------------------------------------------
# remove clients by index
for i in sorted(clients_to_remove, reverse=True):
    del records[i]

if REMOVE_BAD_DQ_CLIENTS and TRACK_RUN:
    dq_rows = []
    for k, v in dq_clients.items():
        err_msg = str(v['error_message'])
        client_nm = str(v['CLIENT_NAME_TABLEAU'])
        client_type = str(v['CLIENT_TYPE'])
        curr_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dq_rows.append(f"('{k}', '{client_nm}','{client_type}', '---', '{RUN_TYPE_TABLEAU}', '---', '{curr_time}', 'Failed', 'Data Quality', '{err_msg}', '{BATCH_ID}')")
        print(err_msg)
    dq_values = ',\n'.join(dq_rows)
    bq_client.query(query_client_tracking.format(dq_values=dq_values))

if args.ODP_Event_Key != '':
    storage_client = storage.Client()
    bucket = storage_client.get_bucket('pbm-mac-lp-prod-user')
    blob = bucket.blob('clientpharmacymacoptimization/client_params.json')
    blob.upload_from_filename('client_params.json')

assert len(records) > 0, 'No clients were eligible to run'

cmpas_query = """ 
    SELECT DISTINCT
    customer_id,
    FROM `pbm-mac-lp-prod-ai.pricing_management.inclusion_table`
    WHERE rec_curr_ind = 'Y' AND CURRENT_DATE() BETWEEN start_date AND end_date and algo = 'CMPAS'
    """
cmpas_incl = bq_client.query(cmpas_query).to_dataframe()
cmpas_incl = cmpas_incl['customer_id'].to_list()
cmpas_clients = [i for i in customer_ids if i in cmpas_incl]
non_cmpas_clients = [i for i in customer_ids if i not in cmpas_incl]
if args.CMPAS == 'True':
    assert len(non_cmpas_clients)==0, 'Non-CMPAS clients are included in the customer_id list'
else:
    assert len(cmpas_clients)==0, 'CMPAS clients are included in the customer_id list'

with open('client_params.json', 'w') as outfile:
    json.dump(records, outfile)

#Printing out the order of the customer id that will be run
print('BATCH_ID:',BATCH_ID)
print_stuff = []
for i in records:
    print_stuff.append(i['CUSTOMER_ID'])
print_stuff.reverse()
for i in print_stuff:
    print(i)
