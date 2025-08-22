#Imports
from google.cloud import bigquery
import json
import socket
import datetime as dt
from batch_run_sql import query_data_quality, query_client_tracking
import argparse 

parser = argparse.ArgumentParser(description='CICD Param')
parser.add_argument('-u','--batch_id', help='batch_id', required=True)

args = vars(parser.parse_args())
batch_id_var = args['batch_id'] 


custom_params = []

based_parameters = {
    "TIMESTAMP": '\"'+ dt.datetime.now().strftime('%Y%m%d') +'\"',
    "USER": '\"'+ socket.gethostname() +'\"',
    "BQ_INPUT_PROJECT_ID": 'pbm-mac-lp-prod-ai',
    "BQ_INPUT_DATASET_ENT_CNFV_PROD": "ds_lp_cicd_anbc_temp",
    "BQ_INPUT_DATASET_ENT_ENRV_PROD": "ds_lp_cicd_anbc_temp",
    "BQ_INPUT_DATASET_DS_PRO_LP": "ds_lp_cicd_anbc_temp", 
    "BQ_INPUT_DATASET_SANDBOX": "ds_lp_cicd_anbc_temp",
    "BQ_OUTPUT_PROJECT_ID": 'pbm-mac-lp-prod-ai',
    "BQ_OUTPUT_DATASET": 'ds_development_lp',
    "PROGRAM_INPUT_PATH": 'gs://pbm-mac-lp-prod-ai-bucket/shared_input',
    "READ_FROM_BQ": True,
    "WRITE_TO_BQ": False,
    "UNC_OPT": False,
    "LAST_DATA": "dt.datetime.strptime('06/30/2021', '%m/%d/%Y')",
    "GO_LIVE": "dt.datetime.strptime('07/18/2021', '%m/%d/%Y')",
    "FLOOR_GPI_LIST": "'20201209_Floor_GPIs.csv'",
    "FLOOR_PRICE": True,
    "UNC_ADJUST": False,
    "DATA_START_DAY": '2021-01-01',
    "TIERED_PRICE_LIM": True,
    "APPLY_GENERAL_MULTIPLIER":False,
    "APPLY_MAIL_MULTIPLIER":False,
    "GPI_UP_FAC": 0.2,
    "GPI_LOW_FAC":0.8, 
    "TRACK_RUN": True, ## If you wish to track the cicd run set it to True
    "BATCH_ID": batch_id_var,
    "RUN_TYPE_TABLEAU": "CICD"
}

#Client original list
# customer_ids = ['2020', # MED-D client
#                 '2023', # MED-D client
#                 '3016', # WRAP client + B/G offset
#                 '3061', # EDWP & WRAP client
#                 '3066', # EDWP & WRAP client + B/G offset
#                 '3152', # GRx candidate - commercial
#                 '3731', # Locked
#                 '4025', # Medicaid + B/G offset
#                 '4454', # Vanilla - commercial
#                 '4475', # UNC candidate - commercial
#                 '4490', # WTW client
#                 '4588', # NonOffsetting - commercial
#                 '4658'] # EDWP client


customer_ids = ['5084',  # MCHOICE KRG and CVS Pharmacy
                '3066',  # EDWP & WRAP client + B/G offset
                '3152',  # GRx candidate - commercial
                '3731',  # Locked
                '4475',  # UNC candidate - commercial
                '4490',  # WTW client
                '4588',
                '2012',  # MedD with P/NP
                '4658',  # EDWP client
                '4442',  # State Parity client/Offsetting commercial
                '8230',  # Locked R90OK client
                '4520',  # Transparent R90OK client
                'A1002', # Aetna CS, Leakage OPT, AGG
                'A1009', # Aetna NonCS, Leakage OPT, 2 VCML
                #'A1446',# Aetna CS, NonLeakage, ----- Comment out the client as it was failing with infeasibility even after setting the HANDLE_INFEASIBLE to True ---
                'A1028', # Aetna Offsetting, CS, NonLeakage,
                'A2201'] # Aetna multiple vcmls
                 

custom_params_5084_no_cs_no_intrcpt = {
    "CUSTOMER_ID": "['5084']",
    "CLIENT_NAME_TABLEAU":"MERCER NCP - DUN AND BRADSTREET (JAN-JUN)",
    "DATA_ID": "'NO_CS_NO_INTRCPT_2025_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/5084_NO_CS_NO_INTRCPT_2025""",
    "BIG_CAPPED_PHARMACY_LIST": "['CVS','RAD','WAG','KRG','WMT']",
    "SMALL_CAPPED_PHARMACY_LIST": "['MCHOICE', 'ABS' , 'PBX' , 'AHD' , 'AMZ' , 'GEN' , 'HYV' , 'GIE' , 'MJR']",
    "NON_CAPPED_PHARMACY_LIST": "['ART','CST','AHS','CHD', 'TPS' , 'RUR']",
    "COGS_PHARMACY_LIST": "['NONPREF_OTH']",
    "PSAO_LIST": "[ 'EPC' , 'HMA' , 'CAR' , 'ELE']",
    "CLIENT_TYPE": "COMMERCIAL",
    "OVER_REIMB_CHAINS": "['CVS', 'RAD', 'WAG', 'WMT']", 
    "UNC_CLIENT": False,
    "APPLY_GENERAL_MULTIPLIER":False,
    "GENERAL_MULTIPLIER":"[3]",
    "APPLY_MAIL_MULTIPLIER": True,
    "MAIL_MULTIPLIER":"[3]",
    "DATA_START_DAY": "2025-01-01", 
    "GO_LIVE": "dt.datetime.strptime('05/10/2025', '%m/%d/%Y')", 
    "LAST_DATA": "dt.datetime.strptime('03/12/2025', '%m/%d/%Y')",
    "PSAO_TREATMENT": False,
    "CONFLICT_GPI_AS_TIERS":True,
    "MAIL_RETAIL_BOUND":"0.0",
    "UPPER_BOUND" : "[8, 25, 50, 100, 999999]",
    "MAX_PERCENT_INCREASE" : "[999999, 1.5, 1, .5, .35]",
    "MAX_DOLLAR_INCREASE" : "[12, 999999, 999999,  999999,  999999]",
    "APPLY_FLOORS_MAIL": False, 
    "CONFLICT_GPI_CUTOFF": 100,
    "COSTSAVER_CLIENT" : False,
    "MARKETPLACE_CLIENT" : False,
    "INTERCEPTOR_OPT" : False,
}

custom_params_5084_cs_no_intrcpt = {
    "CUSTOMER_ID": "['5084']",
    "CLIENT_NAME_TABLEAU":"MERCER NCP - DUN AND BRADSTREET (JAN-JUN)",
    "DATA_ID": "'CS_NO_INTRCPT_2025_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/5084_CS_NO_INTRCPT_2025""",
    "BIG_CAPPED_PHARMACY_LIST": "['CVS','RAD','WAG','KRG','WMT']",
    "SMALL_CAPPED_PHARMACY_LIST": "['MCHOICE', 'ABS' , 'PBX' , 'AHD' , 'AMZ' , 'GEN' , 'HYV' , 'GIE' , 'MJR']",
    "NON_CAPPED_PHARMACY_LIST": "['ART','CST','AHS','CHD', 'TPS' , 'RUR']",
    "COGS_PHARMACY_LIST": "['NONPREF_OTH']",
    "PSAO_LIST": "[ 'EPC' , 'HMA' , 'CAR' , 'ELE']",
    "CLIENT_TYPE": "COMMERCIAL",
    "OVER_REIMB_CHAINS": "['CVS', 'RAD', 'WAG', 'WMT']", 
    "UNC_CLIENT": False,
    "APPLY_GENERAL_MULTIPLIER":False,
    "GENERAL_MULTIPLIER":"[3]",
    "APPLY_MAIL_MULTIPLIER": True,
    "MAIL_MULTIPLIER":"[3]",
    "DATA_START_DAY": "2025-01-01", 
    "GO_LIVE": "dt.datetime.strptime('05/10/2025', '%m/%d/%Y')", 
    "LAST_DATA": "dt.datetime.strptime('03/12/2025', '%m/%d/%Y')",
    "PSAO_TREATMENT": False,
    "CONFLICT_GPI_AS_TIERS":True,
    "MAIL_RETAIL_BOUND":"0.0",
    "UPPER_BOUND" : "[8, 25, 50, 100, 999999]",
    "MAX_PERCENT_INCREASE" : "[999999, 1.5, 1, .5, .35]",
    "MAX_DOLLAR_INCREASE" : "[12, 999999, 999999,  999999,  999999]",
    "APPLY_FLOORS_MAIL": False, 
    "CONFLICT_GPI_CUTOFF": 100,
    "COSTSAVER_CLIENT" : True,
    "MARKETPLACE_CLIENT" : False,
    "INTERCEPTOR_OPT" : False,
}

custom_params_5084_cs_no_mkt = {
    "CUSTOMER_ID": "['5084']",
    "CLIENT_NAME_TABLEAU":"MERCER NCP - DUN AND BRADSTREET (JAN-JUN)",
    "DATA_ID": "'CS_NO_MKT_2025_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/5084_CS_NO_MKT_2025""",
    "BIG_CAPPED_PHARMACY_LIST": "['CVS','RAD','WAG','KRG','WMT']",
    "SMALL_CAPPED_PHARMACY_LIST": "['MCHOICE', 'ABS' , 'PBX' , 'AHD' , 'AMZ' , 'GEN' , 'HYV' , 'GIE' , 'MJR']",
    "NON_CAPPED_PHARMACY_LIST": "['ART','CST','AHS','CHD', 'TPS' , 'RUR']",
    "COGS_PHARMACY_LIST": "['NONPREF_OTH']",
    "PSAO_LIST": "[ 'EPC' , 'HMA' , 'CAR' , 'ELE']",
    "CLIENT_TYPE": "COMMERCIAL",
    "OVER_REIMB_CHAINS": "['CVS', 'RAD', 'WAG', 'WMT']", 
    "UNC_CLIENT": False,
    "APPLY_GENERAL_MULTIPLIER":False,
    "GENERAL_MULTIPLIER":"[3]",
    "APPLY_MAIL_MULTIPLIER": True,
    "MAIL_MULTIPLIER":"[3]",
    "DATA_START_DAY": "2025-01-01", 
    "GO_LIVE": "dt.datetime.strptime('05/10/2025', '%m/%d/%Y')", 
    "LAST_DATA": "dt.datetime.strptime('03/12/2025', '%m/%d/%Y')",
    "PSAO_TREATMENT": False,
    "CONFLICT_GPI_AS_TIERS":True,
    "MAIL_RETAIL_BOUND":"0.0",
    "UPPER_BOUND" : "[8, 25, 50, 100, 999999]",
    "MAX_PERCENT_INCREASE" : "[999999, 1.5, 1, .5, .35]",
    "MAX_DOLLAR_INCREASE" : "[12, 999999, 999999,  999999,  999999]",
    "APPLY_FLOORS_MAIL": False, 
    "CONFLICT_GPI_CUTOFF": 100,
    "COSTSAVER_CLIENT" : True,
    "MARKETPLACE_CLIENT" : False,
    "INTERCEPTOR_OPT" : True,
}

custom_params_5084_cs_mkt = {
    "CUSTOMER_ID": "['5084']",
    "CLIENT_NAME_TABLEAU":"MERCER NCP - DUN AND BRADSTREET (JAN-JUN)",
    "DATA_ID": "'CS_MKT_2025_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/5084_CS_MKT_2025""",
    "BIG_CAPPED_PHARMACY_LIST": "['CVS','RAD','WAG','KRG','WMT']",
    "SMALL_CAPPED_PHARMACY_LIST": "['MCHOICE', 'ABS' , 'PBX' , 'AHD' , 'AMZ' , 'GEN' , 'HYV' , 'GIE' , 'MJR']",
    "NON_CAPPED_PHARMACY_LIST": "['ART','CST','AHS','CHD', 'TPS' , 'RUR']",
    "COGS_PHARMACY_LIST": "['NONPREF_OTH']",
    "PSAO_LIST": "[ 'EPC' , 'HMA' , 'CAR' , 'ELE']",
    "CLIENT_TYPE": "COMMERCIAL",
    "OVER_REIMB_CHAINS": "['CVS', 'RAD', 'WAG', 'WMT']", 
    "UNC_CLIENT": False,
    "APPLY_GENERAL_MULTIPLIER":False,
    "GENERAL_MULTIPLIER":"[3]",
    "APPLY_MAIL_MULTIPLIER": True,
    "MAIL_MULTIPLIER":"[3]",
    "DATA_START_DAY": "2025-01-01", 
    "GO_LIVE": "dt.datetime.strptime('05/10/2025', '%m/%d/%Y')", 
    "LAST_DATA": "dt.datetime.strptime('03/12/2025', '%m/%d/%Y')",
    "PSAO_TREATMENT": False,
    "CONFLICT_GPI_AS_TIERS":True,
    "MAIL_RETAIL_BOUND":"0.0",
    "UPPER_BOUND" : "[8, 25, 50, 100, 999999]",
    "MAX_PERCENT_INCREASE" : "[999999, 1.5, 1, .5, .35]",
    "MAX_DOLLAR_INCREASE" : "[12, 999999, 999999,  999999,  999999]",
    "APPLY_FLOORS_MAIL": False, 
    "CONFLICT_GPI_CUTOFF": 100,
    "COSTSAVER_CLIENT" : True,
    "MARKETPLACE_CLIENT" : True,
    "INTERCEPTOR_OPT" : True,
}

custom_params_2012 = {
    "CUSTOMER_ID": "['2012']",
    "CLIENT_NAME_TABLEAU":"EVOLENT HEALTH MED D",
    "DATA_ID": "'P-NP_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/2012""",
    "BIG_CAPPED_PHARMACY_LIST": "['CVS', 'RAD', 'WAG', 'KRG', 'WMT']",
    "SMALL_CAPPED_PHARMACY_LIST": "['MCHOICE']",
    "NON_CAPPED_PHARMACY_LIST": "['NONPREF_OTH','PREF_OTH','ABS','HMA','AHS','PBX','AHD','MJR','ART','CAR','ELE','EPC','TPS', 'GEN']",
    "COGS_PHARMACY_LIST": "[]",
    "PSAO_LIST": "[]",
    "CLIENT_TYPE": "MEDD",
    "OVER_REIMB_CHAINS":"['WAG', 'WMT']",
    "APPLY_GENERAL_MULTIPLIER":False,
    "GENERAL_MULTIPLIER":"[3]",
    "APPLY_MAIL_MULTIPLIER": False,
    "MAIL_MULTIPLIER":"[3]",
    "DATA_START_DAY": '2023-01-01',
    "GO_LIVE":"dt.datetime.strptime('07/01/2023', '%m/%d/%Y')",
    "LAST_DATA":"dt.datetime.strptime('06/27/2023', '%m/%d/%Y')",
    "PSAO_TREATMENT": False,
    "HANDLE_CONFLICT_GPI":True,
    "CONFLICT_GPI_AS_TIERS":True,
    "CONFLICT_GPI_LIST_FILE":"'conflict_exclusion_gpis_cicd_2012.csv'",
    "MAIL_RETAIL_BOUND":"0.1",
    "INFEASIBLE_EXCLUSION_FILE": "'infeasible_exclusion_gpis_cicd_2012.csv'",
    "HANDLE_INFEASIBLE": True,
    "UPPER_BOUND": "[3, 6, 999999]",
    "MAX_PERCENT_INCREASE": "[20000, 200, 3]",
    "MAX_DOLLAR_INCREASE": "[20, 999999, 999999]",
    "APPLY_FLOORS_MAIL": False
}

custom_params_3066 = {
    "CUSTOMER_ID": "['3066']",
    'CLIENT_NAME_TABLEAU':'The World Bank',
    "DATA_ID": "'EGWP_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/3066""",
    "BIG_CAPPED_PHARMACY_LIST": "['CVS', 'RAD', 'WAG', 'WMT', 'KRG']",
    "SMALL_CAPPED_PHARMACY_LIST": "[]",
    "NON_CAPPED_PHARMACY_LIST": "['NONPREF_OTH', 'MCHOICE']",
    "PSAO_LIST": "[]",
    "CLIENT_TYPE": "MEDD",
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": False,
    "UNC_OPT": False,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": False,
    "CONFLICT_GPI_LIST_FILE": "'conflict_exclusion_gpis_cicd_3066.csv'",
    "HANDLE_CONFLICT_GPI": True,
    "APPLY_FLOORS_MAIL": False
}

custom_params_3152_Int = {
    "CUSTOMER_ID": "['3152']",
    'CLIENT_NAME_TABLEAU':'Whole Foods Int',
    "DATA_ID": "'Interceptor_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/3152_Int""",
    "CLIENT_TYPE": "COMMERCIAL",
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": False,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": True,
    "COSTSAVER_CLIENT": True,
    "READ_FROM_BQ": True,
    "WRITE_TO_BQ": True,
    "RAW_GOODRX": "''",
    "INTERCEPTOR_ZBD_TIME_LAG":20,
    "INFEASIBLE_EXCLUSION_FILE": "'infeasible_exclusion_gpis_cicd_INTERCEPTOR_3152.csv'",
    "HANDLE_INFEASIBLE": True,
    "CONFLICT_GPI_LIST_FILE":"'conflict_exclusion_gpis_cicd_3152_Int.csv'",
    "HANDLE_CONFLICT_GPI": True,
    "IGNORE_PERFORMANCE_CHECK": True, # Current run is getting lamdba_performance - model_performance = 0.0013*AWP
    "APPLY_FLOORS_MAIL": False
}

custom_params_3152_Int_UNC = {
    "CUSTOMER_ID": "['3152']",
    'CLIENT_NAME_TABLEAU':'Whole Foods Int',
    "DATA_ID": "'Interceptor_UNC_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/3152_Int_UNC""",
    "CLIENT_TYPE": "COMMERCIAL",
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": False,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": True,
    "COSTSAVER_CLIENT": True,
    "READ_FROM_BQ": True,
    "WRITE_TO_BQ": True,
    "RAW_GOODRX": "''",
    "INTERCEPTOR_ZBD_TIME_LAG":20,
    "INFEASIBLE_EXCLUSION_FILE": "'infeasible_exclusion_gpis_cicd_INTERCEPTOR_3152.csv'",
    "HANDLE_INFEASIBLE": True,
    "CONFLICT_GPI_LIST_FILE":"'conflict_exclusion_gpis_cicd_3152_Int.csv'",
    "HANDLE_CONFLICT_GPI": True,
    "UNC_OPT": True,
    "UNC_PHARMACY": True,
    "APPLY_FLOORS_MAIL": False
}

custom_params_3152_no_mail = {
    "CUSTOMER_ID": "['3152']",
    'CLIENT_NAME_TABLEAU':'Whole Foods Int',
    "DATA_ID": "'Interceptor_UNC_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/3152_no_mail""",
    "CLIENT_TYPE": "COMMERCIAL",
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": False,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": True,
    "COSTSAVER_CLIENT": True,
    "READ_FROM_BQ": True,
    "WRITE_TO_BQ": True,
    "RAW_GOODRX": "''",
    "INTERCEPTOR_ZBD_TIME_LAG":20,
    "INFEASIBLE_EXCLUSION_FILE": "'infeasible_exclusion_gpis_cicd_INTERCEPTOR_3152.csv'",
    "HANDLE_INFEASIBLE": True,
    "CONFLICT_GPI_LIST_FILE":"'conflict_exclusion_gpis_cicd_3152_Int.csv'",
    "HANDLE_CONFLICT_GPI": True,
    "UNC_OPT": True,
    "UNC_PHARMACY": True, 
    "NO_MAIL": True, 
    "IGNORED_VCMLS":['10','12','13','SX','S3','S9', 'E23', 'N23','E92', '92', '24', 'E77', 'P77', '77', 'E78', '78', 'XR', 'XT', 'XA','88', '2'],
    "APPLY_FLOORS_MAIL": False
}

custom_params_3152_GRx = {
    "CUSTOMER_ID": "['3152']",
    'CLIENT_NAME_TABLEAU':'Whole Foods GRx',
    "DATA_ID": "'GoodRx_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/3152_GRx""",
    "CLIENT_TYPE": "COMMERCIAL",
    "GOODRX_OPT": True,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": False,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": False,
    "RAW_GOODRX": "''",
    "INFEASIBLE_EXCLUSION_FILE": "'infeasible_exclusion_gpis_cicd_3152.csv'",
    "HANDLE_INFEASIBLE": True,
    "CONFLICT_GPI_LIST_FILE":"'conflict_exclusion_gpis_cicd_3152_GRx.csv'",
    "HANDLE_CONFLICT_GPI": True,
    "APPLY_FLOORS_MAIL": False
}

custom_params_3152_PharmUNC = {
    "CUSTOMER_ID": "['3152']",
    'CLIENT_NAME_TABLEAU':'Whole Foods PharmUNC',
    "DATA_ID": "'PharmUNC_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/3152_PharmUNC""",
    "CLIENT_TYPE": "COMMERCIAL",
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": False,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": False,
    "UNC_OPT": True,
    "UNC_PHARMACY": True,
    "ZERO_QTY_TIGHT_BOUNDS": True,
    "CONFLICT_GPI_LIST_FILE":"'conflict_exclusion_gpis_cicd_3152_PharmUNC.csv'",
    "HANDLE_CONFLICT_GPI": True,
    "APPLY_FLOORS_MAIL": False
}

custom_params_3152_RMS = {
    "CUSTOMER_ID": "['3152']",
    'CLIENT_NAME_TABLEAU':'Whole Foods RMS',
    "DATA_ID": "'RMS_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/3152_RMS""",
    "CLIENT_TYPE": "COMMERCIAL",
    "RMS_OPT": True,
    "APPLY_GENERAL_MULTIPLIER": True,
    "APPLY_MAIL_MULTIPLIER": True,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": False,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": False,
    "RAW_GOODRX": "''",
    "INFEASIBLE_EXCLUSION_FILE": "'infeasible_exclusion_gpis_cicd_3152_RMS.csv'",
    "HANDLE_INFEASIBLE": True,
    "CONFLICT_GPI_LIST_FILE":"'conflict_exclusion_gpis_cicd_3152_RMS.csv'",
    "HANDLE_CONFLICT_GPI": True,
    "APPLY_FLOORS_MAIL": False
}

custom_params_3731 = {
    "CUSTOMER_ID": "['3731']",
    'CLIENT_NAME_TABLEAU':'Advance Auto Parts',
    "DATA_ID": "'Locked_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/3731""",
    "CLIENT_TYPE": "COMMERCIAL",
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": False,
    "UNC_OPT": False,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": False,
    "BRAND_SURPLUS_READ_CSV":True,
    "BRAND_SURPLUS_FILE":"''",
    "MAX_PERCENT_INCREASE": "[20, 2, 1, 0.5, 0.5, 0.2]",
    "MAX_DOLLAR_INCREASE" : "[5, 15, 20, 40, 80, 20000]",
    "APPLY_FLOORS_MAIL": False
}

custom_params_3731_ZBD = {
    "CUSTOMER_ID": "['3731']",
    'CLIENT_NAME_TABLEAU':'Advance Auto Parts',
    "DATA_ID": "'ZBD_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/3731_ZBD""",
    "CLIENT_TYPE": "COMMERCIAL",
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": False,
    "UNC_OPT": False,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": False,
    "BRAND_SURPLUS_READ_CSV":True,
    "BRAND_SURPLUS_FILE":"''",
    "MAX_PERCENT_INCREASE": "[20, 2, 1, 0.5, 0.5, 0.2]",
    "MAX_DOLLAR_INCREASE" : "[5, 15, 20, 40, 80, 20000]",
    "ZBD_OPT" : True,
    "LOCKED_CLIENT" : True,
    "CONFLICT_GPI_LIST_FILE": "'conflict_exclusion_gpis_cicd_3731_ZBD.csv'",
    "HANDLE_CONFLICT_GPI": True,
    "LEAKAGE_LIST": "'Legacy'",
    "APPLY_FLOORS_MAIL": True
}

custom_params_3731_LOPT = {
    "CUSTOMER_ID": "['3731']",
    'CLIENT_NAME_TABLEAU':'Advance Auto Parts',
    "DATA_ID": "'LOPT_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/3731_LOPT""",
    "CLIENT_TYPE": "COMMERCIAL",
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": False,
    "UNC_OPT": False,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": False,
    "BRAND_SURPLUS_READ_CSV":True,
    "BRAND_SURPLUS_FILE":"''",
    "MAX_PERCENT_INCREASE": "[20, 2, 1, 0.5, 0.5, 0.2]",
    "MAX_DOLLAR_INCREASE" : "[5, 15, 20, 40, 80, 20000]",
    "LEAKAGE_OPT" : True,
    "LOCKED_CLIENT" : True,
    "CONFLICT_GPI_LIST_FILE": "'conflict_exclusion_gpis_cicd_3731_ZBD.csv'",
    "HANDLE_CONFLICT_GPI": True,
    "LEAKAGE_LIST": "'All'",
    "APPLY_FLOORS_MAIL": False, 
    "IGNORE_PERFORMANCE_CHECK": True 
}

custom_params_3731_spclty_offset = {
    "CUSTOMER_ID": "['3731']",
    'CLIENT_NAME_TABLEAU':'Advance Auto Parts',
    "DATA_ID": "'spclty_offset_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/3731_spclty_offset""",
    "CLIENT_TYPE": "COMMERCIAL",
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": False,
    "UNC_OPT": False,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": False,
    "BRAND_SURPLUS_READ_CSV":True,
    "BRAND_SURPLUS_FILE":"''",
    "MAX_PERCENT_INCREASE": "[20, 2, 1, 0.5, 0.5, 0.2]",
    "MAX_DOLLAR_INCREASE" : "[5, 15, 20, 40, 80, 20000]",
    "GUARANTEE_CATEGORY": 'NonOffsetting R30/R90',
    "SPECIALTY_OFFSET": True,
    "SPECIALTY_SURPLUS_DATA": "'spclty_surplus_CICD.csv'",
    "APPLY_FLOORS_MAIL": False
}

custom_params_4475 = {
    "CUSTOMER_ID": "['4475']",
    'CLIENT_NAME_TABLEAU':'King County',
    "DATA_ID": "'UNC_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/4475""",
    "CLIENT_TYPE": "COMMERCIAL",
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": True,
    "UNC_OPT": True,
    "UNC_CLIENT": True,
    "UNC_PHARMACY": False,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": False,
    "CONFLICT_GPI_LIST_FILE": "'conflict_exclusion_gpis_cicd_4475.csv'",
    "HANDLE_CONFLICT_GPI": True,
    "CONFLICT_GPI_AS_TIERS": False,
    "APPLY_FLOORS_MAIL": False
}

custom_params_4475_dualUNC = {
    "CUSTOMER_ID": "['4475']",
    'CLIENT_NAME_TABLEAU':'King County',
    "DATA_ID": "'DUALUNC_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/4475_DualUNC""",
    "CLIENT_TYPE": "COMMERCIAL",
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": True,
    "UNC_OPT": True,
    "UNC_CLIENT": True,
    "UNC_PHARMACY": True,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": False,
    "CONFLICT_GPI_LIST_FILE": "'conflict_exclusion_gpis_cicd_4475.csv'",
    "HANDLE_CONFLICT_GPI": True,
    "CONFLICT_GPI_AS_TIERS": False,
    "APPLY_FLOORS_MAIL": False
}

custom_params_4475_dualUNC_INT = {
    "CUSTOMER_ID": "['4475']",
    'CLIENT_NAME_TABLEAU':'King County',
    "DATA_ID": "'DUALUNC_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/4475_DualUNC_INT""",
    "CLIENT_TYPE": "COMMERCIAL",
    "GOODRX_OPT": True,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": True,
    "UNC_OPT": True,
    "UNC_CLIENT": True,
    "UNC_PHARMACY": True,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": True,
    "COSTSAVER_CLIENT": True,
    "CONFLICT_GPI_LIST_FILE": "'conflict_exclusion_gpis_cicd_4475.csv'",
    "HANDLE_CONFLICT_GPI": True,
    "CONFLICT_GPI_AS_TIERS": False,
    "APPLY_FLOORS_MAIL": False
}


custom_params_4475_UNC_FY = {
    "CUSTOMER_ID": "['4475']",
    'CLIENT_NAME_TABLEAU':'King County',
    "DATA_ID": "'UNC_FY_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/4475_UNC_FY""",
    "CLIENT_TYPE": "COMMERCIAL",
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": True,
    "UNC_OPT": True,
    "UNC_CLIENT": True,
    "UNC_PHARMACY": True,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": False,
    "CONFLICT_GPI_LIST_FILE": "'conflict_exclusion_gpis_cicd_4475.csv'",
    "HANDLE_CONFLICT_GPI": True,
    "CONFLICT_GPI_AS_TIERS": False,
    "FULL_YEAR": True,
    "PHARM_GUARANTEE_COMM_NY": "'Pharmacy_Rate_Commercial_Next_Year_2025MAY_UPDATE.csv'",
    "BRAND_SURPLUS_READ_CSV": True,
    "GO_LIVE" : "dt.datetime.strptime('01/01/2022', '%m/%d/%Y')",
    "WS_SUFFIX": "''",
    "SPECIALTY_OFFSET": False,
    "APPLY_FLOORS_MAIL": False
}


custom_params_4490 = {
    "CUSTOMER_ID": "['4490']",
    'CLIENT_NAME_TABLEAU':'WTW-ConAgra',
    "DATA_ID": "'WTW_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/4490""",
    "CLIENT_TYPE": "COMMERCIAL",
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": True,
    "UNC_OPT": False,
    "TIERED_PRICE_LIM": False,
    "INTERCEPTOR_OPT": False,
    "GPI_UP_FAC": 0.24,
    "GPI_LOW_FAC": 0.5,
    "CONFLICT_GPI_LIST_FILE": "'conflict_exclusion_gpis_cicd_4490.csv'",
    "HANDLE_CONFLICT_GPI": True,
    "APPLY_FLOORS_MAIL": False
}

custom_params_4490_FY = {
    "CUSTOMER_ID": "['4490']",
    'CLIENT_NAME_TABLEAU':'WTW-ConAgra FY',
    "DATA_ID": "'FullYear_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/4490_FY""",
    "CLIENT_TYPE": "COMMERCIAL",
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": False,
    "FULL_YEAR": True,
    "PHARM_GUARANTEE_COMM_NY": "'Pharmacy_Rate_Commercial_Next_Year_2025MAY_UPDATE.csv'",
    "REMOVE_WTW_RESTRICTION": True, 
    "BRAND_SURPLUS_READ_CSV": True,
    "WS_SUFFIX": "''",
    "DATA_START_DAY": '2021-01-01',
    "LAST_DATA": "dt.datetime.strptime('10/30/2021', '%m/%d/%Y')",
    "GO_LIVE" : "dt.datetime.strptime('01/01/2022', '%m/%d/%Y')",
    "TIERED_PRICE_LIM": False,
    "GPI_UP_FAC": 0.24,
    "GPI_LOW_FAC": 0.5,
    "CONFLICT_GPI_LIST_FILE": "'conflict_exclusion_gpis_cicd_4490_FY.csv'",
    "HANDLE_CONFLICT_GPI": True,
    "SPECIALTY_OFFSET": False,
    "APPLY_FLOORS_MAIL": False
}

custom_params_4658 = {
    "CUSTOMER_ID": "['4658']",
    'CLIENT_NAME_TABLEAU':'State of Oklahoma',
    "DATA_ID": "'EGWP_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/4658""",
    "BIG_CAPPED_PHARMACY_LIST": "['CVS', 'RAD', 'WAG', 'WMT', 'KRG']",
    "SMALL_CAPPED_PHARMACY_LIST": "[]",
    "NON_CAPPED_PHARMACY_LIST": "['NONPREF_OTH']",
    "PSAO_LIST": "[]",
    "CLIENT_TYPE": "MEDD",
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": False,
    "UNC_OPT": False,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": False,
    "APPLY_FLOORS_MAIL": False,
    "BENCHMARK_CAP_MULTIPLIER": 10.0
}

custom_params_4588 = {
    "CUSTOMER_ID": "['4588']",
    'CLIENT_NAME_TABLEAU':'SoGA',
    "DATA_ID": "'UNC_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/4588""",
    "CLIENT_TYPE": "COMMERCIAL",
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": False,
    "UNC_OPT": False,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": False,
    "LAST_DATA": "dt.datetime.strptime('12/28/2021', '%m/%d/%Y')",
    "GO_LIVE": "dt.datetime.strptime('12/30/2021', '%m/%d/%Y')",
    "CONFLICT_GPI_LIST_FILE": "'conflict_exclusion_gpis_cicd_4588.csv'",
    "HANDLE_CONFLICT_GPI": True,
    "APPLY_FLOORS_MAIL": False
}

custom_params_4442 = {
    "CUSTOMER_ID": "['4442']",
    'CLIENT_NAME_TABLEAU':'JPMC',
    "DATA_ID": "'SP_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/4442""",
    "CLIENT_TYPE": "COMMERCIAL",
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": False,
    "UNC_OPT": False,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": False,
    "BRAND_SURPLUS_READ_CSV":True,
    "BRAND_SURPLUS_FILE":"''",
    "DATA_START_DAY": '2022-01-01',
    "LAST_DATA": "dt.datetime.strptime('05/31/2022', '%m/%d/%Y')",
    "GO_LIVE" : "dt.datetime.strptime('06/30/2022', '%m/%d/%Y')",
    "COGS_PHARMACY_LIST":"['NONPREF_OTH']",
    "NON_CAPPED_PHARMACY_LIST":"['ART']",
    "APPLY_FLOORS_MAIL": False, 
    "IGNORE_PERFORMANCE_CHECK": True

}

custom_params_8230 = {
    "CUSTOMER_ID": "['8230']",
    'CLIENT_NAME_TABLEAU':'AMERICAN AIRLINES',
    "DATA_ID": "'R90OK_LOCKED_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/8230""",
    "CLIENT_TYPE": "COMMERCIAL",
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": False,
    "UNC_OPT": False,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": False,
    "BRAND_SURPLUS_READ_CSV":True,
    "BRAND_SURPLUS_FILE":"''",
    "DATA_START_DAY": '2023-01-01',
    "LAST_DATA": "dt.datetime.strptime('08/31/2023', '%m/%d/%Y')",
    "GO_LIVE" : "dt.datetime.strptime('09/30/2023', '%m/%d/%Y')",
    "CONFLICT_GPI_LIST_FILE": "'conflict_exclusion_gpis_cicd_8230.csv'",
    "HANDLE_CONFLICT_GPI": True,
    "APPLY_FLOORS_MAIL": False
}

custom_params_4520 = {
    "CUSTOMER_ID": "['4520']",
    'CLIENT_NAME_TABLEAU':'RAYTHEON COMPANY',
    "DATA_ID": "'R90OK_COMMERCIAL_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/4520""",
    "CLIENT_TYPE": "COMMERCIAL",
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": False,
    "UNC_OPT": False,
    "TIERED_PRICE_LIM": True,
    "INTERCEPTOR_OPT": False,
    "BRAND_SURPLUS_READ_CSV":True,
    "BRAND_SURPLUS_FILE":"''",
    "PARITY_KNOWN_FIXED_PRICE_ISSUE": True,
    "HANDLE_CONFLICT_GPI":True,
    "CONFLICT_GPI_LIST_FILE":"'conflict_exclusion_gpis_cicd_4520.csv'",
    "DATA_START_DAY": '2023-01-01',
    "LAST_DATA": "dt.datetime.strptime('08/31/2023', '%m/%d/%Y')",
    "GO_LIVE" : "dt.datetime.strptime('09/30/2023', '%m/%d/%Y')",
    "APPLY_FLOORS_MAIL": False
}

custom_params_A1002 = {
    "CLIENT_LOB" : "AETNA",
    "WS_SUFFIX" : "''",
    "CUSTOMER_ID": "['A1002']",
    "CLIENT_NAME_TABLEAU":"A PLACE FOR ROVER, INC",
    "DATA_ID": "'Aetna_CS_Locked_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/A1002""",
    "CLIENT_TYPE": "COMMERCIAL",
    "TIERED_PRICE_LIM": True,
    "CROSS_CONTRACT_PROJ" :False,
    "APPLY_GENERAL_MULTIPLIER":False, 
    "ALLOW_INTERCEPT_LIMIT":True,
    "RMS_OPT":False,
    "INTERCEPTOR_OPT": True,
    "COSTSAVER_CLIENT": True,
    "BRAND_SURPLUS_READ_CSV":False,
    "LEAKAGE_OPT": True, 
    "DATA_START_DAY": "2024-01-01",
    "LAST_DATA": "dt.datetime.strptime('05/15/2024', '%m/%d/%Y')",
    "GO_LIVE" : "dt.datetime.strptime('06/01/2024', '%m/%d/%Y')",
    "HANDLE_CONFLICT_GPI":True,
    "CONFLICT_GPI_LIST_FILE":"'conflict_exclusion_gpis_cicd_A1002.csv'",
}

custom_params_A1009 = {
    "CLIENT_LOB" : "AETNA",
    "WS_SUFFIX" : "''",
    "CUSTOMER_ID": "['A1009']",
    "CLIENT_NAME_TABLEAU":"ACCESS SERVICES, INC.",
    "DATA_ID": "'Aetna_Locked_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/A1009""",
    "CLIENT_TYPE": "COMMERCIAL",
    "TIERED_PRICE_LIM": True,
    "CROSS_CONTRACT_PROJ" :False,
    "APPLY_GENERAL_MULTIPLIER":False, 
    "ALLOW_INTERCEPT_LIMIT":True,
    "RMS_OPT":False,
    "INTERCEPTOR_OPT": False,
    "BRAND_SURPLUS_READ_CSV":False,
    "LEAKAGE_OPT": True, 
    "DATA_START_DAY": "2024-01-01",
    "LAST_DATA": "dt.datetime.strptime('05/15/2024', '%m/%d/%Y')",
    "GO_LIVE" : "dt.datetime.strptime('06/01/2024', '%m/%d/%Y')",
    "HANDLE_CONFLICT_GPI":True,
    "CONFLICT_GPI_LIST_FILE":"'conflict_exclusion_gpis_cicd_A1009.csv'",
    "BIG_CAPPED_PHARMACY_LIST" : "['CVS', 'RAD', 'WAG', 'WMT']"
}

## Comment out the client as it was failing with infeasibility even after setting the HANDLE_INFEASIBLE to True 
# custom_params_A1446 = {
#     "CLIENT_LOB" : "AETNA",
#     "WS_SUFFIX" : "''",
#     "CUSTOMER_ID": "['A1446']",
#     "CLIENT_NAME_TABLEAU":"GWINNETT COUNTY BOARD OF COMMISSIONERS",
#     "DATA_ID": "'Aetna_CS_{}'.format(CUSTOMER_ID[0])",
#     "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/A1446""",
#     "CLIENT_TYPE": "COMMERCIAL",
#     "TIERED_PRICE_LIM": True,
#     "CROSS_CONTRACT_PROJ" :False,
#     "APPLY_GENERAL_MULTIPLIER":False, 
#     "ALLOW_INTERCEPT_LIMIT":True,
#     "RMS_OPT":False,
#     "INTERCEPTOR_OPT": True,
#     "BRAND_SURPLUS_READ_CSV":False,
#     "DATA_START_DAY": "2024-02-01",
#     "LAST_DATA": "dt.datetime.strptime('05/15/2024', '%m/%d/%Y')",
#     "GO_LIVE" : "dt.datetime.strptime('06/01/2024', '%m/%d/%Y')",
#     "HANDLE_CONFLICT_GPI":True,
#     "CONFLICT_GPI_LIST_FILE":"'conflict_exclusion_gpis_cicd_A1446.csv'",
#     "HANDLE_INFEASIBLE": True,
# }

custom_params_A1028 = {
    "CLIENT_LOB" : "AETNA",
    "WS_SUFFIX" : "''",
    "CUSTOMER_ID": "['A1028']",
    "CLIENT_NAME_TABLEAU":"AGS, LLC.",
    "DATA_ID": "'Aetna_Offsetting_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/A1028""",
    "CLIENT_TYPE": "COMMERCIAL",
    "TIERED_PRICE_LIM": True,
    "CROSS_CONTRACT_PROJ" :False,
    "APPLY_GENERAL_MULTIPLIER":False, 
    "ALLOW_INTERCEPT_LIMIT":True,
    "RMS_OPT":False,
    "INTERCEPTOR_OPT": True,
    "COSTSAVER_CLIENT": True,
    "BRAND_SURPLUS_READ_CSV":False,
    "DATA_START_DAY": "2024-01-01",
    "LAST_DATA": "dt.datetime.strptime('05/15/2024', '%m/%d/%Y')",
    "GO_LIVE" : "dt.datetime.strptime('06/01/2024', '%m/%d/%Y')",
    "HANDLE_CONFLICT_GPI":True,
    "CONFLICT_GPI_LIST_FILE":"'conflict_exclusion_gpis_cicd_A1028.csv'",
}

custom_params_A2201 = {
    "CLIENT_LOB" : "AETNA",
    "WS_SUFFIX" : "''",
    "CUSTOMER_ID": "['A2201']",
    "CLIENT_NAME_TABLEAU":"TKO OPERATING COMPANY, LLC",
    "DATA_ID": "'Aetna_Multi_VCML_{}'.format(CUSTOMER_ID[0])",
    "PROGRAM_OUTPUT_PATH": f"""gs://pbm-mac-lp-prod-ai-bucket/CICD/A2201""",
    "CLIENT_TYPE": "COMMERCIAL",
    "TIERED_PRICE_LIM": True,
    "CROSS_CONTRACT_PROJ" :False,
    "APPLY_GENERAL_MULTIPLIER":False, 
    "ALLOW_INTERCEPT_LIMIT":True,
    "RMS_OPT":False,
    "INTERCEPTOR_OPT": True,
    "COSTSAVER_CLIENT": True,
    "BRAND_SURPLUS_READ_CSV":False,
    "DATA_START_DAY": "2024-01-01",
    "LAST_DATA": "dt.datetime.strptime('05/15/2024', '%m/%d/%Y')",
    "GO_LIVE" : "dt.datetime.strptime('06/01/2024', '%m/%d/%Y')",
    "HANDLE_CONFLICT_GPI":True,
    "CONFLICT_GPI_LIST_FILE":"'conflict_exclusion_gpis_cicd_A2201.csv'",
}

def appending_dic(parameter_list, base_parameter_dic, client_parameter_dic):
    base_parameter_dic_copy = base_parameter_dic.copy()
    base_parameter_dic_copy.update(client_parameter_dic)
    parameter_list.append(base_parameter_dic_copy)
    del base_parameter_dic_copy
    

for client in customer_ids:
    if client == '5084':
        appending_dic(custom_params, based_parameters, custom_params_5084_no_cs_no_intrcpt)
        appending_dic(custom_params, based_parameters, custom_params_5084_cs_no_intrcpt)
        appending_dic(custom_params, based_parameters, custom_params_5084_cs_no_mkt)
        appending_dic(custom_params, based_parameters, custom_params_5084_cs_mkt)
    if client == '3066':
        appending_dic(custom_params, based_parameters, custom_params_3066)
    if client == '3152':
        appending_dic(custom_params, based_parameters, custom_params_3152_Int)
        appending_dic(custom_params, based_parameters, custom_params_3152_Int_UNC)
        appending_dic(custom_params, based_parameters, custom_params_3152_GRx)
        appending_dic(custom_params, based_parameters, custom_params_3152_PharmUNC)
        appending_dic(custom_params, based_parameters, custom_params_3152_RMS)
        appending_dic(custom_params, based_parameters, custom_params_3152_no_mail) 
    if client == '3731':
        appending_dic(custom_params, based_parameters, custom_params_3731)
        appending_dic(custom_params, based_parameters, custom_params_3731_ZBD)
        appending_dic(custom_params, based_parameters, custom_params_3731_LOPT)
        appending_dic(custom_params, based_parameters, custom_params_3731_spclty_offset)
    if client == '4475':
        appending_dic(custom_params, based_parameters, custom_params_4475)
        appending_dic(custom_params, based_parameters, custom_params_4475_dualUNC)
        appending_dic(custom_params, based_parameters, custom_params_4475_dualUNC_INT)
        appending_dic(custom_params, based_parameters, custom_params_4475_UNC_FY)
    if client == '4490':
        appending_dic(custom_params, based_parameters, custom_params_4490)
        appending_dic(custom_params, based_parameters, custom_params_4490_FY)
    if client == '4658':
        appending_dic(custom_params, based_parameters, custom_params_4658)
    if client == '4588':
        appending_dic(custom_params, based_parameters, custom_params_4588)
    if client == '4442':
        appending_dic(custom_params, based_parameters, custom_params_4442)
    if client == '2012':
        appending_dic(custom_params, based_parameters, custom_params_2012)
    if client == '4520':
        appending_dic(custom_params, based_parameters, custom_params_4520)
    if client == '8230':
        appending_dic(custom_params, based_parameters, custom_params_8230)
    if client == 'A1002':
        appending_dic(custom_params, based_parameters, custom_params_A1002)
    if client == 'A1009':
        appending_dic(custom_params, based_parameters, custom_params_A1009)
    if client == 'A1446':
        appending_dic(custom_params, based_parameters, custom_params_A1446)
    if client == 'A1028':
        appending_dic(custom_params, based_parameters, custom_params_A1028)
    if client == 'A2201':
        appending_dic(custom_params, based_parameters, custom_params_A2201)

with open('client_params.json', 'w') as outfile:
    json.dump(custom_params, outfile)
