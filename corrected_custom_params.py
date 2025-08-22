#!/usr/bin/env python3
"""
Corrected custom parameters for the optimization pipeline
"""

import datetime as dt
import socket

# Corrected custom parameters
custom_params = {
    "TIMESTAMP": dt.datetime.now().strftime('%Y-%m-%d_%H%M%S%f'),
    "USER": socket.gethostname(),
    "FULL_YEAR": False,
    "CUSTOMER_ID": ['4590'],  # Fixed: Use actual list, not string
    'CLIENT_NAME_TABLEAU': 'SoGA',
    "DATA_ID": f"CHANGE_4590_DATE",  # Fixed: Remove the problematic format string
    "BQ_INPUT_PROJECT_ID": "pbm-mac-lp-prod-ai",
    "BQ_OUTPUT_DATASET": "ds_development_lp",
    "BQ_INPUT_DATASET_DS_PRO_LP": "ds_sandbox",  # Fixed: Use correct dataset name
    "PROGRAM_INPUT_PATH": "gs://pbm-mac-lp-prod-ai-bucket/shared_input",
    "PROGRAM_OUTPUT_PATH": "/home/jupyter/Output",
    "BQ_INPUT_DATASET_ENT_CNFV_PROD": "anbc-prod",
    "READ_FROM_BQ": True,
    "WRITE_TO_BQ": False,
    "UNC_OPT": False,
    "DROP_TABLES": False,
    "CLIENT_TYPE": "COMMERCIAL",
    "LAST_DATA": dt.datetime.strptime('01/01/2025', '%m/%d/%Y'),
    "GO_LIVE": dt.datetime.strptime('08/10/2025', '%m/%d/%Y'),
    "RAW_GOODRX": "GoodRx price Jan file 04192021.xlsx",  # Fixed: Remove extra quotes
    "FLOOR_GPI_LIST": "20201209_Floor_GPIs.csv",  # Fixed: Remove extra quotes
    "GOODRX_OPT": False,
    "FLOOR_PRICE": True,
    "UNC_ADJUST": True,
    "USE_PROJECTIONS_BQ": True,
    "DATA_START_DAY": '2022-01-01',
    "TIERED_PRICE_LIM": True,
    "CLIENT_LOB": 'CMK',
    "SMALL_CAPPED_PHARMACY_LIST": {
        'GNRC': ['MCHOICE', 'THF', 'AMZ', 'HYV', 'KIN', 'ABS', 'PBX', 'AHD', 'GIE', 'MJR', 'GEN', 'TPS'],
        'BRND': ['MCHOICE', 'ABS', 'AHD', 'PBX', 'MJR', 'WGS', 'SMC', 'CST', 'KIN', 'GIE', 'HYV', 'TPM', 'SMR', 'ARX', 'WIS', 'GEN', 'BGY', 'DDM', 'MCY', 'MGM', 'PMA', 'GUA', 'FVW', 'BRI', 'AMZ', 'THF', 'TPS']
    },
    "GENERIC_OPT": False, 
    "BRAND_OPT": True,
    "ws_suffix": "_TC_eb_Test",  # Fixed: Use lowercase to match template
    "UCL_CLIENT": False,
    "TRUECOST_CLIENT": True,
}

print("Corrected custom parameters:")
for key, value in custom_params.items():
    print(f"{key}: {value}")

# Test the problematic DATA_ID parameter
print(f"\nDATA_ID value: {custom_params['DATA_ID']}")
print(f"ws_suffix value: {custom_params['ws_suffix']}")