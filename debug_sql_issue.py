#!/usr/bin/env python3
"""
Debug script to test the SQL formatting issue
"""

import datetime as dt
import socket

# Test the custom parameters formatting
def test_customer_id_formatting():
    print("=== Testing Customer ID Formatting ===")
    
    # Test case 1: String format (what you had)
    custom_params_string = {
        "CUSTOMER_ID": "['4590']",
        "BQ_INPUT_PROJECT_ID": "pbm-mac-lp-prod-ai",
        "BQ_INPUT_DATASET_ENT_CNFV_PROD": "anbc-prod",
        "DATA_START_DAY": '2022-01-01',
        "LAST_DATA": dt.datetime.strptime('01/01/2025', '%m/%d/%Y'),
        "WS_SUFFIX": "_TC_eb_Test"
    }
    
    # Test case 2: List format (what you tried)
    custom_params_list = {
        "CUSTOMER_ID": ['4590'],
        "BQ_INPUT_PROJECT_ID": "pbm-mac-lp-prod-ai",
        "BQ_INPUT_DATASET_ENT_CNFV_PROD": "anbc-prod",
        "DATA_START_DAY": '2022-01-01',
        "LAST_DATA": dt.datetime.strptime('01/01/2025', '%m/%d/%Y'),
        "WS_SUFFIX": "_TC_eb_Test"
    }
    
    # Simulate the get_formatted_string function
    def get_formatted_string(id_list):
        if isinstance(id_list, str):
            # If it's a string like "['4590']", extract the actual values
            import ast
            try:
                id_list = ast.literal_eval(id_list)
            except:
                # If parsing fails, treat as single element
                id_list = [id_list]
        
        formatted_str = '\'' + '\',\''.join(str(x) for x in id_list) + '\''
        return formatted_str
    
    # Test the formatting
    print("String format CUSTOMER_ID:", custom_params_string["CUSTOMER_ID"])
    print("List format CUSTOMER_ID:", custom_params_list["CUSTOMER_ID"])
    
    formatted_string = get_formatted_string(custom_params_string["CUSTOMER_ID"])
    formatted_list = get_formatted_string(custom_params_list["CUSTOMER_ID"])
    
    print("Formatted string result:", formatted_string)
    print("Formatted list result:", formatted_list)
    
    # Test the SQL query formatting
    contract_info_custom = """
SELECT DISTINCT CASE WHEN customer_id like 'A%' THEN contract_eff_dt ELSE original_contract_eff_dt END AS contract_eff_dt,
contract_eff_dt AS market_check_dt, contract_exprn_dt
FROM {_project}.{_landing_dataset}.{_table_id}
WHERE customer_ID IN ({_customer_id})
    AND '{_last_data}' BETWEEN contract_eff_dt AND contract_exprn_dt
    AND NOT REGEXP_CONTAINS(LOWER(client), r'(_alf|biosimilar|exc zbd|hif|ldd|ltc|specialty|wrap)')
"""
    
    # Format with string version
    sql_string = contract_info_custom.format(
        _customer_id=formatted_string,
        _data_start=custom_params_string["DATA_START_DAY"],
        _last_data=dt.datetime.strftime(custom_params_string["LAST_DATA"], '%Y-%m-%d'),
        _project=custom_params_string["BQ_INPUT_PROJECT_ID"],
        _landing_dataset=custom_params_string["BQ_INPUT_DATASET_ENT_CNFV_PROD"],
        _table_id='gms_ger_opt_customer_info_all_algorithm' + custom_params_string["WS_SUFFIX"]
    )
    
    # Format with list version
    sql_list = contract_info_custom.format(
        _customer_id=formatted_list,
        _data_start=custom_params_list["DATA_START_DAY"],
        _last_data=dt.datetime.strftime(custom_params_list["LAST_DATA"], '%Y-%m-%d'),
        _project=custom_params_list["BQ_INPUT_PROJECT_ID"],
        _landing_dataset=custom_params_list["BQ_INPUT_DATASET_ENT_CNFV_PROD"],
        _table_id='gms_ger_opt_customer_info_all_algorithm' + custom_params_list["WS_SUFFIX"]
    )
    
    print("\n=== SQL Query Results ===")
    print("String format SQL:")
    print(sql_string)
    print("\nList format SQL:")
    print(sql_list)
    
    # Check for syntax issues
    print("\n=== Syntax Analysis ===")
    if "'['" in sql_string or "']'" in sql_string:
        print("❌ String format has syntax issues - contains malformed quotes")
    else:
        print("✅ String format looks syntactically correct")
    
    if "'['" in sql_list or "']'" in sql_list:
        print("❌ List format has syntax issues - contains malformed quotes")
    else:
        print("✅ List format looks syntactically correct")

if __name__ == "__main__":
    test_customer_id_formatting()