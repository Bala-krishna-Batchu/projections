import json
import logging
import re
import datetime as dt

import pandas as pd
from google.cloud import storage
from google.cloud import bigquery
from google.cloud import bigquery_storage

from typing import (
    List,
    Optional,
)

    
def write_params(params_file_in: str):
    '''Helper function for importing parameters'''
    if not params_file_in: return 
    import os
    import shutil
    from google.cloud import storage
    params_local_path = os.path.join(os.path.dirname(__file__), 'CPMO_parameters.py')
    # copy params file
    if 'gs://' == params_file_in[:5]:  # gcp storage path
        client = storage.Client()
        bp_blob = params_file_in[5:].split('/')
        b = bp_blob[0]    
        blob = '/'.join(bp_blob[1:])
        bucket = client.get_bucket(b)
        blob = bucket.get_blob(blob)
        assert blob, f'FileNotFound: Could not find parameters file: {params_file_in}'
        blob.download_to_filename(params_local_path)
    else:  # local path
        shutil.copy(params_file_in, params_local_path)
    return 1

# storage upload helper function
def upload_blob(bucket_name, source_file, destination_blob):
    """Uploads a file to the bucket."""
    if bucket_name in destination_blob:
        destination_blob = destination_blob.replace(f'gs://{bucket_name}/', '')

    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob)
    blob.upload_from_filename(source_file)

def upload_outputs_to_bucket(
    bucket, 
    blob_dir,
    file_list=False,
    local_dirs=['Inputs', 'Output', 'Dynamic_Input', 'Logs']):
    '''upload files in file_list to cloud storage blob_dir under bucket'''
    for d in local_dirs:
        for f in os.listdir(d):
            if file_list:
                if os.path.isfile(f) and f not in file_list: continue
            uf.upload_blob(bucket, os.path.join(d, f), os.path.join(blob_dir, d, f))

##LP mod change- added schema evolution
def write_to_bq(
    df: pd.DataFrame,
    project_output: str,
    dataset_output: str,
    table_id: str,
    client_name_param: str,
    timestamp_param: str,
    run_id: str,
    schema: Optional[List[bigquery.SchemaField]] = None,
    query = None
) -> None:
    """
    Writes a copy of a Pandas DataFrame to GCP BigQuery. The input DataFrame
    is not altered in any way.
    
    In BigQuery, the schema is configured automatically if not specified.
    The table allows for new data to be automatically appended to the table
    on write, and it also allows for an evolving schema
    (i.e. new fields can be added later).
    
    For the DataFrame: mixed data-type fields are cast as string. Any
    whitespace or dots (".") in the column names are erased.
    The following 3-4 metadata columns are added:
    
    |New Column |Data Type|Calculated   |Always Added|Optionally Added When     |
    |-----------|---------|-------------|------------|--------------------------|
    |client_name|str      |from input   |True        |                          |
    |timestamp  |str      |from input   |True        |                          |
    |Dm_Begn_Dtm|datetime |automatically|True        |                          |
    |AT_RUN_ID  |str      |from input   |False       |"AT_RUN_ID" not in columns|
    
    Any errors that are raised during write are ignored and this function will
    not halt an executing program.
    
    Parameters
    ----------
    df : pd.DataFrame
        The Pandas DataFrame, a copy of which will be written with some
        modifications. This DataFrame is not mutated inplace.
    project_output : str
        GCP project name.
    dataset_output : str
        GCP BigQuery dataset name.
    table_id : str
        Name of the table in BigQuery.
    client_name_param : str
        Name of the client.
    timestamp_param : str
        Timestamp of when parameter file was created.
    run_id : str
        Program run id.
    schema : list of bigquery.SchemaField, default None
        A schema of the BigQuery table; autodetects from the pandas
        DataFrame if not given.
    query: str, optional
        The update query to modify existing data in a BigQuery table. If not provided (default value is None), 
        the function will append a dataframe as new rows to the table. When provided, the query should be a string 
        containing a valid update query to modify data already in a table.
    
    Returns
    -------
    None
    
    Raises
    ------
    Any exceptions raised during this operation are ignored.
    
    Examples
    --------
    >>> import pandas as pd
    >>> from google.cloud import bigquery
    >>> import GER_LP_Code.CPMO_parameters as p
    >>> df = pd.DataFrame({'HELLO THERE': range(3), '1.HI': range(3)})
    >>> write_to_bq(df,
    ...             p.BQ_OUTPUT_PROJECT_ID,
    ...             p.BQ_OUTPUT_DATASET,
    ...             'df_table',
    ...             ', '.join(sorted(p.CUSTOMER_ID)),
    ...             p.TIMESTAMP,
    ...             p.AT_RUN_ID)
    >>> query = f'''
    ... SELECT *
    ... FROM `{p.BQ_OUTPUT_PROJECT_ID}.{p.BQ_OUTPUT_DATASET}.df_table`
    ...'''
    >>> read_back = bqclient.query(query).to_dataframe()
    >>> read_back.columns.tolist()
    ['HELLOTHERE', 'num1HI', 'client_name', 'timestamp', 'Dm_Begn_Dtm', 'AT_RUN_ID']
    >>> df.columns.tolist()
    ['HELLO THERE', '1.HI']
    """
    import traceback
    import datetime as dt
    import numpy as np
    import pandas as pd
    import datetime as dt
    import time
    import random
 
    bqclient = bigquery.Client()
    retry_delay = 0
    retry_delay_temp = 60  # Delay in seconds between retries
    max_retries = 10  # Maximum number of retries
    try_count = 1     # Counter from 1 - 10 
    success = False
    if query:
        while try_count <= max_retries and not success:
            random_number = random.randint(1, 10)
            retry_delay = retry_delay_temp + random_number
            try:
                job = bqclient.query(query)
                job.result()
                success = True
            except Exception as e:
                error_message = str(e)
                error_list = ['400 Resources exceeded during query execution:', 'Could not serialize access to table']
                if any(e in error_message for e in error_list):
                    print("Encountered 400 DML error. Retrying after {} seconds...".format(retry_delay))
                    time.sleep(retry_delay)
                    try_count += 1
                else:
                    raise e
        if not success:
            assert False, f"Maximum retries reached. Unable to complete the operation. Data upload failure for {table_id}"
    else:
        table_id = f"{project_output}.{dataset_output}.{table_id}"
        print(table_id)

        autoschema = {'schema': schema} if schema else {'autodetect': True}
        job_config = bigquery.LoadJobConfig(**{
            **autoschema,
            'write_disposition': bigquery.WriteDisposition.WRITE_APPEND,
            'schema_update_options': bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION,
        })

        def standardize_col(col: str) -> str:
            """
            >>> standardize_col('HELLO WORLD')
            'HELLOWORLD'
            >>> standardize_col('HELLO.WORLD')
            'HELLOWORLD'
            >>> standardize_col('1HELLOWORLD')
            'num1HELLOWORLD'
            >>> standardize_col('2HELLO.THERE WORLD')
            'num2HELLOTHEREWORLD'
            """
            new_col = col if col[0].isalpha() else f'num{col}'
            return new_col.replace(' ', '').replace('.', '')

        # any columns with a mixed type will have to be cast as string
        # otherwise bigquery's pyarrow backend complains
        otype_cols = (df
                      .dtypes
                      .loc[lambda row: row == np.dtype('O')]
                      .index
                      .tolist())

        additional_columns = dict(client_name=client_name_param,
                                  timestamp=timestamp_param,
                                  Dm_Begn_Dtm=dt.datetime.now(),)
        possible_addition = (dict(AT_RUN_ID=run_id)
                             if 'AT_RUN_ID' not in df.columns else dict())
        ##
        # code to make sure quota exceed issue is prevented 
        ##
        while try_count <= max_retries and not success:
            random_number = random.randint(1, 10)
            retry_delay = retry_delay_temp + random_number
            try:
                job = bqclient.load_table_from_dataframe(
                    (df
                    .astype({col: str for col in otype_cols})
                    .rename(columns=standardize_col)
                    .assign(**{**additional_columns, **possible_addition})),
                    table_id,
                    job_config=job_config
                )
                job.result()
                success = True
            except Exception as e:
                error_message = str(e)
                if '403 Quota exceeded:' in error_message:
                    print("Encountered 403 quota error. Retrying after {} seconds...".format(retry_delay))
                    time.sleep(retry_delay)
                    try_count += 1
                else:
                    raise e
        if not success:
            assert False, f"Maximum retries reached. Unable to complete the operation. Data upload failure for {table_id}"

        assert (job.output_rows == df.shape[0]) & (df.shape[0] > 0) & (job.errors == None), f'Data upload failure for {table_id}'
        print(f"Loaded {len(df)} rows to table {table_id}")
        

def log_setup(log_file_path=None, loglevel="INFO"):    
    logger = logging.getLogger('cpmo_log')
    logger.setLevel(loglevel)
    logger.propagate = False
    if (logger.hasHandlers()):
        return logger

    if 'gs://' not in log_file_path:  # (assume local)
        file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        current_time = dt.datetime.now()
        time_string = current_time.strftime("%Y_%m_%d_%H_%M_%S")
        fh = logging.FileHandler(log_file_path)
        fh.setLevel(loglevel)
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    screen_formatter = logging.Formatter('%(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(loglevel)
    ch.setFormatter(screen_formatter)
    logger.addHandler(ch)
    
    return logger

def read_BQ_data(
    query, 
    project_id, 
    dataset_id, 
    table_id, 
    run_id = None,
    client = None, 
    period = None, 
    output = False, 
    custom = False, 
    mac = False, 
    customer = False, 
    debug = False, 
    claim_start = False,
    claim_date = False,
    vcml_ref_table_id = None,
    unc_included = False,
    proj_filter = None,
    locked_fix_up = None
):
    import CPMO_parameters as p
    custom_filter = None
    
    # Tables to filter on BRAND/GENERIC opt and filter out specialty
    bg_spec_tables = ["combined_mapped_claims","combined_measurement_mapping","combined_daily_totals",
                "ger_opt_unc_daily_total","ger_opt_unc_ndc_percentiles_constrained","ger_opt_unc_gpi_percentiles_constrained",
                "client_guarantees","client_guarantees_premc",
                "YTD_COMBINED_CLAIMS","plan_design_ndc","plan_design_gpi"]
    # Tables to filter on BRAND/GENERIC opt only
    bg_only_tables=["commercial_pharm_guarantees","medicaid_pharm_guarantees", "awp_history_table"]

    bg_spec_tables = [table + p.WS_SUFFIX for table in bg_spec_tables] 
    bg_only_tables = [table + p.WS_SUFFIX for table in bg_only_tables]
    
    bqclient = bigquery.Client()
    
    if p.CLIENT_LOB =='AETNA':
        if dataset_id not in  [p.BQ_OUTPUT_DATASET, 'pricing_management','fdl_ent_cnfv_prod']:
            table_id = p.AETNA_TABLE_ID_PREFIX + table_id.lower()
        if mac:
            vcml_ref_table_id = p.AETNA_TABLE_ID_PREFIX + vcml_ref_table_id.lower()
        
    if custom:
        df = bqclient.query(query).to_dataframe()
    else:
        query += f"""FROM \n  `{project_id}.{dataset_id}.{table_id}`"""
        if output:
            query += f"""\nWHERE client_name in ("{client}") AND timestamp = '{period}' AND AT_RUN_ID = '{run_id}'"""
        else:
            if period:
                period = f"period = '{period}'"
            if client:
                client = f"client in ('{client}')"
            if proj_filter:
                if unc_included:
                    proj_filter = f""" date = (SELECT 
                                                   date
                                               FROM 
                                                  `{project_id}.{dataset_id}.{table_id}`
                                               GROUP BY 1
                                               HAVING COUNT(DISTINCT UNC_INCLUDED) > 1)       
                                   """
                else:
                    proj_filter = f""" date = (SELECT 
                                                   date 
                                               FROM 
                                                  `{project_id}.{dataset_id}.{table_id}`
                                               GROUP BY 1
                                               HAVING COUNT(DISTINCT UNC_INCLUDED) > 1)
                                        AND unc_included = 0         
                                   """
            if mac:
                client = f"""mac IN (
                                SELECT
                                DISTINCT vcml_id
                                FROM
                                `{project_id}.{dataset_id}.{vcml_ref_table_id}`
                                WHERE
                                customer_id in ('{customer}') )
                    """
                customer = False
            elif customer:
                customer = f"customer_id in ('{customer}') " 
            if claim_start:
                claim_start = f"claim_date >= '{claim_start}'" 
            if claim_date: 
                claim_date = f"claim_date <= PARSE_DATE('%m/%d/%Y', '{claim_date}')"
            if table_id in bg_spec_tables:
                if p.GENERIC_OPT and not p.BRAND_OPT:
                    custom_filter = f"BG_FLAG = 'G' AND (MEASUREMENT NOT LIKE '%S%' OR UPPER(BREAKOUT_OLD) NOT LIKE '%SPECIALTY%')"
                elif not p.GENERIC_OPT and p.BRAND_OPT:
                    custom_filter = f"BG_FLAG = 'B' AND (MEASUREMENT NOT LIKE '%S%' OR UPPER(BREAKOUT_OLD) NOT LIKE '%SPECIALTY%')"
                elif p.GENERIC_OPT and p.BRAND_OPT:
                    custom_filter = f"(MEASUREMENT NOT LIKE '%S%' OR UPPER(BREAKOUT_OLD) NOT LIKE '%SPECIALTY%')"
            if table_id in bg_only_tables:
                if p.GENERIC_OPT and not p.BRAND_OPT:
                    custom_filter = f"BG_FLAG = 'G'"
                elif not p.GENERIC_OPT and p.BRAND_OPT:
                    custom_filter = f"BG_FLAG = 'B'"
            if run_id:
                run_id = f"RUN_ID = '{run_id}'"
            filter_clause = [client, proj_filter, period, customer, claim_start, claim_date, run_id, custom_filter]
            filter_clause = [i for i in filter_clause if i]
            filter_clause = " \nAND \n  ".join(filter_clause)       
            if filter_clause:
                query += "\nWHERE \n  " + filter_clause
        df = (
            bqclient.query(query).to_dataframe()
        )
        
    if locked_fix_up:
        pharm_cols = ['PHARMACY_CLAIMS', 'PHARMACY_QTY', 'PHARMACY_AWP', 'PHARMACY_NADAC', 'PHARMACY_ACC', 'PHARMACY_SPEND', 'PHARMACY_MEMBER_COST',
               'PHARMACY_DISP_FEE', 'PHARMACY_CLAIMS_ZBD', 'PHARMACY_QTY_ZBD', 'PHARMACY_AWP_ZBD', 'PHARMACY_NADAC_ZBD', 'PHARMACY_ACC_ZBD', 
               'PHARMACY_SPEND_ZBD', 'PHARMACY_MEMBER_COST_ZBD', 'PHARMACY_DISP_FEE_ZBD', 'PHARMACY_PCT25_UCAMT_UNIT', 'PHARMACY_PCT50_UCAMT_UNIT']
        client_cols = ['CLAIMS', 'QTY', 'AWP','AWP', 'AWP', 'SPEND', 'MEMBER_COST', 'DISP_FEE','CLAIMS_ZBD', 'QTY_ZBD', 'AWP_ZBD', 'AWP_ZBD', 'AWP_ZBD', 
               'SPEND_ZBD', 'MEMBER_COST_ZBD', 'DISP_FEE_ZBD', 'CLIENT_PCT25_UCAMT_UNIT', 'CLIENT_PCT50_UCAMT_UNIT']
        df[pharm_cols] = df[client_cols]
        
    if debug:
        print(query)
        print(df.head(2))
    print(f"Imported {df.shape[0]} rows and {df.shape[1]} columns from BQ table {table_id}")
    return df


def get_formatted_string(id_list):
    formatted_str = '\'' + '\',\''.join(id_list)+ '\''
    return formatted_str

def get_formatted_client_name(cutomer_id_lst):
    ##customer_id_lst should be in almost all cases equal to p.CUSTOMER_ID list,
    ##which is a list of string customer ids
    client_name = "_".join(sorted(cutomer_id_lst))
    return client_name

def get_formatted_table_id(table_id):
    import CPMO_parameters as p
    
    #Aetna has to put the prefix, 'gms_aetna' to the table id
    if p.CLIENT_LOB =='AETNA':
        if 'gms_gms_' in table_id:
            table_id = p.AETNA_TABLE_ID_PREFIX + table_id.replace('gms_gms_','').lower()
        else:
            table_id = p.AETNA_TABLE_ID_PREFIX + table_id.replace('gms_','').lower()
    
    return table_id

def get_formatted_dataset_id(dataset_id):
    import CPMO_parameters as p
    
    #If a different dataset ID is needed, this function can be used to swap it. For instance, Aetna's taxonomy table is saved in anbc-pss-prod.de_gms_enrv_pss_prod.gms_aetna_ger_opt_taxonomy_final, while Caremark's table is saved in FDL as anbc-pss-prod.fdm_cnfv_pss_prod.gms_ger_opt_taxonomy_final. 
    if p.CLIENT_LOB =='AETNA': 
        dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP
    
    return dataset_id

def get_formatted_project_id(project_id):
    import CPMO_parameters as p
    
    ##If a different project ID is needed, this function can be used to swap it. For instance, all of Aetna's WS tables are saved in anbc-pss-test instead of anbc-pss-prod. However, some Caremark-specific tables are saved in FDL such as gms_ger_opt_taxonomy_final or gms_ger_opt_customer_info_all_algorithm. 
    if p.CLIENT_LOB =='AETNA':
        project_id = p.BQ_INPUT_PROJECT_ID
    return project_id
        
def file_exists_gcs(file_location):
    """ 
    Function to get if file is present in GCS
    file_location: variable needs to be in os.path format 
                   e.g. os.path.join('gs://pbm-mac-lp-prod-ai-bucket/shared_input/Input/', 'conflict_exclusion_gpis_cicd_3731_ZBD.csv')
    """
    from google.cloud import storage
    
    client = storage.Client()
    blob_list = file_location[5:].split('/')
    b = blob_list[0]    
    blob = '/'.join(blob_list[1:])
    bucket = client.get_bucket(b)
    exists = storage.Blob(bucket=bucket, name=blob).exists(client)

    return exists