# -*- coding: utf-8 -*-
"""
This file implements a class to track parameter settings
for algorithm runs. 

Note: This was an MVP and may need changes with new code 
developments. Example:
1. This code references a folder called Audit_Trail_Data
    which is in cloud storage
2. References a BQ dataset called pricing_management
"""

class AuditTrail():
    
    def __init__(
        self, 
        git_branch,
        git_hash,
        algo_version = 'LP',
        version_iteration = '0', # change as needed
        version_type = 'WIP',     # Use 'PROD' only for official versions
        project_name = 'pbm-mac-lp-prod-ai',
        bucket_name = 'pbm-mac-lp-prod-ai-bucket',
        audit_trail_folder = 'Audit_Trail_Data',
        audit_trail_dataset_name = 'pricing_management', 
        audit_trail_table_name = 'AT_Run_ID',
        gcp_registry_name = 'us.gcr.io',
        base_name = 'pbm_base',
        script_run_name = 'pbm_script_run',
        opt_name = 'pbm_opt',
        client_lob = 'CMK',
        odp_project_name = 'pbm-mac-lp-prod-ai',
        odp_dataset_name = 'pricing_management',
        odp_table_name = 'odp_event_id_and_run_id_map_tbl',
        odp_lp = '',
        cmpas = 'False',
        uclclient = False,
    ):
        
        from google.cloud import bigquery
        import os
        
        self.algo_version = algo_version
        self.bqclient = bigquery.Client()
        self.project_name = project_name
        self.bucket_name = bucket_name
        self.audit_trail_folder = audit_trail_folder
        self.audit_trail_dataset_name = audit_trail_dataset_name
        self.audit_trail_table_name = audit_trail_table_name
        self.AT_RUN_ID = 0
        self.AT_COLUMNS_FROM_BQ = self.get_audit_trail_columns()
        self.client_lob = client_lob
        self.odp_project_name = odp_project_name
        self.odp_dataset_name = odp_dataset_name
        self.odp_table_name = odp_table_name
        self.odp_lp = odp_lp
        self.cmpas = cmpas
        self.uclclient = uclclient
        
        #registry names
        self.base_name = base_name
        self.script_run_name = script_run_name
        self.opt_name = opt_name
        
        # Image Version Setup
        self.version_iteration = version_iteration
        self.version_type = version_type
        
        self.git_branch = git_branch
        self.git_hash = git_hash
        
        """
        #get latest git branch and git hash
        if self.git_branch == None:
            self.git_branch = os.popen("git rev-parse --abbrev-ref HEAD").read().rstrip()
            self.git_hash = os.popen("git rev-parse --short HEAD").read().rstrip() 
        
        """
        self.VERSION = f'{self.git_branch}-{self.git_hash}-{self.version_type}-{self.version_iteration}'
        self.BASE_TAG = f"{gcp_registry_name}/{project_name}/{base_name}:{self.VERSION}"
        self.SCRIPT_RUN_TAG = f"{gcp_registry_name}/{project_name}/{script_run_name}:{self.VERSION}"
        self.OPT_TAG = f"{gcp_registry_name}/{project_name}/{opt_name}:{self.VERSION}"
    
    def get_latest_run_id(self,
        project_name = 'pbm-mac-lp-prod-ai',
        dataset_name = 'pricing_management',
        table_name = 'AT_Run_ID'
    ):
        import datetime as dt
        import random
        import numpy as np
        from pytz import timezone
        
        query = '''
            SELECT * FROM 
            {PROJECT}.{DATASET}.{TABLE}
               WHERE RUN_ID = (
                   SELECT MAX(RUN_ID) FROM 
                   {PROJECT}.{DATASET}.{TABLE}
                   )
        '''.format(PROJECT=project_name,DATASET=dataset_name,TABLE=table_name)
        df = self.bqclient.query(query).result().to_dataframe()
        self.AT_COLUMNS_FROM_BQ = df.columns.to_list()

        #Note: the entire value is like 'LP20210615181014049595690'
        timestamp = dt.datetime.now(timezone('US/Eastern')).strftime('%Y%m%d%H%M%S%f')
        if self.client_lob == 'AETNA':
            self.AT_RUN_ID = 'ALP' + timestamp + str(random.randint(10,99))
        elif self.odp_lp:
            self.AT_RUN_ID = 'ODPLP' + timestamp + str(random.randint(100,999))
        elif self.cmpas == 'True':
            self.AT_RUN_ID = 'CMPLP' + timestamp + str(random.randint(100,999))
        elif self.uclclient:
            if self.client_lob == 'AETNA':
                self.AT_RUN_ID = 'UCLALP' + timestamp + str(random.randint(10,99))
            else:
                self.AT_RUN_ID = 'UCLLP' + timestamp + str(random.randint(100,999))
        else:
            self.AT_RUN_ID = 'LP' + timestamp + str(random.randint(100,999))
        
        self.save_audit_trail_columns(
            data = {'AT_RUN_ID': self.AT_RUN_ID, 'AT_COLUMNS_FROM_BQ': df.columns.to_list()})
        
        return self.AT_RUN_ID 

    
    def save_audit_trail_columns(self, data):
        import os
        
        #save data as local pickle file to preseve types when loaded in python
        #(need to save to local first then upload. Better is to save to storage bucket directly)
        source_file_name='audit_trail_columns.pickle'
        self.save_pickle(data, source_file_name)

        #upload local pickle file to cloud storage
        self.upload_blob(
            bucket_name='pbm-mac-lp-prod-ai-bucket', 
            source_file_name='audit_trail_columns.pickle', 
            destination_blob_name='Audit_Trail_Data/audit_trail_columns.pickle'
        )
        os.remove(source_file_name)
    
    
    def update_audit_trail(
        self, 
        params_file_in: str,
        odp_lp: str, ipynb = False):
        if not ipynb: 
            self.save_params_to_cloud_bucket(params_file_in)
        row_to_add, odp_row_to_add = self.get_audit_trail_data_update(params_file_in, odp_lp, ipynb = ipynb)
        self.load_data_to_bq(row_to_add, self.project_name, self.audit_trail_dataset_name, self.audit_trail_table_name, odp_load = False)
        if self.odp_lp:
            self.load_data_to_bq(odp_row_to_add, self.odp_project_name, self.odp_dataset_name, self.odp_table_name, odp_load = True)
    
    def save_params_to_cloud_bucket(self, params_file_in: str):
        
        """
            Moving files between directories or buckets. it will use GCP's  
            copy function then delete the blob from the old location.
        
        Parameters
        -----
            params_file_in: path to parameter file in cloud storage
            gs://pbm-mac-lp-prod-ai-bucket/marcel-test/Georgia_4588/Run1/CPMO_parameters.py
                
        """
        from google.cloud import storage
        import os
        
        #download to local as parameters.py
        self._download_params_to_local(params_file_in)
        import parameters as p
        
        #using blob, the path should not contain the bucket part
        source_blob_name = params_file_in.replace('gs://pbm-mac-lp-prod-ai-bucket/','')
        
        storage_client = storage.Client()
        source_bucket = storage_client.get_bucket(self.bucket_name)
        source_blob = source_bucket.blob(source_blob_name)
        destination_bucket = storage_client.get_bucket(self.bucket_name)
    
        # copy to new destination
        dest_blob_name = '{}/{}_CPMO_parameters.py'.format(p.USER, p.TIMESTAMP)
        dest_blob_name = os.path.join(self.audit_trail_folder,dest_blob_name)
        new_blob_name = source_bucket.copy_blob(
            source_blob, destination_bucket, dest_blob_name)
    
        print(f'File moved from {source_blob} to {dest_blob_name}')
    
    
    def get_audit_trail_data_update(self, params_file_in:str, odp_lp:str, ipynb = False ):
        
        """
           
        columns_from_bq:
            a list of parameter names from audit trail table
    
        Returns
        -------
            single row dataframe with same columns as audit trail table
            
        """
        import pandas as pd
        import numpy as np
        import datetime as dt
         
        if ipynb: 
            import GER_LP_Code.CPMO_parameters as p
        
        else: 
            self._download_params_to_local(params_file_in)
            import parameters as p

        #get audit trail columns stored in cloud storage as pickle 
        self.download_to_local(
            source_file_path = 'Audit_Trail_Data/audit_trail_columns.pickle', 
            dest_file_path = 'audit_trail_columns.pickle')
        data_dic = self.load_pickle(filename='audit_trail_columns.pickle')
        columns_from_bq = data_dic['AT_COLUMNS_FROM_BQ']
  
        manual_updates = {
            # Use this dictionary to convert parameters that have datatypes not handled by SQL to datatypes that can be used by SQL
            # REC_ADD_TS is a manual addition to the BQ columns not in CPMO_parameters.py
            'PARAM_LOCATION': params_file_in,
            'RUN_ID': p.AT_RUN_ID,
            'CLIENT_LIST': "|".join(p.CUSTOMER_ID*2),  #note that CUSTOMER_ID is a list, like [4545]
            'CLIENT_NAME': p.CLIENT_NAME_TABLEAU,
            'ALGO_VERSION': self.algo_version,
            'CODE_VERSION': self.VERSION, 
            'TIER_LIST': str(p.TIER_LIST),
            'PRICE_BOUNDS_UPPER': str(p.UPPER_BOUND),
            'PRICE_BOUNDS_PERCENT_INCREASE': str(p.MAX_PERCENT_INCREASE),
            'PRICE_BOUNDS_DOLLAR_INCREASE': str(p.MAX_DOLLAR_INCREASE),
            'OVER_REIMB_CHAINS': str(p.OVER_REIMB_CHAINS),
            'BIG_CAPPED_PHARMACY_LIST':str(p.BIG_CAPPED_PHARMACY_LIST) , 
            'SMALL_CAPPED_PHARMACY_LIST': str(p.SMALL_CAPPED_PHARMACY_LIST), 
            'NON_CAPPED_PHARMACY_LIST': str(p.NON_CAPPED_PHARMACY_LIST), 
            'COGS_PHARMACY_LIST': str(p.COGS_PHARMACY_LIST), 
            'PSAO_LIST': str(p.PSAO_LIST), 
            'FORCE_FLOOR_PHARMACY_SUBGROUP_LIST': str(p.FORCE_FLOOR_PHARMACY_SUBGROUP_LIST), 
            'FULL_YEAR_LV_1_UPPER_BOUND': str(p.FULL_YEAR_LV_1_UPPER_BOUND),
            'FULL_YEAR_LV_1_MAX_PERCENT_INCREASE': str(p.FULL_YEAR_LV_1_MAX_PERCENT_INCREASE), 
            'FULL_YEAR_LV_1_MAX_DOLLAR_INCREASE': str(p.FULL_YEAR_LV_1_MAX_DOLLAR_INCREASE), 
            'FULL_YEAR_LV_2_UPPER_BOUND': str(p.FULL_YEAR_LV_2_UPPER_BOUND),
            'FULL_YEAR_LV_2_MAX_PERCENT_INCREASE': str(p.FULL_YEAR_LV_2_MAX_PERCENT_INCREASE), 
            'FULL_YEAR_LV_2_MAX_DOLLAR_INCREASE': str(p.FULL_YEAR_LV_2_MAX_DOLLAR_INCREASE), 
            'FULL_YEAR_LV_3_UPPER_BOUND': str(p.FULL_YEAR_LV_3_UPPER_BOUND),
            'FULL_YEAR_LV_3_MAX_PERCENT_INCREASE': str(p.FULL_YEAR_LV_3_MAX_PERCENT_INCREASE), 
            'FULL_YEAR_LV_3_MAX_DOLLAR_INCREASE': str(p.FULL_YEAR_LV_3_MAX_DOLLAR_INCREASE),
            'RUN_TYPE_TABLEAU': str(p.RUN_TYPE_TABLEAU),
            'IGNORED_VCMLS' : str(p.IGNORED_VCMLS),
            'REC_ADD_TS': dt.datetime.now(),
            'GPI_LOW_FAC':str(p.GPI_LOW_FAC),
            'GPI_UP_FAC':str(p.GPI_UP_FAC),
            'GPI_UP_DOLLAR':str(p.GPI_UP_DOLLAR)
        }   
        
        #all_params_attributes = dir(p)
        values = {
            name: getattr(p, name.rstrip()) 
            for name in columns_from_bq if name not in manual_updates.keys()
        }
        #update the dictionary with manually added parameters
        values.update(manual_updates)

        assert len(columns_from_bq) == len(values),\
             'expected {} but got {}'.format(len(values),len(columns_from_bq))
        
        data = pd.DataFrame(data=values,index=[0])  
        
        # Add parameters with datatypes that SQL will not interpret correctly (string, float or int value)
        data['RUN_ID'] = data['RUN_ID'].astype(str)
        data['CUSTOMER_ID'] = data['CUSTOMER_ID'].astype(str)
        data['CLIENT_LIST'] = data['CLIENT_LIST'].astype(str)
        data['DATA_START_DAY'] = data['DATA_START_DAY'].astype(str)
        data['LAST_DATA'] = data['LAST_DATA'].astype(str)
        data['GO_LIVE'] = data['GO_LIVE'].astype(str)
        data['GR_SF'] = data['GR_SF'].astype(np.int64)
        data['COST_GAMMA'] = data['COST_GAMMA'].astype(np.float64)
        data['PREF_OTHER_FACTOR'] = data['PREF_OTHER_FACTOR'].astype(np.float64)
        data['GOODRX_FACTOR'] = data['GOODRX_FACTOR'].astype(np.float64)
        data['ZBD_CVS_IND_SCALAR'] = data['ZBD_CVS_IND_SCALAR'].astype(np.float64)
        data['ZBD_CAPPED_SCALAR'] = data['ZBD_CAPPED_SCALAR'].astype(np.float64)
        data['ZBD_CURRENT_PRICE_SCALAR'] = data['ZBD_CURRENT_PRICE_SCALAR'].astype(np.float64)
        data['PARITY_PRICE_DIFFERENCE_COLLAR_HIGH'] = data['PARITY_PRICE_DIFFERENCE_COLLAR_HIGH'].astype(np.float64)
        data['PARITY_PRICE_DIFFERENCE_COLLAR_LOW'] = data['PARITY_PRICE_DIFFERENCE_COLLAR_LOW'].astype(np.float64)
        data['MAIL_RETAIL_BOUND'] = data['MAIL_RETAIL_BOUND'].astype(np.float64)
        data['RETAIL_RETAIL_BOUND'] = data['RETAIL_RETAIL_BOUND'].astype(np.float64)        
        data['GENERAL_MULTIPLIER'] = data['GENERAL_MULTIPLIER'].astype(np.float64)
        data['MAIL_MULTIPLIER'] = data['MAIL_MULTIPLIER'].astype(np.float64)        
        data['HANDLE_MAIL_CONFLICT_LEVEL'] = data['HANDLE_MAIL_CONFLICT_LEVEL'].astype(str)
        data['MAIL_TARGET_BUFFER'] = data['MAIL_TARGET_BUFFER'].astype(np.float64) 
        data['RETAIL_TARGET_BUFFER'] = data['RETAIL_TARGET_BUFFER'].astype(np.float64) 
        data['ZERO_QTY_WEIGHT'] = data['ZERO_QTY_WEIGHT'].astype(np.float64)   
        data['PHARM_PERF_WEIGHT'] = data['PHARM_PERF_WEIGHT'].astype(np.float64)
        data['MAIL_NON_MAC_RATE'] = data['MAIL_NON_MAC_RATE'].astype(np.float64)
        data['RETAIL_NON_MAC_RATE'] = data['RETAIL_NON_MAC_RATE'].astype(np.float64)
        data['INTERCEPT_CEILING'] = data['INTERCEPT_CEILING'].astype(np.float64)
        data['MAIL_UNRESTRICTED_CAP'] = data['MAIL_UNRESTRICTED_CAP'].astype(np.float64)
        data['CLIENT_RETAIL_OVRPERF_PEN'] = data['CLIENT_RETAIL_OVRPERF_PEN'].astype(np.float64)
        data['CLIENT_RETAIL_UNRPERF_PEN'] = data['CLIENT_RETAIL_UNRPERF_PEN'].astype(np.float64)
        data['CLIENT_MAIL_OVRPERF_PEN'] = data['CLIENT_MAIL_OVRPERF_PEN'].astype(np.float64)
        data['CLIENT_MAIL_UNRPREF_PEN'] = data['CLIENT_MAIL_UNRPERF_PEN'].astype(np.float64)
        data['CONFLICT_GPI_CUTOFF'] = data['CONFLICT_GPI_CUTOFF'].astype(np.int64)
        data['BENCHMARK_CAP_MULTIPLIER'] = data['BENCHMARK_CAP_MULTIPLIER'].astype(np.float64)
        data['RUR_GUARANTEE_BUFFER'] = data['RUR_GUARANTEE_BUFFER'].astype(np.float64)
        data['PHARMACY_MCHOICE_OVRPERF_PEN'] = data['PHARMACY_MCHOICE_OVRPERF_PEN'].astype(np.float64)
        

        self.row_to_add = data[columns_from_bq]

        if odp_lp:
            odp_dict = {"ODP_Event_Key": str(odp_lp), "Run_ID": data['RUN_ID'].values[0]}
            odp_row_to_add = pd.DataFrame([odp_dict])
        else:
            odp_row_to_add = None

        return self.row_to_add, odp_row_to_add


    def load_data_to_bq(
        self, row_to_add, project, dataset_name, table_name, odp_load):
        from google.cloud import bigquery 
        import time
        
        client = bigquery.Client()

        if odp_load == False:
            schema = [
                bigquery.SchemaField(
                    name='RUN_ID', field_type='STRING', mode='REQUIRED'),
                bigquery.SchemaField(
                    name='CUSTOMER_ID', field_type='STRING', mode='NULLABLE'), 
            ]
        else:
            schema = [
                bigquery.SchemaField(
                    name='ODP_Event_Key', field_type='STRING', mode='REQUIRED'),
                bigquery.SchemaField(
                    name='Run_ID', field_type='STRING', mode='REQUIRED'), 
            ]

        dataset_ref = client.dataset(dataset_name, project = project)
        table_ref = dataset_ref.table(table_name)  
        
        retry_delay = 180  # Delay in seconds between retries
        max_retries = 10  # Maximum number of retries
        try_count = 1     # Counter from 1 - 10 
        success = False
        while try_count <= max_retries and not success:
            try:
                job_config = bigquery.LoadJobConfig(schema=schema) 

                client.load_table_from_dataframe(
                    row_to_add, table_ref, job_config=job_config).result()
                print(f"Loaded {row_to_add.shape[0]} rows to table {table_ref}")
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
            assert False, "Maximum retries reached. Unable to complete the operation. Data upload failure for AT_Run_ID"
        
    
    def _download_params_to_local(self, params_file_in: str, local_file_name = 'parameters.py'):
        '''User input parameter check and prep, 
        copies previously save parameter from storage bucket to code folder, then imports. '''
        import re
        from google.cloud import storage
        import datetime as dt

        # Download parameters file from storage
        tt = dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        local_file_name = 'parameters.py'
        client = storage.Client()
        bp_blob = params_file_in[5:].split('/') #ignore gs:// at the beginning
        b = bp_blob[0]    
        blob = '/'.join(bp_blob[1:])
        bucket = client.get_bucket(b)
        blob = bucket.get_blob(blob)
        assert blob, f'FileNotFound: Could not find parameters file: {params_file_in}'
        blob.download_to_filename(local_file_name)
        
        
    def upload_blob(self, bucket_name, source_file_name, destination_blob_name):
        
        """
        Uploads a local file to storage bucket.
        
            Parameters
        ----------
        bucket_name :  = "your-bucket-name"
            
        source_file_name : "local/path/to/file"
            
        destination_blob_name: "storage-object-name"
        """
       
        from google.cloud import storage
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
    
        blob.upload_from_filename(source_file_name)
    
        print(
            "File {} uploaded to {}.".format(
                source_file_name, destination_blob_name
            )
        )

    def download_to_local(self, source_file_path: str, dest_file_path: str):
        
        """ 
            download file from cloud storage to local computer
        """
    
        from google.cloud import storage

        client = storage.Client()
        b = 'pbm-mac-lp-prod-ai-bucket'

        bucket = client.get_bucket(b)
        blob = bucket.get_blob(source_file_path)
        assert blob, f'FileNotFound: Could not find parameters file: {source_file_path}'
        blob.download_to_filename(dest_file_path)
    

    def save_pickle(self, data, filename):
        import pickle
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, filename):
        import pickle
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)
        return data

    
    
    def get_audit_trail_columns(self):
        
        AT_COLUMNS_FROM_BQ = [
            'RUN_ID', 
            'TIMESTAMP', 
            'USER', 
            'CUSTOMER_ID', 
            'CLIENT_LIST',
            'CLIENT_NAME', 
            'ALGO_VERSION', 
            'CODE_VERSION', 
            'PARAM_LOCATION',
            'CLIENT_TYPE', 
            'DATA_START_DAY', 
            'LAST_DATA', 
            'GO_LIVE',
            'PSAO_TREATMENT', 
            'PRICE_OVERRIDE', 
            'FLOOR_PRICE', 
            'UNC_OPT',
            'UNC_OVERRIDE_GOODRX', 
            'UNC_ADJUST',  
            'TIERED_PRICE_LIM',
            'HIGHEROF_PRICE_LIM',
            'FULL_YEAR', 
            'NEW_YEAR_PRICE_LVL', 
            'CLIENT_TARGET_BUFFER',
            'PHARMACY_TARGET_BUFFER', 
            'PROJ_ALPHA', 
            'GPI_UP_DOLLAR',
            'GPI_UP_FAC', 
            'GPI_LOW_FAC',
            'CLIENT_GR', 
            'GR_SCALE', 
            'GR_SF', 
            'LIM_TIER', 
            'TIER_LIST',
            'CAPPED_ONLY', 
            'LAG_YTD_OVERRIDE', 
            'YTD_OVERRIDE', 
            'NDC_UPDATE',
            'STRENGTH_PRICE_CHANGE_EXCEPTION', 
            'NO_MAIL', 
            'LIMITED_BO',
            'MONTHLY_PHARM_YTD', 
            'PRICE_ZERO_PROJ_QTY', 
            'OUTPUT_FULL_MAC', 
            'CAPPED_OPT', 
            'CLIENT_TIERS', 
            'REMOVE_KRG_WMT_UC',
            'PHARMACY_EXCLUSION',  
            'AGG_UP_FAC', 
            'AGG_LOW_FAC',
            'OVER_REIMB_GAMMA', 
            'COST_GAMMA', 
            'PREF_OTHER_FACTOR',
            'INCLUDE_PLAN_LIABILITY', 
            'PLAN_LIAB_CLIENTS', 
            'PLAN_LIAB_WEIGHT',
            'READ_IN_PLAN_LIAB_WIEGHTS', 
            'GAP_FACTOR', 
            'CAT_FACTOR',
            'CUST_MOVEMENT_WEIGHTS', 
            'UNC_PHARMACY', 
            'PRICE_BOUNDS_UPPER',
            'PRICE_BOUNDS_PERCENT_INCREASE',
            'PRICE_BOUNDS_DOLLAR_INCREASE',
            'UNC_ADJUSTMENT', 
            'PRICE_OVERRIDE_FILE', 
            'RAW_GOODRX', 
            'GOODRX_FILE',
            'WC_SUGGESTED_GUARDRAILS',
            'GOODRX_FACTOR',
            'APPLY_GENERAL_MULTIPLIER',
            'APPLY_MAIL_MULTIPLIER',
            'GENERAL_MULTIPLIER',
            'MAIL_MULTIPLIER',
            'ZBD_OPT ',
            'LEAKAGE_RANK ',
            'ZBD_CVS_IND_SCALAR ',
            'ZBD_CAPPED_SCALAR ',
            'ZBD_CURRENT_PRICE_SCALAR ',
            'PARITY_PRICE_DIFFERENCE_COLLAR_HIGH ',
            'PARITY_PRICE_DIFFERENCE_COLLAR_LOW',   
            'CROSS_CONTRACT_PROJ',
            'ZERO_QTY_TIGHT_BOUNDS',
            'REMOVE_WTW_RESTRICTION',
            'MAIL_RETAIL_BOUND',
            'RETAIL_RETAIL_BOUND',
            'OVER_REIMB_CHAINS',
            'INTERCEPTOR_OPT',
            'ALLOW_INTERCEPT_LIMIT',
            'HANDLE_MAIL_CONFLICT_LEVEL',
            'LEAKAGE_OPT',
            'LEAKAGE_PENALTY',
            'CLIENT_LOB',
            'LEAKAGE_LIST', 
            'EGWP', 
            'BIG_CAPPED_PHARMACY_LIST',
            'SMALL_CAPPED_PHARMACY_LIST',
            'NON_CAPPED_PHARMACY_LIST', 
            'COGS_PHARMACY_LIST', 
            'PSAO_LIST', 
            'HANDLE_INFEASIBLE', 
            'HANDLE_CONFLICT_GPI', 
            'CONFLICT_GPI_AS_TIERS', 
            'COGS_BUFFER', 
            'FORCE_FLOOR', 
            'FORCE_FLOOR_PHARMACY_SUBGROUP_LIST', 
            'LOCKED_CLIENT', 
            'R90_AS_MAIL', 
            'MAIL_TARGET_BUFFER', 
            'RETAIL_TARGET_BUFFER', 
            'FULL_YEAR_LV_1_UPPER_BOUND', 
            'FULL_YEAR_LV_1_MAX_PERCENT_INCREASE', 
            'FULL_YEAR_LV_1_MAX_DOLLAR_INCREASE', 
            'FULL_YEAR_LV_2_UPPER_BOUND', 
            'FULL_YEAR_LV_2_MAX_PERCENT_INCREASE', 
            'FULL_YEAR_LV_2_MAX_DOLLAR_INCREASE', 
            'FULL_YEAR_LV_3_UPPER_BOUND', 
            'FULL_YEAR_LV_3_MAX_PERCENT_INCREASE', 
            'FULL_YEAR_LV_3_MAX_DOLLAR_INCREASE', 
            'ZERO_QTY_WEIGHT',
            'PHARM_PERF_WEIGHT', 
            'REMOVE_SMALL_CAPPED_UC', 
            'MAIL_NON_MAC_RATE', 
            'RETAIL_NON_MAC_RATE', 
            'GUARANTEE_CATEGORY',
            'INTERCEPT_CEILING', 
            'SPECIALTY_OFFSET', 
            'APPLY_FLOORS_MAIL', 
            'MAIL_FLOORS_FACTOR',
            'MAIL_MAC_UNRESTRICTED',
            'MAIL_UNRESTRICTED_CAP',
            'CLIENT_RETAIL_OVRPERF_PEN',
            'CLIENT_RETAIL_UNRPERF_PEN',
            'CLIENT_MAIL_OVRPERF_PEN',
            'CLIENT_MAIL_UNRPERF_PEN',
            'RUN_TYPE_TABLEAU',
            'IGNORED_VCMLS',
            'CONFLICT_GPI_CUTOFF',
            'COSTSAVER_CLIENT',
            'APPLY_BENCHMARK_CAP',
            'BENCHMARK_CAP_MULTIPLIER',
            'UNIFORM_MAC_PRICING',
            'RUR_GUARANTEE_BUFFER',
            'PHARMACY_MCHOICE_OVRPERF_PEN',
            'MEDD_MCHOICE_TARGET_RATE',
            'REC_ADD_TS'
        ]
        return AT_COLUMNS_FROM_BQ
    
    
    def get_audit_trail_schema(self):
        
        from google.cloud import bigquery 
    
        schema = [
            
        bigquery.SchemaField(name='RUN_ID', field_type='INTEGER', mode='REQUIRED'),
        bigquery.SchemaField(name='TIMESTAMP', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='USER', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='CUSTOMER_ID', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='CLIENT_LIST ', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='CLIENT_NAME', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='ALGO_VERSION', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='CODE_VERSION', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='PARAM_LOCATION', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='CLIENT_TYPE', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='DATA_START_DAY', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='LAST_DATA', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='GO_LIVE', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='PSAO_TREATMENT', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='PRICE_OVERRIDE', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='FLOOR_PRICE', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='UNC_OPT', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='UNC_OVERRIDE_GOODRX', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='UNC_ADJUST', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='TIERED_PRICE_LIM', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='HIGHEROF_PRICE_LIM', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='FULL_YEAR', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='NEW_YEAR_PRICE_LVL', field_type='INTEGER', mode='NULLABLE'),
        bigquery.SchemaField(name='CLIENT_TARGET_BUFFER', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='PHARMACY_TARGET_BUFFER', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='PROJ_ALPHA', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='GPI_UP_FAC', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='GPI_LOW_FAC', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='CLIENT_GR', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='GR_SCALE', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='GR_SF', field_type='INTEGER', mode='NULLABLE'),
        bigquery.SchemaField(name='LIM_TIER', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='TIER_LIST', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='CAPPED_ONLY', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='LAG_YTD_OVERRIDE', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='YTD_OVERRIDE', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='NDC_UPDATE', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='STRENGTH_PRICE_CHANGE_EXCEPTION', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='NO_MAIL', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='LIMITED_BO', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='MONTHLY_PHARM_YTD', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='PRICE_ZERO_PROJ_QTY', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='OUTPUT_FULL_MAC', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='CAPPED_OPT', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='CLIENT_TIERS', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='REMOVE_KRG_WMT_UC', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='PHARMACY_EXCLUSION', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='GPI_UP_DOLLAR', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='GPI_UP_FAC', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='GPI_LOW_FAC', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='AGG_UP_FAC', field_type='INTEGER', mode='NULLABLE'),
        bigquery.SchemaField(name='AGG_LOW_FAC ', field_type='INTEGER', mode='NULLABLE'),
        bigquery.SchemaField(name='OVER_REIMB_GAMMA', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='COST_GAMMA', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='PREF_OTHER_FACTOR ', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='INCLUDE_PLAN_LIABILITY', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='PLAN_LIAB_CLIENTS', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='PLAN_LIAB_WEIGHT', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='READ_IN_PLAN_LIAB_WIEGHTS', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='GAP_FACTOR', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='CAT_FACTOR', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='CUST_MOVEMENT_WEIGHTS', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='UNC_PHARMACY', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='PRICE_BOUNDS_UPPER', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='PRICE_BOUNDS_PERCENT_INCREASE', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='PRICE_BOUNDS_DOLLAR_INCREASE', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='UNC_ADJUSTMENT', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='PRICE_OVERRIDE_FILE', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='RAW_GOODRX', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='GOODRX_FILE', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='WC_SUGGESTED_GUARDRAILS', field_type='STRING', mode='NULLABLE'),    
        bigquery.SchemaField(name='GOODRX_FACTOR', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='APPLY_GENERAL_MULTIPLIER', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='APPLY_MAIL_MULTIPLIER', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='GENERAL_MULTIPLIER', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='MAIL_MULTIPLIER', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='ZBD_OPT', field_type='BOOLEAN', mode='NULLABLE'),    
        bigquery.SchemaField(name='LEAKAGE_RANK', field_type='INTEGER', mode='NULLABLE'),
        bigquery.SchemaField(name='ZBD_CVS_IND_SCALAR', field_type='FLOAT', mode='NULLABLE'),    
        bigquery.SchemaField(name='ZBD_CAPPED_SCALAR', field_type='FLOAT', mode='NULLABLE'),    
        bigquery.SchemaField(name='ZBD_CURRENT_PRICE_SCALAR', field_type='FLOAT', mode='NULLABLE'),      
        bigquery.SchemaField(name='PARITY_PRICE_DIFFERENCE_COLLAR_HIGH', field_type='FLOAT', mode='NULLABLE'),    
        bigquery.SchemaField(name='PARITY_PRICE_DIFFERENCE_COLLAR_LOW', field_type='FLOAT', mode='NULLABLE'),    
        bigquery.SchemaField(name='CROSS_CONTRACT_PROJ', field_type='BOOLEAN', mode='NULLABLE'),       
        bigquery.SchemaField(name='ZERO_QTY_TIGHT_BOUNDS', field_type='BOOLEAN', mode='NULLABLE'),    
        bigquery.SchemaField(name='REMOVE_WTW_RESTRICTION', field_type='BOOLEAN', mode='NULLABLE'),  
        bigquery.SchemaField(name='MAIL_RETAIL_BOUND', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='RETAIL_RETAIL_BOUND', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='OVER_REIMB_CHAINS', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='INTERCEPTOR_OPT', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='ALLOW_INTERCEPT_LIMIT', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='HANDLE_MAIL_CONFLICT_LEVEL', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='LEAKAGE_OPT', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='LEAKAGE_PENALTY', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='CLIENT_LOB', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='LEAKAGE_LIST', field_type='STRING', mode='NULLABLE'), 
        bigquery.SchemaField(name='EGWP', field_type='BOOLEAN', mode='NULLABLE'),  
        bigquery.SchemaField(name='BIG_CAPPED_PHARMACY_LIST', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='SMALL_CAPPED_PHARMACY_LIST', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='NON_CAPPED_PHARMACY_LIST', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='COGS_PHARMACY_LIST', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='PSAO_LIST', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='HANDLE_INFEASIBLE', field_type='BOOLEAN', mode='NULLABLE'), 
        bigquery.SchemaField(name='HANDLE_CONFLICT_GPI', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='CONFLICT_GPI_AS_TIERS', field_type='BOOLEAN', mode='NULLABLE'), 
        bigquery.SchemaField(name='COGS_BUFFER', field_type='FLOAT', mode='NULLABLE'), 
        bigquery.SchemaField(name='FORCE_FLOOR', field_type='BOOLEAN', mode='NULLABLE'), 
        bigquery.SchemaField(name='FORCE_FLOOR_PHARMACY_SUBGROUP_LIST', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='LOCKED_CLIENT', field_type='BOOLEAN', mode='NULLABLE'), 
        bigquery.SchemaField(name='R90_AS_MAIL', field_type='BOOLEAN', mode='NULLABLE'), 
        bigquery.SchemaField(name='MAIL_TARGET_BUFFER', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='RETAIL_TARGET_BUFFER', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='FULL_YEAR_LV_1_UPPER_BOUND', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='FULL_YEAR_LV_1_MAX_PERCENT_INCREASE', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='FULL_YEAR_LV_1_MAX_DOLLAR_INCREASE', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='FULL_YEAR_LV_2_UPPER_BOUND', field_type='STRING', mode='NULLABLE'), 
        bigquery.SchemaField(name='FULL_YEAR_LV_2_MAX_PERCENT_INCREASE', field_type='STRING', mode='NULLABLE'), 
        bigquery.SchemaField(name='FULL_YEAR_LV_2_MAX_DOLLAR_INCREASE', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='FULL_YEAR_LV_3_UPPER_BOUND', field_type='STRING', mode='NULLABLE'), 
        bigquery.SchemaField(name='FULL_YEAR_LV_3_MAX_PERCENT_INCREASE', field_type='STRING', mode='NULLABLE'), 
        bigquery.SchemaField(name='FULL_YEAR_LV_3_MAX_DOLLAR_INCREASE', field_type='STRING', mode='NULLABLE'), 
        bigquery.SchemaField(name='ZERO_QTY_WEIGHT', field_type='FLOAT', mode='NULLABLE'), 
        bigquery.SchemaField(name='PHARM_PERF_WEIGHT', field_type='FLOAT', mode='NULLABLE'), 
        bigquery.SchemaField(name='REMOVE_SMALL_CAPPED_UC', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='MAIL_NON_MAC_RATE', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='RETAIL_NON_MAC_RATE', field_type='FLOAT', mode='NULLABLE'), 
        bigquery.SchemaField(name='GUARANTEE_CATEGORY', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='INTERCEPT_CEILING', field_type='FLOAT', mode='NULLABLE'), 
        bigquery.SchemaField(name='SPECIALTY_OFFSET', field_type='BOOLEAN', mode='NULLABLE'), 
        bigquery.SchemaField(name='APPLY_FLOORS_MAIL', field_type='BOOLEAN', mode='NULLABLE'), 
        bigquery.SchemaField(name='MAIL_FLOORS_FACTOR', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='MAIL_MAC_UNRESTRICTED', field_type='BOOLEAN', mode='NULLABLE'), 
        bigquery.SchemaField(name='MAIL_UNRESTRICTED_CAP', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='CLIENT_RETAIL_OVRPERF_PEN', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='CLIENT_RETAIL_UNRPERF_PEN', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='CLIENT_MAIL_OVRPERF_PEN', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='CLIENT_MAIL_UNRPERF_PEN', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='RUN_TYPE_TABLEAU', field_type='STRING', mode='NULLABLE'),
        bigquery.SchemaField(name='IGNORED_VCMLS', field_type='INTEGER', mode='NULLABLE'),
        bigquery.SchemaField(name='CONFLICT_GPI_CUTOFF', field_type='INTEGER', mode='NULLABLE'),
        bigquery.SchemaField(name='COSTSAVER_CLIENT', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='APPLY_BENCHMARK_CAP', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='BENCHMARK_CAP_MULTIPLIER', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='UNIFORM_MAC_PRICING', field_type='BOOLEAN', mode='NULLABLE'),
        bigquery.SchemaField(name='RUR_GUARANTEE_BUFFER', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='PHARMACY_MCHOICE_OVRPERF_PEN', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='MEDD_MCHOICE_TARGET_RATE', field_type='FLOAT', mode='NULLABLE'),
        bigquery.SchemaField(name='REC_ADD_TS', field_type='DATETIME', mode='NULLABLE')
        ]
        
        return schema
    
