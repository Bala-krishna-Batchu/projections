# set/reset working directory to the clientpharmacymacoptimization repo location
import os
# #set it to true for Aetna runs
# client_lob = 'CMK'

# +
import kfp
import os
import importlib
import jinja2 as jj2
import datetime as dt
import random
import numpy as np
import socket
from pytz import timezone
from kfp.components import InputPath, OutputPath, func_to_container_op
from typing import NamedTuple, List
import GER_LP_Code.ClientPharmacyMacOptimization as opt
import GER_LP_Code.QA as qa
import GER_LP_Code.CPMO_reporting_to_IA as rp
import subprocess
import argparse 
import socket

from GER_LP_Code.audit_trail_utils import AuditTrail

parser = argparse.ArgumentParser(description='KF YAML Builder')
parser.add_argument('-v','--version', help='Version Iteration', required=True)
parser.add_argument('-kf','--kf_endpoint', help='Kubeflow Endpoint', default='https://7b9cc36d844fcd2d-dot-us-east1.pipelines.googleusercontent.com/#/runs')
parser.add_argument('-repo','--repo_head', help='Repo Directory', default='/home/jupyter/clientpharmacymacoptimization')
parser.add_argument('-algo_v','--algo_version', help='Algo Version', default='LP')
parser.add_argument('-v_type','--version_type', help='Version Type', default=f'WIP-{socket.gethostname()}')
parser.add_argument('-c_lob','--client_lob', help='Client LOB', default='CMK')
parser.add_argument('-cmpas','--cmpas', help='CMPAS', default='False')


args = vars(parser.parse_args())

def get_short_git_hash():
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip()

def get_git_branch():
    result = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip()


git_branch = get_git_branch()
git_hash = get_short_git_hash()

version_iteration = args['version'] 
kubeflow_endpoint = args['kf_endpoint'] 
repo_head = args['repo_head']
algo_version = args['algo_version']
version_type = args['version_type']
client_lob = args['client_lob']

os.chdir(repo_head)

project_name = 'pbm-mac-lp-prod-ai'
gcp_registry_name = 'us.gcr.io'
base_name = 'pbm_base'
script_run_name = 'pbm_script_run'
opt_name = 'pbm_opt'

audit_obj = AuditTrail(
        git_branch = git_branch,
        git_hash = git_hash, 
        algo_version = algo_version,
        version_iteration = version_iteration, # change as needed
        version_type = version_type,    # Use 'PROD' only for official versions
        project_name = 'pbm-mac-lp-prod-ai',
        bucket_name = 'pbm-mac-lp-prod-ai-bucket',
        audit_trail_folder = 'Audit_Trail_Data',
        audit_trail_dataset_name = 'pricing_management', 
        audit_trail_table_name = 'AT_Run_ID',
        gcp_registry_name = 'us.gcr.io',
        base_name = 'pbm_base',
        script_run_name = 'pbm_script_run',
        opt_name = 'pbm_opt',
        client_lob = client_lob
)
# -

# Set the Container Images/Versions

# # Get Latest Run ID from Audit Trail Table 
#
# - This should be done before the parameter settings because Run_ID needs to be added to parameter object. It would be better to do this another way, probably in the code?

# AT_RUN_ID = audit_obj.get_latest_run_id(table_name = 'AT_Run_ID')
# print("Latest RunID = ", AT_RUN_ID)

# #### Audit trail component update

# +
VERSION = f"{git_branch}-{git_hash}-{version_type}-{version_iteration}"
try:
    VERSION = VERSION.split('/')[1]
except:
    pass
BASE_TAG = f"{gcp_registry_name}/{project_name}/{base_name}:{VERSION}"
SCRIPT_RUN_TAG = f"{gcp_registry_name}/{project_name}/{script_run_name}:{VERSION}"
OPT_TAG = f"{gcp_registry_name}/{project_name}/{opt_name}:{VERSION}"
    
print(VERSION)
print(BASE_TAG)
print(SCRIPT_RUN_TAG)
print(OPT_TAG)


# +
def update_audit_trail(
    params_file_in: str, 
    git_branch: str, 
    git_hash: str, 
    algo_version: str,
    version_type: str,
    version_iteration: str,
    odp_lp: str,
    cmpas : str):
    
    import audit_trail_utils as audit
    audit_obj = audit.AuditTrail(
        git_branch = git_branch,
        git_hash = git_hash,
        algo_version = algo_version,
        version_type = version_type,
        version_iteration = version_iteration,
        odp_lp = odp_lp,
        cmpas = cmpas
    )
    audit_obj.update_audit_trail(params_file_in, odp_lp)

audit_trail_comp = func_to_container_op(
    func=update_audit_trail,
    base_image=BASE_TAG
)


# -

# #### Parameter Components

# +
def prepare_params(
    params_file_in: str
) -> NamedTuple('Outputs', [('LP_RUN', List[int]), ('month_indices', int), ('UNC_ADJUST',bool), ('UNC_FLAG',bool), ('skip_to_opt',bool), ('CONFLICT_GPI_PTH',str)]):
    '''User input parameter check and prep'''
    import re
    from google.cloud import storage
    
    # Download parameters file from storage
    local_file_name = 'CPMO_parameters.py'
    client = storage.Client()
    bp_blob = params_file_in[5:].split('/')
    b = bp_blob[0]    
    blob = '/'.join(bp_blob[1:])
    bucket = client.get_bucket(b)
    blob = bucket.get_blob(blob)
    assert blob, f'FileNotFound: Could not find parameters file: {params_file_in}'
    blob.download_to_filename(local_file_name)
    
    import CPMO_parameters as pp
    # get month indices to iterate over
    m_indices = list(range(len(pp.LP_RUN)))[0]

    return (pp.LP_RUN, m_indices, pp.UNC_ADJUST, False, pp.SKIP_TO_OPT, pp.FILE_OUTPUT_PATH+pp.CONFLICT_GPI_LIST_FILE_THIS_RUN)

params_comp = func_to_container_op(prepare_params, base_image=BASE_TAG)


# +
def lp_run(
    month_index: int, lp_run: List[int]
) -> NamedTuple('Output', [('in_lp_run', bool), ('month', int)]):
    '''Auxiliary func to determine if month is in p.LP_RUN.
    If not, the output is used to skip other steps and run opt.no_lp_run.'''
    month = eval(str(lp_run))[month_index]
    run = month in eval(str(lp_run))
    return (run, month)

lp_run_comp = func_to_container_op(lp_run, base_image='python:3.8-slim-buster')


# +
def conflict_gpi_run(
    params_file_in: str,
    file_in: str,
    unc_adjust_in: bool,
) -> NamedTuple('Outputs', [('conflict_gpi', bool),('unc_flag', bool),('unc_adjust', bool),('run_recursive',bool)]):
    
    '''check conflict gpi file and unc_adjust'''
    
    import pandas as pd
    import re
    import os
    from google.cloud import storage,bigquery
      
    # Download parameters file from storage
    local_file_name = 'CPMO_parameters.py'
    client = storage.Client()
    bp_blob = params_file_in[5:].split('/')
    b = bp_blob[0]    
    blob = '/'.join(bp_blob[1:])
    bucket = client.get_bucket(b)
    blob = bucket.get_blob(blob)
    assert blob, f'FileNotFound: Could not find parameters file: {params_file_in}'
    blob.download_to_filename(local_file_name)
    
    import CPMO_parameters as pp
    
    conflict_gpi_file = pd.read_csv(file_in,dtype={'GPI':str,'NDC':str,'GPI_NDC':str,'CLIENT':str})
    gpi_list= conflict_gpi_file.GPI_NDC[conflict_gpi_file.GPI_NDC.notna()].unique()
    
    if len(conflict_gpi_file) > 0 and pp.RUN_TIME < 2 : 
        
        conflict_gpi = True
        
        # read parameters file from storage
        param_name = 'CPMO_parameters.py'
        client = storage.Client()
        bp_blob = params_file_in[5:].split('/')
        b = bp_blob[0]    
        blob = '/'.join(bp_blob[1:])
        bucket = client.get_bucket(b)
        blob = bucket.get_blob(blob)
        #change the name of the param file
        new_blob = bucket.rename_blob(blob, '/'.join(bp_blob[1:-1]) +'/'+ bp_blob[-1].replace('.py','') + '_before_conflict_gpi.py')
        
        new_blob.download_to_filename(param_name)
        
        import CPMO_parameters as pp
        
        if len(gpi_list) > pp.CONFLICT_GPI_CUTOFF:
            
            param_dic ={'HANDLE_CONFLICT_GPI = False': 'HANDLE_CONFLICT_GPI = True',
                        'CONFLICT_GPI_AS_TIERS = True':'CONFLICT_GPI_AS_TIERS = False',
                        'RUN_TIME = {}'.format(pp.RUN_TIME): 'RUN_TIME = {}'.format(pp.RUN_TIME+1)
                       }
        else:
            
            param_dic ={'HANDLE_CONFLICT_GPI = False': 'HANDLE_CONFLICT_GPI = True',
                        'RUN_TIME = {}'.format(pp.RUN_TIME): 'RUN_TIME = {}'.format(pp.RUN_TIME+1)
                       }

        with open(param_name, 'r+') as file:

            content = file.read()  
            for k, v in param_dic.items():
                content = content.replace(k, v)

            file.seek(0)
            file.truncate()

            file.write(content)

        blob.upload_from_filename(param_name)
        
        #os.makedirs(os.path.join(p.PROGRAM_OUTPUT_PATH, 'Output'), exist_ok=True)
    
    elif len(conflict_gpi_file) > 0 and pp.RUN_TIME >= 2:
        
        conflict_gpi = True
        
        #gpi_list= conflict_gpi_file.GPI_NDC[conflict_gpi_file.GPI_NDC.notna()].unique()
        customer_id = conflict_gpi_file.CLIENT[0]

        bq_client = bigquery.Client()
        vcml_pricing_query = f"""
        select MAC, GPI, NDC, GPI_NDC, PRICE from `{pp.BQ_INPUT_PROJECT_ID}.{pp.BQ_INPUT_DATASET_DS_PRO_LP}.{pp.AETNA_TABLE_ID_PREFIX}mac_list` where GPI_NDC in ("{'", "'.join(gpi_list)}")
        and mac in (
                    select vcml_id from `{pp.BQ_INPUT_PROJECT_ID}.{pp.BQ_INPUT_DATASET_DS_PRO_LP}.{pp.AETNA_TABLE_ID_PREFIX}vcml_reference{pp.WS_SUFFIX}` 
                    where customer_id in ('{customer_id}')
                    )
                              """
        pricing_query_job = bq_client.query(vcml_pricing_query).to_dataframe()

        anomaly_gpi_file = pd.merge(conflict_gpi_file[['CLIENT','GPI_NDC']].drop_duplicates(),pricing_query_job,on='GPI_NDC')
        anomaly_gpi_file['VCML_ID'] = anomaly_gpi_file.MAC.copy()
        anomaly_gpi_file['REGION'] = 'ALL'
        #anomaly_gpi_file['GPI_NDC'] = anomaly_gpi_file['GPI'].astype(str) + "_***********"
        #anomaly_gpi_file['NDC'] = "***********"
        anomaly_gpi_file = anomaly_gpi_file.rename(columns={'PRICE':'PRICE_OVRD_AMT'})
        anomaly_gpi_file = anomaly_gpi_file[['CLIENT','GPI_NDC','GPI','NDC','PRICE_OVRD_AMT','VCML_ID','REGION']]

        if pp.PRICE_OVERRIDE and len(gpi_list) <= pp.CONFLICT_GPI_CUTOFF:

            price_override = pd.read_csv(os.path.join(pp.FILE_INPUT_PATH, pp.PRICE_OVERRIDE_FILE), dtype = pp.VARIABLE_TYPE_DIC)
            price_override_new = pd.concat([anomaly_gpi_file, price_override])
            price_override_new = price_override_new.drop_duplicates()
            # will conflicts in overrides already ?? impossible
            #price_override_new.groupby(['CLIENT','GPI_NDC','GPI','NDC','VCML_ID','REGION'])
            price_override_new.to_csv(os.path.join(pp.FILE_INPUT_PATH, pp.PRICE_OVERRIDE_FILE),index = False)

        else: 
            
            # read parameters file from storage
            param_name = 'CPMO_parameters.py'
            client = storage.Client()
            bp_blob = params_file_in[5:].split('/')
            b = bp_blob[0]    
            blob = '/'.join(bp_blob[1:])
            bucket = client.get_bucket(b)
            blob = bucket.get_blob(blob)
            #change the name of the param file
            new_blob = bucket.rename_blob(blob, '/'.join(bp_blob[1:-1]) +'/'+ bp_blob[-1].replace('.py','') + '_before_conflict_gpi.py')
            new_blob.download_to_filename(param_name)
                      
            if len(gpi_list) > pp.CONFLICT_GPI_CUTOFF:
            
                param_dic ={'HANDLE_CONFLICT_GPI = False': 'HANDLE_CONFLICT_GPI = True',
                            'CONFLICT_GPI_AS_TIERS = True':'CONFLICT_GPI_AS_TIERS = False',
                            'RUN_TIME = {}'.format(pp.RUN_TIME): 'RUN_TIME = {}'.format(pp.RUN_TIME+1)
                           }
            else:

                param_dic ={'HANDLE_CONFLICT_GPI = False': 'HANDLE_CONFLICT_GPI = True',
                            'PRICE_OVERRIDE = False': 'PRICE_OVERRIDE = True',
                            'RUN_TIME = {}'.format(pp.RUN_TIME): 'RUN_TIME = {}'.format(pp.RUN_TIME+1)
                           }

            with open(param_name, 'r+') as file:

                content = file.read()  
                for k, v in param_dic.items():
                    content = content.replace(k, v)

                file.seek(0)
                file.truncate()

                file.write(content)
                
            blob.upload_from_filename(param_name)

            anomaly_gpi_file.to_csv(os.path.join(pp.FILE_INPUT_PATH, pp.PRICE_OVERRIDE_FILE),index = False)
    
    else: conflict_gpi = False
 
    # the logic of just calling recursion once
    if conflict_gpi == True and unc_adjust_in == True:
        # run recursive before the solver then run recursive of the whole 
        run_recursive = True
        unc_flag = False
        unc_adjust = True
    elif conflict_gpi == False and unc_adjust_in == True:
        #run solver then run recursive
        run_recursive = True
        unc_flag = True
        # the key of just run unc_adjust once
        unc_adjust = False
    elif conflict_gpi == True and unc_adjust_in == False:
        #run recursive before the solver
        run_recursive = True
        unc_flag = False
        unc_adjust = False
    elif conflict_gpi == False and unc_adjust_in == False:
        #run solver
        run_recursive = False
        unc_flag = False
        unc_adjust = False
        
    return (conflict_gpi,unc_flag,unc_adjust,run_recursive)

conflict_gpi_run_comp = func_to_container_op(conflict_gpi_run, base_image=BASE_TAG)


# -

# #### Script Run Component

# +
def script_run(
    script_name: str, 
#     local_output_dir: str, 
    params_file_in: str
):
    '''Run script on image and export outputs to directory specified in parameters file'''
    import os
    import logging
    import util_funcs as uf
    
    uf.write_params(params_file_in)
    import CPMO_parameters as p
    import util_funcs as uf
    import subprocess as sp
    
    logging.info(f"running {script_name} ...")
    res = sp.check_call(["python", script_name])
    logging.info(f"finished script {script_name}")

script_run_comp = func_to_container_op(
    func=script_run,
    base_image=SCRIPT_RUN_TAG
)
# -

# #### Optimization Components

opt_prep_comp = func_to_container_op(
    func=opt.opt_preprocessing,
    base_image=OPT_TAG,
)
##Constraint Components
opt_pricing_constraints_comp = func_to_container_op(
    func=opt.consistent_strength_pricing_constraints,
    base_image=OPT_TAG,
)
opt_client_constraints_comp = func_to_container_op(
    func=opt.client_level_constraints,
    base_image=OPT_TAG,
)
pprice_lt_npprice_constraints_comp = func_to_container_op(
    func=opt.preferred_pricing_less_than_non_preferred_pricing_constraints,
    base_image=OPT_TAG    
)
specific_pricing_constraints_comp = func_to_container_op(
    func=opt.specific_pricing_constraints,
    base_image=OPT_TAG
)
brand_generic_pricing_constraints_comp = func_to_container_op(
    func=opt.brand_generic_pricing_constraints,
    base_image=OPT_TAG
)
adj_cap_constraints_comp = func_to_container_op(
    func=opt. adj_cap_constraints,
    base_image=OPT_TAG
)
cvs_parity_price_constraint_comp = func_to_container_op(
    func=opt.cvs_parity_price_constraint,
    base_image=OPT_TAG
)
state_parity_constraint_comp = func_to_container_op(
    func=opt.state_parity_constraints,
    base_image=OPT_TAG
)
mac_constraints_comp = func_to_container_op(
    func=opt.consistent_mac_constraints,
    base_image=OPT_TAG
)
agg_mac_price_change_constraints_comp = func_to_container_op(
    func=opt.agg_mac_constraints,
    base_image=OPT_TAG
)
equal_package_size_constraints_comp = func_to_container_op(
    func=opt.equal_package_size_constraints,
    base_image=OPT_TAG
)
same_difference_package_size_constraints_comp = func_to_container_op(
    func=opt.same_difference_package_size_constraints,
    base_image=OPT_TAG
)
same_therapeutic_constraints_comp = func_to_container_op(
    func=opt.same_therapeutic_constraints,
    base_image=OPT_TAG
)
leakage_optimization_constraints_comp = func_to_container_op(
    func=opt.leakage_opt,
    base_image=OPT_TAG
)
##Generate Conflict GPI
generate_conflict_gpi_comp = func_to_container_op(
    func=opt.generate_conflict_gpi,
    base_image=OPT_TAG
)
##LP Solver
run_solver_comp = func_to_container_op(
    func=opt.run_solver,
    base_image=OPT_TAG
)
##no LP run condition
no_lp_run_comp = func_to_container_op(
    func=opt.no_lp_run,
    base_image=BASE_TAG
)
##LP output component
lp_output_comp = func_to_container_op(
    func=opt.lp_output,
    base_image=OPT_TAG
)

# #### QA.py Components

qa_Pharmacy_Output_comp = func_to_container_op(
    func=qa.qa_Pharmacy_Output,
    base_image=BASE_TAG
)
qa_Price_Check_Output_comp = func_to_container_op(
    func=qa.qa_Price_Check_Output,
    base_image=BASE_TAG
)
qa_price_output_comp = func_to_container_op(
    func=qa.qa_price_output,
    base_image=BASE_TAG
)
qa_price_tiering_rules_REPORT_comp = func_to_container_op(
    func=qa.qa_price_tiering_rules_REPORT,
    base_image=OPT_TAG  # use opt since it has xlsxwriter
)
qa_Prices_above_MAC1026_floor_comp = func_to_container_op(
    func=qa.qa_Prices_above_MAC1026_floor,
    base_image=BASE_TAG
)
qa_pref_nonpref_pharm_pricing_comp = func_to_container_op(
    func=qa.qa_pref_nonpref_pharm_pricing,
    base_image=BASE_TAG
)
qa_test_performance_comp = func_to_container_op(
    func=qa.qa_test_performance,
    base_image=BASE_TAG
)
qa_test_xvcml_comp = func_to_container_op(
    func=qa.qa_test_xvcml_meas_parity,
    base_image=BASE_TAG
)
qa_test_price_changes_file_comp = func_to_container_op(
    func=qa.qa_test_price_changes_file,
    base_image=BASE_TAG
)
qa_goodrx_price_bound_comp = func_to_container_op(
    func=qa.qa_goodrx_price_bound,
    base_image=BASE_TAG
)
qa_r90_as_mail_comp = func_to_container_op(
    func=qa.qa_r90_as_mail,
    base_image=BASE_TAG
)
qa_price_overall_reasonableness_comp = func_to_container_op(
    func=qa.qa_price_overall_reasonableness,
    base_image=BASE_TAG
)
qa_diagnostic_report_comp = func_to_container_op(
    func=qa.qa_diagnostic_report,
    base_image=BASE_TAG
)

# #### CPMO_reporting_to_IA.py Components

# + jupyter={"outputs_hidden": false} pycharm={"name": "#%%\n"}
rp_create_reporting_tables_comp = func_to_container_op(
    func=rp.create_reporting_tables,
    base_image=BASE_TAG
)


# -

# #### Graph component wrapper for preprocessing components before CPMO
#
# `<preprocess_graph>` is a wrapper that replaces duplicate use of the preprocessing components in `UNC_ADJUST = True` and `SKIP_TO_OPT = False` (simulation) modes. Note that, currently, the information required by the components which run after `<preprocess_graph>` are all pickled, i.e., eigther stored in output files or are written to BigQuery.

@kfp.dsl.graph_component
def preprocess_graph(params_file_in: str, skip_to_opt: bool, month):
    #<skip_to_opt = True> would skip the Pre_processing.py, qa_checks.py, and Daily_input_Read.py script runs
    #this is used in multi-price (multi-time) simulation for iterations >= 1 to skip these components
    #also used when <UNC_ADJUST = True> to skip the same components
    
    with kfp.dsl.Condition(skip_to_opt == False, name = 'Full_Pipeline'):
        
        ##Pre_processing.py script run
        prep_op = script_run_comp(
            script_name='Pre_Processing.py',
            #local_output_dir='Input/Data Automation',
            params_file_in=params_file_in
        ).set_display_name(
            'Preprocessing'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
        prep_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
        
        ##QA_INPUT_DATA.py script run
        #    input_qa_op = script_run_comp(
        #    script_name='QA_INPUT_DATA.py',
        #    local_output_dir='Input',
        #    params_file_in=params_file_in
        #).set_display_name(
        #    'QA Input'
        #).set_memory_request('400M').set_cpu_request('1000m').set_timeout(1000)
        #input_qa_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
        #input_qa_op.after(prep_op)
        
        ##qa_checks.py script run
        qa_checks_op = script_run_comp(
            script_name='qa_checks.py',
            params_file_in=params_file_in
        ).set_display_name(
            'QA Checks'
        ).set_memory_request('1G').set_cpu_request('2000m').set_timeout(2000).set_retry(num_retries = 5)
        qa_checks_op.after(prep_op)
        qa_checks_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
        
        ##Daily_Input_Read.py script run
        daily_input_read_op = script_run_comp(
            script_name='Daily_Input_Read.py',
            #local_output_dir='Dynamic_Input',
            params_file_in=params_file_in
        ).set_display_name(
            'Daily Input Read'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
        daily_input_read_op.after(prep_op)
        daily_input_read_op.execution_options.caching_strategy.max_cache_staleness = "P0D"


# #### Graph component wrapper for optimization components
#
# `<optimization_graph>` is a wrapper that replaces duplicate use of the optimization components in `UNC_ADJUST = True` mode. Note that, currently, the information required by the components which run after `<optimization_graph>` is stored in output files or on BQ.

@kfp.dsl.graph_component
def optimization_graph(params_file_in: str, LP_RUN: List[int] , month: int, UNC_ADJUST:bool, UNC_FLAG: bool, CONFLICT_GPI_PTH: str):
    
    ##optimization preprocessing component to prepare dataframes for CMPO run
    opt_prep_op = opt_prep_comp(
        m = month, 
        params_file_in = params_file_in,
        unc_flag = UNC_FLAG
    ).set_display_name(
        'Opt Preprocessing'
    ).set_memory_request('1G').set_cpu_request('2000m').set_timeout(2000).set_retry(num_retries = 5)
    opt_prep_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    ##determine if LP_RUN is true
    ##NOTE: this component might not be neccessary since the simulation mode has changed
    lp_run_op = lp_run_comp(
        month_index = month,
        lp_run = LP_RUN
    ).set_display_name(
        'LP Run Check Support Component'
    ).set_memory_request('10M').set_cpu_request('10m').set_timeout(2000).set_retry(num_retries = 5)
    lp_run_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    ##constraint component to write pricing strenght constraints
    ##price of higher dose > price of lower dose
    pricing_constraints_op = opt_pricing_constraints_comp(
            params_file_in = params_file_in,
            unc_flag = UNC_FLAG,
            lp_data_df_in = opt_prep_op.outputs['lp_data_df_out'],
            price_lambdas_in = opt_prep_op.outputs['price_lambdas_out'],
        ).set_display_name(
            'Consistent Strength Pricing Constraints'
        ).set_memory_request('1G').set_cpu_request('2000m').set_timeout(2000).set_retry(num_retries = 5)
    pricing_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    ##constraint component to write client and pharmacy level performance guarantee constraints
    ##client performance + surplus - liability = client guarantee performance
    ##pharmacy performance + surplus - liability = pharmacy guarantee performance 
    client_constraints_op = opt_client_constraints_comp(
            params_file_in = params_file_in,
            unc_flag = UNC_FLAG,
            lp_data_df_in = opt_prep_op.outputs['lp_data_df_out'],
            client_guarantees_in = opt_prep_op.outputs['client_guarantees_out'],
            pharmacy_guarantees_in = opt_prep_op.outputs['pharmacy_guarantees_out'],
            performance_dict_in = opt_prep_op.outputs['performance_dict_out'],
            breakout_df_in = opt_prep_op.outputs['breakout_df_out'],
            client_list_in = opt_prep_op.outputs['client_list_out'],
            #pharmacy_approx_in = opt_prep_op.outputs['pharmacy_approx_out'],
            oc_eoy_pharm_perf_in = opt_prep_op.outputs['oc_eoy_pharm_perf_out'],
            gen_launch_eoy_dict_in = opt_prep_op.outputs['gen_launch_eoy_dict_out'],
            brand_surplus_eoy_in = opt_prep_op.outputs['brand_surplus_eoy_out'],
            specialty_surplus_eoy_in = opt_prep_op.outputs['specialty_surplus_eoy_out'],
            disp_fee_surplus_eoy_in = opt_prep_op.outputs['disp_fee_surplus_eoy_out'],
            price_lambdas_in = opt_prep_op.outputs['price_lambdas_out'],
            #total_pharm_list_in = opt_prep_op.outputs['total_pharm_list_out'],
            #agreement_pharmacy_list_in = opt_prep_op.outputs['agreement_pharmacy_list_out'],
        ).set_display_name(
            'Client Level Constraints'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
    client_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    ##constraint component to write perferred/non-preferred pricing constraints
    ##preferred pharmacy prices <= non-preferred pharmacy prices
    pprice_lt_npprice_constraints_op = pprice_lt_npprice_constraints_comp(
            params_file_in = params_file_in,
            unc_flag = UNC_FLAG,
            lp_data_df_in = opt_prep_op.outputs['lp_data_df_out'],
            pref_pharm_list_in = opt_prep_op.outputs['pref_pharm_list_out'],
            #total_pharm_list_in = opt_prep_op.outputs['total_pharm_list_out']
        ).set_display_name(
            'Preferred LT Non-Preferred Pricing Constraints'
        ).set_memory_request('1G').set_cpu_request('2000m').set_timeout(2000).set_retry(num_retries = 5)
    pprice_lt_npprice_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    ##constraint component to write measure specific pricing constraints 
    ## M30 prices <= R90 prices <= R30 prices 
    specific_pricing_constraints_op = specific_pricing_constraints_comp(
            params_file_in = params_file_in,
            unc_flag = UNC_FLAG,
            lp_data_df_in = opt_prep_op.outputs['lp_data_df_out'],
            #total_pharm_list_in = opt_prep_op.outputs['total_pharm_list_out']
        ).set_display_name(
            'Measure Specific Pricing Constraints'
        ).set_memory_request('1G').set_cpu_request('2000m').set_timeout(2000).set_retry(num_retries = 5)
    specific_pricing_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    ##constraint component to write brand generic pricing constraints 
    ## Generic prices <= Brand prices
    brand_generic_pricing_constraints_op = brand_generic_pricing_constraints_comp(
            params_file_in = params_file_in,
            unc_flag = UNC_FLAG,
            lp_data_df_in = opt_prep_op.outputs['lp_data_df_out'],
            #total_pharm_list_in = opt_prep_op.outputs['total_pharm_list_out']
        ).set_display_name(
            'Brand Generic Pricing Constraints'
        ).set_memory_request('1G').set_cpu_request('2000m').set_timeout(2000).set_retry(num_retries = 5)
    brand_generic_pricing_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    ##constraint component to write adjudication cap pricing constraints 
    ##adjudication for year < cap * guaranteed price
    adj_cap_constraints_op =  adj_cap_constraints_comp(
            params_file_in = params_file_in,
            unc_flag = UNC_FLAG,
            lp_data_df_in = opt_prep_op.outputs['lp_data_df_out'],
            #total_pharm_list_in = opt_prep_op.outputs['total_pharm_list_out']
        ).set_display_name(
            'Adjudication Cap Pricing Constraints'
        ).set_memory_request('1G').set_cpu_request('2000m').set_timeout(2000).set_retry(num_retries = 5)
    adj_cap_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    ##constraint component to write CVS parity pricing constraints
    ##all pharmacy prices >= CVS prices
    cvs_parity_price_constraint_op = cvs_parity_price_constraint_comp(
            params_file_in = params_file_in,
            unc_flag = UNC_FLAG,
            lp_data_df_in = opt_prep_op.outputs['lp_data_df_out'],
            pref_pharm_list_in = opt_prep_op.outputs['pref_pharm_list_out']                
        ).set_display_name(
            'CVS Parity Constraint'
        ).set_memory_request('1G').set_cpu_request('2000m').set_timeout(2000).set_retry(num_retries = 5)
    cvs_parity_price_constraint_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    ##constraint component to write State Parity pricing constraints
    ##CVS prices should <= IND prices on the CVSSP VCML
    state_parity_constraint_op = state_parity_constraint_comp(
            params_file_in = params_file_in,
            unc_flag = UNC_FLAG,
            lp_data_df_in = opt_prep_op.outputs['lp_data_df_out']
    ).set_display_name(
        'State Parity Constraint'
    ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 5)
    state_parity_constraint_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    ##constraint component to write consistent MAC pricing constraints
    ##common MAC list prices should be equal
    mac_constraints_op = mac_constraints_comp(
            params_file_in = params_file_in,
            unc_flag = UNC_FLAG,
            lp_data_df_in = opt_prep_op.outputs['lp_data_df_out']
        ).set_display_name(
            'Consistent MAC constraints'
        ).set_memory_request('1G').set_cpu_request('2000m').set_timeout(2000).set_retry(num_retries = 5)
    mac_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    ##constraint component to write aggregate MAC price change constraints
    ##prices should perform within an aggregated upper and lower bound w.r.t. MAC price list
    agg_mac_price_change_constraints_op = agg_mac_price_change_constraints_comp(
            params_file_in = params_file_in,
            unc_flag = UNC_FLAG,
            month = lp_run_op.outputs['month'],
            lp_data_df_in = opt_prep_op.outputs['lp_data_df_out']                
        ).set_display_name(
            'Aggregate MAC Price Change Constraints'
        ).set_memory_request('1G').set_cpu_request('2000m').set_timeout(2000).set_retry(num_retries = 5)
    agg_mac_price_change_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    ##constraint component to write package size pricing constraints
    ##equal package size drugs have equal prices
    equal_package_size_constraints_op = equal_package_size_constraints_comp(
            params_file_in = params_file_in,
            unc_flag = UNC_FLAG,
            lp_data_df_in = opt_prep_op.outputs['lp_data_df_out']
        ).set_display_name(
            'Equal Package Size Contraints'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
    equal_package_size_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    ##constraint component to write difference package size pricing constraints
    ##the difference in prices of different package sizes of the same drug is bounded in proportion to ther MAC list price difference
    same_difference_package_size_constraints_op = same_difference_package_size_constraints_comp(
            params_file_in = params_file_in,
            unc_flag = UNC_FLAG,
            lp_data_df_in = opt_prep_op.outputs['lp_data_df_out']            
        ).set_display_name(
            'Same Difference Package Size Constraints'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
    same_difference_package_size_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    ##constraint component to write same therapeutic class constraints
    ##the drugs that belongs to the same therapeutic class are bounded to maintain their price ratio before vs. after
    same_therapeutic_constraints_op = same_therapeutic_constraints_comp(
            params_file_in = params_file_in,
            unc_flag = UNC_FLAG,
            lp_data_df_in = opt_prep_op.outputs['lp_data_df_out']            
        ).set_display_name(
            'Same Therapeutic Class Constraints'
        ).set_memory_request('1G').set_cpu_request('2000m').set_timeout(2000).set_retry(num_retries = 5)
    same_therapeutic_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"

    ##constraint component to write leakage optimization constraints
    ##calculates amount of leakage generated for a given price and adds it as a penalty to the overall objective
    leakage_optimization_constraints_op = leakage_optimization_constraints_comp(
            params_file_in = params_file_in,
            lp_data_df_in = opt_prep_op.outputs['lp_data_df_out']            
        ).set_display_name(
            'Leakage Optimization Constraints'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
    leakage_optimization_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"    
    
    ##generating conflicting GPIs
    generate_conflict_gpi_op = generate_conflict_gpi_comp(
            month = lp_run_op.outputs['month'],
            params_file_in = params_file_in,
            unc_flag = UNC_FLAG,
            t_cost_in = pricing_constraints_op.outputs['t_cost_out'],
            cons_strength_cons_in = pricing_constraints_op.outputs['cons_strength_cons_out'],
            client_constraint_list_in = client_constraints_op.outputs['client_constraint_list_out'],
            client_constraint_target_in = client_constraints_op.outputs['client_constraint_target_out'],
            pref_lt_non_pref_cons_list_in = pprice_lt_npprice_constraints_op.outputs['pref_lt_non_pref_cons_list_out'],
            meas_specific_price_cons_list_in = specific_pricing_constraints_op.outputs['meas_specific_price_cons_list_out'],
            brnd_gnrc_price_cons_list_in = brand_generic_pricing_constraints_op.outputs['brnd_gnrc_price_cons_list_out'],
            adj_cap_price_cons_list_in =  adj_cap_constraints_op.outputs['adj_cap_price_cons_list_out'],
            pref_other_price_cons_list_in = cvs_parity_price_constraint_op.outputs['pref_other_price_cons_list_out'],
            parity_price_cons_list_in = state_parity_constraint_op.outputs['parity_price_cons_list_out'],
            mac_cons_list_in = mac_constraints_op.outputs['mac_cons_list_out'],
            agg_mac_cons_list_in = agg_mac_price_change_constraints_op.outputs['agg_mac_cons_list_out'],
            eq_pkg_sz_cons_list_in = equal_package_size_constraints_op.outputs['eq_pkg_sz_cons_list_out'],
            sm_diff_pkg_sz_cons_list_in = same_difference_package_size_constraints_op.outputs['sm_diff_pkg_sz_cons_list_out'],
            sm_thera_class_cons_list_in = same_therapeutic_constraints_op.outputs['sm_thera_class_cons_list_out'],
            leakage_cost_list_in = leakage_optimization_constraints_op.outputs['leakage_cost_list_out'],
            leakage_const_list_in = leakage_optimization_constraints_op.outputs['leakage_const_list_out'],
            lambda_df_in = client_constraints_op.outputs['lambda_df_out'],
            lp_vol_mv_agg_df_in = opt_prep_op.outputs['lp_vol_mv_agg_df_out'],
            breakout_df_in = opt_prep_op.outputs['breakout_df_out'],
            #total_pharm_list_in = opt_prep_op.outputs['total_pharm_list_out'],
            lp_data_df_in = pricing_constraints_op.outputs['lp_data_df_out'],
            anomaly_gpi_in = pprice_lt_npprice_constraints_op.outputs['anomaly_gpi_out'],
            anomaly_mes_gpi_in = specific_pricing_constraints_op.outputs['anomaly_mes_gpi_out'],
            anomaly_bg_gpi_in = brand_generic_pricing_constraints_op.outputs['anomaly_bg_gpi_out'],
            anomaly_adj_cap_gpi_in =  adj_cap_constraints_op.outputs['anomaly_adj_cap_gpi_out'],
            anomaly_pref_gpi_in = cvs_parity_price_constraint_op.outputs['anomaly_pref_gpi_out'],
            anomaly_const_pkg_sz_in = equal_package_size_constraints_op.outputs['anomaly_const_pkg_sz_out'],
            anomaly_state_parity_gpi_in = state_parity_constraint_op.outputs['anomaly_state_parity_gpi_out'],
            anomaly_const_mac_in =  mac_constraints_op.outputs['anomaly_const_mac_out'],
            anomaly_sm_thera_gpi_in = same_therapeutic_constraints_op.outputs['anomaly_sm_thera_gpi_out'],
        ).set_display_name(
            'Generate Conflict GPI'
        ).set_memory_request('2G').set_cpu_request('1000m').set_timeout(10000).set_retry(num_retries = 3)
    generate_conflict_gpi_op.execution_options.caching_strategy.max_cache_staleness = "P0D"   
    

     ##NOTE: xx
    conflict_gpi_run_op = conflict_gpi_run_comp(
        params_file_in=params_file_in,
        file_in = CONFLICT_GPI_PTH,
        unc_adjust_in = UNC_ADJUST,
    ).set_display_name(
        'Conflict GPI Run Check Support Component'
    ).set_memory_request('10M').set_cpu_request('10m').set_timeout(2000).set_retry(num_retries = 3)
    conflict_gpi_run_op.after(generate_conflict_gpi_op)
    conflict_gpi_run_op.execution_options.caching_strategy.max_cache_staleness = "P0D"

        
    with kfp.dsl.Condition(conflict_gpi_run_op.outputs['conflict_gpi'] == False, name = 'HANDLE_CONFLICT_GPI'):  
        
        ##component to submit the LP to linear solver
        run_solver_op = run_solver_comp(
                month = lp_run_op.outputs['month'],
                params_file_in = params_file_in,
                unc_flag = UNC_FLAG,
                t_cost_in = pricing_constraints_op.outputs['t_cost_out'],
                cons_strength_cons_in = pricing_constraints_op.outputs['cons_strength_cons_out'],
                client_constraint_list_in = client_constraints_op.outputs['client_constraint_list_out'],
                client_constraint_target_in = client_constraints_op.outputs['client_constraint_target_out'],
                pref_lt_non_pref_cons_list_in = pprice_lt_npprice_constraints_op.outputs['pref_lt_non_pref_cons_list_out'],
                meas_specific_price_cons_list_in = specific_pricing_constraints_op.outputs['meas_specific_price_cons_list_out'],
                brnd_gnrc_price_cons_list_in = brand_generic_pricing_constraints_op.outputs['brnd_gnrc_price_cons_list_out'],
                adj_cap_price_cons_list_in =  adj_cap_constraints_op.outputs['adj_cap_price_cons_list_out'],
                pref_other_price_cons_list_in = cvs_parity_price_constraint_op.outputs['pref_other_price_cons_list_out'],
                parity_price_cons_list_in = state_parity_constraint_op.outputs['parity_price_cons_list_out'],
                mac_cons_list_in = mac_constraints_op.outputs['mac_cons_list_out'],
                agg_mac_cons_list_in = agg_mac_price_change_constraints_op.outputs['agg_mac_cons_list_out'],
                eq_pkg_sz_cons_list_in = equal_package_size_constraints_op.outputs['eq_pkg_sz_cons_list_out'],
                sm_diff_pkg_sz_cons_list_in = same_difference_package_size_constraints_op.outputs['sm_diff_pkg_sz_cons_list_out'],
                sm_thera_class_cons_list_in = same_therapeutic_constraints_op.outputs['sm_thera_class_cons_list_out'],
                leakage_cost_list_in = leakage_optimization_constraints_op.outputs['leakage_cost_list_out'],
                leakage_const_list_in = leakage_optimization_constraints_op.outputs['leakage_const_list_out'],
                lambda_df_in = client_constraints_op.outputs['lambda_df_out'],
                lp_vol_mv_agg_df_in = opt_prep_op.outputs['lp_vol_mv_agg_df_out'],
                breakout_df_in = opt_prep_op.outputs['breakout_df_out'],
                #total_pharm_list_in = opt_prep_op.outputs['total_pharm_list_out'],
                lp_data_df_in = pricing_constraints_op.outputs['lp_data_df_out'],
                anomaly_gpi_in = pprice_lt_npprice_constraints_op.outputs['anomaly_gpi_out'],
                anomaly_mes_gpi_in = specific_pricing_constraints_op.outputs['anomaly_mes_gpi_out'],
                anomaly_bg_gpi_in = brand_generic_pricing_constraints_op.outputs['anomaly_bg_gpi_out'],
                anomaly_adj_cap_gpi_in =  adj_cap_constraints_op.outputs['anomaly_adj_cap_gpi_out'],
                anomaly_pref_gpi_in = cvs_parity_price_constraint_op.outputs['anomaly_pref_gpi_out'],
                anomaly_const_pkg_sz_in = equal_package_size_constraints_op.outputs['anomaly_const_pkg_sz_out'],
                anomaly_state_parity_gpi_in =  state_parity_constraint_op.outputs['anomaly_state_parity_gpi_out'],
                anomaly_const_mac_in =  mac_constraints_op.outputs['anomaly_const_mac_out'],
                anomaly_sm_thera_gpi_in = same_therapeutic_constraints_op.outputs['anomaly_sm_thera_gpi_out'],
            ).set_display_name(
                'Run LP Solver'
            ).set_memory_request('2G').set_cpu_request('1000m').set_timeout(10000).set_retry(num_retries = 2)
        run_solver_op.execution_options.caching_strategy.max_cache_staleness = "P0D"

        ##component to produce final outputs
        lp_output_op = lp_output_comp(
                params_file_in = params_file_in,
                #m = m,
                unc_flag = UNC_FLAG,
                month = lp_run_op.outputs['month'],
                lag_price_col = opt_prep_op.outputs['lag_price_col'],
                pharm_lag_price_col = opt_prep_op.outputs['pharm_lag_price_col'],
                lp_data_output_df_in = run_solver_op.outputs['lp_data_output_df_out'],
                performance_dict_in = opt_prep_op.outputs['performance_dict_out'],
                act_performance_dict_in = opt_prep_op.outputs['act_performance_dict_out'],
                ytd_perf_pharm_actuals_dict_in = opt_prep_op.outputs['ytd_perf_pharm_actuals_dict_out'],
                client_list_in = opt_prep_op.outputs['client_list_out'],
                client_guarantees_in = opt_prep_op.outputs['client_guarantees_out'],
                pharmacy_guarantees_in = opt_prep_op.outputs['pharmacy_guarantees_out'],
                oc_eoy_pharm_perf_in = opt_prep_op.outputs['oc_eoy_pharm_perf_out'],
                gen_launch_eoy_dict_in = opt_prep_op.outputs['gen_launch_eoy_dict_out'],
                #pharmacy_approx_in = opt_prep_op.outputs['pharmacy_approx_out'],
                eoy_days_in = opt_prep_op.outputs['eoy_days_out'],
                perf_dict_col_in = opt_prep_op.outputs['perf_dict_col_out'],
                mac_list_df_in = opt_prep_op.outputs['mac_list_df_out'],
                lp_vol_mv_agg_df_actual_in = opt_prep_op.outputs['lp_vol_mv_agg_df_actual_out'],
                oc_pharm_dummy_in = opt_prep_op.outputs['oc_pharm_dummy_out'],
                dummy_perf_dict_in = opt_prep_op.outputs['dummy_perf_dict_out'],
                #pharmacy_approx_dummy_in = opt_prep_op.outputs['pharmacy_approx_dummy_out'],
                #pilot_output_columns_in = run_solver_op.outputs['pilot_output_columns_out'],
                generic_launch_df_in = opt_prep_op.outputs['generic_launch_df_out'],
                pref_pharm_list_in = opt_prep_op.outputs['pref_pharm_list_out'],
                breakout_df_in = opt_prep_op.outputs['breakout_df_out'],
                oc_pharm_surplus_in = opt_prep_op.outputs['oc_pharm_surplus_out'],
                proj_days_in = opt_prep_op.outputs['proj_days_out'],
                lambda_output_df_in = run_solver_op.outputs['lambda_output_df_out'],
                chain_region_mac_mapping_in = opt_prep_op.outputs['chain_region_mac_mapping_out'],
                total_output_columns_in = run_solver_op.outputs['total_output_columns_out'],
                brand_surplus_ytd_in = opt_prep_op.outputs['brand_surplus_ytd_out'],
                brand_surplus_lag_in = opt_prep_op.outputs['brand_surplus_lag_out'],
                brand_surplus_eoy_in = opt_prep_op.outputs['brand_surplus_eoy_out'],
                specialty_surplus_ytd_in = opt_prep_op.outputs['specialty_surplus_ytd_out'],
                specialty_surplus_lag_in = opt_prep_op.outputs['specialty_surplus_lag_out'],
                specialty_surplus_eoy_in = opt_prep_op.outputs['specialty_surplus_eoy_out'],
                disp_fee_surplus_ytd_in = opt_prep_op.outputs['disp_fee_surplus_ytd_out'],
                disp_fee_surplus_lag_in = opt_prep_op.outputs['disp_fee_surplus_lag_out'],
                disp_fee_surplus_eoy_in = opt_prep_op.outputs['disp_fee_surplus_eoy_out']
                #agreement_pharmacy_list_in = opt_prep_op.outputs['agreement_pharmacy_list_out'],
                #non_capped_pharmacy_list_in = opt_prep_op.outputs['non_capped_pharmacy_list_out']
            ).set_display_name(
                'LP Output'
            ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
        lp_output_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
        
        
    with kfp.dsl.Condition(conflict_gpi_run_op.outputs['run_recursive'] == True, name = 'RUN_RECURSIVE'): 
        
        operator_op = optimization_graph(params_file_in, 
                                 LP_RUN, 
                                 month,
                                 conflict_gpi_run_op.outputs['unc_adjust'],
                                 conflict_gpi_run_op.outputs['unc_flag'],
                                 CONFLICT_GPI_PTH).after(lp_output_op)


# #### Graph component wrapper for QA components
#
# `<qa_graph>` is a wrapper that contains all of the QA.py components so that the reporting script does not run if any QA components fail.

@kfp.dsl.graph_component
def qa_graph(params_file_in: str):
    qa_Pharmacy_Output_op = qa_Pharmacy_Output_comp(
        params_in = params_file_in
    ).set_display_name(
        'QA file MedD_LP_Algorithm_Pharmacy_Output_Month'
    ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
    qa_Pharmacy_Output_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    qa_Price_Check_Output_op = qa_Price_Check_Output_comp(
        params_in = params_file_in
    ).set_display_name(
        'QA file Price_Check_Output'
    ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
    qa_Price_Check_Output_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    qa_price_output_op = qa_price_output_comp(
        params_in = params_file_in
    ).set_display_name(
        'QA Test Price Outputs'
    ).set_memory_request('200M').set_cpu_request('2000m').set_timeout(2000).set_retry(num_retries = 5)
    qa_price_output_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    qa_price_tiering_rules_REPORT_op = qa_price_tiering_rules_REPORT_comp(  
        params_in = params_file_in,
        lp_data_output_df_in = qa_price_output_op.outputs['lp_data_output_df_out'],
    ).set_display_name(
        'QA price_tiering_rules_REPORT'
    ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
    qa_price_tiering_rules_REPORT_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    qa_Prices_above_MAC1026_floor_op = qa_Prices_above_MAC1026_floor_comp(
        params_in = params_file_in,
        lp_data_output_df_in = qa_price_output_op.outputs['lp_data_output_df_out'],
        lp_with_final_prices_in = qa_price_output_op.outputs['lp_with_final_prices_out'],
        output_cols_in = qa_price_output_op.outputs['output_cols_out']
    ).set_display_name(
        'QA Prices_above_MAC1026_floor'
    ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
    qa_Prices_above_MAC1026_floor_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    qa_pref_nonpref_pharm_pricing_op = qa_pref_nonpref_pharm_pricing_comp(
        params_in = params_file_in,
        lp_data_output_df_in = qa_price_output_op.outputs['lp_data_output_df_out'],
        lp_with_final_prices_in = qa_price_output_op.outputs['lp_with_final_prices_out'],
        output_cols_in = qa_price_output_op.outputs['output_cols_out']
    ).set_display_name(
        'QA Pref/NonPref Pharm Pricing'
    ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
    qa_pref_nonpref_pharm_pricing_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    qa_test_xvcml_op = qa_test_xvcml_comp(
        params_in = params_file_in
    ).set_display_name(
        'XVCML checks'
    ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
    qa_test_xvcml_op.execution_options.caching_strategy.max_cache_staleness = "P0D"

    qa_test_price_changes_file_op = qa_test_price_changes_file_comp(
        params_in = params_file_in
    ).set_display_name(
        'Output file checks'
    ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
    qa_test_price_changes_file_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    qa_goodrx_price_bound_op = qa_goodrx_price_bound_comp(
        params_in = params_file_in
    ).set_display_name(
        'GoodRx Price Bound Checks'
    ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
    qa_goodrx_price_bound_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    qa_r90_as_mail_op = qa_r90_as_mail_comp(
        params_in = params_file_in,
        lp_data_output_df_in = qa_price_output_op.outputs['lp_data_output_df_out']
    ).set_display_name(
        'R90 as mail QA'
    ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
    qa_r90_as_mail_op.execution_options.caching_strategy.max_cache_staleness = "P0D"

    qa_price_overall_reasonableness_op = qa_price_overall_reasonableness_comp(
        params_in = params_file_in,
        lp_with_final_prices_in = qa_price_output_op.outputs['lp_with_final_prices_out']
    ).set_display_name(
        'Price Overall Reasonablenes Checks'
    ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 0)
    qa_price_overall_reasonableness_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
        
    qa_diagnostic_report_op = qa_diagnostic_report_comp(
    params_in = params_file_in
    ).set_display_name(
        'Diagnostic Report QA'
    ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
    qa_diagnostic_report_op.execution_options.caching_strategy.max_cache_staleness = "P0D"


# #### Pipeline

# This `<pbm_opt_pipe>` function defines the relationship between different components in a kubeflow pipeline. It starts by 
# - setting up parameters, <br>
# Includes `<audit_trail_comp_op>` and `<params_op>` operations to track and set up parameters for each run
#
# - runs preprocessing (if neccessary), <br>
# Makes use of the `<preprocess_graph>` to run `Pre_processing.py`, `qa_checks.py`, and `Daily_Input_Read.py` to prepare and quality check data for an optimization run. Note that in a simulaiton run, only the first iteration requires these scripts to run and thus, the `<preprocess_graph>` wrapper would skip them in subsequent iterations.
#
# - build and solve the lp, <br>
# Makes use of the `<optimization_graph>` to run constraint building components in parallel, submit the resulting lp to the solver, and produce outputs. Note that when <UNC_ADJUST = True>, these components are run twice to produce UC adjustments in the first run and utilize them in the second run
#
# - quality checking the results
# Consists of running QA components,
#
# - and producing report dashboards.
# Runs the reporting components to produce Tableau dashboards based on the results.

##main pipeline function
@kfp.dsl.pipeline(
    name = 'PBM Optimization Pipeline',
    description = 'Kubeflow Pipeline for Client-Pharmacy MAC Optimization'
)
def pbm_opt_pipe(
    params_file_in: str, git_branch: str, git_hash: str, algo_version: str, 
    version_type: str, version_iteration: str, odp_lp: str, cmpas: str):
    
    '''Kubeflow Pipeline to run PBM MAC Optimization workflow'''
    
    ##component to track parameter settings for algorithm runs
    audit_trail_comp_op = audit_trail_comp(
        params_file_in = params_file_in, 
        git_branch = git_branch, 
        git_hash = git_hash, 
        algo_version = algo_version,
        version_type = version_type,
        version_iteration = version_iteration,
        odp_lp = odp_lp,
        cmpas = cmpas 
    ).set_display_name(
        'Audit Trail Update'
    ).set_memory_request('10M').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
    
    ##component to set parameters for CPMO run
    params_op = params_comp(
        params_file_in = params_file_in,
        #months_list = repr(months_list)
    ).set_display_name(
        'Parameter Prep'
    ).set_memory_request('10M').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
    params_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    
    ##preprocessing graph component to prepare data for CPMO run
    ##runs Pre_processing.py, qa_checks.py, and Daily_Input_Read.py scripts
    preprocess_op = preprocess_graph(params_file_in, 
                                     params_op.outputs['skip_to_opt'], 
                                     params_op.outputs['month_indices'])
    preprocess_op.after(params_op)
    
    
    ##optimization graph component to preprocess lp_data, build the lp problem, submit the lp for solving, and produce outputs
    optimization_op = optimization_graph(params_file_in, 
                                         params_op.outputs['LP_RUN'], 
                                         params_op.outputs['month_indices'],
                                         params_op.outputs['UNC_ADJUST'], 
                                         params_op.outputs['UNC_FLAG'],
                                         params_op.outputs['CONFLICT_GPI_PTH'])
    optimization_op.after(preprocess_op)
    
    ##Performance Checks
    qa_test_performance_op = qa_test_performance_comp(
        params_in = params_file_in
    ).set_display_name(
        'Client/Pharmacy Performance Checks'
    ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
    qa_test_performance_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    qa_test_performance_op.after(optimization_op)
    

    ##final QA components
    qa_op = qa_graph(params_file_in)
    qa_op.after(optimization_op)
    
    ##reporting components
    rp_create_reporting_tables_op = rp_create_reporting_tables_comp(
        params_in = params_file_in
    ).set_display_name(
        'Reporting tables'
    ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(2000).after(qa_op).set_retry(num_retries = 0)
    rp_create_reporting_tables_op.execution_options.caching_strategy.max_cache_staleness = "P0D"

# #### Pipeline compile
# Compiling the pipeline to produce a `.yaml` file which is then submitted to the kubeflow.

# +
import yaml

##compile the original pipeline defined above
pipe_path = 'kubeflow_optimization_pipeline.yaml'
kfp.compiler.Compiler().compile(pbm_opt_pipe, pipe_path)
# -

# Remove metadata annotation to reduce the size of the YAML file

with open(pipe_path) as f:
    workflow = yaml.safe_load(f)
for template in workflow['spec']['templates']:
    annotations = template.setdefault('metadata', {}).setdefault('annotations', {})
    if 'pipelines.kubeflow.org/component_spec' in annotations:
        del annotations['pipelines.kubeflow.org/component_spec']
with open(pipe_path, "w") as f:
    yaml.dump(workflow, f)
