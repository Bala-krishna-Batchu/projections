import kfp
import os
import importlib
import jinja2 as jj2
import datetime as dt
from kfp.components import InputPath, OutputPath, func_to_container_op
from typing import NamedTuple, List
import GER_LP_Code.ClientPharmacyMacOptimization as opt
import GER_LP_Code.QA as qa
from GER_LP_Code.audit_trail_utils import AuditTrail

repo_head = '/home/jupyter/clientpharmacymacoptimization'
os.chdir(repo_head)
program_dir = os.path.abspath(os.curdir)


import json
import argparse as ap
def get_args():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-k', '--kf-endpoint',
        type=str,
        help='URL for the Kubeflow endpoint where the job will run',
        default='https://3b2403ddc0816a06-dot-us-east1.pipelines.googleusercontent.com'
        #required=True
    )
    parser.add_argument(
        '--git-branch',
        type=str,
        required=True,
        help='Git Branch to use for the run'
    )
    parser.add_argument(
        '--git-hash',
        type=str,
        required=True,
        help='First few characters of the git hash for the commit version'
    )
    
    parser.add_argument(
        '--version-type',
        type=str,
        required=True,
        help='Work In Progress (WIP) or Production (PROD)',
        #default='WIP'
    )
    parser.add_argument(
        '--version-iteration',
        type=str,
        required=True,
        help='Which iteration, some digit used to name the docker image',
        #default='0'
    )
    parser.add_argument(
        '--project-name',
        type=str,
        #required=True,
        help='name of the gcp project',
        default='pbm-mac-lp-prod-ai'
    )
    
    parser.add_argument(
        '--algo-version',
        type=str,
        help='Which algorithm (LP, SF, MEDD)',
        default='LP'
    )
    
    parser.add_argument(
        '--pipe-yaml',
        help='Path to YAML file for pipeline.',
        default='./kubeflow_optimization_pipeline.yaml'
    )
    parser.add_argument(
        '--gcp-registry-name',
        type=str, 
        help='address name of image registry',
        default='us.gcr.io'
    )
    parser.add_argument(
        '--base-name', 
        type=str, 
        default='pbm_base',
        help='where base images are stored'
    )
    parser.add_argument(
        '--script-run-name', 
        type=str, 
        default='script_run_name',
        help='where script run images are stored'
    )
    parser.add_argument(
        '--opt-name', 
        type=str, 
        default='pbm_opt',
        help='where script opt images are stored'
    )
    
    args = parser.parse_args()
    
    return args

# get args
args = get_args()
if not args.git_branch or not args.git_hash:
    raise Exception('Git branch must be supplied using --git-branch.')

#define version and tag names to be used below
VERSION = f"{args.git_branch}-{args.git_hash}-{args.version_type}-{args.version_iteration}"
BASE_TAG = f"{args.gcp_registry_name}/{args.project_name}/{args.base_name}:{VERSION}"
SCRIPT_RUN_TAG = f"{args.gcp_registry_name}/{args.project_name}/{args.script_run_name}:{VERSION}"
OPT_TAG = f"{args.gcp_registry_name}/{args.project_name}/{args.opt_name}:{VERSION}"
    

#======================Audit trail component update=================
#===================================================================
def update_audit_trail(
    params_file_in: str, 
    git_branch: str, 
    git_hash: str, 
    algo_version: str,
    version_type: str,
    version_iteration: str):
    
    import audit_trail_utils as audit
    audit_obj = audit.AuditTrail(
        git_branch = git_branch,
        git_hash = git_hash,
        algo_version = algo_version,
        version_type = version_type,
        version_iteration = version_iteration
    )
    audit_obj.update_audit_trail(params_file_in)


audit_trail_comp = func_to_container_op(
    func=update_audit_trail,
    base_image=BASE_TAG
)

#===================== Parameter Components =====================
#================================================================
def prepare_params(
    params_file_in: str
) -> NamedTuple('Outputs', [('LP_RUN', List[int])]):
    '''User input parameter check and prep, 
    copies previously save parameter from storage bucket to code folder, then imports. '''
    import re
    from google.cloud import storage
    
    # Download parameters file from storage
    local_file_name = 'CPMO_parameters.py'
    client = storage.Client()
    bp_blob = params_file_in[5:].split('/') #ignore gs:// at the beginning
    b = bp_blob[0]    
    blob = '/'.join(bp_blob[1:])
    bucket = client.get_bucket(b)
    blob = bucket.get_blob(blob)
    assert blob, f'FileNotFound: Could not find parameters file: {params_file_in}'
    blob.download_to_filename(local_file_name)
    
    import CPMO_parameters as pp
    # get month indices to iterate over
    m_indices = list(range(len(pp.LP_RUN)))

    return (pp.LP_RUN, m_indices)

params_comp = func_to_container_op(prepare_params, base_image=BASE_TAG)


#========================== LP Run Component ===========================
#=======================================================================
def lp_run(
    month_index: int, lp_run: List[int]
) -> NamedTuple('Output', [('in_lp_run', bool), ('month', int)]):
    '''Auxiliary func to determine if month is in p.LP_RUN.
    If not, the output is used to skip other steps and run opt.no_lp_run.'''
    month = eval(lp_run)[month_index]
    run = month in eval(lp_run)
    return (run, month) 

lp_run_comp = func_to_container_op(lp_run, base_image='python:3.8-slim-buster')


#========================= Script Run Component ==========================
#=========================================================================
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


#======================== Optimization Components ==========================
#===========================================================================
opt_prep_comp = func_to_container_op(
    func=opt.opt_preprocessing,
    base_image=OPT_TAG,
)
# Constraint Components
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
leakage_optimization_constraints_comp = func_to_container_op(
    func=opt.leakage_opt,
    base_image=OPT_TAG
)
# LP Solver
run_solver_comp = func_to_container_op(
    func=opt.run_solver,
    base_image=OPT_TAG
)
# no LP run condition
no_lp_run_comp = func_to_container_op(
    func=opt.no_lp_run,
    base_image=BASE_TAG
)
# LP output component
lp_output_comp = func_to_container_op(
    func=opt.lp_output,
    base_image=OPT_TAG
)

#========================== QA Components =================================
#==========================================================================
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
#=========================== Pipeline ======================================
#===========================================================================
# pipeline function
# pipeline function
@kfp.dsl.pipeline(
    name='PBM Optimization Pipeline',
    description='Kubeflow Pipeline for Client-Pharmacy MAC Optimization'
)
def pbm_opt_pipe(
    params_file_in: str, git_branch: str, git_hash: str, algo_version: str, 
    version_type: str, version_iteration: str):
    
    '''Kubeflow Pipeline to run PBM MAC Optimization workflow'''
    audit_trail_comp_op = audit_trail_comp(
        params_file_in = params_file_in, 
        git_branch = git_branch, 
        git_hash = git_hash, 
        algo_version = algo_version,
        version_type = version_type,
        version_iteration = version_iteration
    ).set_display_name(
        'Audit Trail Update'
    ).set_memory_request('10M').set_cpu_request('1000m').set_timeout(1000)
    # read in parameters at runtime (as string)
    params_op = params_comp(
        params_file_in=params_file_in,
#         months_list=repr(months_list)
    ).set_display_name(
        'Parameter Prep'
    ).set_memory_request('10M').set_cpu_request('1000m').set_timeout(1000)
    params_op.execution_options.caching_strategy.max_cache_staleness = "P0D"   
    # prep scripts
    # Preprocessing
    prep_op = script_run_comp(
        script_name='Pre_Processing.py',
#         local_output_dir='Input/Data Automation',
        params_file_in=params_file_in
    ).set_display_name(
        'Preprocessing'
    ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(1000)
    prep_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    # QA_INPUT_DATA.py script run
#     input_qa_op = script_run_comp(
#         script_name='QA_INPUT_DATA.py',
#         local_output_dir='Input',
#         params_file_in=params_file_in
#     ).set_display_name(
#         'QA Input'
#     ).set_memory_request('400M').set_cpu_request('1000m').set_timeout(1000)
#     input_qa_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
#     input_qa_op.after(prep_op)
    qa_checks_op = script_run_comp(
        script_name='qa_checks.py',
        params_file_in=params_file_in
    ).set_display_name(
        'QA Checks'
    ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(1000)
    qa_checks_op.after(prep_op)
    qa_checks_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    # Daily_Input_Read.py script run
    daily_input_read_op = script_run_comp(
        script_name='Daily_Input_Read.py',
#         local_output_dir='Dynamic_Input',
        params_file_in=params_file_in
    ).set_display_name(
        'Daily Input Read'
    ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(1000)
    daily_input_read_op.after(prep_op)
    daily_input_read_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    # optimization ops
#     for m, month in enumerate(months_list):       
    opt_prep_op = opt_prep_comp(
        m=m, params_file_in=params_file_in
    ).set_display_name(
        'Opt Preprocessing'
    ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(1000)
    opt_prep_op.after(daily_input_read_op)
    opt_prep_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    # determine if LP_RUN is true
    lp_run_op = lp_run_comp(
        month_index=m, 
        lp_run=params_op.outputs['LP_RUN']
    ).set_display_name(
        'LP Run Check Support Component'
    ).set_memory_request('10M').set_cpu_request('10m').set_timeout(1000)
    lp_run_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    with kfp.dsl.Condition(lp_run_op.outputs['in_lp_run'] == True, name = 'm_in_LP_RUN'):
        # constraints
        # Consistent Strength Pricing constraints
        pricing_constraints_op = opt_pricing_constraints_comp(
            params_file_in=params_file_in,
            lp_data_df_in=opt_prep_op.outputs['lp_data_df_out'],
            price_lambdas_in=opt_prep_op.outputs['price_lambdas_out'],
        ).set_display_name(
            'Consistent Strength Pricing Constraints'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(1000)
        pricing_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
        # Client Level Constraints
        client_constraints_op = opt_client_constraints_comp(
            params_file_in=params_file_in,
            lp_data_df_in=opt_prep_op.outputs['lp_data_df_out'],
            client_guarantees_in=opt_prep_op.outputs['client_guarantees_out'],
            pharmacy_guarantees_in=opt_prep_op.outputs['pharmacy_guarantees_out'],
            performance_dict_in=opt_prep_op.outputs['performance_dict_out'],
            breakout_df_in=opt_prep_op.outputs['breakout_df_out'],
            client_list_in=opt_prep_op.outputs['client_list_out'],
            pharmacy_approx_in=opt_prep_op.outputs['pharmacy_approx_out'],
            oc_eoy_pharm_perf_in=opt_prep_op.outputs['oc_eoy_pharm_perf_out'],
            gen_launch_eoy_dict_in=opt_prep_op.outputs['gen_launch_eoy_dict_out'],
            price_lambdas_in=opt_prep_op.outputs['price_lambdas_out'],
#            total_pharm_list_in=opt_prep_op.outputs['total_pharm_list_out'],
#            agreement_pharmacy_list_in=opt_prep_op.outputs['agreement_pharmacy_list_out'],
        ).set_display_name(
            'Client Level Constraints'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(1000)
        client_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
        # Preferred LT Non-Preferred Pricing Constraints
        pprice_lt_npprice_constraints_op = pprice_lt_npprice_constraints_comp(
            params_file_in=params_file_in,
            lp_data_df_in=opt_prep_op.outputs['lp_data_df_out'],
            pref_pharm_list_in=opt_prep_op.outputs['pref_pharm_list_out'],
#            total_pharm_list_in=opt_prep_op.outputs['total_pharm_list_out']
        ).set_display_name(
            'Preferred LT Non-Preferred Pricing Constraints'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(1000)
        pprice_lt_npprice_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
        # Measure Specific Pricing (ie M < R90 < R30) 
        specific_pricing_constraints_op = specific_pricing_constraints_comp(
            params_file_in=params_file_in,
            lp_data_df_in=opt_prep_op.outputs['lp_data_df_out'],
#            total_pharm_list_in=opt_prep_op.outputs['total_pharm_list_out']
        ).set_display_name(
            'Measure Specific Pricing Constraints'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(1000)
        specific_pricing_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
        # All other pharm pricing greater than CVS
        cvs_parity_price_constraint_op = cvs_parity_price_constraint_comp(
            params_file_in=params_file_in,
            lp_data_df_in=opt_prep_op.outputs['lp_data_df_out'],
            pref_pharm_list_in=opt_prep_op.outputs['pref_pharm_list_out']                
        ).set_display_name(
            'CVS Parity Constraint'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(1000)
        cvs_parity_price_constraint_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
        # CVS prices should <= IND prices on the CVSSP VCML
        state_parity_constraint_op = state_parity_constraint_comp(
            params_file_in = params_file_in,
            lp_data_df_in = opt_prep_op.outputs['lp_data_df_out']
        ).set_display_name(
            'State Parity Constraint'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(1000)
        state_parity_constraint_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
        # Consistent MAC constraints
        mac_constraints_op = mac_constraints_comp(
            params_file_in=params_file_in,
            lp_data_df_in=opt_prep_op.outputs['lp_data_df_out']
        ).set_display_name(
            'Consistent MAC constraints'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(1000)
        mac_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
        # Aggregate MAC Price Change Constraints
        agg_mac_price_change_constraints_op = agg_mac_price_change_constraints_comp(
            params_file_in=params_file_in,
            month=lp_run_op.outputs['month'],
            lp_data_df_in=opt_prep_op.outputs['lp_data_df_out']                
        ).set_display_name(
            'Aggregate MAC Price Change Constraints'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(1000)
        agg_mac_price_change_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
        # Equal Package Size Contraints
        equal_package_size_constraints_op = equal_package_size_constraints_comp(
            params_file_in=params_file_in,
            lp_data_df_in=opt_prep_op.outputs['lp_data_df_out']
        ).set_display_name(
            'Equal Package Size Contraints'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(1000)
        equal_package_size_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
        # Same Difference Package Size Constraints
        same_difference_package_size_constraints_op = same_difference_package_size_constraints_comp(
            params_file_in=params_file_in,
            lp_data_df_in=opt_prep_op.outputs['lp_data_df_out']            
        ).set_display_name(
            'Same Difference Package Size Constraints'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(1000)
        same_difference_package_size_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
        # Leakage Optimization Constraints
        leakage_optimization_constraints_op = leakage_optimization_constraints_comp(
            params_file_in=params_file_in,
            lp_data_df_in = opt_prep_op.outputs['lp_data_df_out']            
        ).set_display_name(
            'Leakage Optimization Constraints'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(2000).set_retry(num_retries = 3)
        leakage_optimization_constraints_op.execution_options.caching_strategy.max_cache_staleness = "P0D"    
        # linear solver
        run_solver_op = run_solver_comp(
            month=lp_run_op.outputs['month'],
            params_file_in=params_file_in,
            t_cost_in=pricing_constraints_op.outputs['t_cost_out'],
            cons_strength_cons_in=pricing_constraints_op.outputs['cons_strength_cons_out'],
            client_constraint_list_in=client_constraints_op.outputs['client_constraint_list_out'],
            client_constraint_target_in=client_constraints_op.outputs['client_constraint_target_out'],
            pref_lt_non_pref_cons_list_in=pprice_lt_npprice_constraints_op.outputs['pref_lt_non_pref_cons_list_out'],
            meas_specific_price_cons_list_in=specific_pricing_constraints_op.outputs['meas_specific_price_cons_list_out'],
            pref_other_price_cons_list_in=cvs_parity_price_constraint_op.outputs['pref_other_price_cons_list_out'],
            parity_price_cons_list_in = state_parity_constraint_op.outputs['parity_price_cons_list_out'],
            mac_cons_list_in=mac_constraints_op.outputs['mac_cons_list_out'],
            agg_mac_cons_list_in=agg_mac_price_change_constraints_op.outputs['agg_mac_cons_list_out'],
            eq_pkg_sz_cons_list_in=equal_package_size_constraints_op.outputs['eq_pkg_sz_cons_list_out'],
            sm_diff_pkg_sz_cons_list_in=same_difference_package_size_constraints_op.outputs['sm_diff_pkg_sz_cons_list_out'],
            leakage_cost_list_in = leakage_optimization_constraints_op.outputs['leakage_cost_list_out'],
            leakage_const_list_in = leakage_optimization_constraints_op.outputs['leakage_const_list_out'],
            lambda_df_in=client_constraints_op.outputs['lambda_df_out'],
            lp_vol_mv_agg_df_in=opt_prep_op.outputs['lp_vol_mv_agg_df_out'],
            breakout_df_in=opt_prep_op.outputs['breakout_df_out'],
#            total_pharm_list_in=opt_prep_op.outputs['total_pharm_list_out'],
            lp_data_df_in=pricing_constraints_op.outputs['lp_data_df_out']
        ).set_display_name(
            'Run LP Solver'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(1000)
        run_solver_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
        # final outputs
        lp_output_op = lp_output_comp(
            params_file_in=params_file_in,
#            m=m,
            month=lp_run_op.outputs['month'],
            lag_price_col=opt_prep_op.outputs['lag_price_col'],
            lp_data_output_df_in=run_solver_op.outputs['lp_data_output_df_out'],
            performance_dict_in=opt_prep_op.outputs['performance_dict_out'],
            act_performance_dict_in=opt_prep_op.outputs['act_performance_dict_out'],
            ytd_perf_pharm_actuals_dict_in=opt_prep_op.outputs['ytd_perf_pharm_actuals_dict_out'],
            client_list_in=opt_prep_op.outputs['client_list_out'],
            client_guarantees_in=opt_prep_op.outputs['client_guarantees_out'],
            pharmacy_guarantees_in=opt_prep_op.outputs['pharmacy_guarantees_out'],
            oc_eoy_pharm_perf_in=opt_prep_op.outputs['oc_eoy_pharm_perf_out'],
            gen_launch_eoy_dict_in=opt_prep_op.outputs['gen_launch_eoy_dict_out'],
            pharmacy_approx_in=opt_prep_op.outputs['pharmacy_approx_out'],
            eoy_days_in=opt_prep_op.outputs['eoy_days_out'],
            perf_dict_col_in=opt_prep_op.outputs['perf_dict_col_out'],
            mac_list_df_in=opt_prep_op.outputs['mac_list_df_out'],
            lp_vol_mv_agg_df_nounc_in=opt_prep_op.outputs['lp_vol_mv_agg_df_nounc_out'],
            oc_pharm_dummy_in=opt_prep_op.outputs['oc_pharm_dummy_out'],
            gen_launch_dummy_in=opt_prep_op.outputs['gen_launch_dummy_out'],
            pharmacy_approx_dummy_in=opt_prep_op.outputs['pharmacy_approx_dummy_out'],
            oc_next_run_pharm_perf_in=opt_prep_op.outputs['oc_next_run_pharm_perf_out'],
#            gen_launch_next_run_dict_in=opt_prep_op.outputs['gen_launch_next_run_dict_out'],
#            pilot_output_columns_in=run_solver_op.outputs['pilot_output_columns_out'],
            generic_launch_df_in=opt_prep_op.outputs['generic_launch_df_out'],
            pref_pharm_list_in=opt_prep_op.outputs['pref_pharm_list_out'],
            breakout_df_in=opt_prep_op.outputs['breakout_df_out'],
            oc_pharm_surplus_in=opt_prep_op.outputs['oc_pharm_surplus_out'],
            proj_days_in=opt_prep_op.outputs['proj_days_out'],
            lambda_output_df_in=run_solver_op.outputs['lambda_output_df_out'],
            chain_region_mac_mapping_in=opt_prep_op.outputs['chain_region_mac_mapping_out'],
            total_output_columns_in=run_solver_op.outputs['total_output_columns_out']
#            agreement_pharmacy_list_in=opt_prep_op.outputs['agreement_pharmacy_list_out'],
#            non_capped_pharmacy_list_in=opt_prep_op.outputs['non_capped_pharmacy_list_out']
        ).set_display_name(
            'LP Output'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(1000)
        lp_output_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    with kfp.dsl.Condition(lp_run_op.outputs['in_lp_run'] == False, name='not_in_LP_RUN'):
        # If not running LP this month
        no_lp_run_op = no_lp_run_comp(
#        params_file_in=params_file_in,
            lp_vol_mv_agg_df_in=opt_prep_op.outputs['lp_vol_mv_agg_df_out']
        ).set_display_name(
            'No LP Run'
        ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(1000)
        no_lp_run_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
        # final outputs
        lp_output_op = lp_output_comp(
            params_file_in=params_file_in,
#            m=m,
            month=lp_run_op.outputs['month'],
            lag_price_col=opt_prep_op.outputs['lag_price_col'],
            lp_data_output_df_in=no_lp_run_op.outputs['lp_data_output_df_out'],
            performance_dict_in=opt_prep_op.outputs['performance_dict_out'],
            act_performance_dict_in=opt_prep_op.outputs['act_performance_dict_out'],
            ytd_perf_pharm_actuals_dict_in=opt_prep_op.outputs['ytd_perf_pharm_actuals_dict_out'],
            client_list_in=opt_prep_op.outputs['client_list_out'],
            client_guarantees_in=opt_prep_op.outputs['client_guarantees_out'],
            pharmacy_guarantees_in=opt_prep_op.outputs['pharmacy_guarantees_out'],
            oc_eoy_pharm_perf_in=opt_prep_op.outputs['oc_eoy_pharm_perf_out'],
            gen_launch_eoy_dict_in=opt_prep_op.outputs['gen_launch_eoy_dict_out'],
            pharmacy_approx_in=opt_prep_op.outputs['pharmacy_approx_out'],
            eoy_days_in=opt_prep_op.outputs['eoy_days_out'],
            perf_dict_col_in=opt_prep_op.outputs['perf_dict_col_out'],
            mac_list_df_in=opt_prep_op.outputs['mac_list_df_out'],
            lp_vol_mv_agg_df_nounc_in=opt_prep_op.outputs['lp_vol_mv_agg_df_nounc_out'],
            oc_pharm_dummy_in=opt_prep_op.outputs['oc_pharm_dummy_out'],
            gen_launch_dummy_in=opt_prep_op.outputs['gen_launch_dummy_out'],
            pharmacy_approx_dummy_in=opt_prep_op.outputs['pharmacy_approx_dummy_out'],
            oc_next_run_pharm_perf_in=opt_prep_op.outputs['oc_next_run_pharm_perf_out'],
#            gen_launch_next_run_dict_in=opt_prep_op.outputs['gen_launch_next_run_dict_out'],
#            pilot_output_columns_in=run_solver_op.outputs['pilot_output_columns_out'],
            generic_launch_df_in=opt_prep_op.outputs['generic_launch_df_out'],
            pref_pharm_list_in=opt_prep_op.outputs['pref_pharm_list_out'],
            breakout_df_in=opt_prep_op.outputs['breakout_df_out'],
            oc_pharm_surplus_in=opt_prep_op.outputs['oc_pharm_surplus_out'],
            proj_days_in=opt_prep_op.outputs['proj_days_out'],
            lambda_output_df_in=run_solver_op.outputs['lambda_output_df_out'],
            chain_region_mac_mapping_in=opt_prep_op.outputs['chain_region_mac_mapping_out'],
            total_output_columns_in=run_solver_op.outputs['total_output_columns_out']
#            agreement_pharmacy_list_in=opt_prep_op.outputs['agreement_pharmacy_list_out'],
#            non_capped_pharmacy_list_in=opt_prep_op.outputs['non_capped_pharmacy_list_out']
        ).set_display_name(
            'LP Output'
        ).set_memory_request('1G').set_cpu_request('1000m').set_timeout(1000)
        lp_output_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    # QA Components
    qa_Pharmacy_Output_op = qa_Pharmacy_Output_comp(
        params_in=params_file_in
    ).set_display_name(
        'QA file MedD_LP_Algorithm_Pharmacy_Output_Month'
    ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(1000).after(lp_output_op)
    qa_Pharmacy_Output_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    qa_Price_Check_Output_op = qa_Price_Check_Output_comp(
        params_in=params_file_in
    ).set_display_name(
        'QA file Price_Check_Output'
    ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(1000).after(lp_output_op)
    qa_Price_Check_Output_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    qa_price_output_op = qa_price_output_comp(
        params_in=params_file_in
    ).set_display_name(
        'QA Test Price Outputs'
    ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(1000).after(lp_output_op)
    qa_price_output_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    qa_price_tiering_rules_REPORT_op = qa_price_tiering_rules_REPORT_comp(  
        params_in=params_file_in,
        lp_data_output_df_in=qa_price_output_op.outputs['lp_data_output_df_out'],
    ).set_display_name(
        'QA price_tiering_rules_REPORT'
    ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(1000).after(lp_output_op)
    qa_price_tiering_rules_REPORT_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    qa_Prices_above_MAC1026_floor_op = qa_Prices_above_MAC1026_floor_comp(
        params_in=params_file_in,
        lp_data_output_df_in=qa_price_output_op.outputs['lp_data_output_df_out'],
        lp_with_final_prices_in=qa_price_output_op.outputs['lp_with_final_prices_out'],
        output_cols_in=qa_price_output_op.outputs['output_cols_out']
    ).set_display_name(
        'QA Prices_above_MAC1026_floor'
    ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(1000)
    qa_Prices_above_MAC1026_floor_op.execution_options.caching_strategy.max_cache_staleness = "P0D"
    qa_pref_nonpref_pharm_pricing_op = qa_pref_nonpref_pharm_pricing_comp(
        params_in=params_file_in,
        lp_data_output_df_in=qa_price_output_op.outputs['lp_data_output_df_out'],
        lp_with_final_prices_in=qa_price_output_op.outputs['lp_with_final_prices_out'],
        output_cols_in=qa_price_output_op.outputs['output_cols_out']
    ).set_display_name(
        'QA Pref/NonPref Pharm Pricing'
    ).set_memory_request('200M').set_cpu_request('1000m').set_timeout(1000)
    qa_pref_nonpref_pharm_pricing_op.execution_options.caching_strategy.max_cache_staleness = "P0D"



if __name__ == '__main__':
    

    # compile pipeline
    pipe_path = args.pipe_yaml
    kfp.compiler.Compiler().compile(pbm_opt_pipe, pipe_path)
    print(f"\n===== Created yaml file name: {pipe_path}")
