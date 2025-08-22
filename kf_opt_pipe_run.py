'''Run the kubeflow pipeline for price optimization on GCP'''
# Get CLI arguments from user
#kubeflow_run_name = f"RunID-{p.AT_RUN_ID}-{p.CUSTOMER_ID[0]}-{p.USER}"
def get_args():
    parser = ap.ArgumentParser()
    parser.add_argument(
        '-k', '--kf-endpoint',
        type=str,
        help='URL for the Kubeflow endpoint where the job will run',
        required=True
    )
    parser.add_argument(
        '-p', '--params',
        help=('Paths to parameters files (i.e. CPMO_parameters.py). '
              'Supply parameters file paths, one per client, space delimited.'),
        required=False,
        default=[],
        nargs='*'
    )
    parser.add_argument(
        '-j', '--json',
        help='Path to JSON file with parameters for each pipeline run.',
        type=str,
        required=False,
        default=None
    )
    parser.add_argument(
        '--pipe-yaml',
        help='Path to YAML file for pipeline.',
        default='./kubeflow_optimization_pipeline.yaml'
    )
    parser.add_argument(
        '--params-template',
        help='Path to parameter template. Used to create parameter files from --json supplied objects',
        default='./GER_LP_Code/CPMO_parameters_TEMPLATE.py'
    )
    parser.add_argument(
        '--run-names', 
        type=str, 
        nargs='*',
        default=[],
        help='Kubeflow run names (must be unique if supplied)'
    )
    parser.add_argument(
        '--run-name-prefix',
        type=str,
        help='Prefix string to append to run name'
    )
    parser.add_argument(
        '-u', '--add-host-as-user',
        help='If true, overwrites the USER parameter with the local system HOSTNAME environment variable. (Set to False if this is not wanted)',
        choices=['True', 'true', 't', '1', 'False', 'false', 'f', '0'],
        default='True'
    )
#     parser.add_argument(
#         '-t', '--tstamp-now',
#         help='Set timestamp parameter to current time',
#         action='store_true'
#     )
    parser.add_argument(
        '--git-branch',
        type=str,
        help='Git Branch to use for the run'
    )
    parser.add_argument(
        '--git-hash',
        type=str,
        help='First few characters of the git hash for the commit version'
    )
    parser.add_argument(
        '--algo-version',
        type=str,
        help='Which algorithm (LP, SF, MEDD)',
        default='LP'
    )
    
    parser.add_argument(
        '--version-type',
        type=str,
        help='Work In Progress (WIP) or Production (PROD)',
        default='WIP'
    )
    parser.add_argument(
        '--version-iteration',
        type=str,
        help='Which iteration, some digit used to name the docker image',
        default='0'
    )
    parser.add_argument(
        '--ODP_Event_Key',
        type=str,
        help='Event key to identify the on demand pricing runs',
        default=''
    )
    parser.add_argument(
        '--schd_mode',
        type=str,
        help='Mode to setup the runs using scheduler',
        default=''
    )
    parser.add_argument(
        '--experiment-name', 
        type=str, 
        help='Kubeflow experiment name'
    )
    parser.add_argument(
        '--namespace', 
        type=str, 
        help='Kubeflow namespace'
    )
    parser.add_argument(
        '--cmpas',
        type=str,
        help='CMPAS Run',
        default=''
    )
    args = parser.parse_args()
    return args

def _add_host_as_user(p):
    if p.lower() in ['True', 'true', 't', '1', True]:
        return True
    if p.lower() in ['False', 'false', 'f', '0', False, '']:
        return False

def get_audit_run_id(
    git_branch,
    git_hash,
    algo_version,
    version_type,
    version_iteration,
    odp_event_key,
    schd_mode,
    cmpas,
    uclclient):
    
    if args.ODP_Event_Key != '' or args.schd_mode != '' :
        import audit_trail_utils as audit
    else:
        import GER_LP_Code.audit_trail_utils as audit
    
    audit_obj = audit.AuditTrail(
        git_branch = git_branch,
        git_hash = git_hash,
        algo_version = algo_version,
        version_type = version_type,
        version_iteration = version_iteration,
        odp_lp = odp_event_key,
        cmpas = cmpas,
        uclclient = uclclient
    )
    AT_RUN_ID = audit_obj.get_latest_run_id(
        project_name = 'pbm-mac-lp-prod-ai',
        dataset_name = 'pricing_management',
        table_name = 'AT_Run_ID'
    )
    return AT_RUN_ID

def _run(
    kf_endpoint,
    pipeline_file,
    git_branch,
    git_hash,
    odp_event_key,
    schd_mode,
    algo_version='LP',
    version_type='WIP',
    version_iteration='0',
    params_files=[],
    json_params=[],
    run_names=[],
    prefix='',
    experiment_name=None,
    namespace=None,
    params_template='GER_LP_Code/CPMO_parameters_TEMPLATE.py',
    tstamp=True,
    add_host_as_user=True,
    cmpas='False'
):
    import os
    import socket
    import jinja2 as jj2
    import datetime as dt

    if args.ODP_Event_Key != '' or args.schd_mode != '' :
        from util_funcs import upload_blob
    else:
        from GER_LP_Code.util_funcs import upload_blob
    
    #argument will be passed to pipeline
    arguments = {
        'params_file_in': '', #this must be supplied below, should be path to params in cloud storage
        'git_branch': git_branch, 
        'git_hash': git_hash, 
        'algo_version': algo_version,
        'version_type': version_type,
        'version_iteration': version_iteration,
        'odp_lp': odp_event_key,
        'cmpas': cmpas
    }
    client = kfp.Client(host=kf_endpoint)
    outpts = []
    if run_names:
        assert (len(run_names) == len(params_files) + len(json_params)), \
            'If run names are supplied, there must be one for each parameter file/object.'
    # Run pipe for each parameter dict in list
    for i, params in enumerate(json_params):
        uclclient = params.get('TRUECOST_CLIENT', False) | params.get('UCL_CLIENT', False)
        
        #following two lines added for AUDIT Trail
        AT_RUN_ID = get_audit_run_id(git_branch, git_hash, algo_version, version_type, version_iteration, odp_event_key, schd_mode, cmpas, uclclient)
        print(f"RunID: {AT_RUN_ID}")
        params['AT_RUN_ID'] = AT_RUN_ID
        
        # get parameters
        template = jj2.Template(open(params_template).read())
        tt = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S%f")
        if tstamp:
            params['TIMESTAMP'] = '"' + tt + '"'
        if add_host_as_user:
            hname = socket.gethostname()
#             hname = sb.run('hostname', capture_output=True).stdout.strip().decode('utf-8')
            params['USER'] = '"' + hname + '"'
        params_string = template.render(**params)
        # write parameters locally
        pfname = f'CPMO_parameters_{tt}.py'
        with open(pfname, 'w') as pf:
            pf.write(params_string)
        # move to cloud for access
        cloud_params_file = os.path.join(params['PROGRAM_OUTPUT_PATH'], pfname)
        arguments['params_file_in'] = cloud_params_file  #IMPORTANT 
        assert 'gs://' in cloud_params_file, 'The PROGRAM_OUTPUT_PATH in parameters must be a GCP bucket path.'
        bucket, blob = cloud_params_file.replace('gs://', '').split('/', 1)
        upload_blob(bucket, pfname, blob)
        # remove local temp params
        os.remove(pfname)
        if run_names:
            rname = prefix + run_names[i] if prefix else run_names[i]
        else:
            #rname = 'run_' + dt.datetime.now().strftime('%Y-%m-%d_%H%M%S_') + str(i+1)
            rname = 'run_' + AT_RUN_ID
            rname = prefix + rname if prefix else rname
        rname += f"_{params['CUSTOMER_ID'][2:-2]}"
        prun = client.create_run_from_pipeline_package(
            pipeline_file=pipeline_file,
            arguments=arguments,
            run_name=rname,
            experiment_name=experiment_name,
            namespace=namespace
        )
        outpts.append((rname, prun, f'Parameter tstamp: {tt}'))
    # Run pipe for each parameter file in path list
    assert all([os.path.exists(ppath) for ppath in params_files]), \
        'Cannot locate one or more supplied parameter files. Check paths.'
    
    for j, ppath in enumerate(params_files):
        idx = len(json_params)+j  # run name index for param file
        if run_names:
            rname = prefix + run_names[idx] if prefix else run_names[i]
        else:
            rname = 'run_' + dt.datetime.now().strftime('%Y-%m-%d_%H%M%S_') + str(idx+1)
            rname = prefix + rname if prefix else rname
        # copy params to cloud for access
        cloud_params_file = os.path.join(params['PROGRAM_OUTPUT_PATH'], os.path.basename(ppath))
        arguments['params_file_in'] = cloud_params_file  #IMPORTANT 
        assert 'gs://' in cloud_params_file, 'The PROGRAM_OUTPUT_PATH in parameters must be a GCP bucket path.'
        bucket, blob = cloud_params_file.replace('gs://', '').split('/', 1)
        upload_blob(bucket, ppath, blob)
        prun = client.create_run_from_pipeline_package(
            pipeline_file=pipeline_file,
            arguments=arguments,
            run_name=rname,
            experiment_name=experiment_name,
            namespace=namespace
        )
        outpts.append((rname, prun, f'Parameter tstamp: {tt}'))        
    return outpts
    
if __name__ == '__main__':
    import kfp
    import json
    import argparse as ap

    # get args
    args = get_args()
    if not args.params and not args.json:
        raise Exception('Parameters must be supplied either using --params or --json argument.')

    # On Demand Pricing specific packages and code
    if args.ODP_Event_Key != '' or args.schd_mode != '' :
        from pyspark import SparkContext
        import pyspark
        import sys, glob,zipfile
        import importlib
        
        sc = SparkContext.getOrCreate()
        sc.addPyFile("GER_LP_Code.zip")
        sys.path.insert(0, 'GER_LP_Code.zip')
    
    # load json
    json_params = json.loads(open(args.json).read()) if args.json else []
    if type(json_params)==dict:  # single run case
        json_params = [json_params]
    # run pipe
    outpts = _run(
        args.kf_endpoint,
        pipeline_file=args.pipe_yaml,
        git_branch=args.git_branch,
        git_hash=args.git_hash,
        algo_version=args.algo_version,
        version_type=args.version_type,   
        version_iteration=args.version_iteration, 
        params_files=args.params,
        json_params=json_params,
        params_template=args.params_template,
        tstamp=True,
        add_host_as_user=_add_host_as_user(args.add_host_as_user),
        run_names=args.run_names,
        prefix=args.run_name_prefix,
        experiment_name=args.experiment_name,
        namespace=args.namespace,
        odp_event_key=args.ODP_Event_Key,
        schd_mode=args.schd_mode,
        cmpas=args.cmpas
    )
    for x in outpts:
        print(x)