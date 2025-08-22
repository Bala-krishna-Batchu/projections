import os
import subprocess
import sys
from google.cloud import storage
from pathlib import Path
import zipfile
import argparse as ap
import socket
from google.cloud import bigquery
import json
from collections import OrderedDict
import encodings
import glob
from datetime import datetime
import pytz
from datetime import date
import time
import random

def copy_dir(bucket_name, gcs_source_dir, current_dir):
    '''Copies local directory to gcs'''
    try:
        subprocess.run(f'gsutil -m cp -r {current_dir} gs://{bucket_name}/{gcs_source_dir}', shell=True, check=True)
        print(f" Upload to gcs is successful ")
                
    except Exception as e:
        print(f" Upload failed. Return code:{e}")
        sys.exit()
        
def create_zip(inner_dir, zip_output_path):
    '''Creates ZIP file from local directory'''
    try:
        # name of the new Zip file
        print(f"Creating ZIP file: {zip_output_path}")

        # Create a ZipFile Object
        with zipfile.ZipFile(zip_output_path, 'w') as zip_object:
            print(f"Adding files from directory: {inner_dir}")
            # iterate over files in source directory
            for filenames in os.listdir(inner_dir):
                #print(filenames)
                file_path = os.path.join(inner_dir,filenames)
                
                if os.path.isfile(file_path) and filenames.endswith('.py'):
                    print(filenames)
                    zip_object.write(filenames, compress_type=zipfile.ZIP_DEFLATED)

            print(f"Zip file created successfully: {zip_output_path}")    
                #zip_object.close()

    except Exception as e:
        print(f"Error while creating ZIP file: {e}")
        sys.exit()

        
def json_split(current_dir,final_gcs_bucket_name,run_kubeflow_endpoint,run_YAML,run_git_branch,run_git_hash,run_algo_version,version_iteration,version_type,run_run_name_prefix,run_add_host_as_user,email_id,run_batch_date,time_24h):
    
    sql_prep = ''
    # Read the 'client_param.json' file
    json_loc = current_dir+'/client_params.json'
    with open(json_loc, 'r',encoding='utf-8') as file:
        data = json.load(file,object_pairs_hook=OrderedDict)     
    # 3 clients runs to go through every ~3 mins for now  
    records_per_file = 3
        
    # Split the records into separate JSON files
    for i in range(0, len(data), records_per_file):  
        filename = current_dir+'/client_params_v%d.json' % (i // records_per_file + 1)
        record = data[i:i+records_per_file]
        batch_id=record[0]['BATCH_ID']
        file_name_temp = os.path.basename(filename)
        sql_prep=sql_prep+f"('{final_gcs_bucket_name}','{run_kubeflow_endpoint}','{run_YAML}','{run_git_branch}','{run_git_hash}','{run_algo_version}','{version_iteration}','{version_type}','{file_name_temp}','CPMO_parameters_TEMPLATE.py','{run_run_name_prefix}','{run_add_host_as_user}','{batch_id}','{email_id}','{run_add_host_as_user}','{run_batch_date}','{time_24h}'),"
        
        with open(filename, 'w') as outfile:
            json.dump(record, outfile, indent=4)

    print('Saved %s' % filename)
    print('Splitting completed.')
    return sql_prep
    

def rm_files(current_dir): 
    os.chdir(current_dir)  # Replace with the actual folder path
    # Get a list of files matching the wildcard pattern
    files_to_delete = glob.glob(os.path.join(current_dir, '*client_params_v*'))
    # Delete the files
    for file_path in files_to_delete:
        os.remove(file_path)
                
            
def main():
    
    parser = ap.ArgumentParser()
    parser.add_argument(
        '--run_batch_date',
        type=str,
        help='provide the load date in %Y-%m-%d format',
        default=None
    )
    parser.add_argument(
        '--run_batch_time',
        type=str,
        help='provide the load hour in %I:%M:%S %p format',
        default = '02:00:00 AM'
    )
    parser.add_argument(
        '--kubeflow_endpoint',
        type=str,
        help='provide the end point',
        default='https://7b9cc36d844fcd2d-dot-us-east1.pipelines.googleusercontent.com'
    )
    parser.add_argument('-algo_v','--algo_version', help='Algo Version', default='LP')
    parser.add_argument('-v_type','--version_type', help='Version Type', default=f'WIP-{socket.gethostname()}')

    args = parser.parse_args()
    # Change these parameters according to project
    bucket_name = 'pbm-mac-lp-prod-ai-bucket'  # in gcs
    local_dir_nm = 'clientpharmacymacoptimization'# local directory for downloaded files
    version_type = args.version_type
    current_ts = str(int(time.time()) + random.randint(1000,2000)) 
    append_version_type = version_type+'_'+current_ts
    gcs_source_dir = 'de_test/auto_scheduling_runs/'+append_version_type+'/'
    final_gcs_bucket_name=f"gs://{bucket_name}/{gcs_source_dir}{local_dir_nm}/" 
    
    print(args.run_batch_date)
    print(args.run_batch_time)

    if args.run_batch_date is None:
        run_batch_date = datetime.now()
        print(run_batch_date)
        cst_zone = pytz.timezone("America/Chicago")
        utc_zone = pytz.utc
        run_batch_frmt_date_time_utc = utc_zone.localize(run_batch_date)
        run_batch_frmt_date_time = run_batch_frmt_date_time_utc.astimezone(cst_zone)
        cnvt_run_batch_date = run_batch_frmt_date_time.strftime("%Y-%m-%d")
        print(cnvt_run_batch_date)
    else:
        run_batch_frmt_date = args.run_batch_date
        run_batch_date_strp = datetime.strptime(run_batch_frmt_date,"%Y-%m-%d")
        cnvt_run_batch_date = run_batch_date_strp.strftime("%Y-%m-%d")

    current_dir = os.getcwd()
    current_dir_nm = os.path.basename(current_dir)
    inner_dir = current_dir+'/GER_LP_Code/'
    init_file_loc = inner_dir+'__init__.py'
    file_path = Path(init_file_loc)
    zip_output_path = current_dir+'/GER_LP_Code.zip'
    gcs_bucket_name=f"gs://{bucket_name}/{gcs_source_dir}"
    
    if current_dir_nm == local_dir_nm:
        
        print("Correct directory")
        os.chdir(inner_dir)
        
        if file_path.is_file():
            print("File already exists ")
        else:
            with open(file_path,"w") as file:
                print("File Created")
                 
        c = bigquery.Client()
        #run_git_branch = !(git rev-parse --abbrev-ref HEAD) #Pulls the current branch
        branch = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True, text=True)
        run_git_branch = branch.stdout.strip()
        hashv = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True)
        run_git_hash = hashv.stdout.strip()
        run_algo_version=args.algo_version
        ls_tag_name = subprocess.run(['docker', 'image', 'ls','--format',"{{.Tag}}"], capture_output=True, text=True)
        tag_name = sorted(ls_tag_name.stdout.splitlines(), reverse=True)[0]
        version_iteration = tag_name.split("-")[-1]
        if version_iteration is None: 
            version_iteration=0 
        #print(version_iteration)
        time_obj = datetime.strptime(args.run_batch_time, '%I:%M:%S %p')

        # Convert the datetime object to a string in 24-hour format
        time_24h = time_obj.strftime('%H:%M:%S')
        email_id='vinothkumar.bangarusamy@cvshealth.com'
        
        run_YAML='kubeflow_optimization_pipeline.yaml'
        run_run_name_prefix ="Autoschd-"
        run_add_host_as_user="False"
        run_kubeflow_endpoint=args.kubeflow_endpoint
    
        print("run_kubeflow_endpoint=",run_kubeflow_endpoint)
        print("run_git_branch=",run_git_branch)
        print("run_git_hash=",run_git_hash)
        print("run_algo_version=",run_algo_version)
        print("version_iteration=",version_iteration)
        print("version_type=",version_type)
        print("run_YAML=",run_YAML)
        print("run_run_name_prefix=",run_run_name_prefix)
        print("run_add_host_as_user=",run_add_host_as_user)
        print(version_type)

        sql_prep_f=''
        create_zip(inner_dir, zip_output_path)
        os.chdir(current_dir)
        sql_prep_f =  json_split(current_dir,final_gcs_bucket_name,run_kubeflow_endpoint,run_YAML,run_git_branch,run_git_hash,run_algo_version,version_iteration,version_type,run_run_name_prefix,run_add_host_as_user,email_id,cnvt_run_batch_date,time_24h)
        copy_dir(bucket_name, gcs_source_dir, current_dir)
        rm_files(current_dir)
        sql_prep_f=sql_prep_f[:-1]+''
        sql_1=f"""insert into pbm-mac-lp-prod-ai.pricing_management.algo_schd_mtdt (base_gcs_path,kubeflow_endpoint,pipe_yaml,git_branch,git_hash,algo_version,version_iteration,version_type,json_file,params_template,run_name_prefix,add_host_as_user,batch_id,email,user_id,job_run_date,job_run_time) values {sql_prep_f}"""
        print(sql_1)
        r_sql_1=c.query(sql_1).result()
    else:  
        print("Incorrect directory")
        sys.exit()

if __name__ == "__main__":
    main()
