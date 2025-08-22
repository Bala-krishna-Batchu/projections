#!/user/bin/bash
set -e

helpFunction()
{
   echo ""
   echo "Usage: $0 -v version_iteration [-r run_method] [-d docker_kf_new] [-b run_batch_date] [-t run_batch_time] [-p run_name_prefix]"
   echo -e "\t Example 1: sh run_multi_client.sh -v 123 -r schedule -d True -b '2025-01-01' -t '12:00:01 AM' -l AETNA"
   echo -e "\t Example 2: sh run_multi_client.sh -v 123 -r instant -d True -p new -c True"
   echo -e "\t-v version_iteration: Version iteration number (required)"
   echo -e "\t-r run_method: Either 'instant' or 'schedule' (default: schedule for Develop | default: instant for non Develop )"
   echo -e "\t-d docker_kf_new: Set to 'True' or 'False' (default: True)"
   echo -e "\t-b run_batch_date: Date in 'YYYY-MM-DD' format (default: Today's date)"
   echo -e "\t-t run_batch_time: Time in 'HH:MM:SS AM/PM' format (default: 1 hour from now in Central Time)"
   echo -e "\t-p run_name_prefix: For scheduler run run_name_prefix is hardcoded to Autoschd-, for default runs update this argument (default: multi-)"
   echo -e "\t-l client_lob: For book of business clients set it to CMK, for Aetna clients set it to AETNA (default: CMK)"
   echo -e "\t-c cmpas: For CMPAS runs set it to True, for non-CMPAS runs set it to False (default: False)"
   echo -e "\t-h Help?"
   exit 1 # Exit script after printing help
}

param_confirm() 
{
    # call with a prompt string or use a default
    read -r -p "${1:-Are you good with the run parameters? [y/n]} " response
    case "$response" in
        [yY][eE][sS]|[yY]) 
            :
            ;;
        *)
            exit 1 #fail 
            ;;
    esac
}

confirm_scheduler_run_time()
{

goon='y'
if [ "$input_date_time" -ge "$current_time" ]; then 
    :
else
    echo "ERROR: Run initiation time: $input_datetime_str is in the past. Hence, no runs are scheduled.."
    exit 1 
fi
if [ "$current_time_plus_twodays" -ge "$input_date_time" ]; then 
    :
else
    echo "WARNING: The run will be scheduled to start more than two days in future."
    read -p  "Do you want to continue [y/n]? " goon 
fi
if [ "$goon" = 'y' ]; then
    echo "The runs are scheduled for $input_datetime_str"
else
    echo "No runs scheduled.."
    exit 1
fi
}

run_method="schedule"  #Pulls the current branch
git_branch=$(git rev-parse --abbrev-ref HEAD) #Pulls the current branch
if [ "$git_branch" != "Develop" ]; then
    # Set defaults for the parameters
    run_method="instant"
fi 
docker_kf_new="True"
run_batch_date=$(TZ="America/Chicago" date +"%Y-%m-%d") # Default to today's date
run_batch_time=$(TZ="America/Chicago" date -d "1 hour" +"%I:%M:%S %p") # Default time 1 hour from now (Central Time)
run_name_prefix="multi-" 
client_lob="CMK"
cmpas="False"


# Parse command line options
while getopts "v:r:d:b:t:p:l:c:h:" opt
do
   case "$opt" in
      v ) version_iteration="$OPTARG" ;;
      r ) run_method="$OPTARG" ;;
      d ) docker_kf_new="$OPTARG" ;;
      b ) run_batch_date="$OPTARG" ;;
      t ) run_batch_time="$OPTARG" ;;
      p ) run_name_prefix="$OPTARG" ;;
      l ) client_lob="$OPTARG" ;;
      c ) cmpas="$OPTARG" ;;
      h ) helpFunction ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Check if required parameter version_iteration is set
if [ -z "$version_iteration" ]; then
   echo "Error: version_iteration is required."
   helpFunction
fi

# Print the parameters for verification
echo "Version Iteration: $version_iteration"
echo "Run Method: $run_method"
echo "Docker KF New: $docker_kf_new"
echo "Client LOB: $client_lob"
echo "CMPAS: $cmpas"

if [ "$run_method" = "schedule" ]; then
    echo "Run Batch Date: $run_batch_date"
    echo "Run Batch Time: $run_batch_time"
    echo "Run Name Prefix: Autoschd-"   
elif [ "$run_method" = "instant" ]; then
    echo "Run Name Prefix: $run_name_prefix"
else
    echo "WARNING: run_method should be either schedule/instant"
fi 

if [ "$client_lob" = "CMK" ]; then
    echo "Initiating book of business run/s"
elif [ "$client_lob" = "AETNA" ]; then
    echo "Initiating AETNA run/s"
else
    echo "ERROR: client_lob (-l) should be set to CMK or AETNA"
    exit 1
fi 

if [ "$cmpas" = "True" ]; then
    echo "Initiating CMPAS run/s"
elif [ "$cmpas" = "False" ]; then
    echo "Initiating non-CMPAS Regular run/s"
else
    echo "ERROR: cmpas (-c) should be set to False or True"
    exit 1
fi 

# confirm with the user if the param look good then proceed 
## If user inputs y then continue running the script else fail
param_confirm

#####################################################################################
#####################################################################################

# Can be constant - Update not required 
kubeflow_endpoint='https://7b9cc36d844fcd2d-dot-us-east1.pipelines.googleusercontent.com' # KF-9
project_name='pbm-mac-lp-prod-ai'
REPOS_DIR='/home/jupyter/clientpharmacymacoptimization'
kf_opt_pipe_run_code='kf_opt_pipe_run.py'
codepy=${REPOS_DIR}/${kf_opt_pipe_run_code}
YAML='kubeflow_optimization_pipeline.yaml'   #This file needs to exist in the current directory
git_hash=$(git rev-parse --short HEAD) 
algo_version='LP'
version_type='WIP-'$(hostname -s) # Pulls the host name

input_datetime_str="$run_batch_date $run_batch_time"
input_date_time=$(TZ="America/Chicago" date -d "$input_datetime_str" +%s)
current_time=$(TZ="America/Chicago" date +%s)
current_time_plus_twodays=$(TZ="America/Chicago" date -d "48 hours" +%s)

if [ "$docker_kf_new" = "True" ]; then
    echo "..... Create Docker Image....."
    sh ./Docker_Image_Management.sh ${version_iteration}
    echo "... Create KubeFlow Optimization Pipeline Builder Code...."
    python Kubeflow_Optimization_Pipeline_Builder.py --version ${version_iteration} --kf_endpoint ${kubeflow_endpoint} --repo_head ${REPOS_DIR} --algo_version ${algo_version} --version_type ${version_type} --client_lob ${client_lob}
else 
    echo "Use previous docker : $(docker image ls --format "{{.Tag}}" | uniq | head -1)"
fi

# Check if docker is present
if [ "$(docker images -q | wc -l)" -eq 0 ]; then
    echo "No Docker images found. Recreate with docker_kf_new = True"
    exit 1
fi

# Check if the older docker image is not build on local branch
if echo "$(docker image ls --format "{{.Tag}}" | uniq | head -1)" | grep -q "$git_branch"; then
    :
else
    echo "WARNING: The older docker image is not build on the branch you have on local" 
fi

echo "..... Creating parameter json file for multiple clients ....."
python json_gen.py --CMPAS ${cmpas}

if [ "$run_method" = "instant" ]; then
    echo "..... Submitting kubeflow job for multiple client params ...."
    # This step will submit kubeflow job using the yaml created in previous step
    python ${codepy} --kf-endpoint ${kubeflow_endpoint} \
        --pipe-yaml ${REPOS_DIR}/${YAML}\
        --git-branch ${git_branch}\
        --git-hash ${git_hash}\
        --algo-version ${algo_version}\
        --version-iteration ${version_iteration}\
        --version-type ${version_type}\
        --json client_params.json\
        --params-template ${REPOS_DIR}/GER_LP_Code/CPMO_parameters_TEMPLATE.py\
        --run-name-prefix ${run_name_prefix}\
        --add-host-as-user false\
        --ODP_Event_Key ""\
        --cmpas ${cmpas}

    echo "... Job submitted, check the GCP pipeline dashboard for progress"

elif [ "$run_method" = "schedule" ]; then
    echo "....Scheduler to submit code...."
    # confirm if the date & time are future CT time 
    confirm_scheduler_run_time    
    # run_name_prefix for the scheduler is "Autoschd-"
    python upload_gcs.py --run_batch_date="${run_batch_date}" --run_batch_time="${run_batch_time}" --kubeflow_endpoint="${kubeflow_endpoint}" --algo_version="${algo_version}" --version_type="${version_type}"
    echo "....Job submitted, using scheduler....."
else
    echo "ERROR: No jobs scheduled for run as run_method is not set to schedule or instant."
    exit 1
fi

