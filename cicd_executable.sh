#!/user/bin/bash
set -e

helpFunction()
{
   echo ""
   echo "Usage: $0 -u username -v version_iteration [-d docker_kf_new]"
   echo -e "Example: sh cicd_executable.sh -u your_name_here -v 123"
   echo -e "\t-u batch_id: Batch ID to use in the run_name_prefix (required)"
   echo -e "\t-v version_iteration: Version iteration number (required)"
   echo -e "\t-d docker_kf_new: Set to 'True' or 'False' (default: True)"
   echo -e "\t-p run_name_prefix: Run name prefix (default: cicd-)"
   echo -e "\t-l client_lob: For book of business clients set it to CMK, for Aetna clients set it to AETNA (default: CMK)"
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


# Set defaults for the parameters
docker_kf_new="True"
run_name_prefix="cicd-"
client_lob="CMK"

# Parse command line options
while getopts "u:v:d:p:l:h:" opt
do
   case "$opt" in
      u ) batch_id="$OPTARG" ;;
      v ) version_iteration="$OPTARG" ;;
      d ) docker_kf_new="$OPTARG" ;;
      p ) run_name_prefix="$OPTARG" ;;
      l ) client_lob="$OPTARG" ;;
      h ) helpFunction ;; # Print helpFunction 
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Check if required parameter username is set
if [ -z "$batch_id" ]; then
   echo "Batch ID is not set"
   helpFunction
fi

# Check if required parameter version_iteration is set
if [ -z "$version_iteration" ]; then
   echo "Error: version_iteration is required."
   helpFunction
fi

# Print the parameters for verification
echo "Batch ID: $batch_id"
echo "Version Iteration: $version_iteration"
echo "Docker KF New: $docker_kf_new"
echo "Client LOB: $client_lob"


if [ "$client_lob" = "CMK" ]; then
    echo "Initiating book of business run/s"
elif [ "$client_lob" = "AETNA" ]; then
    echo "Initiating AETNA run/s"
else
    echo "ERROR: client_lob (-l) should be set to CMK or AETNA"
    error 1
fi 

# confirm with the user if the param look good then proceed 
## If user inputs y then continue running the script else fail
param_confirm

#####################################################################################
#####################################################################################

# Can be constant - Update not required 
kubeflow_endpoint='https://9750bebe2d7a901-dot-us-east1.pipelines.googleusercontent.com'
project_name='pbm-mac-lp-prod-ai'
REPOS_DIR='/home/jupyter/clientpharmacymacoptimization'
kf_opt_pipe_run_code='kf_opt_pipe_run.py'
codepy=${REPOS_DIR}/${kf_opt_pipe_run_code}
YAML='kubeflow_optimization_pipeline.yaml'    # This file needs to exist in the current directory
git_branch=$(git rev-parse --abbrev-ref HEAD) # Pulls the current branch
git_hash=$(git rev-parse --short HEAD) 
algo_version='LP'
version_type='WIP-'$(hostname -s) # Pulls the host name

## Create docker image or use the old docker imaage
if [ "$docker_kf_new" = "True" ]; then
    echo "..... Create Docker Image....."
    sh ./Docker_Image_Management.sh ${version_iteration}
    echo "... Create KubeFlow Optimization Pipeline Builder Code...."
    python Kubeflow_Optimization_Pipeline_Builder.py --version ${version_iteration} --kf_endpoint ${kubeflow_endpoint} --repo_head ${REPOS_DIR} --algo_version ${algo_version} --version_type ${version_type}
else 
    echo "Use previous docker : $(docker image ls --format "{{.Tag}}" | uniq | head -1)"
fi

## Check if docker is present and fail it doesnt find any docker 
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
python cicd_json_gen.py --batch_id ${batch_id}

echo "..... Submitting kubeflow job for multiple client params...."
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
	--run-name-prefix ${run_name_prefix}${batch_id}-\
	--add-host-as-user false\
	--experiment-name CICD

echo "...Job submitted, check the GCP pipeline dashboard for progress...."
