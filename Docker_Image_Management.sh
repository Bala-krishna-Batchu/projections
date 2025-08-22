#!/bin/bash

# Docker Image Management Script

# Optional: Clean up local images
# docker rmi $(docker images -q) -f

# Parameters
HNAME='us.gcr.io'  # Container registry address
PROJECT_ID='pbm-mac-lp-prod-ai'

# Image Names
BASE_NAME='pbm_base'
SCRIPT_RUN_NAME='pbm_script_run'
OPT_NAME='pbm_opt'

# Get Git hash and branch
GIT_HASH=$(git rev-parse --short HEAD)
GIT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Image Version Setup
VERSION_ITERATION=$1  # Change as needed
VERSION_TYPE="WIP-$(hostname)"  # Use 'PROD' or 'DEV' for official versions
VERSION="${GIT_BRANCH}-${GIT_HASH}-${VERSION_TYPE}-${VERSION_ITERATION}"

# Tags for Google Container Registry
BASE_TAG="${HNAME}/${PROJECT_ID}/${BASE_NAME}:${VERSION}"
SCRIPT_RUN_TAG="${HNAME}/${PROJECT_ID}/${SCRIPT_RUN_NAME}:${VERSION}"
OPT_TAG="${HNAME}/${PROJECT_ID}/${OPT_NAME}:${VERSION}"

# Print the version
echo "Version: ${VERSION}"

# List current images/versions
#gcloud container images list --repository ${HNAME}/${PROJECT_ID}
#gcloud container images list-tags ${HNAME}/${PROJECT_ID}/${BASE_NAME}

# Uncomment below to delete images if needed
# VERSION_DELETE='version_to_delete'
# gcloud container images delete --quiet --force-delete-tags ${HNAME}/${PROJECT_ID}/${SCRIPT_RUN_NAME}:${VERSION_DELETE}
# gcloud container images delete --quiet --force-delete-tags ${HNAME}/${PROJECT_ID}/${OPT_NAME}:${VERSION_DELETE}
# gcloud container images delete --quiet --force-delete-tags ${HNAME}/${PROJECT_ID}/${BASE_NAME}:${VERSION_DELETE}

# Uncomment below to delete images by DIGEST
# VERSION_DIGEST='digest_to_delete'
# gcloud container images delete --quiet --force-delete-tags ${HNAME}/${PROJECT_ID}/${SCRIPT_RUN_NAME}@${VERSION_DIGEST}
# gcloud container images delete --quiet --force-delete-tags ${HNAME}/${PROJECT_ID}/${OPT_NAME}@${VERSION_DIGEST}
# gcloud container images delete --quiet --force-delete-tags ${HNAME}/${PROJECT_ID}/${BASE_NAME}@${VERSION_DIGEST}

# Checks if the gurobi file is present
if [ -e gurobi.lic ]
then
    echo "Gurobi file is present"
else
    echo "Gurobi file is not present, please upload the gurobi.lic file to the directory"
    exit 1
fi

# List local docker images
docker image list

# Build Docker Images Locally
echo "Building Base Docker Image..."
docker build -f dockerfile-base -t ${BASE_NAME}:${VERSION} .
docker image tag ${BASE_NAME}:${VERSION} ${BASE_TAG}

# Build Script Run Image
echo "Building Script Run Docker Image..."
cat > dockerfile-script-run << EOF
FROM ${BASE_TAG}
RUN ["pip", "install", "--upgrade", "scikit-learn==0.23.2", "scipy==1.6.2", "statsmodels==0.12.1", "--index-url", "https://nexus-ha.cvshealth.com:9443/repository/pypi-proxy/simple"]
EOF
docker build -f dockerfile-script-run -t ${SCRIPT_RUN_NAME}:${VERSION} .
docker image tag ${SCRIPT_RUN_NAME}:${VERSION} ${SCRIPT_RUN_TAG}

# Build Optimization Image
echo "Building Optimization Docker Image..."
cat > dockerfile-opt << EOF
FROM ${BASE_TAG}
RUN ["pip", "install", "PuLP==2.4", "duckdb==0.8.0", "XlsxWriter==1.3.7", "--index-url", "https://nexus-ha.cvshealth.com:9443/repository/pypi-proxy/simple"]
EOF
docker build -f dockerfile-opt -t ${OPT_NAME}:${VERSION} .
docker image tag ${OPT_NAME}:${VERSION} ${OPT_TAG}

# Upload images to GCP Container Registry
echo "Pushing Docker Images to GCP..."
docker push ${BASE_TAG}
docker push ${SCRIPT_RUN_TAG}
docker push ${OPT_TAG}

echo "Docker Image Management Completed."