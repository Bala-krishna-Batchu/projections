pipeline {
    agent any
    stages {
        stage('Hello World') {
            steps {
                echo 'Hello World'
            }
        }
    }
}


/**

The bellow pipeline is a demo pipeline.  It highlights several key logical points of the desired pipeline including:
* How load a docker image
* How to determine the type of branch
* How to run processes in parallel
* How to run python and pytest

The pipeline is working on the “testing_jenkins” branch, if you want to see what the files have. The dummy pipeline uses:
abc.py is a plain python file with a sum. 
def.py is a python file that loads numpy (with the help from docker).
A modify version of the based docker image


pipeline {
    agent {
        dockerfile {
            filename "dockerfile-base"
        }
    }
    stages {
        stage("Verify Branch") {
            steps {
                echo "Your branch: $GIT_BRANCH"
                echo "Running ${env.BUILD_ID} on ${env.JENKINS_URL}"
            }
        }

        stage('Parallel Stage') {
            parallel {
                
                stage("Running python") {
                    steps {
                        sh(script: """python abc.py""")
                    }
                    post {
                        success {
                            echo "Python ran sucessfully"
                        }
                        failure {
                            echo "This does not work yet"
                        }
                    }
                }

                stage("Running numpy") {
                    steps {
                        sh(script: """python def.py""")
                    }
                    post {
                        success {
                            echo "Python ran sucessfully"
                        }
                        failure {
                            echo "This does not work yet"
                        }
                    }
                }

                stage("Running pytest") {
                    steps {
                        sh "ls -la ${pwd()}"
                        sh(script: """pytest -q ./test_lp_functions.py """)
                        
                    }
                    post {
                        success {
                            echo "Pytest ran sucessfully"
                        }
                        failure {
                            echo "This does not work yet"
                        }
                    }
                }

            }
        }
        

    }
}
*/
