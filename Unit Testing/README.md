# UC Optimization QA Scripts

## QA_INPUT_DATA.py

Run after all PreProcessing and prior to Daily_Input_Read. Checks for all files that are needed and QAs the input data.
Some of the checks inlude:

1. missing values
1. duplication
1. pharmacy (chain_group) coverage

The output includes:
1. A log file in the logs directory that includes WARNINGS (search file for 'WARNING') for missing data and other possible issues. Not all issues require followup.
1. A csv file that includes the pharmacy coverage for the different datasets (in the p.FILE_INPUT_PATH directory).

## QA.py

Run after running ClientPharmacyMacOptimization. These checks require the pilot = True file to be set. This QA code checks output files for accuracy and checks that 
all new prices don't violate price-change constraints. The output is a series of reports saved in the p.FILE_OUTPUT_PATH directory. 

 