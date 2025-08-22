# import numpy as np
# import pandas as pd
# import sys
# sys.path.append('..')
# import CPMO_parameters as p
# from CPMO_shared_functions import standardize_df

# import warnings
# warnings.filterwarnings('ignore')

    
from kfp.components import InputPath, OutputPath  #, func_to_container_op

def arguments():
    import argparse as ap
    import json
    import jinja2 as jj2
    import BQ
    from collections import namedtuple
    
    parser = ap.ArgumentParser()
    parser.add_argument(
        '--custom-args-json',
        help=('JSON file URL, for file supplying custom values. '
              'for example: '
              '{"client_name": "client1", "TIMESTAMP": "2020-12-29", ...}'),
        required=False,
        default=None
    )
    parser.add_argument(
        '-t', '--template',
        help='Path to parameters template.',
        required=False,
        default=None
    )
    parser.add_argument(
        '-p', '--parameter-file',
        help=('Path to hard coded parameters file.'
              'This option will override the --template option if it is specified.'),
        required=False,
        default=None
    )
    parser.add_argument(
        '--loglevel', choices=['INFO', 'DEBUG', 'WARN', 'ERROR', 'FATAL', 'CRITICAL'], 
        default='INFO',
        help='Logging is currently not supported in QA.py; however, this is included for future compatability.'
    )
    args = parser.parse_args()    
    # setup parameters
    outpt = namedtuple('Output', ['params', 'loglevel'])
    # if hard coded parameters are supplied, read and return as string
    if not args.parameter_file and not args.template:
        raise Exception('Either --parameter-file or --template CLI arg must be supplied.')
    if args.parameter_file:
        params = open(args.parameter_file, 'r').read()
        return outpt(params=params, loglevel=args.loglevel)
    # if template path is supplied (and not hard coded parameters), supply custom parameters (if any)
    # and return parameter string
    elif args.template:
        arguments = dict(json.load(open(args.custom_args_json)))
        template = jj2.Template(open(args.template).read())
        return outpt(params=template.render(**arguments), loglevel=args.loglevel)

    
def qa_Pharmacy_Output(params_in: str):
    '''
    Test: that output file LP_Algorithm_Pharmacy_Output_Month looks okay
    '''
    import pandas as pd
    import numpy as np
    # TODO: IS THIS THE RIGHT THING? WE MIGHT NEED TO CHANGE TABLE NAME
    from BQ import MedD_LP_Algorithm_Pharmacy_Output_Month
    from types import ModuleType
    from util_funcs import write_params, read_BQ_data
    import util_funcs as uf
    if isinstance(params_in, ModuleType):
        p = params_in
    else:
        write_params(params_in)
        import CPMO_parameters as p

    from CPMO_shared_functions import update_run_status
    update_run_status(i_error_type='Started QA checks in QA.py')
    try:
        file = p.FILE_OUTPUT_PATH + p.PHARMACY_OUTPUT
        if p.WRITE_TO_BQ:
            # TODO: wHAT IS THE NEW EQUIVALENT
            df = read_BQ_data(
                MedD_LP_Algorithm_Pharmacy_Output_Month,
                project_id = p.BQ_OUTPUT_PROJECT_ID,
                dataset_id = p.BQ_OUTPUT_DATASET,
                table_id = 'MedD_LP_Algorithm_Pharmacy_Output_Month_subgroup',
                run_id = p.AT_RUN_ID,
                client = ', '.join(sorted(p.CUSTOMER_ID)),
                period = p.TIMESTAMP,
                output = True
            )
        else:

            df = pd.read_csv(file, dtype = p.VARIABLE_TYPE_DIC)

        qa_Pharmacy_Output_err_list = []
        columns = ['GPI_NDC', 'GPI', 'NDC', 'PKG_SZ', 'CLIENT', 'BREAKOUT', 'REGION', 'BG_FLAG', 'MEASUREMENT', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'MAC_LIST',
                   'PRICE_MUTABLE','CLAIMS_PROJ_EOY', 'QTY_PROJ_EOY', 'FULLAWP_ADJ_PROJ_EOY', 'OLD_MAC_PRICE', 'UC_UNIT', 'MAC1026_UNIT_PRICE',
                   'GPI_Strength', 'New_Price', 'lb', 'ub', 'LM_CLAIMS', 'LM_QTY', 'LM_FULLAWP_ADJ', 'LM_PRICE_REIMB', 'PRICE_REIMB_CLAIM']
        if df.columns.isin(columns).all(): 
            pass
        else:
             qa_Pharmacy_Output_err_list.append("make sure all columns are included")
        if (df.shape[0] >= 0): 
            pass
        else: 
            qa_Pharmacy_Output_err_list.append("df.shape[0] >= 0")
        if (np.sum([df[column].isna().all() for column in df.columns]) == 0): 
            pass
        else:
            qa_Pharmacy_Output_err_list.append("no columns with all NAs # this is broken due to UC_UNIT")
        if  (df.loc[:, ~df.columns.isin(['PRICE_REIMB_CLAIM'])].isna().sum().any() == 0): 
            pass
        else:
            qa_Pharmacy_Output_err_list.append("no columns with all NAs # this is broken due to UC_UNIT")

        float_cols = ['PKG_SZ', 'CLAIMS_PROJ_EOY', 'QTY_PROJ_EOY', 'FULLAWP_ADJ_PROJ_EOY', 'OLD_MAC_PRICE', 'MAC1026_UNIT_PRICE',
                      'New_Price', 'lb', 'ub', 'LM_CLAIMS', 'LM_QTY', 'LM_FULLAWP_ADJ', 'LM_PRICE_REIMB', 'PRICE_REIMB_CLAIM']
        int_cols = ['GPI_Strength'] # 'REGION', 'CLIENT' got added since they are now the client ID
        char_cols = ['GPI_NDC', 'NDC', 'BREAKOUT', 'MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP', 'CHAIN_SUBGROUP','MAC_LIST','REGION', 'CLIENT']
        if np.all([df[column].dtype == 'float64' for column in float_cols]):
            pass 
        else: 
            qa_Pharmacy_Output_err_list.append("np.all([df[column].dtype == 'float64' for column in float_cols])")

        if df.groupby(['GPI_NDC', 'CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP', 'CHAIN_SUBGROUP']).ngroups == df.shape[0]:
            pass
        else: 
            qa_Pharmacy_Output_err_list.append("df.groupby(['GPI_NDC', 'CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP', 'CHAIN_SUBGROUP']).ngroups == df.shape[0]")

        if df.MEASUREMENT.isin(['R30', 'R90', 'M30']).all():
            pass
        else:
            qa_Pharmacy_Output_err_list.append("df.MEASUREMENT.isin(['R30', 'R90', 'M30']).all()")
        if df.CHAIN_GROUP.isin(list(set(p.PHARMACY_LIST['GNRC'] + p.PSAO_LIST['GNRC'] + p.PHARMACY_LIST['BRND'] + p.PSAO_LIST['BRND'])) + ['MCHOICE', 'MAIL']).all(): 
            pass
        else: 
            qa_Pharmacy_Output_err_list.append("df.CHAIN_GROUP.isin(p.PHARMACY_LIST + p.PSAO_LIST + ['MCHOICE', 'MAIL']).all()")
        if np.all([(df.loc[~df[column].isna(), column] >= 0).all() for column in float_cols + int_cols]): 
            pass
        else: 
            qa_Pharmacy_Output_err_list.append("np.all([(df.loc[~df[column].isna(), column] >= 0).all() for column in float_cols + int_cols])")
        if np.all((df['lb'] > 0) & (df['ub'] > 0) & (df['lb'] <= df['ub'])): 
            pass 
        else:
            qa_Pharmacy_Output_err_list.append("np.all((df['lb'] > 0) & (df['ub'] > 0) & (df['lb'] <= df['ub']))")
        


        if np.all(df['GPI_NDC'].str.len() == 26): 
            pass
        else: 
            qa_Pharmacy_Output_err_list.append("np.all(df['GPI_NDC'].str.len() == 26)")

        # MedD fix, the following assert will fail for MedD clients because MedD clients may have more than one customer id and the breakouts are not similar to commercial
        # assert np.size(df.CLIENT.unique()) == 1
        # assert np.size(np.unique([x[0:4] for x in df['MAC_LIST'].astype(str).unique()])) == 1
        # assert ( ( df['BREAKOUT'] == 'RETAIL') | \
        #    (df['BREAKOUT'] == df['CLIENT'] + '_RETAIL') | (df['BREAKOUT'] == df['CLIENT'] + '_R30') | \
        #    (df['BREAKOUT'] == df['CLIENT'] + '_R90') | (df['BREAKOUT'] == 'MAIL')  ).all()

        # Quick fix, line bellow commented out since it does not longer exist.
        # assert np.all((np.round(df['MAC1026_UNIT_PRICE'], 4) <= np.round(df['New_Price'], 4)) | (df['PRICE_MUTABLE'] == 0))
        # assert np.all((np.round(df['lb'], 2) <= np.round(df['New_Price'], 2)) | (df['PRICE_MUTABLE'] == 0))
        # assert np.all((np.round(df['ub'], 2) >= np.round(df['New_Price'], 2)) | (df['PRICE_MUTABLE'] == 0))  # check this!
        if np.all((df['OLD_MAC_PRICE'] > 0) & (df['New_Price'] > 0)): 
            pass 
        else: 
            qa_Pharmacy_Output_err_list.append("np.all((df['OLD_MAC_PRICE'] > 0) & (df['New_Price'] > 0))")
        
        if len(qa_Pharmacy_Output_err_list) != 0: 
            print("Failing QA/s : ")
            print(qa_Pharmacy_Output_err_list)
            assert False, qa_Pharmacy_Output_err_list
    except Exception as e:
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'QA File MedD_LP_Algorithm_Pharmacy_Output_Month', repr(e), error_loc)
        raise e


def qa_Price_Check_Output(params_in: str):
    '''
    Test: that output file Price_Check_Output looks okay
    '''
    import pandas as pd
    import numpy as np
    from BQ import MedD_LP_Algorithm_Pharmacy_Output_Month, Price_Check_Output
    from types import ModuleType
    from util_funcs import write_params, read_BQ_data
    import util_funcs as uf
    if isinstance(params_in, ModuleType):
        p = params_in
    else:
        write_params(params_in)
        import CPMO_parameters as p

    from CPMO_shared_functions import update_run_status
    try:
        file = p.FILE_OUTPUT_PATH + p.PHARMACY_OUTPUT
        if p.WRITE_TO_BQ:
            df = read_BQ_data(
                MedD_LP_Algorithm_Pharmacy_Output_Month,
                project_id = p.BQ_OUTPUT_PROJECT_ID,
                dataset_id = p.BQ_OUTPUT_DATASET,
                table_id = 'MedD_LP_Algorithm_Pharmacy_Output_Month_subgroup',
                run_id = p.AT_RUN_ID,
                client = ', '.join(sorted(p.CUSTOMER_ID)),
                period = p.TIMESTAMP,
                output = True)
        else:
            df = pd.read_csv(file, dtype = p.VARIABLE_TYPE_DIC)

        file = p.FILE_OUTPUT_PATH + p.PRICE_CHECK_OUTPUT
        if p.WRITE_TO_BQ:
            df2 = read_BQ_data(
                Price_Check_Output,
                project_id = p.BQ_OUTPUT_PROJECT_ID,
                dataset_id = p.BQ_OUTPUT_DATASET,
                table_id = 'Price_Check_Output_subgroup',
                run_id = p.AT_RUN_ID,
                client = ', '.join(sorted(p.CUSTOMER_ID)), #p.client_name_BQ,
                period = p.TIMESTAMP,
                output = True)
        else:
            df2 = pd.read_csv(file, dtype = p.VARIABLE_TYPE_DIC)

        columns = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'MAC_LIST', 'GPI_NDC', 'GPI', 'NDC',
                                     'OLD_MAC_PRICE', 'New_Price', 'Final_Price', 'PKG_SZ', 'QTY_PROJ_EOY', 'GPI_CHANGE_EXCEPT', 'FULLAWP_ADJ_PROJ_EOY',
                                     'CLAIMS_PROJ_EOY', 'PRICE_MUTABLE', 'MAC1026_UNIT_PRICE']
        
        qa_Price_Check_Output = []
        
        if df2.shape[0] == df.shape[0]: 
            pass 
        else: 
            qa_Price_Check_Output.append("df2.columns.isin(columns).all()")
        if df2.shape[0] == df.shape[0]: 
            pass
        else:
            qa_Price_Check_Output.append( "df2.shape[0] == df.shape[0]")
        if np.sum([df[column].isna().any() for column in df.columns[:-1]]) == 0:
            pass
        else: 
            qa_Price_Check_Output.append("np.sum([df[column].isna().any() for column in df.columns[:-1]]) == 0")
        float_cols = ['OLD_MAC_PRICE', 'New_Price', 'Final_Price', 'PKG_SZ', 'QTY_PROJ_EOY', 'FULLAWP_ADJ_PROJ_EOY', 'CLAIMS_PROJ_EOY', 'MAC1026_UNIT_PRICE']
        int_cols = [ 'GPI_CHANGE_EXCEPT']
        char_cols = ['CLIENT','REGION','MAC_LIST', 'BREAKOUT',  'MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GPI_NDC', 'NDC']
        if np.all([df2[column].dtype == 'float64' for column in float_cols]): 
            pass
        else:
            qa_Price_Check_Output.append("np.all([df2[column].dtype == 'float64' for column in float_cols])")
        
    #     assert np.all([df2[column].dtype == 'int64' for column in int_cols])
    #     assert np.all([df2[column].dtype == 'object' for column in char_cols])

        if df2.groupby(['GPI_NDC', 'CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP', 'CHAIN_SUBGROUP']).ngroups == df2.shape[0]: 
            pass 
        else: 
            qa_Price_Check_Output.append("df2.groupby(['GPI_NDC', 'CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP', 'CHAIN_SUBGROUP']).ngroups == df2.shape[0]")

        if df2.MEASUREMENT.isin(['R30', 'R90', 'M30']).all(): 
            pass
        else:    
            qa_Price_Check_Output.append("df2.MEASUREMENT.isin(['R30', 'R90', 'M30']).all()")
        if df2.CHAIN_GROUP.isin(list(set(p.PHARMACY_LIST['GNRC'] + p.PSAO_LIST['GNRC'] + p.PHARMACY_LIST['BRND'] + p.PSAO_LIST['BRND'])) + ['MCHOICE', 'MAIL']).all():
            pass
        else:
            qa_Price_Check_Output.append("df2.CHAIN_GROUP.isin(p.PHARMACY_LIST + p.PSAO_LIST + ['MCHOICE', 'MAIL']).all()")
        if np.all([(df2[column] >= 0).all() for column in float_cols + int_cols]):
            pass
        else:
            qa_Price_Check_Output.append("np.all([(df2[column] >= 0).all() for column in float_cols + int_cols])")
        if np.all(df2['GPI_NDC'].str.len() == 26):
            pass
        else:
            qa_Price_Check_Output.append("np.all(df2['GPI_NDC'].str.len() == 26)")
        if np.all((np.isclose(df2['Final_Price'],df2['New_Price'], atol=1e-4)) | \
                      (np.isclose(df2['Final_Price'],df2['OLD_MAC_PRICE'],atol=1e-4)) | \
                      (df2['PRICE_MUTABLE'] == 0)): 
            pass             
        else: 
            qa_Price_Check_Output.append("Final_Price is close to New_Price, Final_Price isclose to OLD_MAC_PRICE, where PRICE_MUTABLE == 0")
        if np.all((df2['OLD_MAC_PRICE'] > 0) & (df2['New_Price'] > 0) & (df2['Final_Price'] > 0)):
            pass
        else:
            qa_Price_Check_Output.append("np.all((df2['OLD_MAC_PRICE'] > 0) & (df2['New_Price'] > 0) & (df2['Final_Price'] > 0))")

        # MedD fix, the following assert will fail for MedD clients because MedD clients may have more than one customer id and the breakouts are not similar to commercial
        # assert np.size(df2.CLIENT.unique()) == 1
        # assert np.size(np.unique([x[0:4] for x in df2['MAC_LIST'].astype(str).unique()])) == 1
        # assert ( ( df['BREAKOUT'] == 'RETAIL') | \
        #    (df['BREAKOUT'] == df['CLIENT'] + '_RETAIL') | (df['BREAKOUT'] == df['CLIENT'] + '_R30') | \
        #    (df['BREAKOUT'] == df['CLIENT'] + '_R90') | (df['BREAKOUT'] == 'MAIL')  ).all()

        #Added to accomodate that now Mail can go below the Mac1026, however retail should not.  The lines above check for both types and the last ones check only for mail
        df2_sub = df2[df2['MEASUREMENT'].isin(['R30','R90','RETAIL'])]
        if np.all((np.round(df2_sub['MAC1026_UNIT_PRICE'], 4) <= np.round(df2_sub['New_Price'], 4)) | (df2_sub['PRICE_MUTABLE'] == 0)):
            pass
        else:
            qa_Price_Check_Output.append("np.all((np.round(df2_sub['MAC1026_UNIT_PRICE'], 4) <= np.round(df2_sub['New_Price'], 4)) | (df2_sub['PRICE_MUTABLE'] == 0))")
        if np.all((np.round(df2_sub['MAC1026_UNIT_PRICE'], 4) <= np.round(df2_sub['Final_Price'], 4)) | (df2_sub['PRICE_MUTABLE'] == 0)):
            pass
        else:
            qa_Price_Check_Output.append("np.all((np.round(df2_sub['MAC1026_UNIT_PRICE'], 4) <= np.round(df2_sub['Final_Price'], 4)) | (df2_sub['PRICE_MUTABLE'] == 0))")


        if len(qa_Price_Check_Output) !=0:
            print("Failing QA : ")
            print(qa_Price_Check_Output)
            assert False, qa_Price_Check_Output
        else: 
            print("QA PASS")

        # df2_sub = df2[df2['MEASUREMENT'].isin(['M30','MAIL'])]
        # assert np.all((np.round(df2_sub['MAC1026_UNIT_PRICE'], 4) <= np.round(df2_sub['New_Price'], 4)) | (df2_sub['PRICE_MUTABLE'] == 0))
        # assert np.all((np.round(df2_sub['MAC1026_UNIT_PRICE'], 4) <= np.round(df2_sub['Final_Price'], 4)) | (df2_sub['PRICE_MUTABLE'] == 0))

        # Test: that output file Model_0.2_Performance looks okay

        # Test: that output file NEW_PRICES_MONTHLY_PROJECTIONS looks okay

        # Test: that output file OLD_PRICES_MONTHLY_PROJECTIONS looks okay

        # Test: that NEW_PRICES_* and OLD_PRICES_* are consistent where they should be

        # Test: Spend_data_* looks okay

        # Test: Lambda_Performance looks okay

        # Test: Model_Performance and Lambda_Performance are kind of similar

        # Test: all vcml on the vcml file have a vcml on the output file (no vcml is getting drop)
    except Exception as e:
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'QA File Price_Check_Output', repr(e), error_loc)
        raise e


def qa_price_output(
    params_in: str,
    lp_data_output_df_out: OutputPath('pickle'),
    lp_with_final_prices_out: OutputPath('pickle'),
    output_cols_out: OutputPath('pickle'),
    tolerance:float = 0.005):
    '''
    # Test: make sure no VCML get dropped on the price recommendations
    # Test: do prices in "formal" output file, Price_Changes, match prices in lp_data_output (mainly checking that the nanmin() worked:)
    '''
    import os
    import io
    import pickle
    import pandas as pd
    import numpy as np
    import BQ
    from google.cloud import storage
    from types import ModuleType
    from util_funcs import write_params, write_to_bq, read_BQ_data
    if isinstance(params_in, ModuleType):
        p = params_in
    else:
        write_params(params_in)
        import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df
    import util_funcs as uf

    from CPMO_shared_functions import update_run_status
    try:
        FILE_OUTPUT_PATH = p.FILE_OUTPUT_PATH

        if False:
            lp_data_output_df = read_BQ_data(
                BQ.lp_total_output_df,
                project_id = p.BQ_OUTPUT_PROJECT_ID,
                dataset_id = p.BQ_OUTPUT_DATASET,
                table_id = "Total_Output_subgroup",
                run_id = p.AT_RUN_ID,
                client = ', '.join(sorted(p.CUSTOMER_ID)),
                period = p.TIMESTAMP,
                output = True)
        else:
            try:
                lp_data_output_df = pd.read_csv(p.FILE_OUTPUT_PATH + p.TOTAL_OUTPUT, dtype = p.VARIABLE_TYPE_DIC)
            except:
                print('Missing Total_Output_ file: ' + p.FILE_OUTPUT_PATH + p.TOTAL_OUTPUT + '  Need to run lp with WRITE_OUTPUT = True')
    #     try:
    # #TODO: use standardize_df() and capitolize all the fields below
    #         lp_data_output_df = pd.read_csv(p.FILE_OUTPUT_PATH + p.TOTAL_OUTPUT, dtype = p.VARIABLE_TYPE_DIC)
    #     except:
    #         # TODO: WHAT HAPPEND TO PILOT?
    #         print('Missing Total_Output_ file: ' + p.FILE_OUTPUT_PATH + p.TOTAL_OUTPUT + '.csv  Need to run lp with WRITE_OUTPUT = True')

        try:
            # TODO: ADD TO THE PARAMETERS FILE
            fname = p.PRICE_CHANGES_OUTPUT_FILE
            fpath = os.path.join(p.FILE_OUTPUT_PATH, fname)
            price_changes = pd.read_excel(fpath, sheet_name='RXC_MACLISTS')
        except Exception as e:
            print('Missing Price_Changes: ' + fpath + ' Need to run lp with WRITE_OUTPUT = True')
            print(e)

        # to merge these two together:
        lp_data_output_df.rename(columns={'PKG_SZ':'GPPC','NDC':'NDC11'}, inplace=True)
        lp_data_output_df['MACLIST'] = p.APPLY_VCML_PREFIX + lp_data_output_df['MAC_LIST'].astype(str)
        lp_data_output_df['GPPC'] = lp_data_output_df['GPPC'].astype(str)
        lp_data_output_df.loc[lp_data_output_df.GPPC=='0.0', 'GPPC'] = '********'

        # Test: make sure no VCML get dropped on the price recommendations
        # VCML file. The VCML reference includes the list of unique VCMLs for each client
 
        vcml_df = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.VCML_REFERENCE_FILE, dtype=p.VARIABLE_TYPE_DIC)
        vcml_df = standardize_df(vcml_df)

        # Test on number of VCMLs
        if p.CLIENT_TYPE == 'MEDD' and p.FULL_YEAR and not p.EGWP:
            #For MEDD new year pricing the current vcmls is MAC_LIST_OLD  
            lp_data_output_df['MACLIST_OLD'] = p.APPLY_VCML_PREFIX + lp_data_output_df['MAC_LIST_OLD'].astype(str)
            lp_data_output_df_vcml = set(lp_data_output_df['MACLIST_OLD'])
        else:
            lp_data_output_df_vcml = set(lp_data_output_df['MACLIST'])
            
        flagged_vcmls = p.IGNORED_VCMLS 
       
        if len(set(['34', '35', '36', '37', '38']) - set(vcml_df.VCML_ID.astype(str).str[len(p.CUSTOMER_ID[0])+3:])) == 0:
            flagged_vcmls.append('9')
        
        if len(set(['177', '199']) - set(vcml_df.VCML_ID.astype(str).str[len(p.CUSTOMER_ID[0])+3:])) == 0:
            flagged_vcmls.append('1')
            
        if (len(set(['377', '399']) - set(vcml_df.VCML_ID.astype(str).str[len(p.CUSTOMER_ID[0])+3:])) == 0):
            flagged_vcmls.append('3')
            
        vcml = f"MAC{uf.get_formatted_string(p.CUSTOMER_ID)[1:-1]}"
        all_vcml = set(vcml_df.loc[~(vcml_df['VCML_ID'].str[len(vcml):].isin(flagged_vcmls)),'VCML_ID'])
        
        if all_vcml == lp_data_output_df_vcml:
            print('All VCML_IDs are included in the TOTAL_OUTPUT file')
        
        # Test: do prices in "formal" output file, Price_Changes, match prices in lp_data_output (mainly checking that the nanmin() worked:)
        # allow QA checks below to work with UNC_OPT = False
        for column in ['PRICE_CHANGED_UC', 'RAISED_PRICE_UC']:
            if column not in lp_data_output_df.columns:
                lp_data_output_df[column] = False
        for column in ['PRE_UC_MAC_PRICE', 'MAC_PRICE_UPPER_LIMIT_UC', 'VCML_AVG_AWP', 'VCML_AVG_CLAIM_QTY']:
            if column not in lp_data_output_df.columns:
                lp_data_output_df[column] = np.nan

        price_changes["GPI"] = price_changes["GPI"].astype("str")
        price_changes['GPI'] = price_changes['GPI'].str.zfill(14)
          
        if 'GNRC_CurrentMAC' not in price_changes.columns:
            price_changes.rename(columns={'MACPRC': 'GNRC_MACPRC',
                                    'Current MAC': 'GNRC_CurrentMAC'}, inplace=True)

        gnrc_lp_with_final_prices = lp_data_output_df[lp_data_output_df['BG_FLAG']=='G'].merge(
            price_changes[['MACLIST','GPI','NDC11','GPPC','GNRC_MACPRC','GNRC_CurrentMAC']],
            how='inner',on=['MACLIST','GPI','NDC11','GPPC'])
        gnrc_lp_with_final_prices.rename(columns={'GNRC_MACPRC':'MACPRC','GNRC_CurrentMAC':'Current MAC'},inplace=True)
        if p.BRAND_OPT:
            brnd_lp_with_final_prices = lp_data_output_df[lp_data_output_df['BG_FLAG']=='B'].merge(
                price_changes[['MACLIST','GPI','NDC11','GPPC','BRND_MACPRC','BRND_CurrentMAC']],
                how='inner',on=['MACLIST','GPI','NDC11','GPPC'])
            brnd_lp_with_final_prices.rename(columns={'BRND_MACPRC':'MACPRC','BRND_CurrentMAC':'Current MAC'},inplace=True)
            lp_with_final_prices = pd.concat([gnrc_lp_with_final_prices,brnd_lp_with_final_prices])
        else:
            lp_with_final_prices = gnrc_lp_with_final_prices.copy()
            
        output_cols = [
                'CLIENT','BREAKOUT','REGION','MEASUREMENT','BG_FLAG','GPI','NDC11','MACLIST','GPPC','CHAIN_GROUP', 'CHAIN_SUBGROUP','PRICE_MUTABLE',
                'QTY','CLAIMS','CLAIMS_PROJ_EOY','PRICING_CLAIMS_PROJ_EOY','GPI_NDC','PRICING_QTY_PROJ_EOY',
                'MAC1026_UNIT_PRICE','AVG_AWP','VCML_AVG_AWP','VCML_AVG_CLAIM_QTY','RAISED_PRICE_UC',
                'CURRENT_MAC_PRICE', 'MAC_PRICE_UNIT_ADJ','OLD_MAC_PRICE','PRICE_OVRD_AMT','lb','ub','lb_name','ub_name','lower_bound_ordered',
                'upper_bound_ordered','EFF_UNIT_PRICE_new','Final_Price','MACPRC', 'Current MAC', 'GOODRX_UPPER_LIMIT','ZBD_UPPER_LIMIT',
                'PRICING_AVG_AWP', 'PRICING_QTY', 'PRICING_CLAIMS', 'PRICING_PRICE_REIMB_CLAIM', 'UNC_OVRD_AMT']
            
        if p.UNC_OPT:
            output_cols += [
                    'PRE_UC_MAC_PRICE','PRICE_CHANGED_UC','MAC_PRICE_UPPER_LIMIT_UC',
                    'IS_MAINTENANCE_UC', 'MATCH_VCML', 'PCT90_UCAMT_UNIT', 'PCT50_UCAMT_UNIT']
        if p.INTERCEPTOR_OPT:
            if p.GO_LIVE.year >= 2025: 
                output_cols += [
                    'CS_PHARMACY_RATE','VENDOR_PRICE','VENDOR_AVAILABLE','PHARMACY_GER','VENDOR_CONFLICT', 
                    'INTERCEPT_REASON','IMMUTABLE_REASON','CURRENT_KEEP_SEND','DESIRE_KEEP_SEND','ACTUAL_KEEP_SEND','INTERCEPT_HIGH','INTERCEPT_LOW']
            else:
                output_cols += [
                    'VCML_PHARMACY_RATE','VENDOR_PRICE','VENDOR_AVAILABLE','PHARMACY_GER','VENDOR_CONFLICT', 
                    'INTERCEPT_REASON','IMMUTABLE_REASON','CURRENT_KEEP_SEND','DESIRE_KEEP_SEND','ACTUAL_KEEP_SEND','INTERCEPT_HIGH','INTERCEPT_LOW']
        if p.TRUECOST_CLIENT:
            output_cols += ['IMMUTABLE_REASON', 'SOFT_CONST_BENCHMARK_PRICE', 'DISP_FEE', 'TARGET_DISP_FEE']
            
        lp_with_final_prices = lp_with_final_prices[output_cols]

        qa_price_output_err_list = []
        
        if  ((lp_with_final_prices['Final_Price'] - lp_with_final_prices['MACPRC']).abs() > tolerance).sum() > 0:
            missmatchedPrices = ((lp_with_final_prices['Final_Price'] - lp_with_final_prices['MACPRC']).abs() > tolerance).sum()
            print('')
            print('*Warning: there are {} missmatched prices between Total_Output and Price_Changes. Check the missmatched_prices_REPORT'.format(missmatchedPrices))
            miss = lp_with_final_prices[((lp_with_final_prices['Final_Price'] - lp_with_final_prices['MACPRC']).abs() > tolerance)]
            if len(miss) > 0:
                if False:#p.WRITE_TO_BQ:
                    write_to_bq(
                        miss,
                        project_output = p.BQ_OUTPUT_PROJECT_ID,
                        dataset_output = p.BQ_OUTPUT_DATASET,
                        table_id = "missmatched_prices_REPORT",
                        client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                        timestamp_param = p.TIMESTAMP,
                        run_id = p.AT_RUN_ID,
                        schema = None
                    )
                else:
                    miss.to_csv(FILE_OUTPUT_PATH + 'missmatched_prices_REPORT.csv',index=False)
            if len(miss) == 0:
                qa_price_output_err_list.append('There are {} missmatched prices between Total_Output and Price_Changes. Check the missmatched_prices_REPORT'.format(missmatchedPrices))
            
        else:
            print('')
            print('No missmatched prices between Total_Output and Price_Changes.')
        
        # same dataset
        # Test: PHARMACY_EXCLUSION flag

        if p.PHARMACY_EXCLUSION:
            lp_with_final_prices_sub = lp_with_final_prices[lp_with_final_prices['CHAIN_GROUP'].isin(p.LIST_PHARMACY_EXCLUSION)]
            lp_with_final_prices_sub['MIN_PRICE'] = lp_with_final_prices_sub[['MAC_PRICE_UNIT_ADJ','MAC1026_UNIT_PRICE']].max(axis=1)

            if ((lp_with_final_prices_sub['Final_Price'] - lp_with_final_prices_sub['MIN_PRICE']) > tolerance).sum() > 0:
                missmatchedPrices = ((lp_with_final_prices_sub['Final_Price'] - lp_with_final_prices_sub['MIN_PRICE']).abs() > tolerance).sum()

                print('')
                print('*Warning: there are {} prices that are not following the PHARMACY_EXCLUSION rule'.format(missmatchedPrices))
                miss = lp_with_final_prices_sub[((lp_with_final_prices_sub['Final_Price'] - lp_with_final_prices_sub['MIN_PRICE']).abs() > tolerance)]
                # TODO: THIS FILE DID NOT EXIST BEFORE CHEK THE BQ PARAMETERS
                if len(miss) > 0:
                    if False:#p.WRITE_TO_BQ:
                        write_to_bq(
                            miss,
                            project_output = p.BQ_OUTPUT_PROJECT_ID,
                            dataset_output = p.BQ_OUTPUT_DATASET,
                            table_id = "missmatched_prices_REPORT",  # TODO: find the right table for this
                            client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                            timestamp_param = p.TIMESTAMP,
                            run_id = p.AT_RUN_ID,
                            schema = None
                        )
                    else:
                        miss.to_csv(FILE_OUTPUT_PATH + 'PHARMACY_EXCLUSION_REPORT_' + p.TIMESTAMP + '.csv', index=False)
                if len(miss) == 0:
                    qa_price_output_err_list.append('There are {} prices that are not following the PHARMACY_EXCLUSION rule'.format(missmatchedPrices))
            else:
                print('')
                print('No prices break the PHARMACY_EXCLUSION rule.')

        # same dataset:
        # Test: whether existing drug prices satisfy upper and lower bound constraints
        
        lp_with_final_prices["MACLIST_BG_GPI"] = lp_with_final_prices["MACLIST"] + lp_with_final_prices["BG_FLAG"] + lp_with_final_prices["GPI"]

        if p.FLOOR_PRICE:
            floor_gpis = pd.read_csv(p.FILE_INPUT_PATH + p.FLOOR_GPI_LIST, dtype = p.VARIABLE_TYPE_DIC)[['GPI']]
            if 'BG_FLAG' not in floor_gpis.columns:
                floor_gpis['BG_FLAG'] = 'G'
            floor_gpis['BG_GPI'] = floor_gpis['BG_FLAG'] + floor_gpis['GPI']
        else:
            floor_gpis = pd.DataFrame({'BG_GPI': []})

        mac_price_override = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_PRICE_OVERRIDE_FILE, dtype = p.VARIABLE_TYPE_DIC))
        mac_price_override["MACLIST_BG_GPI"] = mac_price_override['VCML_ID'] + mac_price_override['BG_FLAG'] + mac_price_override['GPI']
        
        if p.PRICE_OVERRIDE:
            price_override =standardize_df(pd.read_csv(os.path.join(p.FILE_INPUT_PATH, p.PRICE_OVERRIDE_FILE), dtype = p.VARIABLE_TYPE_DIC))
            price_override["MACLIST_BG_GPI"] = price_override['VCML_ID'] + price_override['BG_FLAG'] + price_override['GPI']
        
        if  (~lp_with_final_prices['Final_Price'].between(lp_with_final_prices['lb']-tolerance,lp_with_final_prices['ub']+tolerance)).sum() > 0:
            miss = lp_with_final_prices[(~lp_with_final_prices['Final_Price'].between(lp_with_final_prices['lb']-tolerance,lp_with_final_prices['ub']+tolerance))]
            outofbounds = len(miss)
            miss['REASON_CODE'] = 'unknown'

            #TODO: add the floor price to the report
            miss.loc[(miss['BG_FLAG']+miss['GPI']).isin(floor_gpis['BG_GPI']),'REASON_CODE'] = 'Floor GPI'
            miss.loc[miss['MACLIST_BG_GPI'].isin(mac_price_override['MACLIST_BG_GPI']),'REASON_CODE'] = 'Mac Price Override'
            miss = miss.merge(mac_price_override[['MACLIST_BG_GPI','PRICE_OVRD_AMT']], on = 'MACLIST_BG_GPI', how='left')
            if p.PRICE_OVERRIDE:
                miss.loc[miss['MACLIST_BG_GPI'].isin(price_override['MACLIST_BG_GPI']),'REASON_CODE'] = 'Price Override'
                miss = miss.merge(price_override[['MACLIST_BG_GPI','PRICE_OVRD_AMT']], on = 'MACLIST_BG_GPI', how='left')
            
            unknowns = len(miss[miss['REASON_CODE'] == 'unknown'])

            print('*Warning: there are {} prices out of lp constraints, {} of them with unknown reasons. Check the price_bounds_REPORT'.format(outofbounds,unknowns))
            if unknowns:
                if False:#p.WRITE_TO_BQ:
                    write_to_bq(
                        miss,
                        project_output = p.BQ_OUTPUT_PROJECT_ID,
                        dataset_output = p.BQ_OUTPUT_DATASET,
                        table_id = "price_bounds_REPORT",
                        client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                        timestamp_param = p.TIMESTAMP,
                        run_id = p.AT_RUN_ID,
                        schema = None
                    )
                else:
                    miss.to_csv(FILE_OUTPUT_PATH + 'price_bounds_REPORT.csv',index=False)
            if unknowns != 0:
                qa_price_output_err_list.append('There are {} prices out of lp constraints for unknown reasons. Check the price_bounds_REPORT'.format(unknowns))

        else:
            print('')
            print('No prices out of lp constraints.')
            
       ##### TRUECOST SPECIFIC QAs #####
        if p.TRUECOST_CLIENT:
            # Test: check that no price changes to immutable rows are applied to truecost clients
            tru_immutable_check = lp_with_final_prices[lp_with_final_prices['PRICE_MUTABLE'] == 0]
            tru_immutable_check['REASON_CODE'] = 'unknown'
            
            tru_immutable_check.loc[~((abs(tru_immutable_check['MACPRC'] - tru_immutable_check['Current MAC']) <= tolerance) |
                                      (tru_immutable_check['MACPRC'].isna()))]
            
            # Check that no overrides are being applied to specialty drugs
            specialty_drugs = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.SPECIALTY_EXCLUSION_FILE).drop(columns=['CLIENT', 'REGION'])
            specialty_drugs['SPEC_IDX'] = 1
            tru_immutable_check = tru_immutable_check.merge(specialty_drugs, on = ['GPI', 'BG_FLAG'], how='left', validate='m:1')
            tru_immutable_check.loc[(abs(tru_immutable_check['MACPRC'] - tru_immutable_check['PRICE_OVRD_AMT']) <= tolerance) & 
                                    ((tru_immutable_check['SPEC_IDX'].isna()) | (tru_immutable_check['SPEC_IDX'] != 1)), 
                                         'REASON_CODE'] = 'Non-Specialty Override'
            
            tru_immutable_report = tru_immutable_check.loc[tru_immutable_check['REASON_CODE'] == 'unknown']
            if len(tru_immutable_check) > 0:
                if len(tru_immutable_report) > 0:
                    tru_immutable_check.to_csv(FILE_OUTPUT_PATH + 'TRUE_PRICE_IMMUTABLE_DIFF_REPORT.csv')
                    qa_price_output_err_list.append(f'There are {len(tru_immutable_check)} prices for truecost clients that were set as immutable, but still have price changes, {len(tru_immutable_report)} of them with unknown reasons. Check the TRUE_PRICE_IMMUTABLE_DIFF_REPORT.')
                elif len(tru_immutable_check) > 0:
                    print(f'*Warning: There are {len(tru_immutable_check)} prices for truecost clients that were set as immutable that did have price changes, 0 with unknown reasons.')
                        
                        
            # Test: check that no forced floors are applied to truecost clients
            tru_floor_check = lp_data_output_df.loc[(lp_data_output_df['IMMUTABLE_REASON'] == 'FLOOR_GPI'), 
                                                    [col for col in output_cols if col not in ['Current MAC', 'MACPRC']]]
            
            if len(tru_floor_check) > 0:
                tru_floor_check.to_csv(FILE_OUTPUT_PATH + 'TRUE_FLOOR_PRICE_REPORT.csv')
                qa_price_output_err_list.append(f'There are {len(tru_floor_check)} prices for truecost clients that have a forced floor price. Check the TRUE_FLOOR_PRICE_REPORT.')
            
            
            #Test: Make sure for TrueCost client there is no non-expected price increase over TRUECOST_UPPER_MULTIPLIER * NET_COST_GUARANTEE_UNIT
            #      NET_COST_GUARANTEE_UNIT value is in the column SOFT_CONST_BENCHMARK_PRICE 
            over_benchmark_cap_check = lp_data_output_df.loc[((lp_data_output_df['BG_FLAG']=='G') & 
                                                              (lp_data_output_df['Final_Price'] > p.TRUECOST_UPPER_MULTIPLIER_GNRC * lp_data_output_df['SOFT_CONST_BENCHMARK_PRICE']+tolerance)) |
                                                              ((lp_data_output_df['BG_FLAG']=='B') & 
                                                              (lp_data_output_df['Final_Price'] > p.TRUECOST_UPPER_MULTIPLIER_BRND * lp_data_output_df['SOFT_CONST_BENCHMARK_PRICE']+tolerance)), [col for col in output_cols if col not in ['Current MAC', 'MACPRC']]]
            
            over_benchmark_cap_check = over_benchmark_cap_check.loc[~((lp_data_output_df['BG_FLAG']=='G') & (over_benchmark_cap_check['QTY'] != 0) 
                                                                      & (over_benchmark_cap_check['Final_Price'] + 
                                                                         over_benchmark_cap_check['DISP_FEE'] / over_benchmark_cap_check['QTY']  <= 
                                                                         p.TRUECOST_UPPER_MULTIPLIER_GNRC * (over_benchmark_cap_check['SOFT_CONST_BENCHMARK_PRICE'] +
                                                                                                        over_benchmark_cap_check['TARGET_DISP_FEE'] / over_benchmark_cap_check['QTY']) + tolerance)), :]

            over_benchmark_cap_check = over_benchmark_cap_check.loc[~((lp_data_output_df['BG_FLAG']=='B') & (over_benchmark_cap_check['QTY'] != 0) 
                                                                      & (over_benchmark_cap_check['Final_Price'] + 
                                                                         over_benchmark_cap_check['DISP_FEE'] / over_benchmark_cap_check['QTY']  <= 
                                                                         p.TRUECOST_UPPER_MULTIPLIER_BRND * (over_benchmark_cap_check['SOFT_CONST_BENCHMARK_PRICE'] +
                                                                                                        over_benchmark_cap_check['TARGET_DISP_FEE'] / over_benchmark_cap_check['QTY']) + tolerance)), :]
            
            over_benchmark_cap_check['REASON_CODE'] = 'unknown'
            over_benchmark_cap_check.loc[(over_benchmark_cap_check['BG_FLAG'] == 'G') & 
                                         (over_benchmark_cap_check['MAC1026_UNIT_PRICE'] > p.TRUECOST_UPPER_MULTIPLIER_GNRC * over_benchmark_cap_check['SOFT_CONST_BENCHMARK_PRICE']) & 
                                         (over_benchmark_cap_check['MAC1026_UNIT_PRICE'] - tolerance < over_benchmark_cap_check['Final_Price']) &
                                         (over_benchmark_cap_check['Final_Price'] < over_benchmark_cap_check['MAC1026_UNIT_PRICE'] + tolerance), 
                                         'REASON_CODE'] = 'MAC1026'
            
            over_benchmark_cap_check.loc[(over_benchmark_cap_check['IMMUTABLE_REASON'] == 'SPECIALTY_EXCLUSION')
                                         & (over_benchmark_cap_check['QTY'] == 0), 
                                         'REASON_CODE'] = 'SPECIALTY_EXCLUSION'
            
            over_benchmark_cap_report = over_benchmark_cap_check.loc[over_benchmark_cap_check['REASON_CODE'] == 'unknown']
            
            if len(over_benchmark_cap_check[over_benchmark_cap_check['PRICE_MUTABLE'] == 1]) > 0:
                if len(over_benchmark_cap_report[over_benchmark_cap_report['PRICE_MUTABLE'] == 1]) > 0:
                    over_benchmark_cap_report[over_benchmark_cap_report['PRICE_MUTABLE'] == 1].to_csv(FILE_OUTPUT_PATH + 'PRICE_OVER_BENCHMARK_CAP_REPORT.csv')
                    qa_price_output_err_list.append(f'''There are {len(over_benchmark_cap_check[over_benchmark_cap_check['PRICE_MUTABLE'] == 1])} price points above {p.TRUECOST_UPPER_MULTIPLIER_GNRC} * benchmark generic price or {p.TRUECOST_UPPER_MULTIPLIER_BRND} * benchmark brand price, {len(over_benchmark_cap_report[over_benchmark_cap_report['PRICE_MUTABLE'] == 1])} of them with unknown reasons. Check the PRICE_OVER_BENCHMARK_CAP_REPORT''')
                else:
                    print(f'''*Warning: There are {len(over_benchmark_cap_check[over_benchmark_cap_check['PRICE_MUTABLE'] == 1])} price points above {p.TRUECOST_UPPER_MULTIPLIER_GNRC} * benchmark generic price or {p.TRUECOST_UPPER_MULTIPLIER_BRND} * benchmark brand price, 0 with unknown reasons''')
                    
            if len(over_benchmark_cap_check[over_benchmark_cap_check['PRICE_MUTABLE'] == 0]) > 0:
                if len(over_benchmark_cap_report[over_benchmark_cap_report['PRICE_MUTABLE'] == 0]) > 0:
                    over_benchmark_cap_report[over_benchmark_cap_report['PRICE_MUTABLE'] == 0].to_csv(FILE_OUTPUT_PATH + 'PRICE_OVER_BENCHMARK_CAP_REPORT_Immutable.csv')
                    print(f'''Warning: There are {len(over_benchmark_cap_check[over_benchmark_cap_check['PRICE_MUTABLE'] == 0])} price points above {p.TRUECOST_UPPER_MULTIPLIER_GNRC} * benchmark generic price or {p.TRUECOST_UPPER_MULTIPLIER_BRND} * benchmark brand price that are IMMUTABLE, {len(over_benchmark_cap_report[over_benchmark_cap_report['PRICE_MUTABLE'] == 0])} of them with unknown reasons. Check the PRICE_OVER_BENCHMARK_CAP_REPORT''')
                else:
                    print(f'''*Warning: There are {len(over_benchmark_cap_check[over_benchmark_cap_check['PRICE_MUTABLE'] == 0])} price points above {p.TRUECOST_UPPER_MULTIPLIER_GNRC} * benchmark generic price or {p.TRUECOST_UPPER_MULTIPLIER_BRND} * benchmark brand price that are IMMUTABLE, 0 with unknown reasons''')

        # Test: Make sure for TrueCost client the price increases due to MAC1026 are reasonable
            MAX_AVERAGE_INCREASE = 1
            price_check = lp_data_output_df.loc[((lp_data_output_df['BG_FLAG']=='G') & 
                                                              (lp_data_output_df['Final_Price'] > p.TRUECOST_UPPER_MULTIPLIER_GNRC * lp_data_output_df['SOFT_CONST_BENCHMARK_PRICE']+tolerance))]
            mac1026_price_check = price_check.loc[(price_check['lb_name'] == 'floor 1026') 
                                                  & (price_check['MAC1026_UNIT_PRICE'] > price_check['SOFT_CONST_BENCHMARK_PRICE'])]
            mac1026_price_check['PERCENT_INCREASE'] = ((mac1026_price_check['MAC1026_UNIT_PRICE'] - mac1026_price_check['SOFT_CONST_BENCHMARK_PRICE']) / mac1026_price_check['SOFT_CONST_BENCHMARK_PRICE'])
            
            weighted_average_increase = (mac1026_price_check['PERCENT_INCREASE'] * mac1026_price_check['CLAIMS']).sum() / mac1026_price_check['CLAIMS'].sum()
            
            if weighted_average_increase > MAX_AVERAGE_INCREASE:
                mac1026_price_check.to_csv(FILE_OUTPUT_PATH + 'MAC1026_PRICE_INCREASE_REPORT.csv')
                # qa_price_output_err_list.append(f'The claim weighted average percentage increase due to MAC1026 is {weighted_average_increase:.2f}, which is greater than the allowed maximum of {MAX_AVERAGE_INCREASE}. Check the MAC1026_PRICE_INCREASE_REPORT.')
            else:
                print(f'*Warning: There are {len(mac1026_price_check)} price point above {p.TRUECOST_UPPER_MULTIPLIER_GNRC} * benchmark generic price due to MAC1026 Floors, 0 above the allowed maximum of {MAX_AVERAGE_INCREASE}x.')

        # Test: Checks Adjudication at GPI level is below Guaranteed price * cap
        if p.TRUECOST_CLIENT:
            claims_prices = lp_data_output_df.loc[
                (lp_data_output_df['CLAIMS'] > 0)].copy()
            
            claims_prices = claims_prices[claims_prices['BG_FLAG'] == 'B'] # remove once we know generic adj cap
            
            claims_prices.loc[claims_prices['NADAC'] == 0,'NADAC'] = None
            claims_prices.loc[claims_prices['WAC'] == 0,'WAC'] = None
            
            claims_prices['TARG_INGCOST_NW_YTD'] = claims_prices[['TARG_INGCOST_ADJ','NADAC_WAC_YTD']].max(axis=1)
            claims_prices['TARG_INGCOST_NW_LAG'] = np.nanmax([claims_prices['TARG_INGCOST_ADJ_PROJ_LAG'],
                                                                     claims_prices.NADAC.combine_first(claims_prices.WAC) * \
                                                                     claims_prices.QTY_PROJ_LAG],  axis=0)
            claims_prices['TARG_INGCOST_NW_EOY'] = np.nanmax([claims_prices['TARG_INGCOST_ADJ_PROJ_EOY'],
                                                                     claims_prices.NADAC.combine_first(claims_prices.WAC) * \
                                                                     claims_prices.QTY_PROJ_EOY], axis=0)
            
            if p.FULL_YEAR:
                claims_prices['GUARANTEED_SCRIPT_PRICE'] = claims_prices['TARG_INGCOST_NW_EOY'] + claims_prices['TARGET_DISP_FEE_PROJ_EOY']
                claims_prices['ACTUAL_SCRIPT_PRICE'] = claims_prices['Price_Effective_Reimb_Proj'] + claims_prices['DISP_FEE_PROJ_EOY']
            else:
                claims_prices['GUARANTEED_SCRIPT_PRICE'] = claims_prices['TARG_INGCOST_NW_YTD'] + claims_prices['TARG_INGCOST_NW_LAG'] + claims_prices['TARG_INGCOST_NW_EOY'] + claims_prices['TARGET_DISP_FEE'] + claims_prices['TARGET_DISP_FEE_PROJ_LAG'] + claims_prices['TARGET_DISP_FEE_PROJ_EOY']
                claims_prices['ACTUAL_SCRIPT_PRICE'] = claims_prices['PRICE_REIMB'] + claims_prices['LAG_REIMB'] + claims_prices['Price_Effective_Reimb_Proj'] + claims_prices['DISP_FEE'] + claims_prices['DISP_FEE_PROJ_LAG'] + claims_prices['DISP_FEE_PROJ_EOY']
            
            gpi_agg_df = claims_prices.groupby(['GPI', 'BG_FLAG']).agg({
                'ACTUAL_SCRIPT_PRICE': 'sum',
                'GUARANTEED_SCRIPT_PRICE': 'sum',
                'IMMUTABLE_REASON':'first'
            }).reset_index()
            gpi_agg_df['RATIO'] = gpi_agg_df['ACTUAL_SCRIPT_PRICE'] / gpi_agg_df['GUARANTEED_SCRIPT_PRICE']
            
            ratio_tolerance = .00001
            price_over_cap =  gpi_agg_df.loc[((gpi_agg_df['BG_FLAG'] == 'G') & (gpi_agg_df['RATIO'] - ratio_tolerance > 1.499)) | 
                                                  ((gpi_agg_df['BG_FLAG'] == 'B') & (gpi_agg_df['RATIO'] - ratio_tolerance  > 1.149))]
                
            price_over_cap_mutable = price_over_cap[price_over_cap['IMMUTABLE_REASON'] != 'NDC_PRICING'].drop(columns=['IMMUTABLE_REASON'])
            if len(price_over_cap_mutable) > 0:
                price_over_cap_mutable.to_csv(FILE_OUTPUT_PATH + 'PRICE_OVER_ADJ_CAP_REPORT.csv')
                qa_price_output_err_list.append(f'''There are {len(price_over_cap_mutable)} NDC-* GPI(s) where the total reimbursement for the year exceeds the adjudication cap over the guaranteed price. Check the PRICE_OVER_ADJ_CAP_REPORT.''')
            else:
                print('All NDC-* GPIs are within the adjudication cap for the full year.')
                
            price_over_cap_ndc = price_over_cap[price_over_cap['IMMUTABLE_REASON'] == 'NDC_PRICING']
            if len(price_over_cap_ndc) > 0:
                price_over_cap_ndc.to_csv(FILE_OUTPUT_PATH + 'PRICE_OVER_ADJ_CAP_NDC_REPORT.csv')
                print(f'''Warning: There are {len(price_over_cap_ndc)} non NDC-* GPI(s) where the total reimbursement for the year exceeds the adjudication cap over the guaranteed price. Check the PRICE_OVER_ADJ_CAP_NDC_REPORT.''')
        ##### END OF TRUECOST SPECIFIC QAs #####

        # Test: Ensure no brand prices are modified due to MAC1026 floors
        if p.BRAND_OPT:
            brand_price_check = lp_data_output_df.loc[
                (lp_data_output_df['BG_FLAG'] == 'B') & 
                (lp_data_output_df['lb_name'] == 'floor 1026') &
                (lp_data_output_df['MAC1026_UNIT_PRICE'] == lp_data_output_df['Final_Price'])]

            if len(brand_price_check) > 0:
                brand_price_check.to_csv(FILE_OUTPUT_PATH + 'BRAND_PRICE_DUE_TO_FLOORS_REPORT.csv')
                qa_price_output_err_list.append(
                    f'There are {len(brand_price_check)} brand price points modified due to MAC1026 floors. Check the BRAND_PRICE_DUE_TO_FLOORS_REPORT.')
            else:
                print('No brand price points were modified due to MAC1026 floors.')

        # Test: Make sure Parity Price Difference collars are respected
        if 'CVSSP' in lp_data_output_df['CHAIN_SUBGROUP'].unique():
            # Isolate prices of interest, currently where chain_subgroup is either CVS or CVSSP
            parity_collar_check = lp_data_output_df[['CLIENT','BREAKOUT','REGION','MEASUREMENT', 'BG_FLAG', 'CHAIN_SUBGROUP',
                                                     'GPI_NDC','OLD_MAC_PRICE','Final_Price', 'PRICE_MUTABLE']].loc[lp_data_output_df['CHAIN_SUBGROUP'].isin(['CVS','CVSSP'])]

            # Pivot to put CVS and CVSSP prices in the same row for price comparison
            parity_collar_check = parity_collar_check.pivot(index = ['CLIENT','BREAKOUT','REGION','MEASUREMENT','BG_FLAG','GPI_NDC'],
                                                            columns = 'CHAIN_SUBGROUP',
                                                            values = ['OLD_MAC_PRICE','Final_Price','PRICE_MUTABLE'])

            # Remove rows with prices of 0 or NaN
            parity_collar_check = parity_collar_check.loc[parity_collar_check[('OLD_MAC_PRICE','CVS')] > 0].dropna(subset = [('OLD_MAC_PRICE','CVS')])

            # Calculate price collar, based on settings in CPMO_parameters.py
            parity_collar_check['COLLAR_LOW_PRICE'] = round((parity_collar_check[('Final_Price','CVSSP')]-tolerance) * p.PARITY_PRICE_DIFFERENCE_COLLAR_LOW, 4)
            parity_collar_check['COLLAR_HIGH_PRICE'] = round((parity_collar_check[('Final_Price','CVSSP')]+tolerance) * p.PARITY_PRICE_DIFFERENCE_COLLAR_HIGH, 4)

            # Check if prices are within the collar
            parity_collar_check['REASON_CODE'] = 'unknown'
            parity_collar_check.loc[parity_collar_check[('Final_Price','CVS')].round(4) + tolerance < (parity_collar_check[('COLLAR_LOW_PRICE')] - tolerance), 
                                    'REASON_CODE'] = 'CVS price below price collar'
            parity_collar_check.loc[parity_collar_check[('Final_Price','CVS')].round(4) - tolerance > (parity_collar_check[('COLLAR_HIGH_PRICE')] + tolerance), 
                                    'REASON_CODE'] = 'CVS price above price collar'
            parity_collar_check.loc[(parity_collar_check[('Final_Price','CVS')].round(4) + tolerance >= (parity_collar_check['COLLAR_LOW_PRICE'] - tolerance)) & 
                                    (parity_collar_check[('Final_Price','CVS')].round(4) - tolerance <= (parity_collar_check['COLLAR_HIGH_PRICE'] + tolerance)), 
                                    'REASON_CODE'] = 'Within collar'
            parity_collar_check.loc[parity_collar_check['REASON_CODE'].isin(['CVS price below price collar', 'CVS price above price collar'] )
                                    & ((parity_collar_check[('PRICE_MUTABLE', 'CVS')]==0) & (parity_collar_check[('PRICE_MUTABLE', 'CVSSP')]==0)), 'REASON_CODE'] = 'Immutable parity pricing issue'
            parity_collar_check.loc[(parity_collar_check[('Final_Price','CVS')].isna() | parity_collar_check[('Final_Price','CVSSP')].isna()), 'REASON_CODE'] = 'Known MAC list length issue'
                                                            
            

            # Report output
            parity_collar_report = parity_collar_check.loc[parity_collar_check['REASON_CODE'] != 'Within collar']
            parity_collar_check = parity_collar_check.loc[~parity_collar_check['REASON_CODE'].isin(['Within collar', 'Immutable parity pricing issue', 'Known MAC list length issue'])]

            if parity_collar_check.shape[0] > 0:
                parity_collar_report.to_csv(FILE_OUTPUT_PATH + 'STATE_PARITY_COLLAR_REPORT.csv')
            elif parity_collar_check.shape[0] != 0:
                    qa_price_output_err_list.append( "CVS prices were set outside of price collars. Check STATE_PARITY_COLLAR_REPORT.csv.")
            elif parity_collar_report.shape[0] > 0:
                parity_collar_report.to_csv(FILE_OUTPUT_PATH + 'STATE_PARITY_COLLAR_REPORT.csv')
                print("*Warning: Known errors in state parity collars.")
            elif parity_collar_check.shape[0] == 0:
                print("All prices within state parity collars.")
            else:
                qa_price_output_err_list.append("Unhandled error in state parity collar check.")    
        
        if len(qa_price_output_err_list) > 0: 
            print("Failing QA : ")
            print(qa_price_output_err_list)
            assert False, qa_price_output_err_list 


        lp_data_output_df['AVG_QTY_PER_RXS_PROJ'] = np.where(lp_data_output_df['PRICING_CLAIMS_PROJ_EOY']==0, 0, lp_data_output_df.PRICING_QTY_PROJ_EOY/lp_data_output_df.PRICING_CLAIMS_PROJ_EOY)
        lp_data_output_df['AVG_QTY_PER_RXS'] = np.where(lp_data_output_df['PRICING_CLAIMS']==0, 0, lp_data_output_df.PRICING_QTY/lp_data_output_df.PRICING_CLAIMS)
        lp_data_output_df['ORIG_PRICE_CLAIM'] = lp_data_output_df.PRICING_PRICE_REIMB_CLAIM
        lp_data_output_df['FINAL_PRICE_CLAIM'] = lp_data_output_df.Final_Price*lp_data_output_df.AVG_QTY_PER_RXS
        lp_data_output_df['FINAL_PRICE_CLAIM_PROJ'] = lp_data_output_df.Final_Price*lp_data_output_df.AVG_QTY_PER_RXS_PROJ
        lp_data_output_df['PRICE_CHANGE_CLAIM'] = lp_data_output_df['FINAL_PRICE_CLAIM'] - lp_data_output_df['ORIG_PRICE_CLAIM']
        lp_data_output_df['PRICE_CHANGE_CLAIM_PROJ'] = lp_data_output_df['FINAL_PRICE_CLAIM_PROJ'] - lp_data_output_df['ORIG_PRICE_CLAIM']
        lp_data_output_df['PERCENT_CHANGE'] = (lp_data_output_df.Final_Price - lp_data_output_df.OLD_MAC_PRICE)  / (lp_data_output_df.OLD_MAC_PRICE )
        lp_data_output_df["CHANGE_BUCKET"] = 'Within range'
        lp_data_output_df['PRICE_TIER_BUCKET'] = np.nan
        lp_data_output_df['PRICE_TIER_REASON'] = 'non needed'

        with open(lp_data_output_df_out, 'wb') as f:
            lp_data_output_df.to_pickle(f)
        with open(lp_with_final_prices_out, 'wb') as f:
            lp_with_final_prices.to_pickle(f)
        with open(output_cols_out, 'wb') as f:
            pickle.dump(output_cols, f)
    except Exception as e:
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'QA Test Price Outputs', repr(e), error_loc)
        raise e

def qa_price_tiering_rules_REPORT(
    params_in: str,
    lp_data_output_df_in: InputPath('pickle'),
    tolerance: float = 0.005
):
    import os
    import pandas as pd
    import numpy as np
    from types import ModuleType
    from util_funcs import write_params, upload_blob
    import bisect
    if isinstance(params_in, ModuleType):
        p = params_in
    else:
        write_params(params_in)
        import CPMO_parameters as p
 
    from CPMO_shared_functions import update_run_status, standardize_df
 
    try:
        with open(lp_data_output_df_in, 'rb') as f:
            lp_data_output_df = pd.read_pickle(f)
 
        FILE_OUTPUT_PATH = p.FILE_OUTPUT_PATH  # os.path.join(p.PROGRAM_OUTPUT_PATH, 'QA', '')

        def tiered_price_check(row):
            idx = bisect.bisect_left(p.PRICE_BOUNDS_DF['upper_bound'], row.ORIG_PRICE_CLAIM)
            if row.PRICING_QTY_PROJ_EOY == 0.00:
                row['PRICE_TIER_BUCKET'] = 'Zero Projected Claims'
                if p.ZERO_QTY_TIGHT_BOUNDS:
                    row['CHANGE_BUCKET'] = "Within range"
                    if p.TIERED_PRICE_LIM:
                        if (row.PERCENT_CHANGE > (p.PRICE_BOUNDS_DF['max_percent_increase'][1] + tolerance)):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                    else:
                        if (row.PERCENT_CHANGE > (p.GPI_UP_FAC + tolerance)):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                    if (row.PERCENT_CHANGE < (- (p.GPI_LOW_FAC) - tolerance)) and (row.ORIG_PRICE_CLAIM > 0.00):
                        row['CHANGE_BUCKET'] = "Less than allowable price decrease"
                else:
                    row['CHANGE_BUCKET'] = 'No bounds for zero projected claims'
 
            # check increases
            elif row.PERCENT_CHANGE > 0:
                if p.TIERED_PRICE_LIM:
                    if idx == 0:  # Message for first bucket
                        row['PRICE_TIER_BUCKET'] = f'Less than or equal to ${p.PRICE_BOUNDS_DF["upper_bound"][0]}'
                        if row.PRICE_CHANGE_CLAIM > p.PRICE_BOUNDS_DF['max_dollar_increase'][idx] + (
                                tolerance * 10) or row.PERCENT_CHANGE > (
                                p.PRICE_BOUNDS_DF['max_percent_increase'][idx] + tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
                    elif idx == len(p.PRICE_BOUNDS_DF['upper_bound']) - 1:  # Message for last bucket
                        row['PRICE_TIER_BUCKET'] = f'Greater than ${p.PRICE_BOUNDS_DF["upper_bound"][idx - 1]}'
                        if row.PERCENT_CHANGE > (
                                p.PRICE_BOUNDS_DF["max_percent_increase"][idx] + tolerance) or row.PERCENT_CHANGE > (
                                p.PRICE_BOUNDS_DF['max_percent_increase'][idx] + tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
                    else:  # Message for other buckets
                        row[
                            'PRICE_TIER_BUCKET'] = f'${p.PRICE_BOUNDS_DF["upper_bound"][idx - 1]} - ${p.PRICE_BOUNDS_DF["upper_bound"][idx] - .01}'
                        if row.PRICE_CHANGE_CLAIM > p.PRICE_BOUNDS_DF['max_dollar_increase'][idx] + (tolerance * 10) \
                                or row.PERCENT_CHANGE > (p.PRICE_BOUNDS_DF['max_percent_increase'][idx] + tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
                elif p.HIGHEROF_PRICE_LIM:
                    row['mbr_bound'] = np.where(row.PRICING_QTY_PROJ_EOY > 0,
                                         (row.PRICING_CLAIMS / row.PRICING_QTY)
                                         * (row.PRICING_PRICE_REIMB_CLAIM + p.GPI_UP_DOLLAR),
                                         0.000001) 
                    if (row.Final_Price > (np.nanmax([row.OLD_MAC_PRICE * (1.0 + p.GPI_UP_FAC), row.mbr_bound]) + tolerance)):
                        row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                else:
                    if (row.PERCENT_CHANGE > (p.GPI_UP_FAC + tolerance)):
                        row['CHANGE_BUCKET'] = "Greater than allowable price increase"
 
            # check decreases
            elif (row.PERCENT_CHANGE < (- p.GPI_LOW_FAC - tolerance)) and (row.ORIG_PRICE_CLAIM > 0.00):
                row['CHANGE_BUCKET'] = "Less than allowable price decrease"
 
            # assign reasons
            if row['CHANGE_BUCKET'] == 'Greater than allowable price increase':
                if (row['OLD_MAC_PRICE'] < row['MAC1026_UNIT_PRICE']) and (
                        row['Final_Price'] - row['MAC1026_UNIT_PRICE'] < tolerance):
                    row['PRICE_TIER_REASON'] = 'OLD_MAC_PRICE < MAC1026_UNIT_PRICE and Final_Price = MAC1026_UNIT_PRICE'
                elif idx == 0 and (
                        row.PRICE_CHANGE_CLAIM_PROJ <= p.PRICE_BOUNDS_DF['max_dollar_increase'][idx] + (tolerance)):
                    row[
                        'PRICE_TIER_REASON'] = f'Price change <= ${p.PRICE_BOUNDS_DF["upper_bound"][idx]} satisfies tier constraint using PROJ QTY and CLAIMS'
                # TODO: this reason code is not quite right, need to check the VCML constraints:
                elif (row['RAISED_PRICE_UC']):
                    row['PRICE_TIER_REASON'] = 'U&C VCML constraint'
                elif (((row.New_Price * row.AVG_QTY_PER_RXS) - row.ORIG_PRICE_CLAIM) / row.ORIG_PRICE_CLAIM <= (
                        p.PRICE_BOUNDS_DF['max_percent_increase'][idx] + tolerance)) \
                        or ((row.New_Price * row.AVG_QTY_PER_RXS) - row.ORIG_PRICE_CLAIM <= (
                        p.PRICE_BOUNDS_DF['max_dollar_increase'][idx] + tolerance)):
                    row['PRICE_TIER_REASON'] = 'Rounding error'
                elif (((row.New_Price - row.OLD_MAC_PRICE) / row.OLD_MAC_PRICE) <= (
                        p.GPI_UP_FAC + tolerance)) and not p.TIERED_PRICE_LIM:
                    row['PRICE_TIER_REASON'] = 'Rounding error'
                elif row['PRICE_TIER_BUCKET'] == 'Zero Projected Claims':
                    row['PRICE_TIER_REASON'] = 'Soft bound violated'
                else:
                    row['PRICE_TIER_REASON'] = 'unknown'
 
            if row['CHANGE_BUCKET'] == 'Less than allowable price decrease':
                if (row['lb'] < np.nanmax(
                        [row['MAC1026_UNIT_PRICE'], row['ORIG_PRICE_CLAIM'] * (1 - p.GPI_LOW_FAC), .0001])):
                    row[
                        'PRICE_TIER_REASON'] = 'lower bound less than max(MAC1026_UNIT_PRICE,ORIG_PRICE_CLAIM * (1-p.GPI_LOW_FAC))'
                elif (row['GOODRX_UPPER_LIMIT'] < row['ORIG_PRICE_CLAIM'] * (
                        1 - p.GPI_LOW_FAC)) and (p.CLIENT_TYPE == 'COMMERCIAL' or p.CLIENT_TYPE == 'MEDICAID'):
                    if 'R' in row['BREAKOUT']:
                        row['PRICE_TIER_REASON'] = 'GoodRX less than max price decrease, retail'
                    else:
                        row['PRICE_TIER_REASON'] = 'GoodRX less than max price decrease, mail'
                elif round(row['Final_Price'], 4) == round(row['lb'], 4):
                    row['PRICE_TIER_REASON'] = 'Rounding error'
                elif row.PRICE_TIER == 'CONFLICT' and p.HANDLE_CONFLICT_GPI and p.CONFLICT_GPI_AS_TIERS and (
                        row['lb'] > row['ORIG_PRICE_CLAIM'] * (1 - 0.9999)):
                    row['PRICE_TIER_REASON'] = 'Conflict GPI Tier'
                else:
                    row['PRICE_TIER_REASON'] = 'unknown'
 
            # To check if the price decrease is due to lower INTERCEPT_HIGH pro
            if p.INTERCEPTOR_OPT:
                if row['CHANGE_BUCKET'] == 'Less than allowable price decrease':
                    if (pd.notna(row['VENDOR_PRICE']) & (row['INTERCEPT_HIGH'] < row['ORIG_PRICE_CLAIM'] * (1 - p.GPI_LOW_FAC))):
                        row['PRICE_TIER_REASON'] = 'Intercept High less than max price decrease'
                    elif round(row['Final_Price'], 4) == round(row['lb'], 4):
                        row['PRICE_TIER_REASON'] = 'Rounding error'
                    else:
                        row['PRICE_TIER_REASON'] = 'unknown'
                
                if p.ALLOW_INTERCEPT_LIMIT and not ((p.CLIENT_NAME_TABLEAU.startswith('WTW') and not p.REMOVE_WTW_RESTRICTION) or (p.CLIENT_NAME_TABLEAU.startswith('AON'))):
                    if row['CHANGE_BUCKET'] == 'Greater than allowable price increase':
                        if ((row['ACTUAL_KEEP_SEND'] == 0.0) & (row['INTERCEPT_LOW'] > row['ORIG_PRICE_CLAIM'] * (1 + p.GPI_UP_FAC))):
                            row['PRICE_TIER_REASON'] = 'Intercept Low greater than max price increase'
                        elif round(row['Final_Price'], 4) == round(row['lb'], 4):
                            row['PRICE_TIER_REASON'] = 'Rounding error'
                        else:
                            row['PRICE_TIER_REASON'] = 'unknown'

            return row
 
        # end of tiered_price_check(row)
 
        def tiered_price_check_OneOne(row):
            if p.NEW_YEAR_PRICE_LVL == 1:
                if row.PRICING_QTY_PROJ_EOY == 0.00:
                    row['PRICE_TIER_BUCKET'] = 'Zero Projected Claims'
                    row['CHANGE_BUCKET'] = 'No bounds for zero projected claims'
 
                # check increases
                elif row.PERCENT_CHANGE > 0:
                    '''
                    if row.ORIG_PRICE_CLAIM <= 8.00 :
                        row['PRICE_TIER_BUCKET'] = 'Less than $8'
                        if ( row.PRICE_CHANGE_CLAIM > 8+(tolerance) ):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
 
                    elif row.ORIG_PRICE_CLAIM  <= 25.00 :
                        row['PRICE_TIER_BUCKET'] = '$8.01 - $25'
                        if row.PERCENT_CHANGE > (.6 +tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
 
                    elif row.ORIG_PRICE_CLAIM  <= 50.: 
                        row['PRICE_TIER_BUCKET'] = '$25.01 - $50'
                        if row.PERCENT_CHANGE > (0.35 +tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range" 
 
                    elif row.ORIG_PRICE_CLAIM  <= 100.: 
                        row['PRICE_TIER_BUCKET'] = '$50.01 - $100'
                        if row.PERCENT_CHANGE > (0.25 +tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
 
                    elif row.ORIG_PRICE_CLAIM  >100.:
                        row['PRICE_TIER_BUCKET'] = 'Greater than $100'
                        if row.PERCENT_CHANGE> (0.15 + tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range" 
                    '''
                    idx = bisect.bisect_left(p.FULL_YEAR_LV_1_PRICE_BOUNDS_DF['upper_bound'], row.ORIG_PRICE_CLAIM)
                    # a la zero quantity tight bounds
                    if np.isnan(row.ORIG_PRICE_CLAIM) or np.isinf(row.ORIG_PRICE_CLAIM):
                        idx = 1
                    if idx == 0:  # Message for first bucket
                        row[
                            'PRICE_TIER_BUCKET'] = f'Less than or equal to ${p.FULL_YEAR_LV_1_PRICE_BOUNDS_DF["upper_bound"][0]}'
                        if row.PRICE_CHANGE_CLAIM > p.FULL_YEAR_LV_1_PRICE_BOUNDS_DF['max_dollar_increase'][idx] + (
                                tolerance * 10) or row.PERCENT_CHANGE > (
                                p.FULL_YEAR_LV_1_PRICE_BOUNDS_DF['max_percent_increase'][idx] + tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
                    elif idx == len(p.FULL_YEAR_LV_1_PRICE_BOUNDS_DF['upper_bound']) - 1:  # Message for last bucket
                        row[
                            'PRICE_TIER_BUCKET'] = f'Greater than ${p.FULL_YEAR_LV_1_PRICE_BOUNDS_DF["upper_bound"][idx - 1]}'
                        if row.PERCENT_CHANGE > (p.FULL_YEAR_LV_1_PRICE_BOUNDS_DF["max_percent_increase"][
                                                     idx] + tolerance) or row.PERCENT_CHANGE > (
                                p.FULL_YEAR_LV_1_PRICE_BOUNDS_DF['max_percent_increase'][idx] + tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
                    else:  # Message for other buckets
                        row[
                            'PRICE_TIER_BUCKET'] = f'${p.FULL_YEAR_LV_1_PRICE_BOUNDS_DF["upper_bound"][idx - 1]} - ${p.FULL_YEAR_LV_1_PRICE_BOUNDS_DF["upper_bound"][idx] - .01}'
                        if row.PRICE_CHANGE_CLAIM > p.FULL_YEAR_LV_1_PRICE_BOUNDS_DF['max_dollar_increase'][idx] + (
                                tolerance * 10) \
                                or row.PERCENT_CHANGE > (
                                p.FULL_YEAR_LV_1_PRICE_BOUNDS_DF['max_percent_increase'][idx] + tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
 
 
                # check decreases
                elif (row.PERCENT_CHANGE < (- p.GPI_LOW_FAC - tolerance)) and (row.ORIG_PRICE_CLAIM > 0.00):
                    row['CHANGE_BUCKET'] = "Less than allowable price decrease"
 
                # assign reasons
                if row['CHANGE_BUCKET'] == 'Greater than allowable price increase':
                    if (row['OLD_MAC_PRICE'] < row['MAC1026_UNIT_PRICE']) and (
                            row['Final_Price'] - row['MAC1026_UNIT_PRICE'] < tolerance):
                        row[
                            'PRICE_TIER_REASON'] = 'OLD_MAC_PRICE < MAC1026_UNIT_PRICE and Final_Price = MAC1026_UNIT_PRICE'
                    elif (row['PRICE_TIER_BUCKET'] == 'Less than $8') and (
                            row.PRICE_CHANGE_CLAIM_PROJ <= 8 + (tolerance)):
                        row[
                            'PRICE_TIER_REASON'] = 'Price change < $8 satisfies tier constraint using PROJ QTY and CLAIMS'
                    # TODO: this reason code is not quite right, need to check the VCML constraints:
                    elif (row['RAISED_PRICE_UC']):
                        row['PRICE_TIER_REASON'] = 'U&C VCML constraint'
                    elif (((row['PRICE_TIER_BUCKET'] == 'Less than $8') & (
                            ((row.New_Price * row.AVG_QTY_PER_RXS) - row.ORIG_PRICE_CLAIM) <= (8 + tolerance)))
                          | ((row['PRICE_TIER_BUCKET'] == '$8.01 - $25') & (
                                    ((row.New_Price - row.OLD_MAC_PRICE) / row.OLD_MAC_PRICE) <= (0.6 + tolerance)))
                          | ((row['PRICE_TIER_BUCKET'] == '$25.01 - $50') & (
                                    ((row.New_Price - row.OLD_MAC_PRICE) / row.OLD_MAC_PRICE) <= (0.35 + tolerance)))
                          | ((row['PRICE_TIER_BUCKET'] == '$50.01 - $100') & (
                                    ((row.New_Price - row.OLD_MAC_PRICE) / row.OLD_MAC_PRICE) <= (0.25 + tolerance)))
                          | ((row['PRICE_TIER_BUCKET'] == 'Greater than $100') & (
                                    ((row.New_Price - row.OLD_MAC_PRICE) / row.OLD_MAC_PRICE) <= (0.15 + tolerance)))
                    ):
                        row['PRICE_TIER_REASON'] = 'Rounding error'
                    else:
                        row['PRICE_TIER_REASON'] = 'unknown'
 
                if row['CHANGE_BUCKET'] == 'Less than allowable price decrease':
                    if (row['lb'] < np.nanmax(
                            [row['MAC1026_UNIT_PRICE'], row['ORIG_PRICE_CLAIM'] * (1 - p.GPI_LOW_FAC), .0001])):
                        row[
                            'PRICE_TIER_REASON'] = 'lower bound less than max(MAC1026_UNIT_PRICE,ORIG_PRICE_CLAIM * (1-p.GPI_LOW_FAC))'
                    elif (row['GOODRX_UPPER_LIMIT'] < row['ORIG_PRICE_CLAIM'] * (
                            1 - p.GPI_LOW_FAC)) and (p.CLIENT_TYPE == 'COMMERCIAL' or p.CLIENT_TYPE == 'MEDICAID'):
                        if 'R' in row['BREAKOUT']:
                            row['PRICE_TIER_REASON'] = 'GoodRX less than max price decrease, retail'
                        else:
                            row['PRICE_TIER_REASON'] = 'GoodRX less than max price decrease, mail'
                    elif row.PRICE_TIER == 'CONFLICT' and p.HANDLE_CONFLICT_GPI and p.CONFLICT_GPI_AS_TIERS and (
                            row['lb'] > row['ORIG_PRICE_CLAIM'] * (1 - 0.9999)):
                        row['PRICE_TIER_REASON'] = 'Conflict GPI Tier'
                    else:
                        row['PRICE_TIER_REASON'] = 'unknown'
                            
                #allow higher increases/decreases for interceptor
                if p.INTERCEPTOR_OPT:
                    if row['CHANGE_BUCKET'] == 'Less than allowable price decrease':
                        if (pd.notna(row['VENDOR_PRICE']) & (row['INTERCEPT_HIGH'] < row['ORIG_PRICE_CLAIM'] * (1 - p.GPI_LOW_FAC))):
                            row['PRICE_TIER_REASON'] = 'Intercept High less than max price decrease'
                        elif round(row['Final_Price'], 4) == round(row['lb'], 4):
                            row['PRICE_TIER_REASON'] = 'Rounding error'
                        else:
                            row['PRICE_TIER_REASON'] = 'unknown'

                    if p.ALLOW_INTERCEPT_LIMIT and not ((p.CLIENT_NAME_TABLEAU.startswith('WTW') and not p.REMOVE_WTW_RESTRICTION) or (p.CLIENT_NAME_TABLEAU.startswith('AON'))):
                        if row['CHANGE_BUCKET'] == 'Greater than allowable price increase':
                            if ((row['ACTUAL_KEEP_SEND'] == 0.0) & (row['INTERCEPT_LOW'] > row['ORIG_PRICE_CLAIM'] * (1 + p.GPI_UP_FAC))):
                                row['PRICE_TIER_REASON'] = 'Intercept Low greater than max price increase'
                            elif round(row['Final_Price'], 4) == round(row['lb'], 4):
                                row['PRICE_TIER_REASON'] = 'Rounding error'
                            else:
                                row['PRICE_TIER_REASON'] = 'unknown'
 
            if p.NEW_YEAR_PRICE_LVL == 2:
                if row.PRICING_QTY_PROJ_EOY == 0.00:
                    row['PRICE_TIER_BUCKET'] = 'Zero Projected Claims'
                    row['CHANGE_BUCKET'] = 'No bounds for zero projected claims'
 
                # check increases
                elif row.PERCENT_CHANGE > 0:
                    '''
                    if row.ORIG_PRICE_CLAIM <= 8.00 :
                        row['PRICE_TIER_BUCKET'] = 'Less than $8'
                        if ( row.PRICE_CHANGE_CLAIM > 10+(tolerance) ):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
 
                    elif row.ORIG_PRICE_CLAIM  <= 25.00 :
                        row['PRICE_TIER_BUCKET'] = '$8.01 - $25'
                        if row.PERCENT_CHANGE > (1.0 +tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
 
                    elif row.ORIG_PRICE_CLAIM  <= 50.: 
                        row['PRICE_TIER_BUCKET'] = '$25.01 - $50'
                        if row.PERCENT_CHANGE > (0.75 +tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range" 
 
                    elif row.ORIG_PRICE_CLAIM  <= 100.: 
                        row['PRICE_TIER_BUCKET'] = '$50.01 - $100'
                        if row.PERCENT_CHANGE > (0.35 +tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
 
                    elif row.ORIG_PRICE_CLAIM  >100.:
                        row['PRICE_TIER_BUCKET'] = 'Greater than $100'
                        if row.PERCENT_CHANGE> (0.25 + tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
                    '''
                    idx = bisect.bisect_left(p.FULL_YEAR_LV_2_PRICE_BOUNDS_DF['upper_bound'], row.ORIG_PRICE_CLAIM)
                    if idx == 0:  # Message for first bucket
                        row[
                            'PRICE_TIER_BUCKET'] = f'Less than or equal to ${p.FULL_YEAR_LV_2_PRICE_BOUNDS_DF["upper_bound"][0]}'
                        if row.PRICE_CHANGE_CLAIM > p.FULL_YEAR_LV_2_PRICE_BOUNDS_DF['max_dollar_increase'][idx] + (
                                tolerance * 10) or row.PERCENT_CHANGE > (
                                p.FULL_YEAR_LV_2_PRICE_BOUNDS_DF['max_percent_increase'][idx] + tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
                    elif idx == len(p.FULL_YEAR_LV_2_PRICE_BOUNDS_DF['upper_bound']) - 1:  # Message for last bucket
                        row[
                            'PRICE_TIER_BUCKET'] = f'Greater than ${p.FULL_YEAR_LV_2_PRICE_BOUNDS_DF["upper_bound"][idx - 1]}'
                        if row.PERCENT_CHANGE > (p.FULL_YEAR_LV_2_PRICE_BOUNDS_DF["max_percent_increase"][
                                                     idx] + tolerance) or row.PERCENT_CHANGE > (
                                p.FULL_YEAR_LV_2_PRICE_BOUNDS_DF['max_percent_increase'][idx] + tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
                    else:  # Message for other buckets
                        row[
                            'PRICE_TIER_BUCKET'] = f'${p.FULL_YEAR_LV_2_PRICE_BOUNDS_DF["upper_bound"][idx - 1]} - ${p.FULL_YEAR_LV_2_PRICE_BOUNDS_DF["upper_bound"][idx] - .01}'
                        if row.PRICE_CHANGE_CLAIM > p.FULL_YEAR_LV_2_PRICE_BOUNDS_DF['max_dollar_increase'][idx] + (
                                tolerance * 10) \
                                or row.PERCENT_CHANGE > (
                                p.FULL_YEAR_LV_2_PRICE_BOUNDS_DF['max_percent_increase'][idx] + tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
 
                # check decreases
                elif (row.PERCENT_CHANGE < (- p.GPI_LOW_FAC - tolerance)) and (row.ORIG_PRICE_CLAIM > 0.00):
                    row['CHANGE_BUCKET'] = "Less than allowable price decrease"
 
                # assign reasons
                if row['CHANGE_BUCKET'] == 'Greater than allowable price increase':
                    if (row['OLD_MAC_PRICE'] < row['MAC1026_UNIT_PRICE']) and (
                            row['Final_Price'] - row['MAC1026_UNIT_PRICE'] < tolerance):
                        row[
                            'PRICE_TIER_REASON'] = 'OLD_MAC_PRICE < MAC1026_UNIT_PRICE and Final_Price = MAC1026_UNIT_PRICE'
                    elif (row['PRICE_TIER_BUCKET'] == 'Less than $8') and (
                            row.PRICE_CHANGE_CLAIM_PROJ <= 8 + (tolerance)):
                        row[
                            'PRICE_TIER_REASON'] = 'Price change < $8 satisfies tier constraint using PROJ QTY and CLAIMS'
                    # TODO: this reason code is not quite right, need to check the VCML constraints:
                    elif (row['RAISED_PRICE_UC']):
                        row['PRICE_TIER_REASON'] = 'U&C VCML constraint'
                    elif (((row['PRICE_TIER_BUCKET'] == 'Less than $8') & (
                            ((row.New_Price * row.AVG_QTY_PER_RXS) - row.ORIG_PRICE_CLAIM) <= (10 + tolerance)))
                          | ((row['PRICE_TIER_BUCKET'] == '$8.01 - $25') & (
                                    ((row.New_Price - row.OLD_MAC_PRICE) / row.OLD_MAC_PRICE) <= (1.0 + tolerance)))
                          | ((row['PRICE_TIER_BUCKET'] == '$25.01 - $50') & (
                                    ((row.New_Price - row.OLD_MAC_PRICE) / row.OLD_MAC_PRICE) <= (0.75 + tolerance)))
                          | ((row['PRICE_TIER_BUCKET'] == '$50.01 - $100') & (
                                    ((row.New_Price - row.OLD_MAC_PRICE) / row.OLD_MAC_PRICE) <= (0.35 + tolerance)))
                          | ((row['PRICE_TIER_BUCKET'] == 'Greater than $100') & (
                                    ((row.New_Price - row.OLD_MAC_PRICE) / row.OLD_MAC_PRICE) <= (0.25 + tolerance)))
                    ):
                        row['PRICE_TIER_REASON'] = 'Rounding error'
                    else:
                        row['PRICE_TIER_REASON'] = 'unknown'
 
                if row['CHANGE_BUCKET'] == 'Less than allowable price decrease':
                    if (row['lb'] < np.nanmax(
                            [row['MAC1026_UNIT_PRICE'], row['ORIG_PRICE_CLAIM'] * (1 - p.GPI_LOW_FAC), .0001])):
                        row[
                            'PRICE_TIER_REASON'] = 'lower bound less than max(MAC1026_UNIT_PRICE,ORIG_PRICE_CLAIM * (1-p.GPI_LOW_FAC))'
                    elif (row['GOODRX_UPPER_LIMIT'] < row['ORIG_PRICE_CLAIM'] * (
                            1 - p.GPI_LOW_FAC)) and (p.CLIENT_TYPE == 'COMMERCIAL' or p.CLIENT_TYPE == 'MEDICAID'):
                        if 'R' in row['BREAKOUT']:
                            row['PRICE_TIER_REASON'] = 'GoodRX less than max price decrease, retail'
                        else:
                            row['PRICE_TIER_REASON'] = 'GoodRX less than max price decrease, mail'
                    elif row.PRICE_TIER == 'CONFLICT' and p.HANDLE_CONFLICT_GPI and p.CONFLICT_GPI_AS_TIERS and (
                            row['lb'] > row['ORIG_PRICE_CLAIM'] * (1 - 0.9999)):
                        row['PRICE_TIER_REASON'] = 'Conflict GPI Tier'
                    else:
                        row['PRICE_TIER_REASON'] = 'unknown'

                #allow higher increases/decreases for interceptor
                if p.INTERCEPTOR_OPT:
                    if row['CHANGE_BUCKET'] == 'Less than allowable price decrease':
                        if (pd.notna(row['VENDOR_PRICE']) & (row['INTERCEPT_HIGH'] < row['ORIG_PRICE_CLAIM'] * (1 - p.GPI_LOW_FAC))):
                            row['PRICE_TIER_REASON'] = 'Intercept High less than max price decrease'
                        elif round(row['Final_Price'], 4) == round(row['lb'], 4):
                            row['PRICE_TIER_REASON'] = 'Rounding error'
                        else:
                            row['PRICE_TIER_REASON'] = 'unknown'

                    if p.ALLOW_INTERCEPT_LIMIT and not ((p.CLIENT_NAME_TABLEAU.startswith('WTW') and not p.REMOVE_WTW_RESTRICTION) or (p.CLIENT_NAME_TABLEAU.startswith('AON'))):
                        if row['CHANGE_BUCKET'] == 'Greater than allowable price increase':
                            if ((row['ACTUAL_KEEP_SEND'] == 0.0) & (row['INTERCEPT_LOW'] > row['ORIG_PRICE_CLAIM'] * (1 + p.GPI_UP_FAC))):
                                row['PRICE_TIER_REASON'] = 'Intercept Low greater than max price increase'
                            elif round(row['Final_Price'], 4) == round(row['lb'], 4):
                                row['PRICE_TIER_REASON'] = 'Rounding error'
                            else:
                                row['PRICE_TIER_REASON'] = 'unknown'
                            
            if p.NEW_YEAR_PRICE_LVL == 3:
                if row.PRICING_QTY_PROJ_EOY == 0.00:
                    row['PRICE_TIER_BUCKET'] = 'Zero Projected Claims'
                    row['CHANGE_BUCKET'] = 'No bounds for zero projected claims'
 
                # check increases
                elif row.PERCENT_CHANGE > 0:
                    '''
                    if row.ORIG_PRICE_CLAIM <= 3.00 :
                        row['PRICE_TIER_BUCKET'] = 'Less than $3'
                        if ( row.PRICE_CHANGE_CLAIM > 20+(tolerance) ):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
 
                    elif row.ORIG_PRICE_CLAIM  <= 6.00 :
                        row['PRICE_TIER_BUCKET'] = '$3.01 - $6'
                        if ( row.PRICE_CHANGE_CLAIM > 30+(tolerance) ):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
 
                    elif row.ORIG_PRICE_CLAIM  > 6.:
                        row['PRICE_TIER_BUCKET'] = 'Greater than $6'
                        if row.PERCENT_CHANGE> (3.0 + tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
                    '''
                    idx = bisect.bisect_left(p.FULL_YEAR_LV_3_PRICE_BOUNDS_DF['upper_bound'], row.ORIG_PRICE_CLAIM)
                    if idx == 0:  # Message for first bucket
                        row[
                            'PRICE_TIER_BUCKET'] = f'Less than or equal to ${p.FULL_YEAR_LV_3_PRICE_BOUNDS_DF["upper_bound"][0]}'
                        if row.PRICE_CHANGE_CLAIM > p.FULL_YEAR_LV_3_PRICE_BOUNDS_DF['max_dollar_increase'][idx] + (
                                tolerance * 10) or row.PERCENT_CHANGE > (
                                p.FULL_YEAR_LV_3_PRICE_BOUNDS_DF['max_percent_increase'][idx] + tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
                    elif idx == len(p.FULL_YEAR_LV_3_PRICE_BOUNDS_DF['upper_bound']) - 1:  # Message for last bucket
                        row[
                            'PRICE_TIER_BUCKET'] = f'Greater than ${p.FULL_YEAR_LV_3_PRICE_BOUNDS_DF["upper_bound"][idx - 1]}'
                        if row.PERCENT_CHANGE > (p.FULL_YEAR_LV_3_PRICE_BOUNDS_DF["max_percent_increase"][
                                                     idx] + tolerance) or row.PERCENT_CHANGE > (
                                p.FULL_YEAR_LV_3_PRICE_BOUNDS_DF['max_percent_increase'][idx] + tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
                    else:  # Message for other buckets
                        row[
                            'PRICE_TIER_BUCKET'] = f'${p.FULL_YEAR_LV_3_PRICE_BOUNDS_DF["upper_bound"][idx - 1]} - ${p.FULL_YEAR_LV_3_PRICE_BOUNDS_DF["upper_bound"][idx] - .01}'
                        if row.PRICE_CHANGE_CLAIM > p.FULL_YEAR_LV_3_PRICE_BOUNDS_DF['max_dollar_increase'][idx] + (
                                tolerance * 10) \
                                or row.PERCENT_CHANGE > (
                                p.FULL_YEAR_LV_3_PRICE_BOUNDS_DF['max_percent_increase'][idx] + tolerance):
                            row['CHANGE_BUCKET'] = "Greater than allowable price increase"
                        else:
                            row['CHANGE_BUCKET'] = "Within range"
 
                # check decreases
                elif (row.PERCENT_CHANGE < (- p.GPI_LOW_FAC - tolerance)) and (row.ORIG_PRICE_CLAIM > 0.00):
                    row['CHANGE_BUCKET'] = "Less than allowable price decrease"
 
                # assign reasons
                if row['CHANGE_BUCKET'] == 'Greater than allowable price increase':
                    if (row['OLD_MAC_PRICE'] < row['MAC1026_UNIT_PRICE']) and (
                            row['Final_Price'] - row['MAC1026_UNIT_PRICE'] < tolerance):
                        row[
                            'PRICE_TIER_REASON'] = 'OLD_MAC_PRICE < MAC1026_UNIT_PRICE and Final_Price = MAC1026_UNIT_PRICE'
                    elif (row['PRICE_TIER_BUCKET'] == 'Less than $3') and (
                            row.PRICE_CHANGE_CLAIM_PROJ <= 20 + (tolerance)):
                        row[
                            'PRICE_TIER_REASON'] = 'Price change < $3 satisfies tier constraint using PROJ QTY and CLAIMS'
                    elif (row['PRICE_TIER_BUCKET'] == 'Less than $6') and (
                            row.PRICE_CHANGE_CLAIM_PROJ <= 30 + (tolerance)):
                        row[
                            'PRICE_TIER_REASON'] = 'Price change < $6 satisfies tier constraint using PROJ QTY and CLAIMS'
                    # TODO: this reason code is not quite right, need to check the VCML constraints:
                    elif (row['RAISED_PRICE_UC']):
                        row['PRICE_TIER_REASON'] = 'U&C VCML constraint'
                    elif (((row['PRICE_TIER_BUCKET'] == 'Less than $3') & (
                            ((row.New_Price * row.AVG_QTY_PER_RXS) - row.ORIG_PRICE_CLAIM) <= (20 + tolerance)))
                          | ((row['PRICE_TIER_BUCKET'] == '$3.01 - $6') & (
                                    ((row.New_Price * row.AVG_QTY_PER_RXS) - row.ORIG_PRICE_CLAIM) <= (30 + tolerance)))
                          | ((row['PRICE_TIER_BUCKET'] == 'Greater than $6') & (
                                    ((row.New_Price - row.OLD_MAC_PRICE) / row.OLD_MAC_PRICE) <= (3.0 + tolerance)))
                    ):
                        row['PRICE_TIER_REASON'] = 'Rounding error'
                    else:
                        row['PRICE_TIER_REASON'] = 'unknown'
 
                if row['CHANGE_BUCKET'] == 'Less than allowable price decrease':
                    if (row['lb'] < np.nanmax(
                            [row['MAC1026_UNIT_PRICE'], row['ORIG_PRICE_CLAIM'] * (1 - p.GPI_LOW_FAC), .0001])):
                        row[
                            'PRICE_TIER_REASON'] = 'lower bound less than max(MAC1026_UNIT_PRICE,ORIG_PRICE_CLAIM * (1-p.GPI_LOW_FAC))'
                    elif (row['GOODRX_UPPER_LIMIT'] < row['ORIG_PRICE_CLAIM'] * (
                            1 - p.GPI_LOW_FAC)) and (p.CLIENT_TYPE == 'COMMERCIAL' or p.CLIENT_TYPE == 'MEDICAID'):
                        if 'R' in row['BREAKOUT']:
                            row['PRICE_TIER_REASON'] = 'GoodRX less than max price decrease, retail'
                        else:
                            row['PRICE_TIER_REASON'] = 'GoodRX less than max price decrease, mail'
                    elif row.PRICE_TIER == 'CONFLICT' and p.HANDLE_CONFLICT_GPI and p.CONFLICT_GPI_AS_TIERS and (
                            row['lb'] > row['ORIG_PRICE_CLAIM'] * (1 - 0.9999)):
                        row['PRICE_TIER_REASON'] = 'Conflict GPI Tier'
                    else:
                        row['PRICE_TIER_REASON'] = 'unknown'
                        
                #allow higher increases/decreases for interceptor
                if p.INTERCEPTOR_OPT:
                    if row['CHANGE_BUCKET'] == 'Less than allowable price decrease':
                        if (pd.notna(row['VENDOR_PRICE']) & (row['INTERCEPT_HIGH'] < row['ORIG_PRICE_CLAIM'] * (1 - p.GPI_LOW_FAC))):
                            row['PRICE_TIER_REASON'] = 'Intercept High less than max price decrease'
                        elif round(row['Final_Price'], 4) == round(row['lb'], 4):
                            row['PRICE_TIER_REASON'] = 'Rounding error'
                        else:
                            row['PRICE_TIER_REASON'] = 'unknown'

                    if p.ALLOW_INTERCEPT_LIMIT and not ((p.CLIENT_NAME_TABLEAU.startswith('WTW') and not p.REMOVE_WTW_RESTRICTION) or (p.CLIENT_NAME_TABLEAU.startswith('AON'))):
                        if row['CHANGE_BUCKET'] == 'Greater than allowable price increase':
                            if ((row['ACTUAL_KEEP_SEND'] == 0.0) & (row['INTERCEPT_LOW'] > row['ORIG_PRICE_CLAIM'] * (1 + p.GPI_UP_FAC))):
                                row['PRICE_TIER_REASON'] = 'Intercept Low greater than max price increase'
                            elif round(row['Final_Price'], 4) == round(row['lb'], 4):
                                row['PRICE_TIER_REASON'] = 'Rounding error'
                            else:
                                row['PRICE_TIER_REASON'] = 'unknown'
 
            return row
 
        # end of tiered_price_check(row)
 
        if p.FULL_YEAR:
            if p.TIERED_PRICE_LIM:
                lp_data_output_df = lp_data_output_df.apply(lambda x: tiered_price_check_OneOne(x), axis=1)
            else:
                lp_data_output_df = lp_data_output_df.apply(lambda x: tiered_price_check(x), axis=1)
        else:
            lp_data_output_df = lp_data_output_df.apply(lambda x: tiered_price_check(x), axis=1)
 
        # If the GPI is listed in the floor GPI file and the final price is being set to MAC1026, then the following lines set the reason for price tier vioaltion as 'Floor_GPI'
        floor_gpi = standardize_df(pd.read_csv(p.FILE_INPUT_PATH + p.FLOOR_GPI_LIST, dtype=p.VARIABLE_TYPE_DIC))
        if 'BG_FLAG' not in floor_gpi.columns:
                floor_gpi['BG_FLAG'] = 'G' 
        floor_gpi_gpis = (floor_gpi['BG_FLAG'] + floor_gpi['GPI']).drop_duplicates()
        lp_data_output_df['BG_GPI'] = lp_data_output_df['BG_FLAG'] + lp_data_output_df['GPI']
        lp_data_output_df.loc[lp_data_output_df.BG_GPI.isin(floor_gpi_gpis) & abs(
            lp_data_output_df.MAC1026_UNIT_PRICE - lp_data_output_df.Final_Price) < tolerance, 'PRICE_TIER_REASON'] = 'Floor_GPI'
 
        price_tier_violations = lp_data_output_df[(lp_data_output_df['CHANGE_BUCKET'] != 'Within range')]
        price_tier_violations = price_tier_violations[
            ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'BG_FLAG', 'GPI', 'NDC11', 'MACLIST', 'CHAIN_GROUP', 'CHAIN_SUBGROUP',
             'PRICE_MUTABLE', 'QTY', 'CLAIMS', 'MAC1026_UNIT_PRICE', 'PRICING_AVG_AWP', 'AVG_AWP', 'VCML_AVG_AWP',
             'VCML_AVG_CLAIM_QTY', 'CURRENT_MAC_PRICE', 'PRE_UC_MAC_PRICE', 'PRICE_CHANGED_UC', 'RAISED_PRICE_UC', 
             'MAC_PRICE_UNIT_ADJ','OLD_MAC_PRICE','CLAIMS_PROJ_EOY', 'PRICING_CLAIMS_PROJ_EOY', 'QTY_PROJ_EOY', 
             'PRICING_QTY_PROJ_EOY', 'lb', 'ub', 'lb_name', 'ub_name', 'lower_bound_ordered', 'upper_bound_ordered',
             'EFF_UNIT_PRICE_new', 'Final_Price', 'AVG_QTY_PER_RXS', 'ORIG_PRICE_CLAIM', 'FINAL_PRICE_CLAIM', 
             'PRICE_CHANGE_CLAIM', 'PERCENT_CHANGE', 'PRICE_TIER_BUCKET', 'CHANGE_BUCKET', 'PRICE_TIER_REASON']]
        
        price_tier_violations.sort_values(['PRICE_TIER_BUCKET', 'CHANGE_BUCKET', 'PRICE_TIER_REASON'])
 
        # create the report:
        fname_violations = 'price tiering rules violations REPORT.csv'
        fpath_violations = os.path.join(p.FILE_OUTPUT_PATH, fname_violations)
 
        if 'gs://' in p.FILE_OUTPUT_PATH:
            price_tier_violations.to_csv(fname_violations, index=False)
        else:
            price_tier_violations.to_csv(fpath_violations, index=False)
 
        if 'gs://' in p.FILE_OUTPUT_PATH:
            # COPY TO CLOUD
            local_fpath = fname_violations
            cloud_path = os.path.join(p.FILE_OUTPUT_PATH, fname_violations)
            bucket = p.FILE_OUTPUT_PATH[5:].split('/', 1)[0]
            assert os.path.exists(local_fpath), f'Path not found locally (on container): {local_fpath}'
            print(f'Uploading file {fname_violations} to cloud path: {cloud_path}')
            upload_blob(bucket, local_fpath, cloud_path)
 
        if ((lp_data_output_df['CHANGE_BUCKET'].isin(
                ['Greater than allowable price increase', 'Less than allowable price decrease'])) & (
                    lp_data_output_df['PRICE_TIER_REASON'].isin(['unknown', 'Zero Projected Claims']))).sum() > 0:
            number_of_unexplained_violations = ((lp_data_output_df['CHANGE_BUCKET'].isin(
                ['Greater than allowable price increase', 'Less than allowable price decrease'])) & (
                                                    lp_data_output_df['PRICE_TIER_REASON'].isin(
                                                        ['unknown', 'Zero Projected Claims']))).sum()
            print('')
            print(
                '*WARNING: Data has price tier violations: {} unexplained violations. Check price tiering rules violations REPORT.csv in the output folder'.format(
                    number_of_unexplained_violations))
            assert number_of_unexplained_violations == 0, '*WARNING: Data has price tier violations: {} unexplained violations. Check price tiering rules violations REPORT.csv in the output folder'.format(
                number_of_unexplained_violations)
        else:
            number_of_violations = (lp_data_output_df['CHANGE_BUCKET'].isin(
                ['Greater than allowable price increase', 'Less than allowable price decrease'])).sum()
            print('')
            print(
                'No unexplained price tier violations. {} explained violations To see reasons check price tiering rules violations REPORT.csv in the output folder'.format(
                    number_of_violations))
 
    except Exception as e:
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'QA price_tiering_rules Report', repr(e), error_loc)
        raise e


def qa_Prices_above_MAC1026_floor(
        params_in: str,
        lp_data_output_df_in: InputPath('pickle'),
        lp_with_final_prices_in: InputPath('pickle'),
        output_cols_in: InputPath('pickle'),
        tolerance: float = 0.005
):
    '''
    Test: Prices above MAC1026 floor
    '''
    import os
    import pickle
    import importlib
    import pandas as pd
    import numpy as np
    import util_funcs as uf
    from types import ModuleType
 
    from util_funcs import write_params, write_to_bq, get_formatted_string
    if isinstance(params_in, ModuleType):
        p = params_in
    else:
        write_params(params_in)
        import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df
    from CPMO_shared_functions import update_run_status, round_to

    try:
        # read input
        with open(lp_with_final_prices_in, 'rb') as f:
            lp_with_final_prices = pd.read_pickle(f)
        with open(lp_data_output_df_in, 'rb') as f:
            lp_data_output_df = pd.read_pickle(f)
        with open(output_cols_in, 'rb') as f:
            output_cols = pickle.load(f)
        FILE_OUTPUT_PATH = p.FILE_OUTPUT_PATH
 
        for column in ['PRICE_CHANGED_UC', 'RAISED_PRICE_UC']:
            if column not in lp_with_final_prices.columns:
                lp_with_final_prices[column] = False
            if column not in lp_data_output_df.columns:
                lp_data_output_df[column] = False

        qa_Prices_above_MAC1026_floor_err_list = []
 
        # Test: Prices below non_mac_rate per channel off AWP ceiling
        # CPMO_lp_functions.py uses df.PRICING_AVG_AWP * AVG_FAC where AVG_FAC is default at 0.75 and defined below as (1- mail NMR) or (1- retail NMR) based on channel
        lp_with_final_prices['AVG_FAC'] = 1 - p.RETAIL_NON_MAC_RATE
        lp_with_final_prices.loc[(lp_with_final_prices['CHAIN_GROUP'] == 'MAIL') | (
                    lp_with_final_prices['CHAIN_GROUP'] == 'MCHOICE'), 'AVG_FAC'] = 1 - p.MAIL_NON_MAC_RATE
        lp_with_final_prices.loc[(lp_with_final_prices['BG_FLAG'] == 'B'), 'AVG_FAC'] = 1 - p.BRAND_NON_MAC_RATE
 
        if ((lp_with_final_prices['PRICING_QTY_PROJ_EOY'] > 0) & (lp_with_final_prices['Final_Price'] > (
                (lp_with_final_prices['PRICING_AVG_AWP'] * lp_with_final_prices['AVG_FAC']) + tolerance))).sum() > 0:
            outAWP = lp_with_final_prices[(lp_with_final_prices['PRICING_QTY_PROJ_EOY'] > 0) & (
                        lp_with_final_prices['Final_Price'] > (
                            lp_with_final_prices['PRICING_AVG_AWP'] * lp_with_final_prices['AVG_FAC'] + tolerance))]
            outofbounds = len(outAWP)
            outAWP.loc[:, 'REASON_CODE'] = 'unknown'
            outAWP.loc[(outAWP['MAC1026_UNIT_PRICE'] > outAWP['PRICING_AVG_AWP'] * outAWP['AVG_FAC']) & (abs(
                outAWP['Final_Price'] - outAWP[
                    'MAC1026_UNIT_PRICE']) < tolerance), 'REASON_CODE'] = 'new-price = MAC1026 > PRICING_AVG_AWP * (1 - non-mac rate)'
            outAWP.loc[
                (outAWP['OLD_MAC_PRICE'] * (1 - p.GPI_LOW_FAC) > outAWP['PRICING_AVG_AWP'] * outAWP['AVG_FAC']) & (
                            outAWP['MAC_PRICE_UNIT_ADJ'] * (1 - p.GPI_LOW_FAC) > outAWP[
                        'MAC1026_UNIT_PRICE']), 'REASON_CODE'] = 'lower tier bound price * (1-.8) > MAC1026 and AVG_AWP * (1 - non-mac rate)'
            outAWP.loc[(outAWP['Final_Price'] < (outAWP['VCML_AVG_AWP'] * outAWP['AVG_FAC'] + tolerance)) & (outAWP[
                'RAISED_PRICE_UC']), 'REASON_CODE'] = 'U&C raised price & AVG_AWP * (1 - non-mac rate) < Final price < VCML_AVG_AWP * (1 - non-mac rate)'
            outAWP.loc[
                (outAWP['MAC_PRICE_UNIT_ADJ'] * (1 - p.GPI_LOW_FAC) > outAWP['PRICING_AVG_AWP'] * outAWP['AVG_FAC']) &
                (outAWP['ub'] - outAWP[
                    'lb'] - 1e4 < 1e4), 'REASON_CODE'] = 'low & up bounds set by MAC_PRICE_UNIT_ADJ hence ub = lb+1e4'
            outAWP.loc[(pd.notna(outAWP['UNC_OVRD_AMT'])) & (
                        outAWP['UNC_OVRD_AMT'] > outAWP['PRICING_AVG_AWP'] * outAWP[
                    'AVG_FAC']), 'REASON_CODE'] = 'UNC Override Price > Standard AWP Discount'
            outAWP.loc[outAWP['PRICE_MUTABLE'] == 0, 'REASON_CODE'] = 'Immutable'
            if p.INTERCEPTOR_OPT:
                outAWP.loc[(outAWP['lb'].apply(round_to) == outAWP['INTERCEPT_LOW'].apply(round_to)) & (outAWP['ub'] - outAWP['lb'] - 1e4 < 1e4) & (
                            outAWP['INTERCEPT_LOW'] > outAWP['PRICING_AVG_AWP'] * outAWP[
                        'AVG_FAC']), 'REASON_CODE'] = 'low & up bounds set by Interceptor Logic and INTERCEPTOR_LOW > AVG_AWP * (1 - non-mac rate)'
 
            if p.PHARMACY_EXCLUSION:
                outAWP['MIN_PRICE'] = outAWP[['MAC_PRICE_UNIT_ADJ', 'MAC1026_UNIT_PRICE']].max(axis=1)
                outAWP.loc[(outAWP['Final_Price'] >= outAWP['MIN_PRICE']) &
                           (outAWP['Final_Price'] <= outAWP['MIN_PRICE'] + 0.0001) &
                           (outAWP['REASON_CODE'] == 'unknown') &
                           (outAWP['CHAIN_GROUP'].isin(
                               p.LIST_PHARMACY_EXCLUSION)), 'REASON_CODE'] = 'following the PHARMACY_EXCLUSION rule'
            unknowns = (outAWP['REASON_CODE'] == 'unknown').sum()
            print('')
            print(
                '*Warning: there are {} prices above AVG_AWP * (1 - non-mac rate), of which {} are of unknown reasons. check AWP_Standard_Discount_REPORT.csv'.format(
                    outofbounds, unknowns))
            if unknowns:
                if False:  # p.WRITE_TO_BQ:
                    write_to_bq(
                        outAWP,
                        project_output=p.BQ_OUTPUT_PROJECT_ID,
                        dataset_output=p.BQ_OUTPUT_DATASET,
                        table_id="AWP_Standard_Discount_REPORT",
                        client_name_param=', '.join(sorted(p.CUSTOMER_ID)),
                        timestamp_param=p.TIMESTAMP,
                        run_id=p.AT_RUN_ID,
                        schema=None
                    )
                else:
                    outAWP.to_csv(FILE_OUTPUT_PATH + 'AWP_Standard_Discount_REPORT.csv', index=False)
                    qa_Prices_above_MAC1026_floor_err_list.append('There are {} prices above AVG AWP * (1 - non-mac rate) for unknown reasons. Check AWP_Standard_Discount_REPORT.csv'.format(unknowns))
        else:
            print('')
            print('No prices above AWP * (1 - non-mac rate).')
 
        # Test: Pharmacies on the same VCML have the same output price
        vary_across_vcml = lp_data_output_df.groupby(['MACLIST', 'BG_FLAG', 'GPI_NDC'])['Final_Price'].nunique() != 1
        if vary_across_vcml.any():
            print('')
            print('*WARNING: {} GPIs had different output prices for different pharmacies on the same VCML'.format(
                vary_across_vcml.sum()))
            num_prices_table = lp_data_output_df.groupby(['MACLIST', 'BG_FLAG', 'GPI_NDC'], as_index=False).agg(
                {'Final_Price': 'nunique'}).rename(columns={'Final_Price': 'Num_Prices'})
            num_prices_table = lp_data_output_df.merge(num_prices_table, how='left', on=['MACLIST', 'BG_FLAG', 'GPI_NDC'])
            new_output_cols = output_cols
            new_output_cols.remove('Current MAC')
            new_output_cols.remove('MACPRC')
            new_output_cols = new_output_cols + ['Num_Prices']
            num_prices_table = num_prices_table[(num_prices_table.Num_Prices > 1)][new_output_cols]
            vcml_price_inc_df = num_prices_table.loc[(num_prices_table.Num_Prices > 1), output_cols + ['Num_Prices']]
            if len(vcml_price_inc_df) > 0:
                if False:  # p.WRITE_TO_BQ:
                    write_to_bq(
                        num_prices_table.loc[(num_prices_table.Num_Prices > 1), output_cols + ['Num_Prices']],
                        project_output=p.BQ_OUTPUT_PROJECT_ID,
                        dataset_output=p.BQ_OUTPUT_DATASET,
                        table_id="vcml_price_inconsistencies",
                        client_name_param=', '.join(sorted(p.CUSTOMER_ID)),
                        timestamp_param=p.TIMESTAMP,
                        run_id=p.AT_RUN_ID,
                        schema=None)
                else:
                    vcml_price_inc_df.to_csv(FILE_OUTPUT_PATH + 'vcml_price_inconsistencies.csv', index=False)
                    #assert len(vcml_price_inc_df) == 0, '{} GPIs had different output prices for different pharmacies on the same VCML'.format(
                #    vary_across_vcml.sum())
            else:
                qa_Prices_above_MAC1026_floor_err_list.append('{} GPIs had different output prices for different pharmacies on the same VCML'.format(
                    vary_across_vcml.sum()))
 
        else:
            print('')
            print('All GPIs on the same VCML have the same output price.')
 
        def qa_measurement_price_check(df):
            r30_r90_violating_GPIs = []
            mail_retail_violating_GPIs = []
            immutable_GPIS_report_r30_r90 = []
            immutable_GPIS_report_mail_retail = []
            qa_measurement_price_check_err_list = []
 
            #############Part 1: Discrepancy if R90 prices are more than R30 prices for a given combo of 'CLIENT','REGION','GPI_NDC','CHAIN_SUBGROUP'#################
 
            group_by_master_df = df.groupby(['CLIENT', 'REGION', 'GPI_NDC', 'CHAIN_SUBGROUP', 'MEASUREMENT', 'BG_FLAG']) \
                .agg(max_final_price=('Final_Price', 'max'), \
                     min_final_price=('Final_Price', 'min') \
                     ).reset_index()
 
            r90_group_df = group_by_master_df[group_by_master_df['MEASUREMENT'] == 'R90']
            r30_group_df = group_by_master_df[group_by_master_df['MEASUREMENT'] == 'R30']
 
            ##denormalizing the dataframe by juxtaposing r_90 and r_30 values and m_30 values together
            r90_30_group_df = r90_group_df.merge(r30_group_df, how='inner',
                                                 on=['CLIENT', 'REGION', 'GPI_NDC', 'CHAIN_SUBGROUP', 'BG_FLAG'], suffixes=(
                    '_90', '_30'))  ##1st dataframne to show R30 and R90 price comparisons
            if r90_group_df.CHAIN_SUBGROUP.str.contains('_R90OK').any():
                # In the main constraint we rewrite the R90OK subgroup to look like the "normal" one
                # In this constraint, we rewrite the R30 subgroup to look like the "OK" one so our debugging outputs make sense
                # (and also doing it two different ways is a more useful test of the logic)
                r90ok_group_df = r90_group_df.loc[r90_group_df.CHAIN_SUBGROUP.str.contains('_R90OK')]
                r30ok_group_df = r30_group_df[~(r30_group_df['CHAIN_SUBGROUP'].str.contains('_EXTRL'))].groupby(['CLIENT', 'REGION', 'GPI_NDC', 'MEASUREMENT', 'BG_FLAG'], as_index=False).agg({'min_final_price': 'min', 'max_final_price': 'max'})
                r90ok_30_group_df = r90ok_group_df.merge(r30ok_group_df, how='inner',
                                                 on=['CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG'], suffixes=(
                    '_90', '_30'))
                r90ok_30_group_df['MEASUREMENT_30'] = r90ok_30_group_df['MEASUREMENT_90'] # to ensure proper join below
                r90_30_group_df = pd.concat([r90_30_group_df, r90ok_30_group_df])
            r90_30_violation_df = r90_30_group_df[
                r90_30_group_df.max_final_price_90 > r90_30_group_df.min_final_price_30 + 0.0001]
            r90_30_group_df_merged = pd.merge(df, r90_30_violation_df, how='inner',
                                             left_on=['CLIENT', 'REGION', 'GPI_NDC', 'CHAIN_SUBGROUP', 'MEASUREMENT', 'BG_FLAG'],
                                              right_on=['CLIENT', 'REGION', 'GPI_NDC', 'CHAIN_SUBGROUP', 'MEASUREMENT_30', 'BG_FLAG'])
            r30_r90_violating_GPIs.extend(
                r90_30_group_df_merged[r90_30_group_df_merged['PRICE_MUTABLE'] == 1][['GPI_NDC','BG_FLAG']].apply(tuple, axis=1).tolist())
            immutable_GPIS_report_r30_r90.extend(
                r90_30_group_df_merged[r90_30_group_df_merged['PRICE_MUTABLE'] == 0][['GPI_NDC','BG_FLAG']].apply(tuple, axis=1).tolist())
 
            if len(r30_r90_violating_GPIs) > 0:
                r90_30_price_violation_report = df[df[['GPI_NDC','BG_FLAG']].apply(tuple, axis=1).isin(r30_r90_violating_GPIs)]
                r90_30_price_violation_report.to_csv(p.FILE_OUTPUT_PATH + 'R9030_price_REPORT.csv', index=False)
                print('WARNING: {} GPIs have R90 price > R30 price. Look at R9030_price_REPORT.csv'.format(
                    len(set(r30_r90_violating_GPIs))))
                # assert len(r30_r90_violating_GPIs) == 0, '{} GPIs have R90 price > R30 price. Look at R9030_price_REPORT.csv'.format(
                #    len(set(r30_r90_violating_GPIs)))
                qa_measurement_price_check_err_list.append('{} GPIs have R90 price > R30 price. Look at R9030_price_REPORT.csv'.format(
                    len(set(r30_r90_violating_GPIs))))
 
            if len(immutable_GPIS_report_r30_r90) > 0:
                r90_30_price_violation_report_immutable = df[df[['GPI_NDC','BG_FLAG']].apply(tuple, axis=1).isin(immutable_GPIS_report_r30_r90)]
                r90_30_price_violation_report_immutable.to_csv(p.FILE_OUTPUT_PATH + 'R9030_price_REPORT_Immutable.csv',
                                                               index=False)
                print(
                    'WARNING : {} GPIs have R90 price > R30 price that are immutable. Look at R9030_price_REPORT_Immutable.csv'.format(
                        len(set(immutable_GPIS_report_r30_r90))))
                #########################################################################################################################################################
 
            #############Part 2 Discrepancy if M30 prices are more than non M30 prices for a given combo of 'CLIENT','REGION','GPI_NDC'##############################

            # Mail may go above retail if it's allowed by the client contract
            if p.MAIL_MAC_UNRESTRICTED:
                m30_cons_cap = p.MAIL_UNRESTRICTED_CAP
            else:
                m30_cons_cap = 1.0

            # Adding m30_cons_cap to aid in diagnosing violations
            df['MAIL_UNRESTRICTED_CAP'] = m30_cons_cap
 
            group_by_master_df_2 = df.groupby(['CLIENT', 'REGION', 'GPI_NDC', 'MEASUREMENT', 'BG_FLAG']) \
                .agg(max_final_price=('Final_Price', 'max'), \
                     min_final_price=('Final_Price', 'min') \
                     ).reset_index()
 
            m30_group_df = group_by_master_df_2[group_by_master_df_2['MEASUREMENT'] == 'M30']
            m_non_30_group_df = group_by_master_df_2[group_by_master_df_2['MEASUREMENT'] != 'M30']
 
            m_30_non_30_group_df = m30_group_df.merge(m_non_30_group_df, how='inner',
                                                      on=['CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG'],
                                                      suffixes=('_m30', '_non_m30'))
 
            ###############Violation lists#####################################################################
            m_30_non_30_group_df['MAIL_CAP_PRICE'] = ((m30_cons_cap * (m_30_non_30_group_df['min_final_price_non_m30'] * 100000)).astype(int) / 100000).round(4)
            mail_retail_violation_df = m_30_non_30_group_df[
                m_30_non_30_group_df.max_final_price_m30 > m_30_non_30_group_df.MAIL_CAP_PRICE + tolerance]
            ##merging with df to fetch the PRICE_MUTABLE value for further filtering
            mail_retail_violation_df_merged = pd.merge(df, mail_retail_violation_df, how='inner',
                                                       left_on=['CLIENT', 'REGION', 'GPI_NDC', 'MEASUREMENT', 'BG_FLAG'],
                                                       right_on=['CLIENT', 'REGION', 'GPI_NDC', 'MEASUREMENT_m30', 'BG_FLAG'])

            mail_retail_violating_GPIs.extend(
                mail_retail_violation_df_merged[mail_retail_violation_df_merged['PRICE_MUTABLE'] == 1][[
                    'GPI_NDC', 'BG_FLAG']].apply(tuple, axis=1).tolist())
            immutable_GPIS_report_mail_retail.extend(
                mail_retail_violation_df_merged[mail_retail_violation_df_merged['PRICE_MUTABLE'] == 0][[
                    'GPI_NDC','BG_FLAG']].apply(tuple, axis=1).tolist())
 
            if len(mail_retail_violating_GPIs) > 0:
                MR_price_REPORT = df[df[['GPI_NDC','BG_FLAG']].apply(tuple, axis=1).isin(mail_retail_violating_GPIs)]
                MR_price_REPORT.to_csv(p.FILE_OUTPUT_PATH + 'MR_price_REPORT.csv', index=False)
                qa_measurement_price_check_err_list.append('{} GPIs have mail price > MAIL_UNRESTRICTED_CAP * retail price. Look at MR_price_REPORT.csv'.format(len(set(mail_retail_violating_GPIs))))

 
            if len(immutable_GPIS_report_mail_retail) > 0:
                mail_retail_price_violation_report_immutable = df[df[['GPI_NDC','BG_FLAG']].apply(tuple, axis=1).isin(immutable_GPIS_report_mail_retail)]
                mail_retail_price_violation_report_immutable.to_csv(
                    p.FILE_OUTPUT_PATH + 'MR_price_REPORT_Immutable.csv', index=False)
                print(
                    'WARNING : {} GPIs have mail price > MAIL_UNRESTRICTED_CAP * retail price that are immutable. Look at MR_price_REPORT_Immutable.csv'.format(
                        len(set(immutable_GPIS_report_mail_retail))))
            
            return qa_measurement_price_check_err_list
        
        def qa_brnd_gnrc_price_check(df):
            '''
            Verifies that all brand prices are more than their generic counterpart. Creates an error report if there are
            violations.
            
            Input:
            df: total output
            
            Output:
            qa_brnd_gnrc_price_check_err_list: List of QA errors where generic > brand. This will eventually get added to 
            a larger list of QA errors. If the list is empty, that means that the QA has passed
            '''            
            gnrc_brnd_violating_GPIs = []
            immutable_GPIS_report_gnrc_brnd = []
            qa_brnd_gnrc_price_check_err_list = []
 
            #############Discrepancy if generic prices are more than brand prices for a given combo of 'CLIENT','REGION','GPI_NDC','CHAIN_SUBGROUP','MEASUREMENT'##############################

            # Generic may go above brand if it's allowed by the client contract
            if p.BRAND_GENERIC_UNRESTRICTED:
                gnrc_cons_cap = p.GENERIC_UNRESTRICTED_CAP
            else:
                gnrc_cons_cap = 1.0

            # Adding gnrc_cons_cap to aid in diagnosing violations
            df['GENERIC_UNRESTRICTED_CAP'] = gnrc_cons_cap
            
            if not p.TRUECOST_CLIENT:
                df['NET_COST_GUARANTEE_UNIT'] = 1.0
                df['PCD_IDX'] = '1'
 
            group_by_master_df = df.groupby(['CLIENT', 'REGION', 'GPI_NDC', 'CHAIN_SUBGROUP', 'MEASUREMENT', 'BG_FLAG']) \
                .agg(max_final_price=('Final_Price', 'max'), \
                     min_final_price=('Final_Price', 'min'), \
                     MAC1026_UNIT_PRICE=('MAC1026_UNIT_PRICE', 'first'), \
                     NET_COST_GUARANTEE=('NET_COST_GUARANTEE_UNIT', 'first'), \
                     PCD_IDX=('PCD_IDX', 'first')
                     ).reset_index()
 
            gnrc_group_df = group_by_master_df[group_by_master_df['BG_FLAG'] == 'G']
            brnd_group_df = group_by_master_df[group_by_master_df['BG_FLAG'] == 'B']
 
            gnrc_brnd_group_df = gnrc_group_df.merge(brnd_group_df, how='inner',
                                                      on=['CLIENT', 'REGION', 'GPI_NDC', 'CHAIN_SUBGROUP', 'MEASUREMENT'],
                                                      suffixes=('_gnrc', '_brnd'),
                                                      validate='1:1')
 
            ###############Violation lists#####################################################################
            gnrc_brnd_group_df['GNRC_CAP_PRICE'] = ((gnrc_cons_cap * (gnrc_brnd_group_df['min_final_price_brnd'] * 100000)).astype(int) / 100000).round(4)
            gnrc_brnd_violation_df = gnrc_brnd_group_df[
                gnrc_brnd_group_df.max_final_price_gnrc > gnrc_brnd_group_df.GNRC_CAP_PRICE + tolerance]
            
            if p.TRUECOST_CLIENT:
                    # filter out when gen floor > brand guarantee
                    gnrc_brnd_violation_df = gnrc_brnd_violation_df[~((gnrc_brnd_violation_df['MAC1026_UNIT_PRICE_gnrc'] > gnrc_brnd_violation_df['NET_COST_GUARANTEE_brnd']))]

                    # filter out when gen PCD > brand NTM guarantee
                    gnrc_brnd_violation_df = gnrc_brnd_violation_df[~((gnrc_brnd_violation_df['PCD_IDX_brnd'].astype(str) != '1') & 
                                                                      (gnrc_brnd_violation_df['NET_COST_GUARANTEE_gnrc'] > gnrc_brnd_violation_df['NET_COST_GUARANTEE_brnd']))]
                    
            ##merging with df to fetch the PRICE_MUTABLE value for further filtering
            gnrc_brnd_violation_df_merged = pd.merge(df, gnrc_brnd_violation_df, how='inner',
                                                       left_on=['CLIENT', 'REGION', 'GPI_NDC', 'CHAIN_SUBGROUP', 'MEASUREMENT', 'BG_FLAG'],
                                                       right_on=['CLIENT', 'REGION', 'GPI_NDC', 'CHAIN_SUBGROUP', 'MEASUREMENT', 'BG_FLAG_gnrc'])

            gnrc_brnd_violating_GPIs.extend(
                gnrc_brnd_violation_df_merged[gnrc_brnd_violation_df_merged['PRICE_MUTABLE'] == 1][
                    'GPI_NDC'].tolist())
            immutable_GPIS_report_gnrc_brnd.extend(
                gnrc_brnd_violation_df_merged[gnrc_brnd_violation_df_merged['PRICE_MUTABLE'] == 0][
                    'GPI_NDC'].tolist())
 
            if len(gnrc_brnd_violating_GPIs) > 0:
                BG_price_REPORT = df[df['GPI_NDC'].isin(gnrc_brnd_violating_GPIs)]
                BG_price_REPORT.to_csv(p.FILE_OUTPUT_PATH + 'BG_price_REPORT.csv', index=False)
                qa_brnd_gnrc_price_check_err_list.append('{} GPIs have generic price > GENERIC_UNRESTRICTED_CAP * brand price. Look at BG_price_REPORT.csv'.format(len(set(gnrc_brnd_violating_GPIs))))

 
            if len(immutable_GPIS_report_gnrc_brnd) > 0:
                gnrc_brnd_price_violation_report_immutable = df[df['GPI_NDC'].isin(immutable_GPIS_report_gnrc_brnd)]
                gnrc_brnd_price_violation_report_immutable.to_csv(
                    p.FILE_OUTPUT_PATH + 'BG_price_REPORT_Immutable.csv', index=False)
                print(
                    'WARNING : {} GPIs have generic price > GENERIC_UNRESTRICTED_CAP * brand price that are immutable. Look at BG_price_REPORT_Immutable.csv'.format(
                        len(set(immutable_GPIS_report_gnrc_brnd))))
            
            return qa_brnd_gnrc_price_check_err_list
 
        qa_measurement_price_chk_err_list = qa_measurement_price_check(lp_data_output_df)
        qa_brnd_gnrc_price_chk_err_list = qa_brnd_gnrc_price_check(lp_data_output_df)

        qa_Prices_above_MAC1026_floor_err_list.extend(qa_measurement_price_chk_err_list)
        qa_Prices_above_MAC1026_floor_err_list.extend(qa_brnd_gnrc_price_chk_err_list)
        # Test: Equal package size constraints
 
        # Test: GOODRX prices
        if p.GOODRX_OPT or p.RMS_OPT or p.APPLY_GENERAL_MULTIPLIER:
            if ((lp_with_final_prices['GOODRX_UPPER_LIMIT'].round(4) < lp_with_final_prices[
                'MACPRC'] - tolerance)).sum() > 0:
                goodrx_violated = lp_with_final_prices.loc[(lp_with_final_prices['GOODRX_UPPER_LIMIT'].round(4) <
                                                            lp_with_final_prices['MACPRC'] - tolerance), :]
                goodrx_violated_mac1026 = goodrx_violated.loc[(goodrx_violated['GOODRX_UPPER_LIMIT'].round(4) <
                                                               goodrx_violated['MAC1026_UNIT_PRICE']), :]
                goodrx_violated_override = goodrx_violated.loc[(goodrx_violated['GOODRX_UPPER_LIMIT'].round(4) <
                                                                goodrx_violated['PRICE_OVRD_AMT']), :]
                goodrx_violated_unc = goodrx_violated.loc[(goodrx_violated['GOODRX_UPPER_LIMIT'].round(4) <
                                                           goodrx_violated['UNC_OVRD_AMT']), :]
                goodrx_violated_unc_rx_override = goodrx_violated.loc[goodrx_violated['RAISED_PRICE_UC'] & (
                            goodrx_violated['GOODRX_UPPER_LIMIT'].round(4) < goodrx_violated[
                        'MACPRC'] - tolerance) & p.UNC_OVERRIDE_GOODRX,
                                                  :]  # prices where we want U&C to override goodrx
 
                print(
                    'Warning: {} GPIs violate the GoodRX upper limit, {} because of the MAC1026 floor, {} because of MAC overrides , {} because of UNC overrides, {} because of U&C overriding Goodrx due to p.UNC_OVERRIDE_GOODRX being true'.format(
                        len(goodrx_violated), len(goodrx_violated_mac1026), len(goodrx_violated_override),
                        len(goodrx_violated_unc), len(goodrx_violated_unc_rx_override)))
                print("See GoodRX_price_changes_REPORT.csv in the output folder")
                unex_goodrx = len(goodrx_violated) - len(goodrx_violated_mac1026) - len(goodrx_violated_override) - len(
                    goodrx_violated_unc) - len(goodrx_violated_unc_rx_override)
                if unex_goodrx != 0:
                    goodrx_violated.to_csv(p.FILE_OUTPUT_PATH + 'GoodRX_price_changes_REPORT.csv', index=False)
                    qa_Prices_above_MAC1026_floor_err_list.append('{} GPIs violate the GoodRX upper limit not due to the MAC1026 floor or UNC Overrides'.format(unex_goodrx))

            else:
                print('No GoodRX price violations.')
 
        # Test: Interceptor prices
        if p.INTERCEPTOR_OPT:
            if ((lp_data_output_df['PRICE_MUTABLE'] == 1) &
                (lp_data_output_df['BG_FLAG'] == 'G') & # Assuming interceptor_opt only applies to generic 
                (lp_data_output_df['lb_name'] != 'UNC_OVRD_AMT') &
                (lp_data_output_df['lb_name'] != 'Immutable') &
                ((lp_data_output_df['INTERCEPT_HIGH'].round(4) < lp_data_output_df['Final_Price'] - tolerance) |
                 (lp_data_output_df['INTERCEPT_LOW'].round(4) - tolerance > lp_data_output_df[
                     'Final_Price']))).sum() > 0:
 
                interceptor_high_violated = lp_data_output_df.loc[(lp_data_output_df['PRICE_MUTABLE'] == 1) &
                                                                  (lp_data_output_df['BG_FLAG'] == 'G') &
                                                                     (lp_data_output_df['INTERCEPT_HIGH'].round(4) <
                                                                      lp_data_output_df['Final_Price'] - tolerance), :]
                interceptor_low_violated = lp_data_output_df.loc[(lp_data_output_df['PRICE_MUTABLE'] == 1) &
                                                                 (lp_data_output_df['BG_FLAG'] == 'G') &
                                                                    (lp_data_output_df['INTERCEPT_LOW'].round(
                                                                        4) - tolerance > lp_data_output_df[
                                                                         'Final_Price']), :]
                interceptor_high_violated_mac1026 = interceptor_high_violated.loc[(interceptor_high_violated[
                                                                                       'INTERCEPT_HIGH'].round(4) <
                                                                                   interceptor_high_violated[
                                                                                       'MAC1026_UNIT_PRICE']), :]
                
                interceptor_high_violated_unc_overrides = interceptor_high_violated.loc[interceptor_high_violated.INTERCEPT_HIGH.round(4) <
                                                                                      interceptor_high_violated.UNC_OVRD_AMT.round(4),:]
                
                interceptor_low_violated_wtwlimits = interceptor_low_violated.loc[interceptor_low_violated.lb_name == 'wtw_upper_limit',:]
                interceptor_low_violated_unc_overrides = interceptor_low_violated.loc[interceptor_low_violated.INTERCEPT_LOW.round(4) >
                                                                                      interceptor_low_violated.UNC_OVRD_AMT.round(4),:]
                
                interceptor_high_violated_nozbd = interceptor_high_violated.loc[interceptor_high_violated.QTY_ZBD_FRAC==0,:]
                interceptor_low_violated_nozbd = interceptor_low_violated.loc[interceptor_low_violated.QTY_ZBD_FRAC==0,:]
                
                print('*Warning: {} GPIs violate the Interceptor high limit, {} because of the MAC1026 floor, {} because of unc overrides, and {} because the ZBD fraction is 0'.format(
                len(interceptor_high_violated), len(interceptor_high_violated_mac1026), len(interceptor_high_violated_unc_overrides), len(interceptor_high_violated_nozbd)))
                print('*Warning: {} GPIs violate the Interceptor low limit, {} because of the wtw limits, {} because of the unc overrides, and {} because the ZBD fraction is 0'.format(
                    len(interceptor_low_violated), len(interceptor_low_violated_wtwlimits), len(interceptor_low_violated_unc_overrides), len(interceptor_low_violated_nozbd)))
                print("See Interceptor_price_changes_REPORT.csv in the output folder")
 
                unex_interceptor = len(interceptor_high_violated) + len(interceptor_low_violated) - \
                                    len(interceptor_high_violated_mac1026) - len(interceptor_high_violated_unc_overrides) - \
                                      len(interceptor_low_violated_wtwlimits) - len(interceptor_low_violated_unc_overrides) - \
                                      len(interceptor_low_violated_nozbd) - len(interceptor_high_violated_nozbd)
 
                if unex_interceptor > 0:
                    interceptor_high_violated['Reason'] = 'Priced above Interceptor High'
                    interceptor_high_violated.loc[(
                                interceptor_high_violated['INTERCEPT_HIGH'].round(4) < interceptor_high_violated[
                            'MAC1026_UNIT_PRICE']), 'Reason'] = 'Interceptor High below Mac 1026'
                    
                    interceptor_high_violated.loc[interceptor_high_violated.QTY_ZBD_FRAC==0,'Reason'] = 'No ZBD claims'
                    interceptor_high_violated.loc[interceptor_high_violated.INTERCEPT_HIGH.round(4) <
                                                 interceptor_high_violated.UNC_OVRD_AMT.round(4),'Reason'] = 'Intercept_High lower than UNC override'
                    
                    
                    interceptor_low_violated['Reason'] = 'Price below Interceptor Low'
                    interceptor_low_violated.loc[interceptor_low_violated.lb_name == 'wtw_upper_limit', 'Reason']\
                                                        = 'Intercept_Low higher than wtw_upper_limit'
                    
                    interceptor_low_violated.loc[interceptor_low_violated.INTERCEPT_LOW.round(4) >
                                                 interceptor_low_violated.UNC_OVRD_AMT.round(4),'Reason'] = 'Intercept_Low higher than UNC override'
                    interceptor_low_violated.loc[interceptor_low_violated.QTY_ZBD_FRAC==0,'Reason'] = 'No ZBD claims'
                    
                    interceptor_violated = interceptor_high_violated.append(interceptor_low_violated)
                    interceptor_violated.to_csv(p.FILE_OUTPUT_PATH + 'Interceptor_price_changes_REPORT.csv',
                                                index=False)
                    #assert unex_interceptor == 0, '{} GPIs violate the Interceptor bounds not due to the MAC1026 floor or wtw limits or unc overrides'.format(
                        #unex_interceptor)
                    qa_Prices_above_MAC1026_floor_err_list.append('{} GPIs violate the Interceptor bounds not due to the MAC1026 floor or wtw limits or unc overrides'.format(unex_interceptor))
                    
            else:
                print('No Interceptor price violations.')
 
            # Check if the prices are set as per Keep/Send logic for mutable gpis
 
            keep_mask = lp_with_final_prices.loc[
                (lp_with_final_prices['PRICE_MUTABLE'] == 1) & (lp_with_final_prices['ACTUAL_KEEP_SEND'] == 1.0) & (lp_with_final_prices['BG_FLAG'] == 'G')]
            send_mask = lp_with_final_prices.loc[
                (lp_with_final_prices['PRICE_MUTABLE'] == 1) & (lp_with_final_prices['ACTUAL_KEEP_SEND'] == 0.0) & (lp_with_final_prices['BG_FLAG'] == 'G')]
            keep_violation = keep_mask.loc[keep_mask['INTERCEPT_HIGH'].round(4) < keep_mask['MACPRC'] - tolerance, :]
            send_violation = send_mask.loc[send_mask['INTERCEPT_LOW'].round(4) - tolerance > send_mask['MACPRC'], :]
            send_violation_goodrx_unavailable = send_mask.loc[send_mask['VENDOR_AVAILABLE'] == False, :]
            print('*Warning: {} GPIs supposed to be kept with Caremark are priced above INTERCEPT_HIGH '.format(
                len(keep_violation)))
            print('*Warning: {} GPIs supposed to be sent to Marketplace (GoodRx etc.) are priced below INTERCEPT_LOW'.format(
                len(send_violation)))
            print('*Warning: {} GPIs without Marketplace Vendor (GoodRx etc.) information are labelled Send'.format(
                len(send_violation_goodrx_unavailable)))
            total_violation = keep_violation.append(send_violation)
 
            if len(total_violation) != 0:
                total_violation.loc[:,'Reason'] = 'Priced above INTERCEPT_HIGH'
                total_violation.loc[(total_violation['ACTUAL_KEEP_SEND'] == 0.0) & \
                                    (total_violation['VENDOR_AVAILABLE'] == False), 'Reason'] = 'Vendor data unavailable - follow keep logic'
                total_violation.loc[(total_violation['ACTUAL_KEEP_SEND'] == 0.0) & \
                                    (total_violation['VENDOR_AVAILABLE'] == True), 'Reason'] = 'Priced below INTERCEPT_LOW'
                total_violation.to_csv(p.FILE_OUTPUT_PATH + 'Interceptor_Keep_Send_Violation_REPORT.csv', index=False)
                print("See Interceptor_Keep_Send_Violation_REPORT.csv in the output folder")
                qa_Prices_above_MAC1026_floor_err_list.append('{} GPIs violate the Keep/Send Logic'.format(len(total_violation)))
 
        # Test: U&C prices go in the right direction
        if p.UNC_OPT:
            if p.UNC_PHARMACY:
                # CVS prices are changed for parity reasons only in the U&C pharmacy module, and will be checked
                # by other elements of the QA script. Exclude them from these checks
                # which may otherwise produce false positives.
                unc_price_check_df = lp_with_final_prices[lp_with_final_prices['CHAIN_GROUP'] != 'CVS']
            else:
                unc_price_check_df = lp_with_final_prices
            # Call the actual unc high threshold that was used to define MAC price
            unc_price_check_df.loc[(unc_price_check_df['CHAIN_GROUP'] == 'WMT'), 'unc_high'] = unc_price_check_df.loc[
                (unc_price_check_df['CHAIN_GROUP'] == 'WMT'), 'PCT50_UCAMT_UNIT']
            unc_price_check_df.loc[(unc_price_check_df['CHAIN_GROUP'] != 'WMT'), 'unc_high'] = unc_price_check_df.loc[
                (unc_price_check_df['CHAIN_GROUP'] != 'WMT'), 'PCT90_UCAMT_UNIT']
            # If it's COMMERCIAL WMT,then allow 10% buffer on the price check range; Else use the original logic
            # Exclude the rows with UNC overrides in place
            unc_price_check_df = unc_price_check_df.loc[unc_price_check_df.UNC_OVRD_AMT.isna(), :]
            # maintenance U&C is less strict than actual U&C optimization, and match VCML is there to ensure *other* checks pass correctly
            # Neither needs to be checked by these next functions               
            if ((unc_price_check_df['RAISED_PRICE_UC']) & (
                    ~unc_price_check_df['IS_MAINTENANCE_UC']) & (
                    ~unc_price_check_df['MATCH_VCML']) & (
                    unc_price_check_df['MACPRC'] <= unc_price_check_df['PRE_UC_MAC_PRICE']) & (
                        unc_price_check_df['ub_name'] != 'goodrx_upper_limit')).sum() > 0 or \
                    ((~unc_price_check_df['RAISED_PRICE_UC']) & (~unc_price_check_df['IS_MAINTENANCE_UC']) & (
                            ~unc_price_check_df['MATCH_VCML'])  & (unc_price_check_df['PRICE_CHANGED_UC']) & (
                            unc_price_check_df['MACPRC'] >= unc_price_check_df['PRE_UC_MAC_PRICE'])).sum() > 0:
                raised_prices_not_raised = unc_price_check_df.loc[(unc_price_check_df['RAISED_PRICE_UC']) & (
                            ~unc_price_check_df['IS_MAINTENANCE_UC']) & (~unc_price_check_df['MATCH_VCML']) & (
                            unc_price_check_df['MACPRC'] <= unc_price_check_df['PRE_UC_MAC_PRICE']) & (
                                                                              unc_price_check_df[
                                                                                  'ub_name'] != 'goodrx_upper_limit'),
                                           :]
                lowered_prices_not_lowered = unc_price_check_df.loc[(~unc_price_check_df['RAISED_PRICE_UC']) & (
                    ~unc_price_check_df['IS_MAINTENANCE_UC']) & (~unc_price_check_df['MATCH_VCML']) & (
                    unc_price_check_df['PRICE_CHANGED_UC']) & (unc_price_check_df['MACPRC'] >= unc_price_check_df[
                    'PRE_UC_MAC_PRICE']), :]
                print('')
                print(
                    '*WARNING: {} GPIs were raised by the U&C optimization but final price < original MAC price. These can be inspected in U&C_price_changes_REPORT.csv'.format(
                        len(raised_prices_not_raised)))
                print('')
                print(
                    '*WARNING: {} GPIs were lowered by the U&C optimization but final price > original MAC price. These can be inspected in U&C_price_changes_REPORT.csv'.format(
                        len(lowered_prices_not_lowered)))
                uc_price_change_df = pd.concat([raised_prices_not_raised, lowered_prices_not_lowered], axis=0)
                if len(uc_price_change_df) > 0:
                    if False:  # p.WRITE_TO_BQ:
                        write_to_bq(
                            pd.concat([raised_prices_not_raised, lowered_prices_not_lowered], axis=0),
                            project_output=p.BQ_OUTPUT_PROJECT_ID,
                            dataset_output=p.BQ_OUTPUT_DATASET,
                            table_id="UC_price_changes_REPORT",
                            client_name_param=', '.join(sorted(p.CUSTOMER_ID)),
                            timestamp_param=p.TIMESTAMP,
                            run_id=p.AT_RUN_ID,
                            schema=None
                        )
                    else:
                        uc_price_change_df.to_csv(FILE_OUTPUT_PATH + 'U&C_price_changes_REPORT.csv', index=False)
                    qa_Prices_above_MAC1026_floor_err_list.append('{} GPIs were raised and {} GPIs were lowered by the U&C optimization but the final price < original Mac price (> for lowered GPIS)'.format(len(raised_prices_not_raised), len(lowered_prices_not_lowered)))
            else:
                print('No U&C price changes went in the wrong direction.')
 
            # Test: Raised U&C prices are where they should be
            checks = (unc_price_check_df['RAISED_PRICE_UC']) & (
                unc_price_check_df['ub_name'].ne('goodrx_upper_limit')) & \
                     (~unc_price_check_df['IS_MAINTENANCE_UC']) & (~unc_price_check_df['MATCH_VCML']) & \
                     (
                             ((unc_price_check_df['MACPRC'] - unc_price_check_df['Final_Price']).abs() > tolerance) | \
                             ((unc_price_check_df['MACPRC'] - unc_price_check_df['lb']).abs() > tolerance) | \
                             ((unc_price_check_df['MACPRC'] - unc_price_check_df['ub']).abs() > tolerance) | \
                             ((unc_price_check_df['MACPRC'] - unc_price_check_df[
                                 'CURRENT_MAC_PRICE']).abs() > tolerance) | \
                             (np.round(unc_price_check_df['MACPRC'], 4) <= np.round(
                                 unc_price_check_df['PRE_UC_MAC_PRICE'], 4) - tolerance) | \
                             (np.round(unc_price_check_df['MACPRC'], 4) <= np.round(unc_price_check_df['OLD_MAC_PRICE'],
                                                                                    4) - tolerance)
                     )
 
            if checks.sum() > 0:
                raised_prices_to_inspect = unc_price_check_df.loc[checks, :]
                print(
                    '*WARNING: {} GPIs raised by U&C optimization may not have retained the correct value. These can be inspected in U&C_price_raises_REPORT.csv'.format(
                        len(raised_prices_to_inspect)))
                if len(raised_prices_to_inspect) > 0:
                    if False:  # p.WRITE_TO_BQ:
                        write_to_bq(
                            raised_prices_to_inspect,
                            project_output=p.BQ_OUTPUT_PROJECT_ID,
                            dataset_output=p.BQ_OUTPUT_DATASET,
                            table_id="UC_price_raises_REPORT",
                            client_name_param=', '.join(sorted(p.CUSTOMER_ID)),
                            timestamp_param=p.TIMESTAMP,
                            run_id=p.AT_RUN_ID,
                            schema=None
                        )
                    else:
                        raised_prices_to_inspect.to_csv(FILE_OUTPUT_PATH + 'U&C_price_raises_REPORT.csv', index=False)
                    qa_Prices_above_MAC1026_floor_err_list.append('{} GPIs raised by U&C optimization may not have retained the correct value. These can be inspected in U&C_price_raises_REPORT.csv'.format(len(raised_prices_to_inspect)))
            else:
                print('All U&C price raises retained the expected values.')
 
            # Test: Lowered U&C prices are below the upper U&C bound
            # Additional tests guaranteeing that the current price change is a drop have been commented out due to the simulation code
            # As drugs lowered below the UNC ceiling may increase to the ceiling in subsequent runs and would fail as a result
 
            checks = (unc_price_check_df['PRICE_CHANGED_UC']) & (~unc_price_check_df['RAISED_PRICE_UC']) & \
                     (~unc_price_check_df['IS_MAINTENANCE_UC']) & (~unc_price_check_df['MATCH_VCML']) & \
                     (
                             ((unc_price_check_df['MACPRC'] - unc_price_check_df['Final_Price']).abs() > tolerance) | \
                             # (np.round(unc_price_check_df['MACPRC'], 4) >= np.round(unc_price_check_df['MAC_PRICE_UNIT_ADJ'], 4) + tolerance) | \
                             # (np.round(unc_price_check_df['MACPRC'], 4) >= np.round(unc_price_check_df['CURRENT_MAC_PRICE'], 4) + tolerance) | \
                             (np.round(unc_price_check_df['MACPRC'], 4) >= np.round(
                                 unc_price_check_df['PRE_UC_MAC_PRICE'], 4) + tolerance) | \
                             # (np.round(unc_price_check_df['MACPRC'], 4) >= np.round(unc_price_check_df['OLD_MAC_PRICE'], 4) + tolerance) | \
                             (np.round(unc_price_check_df['MACPRC'], 4) >= np.round(
                                 unc_price_check_df['MAC_PRICE_UPPER_LIMIT_UC'], 4) + tolerance)
                     )
 
            if checks.sum() > 0:
                lowered_prices_to_inspect = unc_price_check_df.loc[checks, :]
                print(
                    '*WARNING: {} GPIs lowered by U&C optimization may not have retained an appropriate value. These can be inspected in U&C_price_lowered_REPORT.csv'.format(
                        len(lowered_prices_to_inspect)))
                if len(lowered_prices_to_inspect) > 0:
                    if False:  # p.WRITE_TO_BQ:
                        write_to_bq(
                            lowered_prices_to_inspect,
                            project_output=p.BQ_OUTPUT_PROJECT_ID,
                            dataset_output=p.BQ_OUTPUT_DATASET,
                            table_id="UC_price_lowered_REPORT",
                            client_name_param=', '.join(sorted(p.CUSTOMER_ID)),
                            timestamp_param=p.TIMESTAMP,
                            run_id=p.AT_RUN_ID,
                            schema=None
                        )
                    else:
                        lowered_prices_to_inspect.to_csv(p.FILE_OUTPUT_PATH + 'U&C_price_lowered_REPORT.csv',
                                                         index=False)
               
                    qa_Prices_above_MAC1026_floor_err_list.append('{} GPIs lowered by U&C optimization may not have retained an appropriate value. These can be inspected in U&C_price_lowered_REPORT.csv'.format(len(lowered_prices_to_inspect)))
            else:
                print('All lowered U&C prices retained appropriate values.')
 
            # Output report of U&C GPIs that conflict with GoodRx Optimization
            if ((unc_price_check_df['RAISED_PRICE_UC']) & (
                    unc_price_check_df['MACPRC'] <= unc_price_check_df['PRE_UC_MAC_PRICE']) & (
                        unc_price_check_df['ub_name'] == 'goodrx_upper_limit')).sum() > 0:
                goodrx_conflict_gpis = unc_price_check_df.loc[(unc_price_check_df['RAISED_PRICE_UC']) & (
                            unc_price_check_df['MACPRC'] <= unc_price_check_df['PRE_UC_MAC_PRICE']) & (
                                                                          unc_price_check_df[
                                                                              'ub_name'] == 'goodrx_upper_limit'), :]
                print('')
                print(
                    '*WARNING: {} GPIs were raised by the U&C optimization but capped by GoodRx Upper Limit. These can be inspected in U&C_GRX_conflicts_REPORT.csv'.format(
                        len(goodrx_conflict_gpis)))
                print('')
                goodrx_conflict_gpis.to_csv(p.FILE_OUTPUT_PATH + 'U&C_GRX_conflicts_REPORT.csv', index=False)
 
        # Test: non-mutable prices don't change
        # TODO: CHANGE PATH TO THE CORRECT ONE
        if p.FLOOR_PRICE:
            floor_gpis = pd.read_csv(p.FILE_INPUT_PATH + p.FLOOR_GPI_LIST, dtype=p.VARIABLE_TYPE_DIC)[['GPI']]
            if 'BG_FLAG' not in floor_gpis.columns:
                floor_gpis['BG_FLAG'] = 'G'
            floor_gpis['BG_GPI'] = floor_gpis['BG_FLAG'] + floor_gpis['GPI']
        else:
            floor_gpis = pd.DataFrame({'BG_GPI': []})
 
        mac_price_override = standardize_df(
            pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_PRICE_OVERRIDE_FILE, dtype=p.VARIABLE_TYPE_DIC))
        mac_price_override["MACLIST_BG_GPI"] = mac_price_override['VCML_ID'] + mac_price_override['BG_FLAG'] + mac_price_override['GPI']

 
        if p.PRICE_OVERRIDE:
            price_override = standardize_df(
                pd.read_csv(os.path.join(p.FILE_INPUT_PATH, p.PRICE_OVERRIDE_FILE), dtype=p.VARIABLE_TYPE_DIC))
            price_override["MACLIST_BG_GPI"] = price_override['VCML_ID'] + price_override['BG_FLAG'] + price_override['GPI']
 
        if ((lp_data_output_df['PRICE_MUTABLE'] != 1) & (~lp_data_output_df['RAISED_PRICE_UC']) & (
                np.round(lp_data_output_df['OLD_MAC_PRICE'], 4) != np.round(lp_data_output_df['Final_Price'],
                                                                            4))).sum() > 0:
            immutable_price_changes = lp_with_final_prices.loc[(lp_with_final_prices['PRICE_MUTABLE'] != 1), :]
            immutable_price_changes['REASON_CODE'] = 'unknown'
            
            if p.OUTPUT_FULL_MAC:
                immutable_price_changes.loc[immutable_price_changes['OLD_MAC_PRICE'] == immutable_price_changes['Final_Price'], 'REASON_CODE'] = 'No price change, p.OUTPUT_FULL_MAC=True'
                
            immutable_price_changes.loc[immutable_price_changes[
                'RAISED_PRICE_UC'], 'REASON_CODE'] = 'UC algorithm raised price and set to immutable'
            immutable_price_changes.loc[(immutable_price_changes['BG_FLAG']+immutable_price_changes['GPI']).isin(floor_gpis['BG_GPI']), 'REASON_CODE'] = 'Floor GPI'
            immutable_price_changes.loc[immutable_price_changes['MACLIST_BG_GPI'].isin(
                mac_price_override['MACLIST_BG_GPI']), 'REASON_CODE'] = 'Mac Price Override'
            immutable_price_changes = immutable_price_changes.merge(
                mac_price_override[['MACLIST_BG_GPI', 'PRICE_OVRD_AMT']], on='MACLIST_BG_GPI', how='left')
            if p.PRICE_OVERRIDE:
                immutable_price_changes.loc[immutable_price_changes['MACLIST_BG_GPI'].isin(
                    price_override['MACLIST_BG_GPI']), 'REASON_CODE'] = 'Price Override'
                immutable_price_changes = immutable_price_changes.merge(
                    price_override[['MACLIST_BG_GPI', 'PRICE_OVRD_AMT']], on='MACLIST_BG_GPI', how='left')
            unknowns = len(immutable_price_changes[immutable_price_changes['REASON_CODE'] == 'unknown'])
 
            print('')
            print(
                '*WARNING: {} immutable GPIs recorded a price change. These can be inspected in immutable_price_changes_REPORT.csv'.format(
                    len(immutable_price_changes)))
            if unknowns:
                if False:  # p.WRITE_TO_BQ:
                    write_to_bq(
                        immutable_price_changes,
                        project_output=p.BQ_OUTPUT_PROJECT_ID,
                        dataset_output=p.BQ_OUTPUT_DATASET,
                        table_id="immutable_price_changes_REPORT",
                        client_name_param=', '.join(sorted(p.CUSTOMER_ID)),
                        timestamp_param=p.TIMESTAMP,
                        run_id=p.AT_RUN_ID,
                        schema=None
                    )
                else:
                    immutable_price_changes.to_csv(FILE_OUTPUT_PATH + 'immutable_price_changes_REPORT.csv', index=False)
                qa_Prices_above_MAC1026_floor_err_list.append('{} immutable GPIs recorded a price change for unknown reasons'.format(unknowns))
        else:
            print('All price changes occurred with mutable GPIs.')
 
        # Test: all GPIs in price_override are non-mutable
 
        # TODO:  This whole area need to be updated to reflect what the code now does, further changes might be needed
 
        exclusions = standardize_df(
            pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.SPECIALTY_EXCLUSION_FILE, dtype=p.VARIABLE_TYPE_DIC))

        #Every client, regardless of LOB, must have specific exclusions and should never be left empty
        if len(exclusions) == 0: 
            print('\n ***** ALERT: Exclusions is empty')
            assert False
            
        mac_price_override = standardize_df(
            pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_PRICE_OVERRIDE_FILE, dtype=p.VARIABLE_TYPE_DIC))
        
        ndc_prices = mac_price_override[mac_price_override['NDC'] != '***********']
        ndc_prices = ndc_prices.drop_duplicates(subset=['CLIENT', 'REGION', 'GPI','BG_FLAG']).drop(columns=['VCML_ID', 'NDC'])
 
        single_vcml = mac_price_override.groupby(['CLIENT', 'REGION', 'GPI', 'BG_FLAG']).agg({'VCML_ID': 'count'}).reset_index()
        no_vclm = mac_price_override['VCML_ID'].unique().shape[0]
        single_vcml = single_vcml[single_vcml['VCML_ID'] != no_vclm].drop(columns=['VCML_ID'])
 
        exclusions = pd.concat([exclusions, ndc_prices, single_vcml])
 
        exclusions = exclusions[(exclusions.CLIENT == p.CUSTOMER_ID[0]) |  # for commercial
                                (exclusions.CLIENT == uf.get_formatted_client_name(p.CUSTOMER_ID)) |  # for MedD
                                (exclusions.CLIENT == 'ALL')]
        exclusions['CLIENT'] = lp_data_output_df['CLIENT'].unique()[0]
        # below won't work for MedD clients. Therefore, Trying to remove REGION from all subsequent operations
        # this is because the current version of gpi exclusion file and mac price override file have region = 'ALL'
        # while the lp_data_output has a unique region for Commercial clients and potentially multiple regions for MedD clients
        # exclusions['REGION'] = lp_data_output_df['REGION'].unique()[0]
 
        if p.PRICE_OVERRIDE:
            price_override = standardize_df(
                pd.read_csv(p.FILE_INPUT_PATH + p.PRICE_OVERRIDE_FILE, dtype=p.VARIABLE_TYPE_DIC))
            # exclusions = pd.concat([exclusions, price_override])
            mac_price_override = pd.concat([mac_price_override, price_override])
            # mac_price_override['VCML_ID'] = mac_price_override['VCML_ID'].astype(np.int64)
 
        # exclusions['CLIENT'] = np.where(exclusions['CLIENT'] == 'ALL', lp_data_output_df['CLIENT'].unique()[0], exclusions['CLIENT'])
        # exclusions['REGION'] = np.where(exclusions['REGION'] == 'ALL', lp_data_output_df['REGION'].unique()[0], exclusions['REGION'])
 
        lp_data_excluded_temp1 = lp_data_output_df.merge(exclusions.drop(columns=['REGION'], axis=1),
                                                         on=['CLIENT', 'GPI', 'BG_FLAG'],
                                                         how='inner')
        lp_data_excluded_temp2 = lp_data_output_df.merge(mac_price_override.drop(columns=['REGION'], axis=1),
                                                         left_on=['CLIENT', 'MAC_LIST', 'GPI', 'BG_FLAG'],
                                                         right_on=['CLIENT', 'VCML_ID', 'GPI', 'BG_FLAG'],
                                                         how='inner')
 
        # lp_data_excluded = lp_data_output_df.merge(exclusions, on = ['CLIENT', 'REGION', 'GPI'], how = 'inner')
        lp_data_excluded = pd.concat([lp_data_excluded_temp1, lp_data_excluded_temp2])
 
        if p.FLOOR_PRICE:
            lp_data_excluded = lp_data_excluded[~(lp_data_excluded['BG_FLAG'] + lp_data_excluded['GPI']).isin(floor_gpis['BG_GPI'])]
 
        if (lp_data_excluded.PRICE_MUTABLE == 1).any():
            new_cols = [x for x in output_cols if x not in ['GPI_NDC', 'MACPRC', 'Current MAC']]
            lp_data_mutable_exclusions = lp_data_excluded.loc[(lp_data_excluded.PRICE_MUTABLE == 1), new_cols]
            num_exclusions = lp_data_mutable_exclusions.groupby(['CLIENT', 'REGION', 'GPI','BG_FLAG']).ngroups
            print('')
            print(
                '*WARNING: {} GPIs should have been excluded but were kept mutable. These can be inspected in mutable_exclusions_REPORT.csv'.format(
                    num_exclusions))
            if num_exclusions:
                if False:  # p.WRITE_TO_BQ:
                    write_to_bq(
                        lp_data_mutable_exclusions,
                        project_output=p.BQ_OUTPUT_PROJECT_ID,
                        dataset_output=p.BQ_OUTPUT_DATASET,
                        table_id="mutable_exclusions_REPORT",
                        client_name_param=', '.join(sorted(p.CUSTOMER_ID)),
                        timestamp_param=p.TIMESTAMP,
                        run_id=p.AT_RUN_ID,
                        schema=None
                    )
                else:
                    lp_data_mutable_exclusions.to_csv(FILE_OUTPUT_PATH + 'mutable_exclusions_REPORT.csv', index=False)
                qa_Prices_above_MAC1026_floor_err_list.append('{} immutable GPIs recorded a price change for unknown reasons'.format(unknowns))
        else:
            print('All excluded GPIs were correctly set as immutable.')
 
        if (lp_with_final_prices['Final_Price'] < (lp_with_final_prices['MAC1026_UNIT_PRICE'] - tolerance)).sum() > 0:
            outofbounds = (lp_with_final_prices['Final_Price'] < (
                        lp_with_final_prices['MAC1026_UNIT_PRICE'] - tolerance)).sum()
            out1026 = lp_with_final_prices[
                lp_with_final_prices['Final_Price'] < (lp_with_final_prices['MAC1026_UNIT_PRICE'] - tolerance)]
            out1026.loc[:, 'REASON_CODE'] = 'unknown'
            out1026.loc[out1026['MEASUREMENT'].isin(
                ['M30']), 'REASON_CODE'] = 'M30 measurement, Mail prices can be less than MAC1026 floor'
            out1026.loc[(out1026['BG_FLAG'] + out1026['GPI']).isin(
                (lp_data_excluded['BG_FLAG'] + lp_data_excluded['GPI'])), 'REASON_CODE'] = 'GPI excluded or overwriten by one of the 3 possilbe files'
            if p.OUTPUT_FULL_MAC:
                out1026.loc[(out1026['lb_name'] == 'Immutable') &
                            (out1026['OLD_MAC_PRICE'] == out1026['Final_Price']) &
                            (out1026['REASON_CODE'] == 'unknown'), 'REASON_CODE'] = 'Immutable, p.OUTPUT_FULL_MAC=True'
            unknowns = (out1026['REASON_CODE'] == 'unknown').sum()
            if p.INTERCEPTOR_OPT:
                out1026.loc[out1026['INTERCEPT_HIGH'] < out1026[
                    'MAC1026_UNIT_PRICE'], 'REASON_CODE'] = 'Intercept high set below MAC1026'
                intercept_outofbounds = (out1026['REASON_CODE'] == 'Intercept high set below MAC1026').sum()
            print('')
            print(
                '*Warning: there are {} prices below MAC_1026, of which {} are of unknown reasons. check MAC_1026_REPORT.csv'.format(
                    outofbounds, unknowns))
            if unknowns:
                if False:  # p.WRITE_TO_BQ:
                    write_to_bq(
                        out1026,
                        project_output=p.BQ_OUTPUT_PROJECT_ID,
                        dataset_output=p.BQ_OUTPUT_DATASET,
                        table_id="MAC_1026_REPORT",
                        client_name_param=', '.join(sorted(p.CUSTOMER_ID)),
                        timestamp_param=p.TIMESTAMP,
                        run_id=p.AT_RUN_ID,
                        schema=None
                    )
                else:
                    out1026.to_csv(FILE_OUTPUT_PATH + 'MAC_1026_REPORT.csv', index=False)
                qa_Prices_above_MAC1026_floor_err_list.append('There are {} prices below MAC_1026 for unknown reasons. Check MAC_1026_REPORT.csv'.format(unknowns))
                if p.INTERCEPTOR_OPT:
                    if intercept_outofbounds != 0:
                        qa_Prices_above_MAC1026_floor_err_list.append('There are {} Intercept_high bounds set below MAC_1026 . Check MAC_1026_REPORT.csv'.format(intercept_outofbounds))

 
        else:
            print('')
            print('No prices below MAC_1026.')
 
        # Test that the UNC Prices are set correctly as per the UNC price freeze
        unc_override_mask = lp_with_final_prices.loc[lp_with_final_prices.UNC_OVRD_AMT.notna(), :]
        unc_override_violation = unc_override_mask.loc[
            (unc_override_mask.Final_Price - unc_override_mask.UNC_OVRD_AMT).abs() > tolerance]
 
        if len(unc_override_violation) > 0:
            unc_override_violation.to_csv(FILE_OUTPUT_PATH + 'UNC_Price_Freeze_Violation_REPORT.csv', index=False)
            qa_Prices_above_MAC1026_floor_err_list.append('{} GPIs have prices different from the UNC price overrides. Check UNC_Price_Freeze_Violation_REPORT.csv'.format(len(unc_override_violation)))
        else:
            print('')
            print('All UNC Overrides are set correctly')
  
        # Test: All prices on ZBD gpi list at or below ZBD upper limit
        if p.ZBD_OPT:
            zbd_opt_violated = lp_with_final_prices.loc[(lp_with_final_prices['ZBD_UPPER_LIMIT'].round(4) <
                                                         lp_with_final_prices['Final_Price'] - tolerance), :]
            if len(zbd_opt_violated) > 0:
                zbd_opt_violated.loc[:, 'REASON'] = 'unknown'
                if p.UNC_OPT:
                    zbd_opt_violated.loc[
                        zbd_opt_violated['RAISED_PRICE_UC'] == True, 'REASON'] = 'Price Increased due to UNC Logic'
                zbd_opt_violated.loc[
                    (zbd_opt_violated['ZBD_UPPER_LIMIT'].round(4) < zbd_opt_violated['MAC1026_UNIT_PRICE']) & (
                                zbd_opt_violated[
                                    'MEASUREMENT'] != 'M30'), 'REASON'] = 'MAC1026 is higher than ZBD_UPPER_LIMIT'
                zbd_opt_violated.loc[(zbd_opt_violated['ZBD_UPPER_LIMIT'].round(4) < zbd_opt_violated[
                    'UNC_OVRD_AMT']), 'REASON'] = 'UNC Override is higher than ZBD_UPPER_LIMIT'
                zbd_opt_violated.loc[zbd_opt_violated['ub_name'] == 'Immutable', 'REASON'] = 'Price Immutable'
                zbd_opt_violated.to_csv(p.FILE_OUTPUT_PATH + 'ZBD_price_changes_REPORT.csv', index=False)
                unknowns = (zbd_opt_violated['REASON'] == 'unknown').sum()
                if unknowns > 0:
                    print(
                        "Warning: {} GPIs violate the ZBD upper limit, of which {} are because of unknown reasons".format(
                            len(zbd_opt_violated), unknowns))
                    print("See ZBD_price_changes_REPORT.csv in the output folder")
                    #assert unknowns == 0, '{} GPIs violated the ZBD upper limit not due to the MAC1026 floor or UNC prices'.format(unknowns)
                    qa_Prices_above_MAC1026_floor_err_list.append('{} GPIs violated the ZBD upper limit not due to the MAC1026 floor or UNC prices'.format(unknowns))
                    
                else:
                    print("No ZBD optimization violations.")

        if len(qa_Prices_above_MAC1026_floor_err_list) > 0: 
            print(qa_Prices_above_MAC1026_floor_err_list)
            assert False, qa_Prices_above_MAC1026_floor_err_list 
        else: 
            print("qa_Prices_above_MAC1026_floor_err passed")
 
        # Test: All other pharmacies >= 90% CVS pricing
 
    except Exception as e:
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'QA Prices Above MAC1026 Floor', repr(e), error_loc)
        raise e

# =============================================================================
#     list(lp_data_output_df.columns)
#     ndc_check = lp_data_output_df.NDC11
# =============================================================================

def qa_pref_nonpref_pharm_pricing(
        params_in: str,
        lp_data_output_df_in: InputPath('pickle'),
        lp_with_final_prices_in: InputPath('pickle'),
        output_cols_in: InputPath('pickle')
):
    import os
    import pickle
    import pandas as pd
    import numpy as np
    from types import ModuleType
    from util_funcs import write_params, write_to_bq, read_BQ_data
    if isinstance(params_in, ModuleType):
        p = params_in
    else:
        write_params(params_in)
        import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df
    import util_funcs as uf
    from CPMO_shared_functions import update_run_status
    try:
        # read input
        with open(lp_data_output_df_in, 'rb') as f:
            lp_data_output_df = pd.read_pickle(f)
        with open(lp_with_final_prices_in, 'rb') as f:
            lp_with_final_prices = pd.read_pickle(f)
        with open(output_cols_in, 'rb') as f:
            output_cols = pickle.load(f)

        FILE_OUTPUT_PATH = p.FILE_OUTPUT_PATH

        '''
        adding new QA that uses total output file for CVS < all other pharmacies check instead of price change file and checks violations across measurement levels and
        for every client, for every region, for every GPI_NDC, for every measurement.
        We also check if CVS is a preferred pharmacy, then CVS is compared against all other pharmacies, if not CVS is compared against non-preferred pharmacies.
        This QA also creates a report when prices are immutable but they violate this check. (if prices are mutable, QA should triggered)
        '''

        def pharm_comp(df, tolerance = 0.01):
            violating_GPIs = {'cvs_ind': [], 'immutable_cvs_ind': []}

            grouped_df = df.groupby(['CLIENT', 'REGION', 'GPI_NDC', 'MEASUREMENT', 'BG_FLAG'])

            for (client, region, GPI_NDC, measurement, bg), group in grouped_df:
                cvs_parity_subgroup = 'CVSSP' if ((group['CHAIN_SUBGROUP'] == 'CVSSP').any()) else 'CVS'
                cvs_ind_violations = []

                if group['PHARMACY_TYPE'].nunique() != 1:
                    for pharmacy_type, type_group in group.groupby('PHARMACY_TYPE'):
                        df_pref = type_group[type_group['CHAIN_SUBGROUP'] == cvs_parity_subgroup]
                        
                        if (p.CLIENT_TYPE == 'COMMERCIAL' or p.CLIENT_TYPE == 'MEDICAID'):
                            df_oth = type_group[type_group['CHAIN_SUBGROUP'] != cvs_parity_subgroup]
                        elif p.CLIENT_TYPE == 'MEDD':
                            df_oth = type_group[type_group['CHAIN_GROUP'].isin(p.NON_CAPPED_PHARMACY_LIST + p.PSAO_LIST)]

                        if not df_pref.empty and not df_oth.empty:
                            cvs_price = round(float(np.max(df_pref['Final_Price'])), 2)
                            oth_price = round(float(np.min(df_oth['Final_Price'])), 2)
                            
                            if (p.CLIENT_TYPE == 'COMMERCIAL' or p.CLIENT_TYPE == 'MEDICAID'):
                                if cvs_price > oth_price:
                                    cvs_ind_violations.append((GPI_NDC, bg))
                            elif p.CLIENT_TYPE == 'MEDD':
                                if (0.7*cvs_price) - oth_price  > tolerance:
                                    cvs_ind_violations.append((GPI_NDC, bg))

                if group['PRICE_MUTABLE'].values[0] == 1 and cvs_ind_violations:
                    violating_GPIs['cvs_ind'].extend(cvs_ind_violations)
                elif group['PRICE_MUTABLE'].values[0] == 0 and cvs_ind_violations:
                    violating_GPIs['immutable_cvs_ind'].extend(cvs_ind_violations)

            for measurement, GPIs in violating_GPIs.items():
                if GPIs:
                    violation_report = df[df[['GPI_NDC','BG_FLAG']].apply(tuple, axis=1).isin(GPIs)]
                    violation_report.to_csv(f'{p.FILE_OUTPUT_PATH}{measurement}_REPORT.csv', index=False)
                    if measurement == 'cvs_ind':
                        print(
                            f'*WARNING: {len(set(GPIs))} GPIs had CVS pricing > all other pharmacy pricing. These can be inspected in {measurement}_REPORT.csv')
                        assert len(
                            GPIs) == 0, f'{len(set(GPIs))} GPIs had CVS pricing > all other pharmacy pricing. These can be inspected in {measurement}_REPORT.csv'
                    elif measurement == 'immutable_cvs_ind':
                        print(
                            f'*WARNING: {len(set(GPIs))} GPIs had CVS pricing > all other pharmacy pricing that are immutable. These can be inspected in {measurement}_REPORT.csv')


        def generate_awp_report(df):

            df = df[(df.PRICE_MUTABLE == 1) & (df.MEASUREMENT != 'M30')]
            df.loc[:, 'AWP'] = df[['FULLAWP_ADJ', 'FULLAWP_ADJ_PROJ_LAG', 'FULLAWP_ADJ_PROJ_EOY']].sum(axis=1)

            group_cols = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'BG_FLAG']
            df = df[group_cols + ['GPI', 'CHAIN_GROUP', 'CURRENT_MAC_PRICE', 'AWP']].reset_index(drop=True)
            tot_awp = df.AWP.sum()

            df_awp = df.groupby(group_cols + ['CHAIN_GROUP']).agg(TOT_AWP=('AWP', sum)).reset_index()
            df_cvs = df.merge(df[df.CHAIN_GROUP == 'CVS'], how='left', on=group_cols + ['GPI'], suffixes=('', '_CVS'))

            mask = df_cvs.CURRENT_MAC_PRICE_CVS > df_cvs.CURRENT_MAC_PRICE
            if len(df_cvs[mask]) > 0:
                wrong_awp_chain = df_cvs[mask].groupby(group_cols + ["CHAIN_GROUP"]). \
                    agg(GPI=('GPI', pd.Series.nunique), WRONG_AWP=('AWP', sum)).reset_index()

                wrong_awp_cvs = df_cvs[mask][group_cols + ["CHAIN_GROUP_CVS", 'GPI', 'AWP_CVS']].drop_duplicates() \
                    .groupby(group_cols + ["CHAIN_GROUP_CVS"]) \
                    .agg(GPI=('GPI', pd.Series.nunique), WRONG_AWP=('AWP_CVS', sum)) \
                    .reset_index().rename(columns={'CHAIN_GROUP_CVS': 'CHAIN_GROUP'})

                wrong_awp = pd.concat([wrong_awp_chain, wrong_awp_cvs], ignore_index=True)

                report_df = df_awp.merge(wrong_awp, how='left', on=group_cols + ['CHAIN_GROUP'])
                report_df['%_WRONG_AWP'] = report_df['WRONG_AWP'] / report_df['TOT_AWP'] * 100
                report_df['%_TOT_WRONG_AWP'] = report_df['WRONG_AWP'] / tot_awp * 100


                print('{:.2f}% of mutable Retail AWP was incorrect as current CVS Price > Chain Price.'.format(
                    report_df['%_TOT_WRONG_AWP'].sum()))


                report_df.to_csv(
                    os.path.join(p.FILE_OUTPUT_PATH, "CVS_Allpharm_mismatch_awp_report_" + p.DATA_ID + ".csv"),
                    index=False)


                if p.WRITE_TO_BQ:
                    write_to_bq(report_df,
                                project_output=p.BQ_OUTPUT_PROJECT_ID,
                                dataset_output=p.BQ_OUTPUT_DATASET,
                                table_id="WRONG_AWP_REPORT",
                                client_name_param=', '.join(sorted(p.CUSTOMER_ID)),
                                timestamp_param=p.TIMESTAMP,
                                run_id=p.AT_RUN_ID,
                                schema=None)

        pharm_comp(lp_data_output_df)
        generate_awp_report(lp_data_output_df)

        # Test to be written later: check strength consistency constraints

        # Test: check preferred < non-preferred

        if False:  # p.READ_FROM_BQ: Diego, Marcel added this as temporary CAN WE DELETE?
            pref_pharm_list = read_BQ_data(pref_pharm_list, project_id=p.BQ_INPUT_PROJECT_ID,
                                           dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP, table_id='pref_pharm_list')
            pref_pharm_list = standardize_df(pref_pharm_list)
        else:
            pref_pharm_list = standardize_df(
                pd.read_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.PREFERRED_PHARM_FILE), dtype=p.VARIABLE_TYPE_DIC))


        pref_pharm_list['PREF_PHARM'] = pref_pharm_list.PREF_PHARM.apply(lambda x: x.split(','))


        def pref_nonpref(x, pref):
            if x.CHAIN_GROUP.nunique() != 1:
                # x.flag = x.CHAIN_GROUP.apply(lambda x: 'PREF' if x in pref else 'NONPREF')
                flag = np.where(x.CHAIN_GROUP.isin(pref), 'PREF', 'NONPREF')
                if ((flag == 'PREF').any()):
                    if np.round(float(np.max(x.Final_Price[flag == 'PREF'])), 3) > np.round(
                            float(np.min(x.Final_Price[flag == 'NONPREF'])), 3):
                        return True
                else:
                    return False
            else:
                return False


        pref_nonpref_check = pd.DataFrame()
        for client in lp_data_output_df.CLIENT.unique():
            for region in lp_data_output_df.loc[(lp_data_output_df.CLIENT == client) & (
            ~lp_data_output_df.BREAKOUT.str.contains('M')), 'REGION'].unique():
                preferred_chains_temp = pref_pharm_list.loc[
                    (pref_pharm_list.CLIENT == client) & (pref_pharm_list.REGION == region), 'PREF_PHARM'].values
                preferred_chains = []
                for item in preferred_chains_temp:
                    if item[0] not in ['none', 'None', 'NONE']:
                        preferred_chains += list(item)
                pref_temp = lp_data_output_df.loc[
                            (lp_data_output_df.CLIENT == client) & (lp_data_output_df.REGION == region) & (
                                ~lp_data_output_df.BREAKOUT.str.contains('M')) & (
                                        lp_data_output_df.PRICE_MUTABLE == 1), :] \
                    .groupby(['CLIENT', 'MEASUREMENT', 'GPI_NDC','BG_FLAG']) \
                    .filter(lambda x: pref_nonpref(x, preferred_chains))
                pref_nonpref_check = pref_nonpref_check.append(pref_temp, ignore_index=False)


        if len(pref_nonpref_check) > 0:
            num_gpis_violate = pref_nonpref_check.groupby(['CLIENT', 'REGION', 'MEASUREMENT', 'GPI_NDC','BG_FLAG']).ngroups
            print('')
            print(
                '*WARNING: {} GPIs were found to have preferred pharmacy pricing > non-preferred. These can be inspected in pref_nonpref_REPORT.csv'.format(
                    num_gpis_violate))


            # removing columns that are not used in the report
            output_cols.remove('Current MAC')
            output_cols.remove('MACPRC')


            if num_gpis_violate:
                if False:  # p.WRITE_TO_BQ:
                    write_to_bq(
                        pref_nonpref_check[output_cols],
                        project_output=p.BQ_OUTPUT_PROJECT_ID,
                        dataset_output=p.BQ_OUTPUT_DATASET,
                        table_id="pref_nonpref_REPORT",
                        client_name_param=', '.join(sorted(p.CUSTOMER_ID)),
                        timestamp_param=p.TIMESTAMP,
                        run_id=p.AT_RUN_ID,
                        schema=None
                    )
                else:
                    pref_nonpref_check[output_cols].to_csv(FILE_OUTPUT_PATH + 'pref_nonpref_REPORT.csv', index=False)
                assert num_gpis_violate == 0, '{} GPIs were found to have preferred pharmacy pricing > non-preferred. These can be inspected in pref_nonpref_REPORT.csv'.format(
                    num_gpis_violate)
        else:
            print('')
            print('All GPIs correctly had preferred pharmacy pricing less than non-preferred pharmacy pricing.')
    except Exception as e:
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'QA Pref/NonPref Pharm Pricing', repr(e), error_loc)
        raise e

def qa_r90_as_mail(
    params_in: str,
    lp_data_output_df_in: InputPath('pickle')
):
    """
    QA to check that if p.R90_AS_MAIL is set, all mutable R90 prices actually got MAIL pricing. Test is skipped entirely if this parameter
    is False.
    """
    import pickle
    import pandas as pd
    from types import ModuleType
    from util_funcs import write_params
    if isinstance(params_in, ModuleType):
        p = params_in
    else:
        write_params(params_in)
        import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df, update_run_status
    try:
        if p.R90_AS_MAIL:
            with open(lp_data_output_df_in, 'rb') as f:
                lp_data_output_df = standardize_df(pd.read_pickle(f))
            r90_claims = lp_data_output_df[lp_data_output_df['MEASUREMENT']=='R90']
            mail_claims = lp_data_output_df[lp_data_output_df['MEASUREMENT']=='M30']
            assert r90_claims.shape[0]>0, "R90_AS_MAIL is True but no R90 claims found"
            assert mail_claims.shape[0]>0, "R90_AS_MAIL is True but no M30 claims found"
            combo = r90_claims.merge(mail_claims, how='outer', on=['GPI', 'NDC','BG_FLAG'], suffixes=('_R90', '_MAIL'))
            combo = combo[(combo['PRICE_MUTABLE_R90']==1) | (combo['PRICE_MUTABLE_MAIL']==1)]
            combo['diff'] = combo['FINAL_PRICE_R90']-combo['FINAL_PRICE_MAIL']
            assert(combo['FINAL_PRICE_R90']==combo['FINAL_PRICE_MAIL']).all(), "Some R90 prices are different from MAIL prices"
            print('R90 claims correctly assigned to MAIL MAC list.')
    except Exception as e:
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'QA VCML Assignment', repr(e), error_loc)
        raise e

def qa_test_performance(params_in: str):
    '''
    It test the performance files form the model. It helps the DS understand if the model is working and if there are things
    that one should pay closer attention to.
    
    input: p.MODEL_PERFORMANCE_OUTPUT, p.PRE_EXISTING_PERFORMANCE_OUTPUT
    '''
    # TODO: This file does not currently excist on the GCP branch.  It has to be created on CPMO.py before this code can be tested.
    from BQ import performance_files, awp_spend_total, chain_subgroup_cmm
    import pandas as pd
    import numpy as np
    from types import ModuleType
    import util_funcs as uf
    from util_funcs import write_params, read_BQ_data, get_formatted_string
    import os
    if isinstance(params_in, ModuleType):
        p = params_in
    else:
        write_params(params_in)
        import CPMO_parameters as p
    from CPMO_shared_functions import update_run_status, check_run_status
    try:
        def check_perf(comparison: str = 'model-pre'):
            """
            comparison: 'model-pre' will check the model performance against the pre-existing performance,
                    'lambda-pre' will check the lambda performance against the pre-existing performance,
                    'model-lambda' will check the model performance against the lambda performance in terms of the AWP
            """
            if p.WRITE_TO_BQ:
                pre_existing_performance = read_BQ_data(
                    performance_files,
                    project_id = p.BQ_OUTPUT_PROJECT_ID,
                    dataset_id = p.BQ_OUTPUT_DATASET,
                    table_id = "Prexisting_Performance",
                    run_id = p.AT_RUN_ID,
                    client = ', '.join(sorted(p.CUSTOMER_ID)),
                    period = p.TIMESTAMP,
                    output = True)
            else:
                pre_existing_performance = pd.read_csv(p.FILE_OUTPUT_PATH + p.PRE_EXISTING_PERFORMANCE_OUTPUT)

            if comparison in ['model-pre', 'model-lambda']:
                if p.WRITE_TO_BQ:
                    model_performance = read_BQ_data(
                        performance_files,
                        project_id=p.BQ_OUTPUT_PROJECT_ID,
                        dataset_id=p.BQ_OUTPUT_DATASET,
                        table_id="Model_Performance",
                        run_id=p.AT_RUN_ID,
                        client=', '.join(sorted(p.CUSTOMER_ID)),
                        period=p.TIMESTAMP,
                        output=True)
                else:
                    model_performance = pd.read_csv(p.FILE_OUTPUT_PATH + p.MODEL_PERFORMANCE_OUTPUT)
            if comparison in ['lambda-pre', 'model-lambda']:
                if p.WRITE_TO_BQ:
                    lambda_performance = read_BQ_data(
                        performance_files,
                        project_id = p.BQ_OUTPUT_PROJECT_ID,
                        dataset_id = p.BQ_OUTPUT_DATASET,
                        table_id = "Lambda_Performance",
                        run_id = p.AT_RUN_ID,
                        client = ', '.join(sorted(p.CUSTOMER_ID)),
                        period = p.TIMESTAMP,
                        output = True)
                else:
                    lambda_performance = pd.read_csv(p.FILE_OUTPUT_PATH + p.LAMBDA_PERFORMANCE_OUTPUT)

            if comparison == 'model-pre':
                full_performance = pd.merge(model_performance, pre_existing_performance, on='ENTITY')
            elif comparison == 'lambda-pre':
                full_performance = pd.merge(lambda_performance, pre_existing_performance, on='ENTITY')
            else:
                full_performance = pd.merge(model_performance, lambda_performance, on='ENTITY')
            full_performance.rename(columns = {'PERFORMANCE_x': 'MODEL_PERF', 'PERFORMANCE_y': 'PRE_EX_PERF'}, inplace=True)

            chain_subgroup = read_BQ_data(
                        chain_subgroup_cmm,
                        project_id = p.BQ_INPUT_PROJECT_ID,
                        dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
                        table_id = "combined_measurement_mapping" + p.WS_SUFFIX + p.CCP_SUFFIX,
                        client = ', '.join(sorted(p.CUSTOMER_ID)),
                       
            )

            # mask to filter out all but client side performance 
            mask = (~full_performance['ENTITY'].isin(chain_subgroup['CHAIN_SUBGROUP_DISTINCT']))
            
            # The client side 
            df = full_performance[mask].copy()
            
            df_mail = df[df['ENTITY'].str.contains("MAIL")].copy()
            df_retail = df[~df['ENTITY'].str.contains("MAIL")].copy()
            
            df_mail.loc[df_mail['MODEL_PERF']>0,'MODEL_PERF'] = p.CLIENT_MAIL_OVRPERF_PEN * df_mail.loc[df_mail['MODEL_PERF']>0,'MODEL_PERF']
            df_mail.loc[df_mail['PRE_EX_PERF']>0,'PRE_EX_PERF'] = p.CLIENT_MAIL_OVRPERF_PEN * df_mail.loc[df_mail['PRE_EX_PERF']>0,'PRE_EX_PERF']
            df_mail.loc[df_mail['MODEL_PERF']<0,'MODEL_PERF'] = p.CLIENT_MAIL_UNRPERF_PEN * df_mail.loc[df_mail['MODEL_PERF']<0,'MODEL_PERF']
            df_mail.loc[df_mail['PRE_EX_PERF']<0,'PRE_EX_PERF'] = p.CLIENT_MAIL_UNRPERF_PEN * df_mail.loc[df_mail['PRE_EX_PERF']<0,'PRE_EX_PERF']
            df_mail.loc[:,'MODEL_PERF'] = np.abs(df_mail.loc[:,'MODEL_PERF'])
            df_mail.loc[:,'PRE_EX_PERF'] = np.abs(df_mail.loc[:,'PRE_EX_PERF'])
        
            df_retail.loc[df_retail['MODEL_PERF']>0,'MODEL_PERF'] = p.CLIENT_RETAIL_OVRPERF_PEN * df_retail.loc[df_retail['MODEL_PERF']>0,'MODEL_PERF']
            df_retail.loc[df_retail['PRE_EX_PERF']>0,'PRE_EX_PERF'] = p.CLIENT_RETAIL_OVRPERF_PEN * df_retail.loc[df_retail['PRE_EX_PERF']>0,'PRE_EX_PERF']
            df_retail.loc[df_retail['MODEL_PERF']<0,'MODEL_PERF'] = p.CLIENT_RETAIL_UNRPERF_PEN * df_retail.loc[df_retail['MODEL_PERF']<0,'MODEL_PERF']
            df_retail.loc[df_retail['PRE_EX_PERF']<0,'PRE_EX_PERF'] = p.CLIENT_RETAIL_UNRPERF_PEN * df_retail.loc[df_retail['PRE_EX_PERF']<0,'PRE_EX_PERF']
            df_retail.loc[:,'MODEL_PERF'] = np.abs(df_retail.loc[:,'MODEL_PERF'])
            df_retail.loc[:,'PRE_EX_PERF'] = np.abs(df_retail.loc[:,'PRE_EX_PERF'])

            df_client = pd.concat([df_retail,df_mail])
            if comparison in ['model-pre', 'lambda-pre']:
                if np.sum(np.abs(df_client['MODEL_PERF'])) > np.sum(np.abs(df_client['PRE_EX_PERF'])) + 5.0:
                    print('Each client run should improve the overall client situation (at mail and retail) and therefore once should check this by hand.')
                    print('This could happen when there is a off balance between the client breakouts i.e Mail >> Retail')
                    print('Open the Model performance file and determine how big is lost in performance on one of the breakouts relative to the wins on the rest.')
                    print('The overall client performance change from ${:,.2f} to ${:,.2f}'.format(np.sum(np.abs(df_client['PRE_EX_PERF'])), np.sum(np.abs(df_client['MODEL_PERF']))))
                    if comparison in ['model-pre']:
                        df_client.to_csv(os.path.join(p.FILE_OUTPUT_PATH, "Client_Performance_" + p.DATA_ID + ".csv"), index=False)
                    error_report.append('*ERROR: The client performance did not improved on each of the client breakouts. ({})'.format(comparison))
                else:
                    print('The client liability was reduced in all breakouts')


            # # Big pharmacies
            # df = full_performance[full_performance['ENTITY'].isin(p.BIG_CAPPED_PHARMACY_LIST)]
            # df = df.sum()
            # if df['MODEL_PERF'] > df['PRE_EX_PERF']:
            #     print('The big cap pharmacy liability change from ${:,.2f} to ${:,.2f}'.format(df['PRE_EX_PERF'],df['MODEL_PERF']))
            #     assert False, '*ERROR: The big 5 liability did not inmproved.'
            # else:
            #     print('The big cap pharmacy liability was reduced')

            # Objective reduction

            # df1 is the contribution from the pharmacies not the BIG_CAPPED to the objective function
            df = full_performance[full_performance['ENTITY'].isin(set(p.AGREEMENT_PHARMACY_LIST['GNRC']+p.AGREEMENT_PHARMACY_LIST['BRND'])-set(p.BIG_CAPPED_PHARMACY_LIST['GNRC']+p.BIG_CAPPED_PHARMACY_LIST['BRND']))].copy()
            df.loc[:,'OBJECTIVE_MODEL'] = df.loc[:,'MODEL_PERF']
            df.loc[:,'OBJECTIVE_PRE_EX'] = df.loc[:,'PRE_EX_PERF']
            df.loc[df['OBJECTIVE_MODEL']<0,'OBJECTIVE_MODEL'] = 0
            df.loc[df['OBJECTIVE_PRE_EX']<0,'OBJECTIVE_PRE_EX'] = 0

            # df1 is the contribution from the client to the objective function
            df1 = full_performance[mask].copy()
            df_mail = df1[df1['ENTITY'].str.contains("MAIL")].copy()
            df_retail = df1[~df1['ENTITY'].str.contains("MAIL")].copy()
            
            df_mail.loc[:,'OBJECTIVE_MODEL'] = df_mail.loc[:,'MODEL_PERF']
            df_mail.loc[:,'OBJECTIVE_PRE_EX'] = df_mail.loc[:,'PRE_EX_PERF']

            df_mail.loc[df_mail['OBJECTIVE_MODEL']>0,'OBJECTIVE_MODEL'] = p.CLIENT_MAIL_OVRPERF_PEN * df_mail.loc[df_mail['OBJECTIVE_MODEL']>0,'OBJECTIVE_MODEL']
            df_mail.loc[df_mail['OBJECTIVE_PRE_EX']>0,'OBJECTIVE_PRE_EX'] = p.CLIENT_MAIL_OVRPERF_PEN * df_mail.loc[df_mail['OBJECTIVE_PRE_EX']>0,'OBJECTIVE_PRE_EX']
            df_mail.loc[df_mail['OBJECTIVE_MODEL']<0,'OBJECTIVE_MODEL'] = p.CLIENT_MAIL_UNRPERF_PEN * df_mail.loc[df_mail['OBJECTIVE_MODEL']<0,'OBJECTIVE_MODEL']
            df_mail.loc[df_mail['OBJECTIVE_PRE_EX']<0,'OBJECTIVE_PRE_EX'] = p.CLIENT_MAIL_UNRPERF_PEN * df_mail.loc[df_mail['OBJECTIVE_PRE_EX']<0,'OBJECTIVE_PRE_EX']
            df_mail.loc[:,'OBJECTIVE_MODEL'] = np.abs(df_mail.loc[:,'MODEL_PERF'])
            df_mail.loc[:,'OBJECTIVE_PRE_EX'] = np.abs(df_mail.loc[:,'PRE_EX_PERF'])

            df_retail.loc[:,'OBJECTIVE_MODEL'] = df_retail.loc[:,'MODEL_PERF']
            df_retail.loc[:,'OBJECTIVE_PRE_EX'] = df_retail.loc[:,'PRE_EX_PERF']
            df_retail.loc[df_retail['OBJECTIVE_MODEL']>0,'OBJECTIVE_MODEL'] = p.CLIENT_RETAIL_OVRPERF_PEN * df_retail.loc[df_retail['OBJECTIVE_MODEL']>0,'OBJECTIVE_MODEL']
            df_retail.loc[df_retail['OBJECTIVE_PRE_EX']>0,'OBJECTIVE_PRE_EX'] = p.CLIENT_RETAIL_OVRPERF_PEN * df_retail.loc[df_retail['OBJECTIVE_PRE_EX']>0,'OBJECTIVE_PRE_EX']
            df_retail.loc[df_retail['OBJECTIVE_MODEL']<0,'OBJECTIVE_MODEL'] = p.CLIENT_RETAIL_UNRPERF_PEN * df_retail.loc[df_retail['OBJECTIVE_MODEL']<0,'OBJECTIVE_MODEL']
            df_retail.loc[df_retail['OBJECTIVE_PRE_EX']<0,'OBJECTIVE_PRE_EX'] = p.CLIENT_RETAIL_UNRPERF_PEN * df_retail.loc[df_retail['OBJECTIVE_PRE_EX']<0,'OBJECTIVE_PRE_EX']
            df_retail.loc[:,'OBJECTIVE_MODEL'] = np.abs(df_retail.loc[:,'OBJECTIVE_MODEL'])
            df_retail.loc[:,'OBJECTIVE_PRE_EX'] = np.abs(df_retail.loc[:,'OBJECTIVE_PRE_EX'])

            # df2 is the contribution from the BIG_CAPPED to the objective function
            df2 = full_performance[full_performance['ENTITY'].isin(p.OVER_REIMB_CHAINS)].copy()
            df2.loc[:,'OBJECTIVE_MODEL_1'] = df2.loc[:,'MODEL_PERF']
            df2.loc[:,'OBJECTIVE_PRE_EX_1'] = df2.loc[:,'PRE_EX_PERF']
            df2.loc[df2['OBJECTIVE_MODEL_1']<0,'OBJECTIVE_MODEL_1'] = 0
            df2.loc[df2['OBJECTIVE_PRE_EX_1']<0,'OBJECTIVE_PRE_EX_1'] = 0

            df2.loc[:,'OBJECTIVE_MODEL_2'] = df2.loc[:,'MODEL_PERF']
            df2.loc[:,'OBJECTIVE_PRE_EX_2'] = df2.loc[:,'PRE_EX_PERF']
            df2.loc[df2['OBJECTIVE_MODEL_2']>0,'OBJECTIVE_MODEL_2'] = 0
            df2.loc[df2['OBJECTIVE_PRE_EX_2']>0,'OBJECTIVE_PRE_EX_2'] = 0

            df2['OBJECTIVE_MODEL'] = df2['OBJECTIVE_MODEL_1'] + 0.1*df2['OBJECTIVE_MODEL_2']
            df2['OBJECTIVE_PRE_EX'] = df2['OBJECTIVE_PRE_EX_1'] + 0.1*df2['OBJECTIVE_PRE_EX_2']

            df_obj = pd.concat([df_mail,df_retail,df2,df])
            if comparison in ['model-pre', 'lambda-pre']:
                # Checking if model performance is atleast 5 percent better than the pre-existing performance
                if df_obj['OBJECTIVE_MODEL'].sum() > 1.05*(df_obj['OBJECTIVE_PRE_EX'].sum()):
                    if comparison in ['model-pre']:
                        df_obj.iloc[:,:5].to_csv(os.path.join(p.FILE_OUTPUT_PATH, "Objective_Function_" + p.DATA_ID + ".csv"), index=False)
                    error_report.append('*ERROR: The objective function increased after running the model from ${:,.2f} to ${:,.2f} ({})'.format(df_obj['OBJECTIVE_PRE_EX'].sum(), df_obj['OBJECTIVE_MODEL'].sum(), comparison))
                else:
                    print('The objective function was reduced from ${:,.2f} to ${:,.2f}'.format(df_obj['OBJECTIVE_PRE_EX'].sum(), df_obj['OBJECTIVE_MODEL'].sum()))
            elif comparison == 'model-lambda':
                if p.WRITE_TO_BQ:
                    awp_spend = read_BQ_data(
                        awp_spend_total,
                        project_id=p.BQ_OUTPUT_PROJECT_ID,
                        dataset_id=p.BQ_OUTPUT_DATASET,
                        table_id="awp_spend_total_medd_comm_subgroup",
                        run_id=p.AT_RUN_ID,
                        client=', '.join(sorted(p.CUSTOMER_ID)),
                        period=p.TIMESTAMP,
                        output=True)
                else:
                    awp_spend = pd.read_csv(os.path.join(p.FILE_OUTPUT_PATH, "awp_spend_total_" + p.DATA_ID + ".csv"))

                awp_spend['AWP'] = awp_spend[['FULLAWP_ADJ', 'FULLAWP_ADJ_PROJ_LAG', 'FULLAWP_ADJ_PROJ_EOY']].sum(axis=1)

                # (lambda - model) / awp
                if (df_obj['OBJECTIVE_PRE_EX'].sum() - df_obj['OBJECTIVE_MODEL'].sum()) / awp_spend['AWP'].sum() > 0.001:
                    error_report.append('*ERROR: The (lambda performance - model performance) / AWP is greater than 0.1%')
        
        error_report = []
        check_perf()
        check_perf(comparison='lambda-pre')
        check_perf(comparison='model-lambda')

        if error_report and not p.IGNORE_PERFORMANCE_CHECK:
            raise ValueError('  '.join(error_report))
        if p.IGNORE_PERFORMANCE_CHECK: 
            ## Keeping the run_status unaltered  
            if p.FULL_YEAR:
                update_run_status('Started', i_run_type = "-BypassPerformance_WS")
            else:
                update_run_status('Started', i_run_type = "-BypassPerformance")

    except Exception as e:
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        
        if check_run_status(run_status = 'Started') and not p.FULL_YEAR : 
            update_run_status('Complete-BypassPerformance', 'Client/Pharmacy Performance Checks', repr(e), error_loc, "-BypassPerformance")

        if check_run_status(run_status = 'Started') and p.FULL_YEAR: 
            update_run_status('Complete-BypassPerformance', 'Client/Pharmacy Performance Checks', repr(e), error_loc, "-BypassPerformance_WS")

        
        raise e
        
def qa_test_xvcml_meas_parity(params_in: str):
    '''
    It tests below constraints for X VCMLS (XR, XT, XA)
     1.) Mail <= Retail (M30 < XT|XR|XA)
     2.) R90 <= R30 (XT < XR|XA)
     3.) Has both XR and XA vcmls for a given client
    
    '''
    
    import pandas as pd
    import numpy as np
    from types import ModuleType
    from util_funcs import write_params
    import BQ
    if isinstance(params_in, ModuleType):
        p = params_in
    else:
        write_params(params_in)
        import CPMO_parameters as p

    try:        
        df_total = pd.read_csv(p.FILE_OUTPUT_PATH + p.TOTAL_OUTPUT, dtype=p.VARIABLE_TYPE_DIC, usecols=['REGION', 'CLIENT', 'MAC_LIST', 'GPI_NDC', 'CHAIN_SUBGROUP', 'PRICE_MUTABLE', 
                                                                                                        'MEASUREMENT', 'BG_FLAG','Final_Price'])
        df_total.columns = map(str.lower, df_total.columns)
        xgpis = df_total[(df_total['chain_subgroup'].str.contains('_EXTRL')) & (df_total['price_mutable'] == 1)]['gpi_ndc']
        df_qa = df_total[df_total['gpi_ndc'].isin(xgpis)]

        if not p.MAIL_MAC_UNRESTRICTED:
            df = df_qa.copy()
            df['MAIL_RETAIL_MEASUREMENT'] = 'Retail'
            df.loc[df['measurement'] == 'M30', 'MAIL_RETAIL_MEASUREMENT'] = 'M30'
            df = df[(df['chain_subgroup'].str.contains('_EXTRL')) | (df['measurement'] == 'M30')].copy()

            df_mr = df.groupby(['client', 'region', 'gpi_ndc', 'bg_flag', 'MAIL_RETAIL_MEASUREMENT']) \
                        .agg(MAX_Final_Price=('final_price', 'max'), \
                            MIN_Final_Price=('final_price', 'min') \
                            ).reset_index()

            m30_df = df_mr[df_mr['MAIL_RETAIL_MEASUREMENT'] == 'M30']
            xretail_df = df_mr[(df_mr['MAIL_RETAIL_MEASUREMENT'] != 'M30')]
            if len(m30_df) > 0 and len(xretail_df) > 0:
                m30_retail_df = m30_df.merge(xretail_df, how='inner',
                                            on=['client', 'region', 'gpi_ndc','bg_flag'],
                                            suffixes=('_M30', '_RETAIL'))
                m30_retail_df[m30_retail_df['MAX_Final_Price_M30'] > m30_retail_df['MIN_Final_Price_RETAIL']].to_csv(p.FILE_LOG_PATH + 'XVCML_MAIL_RETAIL_FAILED_{}.csv'.format(p.DATA_ID), index=False)

                assert len(m30_retail_df[m30_retail_df['MAX_Final_Price_M30'] > m30_retail_df['MIN_Final_Price_RETAIL']]) == 0, "Mail prices are higher than Retail for EXTRL VCMLS. Check XVCML_MAIL_RETAIL_FAILED.csv report in Logs folder"

        df = df_qa[(df_qa['chain_subgroup'].str.contains('_EXTRL'))].copy()

        assert df[df['measurement'] == 'R30']['mac_list'].nunique() <= 1, "Client has both XR and XA vcmls for R30! This should be a custom run"
        assert df[df['measurement'] == 'R90']['mac_list'].nunique() <= 1, "Client has a new vcml along with XT for R90! This should be a custom run"

        df_mr3090 = df.groupby(['client', 'region', 'gpi_ndc', 'chain_subgroup', 'measurement','bg_flag']) \
                                .agg(MAX_Final_Price=('final_price', 'max'), \
                                    MIN_Final_Price=('final_price', 'min') \
                                    ).reset_index()
        r90_df = df_mr3090[df_mr3090['measurement'] == 'R90']
        r30_df = df_mr3090[df_mr3090['measurement'] == 'R30']

        if len(r90_df) > 0 and len(r30_df) >0 :
            r90_30_df = r90_df.merge(r30_df, how='inner',
                                            on=['client', 'region', 'gpi_ndc', 'chain_subgroup','bg_flag']
                                            ,suffixes=('_90', '_30'))
            r90_30_df[r90_30_df['MAX_Final_Price_90'] > r90_30_df['MIN_Final_Price_30']].to_csv(p.FILE_LOG_PATH + 'XVCML_R30_R90_FAILED_{}.csv'.format(p.DATA_ID), index=False)
            assert len(r90_30_df[r90_30_df['MAX_Final_Price_90'] > r90_30_df['MIN_Final_Price_30']]) == 0, "R90 prices are higher than R30 for EXTRL VCMLS. Check XVCML_R30_R90_FAILED.csv report in Logs folder"
    except Exception as e:
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'XVCML Output File Check', repr(e), error_loc)
        raise e
    
    
def qa_test_price_changes_file(params_in: str):
    '''
    It test that all the price recomendations are at the GPI level, the format of the file, and that all GPI are unique.
    
    intput: p.PRICE_CHANGE_FILE
    
    '''
    # TODO: This file does not currently excist on the GCP branch.  It has to be created on CPMO.py before this code can be tested.
    import pandas as pd
    import numpy as np
    from types import ModuleType
    from util_funcs import write_params, write_to_bq, read_BQ_data
    import util_funcs as uf
    import BQ
    import re
    if isinstance(params_in, ModuleType):
        p = params_in
    else:
        write_params(params_in)
        import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df, update_run_status
    try:
        price_df = pd.read_csv(p.FILE_OUTPUT_PATH + p.PRICE_CHANGE_FILE, sep ='\t', dtype=p.VARIABLE_TYPE_DIC )
        qa_test_price_changes_file_err_list = []

        if p.TRUECOST_CLIENT or p.UCL_CLIENT:
            # Test column names and precence
            if set(price_df.columns) != set(['MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE', 
                                              'GNRC_MACPRC', 'BRND_MACPRC', 'GNRC_CurrentMAC','BRND_CurrentMAC']):
                qa_test_price_changes_file_err_list.append("*ERROR: The output file has has different columns than dessired. It should have: MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE', 'GNRC_MACPRC', 'BRND_MACPRC', 'GNRC_CurrentMAC','BRND_CurrentMAC'")

            else:
                print('The file has the right columns')

            #### Adhoc change for WS 2025, need to remove it later ####
            price_df_TRU = price_df[price_df['MACLIST'].str.contains(p.APPLY_VCML_PREFIX)]
            # Test NDC11 are all stars
            if all(price_df_TRU['NDC11'].astype(str).str.match(pat = '\*{11}')):
                print('All NDC are *s ')
            else:
                qa_test_price_changes_file_err_list.append("*ERROR: Some NDC are not *, implying that we are recomending NDC level prices.")
 
            # Test GPPC are all stars
            if all(price_df_TRU['GPPC'].astype(str).str.match(pat = '\*{8}')):
                print('All GPPC are *s ')
            else:
                qa_test_price_changes_file_err_list.append("*ERROR: Some GPPC are not *s or different lenght.")
 
            # Test all GPIs are unique
            df_unique = price_df_TRU.groupby(['MACLIST', 'GPI']).agg({'GPPC':'nunique'}).reset_index()['GPPC']
 
            if any(df_unique != 1):
                qa_test_price_changes_file_err_list.append("*ERROR: Not all GPIs are unique.")
            else:
                print('All GPIs are unique')
            
            # Test we don't pass any null prices
            price_df_temp = price_df.copy()
            price_df_temp['COALESCED_CurrentMAC'] = price_df_temp['GNRC_CurrentMAC'].combine_first(price_df_temp['BRND_CurrentMAC'])
            price_df_temp['COALESCED_MACPRC'] = price_df_temp['GNRC_MACPRC'].combine_first(price_df_temp['BRND_MACPRC'])
            nans = price_df_temp[['MACLIST', 'GPI', 'GPPC', 'NDC11', 'EFFDATE', 'TERMDATE', 'COALESCED_MACPRC', 'COALESCED_CurrentMAC']].isna().any(axis=1)
            if nans.any() == True:
                qa_test_price_changes_file_err_list.append('*ERROR: Missing data (NaN) in p.PRICE_CHANGE_FILE. It could be due to the wrong format (csv) or missing data.')
            
            # Test we don't modify Brand price in non-Truecost MAC-lists
            maclist_unique_brand = price_df.loc[price_df['BRND_MACPRC'].notnull(),'MACLIST'].unique()
            if any(re.search(r"MAC+", maclist) for maclist in maclist_unique_brand):
                qa_test_price_changes_file_err_list.append('*ERROR: Brand Price being set to non_truecost/UCL Mac-list.')
            
                
            if len(qa_test_price_changes_file_err_list) != 0: 
                print("Failing QA : ")
                print(qa_test_price_changes_file_err_list)
                assert False, qa_test_price_changes_file_err_list

            else:
                print('There are no NaN in the file')


            
        else:
            if set(price_df.columns) != set(['MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE', 'MACPRC', 'Current MAC']):
                #assert False, "*ERROR: The output file has has different columns than dessired. It should have: 'MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE', 'MACPRC' and  'Current MAC'"
                qa_test_price_changes_file_err_list.append("*ERROR: The output file has has different columns than dessired. It should have: 'MACLIST', 'GPI', 'GPPC', 'NDC11', 'NAME', 'EFFDATE', 'TERMDATE', 'MACPRC' and  'Current MAC'")

            else:
                print('The file has the right columns')
            # Test NDC11 are all stars
            if all(price_df['NDC11'].astype(str).str.match(pat = '\*{11}')):
                print('All NDC are *s ')
            else:
                qa_test_price_changes_file_err_list.append("*ERROR: Some NDC are not *, implying that we are recomending NDC level prices.")

            # Test GPPC are all stars
            if all(price_df['GPPC'].astype(str).str.match(pat = '\*{8}')):
                print('All GPPC are *s ')
            else:
                qa_test_price_changes_file_err_list.append("*ERROR: Some GPPC are not *s or different lenght.")

            df_unique = price_df.groupby(['MACLIST', 'GPI']).agg({'GPPC':'nunique'}).reset_index()['GPPC']

            if any(df_unique != 1):
                qa_test_price_changes_file_err_list.append("*ERROR: Not all GPIs are unique.")
            else:
                print('All GPIs are unique')

            nans = price_df[['MACLIST', 'GPI', 'GPPC', 'NDC11', 'EFFDATE', 'TERMDATE', 'MACPRC', 'Current MAC']].isna().any(axis=1)
            if nans.any() == True:
                qa_test_price_changes_file_err_list.append('*ERROR: Missing data (NaN) in p.PRICE_CHANGE_FILE. It could be due to the wrong format (csv) or missing data.')

            if len(qa_test_price_changes_file_err_list) != 0: 
                print("Failing QA : ")
                print(qa_test_price_changes_file_err_list)
                assert False, qa_test_price_changes_file_err_list

            else:
                print('There are no NaN in the file')

        # ============================
        # ============================
        # WTW and AON client report
        # Notice we are currently keeping the report on generic drug only for now,
        # as report for WTW/AON needs more discussion with client/enhancement team
        
        if (p.CLIENT_NAME_TABLEAU.startswith('WTW') and not p.REMOVE_WTW_RESTRICTION) or (p.CLIENT_NAME_TABLEAU.startswith('AON')):

            price_df = price_df.drop(['NAME'], axis=1)
            if False:
                df_total = read_BQ_data(
                    BQ.lp_total_output_df,
                    project_id = p.BQ_OUTPUT_PROJECT_ID,
                    dataset_id = p.BQ_OUTPUT_DATASET,
                    table_id = "Total_Output_subgroup",
                    run_id = p.AT_RUN_ID,
                    client = ', '.join(sorted(p.CUSTOMER_ID)),
                    period = p.TIMESTAMP,
                    output = True)
                df_total = df_total[['MAC_LIST', 'GPI', 'NDC', 'CLAIMS', 'QTY','BG_FLAG']]
                df_total = df_total.loc[(df_total.BG_FLAG == 'G')]['MAC_LIST', 'GPI', 'NDC', 'CLAIMS', 'QTY']
            else:
                df_total = pd.read_csv(p.FILE_OUTPUT_PATH + p.TOTAL_OUTPUT, dtype=p.VARIABLE_TYPE_DIC, usecols=['MAC_LIST', 'GPI', 'NDC', 'CLAIMS', 'QTY', 'BG_FLAG'])
                df_total = df_total.loc[(df_total.BG_FLAG == 'G')]['MAC_LIST', 'GPI', 'NDC', 'CLAIMS', 'QTY']

            df_total['MAC_LIST'] = p.APPLY_VCML_PREFIX + df_total['MAC_LIST'].astype(str)

            tot_claims = df_total['CLAIMS'].sum()
            changes_df = price_df.merge(df_total.groupby(['MAC_LIST', 'GPI', 'NDC']).sum().reset_index(),
                                        how = 'left',
                                        left_on = ['MACLIST', 'GPI', 'NDC11'],
                                        right_on =['MAC_LIST', 'GPI', 'NDC'])

            assert changes_df.isna().sum().sum() == 0, 'The merger should not have any NaNs'

            changes_df = changes_df[changes_df['CLAIMS'] > 0]
            changes_df['PRICE_CHANGE'] = changes_df['QTY'] / changes_df['CLAIMS']*(changes_df['MACPRC'] - changes_df['Current MAC'])
            abv_claims = changes_df[changes_df['PRICE_CHANGE'] > 25]['CLAIMS'].sum()

            detail_report = changes_df[changes_df['PRICE_CHANGE'] > 25][['MAC_LIST', 'GPI', 'NDC','CLAIMS', 'QTY', 'MACPRC', 'Current MAC', 'PRICE_CHANGE']]

            report = pd.DataFrame.from_dict({'Total_Claims': [tot_claims],
                                      'Claims_Above_25': [abv_claims],
                                      'Fraction_Claims': [abv_claims/tot_claims]})
            
            if p.BQ_OUTPUT_DATASET != "ds_development_lp":
                assert p.WRITE_TO_BQ == True, 'The WTW or AON report needs to be recorded on BQ'

            if p.WRITE_TO_BQ:
                write_to_bq(
                    report,
                    project_output = p.BQ_OUTPUT_PROJECT_ID,
                    dataset_output = 'pricing_management',
                    table_id = "WTW_report",
                    client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                    timestamp_param = p.TIMESTAMP,
                    run_id = p.AT_RUN_ID,
                    schema = None # TODO: create schema
                )

            if p.WRITE_TO_BQ and detail_report.shape[0] > 0:
                write_to_bq(
                    detail_report,
                    project_output = p.BQ_OUTPUT_PROJECT_ID,
                    dataset_output = 'pricing_management',
                    table_id = "WTW_report_GPI_lvl",
                    client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                    timestamp_param = p.TIMESTAMP,
                    run_id = p.AT_RUN_ID,
                    schema = None # TODO: create schema
                )
            if not p.REMOVE_WTW_RESTRICTION:
                assert abv_claims/tot_claims < 0.1, 'Not an onboarding issue.  The results of the run must be notifed to CNP'

            price_df.rename(columns={"NDC11": "NDC", "MACLIST": "MAC_LIST"}, inplace = True)
            
            
            if False:
                df_total = read_BQ_data(
                    BQ.lp_total_output_df,
                    project_id = p.BQ_OUTPUT_PROJECT_ID,
                    dataset_id = p.BQ_OUTPUT_DATASET,
                    table_id = "Total_Output_subgroup",
                    run_id = p.AT_RUN_ID,
                    client = ', '.join(sorted(p.CUSTOMER_ID)),
                    period = p.TIMESTAMP,
                    output = True)
                df_total = df_total[['MAC_LIST', 'GPI', 'NDC', 'BEG_PERIOD_PRICE', 'MAC1026_UNIT_PRICE','PRICE_OVRD_AMT','BG_FLAG']]
                df_total = df_total.loc[(df_total.BG_FLAG == 'G')]['MAC_LIST', 'GPI', 'NDC', 'BEG_PERIOD_PRICE', 'MAC1026_UNIT_PRICE','PRICE_OVRD_AMT']
            else:
                
                df_total = pd.read_csv(p.FILE_OUTPUT_PATH + p.TOTAL_OUTPUT, dtype=p.VARIABLE_TYPE_DIC, usecols=['MAC_LIST', 'GPI', 'NDC', 'BEG_PERIOD_PRICE', 'MAC1026_UNIT_PRICE','PRICE_OVRD_AMT'])
                df_total = df_total.loc[(df_total.BG_FLAG == 'G')]['MAC_LIST', 'GPI', 'NDC', 'BEG_PERIOD_PRICE', 'MAC1026_UNIT_PRICE','PRICE_OVRD_AMT']

            df_total['MAC_LIST'] = p.APPLY_VCML_PREFIX + df_total['MAC_LIST'].astype(str)

            price_comparison = price_df.merge(df_total, how = 'left', on= ['GPI', 'MAC_LIST','NDC'])    
            
            #Exclude price changes due to CNP MAC Overrides
            price_comparison = price_comparison[price_comparison.PRICE_OVRD_AMT.isna()]
            
            #Check quarterly increases are not over 25% for WTW clients and monthly increases are not over 25% for AON clients 
            #(BEG_PERIOD_PRICE defined in CPMO.py is BEG_Q_PRICE for WTW and BEG_M_PRICE for AON)
            price_comparison['PERIOD_CAP'] = 1.25*price_comparison['BEG_PERIOD_PRICE']
            price_comparison['PRICE_CAP'] = np.round(price_comparison[['PERIOD_CAP', 'MAC1026_UNIT_PRICE']].max(axis=1), 4)+0.0002

            wtw_violations = price_comparison[price_comparison['MACPRC'] > price_comparison['PRICE_CAP']]
            wtw_violations['FLAG'] = str('LP Violation')
            if p.WRITE_TO_BQ and wtw_violations.shape[0] > 0:
                write_to_bq(
                    wtw_violations,
                    project_output = p.BQ_OUTPUT_PROJECT_ID,
                    dataset_output = 'pricing_management',
                    table_id = "WTW_price_violations",
                    client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                    timestamp_param = p.TIMESTAMP,
                    run_id = p.AT_RUN_ID,
                    schema = None # TODO: create schema
                )
            if not p.REMOVE_WTW_RESTRICTION:
                assert (price_comparison['MACPRC'] <= price_comparison['PRICE_CAP']).all(), "*Some of the prices exceed the 25% threshold on WTW or AON clients. Check the WTW price violations report on BQ."


            if not (price_comparison['Current MAC'] <= price_comparison['PRICE_CAP']).all():
                current_mac_above_cap = price_comparison[price_comparison['Current MAC'] >= price_comparison['PRICE_CAP']]
                current_mac_above_cap['FLAG'] = str('Current Mac')
                if p.WRITE_TO_BQ:
                    write_to_bq(
                        current_mac_above_cap,
                        project_output = p.BQ_OUTPUT_PROJECT_ID,
                        dataset_output = 'pricing_management',
                        table_id = "WTW_price_violations",
                        client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                        timestamp_param = p.TIMESTAMP,
                        run_id = p.AT_RUN_ID,
                        schema = None # TODO: create schema
                    )
                print("*Warning: The initial MAC price is above the 25% increase threshold for {} GPIs for WTW or AON clients. Check the WTW price violations report on BQ for records with flag=current mac.".format(current_mac_above_cap.shape[0]))
                #assert (price_comparison['Current MAC'] <= price_comparison['PRICE_CAP']).all(),  "*Some of the prior maintenace prices are above the 25% threshold on WTW clients."
            print(p.GPI_UP_FAC)
        print ("All price increases within the applicable threshold.")
                
    except Exception as e:
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Output File Checks', repr(e), error_loc)
        raise e


def qa_goodrx_price_bound(params_in: str, tolerance:float = 0.005):
    import pandas as pd
    import numpy as np
    import BQ
    from ast import literal_eval
    from types import ModuleType
    from util_funcs import write_params, write_to_bq, read_BQ_data

    if isinstance(params_in, ModuleType):
        p = params_in
    else:
        write_params(params_in)
        import CPMO_parameters as p

    from CPMO_shared_functions import update_run_status
    try:
        if not p.GOODRX_OPT or not p.RMS_OPT or not p.APPLY_GENERAL_MULTIPLIER:
            return

        if p.UNC_OPT:
            df_cols = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'BG_FLAG', 'GPI', 'CHAIN_GROUP', 'GO_LIVE', 'MAC_LIST',
                       'CURRENT_MAC_PRICE', 'GPI_ONLY', 'CLAIMS', 'QTY', 'MAC1026_UNIT_PRICE','RAISED_PRICE_UC',
                       'GOODRX_UPPER_LIMIT', 'ZBD_UPPER_LIMIT', 'lb', 'ub', 'lb_name', 'ub_name', 'lower_bound_ordered', 'upper_bound_ordered', 'Final_Price']
        else:
            df_cols = ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'BG_FLAG', 'GPI', 'CHAIN_GROUP', 'GO_LIVE', 'MAC_LIST',
                       'CURRENT_MAC_PRICE', 'GPI_ONLY', 'CLAIMS', 'QTY', 'MAC1026_UNIT_PRICE',
                       'GOODRX_UPPER_LIMIT', 'ZBD_UPPER_LIMIT', 'lb', 'ub', 'lb_name', 'ub_name', 'lower_bound_ordered', 'upper_bound_ordered', 'Final_Price']
        grx_df = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.GOODRX_FILE, dtype=p.VARIABLE_TYPE_DIC)
        if False:
            to_df = read_BQ_data(
                BQ.lp_total_output_df,
                project_id = p.BQ_OUTPUT_PROJECT_ID,
                dataset_id = p.BQ_OUTPUT_DATASET,
                table_id = "Total_Output_subgroup",
                run_id = p.AT_RUN_ID,
                client = ', '.join(sorted(p.CUSTOMER_ID)),
                period = p.TIMESTAMP,
                output = True)
        else:
            to_df = pd.read_csv(p.FILE_OUTPUT_PATH + p.TOTAL_OUTPUT, dtype=p.VARIABLE_TYPE_DIC, low_memory=False)

        qa_df = pd.merge(grx_df, to_df, on=['GPI', 'CHAIN_GROUP','BG_FLAG'], suffixes=('', '_output'))
        qa_df = qa_df[qa_df['QTY'] != 0]
        qa_df = qa_df[df_cols]

        # find goodrx price bound from upper_bound_ordered list
        qa_df['total_output_grx'] = qa_df['upper_bound_ordered'].loc[qa_df['upper_bound_ordered'].str.contains(
            'goodrx_upper_limit')].str.split(", 'goodrx_upper_limit'").str[0].str.split('(').str[-1].astype('float64')

        # give reasons for price bound not matching goodrx limit price
        qa_df['REASON'] = None
        qa_df['REASON'].loc[qa_df['upper_bound_ordered'].str.contains('Immutable')] = 'Price immutable'
        if p.UNC_OPT:
            qa_df['REASON'].loc[(qa_df['RAISED_PRICE_UC'] == True) & (qa_df['lb'] == qa_df['ub'])] = 'U&C Raised Price'
        if p.CLIENT_NAME_TABLEAU.find('WTW') >= 0:
            # When the price is raised due to MAC1026 being greater than WTW allowed price limit changes,
            # this comes through with a dominating lower bound of "wtw_upper_limit" and a dominating
            # upper bound of "wtw_upper_limit+0.0001". So we need to check A) that this is the reason
            # given in the ordered bounds, and B) that this limit is actually equal to MAC1026. If that's
            # the case, then the absence of a GoodRx upper price bound is not actually an LP failure,
            # but a legacy of how we code these limits to obey MAC1026.
            equal_1026_unit = (qa_df['lb']-qa_df['MAC1026_UNIT_PRICE']).abs() < tolerance
            # split divides the string into the first item only and returns an array, lambda function keeps only the first element
            qa_df['REASON'].loc[qa_df['upper_bound_ordered'].str.split(')').apply(lambda x: x[0]).str.contains('wtw_upper_limit')
                                & equal_1026_unit
                               ] = 'MAC1026 forced raises above WTW increase limits'
        goodrx_error = '*ERROR: GoodRx price bounds have issues. '

        num_missing = pd.isna(qa_df['total_output_grx']) & pd.isna(qa_df['REASON'])
        if sum(num_missing) > 0:
            goodrx_error += '{} prices should have GoodRx upper bounds but do not. '.format(sum(num_missing))
            qa_df['REASON'].loc[num_missing] = 'Missing GoodRx upper bound'

        # upper bound used in total output should not be greater than the goodrx limit
        qa_df['REASON'].loc[np.isclose(qa_df['MAC1026_UNIT_PRICE'], qa_df['total_output_grx'])] = 'GoodRx price set to mac floor'

        num_mismatch = (qa_df['GOODRX_UPPER_LIMIT'] + 0.00001 < qa_df['total_output_grx']) & pd.isna(qa_df['REASON'])
        if sum(num_mismatch) > 0:
            goodrx_error += '{} prices have GoodRx upper bounds that do not match the GoodRx Limits price. '.format(sum(num_mismatch))
            qa_df.loc[qa_df['num_mismatch'], 'REASON'] = 'unknown'

        qa_df[pd.notna(qa_df['REASON'])].to_csv(p.FILE_LOG_PATH + 'GOODRX_PRICE_REPORT_{}.csv'.format(p.DATA_ID), index=False)
        if sum(num_missing) > 0 or sum(num_mismatch) > 0:
            assert False, goodrx_error
    except Exception as e:
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'GoodRx Price Bound Checks', repr(e), error_loc)
        raise e

def qa_price_overall_reasonableness(
    params_in: str,
    lp_with_final_prices_in: InputPath('pickle')
 ):
    import os
    import pandas as pd
    import numpy as np
    from types import ModuleType
    from util_funcs import write_params, write_to_bq, upload_blob

    import bisect
    if isinstance(params_in, ModuleType):
        p = params_in
    else:
        write_params(params_in)
        import CPMO_parameters as p
 
    from CPMO_shared_functions import update_run_status, standardize_df
 
    try:
        with open(lp_with_final_prices_in, 'rb') as f:
            lp_with_final_prices = pd.read_pickle(f)

        #Some GPIs still have prices lower than MAC1026 due to data issues. For this QA, discarding these bad data points.    
        if (lp_with_final_prices['PRICING_AVG_AWP'] < lp_with_final_prices['MAC1026_UNIT_PRICE']).count() > 0:
            awp_lower_than_mac1026 = (lp_with_final_prices['PRICING_AVG_AWP'] < lp_with_final_prices['MAC1026_UNIT_PRICE']).count()
            print('')
            print('*Warning: there are {} AWPs lower than MAC1026 unit prices'.format(awp_lower_than_mac1026))
            awp_lower_than_mac1026_to_file = lp_with_final_prices[lp_with_final_prices['PRICING_AVG_AWP'] < lp_with_final_prices['MAC1026_UNIT_PRICE']]
            awp_lower_than_mac1026_to_file.to_csv(os.path.join(p.FILE_OUTPUT_PATH, 'AWP_lower_than_MAC1026_' + p.DATA_ID + '.csv'), index=False)
            lp_with_final_prices_no_low_awp = lp_with_final_prices.drop(lp_with_final_prices[lp_with_final_prices['PRICING_AVG_AWP'] < lp_with_final_prices['MAC1026_UNIT_PRICE']].index)
        else:
            lp_with_final_prices_no_low_awp = lp_with_final_prices

        if p.FULL_YEAR: 
            #If total numbers of GPI that exceeded 5x AWP is less than 10, impact is insignificant, pass. If more than 10, fail the run.     
            total_output_price_over_5awp = lp_with_final_prices_no_low_awp[lp_with_final_prices_no_low_awp['Final_Price'] > 5*lp_with_final_prices_no_low_awp['PRICING_AVG_AWP']]
        else:
             #If total numbers of GPI that exceeded 5x AWP is less than 10, impact is insignificant, pass. If more than 10, fail the run.      
            total_output_price_over_5awp = lp_with_final_prices_no_low_awp[lp_with_final_prices_no_low_awp['Final_Price'] > 5*lp_with_final_prices_no_low_awp['PRICING_AVG_AWP']]


        #Excluding specialty prices and MAC price overrides
        exclusions = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.SPECIALTY_EXCLUSION_FILE, dtype=p.VARIABLE_TYPE_DIC))
        
        #Every client, regardless of LOB, must have specific exclusions and should never be left empty
        if len(exclusions) == 0: 
            print('\n ***** ALERT: Exclusions is empty')
            assert False 
        
        mac_price_override = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_PRICE_OVERRIDE_FILE, dtype=p.VARIABLE_TYPE_DIC))
        mac_price_override_5awp = mac_price_override.copy()
        mac_price_override_5awp['MACLIST']= mac_price_override_5awp['VCML_ID']
                                                                      
        mask_exclusions = total_output_price_over_5awp[['GPI','BG_FLAG']].apply(tuple, axis=1).isin(exclusions[['GPI','BG_FLAG']].apply(tuple, axis=1))
        total_output_price_over_5awp = total_output_price_over_5awp[~mask_exclusions]
        
        mask_mac_price_override_5awp = total_output_price_over_5awp.set_index(['MACLIST', 'GPI', 'NDC11','BG_FLAG']).index.isin(mac_price_override_5awp.set_index(['MACLIST', 'GPI', 'NDC','BG_FLAG']).index)
        total_output_price_over_5awp = total_output_price_over_5awp[~mask_mac_price_override_5awp]

        if len(total_output_price_over_5awp)<=10:
            if len(total_output_price_over_5awp)>0:
                total_output_price_over_5awp.to_csv(os.path.join(p.FILE_OUTPUT_PATH, 'Final_Price_Over_5awp_' + p.DATA_ID + '.csv'), index=False)

                if p.WRITE_TO_BQ or p.UPLOAD_TO_DASH:
                    write_to_bq(
                        total_output_price_over_5awp,
                        project_output = p.BQ_OUTPUT_PROJECT_ID,
                        dataset_output = p.BQ_OUTPUT_DATASET,
                        # Saving total_output_price_over_5awp to qa_price_over_2awp to avoid redundant table creation
                        table_id = "qa_price_over_2awp",
                        client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                        timestamp_param = p.TIMESTAMP,
                        run_id=p.AT_RUN_ID,
                        schema = None
                    )

                print('')
                if p.FULL_YEAR: 
                    print('*Warning: there are {} prices exceeding 5 times of AWP. Check file Final_Price_Over_5awp for details.'.format(len(total_output_price_over_5awp)))
                else:
                    print('*Warning: there are {} prices exceeding 5 times of AWP. Check file Final_Price_Over_5awp for details.'.format(len(total_output_price_over_5awp)))

        else:
            total_output_price_over_5awp.to_csv(os.path.join(p.FILE_OUTPUT_PATH, 'Final_Price_Over_5awp_' + p.DATA_ID + '.csv'), index=False)

            if p.WRITE_TO_BQ or p.UPLOAD_TO_DASH:
                write_to_bq(
                    total_output_price_over_5awp,
                    project_output = p.BQ_OUTPUT_PROJECT_ID,
                    dataset_output = p.BQ_OUTPUT_DATASET,
                    # Saving total_output_price_over_5awp to qa_price_over_2awp to avoid redundant table creation
                    table_id = "qa_price_over_2awp",
                    client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
                    timestamp_param = p.TIMESTAMP,
                    run_id=p.AT_RUN_ID,
                    schema = None
                )
            if p.FULL_YEAR: 
                assert len(total_output_price_over_5awp)<=10, "More than 10 GPIs with final prices exceed 5 times of AWP for client {}. Check Final_Price_Over_5awp file in Output folder.".format(p.CUSTOMER_ID[0])    
            else:
                assert len(total_output_price_over_5awp)<=10, "More than 10 GPIs with final prices exceed 5 times of AWP for client {}. Check Final_Price_Over_5awp file in Output folder.".format(p.CUSTOMER_ID[0])    

    except Exception as e:
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Price changes are not within reasonable range', repr(e), error_loc)
        raise e

def qa_diagnostic_report(params_in: str):
    import pandas as pd
    import BQ
    from types import ModuleType
    from util_funcs import write_params, write_to_bq, read_BQ_data
    import util_funcs as uf
    if isinstance(params_in, ModuleType):
        p = params_in
    else:
        write_params(params_in)
        import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df, update_run_status, add_virtual_r90
    try:
        p.VARIABLE_TYPE_DIC['GPI_12'] = str
        # Read in total_output file
        if False:
            df = read_BQ_data(
                BQ.lp_total_output_df,
                project_id = p.BQ_OUTPUT_PROJECT_ID,
                dataset_id = p.BQ_OUTPUT_DATASET,
                table_id = "Total_Output_subgroup",
                run_id = p.AT_RUN_ID,
                client = ', '.join(sorted(p.CUSTOMER_ID)),
                period = p.TIMESTAMP,
                output = True)
        else:
            df = pd.read_csv(p.FILE_OUTPUT_PATH + p.TOTAL_OUTPUT, dtype=p.VARIABLE_TYPE_DIC)

        # CLIENT_GUARANTEE_FILE
        client_guarantees = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.CLIENT_GUARANTEE_FILE, dtype = p.VARIABLE_TYPE_DIC)
        client_guarantees = standardize_df(client_guarantees)
        client_guarantees = client_guarantees[['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'BG_FLAG', 'RATE', 'PHARMACY_TYPE']]\
            .rename(columns={'RATE': 'Client Guarantee Rate'})
        # PHARM_GUARANTEE_FILE
        pharm_guarantee = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.PHARM_GUARANTEE_FILE
                                                     , dtype=p.VARIABLE_TYPE_DIC)) \
            .rename(columns={'PHARMACY': 'CHAIN_GROUP', 'RATE': 'Pharmacy Guarantee Rate'})
        # Merge with Total output use the same way as the measurement report generation
        df = pd.merge(df, client_guarantees, on=['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'BG_FLAG', 'PHARMACY_TYPE'], how='left')
        df = pd.merge(df, pharm_guarantee, on=['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP'], how='left')

        # Generate price changes and check if at bounds
        # df['Price_Raised'] = df['OLD_MAC_PRICE'] < df['Final_Price']
        df['At_Lower_Bound'] = df['Final_Price'] == df['lb']
        df['At_Upper_Bound'] = df['Final_Price'] == df['ub']

        # De-dup the R30/M30/R90 prices, prepare for join and show prices in parallel
        # Min Retail prices !=M30 within group ['CLIENT', 'REGION', 'GPI', 'BG_FLAG'] from QA.py price_compare(x)
        r_prices = df.loc[df['MEASUREMENT'] != 'M30',] \
            .sort_values(by='Final_Price', ascending=True) \
            .drop_duplicates(subset=['CLIENT', 'REGION', 'GPI','BG_FLAG'], keep='first')
        # Max M30 price within group ['CLIENT', 'REGION', 'GPI'] from QA.py price_compare(x)
        m30_prices = df.loc[df['MEASUREMENT'] == 'M30',] \
            .sort_values(by='Final_Price', ascending=False) \
            .drop_duplicates(subset=['CLIENT', 'REGION', 'GPI', 'BG_FLAG'], keep='first')
        # Max R90 price within group ['CLIENT', 'REGION', 'CHAIN_GROUP','GPI'] from R9030_price_compare(x)
        r90_prices = df.loc[df['MEASUREMENT'] == 'R90',] \
            .sort_values(by='Final_Price', ascending=False) \
            .drop_duplicates(subset=['CLIENT', 'REGION', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GPI', 'BG_FLAG'], keep='first')

        # # self join for parallel R30, M30, R90 prices
        join_cols = ['CLIENT', 'REGION', 'GPI', 'BG_FLAG']
        r90_join_cols = ['CLIENT', 'REGION', 'GPI', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'BG_FLAG']
        sliced_cols = ['CLIENT', 'REGION', 'GPI', 'BG_FLAG', 'Final_Price']
        r90_sliced_cols = ['CLIENT', 'REGION', 'GPI', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'BG_FLAG', 'Final_Price']
        df = pd.merge(df, r_prices[sliced_cols].rename(columns={'Final_Price': 'R_Price'}), on=join_cols, how='left')
        df = pd.merge(df, m30_prices[sliced_cols].rename(columns={'Final_Price': 'M30_Price'}), on=join_cols, how='left')
        df = pd.merge(df, r90_prices[r90_sliced_cols].rename(columns={'Final_Price': 'R90_Price'}), on=r90_join_cols,
                      how='left')

        df['PRICE_CHANGE'] = 'no change'
        df.loc[df['OLD_MAC_PRICE'] < df['Final_Price'], 'PRICE_CHANGE'] = 'raised'
        df.loc[df['OLD_MAC_PRICE'] > df['Final_Price'], 'PRICE_CHANGE'] = 'lowered'

        df['CONSTRAINT'] = 'none'
        df.loc[(df['MEASUREMENT'] == 'M30') & (df['Final_Price'] == df['R_Price']), 'CONSTRAINT'] = 'M30 = min(R30,R90)'

        df.loc[(df['MEASUREMENT'] == 'R30') & (df['Final_Price'] == df['R90_Price']), 'CONSTRAINT'] = 'R30 = R90'
        df.loc[(df['MEASUREMENT'] == 'R30') & (df['Final_Price'] == df['M30_Price']), 'CONSTRAINT'] = 'R30 = M30'
        df.loc[(df['MEASUREMENT'] == 'R90') & (df['Final_Price'] == df['M30_Price']), 'CONSTRAINT'] = 'R90 = M30'
        df['ZERO_QTY'] = df['PRICING_QTY_PROJ_EOY'] == 0

        # calculate performance (line by line)
        # this is not quite correct because it does not account for generic launches or brand-surplus (in the case for brand-generic offset)
        # good enough for diagnostics

        # do nothing
        df['CLIENT_SURPLUS_YTD'] = ((1 - df['Client Guarantee Rate']) * df['FULLAWP_ADJ'] - df['PRICE_REIMB'])
        df['CLIENT_SURPLUS_LAG'] = (((1 - df['Client Guarantee Rate']) * df['FULLAWP_ADJ_PROJ_LAG']) - df['LAG_REIMB'])
        df['Old_Price_Effective_Reimb_Proj_EOY'] = df.QTY_PROJ_EOY * df.EFF_CAPPED_PRICE.round(4)
        df['CLIENT_SURPLUS_EOY'] = ((1 - df['Client Guarantee Rate']) * df['FULLAWP_ADJ_PROJ_EOY']) - (
        df['Old_Price_Effective_Reimb_Proj_EOY'])
        df['CLIENT_SURPLUS_FULL'] = df['CLIENT_SURPLUS_YTD'] + df['CLIENT_SURPLUS_LAG'] + df['CLIENT_SURPLUS_EOY']

        df['PHARM_SURPLUS_YTD'] = ((1 - df['Pharmacy Guarantee Rate']) * df['PHARM_FULLAWP_ADJ'] - df['PHARM_PRICE_REIMB'])
        df['PHARM_SURPLUS_LAG'] = ((1 - df['Pharmacy Guarantee Rate']) * df['PHARM_FULLAWP_ADJ_PROJ_LAG'] - df['PHARM_LAG_REIMB'])
        df['PHARM_SURPLUS_EOY'] = ((1 - df['Pharmacy Guarantee Rate']) * df['PHARM_FULLAWP_ADJ_PROJ_EOY'] - df[
            'Pharm_Old_Price_Effective_Reimb_Proj_EOY'])
        df['PHARM_SURPLUS_FULL'] = df['PHARM_SURPLUS_YTD'] + df['PHARM_SURPLUS_LAG'] + df['PHARM_SURPLUS_EOY']

        # Model
        df['Price_Effective_Reimb_Proj_EOY'] = df.QTY_PROJ_EOY * df.EFF_CAPPED_PRICE_new.round(4)
        df['Pharm_Price_Effective_Reimb_Proj_EOY'] = df.PHARM_QTY_PROJ_EOY * df.EFF_CAPPED_PRICE_new.round(4)
        df['CLIENT_SURPLUS_MODEL_EOY'] = ((1 - df['Client Guarantee Rate']) * df['FULLAWP_ADJ_PROJ_EOY']) - \
            (df['Price_Effective_Reimb_Proj_EOY'])
        df['CLIENT_SURPLUS_MODEL_FULL'] = df['CLIENT_SURPLUS_YTD'] + df['CLIENT_SURPLUS_LAG'] + df[
            'CLIENT_SURPLUS_MODEL_EOY']
        df['PHARM_SURPLUS_MODEL_EOY'] = ((1 - df['Pharmacy Guarantee Rate']) * df['PHARM_FULLAWP_ADJ_PROJ_EOY'] - df[
            'Pharm_Price_Effective_Reimb_Proj_EOY'])
        df['PHARM_SURPLUS_MODEL_FULL'] = df['PHARM_SURPLUS_YTD'] + df['PHARM_SURPLUS_LAG'] + df['PHARM_SURPLUS_MODEL_EOY']

        df2 = df.groupby(['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'PHARMACY_TYPE', 'PRICE_MUTABLE', 'ZERO_QTY',
                          'PRICE_CHANGE', 'At_Lower_Bound', 'At_Upper_Bound', 'CONSTRAINT', 'lb', 'ub', 'lb_name', 'ub_name', 'lower_bound_ordered', 'upper_bound_ordered']) \
            .agg({'GPI': 'nunique', 'FULLAWP_ADJ': 'sum'
                     , 'CLIENT_SURPLUS_FULL': 'sum', 'CLIENT_SURPLUS_MODEL_FULL': 'sum'
                     , 'PHARM_SURPLUS_FULL': 'sum', 'PHARM_SURPLUS_MODEL_FULL': 'sum'}).reset_index()

        df2.to_csv(p.FILE_LOG_PATH + p.DIAGNOSTIC_REPORT, index=False)
    except Exception as e:
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Diagnostic Report QA', repr(e), error_loc)
        raise e


if __name__ == '__main__':
    import os
    import sys
    import datetime as dt
    import util_funcs as uf
    import importlib
    import BQ
    
    args = arguments()
    # write parameters
    pfile = f'CPMO_parameters.py'
    ppath = os.path.join(os.path.dirname(__file__), pfile)
    with open(ppath, 'w') as pf:
        pf.write(args.params)
    p = importlib.import_module(pfile.replace('.py', ''), package='PBM_Opt_Code')
#     spec = importlib.util.spec_from_file_location('p', ppath)
#     p = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(p)
#     os.remove(ppath)
    
    tolerance = 0.005
    # create paths for local run
    os.makedirs(os.path.join(p.PROGRAM_OUTPUT_PATH, 'Output'), exist_ok=True)
    output_df_path = os.path.join(p.PROGRAM_OUTPUT_PATH, 'Output', 'lp_data_output_df.pkl')
    final_prices_path = os.path.join(p.PROGRAM_OUTPUT_PATH, 'Output', 'lp_with_final_prices.pkl')
    output_cols_path = os.path.join(p.PROGRAM_OUTPUT_PATH, 'Output', 'output_cols.pkl')

    qa_Pharmacy_Output(p)
    qa_Price_Check_Output(p)
    
    qa_price_output(
        params_in=p,
        lp_data_output_df_out=output_df_path,
        lp_with_final_prices_out=final_prices_path,
        output_cols_out=output_cols_path
    )
    
    qa_price_tiering_rules_REPORT(
        params_in=p,
        lp_data_output_df_in=output_df_path,
        tolerance=tolerance
    )
    
    qa_Prices_above_MAC1026_floor(
        params_in=p,
        lp_data_output_df_in=output_df_path,
        lp_with_final_prices_in=final_prices_path,
        output_cols_in=output_cols_path,
        tolerance=tolerance
    )
    
    qa_pref_nonpref_pharm_pricing(
        params_in=p,
        lp_data_output_df_in=output_df_path,
        lp_with_final_prices_in=final_prices_path,
        output_cols_in=output_cols_path
    )
    
    qa_r90_as_mail(
        params_in=p,
        lp_data_output_df_in=output_df_path
    )

    qa_test_performance(p)
    qa_test_xvcml_meas_parity(p)
    qa_test_price_changes_file(p)
    qa_goodrx_price_bound(p, tolerance=tolerance)

    qa_price_overall_reasonableness(
        params_in=p,
        lp_with_final_prices_in=final_prices_path)

    qa_diagnostic_report(
        params_in=p)
