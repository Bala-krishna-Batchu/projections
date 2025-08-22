def add_cap_columns(df):
    import numpy as np
    ''' Add 3 new columns - BOUND         : Lower or Upper
                            BOUND_REASON  : Reason for LB/UB 
                            CAPPED_REASON : All other reasons other than LB/UB
    '''
        
    df['Final_Price_IS_UB'] = False
    df['Final_Price_IS_LB'] = False
    df.loc[ df['Final_Price'] == df['lb'].astype(float).round(4) , 'Final_Price_IS_LB'] = True
    df.loc[ df['Final_Price'] == df['ub'].astype(float).round(4) , 'Final_Price_IS_UB'] = True
    df['BOUND'] = 'NEITHER'
    df['BOUND_REASON'] = 'NEITHER'
    df['CAPPED_REASON'] = 'UNKNOWN'
    
    # Updating the Bound
    df.loc[ df['Final_Price_IS_LB'] , 'BOUND'] = 'Lower Bound'
    df.loc[ df['Final_Price_IS_UB'] , 'BOUND'] = 'Upper Bound'

    # Updating the Bound reason 
    df['BOUND_REASON'] = np.where(df['Final_Price_IS_LB'], df['lb_name'], df['BOUND_REASON'])
    df['BOUND_REASON'] = np.where(df['Final_Price_IS_UB'], df['ub_name'], df['BOUND_REASON'])
    
    return df
    
    
def check_bounds(df):
    import numpy as np
    ''' assigning the cap reason as Lower / Upper bounded '''
    
    df['CAPPED_REASON'] = np.where(df['Final_Price_IS_LB'] | df['Final_Price_IS_UB'], df['BOUND_REASON'], df['CAPPED_REASON'])
    
    return df

def check_immutable(df):
    import numpy as np
    ''' assigning cap reason with the 'immutable_reason' column if present else "Immutable" - for prices that can't be changed '''

    try:
        df['CAPPED_REASON'] = np.where((df['PRICE_MUTABLE'] == 0), df['IMMUTABLE_REASON'], df['CAPPED_REASON'])
    except:
        df.loc[(df['PRICE_MUTABLE'] == 0), 'CAPPED_REASON'] = 'Immutable'
        
    return df

def check_mail_retail(df, cap):
    import pandas as pd
    ''' assign the cap reason for mail where the mail price couldn't be increased as the Retail price is low '''
    columns = df.columns
    #Combining R30 and R90 as we need to consider both as one for grouping to get max and min prices
    df['MAIL_RETAIL_MEASUREMENT'] = 'Retail'
    df.loc[df['MEASUREMENT'] == 'M30', 'MAIL_RETAIL_MEASUREMENT'] = 'M30'
    
    df_mr = df.groupby(['CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG', 'MAIL_RETAIL_MEASUREMENT']) \
                .agg(MAX_Final_Price=('Final_Price', 'max'), \
                     MIN_Final_Price=('Final_Price', 'min') \
                     ).reset_index()
    
    m30_df = df_mr[df_mr['MAIL_RETAIL_MEASUREMENT'] == 'M30']
    retail_df = df_mr[df_mr['MAIL_RETAIL_MEASUREMENT'] != 'M30']

    m30_retail_df = m30_df.merge(retail_df, how='inner',
                                        on=['CLIENT', 'REGION', 'GPI_NDC','BG_FLAG'],
                                        suffixes=('_M30', '_RETAIL'))
    
    # Only the mail records will join here (MEASUREMENT = M30)
    mail_retail_df_merged = pd.merge(df, m30_retail_df, how='inner',
                                                       left_on=['CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG', 'MEASUREMENT'],
                                                       right_on=['CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG', 'MAIL_RETAIL_MEASUREMENT_M30'])
    
    ## list of GPIs to update the reason for M30 where the price couldn't be increased as the Retail price is low
    ## takes into account the new mail_retail cap

    mail_retail_df_merged['MIN_Final_Price_RETAIL_CAPPED'] = cap * mail_retail_df_merged['MIN_Final_Price_RETAIL']
    mail_retail_limited_gpis = mail_retail_df_merged.query(
                                'Final_Price_IS_UB == False & PRICE_MUTABLE == 1 & MAX_Final_Price_M30 == MIN_Final_Price_RETAIL_CAPPED'
                            )[['GPI_NDC', 'BG_FLAG']].drop_duplicates()
    
    # updating the cap reason
    df.loc[df.set_index(['GPI_NDC', 'BG_FLAG']).index.isin(mail_retail_limited_gpis.set_index(['GPI_NDC', 'BG_FLAG']).index)
        & (df['MEASUREMENT'] == 'M30')
        & (df['CAPPED_REASON'] == 'UNKNOWN'),
        'CAPPED_REASON'] = 'Mail <= Retail'
    
    return df[columns]

def check_r30_r90(df):
    import pandas as pd
    ''' assign the cap reason for R90 where the R90 price couldn't be increased as the R30 price is low '''
    columns = df.columns
    df_mr3090 = df.groupby(['CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG', 'CHAIN_SUBGROUP', 'MEASUREMENT']) \
                .agg(MAX_Final_Price=('Final_Price', 'max'), \
                     MIN_Final_Price=('Final_Price', 'min') \
                     ).reset_index()
    
    r90_df = df_mr3090[df_mr3090['MEASUREMENT'] == 'R90']
    r30_df = df_mr3090[df_mr3090['MEASUREMENT'] == 'R30']
    
    ## dataframe to show R30 and R90 price comparisons
    r90_30_df = r90_df.merge(r30_df, how='inner',
                                     on=['CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG', 'CHAIN_SUBGROUP']
                                     ,suffixes=('_90', '_30'))  
    
    ## merging with df as a LEFT JOIN to ensure the "Reason" can be updated in total output
    df_r90_merged = pd.merge(df, r90_30_df, how='left',
                                            left_on=['CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG', 'CHAIN_SUBGROUP'],
                                            right_on=['CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG', 'CHAIN_SUBGROUP'])
    
    ##Updating the reason for R90 where the price couldn't be increased as the R30 price is low
    df_r90_merged.loc[(df_r90_merged['CAPPED_REASON'] == 'UNKNOWN')
                    &(~df_r90_merged['MIN_Final_Price_30'].isna())
                    &(df_r90_merged['MAX_Final_Price_90']==df_r90_merged['MIN_Final_Price_30'])
                   , 'CAPPED_REASON'] = 'R90 <= R30 limit'
    
    return df_r90_merged[columns]


def check_cvs_others(df):
    import numpy as np
    
    ''' assigning the cap reason for CVS/ CVSSP where the price couldn't be increased as the other pharmacies price is low '''
    columns = df.columns
    cvs_price_limited = []
    other_price_limited = []
    grouped_df = df.groupby(['CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG', 'MEASUREMENT'])

    for (client, region, GPI_NDC, bg_flag, measurement), group in grouped_df:
        cvs_parity_subgroup = 'CVSSP' if ((group['CHAIN_SUBGROUP'] == 'CVSSP').any()) else 'CVS'

        df_pref = group[group['CHAIN_SUBGROUP'] == cvs_parity_subgroup]
        df_oth = group[group['CHAIN_SUBGROUP'] != cvs_parity_subgroup]

        if not df_pref.empty and not df_oth.empty:
                cvs_price = np.max(df_pref['Final_Price'])
                oth_price = np.min(df_oth['Final_Price'])

                if cvs_price == oth_price:
                    cvs_price_limited.append(GPI_NDC+bg_flag+measurement+cvs_parity_subgroup)   
                    for chain in df_oth[df_oth['Final_Price'] == oth_price]['CHAIN_SUBGROUP'].unique().tolist():
                        other_price_limited.append(GPI_NDC+bg_flag+measurement+chain)
                    
    df['GPI_NDC_BG_MEAS_CHAIN'] = [str(i) for i in df['GPI_NDC']+df['BG_FLAG']+df['MEASUREMENT']+df['CHAIN_SUBGROUP']]
    
    ##Updating the reason for CVS/ CVSSP where the price couldn't be increased as the other pharmacies price is low

    df.loc[(df['GPI_NDC_BG_MEAS_CHAIN'].isin(cvs_price_limited))
                      & (df['CAPPED_REASON'] == 'UNKNOWN')
                      , 'CAPPED_REASON'] = 'CVS <= Other Pharmacies'
    
    df.loc[(df['GPI_NDC_BG_MEAS_CHAIN'].isin(other_price_limited))
                      & (df['CAPPED_REASON'] == 'UNKNOWN')
                      , 'CAPPED_REASON'] = 'CVS <= Other Pharmacies'
    
    return df[columns]

def check_pref_nonpref(df):
    import pandas as pd
    ''' assigning the cap reason for Pref pharmacies where the price couldn't be increased as the non-pref pharmacies is low '''
    columns = df.columns
    df_grp = df.groupby(['CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG', 'MEASUREMENT', 'PHARMACY_TYPE']) \
                .agg(MAX_Final_Price=('Final_Price', 'max'), \
                     MIN_Final_Price=('Final_Price', 'min') \
                     ).reset_index()
    
    pref_df = df_grp[(df_grp['PHARMACY_TYPE'] == 'Preferred') & (df_grp['MEASUREMENT'] != 'M30')]
    nonpref_df = df_grp[(df_grp['PHARMACY_TYPE'] == 'Non_Preferred') & (df_grp['MEASUREMENT'] != 'M30')]
    
    pref_non_df = pref_df.merge(nonpref_df, how='inner',
                                 on=['CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG', 'MEASUREMENT']
                                 ,suffixes=('_pref', '_nonpref'))
    
    df = pd.merge(df, pref_non_df, how='left',
                                        left_on=['CLIENT', 'REGION', 'GPI_NDC',  'BG_FLAG', 'MEASUREMENT', 'PHARMACY_TYPE'],
                                        right_on=['CLIENT', 'REGION', 'GPI_NDC', 'BG_FLAG', 'MEASUREMENT', 'PHARMACY_TYPE_pref'])
    
    # Updating the cap reason for Pref pharmacies where the price couldn't be increased as the non-pref pharmacies is low
    df.loc[(df['CAPPED_REASON'] == 'UNKNOWN')
                &(~df['MAX_Final_Price_pref'].isna())
                &(df['MAX_Final_Price_pref']==df['MIN_Final_Price_nonpref'])
                &(df['MAX_Final_Price_pref']==df['Final_Price'])
               , 'CAPPED_REASON'] = 'Pref <= Non-Pref'

    return df[columns]


def check_consistent_mac(df):
    
    ''' assigning the cap reason as consistent mac for unknown cap reason GPIs that share same VCML
        as other GPIs with known cap reasons, within a measurement group 
    '''
    columns = df.columns
    unk_gpi = df[df['CAPPED_REASON'] == 'UNKNOWN'][['GPI_NDC', 'BG_FLAG']].drop_duplicates()

    update_conditions = []
    for idx, row in enumerate(unk_gpi.itertuples(index=False)):
        gpi, bg_flag = row.GPI_NDC, row.BG_FLAG
        df_gpi = df[(df['GPI_NDC'] == gpi) & (df['BG_FLAG'] == bg_flag)]
        df_unk = df_gpi[df_gpi['CAPPED_REASON'] == 'UNKNOWN']
        df_kno = df_gpi[df_gpi['CAPPED_REASON'] != 'UNKNOWN']

        data = df_unk[['MEASUREMENT', 'MAC_LIST', 'Final_Price', 'CAPPED_REASON']].\
                        merge(df_kno[['MEASUREMENT', 'MAC_LIST', 'Final_Price', 'CAPPED_REASON']], how = 'inner', 
                                          on = ['MEASUREMENT', 'MAC_LIST', 'Final_Price'],
                                         suffixes = ('_unk','_kno'))
        
        if not data.empty:
            condition = (
                   (df['GPI_NDC'] == gpi)
                   & (df['BG_FLAG'] == bg_flag)
                   & (df['CAPPED_REASON'] == 'UNKNOWN')
                   & df['MEASUREMENT'].isin(data['MEASUREMENT']) 
                   & df['MAC_LIST'].isin(data['MAC_LIST'])
                   & df['Final_Price'].isin(data['Final_Price'])
                       )
            update_conditions.append(condition)      
    
    if update_conditions:
        import numpy as np
        combined_condition = np.logical_or.reduce(update_conditions)
        df.loc[combined_condition, 'CAPPED_REASON' ] = 'Consistent MAC pricing'
            
    return df[columns]

def check_independent_cvs_medd(df, PREF_OTHER_FACTOR):
    import numpy as np
    ''' assigning cap reason for independent pharmacies (for MEDD client) where their prices couldn't go lower than PREF_OTHER_FACTOR of CVS prices '''
    columns = df.columns
    unk_gpis = df[df['CAPPED_REASON'] == 'UNKNOWN'][['GPI_NDC', 'BG_FLAG']].drop_duplicates()
    noncvs_pref_idx_list = []

    for idx, row in enumerate(unk_gpi.itertuples(index=False)):
        gpi, bg_flag = row.GPI_NDC, row.BG_FLAG
        df_gpi = df[(df['GPI_NDC'] == gpi) & (df['BG_FLAG'] == bg_flag) & (df['MEASUREMENT'] != 'M30')]
        grp_df = df_gpi.groupby('MEASUREMENT')
        for (measurement), group in grp_df:
            cvs_parity_subgroup = 'CVSSP' if ((group['CHAIN_SUBGROUP'] == 'CVSSP').any()) else 'CVS'
            df_cvs_price = np.min(group[group['CHAIN_SUBGROUP'] == cvs_parity_subgroup]['Final_Price'])

            noncvs_pref_idx_list.extend(group.index[
                (group['CHAIN_SUBGROUP'] != cvs_parity_subgroup) 
                & (group['MEASUREMENT'] == measurement)                          
                & (abs(group['Final_Price']-round(PREF_OTHER_FACTOR * df_cvs_price,4)) < 0.0001) 
                & (group['CAPPED_REASON'] == 'UNKNOWN')])
            
    df.loc[(df.index.isin(noncvs_pref_idx_list)), 'CAPPED_REASON'] = 'Independents >= PREF_OTHER_FACTOR * CVS'

    return df[columns]

def check_same_price(df):
    
    ''' assigning the unchanged final prices cap reason to be LP soft constraint as LP tries to minimize number of price point updates '''

    df.loc[(df['CURRENT_MAC_PRICE'].astype(float).round(4) == df['Final_Price'].astype(float).round(4)) & (df['CAPPED_REASON'] == 'UNKNOWN'), 'CAPPED_REASON'] = 'LP Soft Constraint - No price change'
    
    df.drop( ['Final_Price_IS_UB', 'Final_Price_IS_LB'], axis = 1, inplace = True) 
    return df
   

def check_pkg_constraint(df):
    
    ''' assigning the reason as equal package strength for gpis that got capped with final price equal to that of a gpi with different package strength '''
    import pandas as pd
    columns = df.columns
    unk_gpi = df.loc[df['CAPPED_REASON'] == 'UNKNOWN'][['GPI_NDC', 'BG_FLAG']].drop_duplicates()
    df['GPI_BASE'] = df['GPI_NDC'].str[:12]
    update_conditions = []
    
    for idx, row in enumerate(unk_gpi.itertuples(index=False)):
        gpi, bg_flag = row.GPI_NDC, row.BG_FLAG
        df_gpi = df[(df['GPI_NDC'] == gpi) & (df['BG_FLAG'] == bg_flag)][['MEASUREMENT', 'CHAIN_SUBGROUP', 'Final_Price']]
        df_all_pkg = df[(df['GPI_BASE'] == gpi[:12]) & (df['GPI_NDC'] != gpi) & (df['BG_FLAG'] == bg_flag)]
        df_pkg = pd.merge(df_gpi, df_all_pkg, how='inner', on=['MEASUREMENT', 'CHAIN_SUBGROUP', 'Final_Price'])

        if not df_pkg.empty:
            condition = (
                   (df['GPI_NDC'] == gpi)
                   & (df['BG_FLAG'] == bg_flag)
                   & (df['CAPPED_REASON'] == 'UNKNOWN')
                   & df['MEASUREMENT'].isin(df_pkg['MEASUREMENT']) 
                   & df['CHAIN_SUBGROUP'].isin(df_pkg['CHAIN_SUBGROUP'])
                   & df['Final_Price'].isin(df_pkg['Final_Price'])
                       )
            update_conditions.append(condition)
            
    if update_conditions:
        import numpy as np
        combined_condition = np.logical_or.reduce(update_conditions)
        df.loc[combined_condition, 'CAPPED_REASON' ] = 'Equal Package Strength'
                
    return df[columns]

def check_benchmark_ceiling(df, BENCHMARK_CAP):
    
    ''' assigning the reason as Benchmark Ceiling Price for gpis that got capped with their benchmark price times some multiplier '''
    
    benchmark_condition = ( 
                        (df['Final_Price'] == df['BENCHMARK_CEILING_PRICE'] * BENCHMARK_CAP)
                       & (df['Final_Price'] == df['GOODRX_UPPER_LIMIT'])
                       & (df['CAPPED_REASON'] == 'UNKNOWN')
                        )
    df.loc[benchmark_condition, 'CAPPED_REASON' ] = 'Benchmark Ceiling Price'
    return df

def apply_cap_reasons(lp_data_output_df, mail_unrestricted_cap_var, unc_flag, loglevel):
    """
    Analyze and classify price capping reasons for LP optimization results.
    
    Adds BOUND, BOUND_REASON, and CAPPED_REASON columns to identify why prices
    were constrained during optimization.
    """
    import CPMO_parameters as p
    import util_funcs as uf
    import os

    out_path = os.path.join(p.FILE_LOG_PATH, 'ClientPharmacyMacOptimization.log')
    logger = uf.log_setup(log_file_path=out_path, loglevel=loglevel)

    num_rows = len(lp_data_output_df)
    logger.info("Started adding the capped reason columns..")
    # Adding cap columns
    lp_data_output_df = add_cap_columns(lp_data_output_df)  

    # Lower/Upper bound check
    lp_data_output_df = check_bounds(lp_data_output_df)

    # Check for immutable final price
    lp_data_output_df = check_immutable(lp_data_output_df)

    # Mail <= Retail check
    lp_data_output_df = check_mail_retail(lp_data_output_df, mail_unrestricted_cap_var)

    # Same price LP soft constraint check
    lp_data_output_df = check_same_price(lp_data_output_df)

    # R90 <= R30 check
    meas = lp_data_output_df['MEASUREMENT'].unique().tolist()
    if ('R30' in meas) and ('R90' in meas):
        lp_data_output_df = check_r30_r90(lp_data_output_df)

    # CVS <= Other Pharmacies check
    lp_data_output_df = check_cvs_others(lp_data_output_df)

    # Pref <= Non-Pref check
    prefs = lp_data_output_df['PHARMACY_TYPE'].unique().tolist()
    if ('Preferred' in prefs) and ('Non_Preferred' in prefs):
        lp_data_output_df = check_pref_nonpref(lp_data_output_df)

    # Consistent MAC pricing check
    lp_data_output_df = check_consistent_mac(lp_data_output_df)

    # Independents >= PREF_OTHER_FACTOR * CVS for MEDD check
    if p.CLIENT_TYPE == 'MEDD':
        lp_data_output_df = check_independent_cvs_medd(lp_data_output_df, p.PREF_OTHER_FACTOR)

    # Equal package strength check
    lp_data_output_df = check_pkg_constraint(lp_data_output_df)

    # Benchmark Ceiling price check
    if p.APPLY_BENCHMARK_CAP:
        lp_data_output_df = check_benchmark_ceiling(lp_data_output_df, p.BENCHMARK_CAP_MULTIPLIER)


    assert len(lp_data_output_df) == num_rows, "Length of dataframe altered after assigning cap reasons !"

    cap_counts = lp_data_output_df.groupby(['CLIENT', 'REGION', 'BREAKOUT','MEASUREMENT', 'BG_FLAG', 'BOUND','CAPPED_REASON']).agg(UNIQUE_GPI_NDC = ('GPI_NDC' , lambda x: x.nunique()), TOTAL_RECORDS = ('GPI_NDC' , 'count'), AWP = ('FULLAWP_ADJ', 'sum')).sort_values(by = ['MEASUREMENT', 'BOUND', 'UNIQUE_GPI_NDC'], ascending = False).reset_index()

    if p.WRITE_TO_BQ and (unc_flag == True or p.UNC_ADJUST == False):
        uf.write_to_bq(
            cap_counts,
            project_output = p.BQ_OUTPUT_PROJECT_ID,
            dataset_output = p.BQ_OUTPUT_DATASET,
            table_id = "cap_reasons",
            client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
            timestamp_param = p.TIMESTAMP,
            run_id = p.AT_RUN_ID,
            schema = None # TODO: create schema
        )
    else:
        cap_counts.to_csv(os.path.join(p.FILE_OUTPUT_PATH, "cap_counts_" + p.DATA_ID + ".csv"), index=False)

    logger.info("Capped reasons added successfully..")
    
    return lp_data_output_df

def main():
    
    
    """
    
    This main function will read in the TOTAL_OUTPUT file from gs bucket and writes back 
    the same file with 3 new columns - CAPPED_REASON, BOUND_REASON, BOUND to  - 
    
    1.) Same location/folder as the TOTAL_OUTPUT file in gs, if `write_to_gs` flag is True
    2.) Specified local directory, if `write_to_gs` flag is False
    
    """              
                           
    write_to_gs = True

    try:
        output = pd.read_csv(os.path.join(p.FILE_OUTPUT_PATH, p.TOTAL_OUTPUT))
    except Exception as e:
        raise e
     
    num_rows = len(output)
    
    output = add_cap_columns(output)
    print("Adding cap columns completed...\n Shape of the data - \n",output.shape, '\n')
    
    output = check_bounds(output)
    print("Lower/Upper bound check completed...\n\n",output.CAPPED_REASON.value_counts(), '\n')
    
    output = check_immutable(output)
    print("Check for immutable final price completed...\n\n",output.CAPPED_REASON.value_counts(), '\n')
    
    if p.MAIL_MAC_UNRESTRICTED:
        output = check_mail_retail(output, p.MAIL_UNRESTRICTED_CAP)
    else:
        output = check_mail_retail(output,1)
    print("Mail <= Retail check completed...\n\n",output.CAPPED_REASON.value_counts(), '\n')
    
    output = check_same_price(output)
    print("Same price LP soft constraint check completed...\n\n",output.CAPPED_REASON.value_counts(), '\n') 
    
    meas = output['MEASUREMENT'].unique().tolist()
    if ('R30' in meas) and ('R90' in meas):
        output = check_r30_r90(output)
        print("R90 <= R30 check completed...\n\n",output.CAPPED_REASON.value_counts(), '\n')
    else:
        has_meas = 'R90' if 'R90' in meas else 'R30' if 'R30' in meas else 'M30'
        print("Skipping the R90 <= R30 check as the client has only {}".format(has_meas))
    
    output = check_cvs_others(output)
    print("CVS <= Other Pharmacies check completed...\n\n",output.CAPPED_REASON.value_counts(), '\n')
    
    prefs = output['PHARMACY_TYPE'].unique().tolist()
    if ('Preferred' in prefs) and ('Non_Preferred' in prefs):
        output = check_pref_nonpref(output)
        print("Pref <= Non-Pref check completed...\n\n",output.CAPPED_REASON.value_counts(), '\n')
    else:
        pre = 'Preferred' if 'Preferred' in prefs else 'Non_Preferred'
        print("Skipping the Pref <= Non-Pref check as the client has only {}".format(pre))
  
    output = check_consistent_mac(output)
    print("Consistent MAC pricing check completed...\n\n",output.CAPPED_REASON.value_counts(), '\n')
    
    if p.CLIENT_TYPE == 'MEDD':
        output = check_independent_cvs_medd(output, p.PREF_OTHER_FACTOR)
        print("Independents >= PREF_OTHER_FACTOR * CVS for MEDD check completed...\n\n",output.CAPPED_REASON.value_counts(), '\n')
        
    output = check_pkg_constraint(output)
    print("Equal Package Strength check completed...\n\n",output.CAPPED_REASON.value_counts(), '\n')
    
    assert len(output) == num_rows, "Length of dataframe altered!"
    
    print("Unknown capped reasons after all checks - ", round(output.CAPPED_REASON.value_counts(normalize=True)['UNKNOWN']*100,2), " % \n")
    

    if write_to_gs:
        
        output.to_csv(os.path.join(p.FILE_OUTPUT_PATH, p.TOTAL_OUTPUT[:-4]+"_cap_reasons.csv"))
        print("Finished writing the cap reasons file to cloud storage output path..")
    else:
        output.to_csv('cap_reasons.csv', index = False)
    
    
    
if __name__ == '__main__':
    
    import os
    import pandas as pd
    import numpy as np
    from google.cloud import bigquery
    import CPMO_parameters as p
    import warnings
    warnings.filterwarnings("ignore")
    bq = bigquery.Client()
    main()