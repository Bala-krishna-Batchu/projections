# -*- coding: utf-8 -*-
"""
"""

def unique_id_for_production():
    '''
    It verifies that the RUN_ID is unique when running in produciton. This part of the code should be move later and have it part of a more automated check as
    part of the Kubeflow pipeline such that we can not evern run in production without an RUN_ID and that we save to the Audit Trail the results of such run 
    (e.g. if anything in the code fail).
    '''
    
    from google.cloud import bigquery

    if p.BQ_OUTPUT_DATASET == 'ds_production_lp':
        
        if p.AT_RUN_ID == '':
            raise Exception('Audit Trail ID is empty.  Can not be empty for production runs.')
        
        query = '''
            SELECT * FROM pbm-mac-lp-prod-ai.pricing_management.AT_Run_ID
               WHERE RUN_ID = ('{AT_RUN_ID}')
        '''.format(AT_RUN_ID = str(p.AT_RUN_ID))
        bqclient = bigquery.Client()
        df = bqclient.query(query).result().to_dataframe()
        
        if len(df) > 1: # It could be 1 or 0.  If zero the AT_module has not run.  If 1 we might be seeing our run. If >1 more than one run has the same number.
            raise Exception('Audit Trail ID is repeated.  Can not be repeated for production runs.')

def prepare_pharmacy_guarantees():
    """ 
    Generate pharmacy‑level guarantee tables for the three supported lines of business:
    - COMMERCIAL  – joins the client’s guarantee table/sheet to measurement mapping.
    - MEDICAID    – largely mirrors COMMERCIAL but requires AWP-only reconciliation.
    - MEDD        – custom logic that blends claim‑level mapping with BigQuery‑calculated
                    target rates and additional data healing.

    The resulting file (`p.PHARM_GUARANTEE_FILE`) feeds into QA.py, Daily_Input_Read.py, ClientPharmacyMacOptimization.py and other subsequent files.
    
    """
    # ------------------------------------------------------------------
    # Standard library & project imports 
    # ------------------------------------------------------------------
    import BQ
    import numpy as np
    import pandas as pd
    import util_funcs as uf
    import CPMO_parameters as p
    from qa_checks import qa_dataframe
    from CPMO_shared_functions import standardize_df, add_virtual_r90, add_rur_guarantees
    
    # ------------------------------------------------------------------
    # Helper: map verbose chain names -> short codes 
    # ------------------------------------------------------------------
    def _normalize_chain_codes(df):
        mapping = {"AHOLD": "AHD", 
                   "ALBERTSONS": "ABS", 
                   "CARDINAL": "CAR", 
                   "ELEVATE": "ELE",
                   "EPIC": "EPC", 
                   "KD": "KIN", 
                   "KROGER": "KRG",
                   "WALMART": "WMT",
                   "MEIJER": "MJR",
                   "PUBLIX": "PBX",
                   "GENOA": "GEN",
                   "AMAZON": "AMZ",
                   "HYVEE": "HYV",
                   "INGLES": "IGD",
                   "THRIFTY WHITE": "THF",
                   "MCHOICE_KROGER": "MCHOICE_KRG",
                   "ALIGNRX" : "ARX",
                   "BIG Y" : "BGY",
                   "BRIOVA (OPTUM)" : "BRI",
                   "DISCOUNT DRUG MART" : "DDM",
                   "FAIRVIEW" : "FVW",
                   "GUARDIAN" : "GUA",
                   "MARC GLASSMAN" : "MGM",
                   "MERCY" : "MCY",
                   "PHARMERICA" : "PMA",
                   "RALEYS" : "RYS",
                   "RECEPT" : "RPT",
                   "SAVMOR" : "SMR",
                   "SUPERVALU" : "SVU",
                   "WAKEFERN" : "WFN",
                   "WEIS" : "WIS",
                   "COSTCO" : "CST",
                   "FOOD CITY" : "FCD",
                   "GOLUB" : "PCD",
                   "HEB" : "HEB",
                   "INGLES" : "IGD",
                   "LEWIS" : "LWD",
                   "SAM'S CLUB" : "SMC",
                   "TOPS MARKET" : "TPM",
                   "WEGMANS" : "WGS"}
        
        return df.replace(mapping)

    def _read_measurement_mapping():
        if p.READ_FROM_BQ:
            return uf.read_BQ_data(
                BQ.ger_opt_msrmnt_map.format(_customer_id=uf.get_formatted_string(p.CUSTOMER_ID)),
                                             project_id = p.BQ_INPUT_PROJECT_ID,
                                             dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
                                             table_id= "combined_measurement_mapping" + p.WS_SUFFIX + p.CCP_SUFFIX,
                                             customer=", ".join(sorted(p.CUSTOMER_ID)),
            )
        return pd.read_csv(p.FILE_INPUT_PATH + p.MEASUREMENT_MAPPING, dtype=p.VARIABLE_TYPE_DIC)
    
    # ------------------------------------------------------------------
    # COMMERCIAL 
    # ------------------------------------------------------------------
    def _build_commercial():
        """
        Builds the commercial pharmacy guarantees table by merging measurement mapping with commercial guarantees data.
        
        Steps:
        1. Read the commercial pharmacy guarantees data from a file or BigQuery.
        2. Normalize chain codes and standardize the data.
        3. Ensure all guarantees are AWP, NADAC or ACC based
        4. Read and process the measurement mapping data.
        5. Merge the measurement mapping with the commercial guarantees to produce the final guarantees table.
        6. Return the processed pharmacy guarantees table.
        """
        # -----------------------------
        # Read guarantee 
        # -----------------------------
        # Load the commercial pharmacy guarantees based on the configuration:
        # - If FULL_YEAR and GO_LIVE is January 1st, read from a specific file.
        # - If READ_FROM_BQ is enabled, read from BigQuery.
        # - Otherwise, read from the default commercial guarantees file.
        if p.FULL_YEAR and p.GO_LIVE.day == 1 and p.GO_LIVE.month == 1:
            commercial_pharmacy_guarantees = pd.read_csv(p.FILE_INPUT_PATH + p.PHARM_GUARANTEE_COMM_NY, dtype = p.VARIABLE_TYPE_DIC)
        elif p.READ_FROM_BQ:
            commercial_pharmacy_guarantees = uf.read_BQ_data(
                BQ.commercial_pharm_guarantees,
                project_id=p.BQ_INPUT_PROJECT_ID,
                dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                table_id="commercial_pharm_guarantees"
                )
        else:
            commercial_pharmacy_guarantees = pd.read_csv(p.FILE_INPUT_PATH + p.COMMERCIAL_PHARM_GUARANTEES_FILE, dtype = p.VARIABLE_TYPE_DIC)
        
        # Normalize chain codes and standardize the guarantees data
        commercial_pharmacy_guarantees = _normalize_chain_codes(standardize_df(commercial_pharmacy_guarantees))

        # Ensure all guarantees are AWP, NADAC or ACC based
        assert commercial_pharmacy_guarantees.PHRM_GRTE_TYPE.isin(["AWP", "NADAC", "ACC"]).any(), "Commercial Pharmacy Guarantees must contain at least one of AWP, NADAC, or ACC"
        
        # -----------------------------
        # Process measurement mapping
        # -----------------------------
        # Read and standardize the measurement mapping data
        measurement_mapping = standardize_df(_read_measurement_mapping())
        
        # Ensure there is only one guarantee category in the measurement mapping
        guarantee_category = measurement_mapping.GUARANTEE_CATEGORY.unique()
        assert len(guarantee_category) == 1, "Multiple guarantee categories"
        
        # Extract relevant columns for mapping and remove duplicates
        mapping = measurement_mapping[['CLIENT',
                                       'REGION',
                                       'BREAKOUT',
                                       'MEASUREMENT',
                                       'BG_FLAG',
                                       'CHAIN_GROUP',
                                       'CHAIN_SUBGROUP']].drop_duplicates().copy(deep=True)
        
        # Prepare measurement mapping for merging
        # - Remove mail breakout to avoid assigning all pharmacies to mail breakout
        measurement_mapping = measurement_mapping[['CLIENT','BREAKOUT', 'BG_FLAG']].drop_duplicates()
        measurement_mapping = measurement_mapping[~(measurement_mapping['BREAKOUT'].str.contains('M'))]
        
        # -----------------------------
        # Merge to produce guarantee table
        # -----------------------------
        # Merge measurement mapping with commercial guarantees to create the pharmacy guarantees table
        pharmacy_guarantees = measurement_mapping.merge(commercial_pharmacy_guarantees,on = 'BG_FLAG', how = 'outer')\
                                                 .reset_index().drop(columns=['index'])\
                                                 .rename(columns={'PHARMACY_GROUP':'PHARMACY',
                                                                 'PHARMACY_SUBGROUP':'PHARMACY_SUBGROUP_TEMP',
                                                                 'TARGET_RATE':'RATE'})
        
        # Update the breakout column for specific pharmacy subgroups (MCHOICE_CVS and MCHOICE_KRG)
        if len(mapping[(mapping['BREAKOUT'].str.contains('M'))])>0:
            for value in pharmacy_guarantees.loc[(pharmacy_guarantees['PHARMACY_SUBGROUP_TEMP'].isin(['MCHOICE_CVS','MCHOICE_KRG'])),'BG_FLAG'].unique().tolist():
                pharmacy_guarantees.loc[((pharmacy_guarantees['PHARMACY_SUBGROUP_TEMP'].isin(['MCHOICE_CVS','MCHOICE_KRG'])) & (pharmacy_guarantees['BG_FLAG'] == value)),'BREAKOUT'] = mapping[(mapping['BREAKOUT'].str.contains('M')) & (mapping['BG_FLAG'] == value)].BREAKOUT.values[0]
        else:
            pharmacy_guarantees = pharmacy_guarantees.loc[~(pharmacy_guarantees['PHARMACY_SUBGROUP_TEMP'].isin(['MCHOICE_CVS','MCHOICE_KRG']))]
        
        # -----------------------------
        # Merge with mapping and finalize
        # -----------------------------      
        
        # Create mappings for RATE and PHRM_GRTE_TYPE based on PHARMACY
        chain_group_rate_map = pharmacy_guarantees[['PHARMACY', 'BG_FLAG', 'RATE']].drop_duplicates(subset=['PHARMACY', 'BG_FLAG']).set_index('PHARMACY')[['BG_FLAG', 'RATE']]
        chain_group_type_map = pharmacy_guarantees[['PHARMACY', 'BG_FLAG', 'PHRM_GRTE_TYPE']].drop_duplicates(subset=['PHARMACY', 'BG_FLAG']).set_index('PHARMACY')[['BG_FLAG', 'PHRM_GRTE_TYPE']]
        
        # Merge the mapping with the pharmacy guarantees to align chain groups and subgroups
        pharmacy_guarantees = pd.merge(mapping, 
                                       pharmacy_guarantees, 
                                       how='left', 
                                       left_on=['CLIENT', 'BREAKOUT', 'BG_FLAG', 'CHAIN_GROUP', 'CHAIN_SUBGROUP'],
                                       right_on=['CLIENT', 'BREAKOUT', 'BG_FLAG', 'PHARMACY', 'PHARMACY_SUBGROUP_TEMP'])
        
        # Populate missing RATE and PHRM_GRTE_TYPE values using CHAIN_GROUP
        # Populate RATE and PHRM_GRTE_TYPE for rows where PHARMACY_SUBGROUP_TEMP is NaN but CHAIN_GROUP matches PHARMACY
        # We need this code for _R90OK, X VCMLs etc. ( we do not need this code for mchoice_cvs, mchoice_krg, or RUR )
        pharmacy_guarantees['RATE'] = np.where(pharmacy_guarantees['BG_FLAG'] == 'G', 
                                                pharmacy_guarantees['RATE'].fillna(pharmacy_guarantees['CHAIN_GROUP'].map(chain_group_rate_map[chain_group_rate_map['BG_FLAG'] == 'G']['RATE'])),  
                                                pharmacy_guarantees['RATE'].fillna(pharmacy_guarantees['CHAIN_GROUP'].map(chain_group_rate_map[chain_group_rate_map['BG_FLAG'] == 'B']['RATE'])))
        pharmacy_guarantees['PHRM_GRTE_TYPE'] = np.where(pharmacy_guarantees['BG_FLAG'] == 'G', 
                                                pharmacy_guarantees['PHRM_GRTE_TYPE'].fillna(pharmacy_guarantees['CHAIN_GROUP'].map(chain_group_type_map[chain_group_type_map['BG_FLAG'] == 'G']['PHRM_GRTE_TYPE'])),  
                                                pharmacy_guarantees['PHRM_GRTE_TYPE'].fillna(pharmacy_guarantees['CHAIN_GROUP'].map(chain_group_type_map[chain_group_type_map['BG_FLAG'] == 'B']['PHRM_GRTE_TYPE'])))
                                                        
        # Reorder columns, drop duplicates, and rename columns for the final guarantees table
        pharmacy_guarantees = pharmacy_guarantees[['CLIENT',
                                                   'REGION',
                                                   'BREAKOUT',
                                                   'MEASUREMENT',
                                                   'BG_FLAG',
                                                   'CHAIN_GROUP',
                                                   'CHAIN_SUBGROUP',
                                                   'RATE',
                                                   'PHRM_GRTE_TYPE']]\
                                                    .dropna()\
                                                    .drop_duplicates()\
                                                    .rename(columns ={'CHAIN_GROUP':'PHARMACY',
                                                                      'CHAIN_SUBGROUP':'PHARMACY_SUB'})
        
        return pharmacy_guarantees
    # ------------------------------------------------------------------
    # MEDICAID 
    # ------------------------------------------------------------------
    def _build_medicaid(): 
        """
        Builds the Medicaid pharmacy guarantees table by merging measurement mapping with Medicaid guarantees data.
        
        Steps:
        1. Read the Medicaid pharmacy guarantees data from a file or BigQuery.
        2. Normalize chain codes and standardize the data.
        3. Read and process the measurement mapping data.
        4. Remove M30 claims if MCHOICE is not present in the claims pipeline. Note : Medicaid does not have MCHOICE for now.
        5. Merge the measurement mapping with the Medicaid guarantees to produce the final guarantees table.
        6. Return the processed pharmacy guarantees table.
        """
        # -----------------------------
        # Read Medicaid guarantees
        # -----------------------------
        # Load the Medicaid pharmacy guarantees based on the configuration:
        # - If FULL_YEAR and GO_LIVE is January 1st, read from a specific file.
        # - If READ_FROM_BQ is enabled, read from BigQuery.
        # - Otherwise, read from the default Medicaid guarantees file.
        if p.FULL_YEAR and p.GO_LIVE.day == 1 and p.GO_LIVE.month == 1:
            medicaid_pharmacy_guarantees = pd.read_csv(p.FILE_INPUT_PATH + p.PHARM_GUARANTEE_MEDICAID_NY, dtype=p.VARIABLE_TYPE_DIC)
        elif p.READ_FROM_BQ:
            medicaid_pharmacy_guarantees = uf.read_BQ_data(
                BQ.medicaid_pharm_guarantees,
                project_id=p.BQ_INPUT_PROJECT_ID,
                dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                table_id="medicaid_pharm_guarantees"
                )
        else:
            medicaid_pharmacy_guarantees = pd.read_csv(p.FILE_INPUT_PATH + p.MEDICAID_PHARM_GUARANTEES_FILE, dtype = p.VARIABLE_TYPE_DIC)
        
        # Normalize chain codes and standardize the guarantees data
        medicaid_pharmacy_guarantees = _normalize_chain_codes(standardize_df(medicaid_pharmacy_guarantees))
        
        # Ensure all guarantees are AWP-based
        awp_medicaid_pharmacy_guarantees = medicaid_pharmacy_guarantees[~((medicaid_pharmacy_guarantees['BG_FLAG']=='B') & (medicaid_pharmacy_guarantees['PHARMACY_GROUP']=='RAD'))]
        assert awp_medicaid_pharmacy_guarantees.PHRM_GRTE_TYPE.eq("AWP").all(), "Medicaid Pharmacy Guarantees must be AWP based"
    
        # -----------------------------
        # Process measurement mapping
        # -----------------------------
        # Read and standardize the measurement mapping data
        measurement_mapping = standardize_df(_read_measurement_mapping())
        
        # Remove M30 claims if MCHOICE is not present in the claims pipeline Note : Medicaid does not have MCHOICE for now
        if not p.HAS_MCHOICE:
            measurement_mapping = measurement_mapping[measurement_mapping.MEASUREMENT != 'M30']
        
        # Extract relevant columns for mapping and remove duplicates
        mapping = measurement_mapping[['CLIENT',
                                       'REGION',
                                       'BREAKOUT',
                                       'MEASUREMENT',
                                       'BG_FLAG',
                                       'CHAIN_GROUP',
                                       'CHAIN_SUBGROUP']].drop_duplicates().copy(deep=True)
        
        # Prepare measurement mapping for merging
        # - Remove mail breakout to avoid assigning all pharmacies to mail breakout
        measurement_mapping = measurement_mapping[['CLIENT', 'BREAKOUT', 'BG_FLAG']].drop_duplicates()
        measurement_mapping = measurement_mapping[~(measurement_mapping['BREAKOUT'].str.contains('M'))]

        # -----------------------------
        # Merge to produce guarantee table
        # -----------------------------
        # Merge measurement mapping with Medicaid guarantees to create the pharmacy guarantees table
        pharmacy_guarantees = measurement_mapping.merge(medicaid_pharmacy_guarantees, on='BG_FLAG', how='outer')\
                                                 .reset_index().drop(columns=['index'])\
                                                 .rename(columns={'PHARMACY_GROUP': 'PHARMACY',
                                                                  'PHARMACY_SUBGROUP': 'PHARMACY_SUBGROUP_TEMP',
                                                                  'TARGET_RATE': 'RATE'})

        # Fix mail breakout for specific pharmacy subgroups (MCHOICE_CVS and MCHOICE_KRG)
        if len(mapping[(mapping['BREAKOUT'].str.contains('M'))])>0:
            for value in pharmacy_guarantees.loc[(pharmacy_guarantees['PHARMACY_SUBGROUP_TEMP'].isin(['MCHOICE_CVS','MCHOICE_KRG'])),'BG_FLAG'].unique().tolist():
                pharmacy_guarantees.loc[((pharmacy_guarantees['PHARMACY_SUBGROUP_TEMP'].isin(['MCHOICE_CVS','MCHOICE_KRG'])) & (pharmacy_guarantees['BG_FLAG'] == value)),'BREAKOUT'] = mapping[(mapping['BREAKOUT'].str.contains('M')) & (mapping['BG_FLAG'] == value)].BREAKOUT.values[0]
        else:
            pharmacy_guarantees = pharmacy_guarantees.loc[~(pharmacy_guarantees['PHARMACY_SUBGROUP_TEMP'].isin(['MCHOICE_CVS','MCHOICE_KRG']))]
        
        # -----------------------------
        # Merge with mapping and finalize
        # -----------------------------                                                                
                                                                         
        # Create a mapping of CHAIN_GROUP to RATE before the merge
        chain_group_rate_map = pharmacy_guarantees[['PHARMACY', 'BG_FLAG', 'RATE']].drop_duplicates(subset=['PHARMACY', 'BG_FLAG']).set_index('PHARMACY')[['BG_FLAG', 'RATE']]
        chain_group_type_map = pharmacy_guarantees[['PHARMACY', 'BG_FLAG', 'PHRM_GRTE_TYPE']].drop_duplicates(subset=['PHARMACY', 'BG_FLAG']).set_index('PHARMACY')[['BG_FLAG', 'PHRM_GRTE_TYPE']]

        # Merge the mapping with the pharmacy guarantees to align chain groups and subgroups
        pharmacy_guarantees = pd.merge(mapping, 
                                       pharmacy_guarantees, 
                                       how='left', 
                                       left_on=['CLIENT', 'BREAKOUT', 'BG_FLAG', 'CHAIN_GROUP', 'CHAIN_SUBGROUP'],
                                       right_on=['CLIENT', 'BREAKOUT', 'BG_FLAG', 'PHARMACY', 'PHARMACY_SUBGROUP_TEMP'])

        # Populate missing RATE and PHRM_GRTE_TYPE values using CHAIN_GROUP
        # Populate RATE and PHRM_GRTE_TYPE for rows where PHARMACY_SUBGROUP_TEMP is NaN but CHAIN_GROUP matches PHARMACY
        # We need this code for _R90OK, X VCMLs etc. ( we do not need this code for mchoice_cvs, mchoice_krg, or RUR )
        pharmacy_guarantees['RATE'] = np.where(pharmacy_guarantees['BG_FLAG'] == 'G', 
                                                pharmacy_guarantees['RATE'].fillna(pharmacy_guarantees['CHAIN_GROUP'].map(chain_group_rate_map[chain_group_rate_map['BG_FLAG'] == 'G']['RATE'])),  
                                                pharmacy_guarantees['RATE'].fillna(pharmacy_guarantees['CHAIN_GROUP'].map(chain_group_rate_map[chain_group_rate_map['BG_FLAG'] == 'B']['RATE'])))
        pharmacy_guarantees['PHRM_GRTE_TYPE'] = np.where(pharmacy_guarantees['BG_FLAG'] == 'G', 
                                                pharmacy_guarantees['PHRM_GRTE_TYPE'].fillna(pharmacy_guarantees['CHAIN_GROUP'].map(chain_group_type_map[chain_group_type_map['BG_FLAG'] == 'G']['PHRM_GRTE_TYPE'])),  
                                                pharmacy_guarantees['PHRM_GRTE_TYPE'].fillna(pharmacy_guarantees['CHAIN_GROUP'].map(chain_group_type_map[chain_group_type_map['BG_FLAG'] == 'B']['PHRM_GRTE_TYPE'])))
                                                                                  
        # Reorder columns, drop duplicates, and rename columns for the final guarantees table
        pharmacy_guarantees = pharmacy_guarantees[['CLIENT',
                                                   'REGION',
                                                   'BREAKOUT',
                                                   'MEASUREMENT',
                                                   'BG_FLAG',
                                                   'CHAIN_GROUP',
                                                   'CHAIN_SUBGROUP',
                                                   'RATE', 
                                                   'PHRM_GRTE_TYPE']]\
                                                    .dropna()\
                                                    .drop_duplicates()\
                                                    .rename(columns={'CHAIN_GROUP': 'PHARMACY',
                                                                    'CHAIN_SUBGROUP': 'PHARMACY_SUB'})
        
        # Return the final pharmacy guarantees table
        return pharmacy_guarantees
    
    # ------------------------------------------------------------------
    # MEDD 
    # ------------------------------------------------------------------
    def _build_medd(): 
        """
        Builds the MEDD pharmacy guarantees table by merging measurement mapping with MEDD-specific guarantees data.

        Steps:
        1. Read the MEDD pharmacy guarantees data from BigQuery or files.
        2. Normalize and standardize the data.
        3. Merge pharmacy measurement mapping with target rates.
        4. Assign chain subgroups and handle MCHOICE-specific rates.
        5. Handle missing rates and ensure data consistency.
        6. Return the processed pharmacy guarantees table.
        """
        
        # -----------------------------
        # Read MEDD guarantees and measurement mapping
        # -----------------------------
        if p.READ_FROM_BQ:
            if not p.FULL_YEAR:
                # Read contract effective date for target rate query
                eff_date = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.CONTRACT_DATE_FILE, dtype=p.VARIABLE_TYPE_DIC))

                if p.GENERIC_OPT:
                    # Map Pharmacy Chain Group to individual claims
                    pharmacy_measurement_mapping = uf.read_BQ_data(
                        project_id=p.BQ_INPUT_PROJECT_ID,
                        dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                        table_id='pharmacy_measurement_mapping_medd_custom',
                        query=BQ.pharmacy_measurement_mapping_medd_custom.format(
                            _customer_id=uf.get_formatted_string(p.CUSTOMER_ID),
                            _project=p.BQ_INPUT_PROJECT_ID,
                            _landing_dataset=p.BQ_INPUT_DATASET_ENT_ENRV_PROD,
                            _table_id='GER_OPT_CLIENT_PHARM_CLAIMS_STANDARD',
                            _pharm_list=uf.get_formatted_string(p.AGREEMENT_PHARMACY_LIST['GNRC'])
                        ),
                        custom=True
                    )

                    # Read target rates for MEDD
                    target_rate = uf.read_BQ_data(
                        project_id=p.BQ_INPUT_PROJECT_ID,
                        dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                        table_id='pharmacy_target_rate_medd_custom',
                        query=BQ.pharmacy_target_rate_medd_custom.format(
                            _customer_id=uf.get_formatted_string(p.CUSTOMER_ID),
                            _data_start=eff_date['CONTRACT_EFF_DT'][0],
                            _project=p.BQ_INPUT_PROJECT_ID,
                            _landing_dataset=p.BQ_INPUT_DATASET_ENT_ENRV_PROD,
                            _table_id='GER_OPT_CLIENT_PHARM_CLAIMS_STANDARD',
                            _ratio=p.PHARM_TARGET_RATE_RATIO,
                            _pharm_list=uf.get_formatted_string(p.AGREEMENT_PHARMACY_LIST['GNRC'])
                        ),
                        custom=True
                    )
            else:
                # Map Pharmacy Chain Group to individual claims for full year
                pharmacy_measurement_mapping = uf.read_BQ_data(
                    project_id=p.BQ_INPUT_PROJECT_ID,
                    dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                    table_id='pharmacy_measurement_mapping_medd_custom',
                    query=BQ.pharmacy_measurement_mappingNY_medd_custom.format(
                        _customer_id=uf.get_formatted_string(p.CUSTOMER_ID),
                        _project=p.BQ_INPUT_PROJECT_ID,
                        _landing_dataset=p.BQ_INPUT_DATASET_ENT_ENRV_PROD,
                        _table_id='GER_OPT_CLIENT_PHARM_CLAIMS_STANDARD' + p.WS_SUFFIX,
                        _pharm_list=uf.get_formatted_string(p.AGREEMENT_PHARMACY_LIST['GNRC'])
                    ),
                    custom=True
                )

                # Read target rates for full year
                target_rate = uf.read_BQ_data(
                    project_id=p.BQ_INPUT_PROJECT_ID,
                    dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                    table_id='pharmacy_target_rate_medd_custom',
                    query=BQ.pharmacy_target_rateNY_medd_custom.format(
                        _customer_id=uf.get_formatted_string(p.CUSTOMER_ID),
                        _data_start=p.DATA_START_DAY,
                        _project=p.BQ_INPUT_PROJECT_ID,
                        _landing_dataset=p.BQ_INPUT_DATASET_ENT_ENRV_PROD,
                        _table_id='GER_OPT_CLIENT_PHARM_CLAIMS_STANDARD' + p.WS_SUFFIX,
                        _ratio=p.PHARM_TARGET_RATE_RATIO,
                        _pharm_list=uf.get_formatted_string(p.AGREEMENT_PHARMACY_LIST['GNRC'])
                    ),
                    custom=True
                )

            # Set MCHOICE target rate
            mchoice_target_rate = p.MEDD_MCHOICE_TARGET_RATE
        else:
            # Read pharmacy measurement mapping and target rates from files
            pharmacy_measurement_mapping = pd.read_csv(p.FILE_INPUT_PATH + p.PHARMACY_MEASUREMENT_MAPPING_FILE, dtype=p.VARIABLE_TYPE_DIC)
            target_rate = pd.read_csv(p.FILE_INPUT_PATH + p.TARGET_RATE_FILE, dtype=p.VARIABLE_TYPE_DIC)

            if p.FULL_YEAR:
                # Handle full-year target rates
                target_rateNY = standardize_df(pd.read_csv(p.FILE_INPUT_PATH + p.PHARM_GUARANTEE_MEDD_NY, dtype=p.VARIABLE_TYPE_DIC))
                target_rateNY.rename(columns={'PHARMACY': 'CHAIN_GROUP', 'RATE': 'Rate'}, inplace=True)
                target_rate.drop(columns=['Rate'], inplace=True)
                target_rate = pd.merge(target_rate, target_rateNY, how='left', on=['CHAIN_GROUP'])

            # Set MCHOICE target rate
            mchoice_target_rate = p.MEDD_MCHOICE_TARGET_RATE

        # -----------------------------
        # Standardize and merge data
        # -----------------------------
        pharmacy_measurement_mapping = standardize_df(pharmacy_measurement_mapping)
        measurement_mapping = standardize_df(_read_measurement_mapping())

        # Merge measurement mapping with pharmacy measurement mapping
        pharmacy_measurement_mapping = pharmacy_measurement_mapping.merge(measurement_mapping, on=['NETWORK', 'MEASUREMENT_CLEAN', 'CHAIN_GROUP_TEMP'])

        # -----------------------------
        # Handle missing subgroups and finalize
        # -----------------------------
        target_rate_prior = target_rate[target_rate['PERIOD'] == 'PRIOR']
        target_rate_current = target_rate[target_rate['PERIOD'] == 'CURRENT']

        missing_subgroups = set(p.AGREEMENT_PHARMACY_LIST['GNRC']) - set(target_rate_current['CHAIN_GROUP'])
        print(f"Warning: No 'CURRENT' data found for missing subgroups: {missing_subgroups}")
        prior_data = target_rate_prior[target_rate_prior['CHAIN_GROUP'].isin(missing_subgroups)]

        if not prior_data.empty:
            target_rate = pd.concat([target_rate_current, prior_data], ignore_index=True)
        else:
            target_rate = target_rate_current

        # Merge pharmacy measurement mapping with target rates
        pharmacy_guarantees = pharmacy_measurement_mapping.merge(target_rate, on=['CHAIN_GROUP', 'MEASUREMENT_CLEAN', 'NETWORK', 'BG_FLAG'], how='left')

        # Set guarantee types to AWP for MEDD
        pharmacy_guarantees['PHRM_GRTE_TYPE'] = 'AWP'

        # -----------------------------
        # Assign chain subgroups
        # -----------------------------
        def _assign_chain_subgroup_temp(row):
            if row['CHAIN_GROUP'] == 'KRG':
                return ['KRG', 'MCHOICE_KRG']
            elif row['CHAIN_GROUP'] == 'MCHOICE':
                return ['MCHOICE_CVS']
            else:
                return [row['CHAIN_GROUP']]

        pharmacy_guarantees['CHAIN_SUBGROUP'] = pharmacy_guarantees.apply(_assign_chain_subgroup_temp, axis=1)
        pharmacy_guarantees = pharmacy_guarantees.explode('CHAIN_SUBGROUP')

        # Rename columns for consistency
        pharmacy_guarantees.rename(columns={'CHAIN_GROUP': 'PHARMACY',
                                            'CHAIN_SUBGROUP': 'PHARMACY_SUBGROUP_TEMP',
                                            'Rate': 'RATE'}, inplace=True)
        
        # -----------------------------
        # Merge with mapping 
        # -----------------------------
        mapping = measurement_mapping[['CLIENT', 'BREAKOUT', 'MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP', 'CHAIN_SUBGROUP']].drop_duplicates()
        # Merge the mapping with the pharmacy guarantees to align chain groups and subgroups
        pharmacy_guarantees = pd.merge(mapping, 
                                       pharmacy_guarantees, 
                                       how='left', 
                                       left_on=['CLIENT','BREAKOUT','MEASUREMENT', 'BG_FLAG', 'CHAIN_GROUP','CHAIN_SUBGROUP'],
                                       right_on=['CLIENT','BREAKOUT','MEASUREMENT', 'BG_FLAG', 'PHARMACY','PHARMACY_SUBGROUP_TEMP'])

        #Updates the MCHOICE related pharm guar
        pharmacy_guarantees['RATE'] = np.where(pharmacy_guarantees['CHAIN_SUBGROUP'].isin(['MCHOICE_CVS', 'MCHOICE_KRG']),
                                               mchoice_target_rate,
                                               pharmacy_guarantees['RATE'])
        
        # Finalize columns and handle MCHOICE-specific rates
        pharmacy_guarantees = pharmacy_guarantees[['CLIENT', 
                                                   'REGION', 
                                                   'BREAKOUT', 
                                                   'MEASUREMENT', 
                                                   'BG_FLAG',
                                                   'CHAIN_GROUP', 
                                                   'CHAIN_SUBGROUP', 
                                                   'RATE', 
                                                   'PHRM_GRTE_TYPE']].dropna().drop_duplicates().rename(columns={'CHAIN_GROUP': 'PHARMACY',
                                                                                                                'CHAIN_SUBGROUP': 'PHARMACY_SUB'})
        
        # -----------------------------
        # Fix no guarantee rates
        # -----------------------------
        # Fix for bug where some networks have no guarantee rates, but other client/breakout/region/pharmacy/measurement
        # groupings DO have a rate -- if we don't do this, the next block of code fills in the missing values from the other
        # measurement. We can just drop any na rows if they have actual rates elsewhere in the table, instead of filling them
        # in, since we do a .drop_duplicates() below.
        na_rates = pharmacy_guarantees[pharmacy_guarantees['RATE'].isna()].groupby(['CLIENT', 'BREAKOUT', 'REGION', 'PHARMACY','PHARMACY_SUB', 'MEASUREMENT','BG_FLAG','PHRM_GRTE_TYPE'], as_index=False)['RATE'].size().rename(columns={'size': 'Num NA'})
        notna_rates = pharmacy_guarantees[pharmacy_guarantees['RATE'].notna()].groupby(['CLIENT', 'BREAKOUT', 'REGION', 'PHARMACY','PHARMACY_SUB', 'MEASUREMENT','BG_FLAG','PHRM_GRTE_TYPE'], as_index=False)['RATE'].size().rename(columns={'size': 'Num Not NA'})
        # inner join: we only care about cases where there are both NaN and non-NaN rates
        joined_na_rates = na_rates.merge(notna_rates, on=['CLIENT', 'BREAKOUT', 'REGION', 'PHARMACY','PHARMACY_SUB','MEASUREMENT','BG_FLAG','PHRM_GRTE_TYPE'], how='inner')
        joined_na_rates['RATIO'] = joined_na_rates['Num NA']/joined_na_rates['Num Not NA']
        pharmacy_guarantees = pharmacy_guarantees.merge(joined_na_rates, on=['CLIENT', 'BREAKOUT', 'REGION', 'PHARMACY','PHARMACY_SUB', 'MEASUREMENT','BG_FLAG','PHRM_GRTE_TYPE'], how='left')
        # Two conditions: the rate is not NaN, or the ratio is NaN. If the grouping has ONLY all NaNs or no NaNs,
        # then the inner join doesn't produce a row and the ratio is NaN, and we should keep everything.
        # If there are some NaNs and some non NaNs, then the second condition is False, and we only drop the NaN rate rows.
        pharmacy_guarantees = pharmacy_guarantees[(pharmacy_guarantees['RATE'].notna()) | (pharmacy_guarantees['RATIO'].isna())].drop(columns=['RATIO', 'Num NA', 'Num Not NA'])
        
        pharmacy_guarantees_mail = pharmacy_guarantees.loc[pharmacy_guarantees.MEASUREMENT =='M30']

        breakout_mapping = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.BREAKOUT_MAPPING_FILE, dtype = p.VARIABLE_TYPE_DIC)
        pharmacy_guarantees.drop(columns = ['BREAKOUT'], inplace = True) #Delete breakout here for Retail channel, as it will corrected later on when fixing missing value

        pharmacy_guarantees_30 = pharmacy_guarantees.loc[pharmacy_guarantees.MEASUREMENT =='R30']
        pharmacy_guarantees_90 = pharmacy_guarantees.loc[pharmacy_guarantees.MEASUREMENT =='R90']
        pharmacy_guarantees_join = pd.merge(pharmacy_guarantees_30, pharmacy_guarantees_90, how='outer', on=['CLIENT', 'REGION', 'PHARMACY','PHARMACY_SUB','BG_FLAG','PHRM_GRTE_TYPE'], suffixes = ('_30', '_90'))

        #fix R90 missing value
        pharmacy_guarantees_join.loc[pharmacy_guarantees_join.RATE_90.isna(), 'MEASUREMENT_90'] = pharmacy_guarantees_join.loc[pharmacy_guarantees_join.RATE_90.isna(), 'MEASUREMENT_30'].str.replace('30', '90', regex=False)
        pharmacy_guarantees_join = pd.merge(pharmacy_guarantees_join, breakout_mapping[['BG_FLAG','MEASUREMENT','BREAKOUT']].rename(columns = {'MEASUREMENT':'MEASUREMENT_90'}),
                                            how='left', on=['BG_FLAG', 'MEASUREMENT_90']
                                           ).rename(columns = {'BREAKOUT':'BREAKOUT_90'})
        pharmacy_guarantees_join.loc[pharmacy_guarantees_join.RATE_90.isna(), 'RATE_90'] = pharmacy_guarantees_join.loc[pharmacy_guarantees_join.RATE_90.isna(), 'RATE_30']                               
        
        #fix R30 missing value
        pharmacy_guarantees_join.loc[pharmacy_guarantees_join.RATE_30.isna(), 'MEASUREMENT_30'] = pharmacy_guarantees_join.loc[pharmacy_guarantees_join.RATE_30.isna(), 'MEASUREMENT_90'].str.replace('90', '30', regex=False)
        pharmacy_guarantees_join = pd.merge(pharmacy_guarantees_join, breakout_mapping[['BG_FLAG','MEASUREMENT','BREAKOUT']].rename(columns = {'MEASUREMENT':'MEASUREMENT_30'}), 
                                            how='left', on=['BG_FLAG', 'MEASUREMENT_30']
                                           ).rename(columns = {'BREAKOUT':'BREAKOUT_30'})
        pharmacy_guarantees_join.loc[pharmacy_guarantees_join.RATE_30.isna(), 'RATE_30'] = pharmacy_guarantees_join.loc[pharmacy_guarantees_join.RATE_30.isna(), 'RATE_90']

        pharmacy_guarantees_30_2 = pharmacy_guarantees_join[['PHARMACY','PHARMACY_SUB', 'BREAKOUT_30', 'REGION', 'MEASUREMENT_30', 'CLIENT', 'RATE_30',  'BG_FLAG','PHRM_GRTE_TYPE']].rename(columns={'MEASUREMENT_30':'MEASUREMENT', 'BREAKOUT_30':'BREAKOUT', 'RATE_30':'RATE'}).copy(deep=True)
        pharmacy_guarantees_90_2 = pharmacy_guarantees_join[['PHARMACY','PHARMACY_SUB','BREAKOUT_90', 'REGION', 'MEASUREMENT_90', 'CLIENT', 'RATE_90', 'BG_FLAG','PHRM_GRTE_TYPE']].rename(columns={'MEASUREMENT_90':'MEASUREMENT', 'BREAKOUT_90':'BREAKOUT', 'RATE_90':'RATE'}).copy(deep=True)

        pharmacy_guarantees = pd.concat([pharmacy_guarantees_mail, pharmacy_guarantees_30_2,pharmacy_guarantees_90_2])
        pharmacy_guarantees.dropna(inplace=True)
        
        # Return the final pharmacy guarantees table
        return pharmacy_guarantees
        
    # ------------------------------------------------------------------
    # Dispatch table keeps main body flat
    # ------------------------------------------------------------------
    builders = {
        "COMMERCIAL": _build_commercial,
        "MEDICAID": _build_medicaid,
        "MEDD": _build_medd,
    }
    
    pharmacy_guarantees = builders[p.CLIENT_TYPE]()
    print(pharmacy_guarantees)
   
    # ------------------------------------------------------------------
    # Shared post‑processing (identical to original)
    # ------------------------------------------------------------------
    pharmacy_guarantees.drop_duplicates(inplace=True)
    pharmacy_guarantees.rename(columns={"CHAIN_GROUP": "PHARMACY", 
                                        "CHAIN_SUBGROUP": "PHARMACY_SUB", 
                                        "Rate": "RATE"}, inplace=True)

    virtual_r90_flag = add_virtual_r90(update_guarantees=False, return_check=True)
    if p.CLIENT_TYPE != "MEDD":
        pharmacy_guarantees = add_rur_guarantees(pharmacy_guarantees, virtual_r90_flag)

    qa_dataframe(pharmacy_guarantees, dataset=p.PHARM_GUARANTEE_FILE)
    pharmacy_guarantees.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.PHARM_GUARANTEE_FILE, index=False)

def prepare_mac_mapping():
    """ Load vcml_reference file from CSV and transform into format expected by LP save as mac_mapping file
    
    intput: p.VCML_REFERENCE_FILE, p.MEASUREMENT_MAPPING, p.CLIENT_GUARANTEE_FILE, p.PREFERRED_PHARM_FILE
    output: p.MAC_MAPPING_FILE
    1. Create a master list of all possible VCML-codes to MEASUREMENT and CHAIN GROUP
    loop over clients
    2. Use waterfall logic to find the mac list for HMA, nonpref 30, nonpref 90, pref 30, and pref 90
        Enter these into the mac mapping dictionary (used like defaults)
    3. Use a separate vcml mapping dictionary to directly find every big capped and PSAO vcml listed in the vcml table
        For example, 3477 directly gets mapped to CVS R90, does not use a waterfall to find this
    4. Waterfall the unmapped vcmls
        First fill in the defaults, then see if waterfall is needed
    5. Filter out unmapped mac lists
    7. several final checks
    
    output file is used in MAC_CONSTRAINTS and in daily_input_read.py joined to the main lp dataset gpi_vol_awp_agg_ytd by 'CLIENT', 'BREAKOUT', 'MEASUREMENT', 'GPI_NDC', 'GPI', 'NDC', 'CHAIN_GROUP','REGION'

    """
    import pandas as pd
    import numpy as np
    import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df, add_virtual_r90
    from qa_checks import qa_dataframe
    import BQ
    import util_funcs as uf
    
    ###############################################
    # keep track of any issues during mac-mapping
    WARNINGS = False
    WARNING_LIST = []
    
    # VCML file. The VCML reference includes the list of unique VCMLs for each client
    if p.READ_FROM_BQ:
        if p.TRUECOST_CLIENT or p.UCL_CLIENT:
            vcml_df = uf.read_BQ_data(
            BQ.vcml_reference,
            project_id=p.BQ_INPUT_PROJECT_ID,
            dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
            table_id = 'vcml_reference'+p.WS_SUFFIX,
            customer = ', '.join(sorted(p.CUSTOMER_ID)),
            )
            
            tru_vcml_df = vcml_df[vcml_df['VCML_ID'].str.contains(p.APPLY_VCML_PREFIX)]
            mac_vcml_df = vcml_df[vcml_df['VCML_ID'].str.contains('MAC')]
            mac_vcml_df['VCML_ID'] = p.APPLY_VCML_PREFIX + mac_vcml_df['VCML_ID'].str[3:]            
            mac_vcml_df['BASE_MAC_LIST_ID'] = tru_vcml_df['BASE_MAC_LIST_ID'].unique()[0]
            vcml_df_final = pd.concat([tru_vcml_df,mac_vcml_df]).drop_duplicates(subset = tru_vcml_df.columns.difference(['REC_EFFECTIVE_DATE','REC_EXPIRATION_DATE']).tolist(), keep='first')
            vcml_df = vcml_df_final.copy()

        else:
            vcml_df = uf.read_BQ_data(
                BQ.vcml_reference,
                project_id=p.BQ_INPUT_PROJECT_ID,
                dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                table_id = 'vcml_reference'+p.WS_SUFFIX,
                customer = ', '.join(sorted(p.CUSTOMER_ID))
            )
    else:
        vcml_df = pd.read_csv(p.FILE_INPUT_PATH + p.VCML_REFERENCE_FILE, dtype = p.VARIABLE_TYPE_DIC)
    vcml_df = standardize_df(vcml_df)
    # Filter out to make sure only current information is used
    vcml_df = vcml_df[vcml_df['REC_CURR_IND'] == 'Y']

    # Save for later use
    vcml_df.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.VCML_REFERENCE_FILE, index=False)
    # rename VCML_ID for merge later
    vcml_df = vcml_df[['CUSTOMER_ID','VCML_ID','CHNL_IND']].rename(columns={'VCML_ID':'CLIENT_VCML_ID'})
    
    # remove MAC and the CUSTOMER_ID from the VCML_ID
    vcml_df_list = []
    # Since the customer IDs may have different lengths, loop through each one and then concat the results
    for cid in vcml_df['CUSTOMER_ID'].unique():
        t_vcml_df = vcml_df[vcml_df['CUSTOMER_ID']==cid]
        t_vcml_df['CLIENT_MAC_LIST'] = t_vcml_df['CLIENT_VCML_ID'].str[3+len(cid):]
        vcml_df_list.append(t_vcml_df)
    vcml_df = pd.concat(vcml_df_list)
    
    # measurement mapping used to identify VCML_ID combinations with no claims
    if p.READ_FROM_BQ:
        measurement_mapping = uf.read_BQ_data(
                BQ.ger_opt_msrmnt_map.format(_customer_id=uf.get_formatted_string(p.CUSTOMER_ID)),
                project_id=p.BQ_INPUT_PROJECT_ID,
                dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                table_id="combined_measurement_mapping" + p.WS_SUFFIX + p.CCP_SUFFIX,
                customer=', '.join(sorted(p.CUSTOMER_ID)))
    else:
        measurement_mapping = pd.read_csv(p.FILE_INPUT_PATH + p.MEASUREMENT_MAPPING, dtype = p.VARIABLE_TYPE_DIC)
    measurement_mapping = standardize_df(measurement_mapping)
    x_mapping = measurement_mapping[measurement_mapping['NETWORK'].isin(['1NCSTN', '2NVSTN', '1EDSTN'])][['CLIENT', 'NETWORK']].drop_duplicates(keep='first')
    x_mapping['VCML'] = x_mapping['NETWORK'].map({'1EDSTN' : 'XT', '1NCSTN': 'XR', '2NVSTN': 'XA'})
    measurement_mapping = measurement_mapping.groupby(['CLIENT','MEASUREMENT_CLEAN','CHAIN_GROUP','CHAIN_SUBGROUP'])['FULLAWP']\
                                                        .sum() \
                                                        .reset_index()
                                                        
    # Stop the Run if R90OK chain_subgroups are present (meaning R90OK network is set up) but no R90OK VCML is present.
    for customer_id in measurement_mapping['CLIENT'].unique():
        if (p.APPLY_VCML_PREFIX+customer_id+'OK') in vcml_df.CLIENT_VCML_ID.unique():
            assert measurement_mapping.loc[measurement_mapping.CLIENT==customer_id, 'CHAIN_SUBGROUP'].str.contains('_R90OK').any(), \
                   r"Client {customer_id} has R90OK Maclist but no R90OK claims"
        if measurement_mapping.loc[measurement_mapping.CLIENT==customer_id, 'CHAIN_SUBGROUP'].str.contains('_R90OK').any():
            assert (p.APPLY_VCML_PREFIX+customer_id+'OK') in vcml_df.CLIENT_VCML_ID.unique(), f"Client {customer_id} has R90OK claims but no R90OK Maclist"
        if ((measurement_mapping.CHAIN_SUBGROUP.str.contains('_R90OK')) & (measurement_mapping.MEASUREMENT_CLEAN!='R90')).any():
            assert False, "R90OK claims should only appear with MEASUREMENT=R90. Check measurement mapping"
    
    # client guarantees - used to identify VCML_ID combinations without guarantees.
    if p.TRUECOST_CLIENT:
        client_guarantees = uf.read_BQ_data(
            project_id=p.BQ_INPUT_PROJECT_ID,
            dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
            table_id='combined_measurement_mapping' + p.WS_SUFFIX,
            client = ', '.join(sorted(p.CUSTOMER_ID)),
            query = BQ.client_guarantees_dummy_TC.format(
                _customer_id = uf.get_formatted_string(p.CUSTOMER_ID),
                _project = p.BQ_INPUT_PROJECT_ID, 
                _landing_dataset = p.BQ_INPUT_DATASET_DS_PRO_LP,
                _table_id = 'combined_measurement_mapping'+ p.WS_SUFFIX),
            custom = True)
    elif p.READ_FROM_BQ:
        client_guarantees = uf.read_BQ_data(
            BQ.client_guarantees,
            project_id=p.BQ_INPUT_PROJECT_ID,
            dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
            table_id='client_guarantees' + p.WS_SUFFIX,
            client = ', '.join(sorted(p.CUSTOMER_ID)))
    else:
        client_guarantees = pd.read_csv(p.FILE_INPUT_PATH + p.CLIENT_GUARANTEE_FILE, dtype = p.VARIABLE_TYPE_DIC)
    
    ## Modify client guarantee according to data read
    if p.GENERIC_OPT and not p.BRAND_OPT:
        client_guarantees = client_guarantees[client_guarantees['BG_FLAG'] == 'G']
    elif not p.GENERIC_OPT and p.BRAND_OPT:
        client_guarantees = client_guarantees[client_guarantees['BG_FLAG'] == 'B']
    client_guarantees = standardize_df(client_guarantees)
    client_guarantees = add_virtual_r90(client_guarantees, update_guarantees = True)
    
    #Read in client guarantees for MEDD WS run
    if p.CLIENT_TYPE == 'MEDD' and p.FULL_YEAR and not p.EGWP:
        #For MEDD new year pricing: read in custom client guarantee file with the new year guarantees
        client_guarantees_ny = standardize_df(pd.read_csv(p.FILE_INPUT_PATH + p.CLIENT_GUARANTEE_MEDD_NY, dtype = p.VARIABLE_TYPE_DIC))
        cid_mapping = standardize_df(pd.read_csv(p.FILE_INPUT_PATH + p.CUSTOMER_ID_MAPPING_MEDD_NY, dtype = {'CUSTOMER_ID': 'string', 'NEW_CUSTOMER_ID':'string'})).set_index('CUSTOMER_ID').to_dict()
        client_guarantees_ny = client_guarantees_ny.loc[client_guarantees_ny.CLIENT == cid_mapping['NEW_CUSTOMER_ID'][p.CUSTOMER_ID[0]]]
        client_guarantees_ny = add_virtual_r90(client_guarantees_ny, update_guarantees = True)
        client_guarantees_ny = client_guarantees_ny[['MEASUREMENT','PHARMACY_TYPE','RATE']]
        client_guarantees_ny = client_guarantees_ny.set_index(['MEASUREMENT','PHARMACY_TYPE'])
        
        client_guarantees = client_guarantees.set_index(['MEASUREMENT','PHARMACY_TYPE'])
        client_guarantees.update(client_guarantees_ny)
        client_guarantees = client_guarantees.reset_index()
        
    if p.READ_FROM_BQ:
        client_guarantees.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.CLIENT_GUARANTEE_FILE, index = False)

    pref_pharm_list = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.PREFERRED_PHARM_FILE, dtype=p.VARIABLE_TYPE_DIC)
    pref_pharm_list = standardize_df(pref_pharm_list)

    ######## STEP #1
    # Build the complete list of VCML-> MEASUREMENT/CHAIN_GROUP mappings using the full pharmacy list from the parameter file.    
    full_pharm_list = list(set(p.PHARMACY_LIST['BRND']+p.PHARMACY_LIST['GNRC']))
    full_pharm_list = list(set(np.concatenate([measurement_mapping[measurement_mapping['CHAIN_GROUP']==cg]['CHAIN_SUBGROUP'].unique() for cg in full_pharm_list])))
    full_pharm_list = list(set(full_pharm_list) | set(p.BIG_CAPPED_PHARMACY_LIST['BRND'] + p.BIG_CAPPED_PHARMACY_LIST['GNRC']))

    # Determine the MAIL_LIST based on the presence of 'MCHOICE_CVS' and 'MCHOICE_KRG' in full_pharm_list
    MAIL_LIST = []
    if ('MCHOICE_CVS' in full_pharm_list) and ('MCHOICE_KRG' in full_pharm_list):
        full_pharm_list.remove('MCHOICE_KRG')  
        full_pharm_list.remove('MCHOICE_CVS')  
        MAIL_LIST = ['MCHOICE_KRG','MCHOICE_CVS','MAIL']
    elif 'MCHOICE_KRG' in full_pharm_list:
        full_pharm_list.remove('MCHOICE_KRG') 
        MAIL_LIST = ['MCHOICE_KRG','MAIL']
    elif 'MCHOICE_CVS' in full_pharm_list:
        full_pharm_list.remove('MCHOICE_CVS')  
        MAIL_LIST = ['MCHOICE_CVS','MAIL']
    else:
        MAIL_LIST = ['MAIL']

    # find the first available vcml in the provided vcml df
    def single_waterfall(wf, vcml_in_df):
        vcml = str(wf[0])
        if len(wf) > 1:
            if any(vcml_in_df.CLIENT_MAC_LIST == vcml):
                return vcml
            else:
                return single_waterfall(wf[1:], vcml_in_df)
        else:
            return vcml

    # find defaults for mac mapping dictionaries
    hma_mapping = single_waterfall(['11'], vcml_df)
    np_30 = single_waterfall(['199', '1'], vcml_df)
    np_90 = single_waterfall(['399', '3', '199', '1'], vcml_df)
    chd_np_30 = single_waterfall(['199', 'C4', '1'], vcml_df)
    chd_np_90 = single_waterfall(['399', 'C9', '3', '199', 'C4', '1'], vcml_df)
    p_30 = single_waterfall(['177', '1'], vcml_df)
    if vcml_df.CLIENT_VCML_ID.astype(str).str.contains(p.APPLY_VCML_PREFIX+p.CUSTOMER_ID[0]+'373', regex=False).any():
        p_90 = single_waterfall(['373', '3', '177', '1'], vcml_df)
    else:
        p_90 = single_waterfall(['377', '3', '177', '1'], vcml_df)
    # the correct is 377 but 373 is a client unique situation
    # small chains that will have EITHER pref OR nonpref VCMLs for Med D -- pick whichever appears
    pbx_30 = single_waterfall(['14', 'P89'], vcml_df)
    pbx_90 = single_waterfall(['42', 'E89'], vcml_df)
    cst_30 = single_waterfall(['17', 'P24'], vcml_df)
    cst_90 = single_waterfall(['47', 'E24'], vcml_df)
    if p.CLIENT_TYPE == 'MEDD':
        # Different mappings for AHD, ABS, MJR between Med D & commercial
        abs_30 = single_waterfall(['15', 'P5'], vcml_df)
        abs_90 = single_waterfall(['45', 'E5'], vcml_df)
        ahd_30 = single_waterfall(['16', 'P91'], vcml_df)
        ahd_90 = single_waterfall(['46', 'E91'], vcml_df)
        mjr_90 = '48'
    else:
        abs_30 = single_waterfall(['15', 'P5'], vcml_df)
        abs_90 = single_waterfall(['43', 'E5'], vcml_df)
        ahd_30 = single_waterfall(['16', 'P91'], vcml_df)
        ahd_90 = single_waterfall(['45', 'E91'], vcml_df)
        mjr_90 = '46'
    
    
    # mapping dict for big capped and PSAO
    bigcap_psao_map = {'4': ('CVS', 'R30'), '477': ('CVS', 'R30'), '499': ('CVS', 'R30'),
                       '34': ('CVS', 'R90'), '3477': ('CVS', 'R90'), '3499': ('CVS', 'R90'),
                       '5': ('WAG', 'R30'), '577': ('WAG', 'R30'), '599': ('WAG', 'R30'), 'P93': ('WAG', 'R30'),
                       '35': ('WAG', 'R90'), '3577': ('WAG', 'R90'), '3599': ('WAG', 'R90'), 'E93': ('WAG', 'R90'),
                       '6': ('RAD', 'R30'), '677': ('RAD', 'R30'), '699': ('RAD', 'R30'),
                       '36': ('RAD', 'R90'), '3677': ('RAD', 'R90'), '3699': ('RAD', 'R90'),
                       '7': ('WMT', 'R30'), '777': ('WMT', 'R30'), '799': ('WMT', 'R30'), 'P37': ('WMT', 'R30'),
                       '37': ('WMT', 'R90'), '3777': ('WMT', 'R90'), '3799': ('WMT', 'R90'), 'E37': ('WMT', 'R90'),
                       '8': ('KRG', 'R30'), '877': ('KRG', 'R30'), '899': ('KRG', 'R30'), 'P40': ('KRG', 'R30'),
                       'P57': ('KRG', 'R30'),
                       '38': ('KRG', 'R90'), '3877': ('KRG', 'R90'), '3899': ('KRG', 'R90'), 'E40': ('KRG', 'R90'),
                       'E57': ('KRG', 'R90'),
                       '11': ('ACH', 'R30'), '117': ('ACH', 'R30'), '119': ('ACH', 'R30'),
                       '311': ('ACH', 'R90'), '3117': ('ACH', 'R90'), '3119': ('ACH', 'R90'), 
                       '20': ('HMA', 'R90'),
                       '22': ('ART', 'R30'), '227': ('ART', 'R30'), '229': ('ART', 'R30'),
                       '322': ('ART', 'R90'), '3227': ('ART', 'R90'), '3229': ('ART', 'R90'),
                       '33': ('CAR', 'R30'), '337': ('CAR', 'R30'), '339': ('CAR', 'R30'), 
                       '333': ('CAR', 'R90'), '3337': ('CAR', 'R90'), '3339': ('CAR', 'R90'), '30': ('CAR', 'R90'),  
                       '41': ('CVSSP', 'R30'),
                       '91': ('CVSSP', 'R90'),
                       '44': ('ELE', 'R30'), '447': ('ELE', 'R30'), '449': ('ELE', 'R30'),
                       '344': ('ELE', 'R90'), '3447': ('ELE', 'R90'), '3449': ('ELE', 'R90'), '40': ('ELE', 'R90'), 
                       '55': ('EPC', 'R30'), '557': ('EPC', 'R30'), '559': ('EPC', 'R30'),
                       '355': ('EPC', 'R90'), '3557': ('EPC', 'R90'), '3559': ('EPC', 'R90'), '50': ('EPC', 'R90'),
                       '66': ('TPS', 'R30'), '667': ('TPS', 'R30'), '669': ('TPS', 'R30'),
                       '366': ('TPS', 'R90'), '3667': ('TPS', 'R90'), '3669': ('TPS', 'R90'), '60': ('TPS', 'R90'),
                       '14': ('PBX', 'R30'), 'P89': ('PBX', 'R30'),
                       '42': ('PBX', 'R90'), 'E89': ('PBX', 'R90'),
                       '15': ('ABS', 'R30'), 'P5': ('ABS', 'R30'),
                       '43': ('ABS', 'R90'), 'E5': ('ABS', 'R90'),
                       '16': ('AHD', 'R30'), 'P91': ('AHD', 'R30'),
                       '45': ('AHD', 'R90'), 'E91': ('AHD', 'R90'),
                       '18': ('MJR', 'R30'),
                       '46': ('MJR', 'R90'), '48': ('MJR', 'R90'),
                       '54': ('LWD', 'R30'),
                       'E54': ('LWD', 'R90'),
                       '17': ('CST', 'R30'), 'P24': ('CST', 'R30'),
                       '47': ('CST', 'R90'), 'E24': ('CST', 'R90'),
                       '90': ('GIE', 'R30'),
                       'E90': ('GIE', 'R90'),
                       '53': ('HYV', 'R30'),
                       'E53': ('HYV', 'R90'),
                       'P58': ('KIN', 'R30'),
                       'E58': ('KIN', 'R90'),
                       'P11': ('WGS', 'R30'), # Wegmans not Walgreens!
                       'E11': ('WGS', 'R90'),
                       'R1': ('RUR', 'R30'), #2025 commercial
                       'R3': ('RUR', 'R90') , 'R9': ('RUR', 'R90'),
                       'C4': ('CHD', 'R30'),
                       'C9': ('CHD', 'R90'), 
                       'H1': ('HYV', 'R30'),
                       'H3': ('HYV', 'R90') ,
                       '59' : ('THF', 'R30'), 'E59' : ('THF', 'R90'), #2025 UPDATE MEDD
                       '61' : ('HRT', 'R30'), 'E61' : ('HRT', 'R90'),
                       '62' : ('PUR', 'R30'), 'E62' : ('PUR', 'R90'),
                       '64' : ('AMZ', 'R30'), 'E64' : ('AMZ', 'R90'),
                       '81' : ('BYD', 'R30'), 'E81' : ('BYD', 'R90'),
                       'P38' : ('FCD', 'R30'), 'E38' : ('FCD', 'R90'),
                       'P55' : ('IGD', 'R30'), 'E55' : ('IGD', 'R90'),
                       'P57' : ('KSD', 'R30'), 'E57' : ('KSD', 'R90'),
                       'P88' : ('PCD', 'R30'),  'E88' : ('PCD', 'R90'),
                       'P95' : ('TND', 'R30'), 'E95' : ('TND', 'R90'),
                       'P93' : ('HAD', 'R30'), 'E93' : ('HAD', 'R90'),
                       '68' : ('TPM', 'R30'), 'E68' : ('TPM', 'R90'),
                       '67' : ('PBX', 'R30'), 'E67' : ('PBX', 'R90'),
                       '72' : ('WGS', 'R30'), 'E72' : ('WGS', 'R90'),
                       '71' : ('PCD', 'R30'), 'E71' : ('PCD', 'R90'),
                       '65' : ('IGD', 'R30'), 'E65' : ('IGD', 'R90'),
                       '85' : ('AMZ', 'R30'), 'E85' : ('AMZ', 'R90'),
                       '83' : ('INP', 'R30'), 'E83' : ('INP', 'R90'),
                       
 
                       }
    if p.CLIENT_TYPE=='MEDD':
        bigcap_psao_map.update({
                       '45': ('ABS', 'R90'),
                       '46': ('AHD', 'R90'),
                       '11': ('HMA', 'R30'),
                        '28':('ELE','R30'),
                       'E28':('ELE','R90')
        })

    #TODO: should not have these hardcoded here -- move to parameters? # move to an input dataset?
    # Both dictionaries (30 and 90) come from Scott Stankey/Dawn Ciullo.  They are current as of Jan 2019 but they can change, both pharmacy names or VCML ID.
    # New chains added Apr 2023 as part of commercial passthrough VCML standardization process.

    mapping_dictionary_30 = {'ABS': abs_30, 'ACH': '1', 'AHD': ahd_30, 'ART': '22', 'CAR': '33', 'CVS': '4', 'CVSSP': '41', 'ELE': '44',
                             'EPC': '55', 'GIE': '90', 'HMA': hma_mapping, 'KIN': 'P58', 'KRG': '8', 'NONPREF_OTH': np_30,
                             'PREF_OTH': p_30, 'RAD': '6', 'SAF': '97', 'TPS': '66', 'WAG': '5', 'WMT': '7', 'LTC': '10', 
                             'GEN': '1', 'PBX': pbx_30, 'MJR': '18','HYV':'53','CST': cst_30,'AHS':'13', 'WGS': 'P11', 'LWD': '54',
                             'HVD':'P48', 'HEB' : 'P49', 'PCD': 'P88','MAD':'P94','AMZ':'64','THF':'59','HRT':'61',
                             'BYD':'81' , 'FCD':'P38' , 'IGD':'P55' , 'KSD':'P57' , 'PCD':'P88' , 'TND' : 'P95' , 'HAD':'P93',
                             'PUR': '62', 'RUR':'R1','CHD':'C4','TPM':'68','INP':'83', 'ARX':'1', 'BGY':'1', 'FVW':'1', 'GUA':'1', 
                             'PMA':'1', 'RPT':'1', 'SMC':'1', 'SMR':'1', 'WIS':'1', 'BRI':chd_np_30, 'DDM':chd_np_30, 'MCY':chd_np_30, 
                             'MGM':chd_np_30, 'RYS':chd_np_30, 'SVU':chd_np_30, 'WFN':chd_np_30}

    mapping_dictionary_90 = {'ABS': abs_90, 'ACH': '3', 'AHD': ahd_90, 'ART': '3', 'CAR': '30', 'CVS': '34', 'CVSSP': '91', 'ELE': '40',
                             'EPC': '50', 'GIE': 'E90', 'HMA': '20', 'KIN': 'E58', 'KRG': '38', 'NONPREF_OTH': np_90,
                             'PREF_OTH': p_90, 'RAD': '36', 'SAF': '39', 'TPS': '60', 'WAG': '35', 'WMT': '37', 
                             'GEN': '3', 'PBX': pbx_90, 'MJR': mjr_90,'HYV':'E53','CST': cst_90, 'WGS': 'E11', 'LWD': 'E54',
                             'HVD':'E48', 'HEB' : 'E49', 'PCD': 'E88', 'MAD':'E94' , 'AMZ':'E64', 'FCD':'E38' , 'IGD':'E55',
                             'KSD':'E57' , 'THF':'E59' , 'HRT':'E61' , 'BYD':'E81' , 'PCD':'E88' , 'TND':'E95' , 'HAD':'E93',
                             'PUR': 'E62', 'RUR':'R3','CHD':'C9','TPM':'E68','INP':'E83', 'ARX':'3', 'BGY':'3', 'FVW':'3', 'GUA':'3', 
                             'PMA':'3', 'RPT':'3', 'SMC':'3', 'SMR':'3', 'WIS':'3', 'BRI':chd_np_90, 'DDM':chd_np_90, 'MCY':chd_np_90, 
                             'MGM':chd_np_90, 'RYS':chd_np_90, 'SVU':chd_np_90, 'WFN':chd_np_90}

    
    mapping_dictionary_30_xr = dict({key+'_EXTRL': 'XR' for key in mapping_dictionary_30})
    mapping_dictionary_30_xa = dict({key+'_EXTRL': 'XA' for key in mapping_dictionary_30})
    mapping_dictionary_90_xt = dict({key+'_EXTRL': 'XT' for key in mapping_dictionary_90})
    # add R90OK VCMLs
    mapping_dictionary_30.update({key+'_R90OK': 'OK' for key in mapping_dictionary_30})
    mapping_dictionary_90.update({key+'_R90OK': 'OK' for key in mapping_dictionary_90})

    custom_mapping = '88'
    # end of hardcoded VCMLs
    
    full_pharm_list_x = [p for p in full_pharm_list if p.endswith('_EXTRL')]
    full_pharm_list_nonx = [p for p in full_pharm_list if not p.endswith('_EXTRL')]
    
    mac_mapping_30 = pd.DataFrame.from_dict({'CHAIN_SUBGROUP': full_pharm_list_nonx, 'MAC_LIST':[mapping_dictionary_30[pharm] for pharm in full_pharm_list_nonx]})
    mac_mapping_30['MEASUREMENT'] = 'R30'
    mac_mapping_90 = pd.DataFrame.from_dict({'CHAIN_SUBGROUP': full_pharm_list_nonx, 'MAC_LIST':[mapping_dictionary_90[pharm] for pharm in full_pharm_list_nonx]})
    mac_mapping_90['MEASUREMENT'] = 'R90'
    
    mac_mapping_30_xr = pd.DataFrame.from_dict({'CHAIN_SUBGROUP': full_pharm_list_x, 'MAC_LIST':[mapping_dictionary_30_xr[pharm] for pharm in full_pharm_list_x]})
    mac_mapping_30_xr['MEASUREMENT'] = 'R30'
    mac_mapping_30_xa = pd.DataFrame.from_dict({'CHAIN_SUBGROUP': full_pharm_list_x, 'MAC_LIST':[mapping_dictionary_30_xa[pharm] for pharm in full_pharm_list_x]})
    mac_mapping_30_xa['MEASUREMENT'] = 'R30' 
    mac_mapping_90_xt = pd.DataFrame.from_dict({'CHAIN_SUBGROUP': full_pharm_list_x, 'MAC_LIST':[mapping_dictionary_90_xt[pharm] for pharm in full_pharm_list_x]})
    mac_mapping_90_xt['MEASUREMENT'] = 'R90'
    
    mac_mapping_mail = pd.DataFrame({'CHAIN_SUBGROUP':MAIL_LIST,'MAC_LIST':['2'] * len(MAIL_LIST),'MEASUREMENT':['M30'] * len(MAIL_LIST)})
    mac_mapping_df = pd.concat([mac_mapping_30,mac_mapping_30_xr,mac_mapping_30_xa,mac_mapping_90,mac_mapping_90_xt,mac_mapping_mail]).sort_values('MAC_LIST')
    del mac_mapping_30,mac_mapping_30_xa,mac_mapping_30_xr,mac_mapping_90_xt,mac_mapping_90,mac_mapping_mail
    # end of complete list of VCML mapping
    
    ######## final mac_mapping with all clients
    all_clients_mac_mapping_columns = ['CUSTOMER_ID','REGION','MEASUREMENT','CHAIN_SUBGROUP','MAC_LIST']
    all_clients_mac_mapping = pd.DataFrame(columns = all_clients_mac_mapping_columns)
    
    # loop through clients
    clients = vcml_df['CUSTOMER_ID'].unique()  
    
    for client in clients:       
        ######## create a client version of all possible VCML-IDs
        client_mac_mapping = mac_mapping_df.copy(deep=True)
        client_mac_mapping['CUSTOMER_ID'] = client
        client_mac_mapping['VCML_ID'] = p.APPLY_VCML_PREFIX + client_mac_mapping['CUSTOMER_ID'].apply(str) + client_mac_mapping['MAC_LIST'].apply(str)
        x_mapping_client = x_mapping[x_mapping['CLIENT'] == client]
        x_filter = list(set(['XA','XR', 'XT']) -set( x_mapping_client['VCML'].tolist()))
        client_mac_mapping = client_mac_mapping[~client_mac_mapping['MAC_LIST'].isin(x_filter)]    
        assert not('XA' in client_mac_mapping['MAC_LIST'].unique() and  'XR' in client_mac_mapping['MAC_LIST'].unique()), "Client has both XA and XR!! Should be a custom run"
        # find preferred pharmacy list by region
        region = client  # TODO: this should be the region when we loop on region
        pref_pharms = pref_pharm_list['PREF_PHARM'].loc[(pref_pharm_list['REGION']==region) & (~(pref_pharm_list['BREAKOUT'].str.contains('M')))].unique()
        assert len(pref_pharms) == 1, "len(pref_pharms) == 1"
        pref_pharms = pref_pharms[0].split(',')
        pref_pharms = list(set(np.concatenate([measurement_mapping[(measurement_mapping['CHAIN_GROUP']==cg) & (measurement_mapping['CLIENT']==client)]['CHAIN_SUBGROUP'].unique() for cg in pref_pharms])))

        # find vcmls for big capped and psao
        vcml_mac_mapping = vcml_df.copy()
        chain_meas_list = vcml_mac_mapping['CLIENT_MAC_LIST'].map(bigcap_psao_map).tolist()
        
        if pd.isna(chain_meas_list[0]):
            chain_meas_list[0] = (None, None)
        vcml_mac_mapping[['CHAIN_SUBGROUP', 'MEASUREMENT']] = pd.DataFrame(chain_meas_list, index=vcml_mac_mapping.index)
        vcml_mac_mapping = vcml_mac_mapping[pd.notna(vcml_mac_mapping['MEASUREMENT'])]
        vcml_mac_mapping = vcml_mac_mapping[['CUSTOMER_ID', 'MEASUREMENT', 'CHAIN_SUBGROUP', 'CLIENT_MAC_LIST']]

        ######## join the all-combinations data to each client's VCML reference
        # to identify which pharmacies/measurements have unique MAC_LISTs for each client
        client_vcml_df = vcml_df[vcml_df['CUSTOMER_ID'] == client]
        client_mac_mapping = client_mac_mapping.merge(client_vcml_df,left_on = ['CUSTOMER_ID','VCML_ID']\
                                                      ,right_on=['CUSTOMER_ID','CLIENT_VCML_ID']\
                                                      , how ='left')
        
        # this checks whether the client has a custom network buildout = VCML_ID = 88
        has_88 = any(client_vcml_df.CLIENT_MAC_LIST == str(custom_mapping))
        if (has_88):
            WARNINGS = True
            WARNING_LIST = WARNING_LIST + ["WARNING, {} has VCML_ID = {}xxxx88, which may be a custom network buildout".format(client, p.APPLY_VCML_PREFIX)]
                
        ################## The waterfall
        client_mac_mapping.set_index(['CUSTOMER_ID', 'MEASUREMENT', 'CHAIN_SUBGROUP'], inplace=True)
        
        vcml_mac_mapping.set_index(['CUSTOMER_ID', 'MEASUREMENT', 'CHAIN_SUBGROUP'], inplace=True)
                    
        client_mac_mapping['CLIENT_MAC_LIST'].update(vcml_mac_mapping['CLIENT_MAC_LIST'])
        client_mac_mapping.reset_index(inplace=True)
        pref_filter = client_mac_mapping['CHAIN_SUBGROUP'].isin(pref_pharms)

        def mac_mapping_waterfall(cmm_df, vcml_in_df, wf, meas, chain_filt, pref_filt):
            waterfall = single_waterfall(wf, vcml_in_df)
            df_filter = (cmm_df['MEASUREMENT'] == meas) & (cmm_df['CHAIN_SUBGROUP'].isin(chain_filt))
            cmm_df.loc[pref_filt & df_filter & (cmm_df.CLIENT_MAC_LIST.isna()), 'CLIENT_MAC_LIST'] = waterfall

        ######## waterfall STEP #1 R30 BIG CHAIN waterfall
        # any missing big chain R30 VCML will : {4,5,6,7,8} -> 4 -> 1
        bigcap_list = list(set(np.concatenate([measurement_mapping[measurement_mapping['CHAIN_GROUP']==cg]['CHAIN_SUBGROUP'].unique() for cg in list(set(p.BIG_CAPPED_PHARMACY_LIST['GNRC']+p.BIG_CAPPED_PHARMACY_LIST['BRND']))])))
        mac_mapping_waterfall(client_mac_mapping, vcml_df, ['477', '4', '177', 'C4', '1'],
                              'R30', bigcap_list, 
                              pref_filter)
        mac_mapping_waterfall(client_mac_mapping, vcml_df, ['499', '4', '199', 'C4', '1'],
                              'R30', bigcap_list,
                              ~pref_filter)
        
        # any missing big chain R90 VCML will: {34,35,36,37,38} -> 9 -> 3 -> {4,5,6,7,8} -> 4 -> 1
        # we do CVSSP separately because a missing CVSSP R90 VCML will adjudicate as 34 if available--
        # but only CVSSP from the bigcap_list!
        if 'CVSSP' in bigcap_list:
            mac_mapping_waterfall(client_mac_mapping, vcml_df, ['34', '977', '9', '377', 'C9', '3', '477', '4', '177', 'C4', '1'],
                                  'R90', ['CVSSP'],
                                  pref_filter)
            mac_mapping_waterfall(client_mac_mapping, vcml_df, ['34', '999', '9', '377', 'C9', '3', '477', '4', '177', 'C4', '1'],
                                  'R90', ['CVSSP'],
                                  ~pref_filter)
            bigcap_list.remove("CVSSP")
            
        mac_mapping_waterfall(client_mac_mapping, vcml_df, ['977', '9', '377', 'C9', '3', '477', '4', '177', 'C4', '1'],
                              'R90', bigcap_list, 
                              pref_filter)
        mac_mapping_waterfall(client_mac_mapping, vcml_df, ['977', '9', '377', 'C9', '3', '477', '4', '177', 'C4', '1'],
                              'R90', bigcap_list,
                              ~pref_filter)

        ######## waterfall STEP #2 PSAO waterfall
        #R30 PSAOs : {PSAO_R30_vcmls} -> 1
        if len(list(set(p.PSAO_LIST['BRND']+p.PSAO_LIST['GNRC'])))>0:
            psao_list = list(set(np.concatenate([measurement_mapping[measurement_mapping['CHAIN_GROUP']==cg]['CHAIN_SUBGROUP'].unique() for cg in list(set(p.PSAO_LIST['GNRC']+p.PSAO_LIST['BRND']))])))
        else:
            psao_list = []
        mac_mapping_waterfall(client_mac_mapping, vcml_df, ['177', '1'],
                              'R30', psao_list, pref_filter)
        mac_mapping_waterfall(client_mac_mapping, vcml_df, ['199', '1'],
                              'R30', psao_list, ~pref_filter)
        
        #R90 PSAOs : 3 -> 1
        mac_mapping_waterfall(client_mac_mapping, vcml_df, ['377', '3', '177', '1'],
                              'R90', psao_list, pref_filter)
        mac_mapping_waterfall(client_mac_mapping, vcml_df, ['399', '3', '199', '1'],
                              'R90', psao_list, ~pref_filter)
        
        ######## waterfall STEP #3: p.SMALL_CAPPED_PHARMACY_LIST + p.NON_CAPPED_PHARMACY_LIST waterfall
        # remove any PSAOs if they were added and MCHOICE
        ALL_OTHER_PHARMACY_LIST = [item for item in (list(set(p.SMALL_CAPPED_PHARMACY_LIST['BRND'] + p.SMALL_CAPPED_PHARMACY_LIST['GNRC']
                                                              + p.NON_CAPPED_PHARMACY_LIST['BRND'] + p.NON_CAPPED_PHARMACY_LIST['GNRC']
                                                              + p.COGS_PHARMACY_LIST['BRND'] + p.COGS_PHARMACY_LIST['GNRC']))) if item not in (list(set(p.PSAO_LIST['BRND']+p.PSAO_LIST['GNRC'])) + MAIL_LIST)]
        ALL_OTHER_PHARMACY_LIST = list(set(np.concatenate([measurement_mapping[measurement_mapping['CHAIN_GROUP']==cg]['CHAIN_SUBGROUP'].unique() for cg in ALL_OTHER_PHARMACY_LIST])))

        #R30: 1
        mac_mapping_waterfall(client_mac_mapping, vcml_df, ['177','C4','1'],
                              'R30', ALL_OTHER_PHARMACY_LIST, pref_filter)
        mac_mapping_waterfall(client_mac_mapping, vcml_df, ['199','C4', '1'],
                              'R30', ALL_OTHER_PHARMACY_LIST, ~pref_filter)
    
        #R90 3 -> 1
        mac_mapping_waterfall(client_mac_mapping, vcml_df, ['377', 'C9', '3', '177', 'C4', '1'],
                              'R90', ALL_OTHER_PHARMACY_LIST, pref_filter)
        mac_mapping_waterfall(client_mac_mapping, vcml_df, ['399', 'C9', '3', '199', 'C4', '1'],
                              'R90', ALL_OTHER_PHARMACY_LIST, ~pref_filter)
        
        
           
        ######################## END of WATERFALL
        ######################## Eliminate unnecessary mac-mapping rows
        ''' 
        This section eliminates any row with no guarantee, no VCML in VCML-reference, AND no claims
        '''
        # filter out R90 if not in measurement mapping
        if not any(measurement_mapping['MEASUREMENT_CLEAN'].isin(['R90','R90P','R90p','R90N','R90n'])):
            client_mac_mapping = client_mac_mapping[client_mac_mapping['MEASUREMENT'].apply(lambda x:x[:3]) != 'R90']

        # filter out pref_oth if preferred pharmacy list is empty
        if pref_pharms == ['None']:
            client_mac_mapping = client_mac_mapping[client_mac_mapping['CHAIN_SUBGROUP'] != 'PREF_OTH']
        
        # filter out MAIL rows that have no CLIENT_VCML_ID, an artifact of our waterfall process for clients without MAIL
        if client_mac_mapping[client_mac_mapping['CHAIN_SUBGROUP']=='MAIL']['CLIENT_VCML_ID'].isna().all():
            WARNINGS = True
            WARNING_LIST = WARNING_LIST + ['\nWARNING: MAC mapping MAIL rows have no VCML_ID. Removing MAIL from MAC mapping.']
            client_mac_mapping = client_mac_mapping[client_mac_mapping['CHAIN_SUBGROUP'] != 'MAIL']
        
        # filter out R90OK rows when the MEASUREMENT is not R90
        client_mac_mapping = client_mac_mapping[~((client_mac_mapping['MEASUREMENT']!='R90') 
                                                  & (client_mac_mapping['CHAIN_SUBGROUP'].str.contains('R90OK')))]
       
        
        ########################## END of unnecessary row elimination
        
        # 199, 177, 1 may appear in the r90 mapping dictionary due to the np_90 and p_90 waterfalls 
        # and we handle OK separately
        r90_only_maclists = set(mapping_dictionary_90.values()) - set(['199', '177', 'C4' ,'1', 'OK'])
        if 'OK' in client_mac_mapping['CLIENT_MAC_LIST'].unique() and not vcml_df['CLIENT_MAC_LIST'].isin(r90_only_maclists).any():
            # In this situation, we have R90OK VCMLs but no actual R90 MAC lists or R90 claims -- so our usual waterfall 
            # produces wrong results.
            client_mac_mapping_r30 = client_mac_mapping[client_mac_mapping['MEASUREMENT']=='R30']
            client_mac_mapping_r90 = client_mac_mapping_r30.copy()
            client_mac_mapping_r90 = client_mac_mapping_r90[~client_mac_mapping_r90['CLIENT_MAC_LIST'].isin(['XA','XR'])]
            client_mac_mapping_r90['MEASUREMENT'] = 'R90'
            client_mac_mapping_r90ok = client_mac_mapping[client_mac_mapping['CLIENT_MAC_LIST']=='OK']
            client_mac_mapping_mail = client_mac_mapping[client_mac_mapping['CHAIN_SUBGROUP'].isin(MAIL_LIST)]
            client_mac_mapping = pd.concat([client_mac_mapping_r30, 
                                            client_mac_mapping_r90, 
                                            client_mac_mapping_r90ok, 
                                            client_mac_mapping_mail])
        if p.R90_AS_MAIL:
            # Rewrite any R90 list that has R30 claims as MAIL.
            # We don't want to overwrite pure R90 lists here, because otherwise they don't get priced
            # But we also don't want to make R30 pricing obey R90 rules when the claims aren't truly
            # adjudicating on those lists
            vcmls_with_r30 = client_mac_mapping.loc[client_mac_mapping['MEASUREMENT']=='R30', 'CLIENT_MAC_LIST'].unique()
            client_mac_mapping.loc[(client_mac_mapping['MEASUREMENT']=='R90') & (client_mac_mapping['CLIENT_MAC_LIST'].isin(vcmls_with_r30)),
                                   'CLIENT_MAC_LIST'] = '2' 

        # Add CUSTOMER_ID to VCML_ID:
        client_mac_mapping['CLIENT_VCML_ID'] = client_mac_mapping['CUSTOMER_ID'] + client_mac_mapping['CLIENT_MAC_LIST']
        #REGION MAPPING
        client_mac_mapping['REGION'] = client_mac_mapping['CUSTOMER_ID']

        # drop, reorder, and rename columns
        client_mac_mapping = client_mac_mapping[['CUSTOMER_ID','REGION','MEASUREMENT','CHAIN_SUBGROUP','CLIENT_VCML_ID']]\
                    .rename(columns={'CLIENT_VCML_ID':'MAC_LIST'})

        all_clients_mac_mapping = pd.concat([all_clients_mac_mapping,client_mac_mapping])
    #end of clients loop
    
    # Test: Check to see if there are claims in non-existing VCMLs or VCMLs without claims    
    cm2 = all_clients_mac_mapping.merge(measurement_mapping, left_on = ['CUSTOMER_ID','MEASUREMENT','CHAIN_SUBGROUP'],
                        right_on=['CLIENT','MEASUREMENT_CLEAN','CHAIN_SUBGROUP'],how = 'outer').drop(columns=['CLIENT','MEASUREMENT_CLEAN'])
    
    if any (cm2['FULLAWP'].isna()):
        WARNINGS = True
        WARNING_LIST = WARNING_LIST + ['\nWARNING: There are VCML-IDs with no claims, run qa_checks.py and check Measurement mapping report']

    if any (cm2['CUSTOMER_ID'].isna()):
        WARNINGS = True
        WARNING_LIST = WARNING_LIST + ['\nWARNING: There are claims that do not belong to any of the VCML_IDs, run qa_checks.py and check Measurement mapping report']

    # Test: make sure that the number of VCMLs match the number in ger_opt_vw_taxonomy: 
    if p.READ_FROM_BQ:
        client_taxonomy = uf.read_BQ_data(
            BQ.client_name_id_mapping_custom.format(_customer_id = uf.get_formatted_string(p.CUSTOMER_ID),
                                                    _project = p.BQ_INPUT_PROJECT_ID,
                                                    _dataset = p.BQ_INPUT_DATASET_DS_PRO_LP,
                                                    _table_id = uf.get_formatted_table_id("ger_opt_taxonomy_final") + p.WS_SUFFIX),
            project_id=p.BQ_INPUT_PROJECT_ID,
            dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
            table_id='client_name_id_mapping', #+ p.WS_SUFFIX, 
            custom = True
        )
    else:
        client_taxonomy = pd.read_csv(p.FILE_INPUT_PATH + p.CLIENT_NAME_MAPPING_FILE, dtype = p.VARIABLE_TYPE_DIC) 
    client_taxonomy = standardize_df(client_taxonomy)
    number_vcmls = all_clients_mac_mapping.groupby(by='CUSTOMER_ID')['MAC_LIST']\
                .nunique().reset_index().rename(columns={'MAC_LIST':'NUMBER_MAC_LISTS'})
    number_vcmls = number_vcmls.merge(client_taxonomy[['CUSTOMER_ID','NO_OF_VCMLS']],on='CUSTOMER_ID',how='inner')
    if not all(number_vcmls.NUMBER_MAC_LISTS == number_vcmls.NO_OF_VCMLS):
        WARNINGS = True
        WARNING_LIST = WARNING_LIST + ['\n* Warning: The number of unique VCMLs in mac-mapping does not mactch the number expected from CLIENT_TAXONOMY. ']
        
    # Test: Test that all vcml in the vcml_df are in mac_mapping_df
    if len(set(vcml_df['CLIENT_VCML_ID'].str[3:].unique()) - set(all_clients_mac_mapping['MAC_LIST'])) != 0:
        WARNINGS = True
        WARNING_LIST = WARNING_LIST + [f"\n* Warning: The number of unique VCMLs in mac-mapping does not mactch the number expected in the original VCML-reference. Check for a VCML_ID = {p.APPLY_VCML_PREFIX}xxxx88."]

    if p.NO_MAIL: 
        all_clients_mac_mapping = all_clients_mac_mapping[all_clients_mac_mapping['MEASUREMENT'] != 'M30']
    
    qa_dataframe(all_clients_mac_mapping,
                 duplist = ['CUSTOMER_ID','REGION','MEASUREMENT','CHAIN_SUBGROUP'],
                 dataset = p.MAC_MAPPING_FILE)
    
    if WARNINGS:
        for i in WARNING_LIST:
            print("{}".format(i))
    
    all_clients_mac_mapping.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_MAPPING_FILE, index=False)
    
    if p.WRITE_TO_BQ:
        uf.write_to_bq(
            all_clients_mac_mapping,
            project_output = p.BQ_OUTPUT_PROJECT_ID,
            dataset_output = p.BQ_OUTPUT_DATASET,
            table_id = "mac_mapping_subgroup",
            client_name_param = ', '.join(sorted(p.CUSTOMER_ID)),
            timestamp_param = p.TIMESTAMP,
            run_id=p.AT_RUN_ID,
            schema = None
        )

    
def prepare_mac_constraints():
    """ transform the mac-mapping into the format needed for building the LP constraints
    
    intput: p.CLIENT_GUARANTEE_FILE, p.MAC_MAPPING_FILE
    output: p.MAC_CONSTRAINT_FILE
    
    Takes the mac-mapping and pivots the pharmacies to columns and assigns integers to the pharmacies that share 
    prices (on the same VCML_ID). These price-sets will be used in the LP as constraints, forcing the prices to be equal.
    """
    import pandas as pd
    import numpy as np
    import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df, add_virtual_r90
    import BQ
    import util_funcs as uf
  
    client_guarantees = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.CLIENT_GUARANTEE_FILE, dtype = p.VARIABLE_TYPE_DIC)
    
    client_guarantees = standardize_df(client_guarantees)
    
    #Read in mac mapping file
    if p.CLIENT_TYPE == 'MEDD' and p.FULL_YEAR and not p.EGWP:
        #For MEDD new year pricing: read in custom mac mapping file with the new year vcml structure
        client_guarantees = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.CLIENT_GUARANTEE_FILE, dtype = p.VARIABLE_TYPE_DIC))
        mac_mapping = standardize_df(pd.read_csv(p.FILE_INPUT_PATH + p.CUSTOM_MAC_MAPPING_FILE, dtype = p.VARIABLE_TYPE_DIC)) 
        mac_mapping = mac_mapping[mac_mapping.CUSTOMER_ID == p.CUSTOMER_ID[0]]
    else:
        mac_mapping = standardize_df(pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_MAPPING_FILE, dtype = p.VARIABLE_TYPE_DIC))
    
    # Set up the rows for mac constraint file
    client_groups = client_guarantees[['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT']].drop_duplicates()
    client_groups = client_groups.merge(mac_mapping, how = 'left', on = ['REGION', 'MEASUREMENT'])

    client_list = client_groups['CLIENT'].unique()
    mac_constraints_all = pd.DataFrame()
    for client in client_list:
        # Chain Group for columns
        chains = client_groups.loc[client_groups['CLIENT'] == client, 'CHAIN_SUBGROUP'].unique()
        
        measurement_list = list(client_groups.loc[client_groups['CLIENT'] == client, 'MEASUREMENT'].unique())
        mac_list = list(client_groups.loc[client_groups['CLIENT'] == client, 'MAC_LIST'].unique())
        r90_mac_list = list(client_groups.loc[(client_groups['CLIENT'] == client) & (client_groups["MEASUREMENT"]=='R90'), 'MAC_LIST'].unique())

        # Mac constraints df with rows and columns
        client_single = client_groups.loc[client_groups['CLIENT'] == client, ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT']].drop_duplicates()
        mac_constraints = client_single.reindex(columns=[*client_single.columns.tolist(), *chains], fill_value=0)

        # Prepare list of constraints to be checked, either mac_list or measurement_list based on the client.
        constraint_list = measurement_list if p.UNIFORM_MAC_PRICING else mac_list
        # Loop through mac/measurement lists to check equals and set constraints
        cons = 0 #initual constant is 0
        
        for constraint in constraint_list:
            if p.UNIFORM_MAC_PRICING:
                mapping_chain = mac_mapping.loc[mac_mapping.MEASUREMENT == constraint]
            else:
                mapping_chain = mac_mapping.loc[mac_mapping.MAC_LIST == constraint]
            if p.R90_AS_MAIL and (mapping_chain.MEASUREMENT.str[0]=='M').any():
                mapping_chain = pd.concat([mapping_chain, mac_mapping.loc[mac_mapping.MEASUREMENT == 'R90']]).drop_duplicates().reset_index()
            elif p.R90_AS_MAIL \
                and ((constraint in r90_mac_list) or (constraint == 'R90')):
                continue
            
            row_count = len(mapping_chain.index)
            if row_count > 1:
                cons = cons + 1
                for index, row in mapping_chain.iterrows():
                    mac_constraints.loc[(mac_constraints.REGION == row['REGION']) &
                                        (mac_constraints.MEASUREMENT == row['MEASUREMENT']), row['CHAIN_SUBGROUP']]= cons
        mac_constraints_all = mac_constraints_all.append(mac_constraints)
    # Write mac constraint file to csv
    mac_constraints_all.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_CONSTRAINT_FILE)


def prepare_generic_launches_file():
    """ Load Generic Launch file from CSV and transform into format expected by LP
    
    This function no longer works beyond producing zeroed out generic launch files, and requires re-establishing the the base tables
    behind the gen_launch_all table (formally owned by Adil), as well as updating it to be compatible with the CHAIN_SUBGROUP column 
    in combined_measurement_mapping.
    
    intput: p.RAW_GEN_LAUNCH_FILE, p.MEASUREMENT_MAPPING
    output: p.GENERIC_LAUNCH_FILE
    1. fix generic-launch file in three ways:
        1.1. fix Rx chain groups to match claims chain groups
        1.2. if no R90 combine R30 and R90 in generic launch file 
        1.3. split 'IND' into 'NONPREF_OTH' and 'PREF_OTH' proportionally by pct of AWP in claims
    2. use measurement mapping from claims to identify all combinations of {client,breakout,preferred,measurement,chain-groups}
    3. add a row for each month for each combinations of {client,breakout,preferred,measurement,chain-groups} in measurement mapping
    4. join to generic-launches and impute zeros 
    
    output file is used in ClientPharmacyMacOptimization.py when calculating performances
    """
    import pandas as pd
    import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df
    from qa_checks import qa_dataframe
    import BQ
    import util_funcs as uf
    
    # read in measurement mapping and remove extra rows
    if p.READ_FROM_BQ:
        measurement_mapping = uf.read_BQ_data(
                BQ.ger_opt_msrmnt_map.format(_customer_id=uf.get_formatted_string(p.CUSTOMER_ID)),
                project_id=p.BQ_INPUT_PROJECT_ID,
                dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                table_id="combined_measurement_mapping" + p.WS_SUFFIX + p.CCP_SUFFIX,
                customer=', '.join(sorted(p.CUSTOMER_ID)))
    else:
        measurement_mapping = pd.read_csv(p.FILE_INPUT_PATH + p.MEASUREMENT_MAPPING, dtype = p.VARIABLE_TYPE_DIC)
    measurement_mapping = standardize_df(measurement_mapping)
    measurement_mapping = measurement_mapping.groupby(['CLIENT','BREAKOUT','REGION','PREFERRED', \
                                                      'MEASUREMENT','BG_FLAG','CHAIN_GROUP','CHAIN_SUBGROUP','GUARANTEE_CATEGORY'])['FULLAWP']\
                                                        .sum() \
                                                        .reset_index() \
                                                        .rename(columns={'FULLAWP':'CLAIMS_AWP'})
    # Add all months to measurement mapping
    measurement_mapping['key'] = 1
    # add all months to measurement_mapping -> generic_launch_full
#TODO: move the range(1,13) to a date parameter
    months_df = pd.DataFrame({'MONTH':list(range(1,13)),'key':1})
    measurementMapping_all_months = measurement_mapping.merge(months_df,on='key',how='outer')

    # until further notice gen_launch is going to be set at zero
    # set this to False for zeros:
    # set to true to use the gen_launch data
    USE_GENERIC_LAUNCH_DATA = False
    
    if not USE_GENERIC_LAUNCH_DATA:
        generic_launch_full = measurementMapping_all_months.copy(deep=True)
        generic_launch_full['QTY'] = 0
        generic_launch_full['FULLAWP'] = 0.0
        generic_launch_full['ING_COST'] = 0.0
        
    elif USE_GENERIC_LAUNCH_DATA:
        if p.READ_FROM_BQ:
            generic_launches = uf.read_BQ_data(
                BQ.gen_launch_all,
                project_id=p.BQ_INPUT_PROJECT_ID,
                dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                table_id='gen_launch_all' + p.WS_SUFFIX,
                customer = ', '.join(sorted(p.CUSTOMER_ID))
            )
        else:
            generic_launches = pd.read_csv(p.FILE_INPUT_PATH + p.RAW_GEN_LAUNCH_FILE, dtype = p.VARIABLE_TYPE_DIC)
        generic_launches = standardize_df(generic_launches)
        # Fix RX naming convensions
        generic_launches.replace(['AH', 'ALB', 'CDL', 'ELV', 'KD'], ['HMA', 'ABS', 'CAR', 'ELE', 'KIN'], inplace=True)
    
        total_generic_awp = generic_launches['FULLAWP'].sum()
        orginal_generic_launches = generic_launches.copy(deep=True)
        # fix R90 and R30 and IND chain group in generic_launches
        clients = measurement_mapping['CLIENT'].unique()
        for client in clients:
            # if the client does not have R90, combine R30 and R90 in generic launch forecasts 
#TODO: is this CVS only?
            if not 'R90' in measurement_mapping[measurement_mapping['CLIENT'] == client]['MEASUREMENT'].unique():
                generic_launches.loc[(generic_launches['CUSTOMER_ID'] == client) & (generic_launches['MEASUREMENT'] == 'R90'),'MEASUREMENT'] = 'R30'
                generic_launches = generic_launches.groupby(['CUSTOMER_ID','MEASUREMENT','CHAIN_GROUP','MONTH'])[['QTY','FULLAWP','ING_COST']] \
                                .sum().reset_index()
                                
            # Assign IND to combination of NONPREF_OTH and PREF_OTH
            chain_groups = measurement_mapping[(measurement_mapping['CLIENT'] == client)]['CHAIN_GROUP'].unique()
            if ('PREF_OTH' in chain_groups) and ('NONPREF_OTH' in chain_groups):
                # proportional allocation
                independents = measurement_mapping[(measurement_mapping['CLIENT'] == client) & (measurement_mapping['CHAIN_GROUP'].isin(['PREF_OTH','NONPREF_OTH']))
                    & (measurement_mapping['MEASUREMENT'].isin(['R30','R90']))]
                total_IND_AWP = independents['CLAIMS_AWP'].sum()
                independents['PROPORTION_AWP'] = independents['CLAIMS_AWP']/total_IND_AWP
                for v in ['PREF_OTH','NONPREF_OTH']:
                    generic_ind = generic_launches[(generic_launches['CUSTOMER_ID'] == client) & (generic_launches['CHAIN_GROUP'] == 'IND')]
                    generic_ind['CHAIN_GROUP'] = v
                    proportion_awp = independents.loc[(independents['CHAIN_GROUP'] == v),'PROPORTION_AWP'].values
                    generic_ind['QTY'] = generic_ind['QTY'] * proportion_awp
                    generic_ind['FULLAWP'] = generic_ind['FULLAWP'] * proportion_awp
                    generic_ind['ING_COST'] = generic_ind['ING_COST'] * proportion_awp
                    # add the row to generic_launches dataset
                    generic_launches = pd.concat([generic_launches,generic_ind])
                    # drop the old IND row
                    dropIndex = generic_launches[(generic_launches['CUSTOMER_ID'] == client) & (generic_launches['CHAIN_GROUP'] == 'IND')].index
                    generic_launches.drop(dropIndex,inplace=True)
                    
            elif ('NONPREF_OTH' in chain_groups):
                generic_launches.loc[(generic_launches['CUSTOMER_ID'] == client) & (generic_launches['CHAIN_GROUP'] == 'IND'),'CHAIN_GROUP'] = 'NONPREF_OTH'
            elif ('PREF_OTH' in chain_groups):
                generic_launches.loc[(generic_launches['CUSTOMER_ID'] == client) & (generic_launches['CHAIN_GROUP'] == 'IND'),'CHAIN_GROUP'] = 'PREF_OTH'
            # end of for:loop
        # make sure nothing broke
        assert abs(generic_launches['FULLAWP'].sum() - total_generic_awp ) < .001,\
            "abs(generic_launches['FULLAWP'].sum() - total_generic_awp ) < .001"
    
        # merge generic_launches to measurement_mapping and fill in zeros for blanks
        generic_launches.loc[(generic_launches['MEASUREMENT'] == 'M30') & (generic_launches['CHAIN_GROUP'] == 'NONPREF_OTH'),'CHAIN_GROUP'] = 'MAIL'

        # join full measurement mappping to generic_launches
        generic_launch_full = measurementMapping_all_months.merge(generic_launches[['CUSTOMER_ID','MEASUREMENT','CHAIN_GROUP','MONTH','QTY','FULLAWP','ING_COST']],
                                                    left_on=['CLIENT','MEASUREMENT','CHAIN_GROUP','MONTH']
                                                    ,right_on=['CUSTOMER_ID','MEASUREMENT','CHAIN_GROUP','MONTH']
                                                    ,how='left').fillna(0)

        assert abs(total_generic_awp - generic_launch_full['FULLAWP'].sum()) < 0.001, \
            "abs(total_generic_awp - generic_launch_full['FULLAWP'].sum()) < 0.001"
    
        # for debugging
        # clients1 = generic_launches.groupby(['CUSTOMER_ID','CHAIN_GROUP','MEASUREMENT'])['FULLAWP'].sum().reset_index()
        # clients2 = generic_launch_full.groupby(['CLIENT','CHAIN_GROUP','MEASUREMENT_CLEAN'])['FULLAWP'].sum().reset_index()
    
    # get into same format as expected:  
    generic_launch_full = generic_launch_full[['CLIENT','BREAKOUT','REGION','MEASUREMENT','BG_FLAG','CHAIN_GROUP','CHAIN_SUBGROUP','MONTH','QTY','FULLAWP','ING_COST']]
    
    qa_dataframe(generic_launch_full,
                duplist = ['CLIENT','BREAKOUT','REGION','MEASUREMENT','BG_FLAG','CHAIN_GROUP','CHAIN_SUBGROUP','MONTH'],
                dataset = p.GENERIC_LAUNCH_FILE)
    
    # The YTD portion of the Generic Launch file is zeroed out, becuase the assumption is those should already be reflected in claims data. 
    # It is important to do that correction in preprocess step, since in simulation, during iterations YTD period changes and we need to assume some part of the generic launch is YTD and we have to include this ytd value to our performance calculations. 
    #Basically, in CPMO_Optimization code - generic launch ytd is included in the performance. In the preprocessing we modify the input file so that we do not double count the generic launch during ytd. 
    from calendar import monthrange
    LAST_DATA_month_len = monthrange(p.LAST_DATA.year, p.LAST_DATA.month)[1]
    generic_launch_full.loc[generic_launch_full['MONTH']<p.LAST_DATA.month, ['QTY','FULLAWP','ING_COST']] = 0 
    temp_last_month_genlaunch = generic_launch_full.loc[generic_launch_full['MONTH']==p.LAST_DATA.month][['QTY','FULLAWP','ING_COST']]
    generic_launch_full.loc[generic_launch_full['MONTH']==p.LAST_DATA.month, ['QTY','FULLAWP','ING_COST']] =  ((LAST_DATA_month_len - p.LAST_DATA.day) / LAST_DATA_month_len) * temp_last_month_genlaunch

    assert (generic_launch_full[['QTY', 'FULLAWP', 'ING_COST']] == 0).all().all(), "Generic launch not supported at this time. Values must be 0"
    generic_launch_full.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.GENERIC_LAUNCH_FILE, index=False)
 
 
def prepare_price_overrides_file():
    import pandas as pd
    import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df
    from qa_checks import qa_dataframe
    import numpy as np
    import BQ
    import util_funcs as uf
    
    if p.CLIENT_LOB=='CMK':
        if p.READ_FROM_BQ:
            mac_price_overrides = uf.read_BQ_data(
                BQ.ger_opt_mac_price_override_custom.format(_customer_id = uf.get_formatted_string(p.CUSTOMER_ID),
                                                            _project = p.BQ_INPUT_PROJECT_ID,
                                                            _landing_dataset = p.BQ_INPUT_DATASET_DS_PRO_LP,
                                                            _table_id = "ger_opt_mac_price_override" + p.WS_SUFFIX),
                project_id=p.BQ_INPUT_PROJECT_ID,
                dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                table_id="ger_opt_mac_price_override" + p.WS_SUFFIX,
                custom = True
            )
        else:
            mac_price_overrides = pd.read_csv(p.FILE_INPUT_PATH + p.RAW_MAC_PRICE_OVERRIDES, dtype = p.VARIABLE_TYPE_DIC)
        mac_price_overrides = standardize_df(mac_price_overrides)
        mac_price_overrides = mac_price_overrides.drop_duplicates(subset = ['CLIENT', 'VCML_ID', 'GPI','NDC','BG_FLAG'])
        mac_price_overrides['VCML_ID'] = mac_price_overrides['VCML_ID'].copy()
        mac_price_overrides['CLIENT'] = uf.get_formatted_client_name(p.CUSTOMER_ID)
        
        assert len(mac_price_overrides[mac_price_overrides['GPI'].str.len()!=14]) == 0, "Incorrectly formatted GPIs in mac_price_overrides"
        
        if len(mac_price_overrides) > 0 or not p.TRUECOST_CLIENT:
            qa_dataframe(mac_price_overrides, duplist = ['CLIENT', 'VCML_ID', 'GPI','NDC','BG_FLAG'], dataset = p.MAC_PRICE_OVERRIDE_FILE)
        else:
            print("WARNING: No overrides for TrueCost client")

        mac_price_overrides.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_PRICE_OVERRIDE_FILE, index=False)

        if p.READ_FROM_BQ:
            wmt_price_overrides = uf.read_BQ_data(
                BQ.wmt_unc_override_custom.format(_customer_id = uf.get_formatted_string(p.CUSTOMER_ID),
                                                  _project = "pbm-mac-lp-prod-ai",
                                                  _dataset = p.BQ_INPUT_DATASET_SANDBOX,
                                                  _table = "wmt_unc_override"),
                project_id =  "pbm-mac-lp-prod-ai",
                dataset_id = p.BQ_INPUT_DATASET_SANDBOX,
                table_id = "wmt_unc_override",
                custom = True
            )
        else:
            wmt_price_overrides = pd.read_csv(p.FILE_INPUT_PATH + p.RAW_WMT_UNC_PRICE_OVERRIDES, dtype = p.VARIABLE_TYPE_DIC)

        wmt_price_overrides = standardize_df(wmt_price_overrides)
        wmt_price_overrides = wmt_price_overrides.drop_duplicates(subset = ['CLIENT', 'VCML_ID', 'GPI','NDC','BG_FLAG'])
        wmt_price_overrides['CLIENT'] = uf.get_formatted_client_name(p.CUSTOMER_ID)

        if p.CLIENT_NAME_TABLEAU.startswith(('WTW','AON')):
            begin_m_q_prices_df = uf.read_BQ_data(
                BQ.begin_m_q_prices_custom.format(_macs = list(wmt_price_overrides.VCML_ID.unique()),
                                                  _project = p.BQ_INPUT_PROJECT_ID,
                                                  _landing_dataset = p.BQ_INPUT_DATASET_ENT_ENRV_PROD,
                                                  _table = "gms_v_drug_mac_hist",
                                                  _go_live = p.GO_LIVE.date()),
                project_id = p.BQ_INPUT_PROJECT_ID,
                dataset_id = p.BQ_INPUT_DATASET_ENT_ENRV_PROD,
                table_id = "gms_v_drug_mac_hist",
                custom = True)

            begin_m_q_prices_df = standardize_df(begin_m_q_prices_df)
            wmt_price_overrides = pd.merge(wmt_price_overrides, begin_m_q_prices_df, how = 'left', on = ['VCML_ID', 'GPI', 'NDC','GPI_NDC','BG_FLAG'])

        wmt_price_overrides['MAC'] = wmt_price_overrides['VCML_ID'].copy()
        wmt_price_overrides.loc[:,"UNC_OVRD_AMT"] = wmt_price_overrides.loc[:,"MAC_PRICE"]

        if p.CLIENT_NAME_TABLEAU.startswith('WTW'):
            wmt_price_overrides = wmt_price_overrides.dropna(subset=['BEG_Q_PRICE'])
            wmt_price_overrides.loc[:,"UNC_OVRD_AMT"] =  round((wmt_price_overrides['BEG_Q_PRICE']*1.24).clip(wmt_price_overrides.MAC_PRICE, wmt_price_overrides.UNC_PRICE),4)

        if p.CLIENT_NAME_TABLEAU.startswith('AON'):
            wmt_price_overrides = wmt_price_overrides.dropna(subset=['BEG_M_PRICE'])
            wmt_price_overrides.loc[:,"UNC_OVRD_AMT"] =  round((wmt_price_overrides['BEG_M_PRICE']*1.24).clip(wmt_price_overrides.MAC_PRICE, wmt_price_overrides.UNC_PRICE),4)

        wmt_price_overrides = wmt_price_overrides[['CLIENT', 'REGION', 'MAC', 'GPI','NDC','BG_FLAG','UNC_OVRD_AMT','DATE_ADDED', 'SOURCE','REASON','GPI_NDC']]

        wmt_price_overrides.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.WMT_UNC_PRICE_OVERRIDE_FILE, index=False)
    else:
        #Create empty files for Aetna 
        mac_price_overrides = pd.DataFrame(columns = ['CLIENT','REGION','VCML_ID','GPI','NDC','BG_FLAG','PRICE_OVRD_AMT','DATE_ADDED','SOURCE','REASON','GPI_NDC'])
        mac_price_overrides.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_PRICE_OVERRIDE_FILE, index=False)

        wmt_price_overrides = pd.DataFrame(columns = ['CLIENT', 'REGION', 'MAC', 'GPI','NDC','BG_FLAG','UNC_OVRD_AMT','DATE_ADDED', 'SOURCE','REASON','GPI_NDC'])
        wmt_price_overrides.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.WMT_UNC_PRICE_OVERRIDE_FILE, index=False)

def prepare_mac_1026(): 
    """ Impute the GPI-level 1026 price for any GPI without a 1026 for NDC='***********'
    
    intput: p.MAC1026_FILE
    output: p.MAC1026_FILE
    """
    import pandas as pd
    import numpy as np
    import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df
    from qa_checks import qa_dataframe
    import BQ
    import util_funcs as uf
       
    # MAC1026
    if p.READ_FROM_BQ:
        mac1026 = uf.read_BQ_data(
            BQ.mac_1026,
            project_id=p.BQ_INPUT_PROJECT_ID,
            dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
            table_id="mac_1026" + p.WS_SUFFIX
        )
    else:
        mac1026 = pd.read_csv(p.FILE_INPUT_PATH + p.RAW_MAC1026_FILE, dtype = p.VARIABLE_TYPE_DIC)
    mac1026 = standardize_df(mac1026)
    mac1026['GPI_PRICE'] = mac1026['NDC'] == '***********'    # create an indicator for a GPI price
    all_GPI = mac1026.groupby(['GPI','BG_FLAG']).agg({'GPI_PRICE':'sum','PRICE':'min'}).reset_index().sort_values(['GPI_PRICE','GPI','BG_FLAG']) # roll up to GPI including GPI-price indicator 
    GPI_only = all_GPI[all_GPI['GPI_PRICE'] == 0].copy(deep=True)
    GPI_only['NDC'] = '***********'
    GPI_only['GPI_NDC'] = GPI_only['GPI'] + '_' + GPI_only['NDC']
    GPI_only['MAC_LIST'] = 'MAC1026'
    mac1026_out = pd.concat([mac1026,GPI_only]).drop(columns='GPI_PRICE')

    # NAs in EFF_DATE caused by imputation above, do not include in NAN check
    qa_dataframe(mac1026_out,
                 duplist = ['GPI_NDC','BG_FLAG'],
                 nanlist=['GPI','GPI_NDC','MAC_LIST','NDC','PRICE'] ,
                 dataset = p.MAC1026_FILE)
    
    mac1026_out.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC1026_FILE, index=False)    


def prepare_gpi_change_exclusions():
    """ add client and region to gpi_change_exclusions file
    
    intput: p.SPECIALTY_EXCLUSION_FILE
    output: p.SPECIALTY_EXCLUSION_FILE 
    
    Adds CLIENT and REGION to gpi_change_exclusions file
    """
    import pandas as pd
    import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df, read_tru_mac_list_prices
    from qa_checks import qa_dataframe
    import BQ
    import util_funcs as uf
    
    if p.READ_FROM_BQ:
        gpi_exclusions = uf.read_BQ_data(
            BQ.gpi_change_exclusion_ndc,
            project_id=p.BQ_INPUT_PROJECT_ID,
            dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
            table_id='gpi_change_exclusion_ndc' + p.WS_SUFFIX
        )
    else:
        gpi_exclusions = pd.read_csv(p.FILE_INPUT_PATH + p.RAW_SPECIALTY_EXCLUSION_FILE, dtype = p.VARIABLE_TYPE_DIC)
    #Every client, regardless of LOB, must have specific exclusions and should never be left empty
    qa_dataframe(gpi_exclusions, dataset = p.RAW_SPECIALTY_EXCLUSION_FILE) 
    
    gpi_exclusions = standardize_df(gpi_exclusions)
    gpi_exclusions['CLIENT'] = 'ALL'
    gpi_exclusions['REGION'] = 'ALL'

    # this is annoying but because writing over same csv file, in case we run this twice
    gpi_exclusions.rename(columns={'GPI_CD':'GPI'},inplace=True)
    gpi_exclusions = gpi_exclusions[['CLIENT','REGION','GPI','BG_FLAG']]
        
    if p.RAW_MONY_EXCLUSION_FILE:
        mony_exclusions = pd.read_csv(p.FILE_INPUT_PATH + p.RAW_MONY_EXCLUSION_FILE, dtype=p.VARIABLE_TYPE_DIC)
        mony_exclusions = standardize_df(mony_exclusions)
        if 'GPI_CD' in mony_exclusions.columns:
            mony_exclusions.rename(columns={'GPI_CD': 'GPI'}, inplace=True)
        mony_exclusions['CLIENT'] = 'ALL'
        mony_exclusions['REGION'] = 'ALL'
        mony_exclusions = mony_exclusions[['CLIENT', 'REGION', 'GPI','BG_FLAG']]
        gpi_exclusions = pd.concat([gpi_exclusions, mony_exclusions])
        
    if p.TRUECOST_CLIENT:
        tc_exclusions = uf.read_BQ_data(
             BQ.truecost_specialty_exclusion_custom.format(_customer_id = uf.get_formatted_string(p.CUSTOMER_ID),
                                          _contract_eff_date = p.GO_LIVE.date()),
             project_id = p.BQ_INPUT_PROJECT_ID,
             dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
             table_id = 'gms_truecost_gpi_14_drug_list',
             customer = ', '.join(sorted(p.CUSTOMER_ID)),
            custom=True)
        tc_exclusions = standardize_df(tc_exclusions)
        tc_exclusions['CLIENT'] = uf.get_formatted_client_name(p.CUSTOMER_ID)
        tc_exclusions['REGION'] = 'ALL'
        tc_exclusions = tc_exclusions[['CLIENT', 'REGION', 'GPI','BG_FLAG']]
        gpi_exclusions = pd.concat([gpi_exclusions, tc_exclusions])
    
    ## Making BOB Specialty, MONY Specialty and TC Specialty at GPI level
    gpi_exclusions = gpi_exclusions[['CLIENT','REGION','GPI','BG_FLAG']]
    gpi_exclusions_brnd = gpi_exclusions.copy()
    gpi_exclusions_brnd['BG_FLAG'] = 'B'
    gpi_exclusions = pd.concat([gpi_exclusions, gpi_exclusions_brnd])
    
    ##For MAC exclusions, we want to keep the at BG_FLAG level instead of GPI level
    ##Because whether a drug is at GPI level or NDC level pricing can distinguish between brand and generic
    if p.READ_FROM_BQ:
        if (p.CLIENT_TYPE == "COMMERCIAL" or p.CLIENT_TYPE == "MEDICAID"):
            if p.TRUECOST_CLIENT or p.UCL_CLIENT:
                """Read TRUECOST mac list from only Big Query"""
                print('Reading in TRUECOST mac list.........')
                assert p.READ_FROM_BQ == True, "Use p.READ_FROM_BQ=True to read table"
                mac_list_prices = read_tru_mac_list_prices()
                mac_list_prices = mac_list_prices[mac_list_prices['PRICE'] > 0]
            else:
                if p.READ_FROM_BQ:
                    mac_list_prices = uf.read_BQ_data(
                        BQ.mac_list,
                        project_id=p.BQ_INPUT_PROJECT_ID,
                        dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                        table_id="mac_list" + p.WS_SUFFIX,
                        mac=True,
                        vcml_ref_table_id='vcml_reference',
                        customer = ', '.join(sorted(p.CUSTOMER_ID))
                    )
        elif p.CLIENT_TYPE == "MEDD":
            #This was added to update vcml_ref table as per DE, later changes can be moved directly to CPMO Params
            #Updated _landing_dataset and _table_id_vcml to accomodate new table
            mac_list_prices = uf.read_BQ_data(
                BQ.mac_list_for_medd_custom.format(_customer_id = uf.get_formatted_string(p.CUSTOMER_ID),
                                                   _project=p.BQ_INPUT_PROJECT_ID,
                                                   _landing_dataset= p.BQ_INPUT_DATASET_ENT_CNFV_PROD,
                                                   _landing_dataset_vcml= p.BQ_INPUT_DATASET_ENT_ENRV_PROD,
                                                   _table_id_base= 'gms_ger_opt_base_mac_lists',
                                                   _table_id_vcml= 'v_cmk_vcml_reference'),
                project_id = p.BQ_INPUT_PROJECT_ID,
                dataset_id = p.BQ_INPUT_DATASET_ENT_CNFV_PROD,
                table_id = 'gms_ger_opt_base_mac_lists',
                custom = True
            )
        else:
            assert False, "CLIENT_TYPE is not of type COMMERCIAL or MEDD or MEDICAID."
    else:
        mac_list_prices = pd.read_csv(p.FILE_INPUT_PATH + p.MAC_LIST_FILE, dtype = p.VARIABLE_TYPE_DIC)
    mac_list_prices = standardize_df(mac_list_prices)
    mac_list_prices = mac_list_prices[mac_list_prices['NDC'] != '***********'][['MAC', 'GPI','BG_FLAG']]
    
    if (p.CLIENT_TYPE == "COMMERCIAL" or p.CLIENT_TYPE == "MEDICAID"):
        mac_list_prices['MAC'] = uf.get_formatted_client_name(p.CUSTOMER_ID)
    elif p.CLIENT_TYPE == "MEDD":
        mac_list_prices['MAC'] = uf.get_formatted_client_name(p.CUSTOMER_ID)
    else:
        assert False, "CLIENT_TYPE is not of type COMMERCIAL or MEDD or MEDICAID."
    mac_list_prices = mac_list_prices.drop_duplicates()
    mac_list_prices = mac_list_prices.rename(columns = {'MAC': 'CLIENT'})
    mac_list_prices['REGION'] = 'ALL'
    # mac_list_prices['REASON'] = 'GPI excuded since it did not have a * NDC'
    
    gpi_exclusions = pd.concat([gpi_exclusions, mac_list_prices])
    
    gpi_exclusions = gpi_exclusions.drop_duplicates()
    
    qa_dataframe(gpi_exclusions, dataset = p.SPECIALTY_EXCLUSION_FILE)
    
    # TrueCost: check that no overrides are being applied to specialty drugs
    if p.TRUECOST_CLIENT:
        mac_price_overrides = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.MAC_PRICE_OVERRIDE_FILE)
        mac_price_overrides = mac_price_overrides.merge(gpi_exclusions.drop_duplicates(subset=['GPI','BG_FLAG']), on = ['GPI', 'BG_FLAG'], how='inner', validate='m:1')
        assert len(mac_price_overrides) == 0, "Specialty overrides are being applied to TrueCost clients"
    
    gpi_exclusions.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.SPECIALTY_EXCLUSION_FILE, index=False)     
    
def prepare_OC_pharm_pref_file():
    ''' create the OC_pharm_pref_file for COMMERCIAL clients
    '''
    import os
    import pandas as pd
    import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df
    if p.CLIENT_TYPE == "MEDD":
        try:
            oc_pharm_pref = pd.read_csv(os.path.join(p.FILE_INPUT_PATH, p.OC_PHARM_PERF_FILE))
            assert oc_pharm_pref.shape[0] == len(p.AGREEMENT_PHARMACY_LIST['BRND']+p.AGREEMENT_PHARMACY_LIST['GNRC']), "oc_pharm_pref.shape[0] == len(p.AGREEMENT_PHARMACY_LIST)"
            oc_pharm_pref.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.OC_PHARM_PERF_FILE, index = False)
        except FileNotFoundError as err:
            print("WARNING: Automatically created {} file with zeros because input file was not found.".format(p.OC_PHARM_PERF_FILE))
            combined_list = p.AGREEMENT_PHARMACY_LIST['BRND'] + p.AGREEMENT_PHARMACY_LIST['GNRC']
            oc_pharm_pref = pd.DataFrame({
                'CHAIN_GROUP': combined_list,
                'BG_FLAG': ['B'] * len(p.AGREEMENT_PHARMACY_LIST['BRND']) + ['G'] * len(p.AGREEMENT_PHARMACY_LIST['GNRC']),
                'SURPLUS': [0] * len(combined_list)
            })
            oc_pharm_pref.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.OC_PHARM_PERF_FILE, index = False)
        except AssertionError as err:
            err_text = "ERROR: input file {} did not have entries for all pharmacies in AGREEMENT_PHARMACY_LIST".format(p.OC_PHARM_PERF_FILE)
            print(err_text)
            err.args += (err_text,)
            raise err
    elif (p.CLIENT_TYPE == "COMMERCIAL" or p.CLIENT_TYPE == "MEDICAID"):
        combined_list = p.AGREEMENT_PHARMACY_LIST['BRND'] + p.AGREEMENT_PHARMACY_LIST['GNRC']
        oc_pharm_pref = pd.DataFrame({
                'CHAIN_GROUP': combined_list,
                'BG_FLAG': ['B'] * len(p.AGREEMENT_PHARMACY_LIST['BRND']) + ['G'] * len(p.AGREEMENT_PHARMACY_LIST['GNRC']),
                'SURPLUS': [0] * len(combined_list)
            })
        oc_pharm_pref.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.OC_PHARM_PERF_FILE, index = False)
        print("WARNING: Automatically created {} file with zeros for commercial or MEDICAID clients.".format(p.OC_PHARM_PERF_FILE))
    else:
        assert False, "CLIENT_TYPE is not of type COMMERCIAL or MEDD or MEDICAID."


    '''
    #old commercial logic
    import pandas as pd
    import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df
    oc_pharm_pref = pd.DataFrame({'CHAIN_GROUP':p.AGREEMENT_PHARMACY_LIST,'SURPLUS':[0] * len(p.AGREEMENT_PHARMACY_LIST)})
    oc_pharm_pref.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.OC_PHARM_PERF_FILE,index=False)
    if p.CLIENT_TYPE == 'MEDD':
        print("WARNING: Automatically created {} file with zeros.".format(p.OC_PHARM_PERF_FILE))
    '''

def prepare_drug_mac_history():
    import os
    import pandas as pd
    import util_funcs as uf
    import BQ
    from CPMO_shared_functions import standardize_df
    '''
    '''
    # Create the drug_mac_history file for clients    
    if p.READ_FROM_BQ:
        assert p.READ_FROM_BQ, "gms_v_drug_mac_hist should be read from BQ"
        for client in p.CUSTOMER_ID:
            drug_mac_hist  = uf.read_BQ_data(
                    project_id=p.BQ_INPUT_PROJECT_ID,
                    dataset_id=p.BQ_INPUT_DATASET_ENT_ENRV_PROD,
                    table_id='gms_v_drug_mac_hist',
                    query=BQ.WTW_AON_beg_q_m_prices_by_cust_id_custom.format(_customer_id=uf.get_formatted_string(p.CUSTOMER_ID),
                                                                                _customer_id_len=len(client),
                                                                                _go_live_date=p.GO_LIVE.date(),
                                                                                _project = p.BQ_INPUT_PROJECT_ID,
                                                                                _landing_dataset = p.BQ_INPUT_DATASET_ENT_ENRV_PROD,
                                                                                _table = "gms_v_drug_mac_hist"),
                    custom=True
                )
                
    drug_mac_hist = standardize_df(drug_mac_hist)
            
    drug_mac_hist[['GPI', 'MAC_LIST', 'BG_FLAG', 'BEG_Q_PRICE', 'BEG_M_PRICE']].to_csv(os.path.join(p.FILE_DYNAMIC_INPUT_PATH, p.DRUG_MAC_HIST_FILE), index=False)

def prepare_non_mac_rate():
    ''' Prepare non-mac-rate data for all clients.
        Input:
            vcml reference
            non-mac-rate table
        Output:
            non mac rate for all VCMLs in the data
    '''
    import pandas as pd
    import CPMO_parameters as p   
    from CPMO_shared_functions import standardize_df
    import BQ
    import util_funcs as uf
    
    if p.READ_FROM_BQ:
    
        nmr_data = uf.read_BQ_data(
                                   BQ.non_mac_rate,
                                   project_id=p.BQ_INPUT_PROJECT_ID,
                                   dataset_id=p.BQ_INPUT_DATASET_ENT_CNFV_PROD,
                                   table_id='gms_grfi_mac_vcml_nmg',
                                   )
    else:
        nmr_data = pd.read_csv(p.FILE_INPUT_PATH + p.NON_MAC_RATE_FILE, dtype = p.VARIABLE_TYPE_DIC)
    nmr_data = standardize_df(nmr_data)
    vcml_data = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.VCML_REFERENCE_FILE)
    vcml_data = standardize_df(vcml_data)
    # When there is seperate NMR for brand we need to merge with a combo of VCML_ID
    if p.GENERIC_OPT and not p.BRAND_OPT:
        nmr_data['BG_FLAG'] = 'G'
    elif not p.GENERIC_OPT and p.BRAND_OPT:
        nmr_data['BG_FLAG'] = 'B'
    elif p.GENERIC_OPT and p.BRAND_OPT:
        nmr_data_g = nmr_data.copy()
        nmr_data_g['BG_FLAG'] = 'G'
        nmr_data_b = nmr_data.copy()
        nmr_data_b['BG_FLAG'] = 'B'
        nmr_data = pd.concat([nmr_data_g, nmr_data_b])
    nmr_data = nmr_data.merge(vcml_data['VCML_ID'], on=['VCML_ID']) # filter for relevant VCML IDs
    # floating-point error buffer
    nmr_data = nmr_data[nmr_data.NON_MAC_RATE.notna() & (nmr_data.NON_MAC_RATE > 1.E-4) & (nmr_data.NON_MAC_RATE < 1-1.E-4)] 
    
    nmr_data.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.NON_MAC_RATE_FILE, index=None)

    
def prepare_pref_pharm_list():
    ''' Create the preferred pharmacy list file for MEDD and COMMERCIAL clients
        Input:
            measurement mapping
        Output:
            preferred pharmacy list w.r.t. client, breakout, and region
        Process:
            for COMMERCIAL clients, there is no preferred pharmacy
            for MEDD clients, preferred pharmacies for client, breakout, and region are joined by a comma
                preferred pharmacy is inferred from the PREFFERED column of MEASUREMENT_MAPPING data set
    '''
    import pandas as pd
    import CPMO_parameters as p   
    from CPMO_shared_functions import standardize_df
    import BQ
    import util_funcs as uf
    
    if p.READ_FROM_BQ:
        measurement_mapping = uf.read_BQ_data(
                BQ.ger_opt_msrmnt_map.format(_customer_id=uf.get_formatted_string(p.CUSTOMER_ID)),
                project_id=p.BQ_INPUT_PROJECT_ID,
                dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                table_id="combined_measurement_mapping" + p.WS_SUFFIX + p.CCP_SUFFIX,
                customer=', '.join(sorted(p.CUSTOMER_ID)))
    else:
        measurement_mapping = pd.read_csv(p.FILE_INPUT_PATH + p.MEASUREMENT_MAPPING, dtype = p.VARIABLE_TYPE_DIC)
    measurement_mapping = standardize_df(measurement_mapping)
    if (p.CLIENT_TYPE == 'COMMERCIAL' or p.CLIENT_TYPE == 'MEDICAID'):
        ##for COMMERCIAL clients, preferred vs. non-preferred does not matter
        pref_pharm_list = measurement_mapping[['CLIENT', 'BREAKOUT', 'REGION']].drop_duplicates()
        pref_pharm_list['PREF_PHARM'] = 'None'
    elif p.CLIENT_TYPE == 'MEDD':
        ##for some MEDD clients, some pharmacies are preferred w.r.t. breakout and region while others will be non-preferred
        ##currently, COMMERCIAL measurement mapping's PREFERRED column stores this information as [Preferred, Non-preferred]
        ##whereas MEDD mapping stores the same information as [PREF, NONPREF]

        ##if measurement mapping contains preferred pharmacies; written to handle both MEDD and COMMERCIAL conventions
        ##otherwise, MEDD preferred pharmacy list is generated similar to COMMERCIAL clients
        if any(measurement_mapping.PREFERRED.str.lower().str.startswith('pref')):
            ##measurement mapping is filtered for clients, breakouts, and regions whose corresponding pharmacy is preferred
            ##then, grouped by client, breakout, and region, preferred pharmacies are joined in a string separated by a comma
            ##non-preferred pharmacies are filled with 'None'
            pharm_list = measurement_mapping[['CLIENT', 'BREAKOUT', 'REGION']].drop_duplicates()
            pref_pharm_list = measurement_mapping.loc[measurement_mapping.PREFERRED.str.lower().str.startswith('pref'), ['CLIENT', 'BREAKOUT', 'REGION', 'CHAIN_GROUP']] \
                                                 .groupby(['CLIENT', 'BREAKOUT', 'REGION'])['CHAIN_GROUP'] \
                                                 .apply(lambda x: ','.join(x.drop_duplicates())) \
                                                 .reset_index() \
                                                 .rename(columns = {'CHAIN_GROUP': 'PREF_PHARM'})
            pref_pharm_list = pharm_list.merge(pref_pharm_list, on = ['CLIENT', 'BREAKOUT', 'REGION'], how = 'left').fillna('None')
        else:
            pref_pharm_list = measurement_mapping[['CLIENT', 'BREAKOUT', 'REGION']].drop_duplicates()
            pref_pharm_list['PREF_PHARM'] = 'None'
    else:
        assert False, "CLIENT_TYPE is not of type COMMERCIAL or MEDD or MEDICAID."

    ##for now, PSAO_GUARANTEE remain ineffective for COMMERCIAL clients
    ##this might be revisited in future
    pref_pharm_list['PSAO_GUARANTEE'] = 'No'
    if p.PREF_PHARM_BYPASS:
        pref_pharm_list['PREF_PHARM'] = 'None'
    pref_pharm_list.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.PREFERRED_PHARM_FILE, index = False)

def prepare_goodrx():
    ''' create the goodrx price limits input file for goodrx client algo runs
    
    intputs: p.DAILY_TOTALS_FILE and SJ's Cleaned goodrx prices
    output: client goodrx price limits used in LP algorithm  
    
    '''
    import pandas as pd
    import CPMO_parameters as p  
    import CPMO_goodrx_functions as grx
    from CPMO_shared_functions import standardize_df
    import BQ
    import util_funcs as uf

    if p.GOODRX_OPT:
        # Use input raw goodrx file instead of BQ if one is submitted
        if p.RAW_GOODRX:
            raw_goodrx_df = pd.read_excel(p.FILE_INPUT_PATH + p.RAW_GOODRX, dtype=p.VARIABLE_TYPE_DIC)
        else:
            if p.READ_FROM_BQ:
                raw_goodrx_df =  uf.read_BQ_data(
                                    project_id=p.BQ_INPUT_PROJECT_ID,
                                    dataset_id=p.BQ_INPUT_DATASET_ENT_CNFV_PROD,
                                    table_id='gms_goodrx_drug_price',
                                    query=BQ.raw_goodrx_custom.format(_project = p.BQ_INPUT_PROJECT_ID,
                                                                      _staging_dataset = p.BQ_INPUT_DATASET_ENT_CNFV_PROD),
                                    custom=True
                                            )
                raw_goodrx_df = raw_goodrx_df.rename(columns={'PRICE_UNIT_QTY': 'UNIT PRICE'})

        if p.READ_FROM_BQ:
            lp_data_df = uf.read_BQ_data(
                BQ.daily_totals_pharm,
                project_id = p.BQ_INPUT_PROJECT_ID,
                dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
                table_id = 'combined_daily_totals' + p.WS_SUFFIX + p.CCP_SUFFIX,
                client = ', '.join(sorted(p.CUSTOMER_ID)),
                claim_date = p.LAST_DATA.strftime('%m/%d/%Y')
            )
        else:
            lp_data_df = pd.read_csv(p.FILE_INPUT_PATH + p.DAILY_TOTALS_FILE, dtype = p.VARIABLE_TYPE_DIC)

        lp_data_df = standardize_df(lp_data_df)
        raw_goodrx_df = standardize_df(raw_goodrx_df)

        merge_method = p.GOODRX_MATCH_METHOD
        if 'QTY' not in raw_goodrx_df.columns:  # if there is no quantity to match on just use the lowest price
            merge_method = 'lowest'
            raw_goodrx_df['QTY'] = pd.np.nan

        cleaned_goodrx_df = grx.goodrx_grxprice_preprocess(raw_goodrx_df)
        lp_data_goodrx_df = grx.goodrx_data_preprocess(lp_data_df, cleaned_goodrx_df,
                                                       merge_method=merge_method,
                                                       qty_method=p.GOODRX_QTY_METHOD)
        rules = grx.goodrx_rules_setup()
        outdf = grx.goodrx_price_limits(lp_data_goodrx_df, rules)
        goodrx_price_limits = grx.goodrx_output_cleanup(outdf)
    
        # Drop NAs (places where recommendations didn't meet criteria)
        goodrx_price_limits = goodrx_price_limits[~goodrx_price_limits.isnull().any(axis=1)]
        if 'BG_FLAG' not in goodrx_price_limits.columns:
            goodrx_price_limits['BG_FLAG'] = 'G'
                
        goodrx_price_limits.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.GOODRX_FILE, index=False)

def prepare_rms():
    ''' Create the RMS price limits input file for Rebalanced MAC Strategy Algo runs for all 
    types of clients, regions, breakout, measurement, pref/nonpref, chains, gpi
    
    intputs: 
    
    YTD_COMBINED_CLAIMS: Claims aggregated per Client, Region, Breakout, Measurement, 
    Preferred, Chain_Group, GPI, NDC 
             
    GOODRX_RAW_DATA: Minimum and Maximum GoodRx price per GPI, NDC, QTY 
    
    output: RMS Upper limits at Client, Region, Breakout, Measurement, Preferred, Chain_Group, GPI
    '''
    import pandas as pd
    import CPMO_parameters as p  
    import CPMO_goodrx_functions as grx
    from CPMO_shared_functions import standardize_df
    import CPMO_rms_functions as rms
    import BQ
    import util_funcs as uf
    from qa_checks import qa_dataframe
    
    #Add assert statements after every data load

    raw_goodrx_df, ytd_claims_df, vcml_gpi_df, costplus_df = rms.import_grx_claims_data()

    #Exclude the claims and GoodRx prices that have an active GPI NDC price on Base MAC
    cleaned_goodrx_df = pd.merge(raw_goodrx_df, vcml_gpi_df, how = 'left', on = ['GPI','NDC'])
    cleaned_goodrx_df = cleaned_goodrx_df[(cleaned_goodrx_df.BASE_MAC.isnull()) | cleaned_goodrx_df.NDC.str.contains('*',regex=False)]

    cleaned_ytd_claims_df = pd.merge(ytd_claims_df, vcml_gpi_df, how = 'left', on = ['GPI','NDC'])
    cleaned_ytd_claims_df = cleaned_ytd_claims_df[(cleaned_ytd_claims_df.BASE_MAC.isnull())]

    #Merge the claims data with goodrx data to get raw goodrx limit    
    ytd_claims_df_grx_raw = rms.grx_claims_mapped(cleaned_goodrx_df, cleaned_ytd_claims_df)

    costplus_df['MAIL_CEILING'] = p.MAIL_MULTIPLIER[0]*costplus_df['MCCP_UNIT_PRICE']
    costplus_df = costplus_df.drop(columns = 'MCCP_UNIT_PRICE')
    costplus_limits = costplus_df.copy(deep=True)
    costplus_limits.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.COSTPLUS_FILE, index=False)

    if p.RMS_OPT:
        if p.SIMPLE_RMS:
            grx_calculated = rms.rms_logic_implementation(ytd_claims_df_grx_raw)
        else:
            mbr_rules = rms.grx_mbr_rules_setup()
            clnt_rules = rms.grx_clnt_rules_setup()
            grx_rules_mapped_df = rms.map_grx_rules(ytd_claims_df_grx_raw, mbr_rules, clnt_rules)
            grx_calculated = rms.apply_grx_logic(grx_rules_mapped_df)
    else:
        grx_calculated = ytd_claims_df_grx_raw.copy(deep=True)
        if p.APPLY_GENERAL_MULTIPLIER:
            grx_calculated['GOODRX_CHAIN_PRICE'] = p.GENERAL_MULTIPLIER[0]*grx_calculated['GOODRX_UNIT_PRICE_SAME']
        
    if p.RMS_OPT or p.APPLY_GENERAL_MULTIPLIER:
        # Saving this file for QCing the logic. We might want to save it in the future too. 
        goodrx_limits = rms.prepare_rms_output(grx_calculated)
        
        if 'BG_FLAG' not in grx_calculated.columns:
            grx_calculated['BG_FLAG'] = 'G'
        if 'BG_FLAG' not in goodrx_limits.columns:
            goodrx_limits['BG_FLAG'] = 'G'
            
        grx_calculated.to_csv(p.FILE_LOG_PATH + 'GOODRX_CALCS_LOG_{}.csv'.format(p.DATA_ID), index=False)
        
        goodrx_limits.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.GOODRX_FILE, index=False)

def prepare_leakage_gpis(leakage_rank = 200):
    '''
    Create the top leakage GPI list
    '''
    import CPMO_parameters as p  
    from CPMO_shared_functions import standardize_df
    import BQ
    from qa_checks import qa_dataframe
    from google.cloud import bigquery
    
    if p.LOCKED_CLIENT and p.LEAKAGE_LIST != 'All':
        if p.READ_FROM_BQ:
            if p.LEAKAGE_LIST == 'Client':
                leakage_gpi_df = uf.read_BQ_data(
                                        project_id = p.BQ_INPUT_PROJECT_ID,
                                        dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
                                        table_id='gpi_leakage_rankings',
                                        query=BQ.gpi_leakage_rankings_custom.format(_project = p.BQ_INPUT_PROJECT_ID,
                                                                                    _dataset = p.BQ_INPUT_DATASET_DS_PRO_LP,
                                                                                    _table_id = 'gpi_leakage_rankings',
                                                                                    _customer_id = uf.get_formatted_string(p.CUSTOMER_ID),
                                                                                    _leakage_rank = leakage_rank),
                                        custom=True)
                leakage_gpi_df = standardize_df(leakage_gpi_df)
                qa_dataframe(leakage_gpi_df, dataset = 'LEAKAGE_GPI_FILE')
                leakage_gpi_df.to_csv(p.FILE_DYNAMIC_INPUT_PATH + 'LEAKAGE_GPI_DF_{}.csv'.format(p.DATA_ID), index=False)
                assert leakage_gpi_df.shape[0] > 0, "No data in the leakage optimization list. Please check the data source."         
            elif p.LEAKAGE_LIST == 'Legacy':
                gpi_query = f'''
                (SELECT GPI, 'G' AS BG_FLAG, TOTAL_LEAKAGE, LEAKAGE_RANK
                FROM `pbm-mac-lp-prod-ai.ds_sandbox.ZBD_GPI_LIST`
                WHERE date = '20221201'
                AND LEAKAGE_RANK <= {leakage_rank});
                '''
                bqclient = bigquery.Client()
                leakage_gpi_df = (bqclient.query(gpi_query).result().to_dataframe())
                leakage_gpi_df = standardize_df(leakage_gpi_df)
                qa_dataframe(leakage_gpi_df, dataset = 'LEAKAGE_GPI_FILE')
                leakage_gpi_df.to_csv(p.FILE_DYNAMIC_INPUT_PATH + 'LEAKAGE_GPI_DF_{}.csv'.format(p.DATA_ID), index=False)
                assert leakage_gpi_df.shape[0] > 0, "No data in the leakage optimization list. Please check the data source."
            else:
                assert False, "No valid value was specified in parameters for p.LEAKAGE_LIST or trying to use p.ZBD_OPT with 'All'."
        else:
            assert False, "p.READ_FROM_BQ must be True in order to use ZBD_OPT"

def check_start_date():
    '''
    read contract effective and expiration dates for MEDD/COMMERCIAL clients
    '''
    import CPMO_parameters as p
    import datetime as dt
    import util_funcs as uf
    import pandas as pd
    from dateutil import relativedelta
    import BQ

    if p.PROGRAM_INPUT_PATH[:3] == 'gs:':
        if p.READ_FROM_BQ:
            df_ = uf.read_BQ_data(
            query = BQ.contract_info_custom.format(_customer_id = uf.get_formatted_string(p.CUSTOMER_ID),
                                                   _data_start = p.DATA_START_DAY,
                                                   _last_data = dt.datetime.strftime(p.LAST_DATA, '%Y-%m-%d'),
                                                   _project = uf.get_formatted_project_id(p.BQ_INPUT_PROJECT_ID),
                                                   _landing_dataset = uf.get_formatted_dataset_id(p.BQ_INPUT_DATASET_ENT_CNFV_PROD),
                                                   _table_id = uf.get_formatted_table_id('gms_ger_opt_customer_info_all_algorithm' + p.WS_SUFFIX)),
                project_id = p.BQ_INPUT_PROJECT_ID,
                dataset_id = uf.get_formatted_dataset_id(p.BQ_INPUT_DATASET_ENT_CNFV_PROD),
                table_id = 'gms_ger_opt_customer_info_all_algorithm' + p.WS_SUFFIX,
                custom = True
            )
        else:
            df_ = pd.read_csv(p.FILE_INPUT_PATH + p.CONTRACT_DATE_FILE, dtype = p.VARIABLE_TYPE_DIC)
            df_.columns = df_.columns.str.upper()
            df_['CONTRACT_EFF_DT'] = pd.to_datetime(df_['CONTRACT_EFF_DT'])
            df_['CONTRACT_EXPRN_DT'] = pd.to_datetime(df_['CONTRACT_EXPRN_DT'])
    else:
        assert False, "The p.PROGRAM_INPUT_PATH should be on GCP or p.READ_FROM_BQ = True "
    if p.TRUECOST_CLIENT:
        df_ = uf.read_BQ_data(
            query = BQ.contract_info_custom.format(_customer_id = uf.get_formatted_string(p.CUSTOMER_ID),
                                                   _data_start = p.DATA_START_DAY,
                                                   _last_data = dt.datetime.strftime(p.LAST_DATA, '%Y-%m-%d'),
                                                   _project = uf.get_formatted_project_id(p.BQ_INPUT_PROJECT_ID),
                                                   _landing_dataset = uf.get_formatted_dataset_id(p.BQ_INPUT_DATASET_ENT_CNFV_PROD),
                                                   _table_id = uf.get_formatted_table_id('gms_truecost_customer_info_all_algorithm')),
                project_id = p.BQ_INPUT_PROJECT_ID,
                dataset_id = uf.get_formatted_dataset_id(p.BQ_INPUT_DATASET_ENT_CNFV_PROD),
                table_id = 'gms_truecost_customer_info_all_algorithm',
                custom = True
            )
        df_['contract_eff_dt'] = dt.datetime.strptime('2025-01-01', '%Y-%m-%d')
    
    df_.columns = df_.columns.str.upper()
    #Temporarily change the last contract expiration date from 2/28 to 2/29 in a leap year 
    temp_contract_exp_dt = pd.to_datetime(df_['CONTRACT_EXPRN_DT']) + dt.timedelta(days = 1)
    if p.FULL_YEAR & (temp_contract_exp_dt[0].strftime("%m-%d") == '02-29'):
        df_['CONTRACT_EXPRN_DT'] = pd.to_datetime(df_['CONTRACT_EXPRN_DT']) + dt.timedelta(days = 1)
    
    data_pull_ln = relativedelta.relativedelta(p.LAST_DATA, dt.datetime.strptime(p.DATA_START_DAY, '%Y-%m-%d'))
    eoy_ln = relativedelta.relativedelta(df_['CONTRACT_EXPRN_DT'][0], p.GO_LIVE)
    curr_cntrct_ln = relativedelta.relativedelta(p.LAST_DATA, df_['CONTRACT_EFF_DT'][0])

    #TODO: once data is set in BQ, check these asserts
    if len(df_['CONTRACT_EFF_DT']) < 1:
        raise AssertionError('No Contract Date Information')
    elif len(df_['CONTRACT_EFF_DT'].unique()) > 1:
        raise AssertionError('Multiple Contract Start Dates')
    elif p.CROSS_CONTRACT_PROJ == False and df_['CONTRACT_EFF_DT'][0] != dt.datetime.strptime(p.DATA_START_DAY, '%Y-%m-%d').date():
        raise AssertionError('Data start day is different from contract start day')
    elif len(df_['CONTRACT_EXPRN_DT'].unique()) > 1:
        raise AssertionError('Multiple Contract End Dates')
    elif p.FULL_YEAR == False and df_['CONTRACT_EXPRN_DT'][0] <= p.GO_LIVE.date():
        raise AssertionError('The client contract is expiring before or on the GO_LIVE date. Change the GO_LIVE date or reconsider appropriateness of the run.')
    elif p.FULL_YEAR & (df_['CONTRACT_EXPRN_DT'][0] + dt.timedelta(days = 1)  != p.GO_LIVE.date()):
        raise AssertionError('Client run incorrectly set as Full Year Run. Reset the FULL_YEAR parameter and rerun')
    elif p.CROSS_CONTRACT_PROJ == True and relativedelta.relativedelta(df_['CONTRACT_EXPRN_DT'][0], df_['CONTRACT_EFF_DT'][0]).months + 1 != 12:
        raise AssertionError('CROSS_CONTRACT_PROJ is not compatible with 6 month contract clients. If you have a different contract length, it should be investigated to determine if using this parameter is appropriate.')
    elif p.CROSS_CONTRACT_PROJ == True and ((data_pull_ln.months + (data_pull_ln.years * 12)) - (curr_cntrct_ln.months + (curr_cntrct_ln.years * 12) + 1)) != (eoy_ln.months + (eoy_ln.years * 12)):
        raise AssertionError('Prior contract data pull from Daily Totals is not the same length as the current contract EOY period. Adjust DATA_START_DAY.')
    
    if p.FULL_YEAR & (temp_contract_exp_dt[0].strftime("%m-%d") == '02-29'):
        df_['CONTRACT_EXPRN_DT'] = pd.to_datetime(df_['CONTRACT_EXPRN_DT']) - dt.timedelta(days = 1)
    
    df_.drop_duplicates(subset = ['CONTRACT_EFF_DT', 'CONTRACT_EXPRN_DT'], inplace = True, ignore_index = True)
    df_.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.CONTRACT_DATE_FILE, index = False)
    print("Start date check successful.")

def populate_brand_generic_df(**kwargs):
    ''' This function is used to query sb_finance_g2_guarantees table to get surplus data by brand and filter it for appropriate client names
        Input: keyword arguments are being used to format the query text. 
        Process:
            query teradata
                if access issue, read a local file
            filter client names
    '''
    import BQ
    import numpy as np
    import pandas as pd
    import datetime as dt
    import util_funcs as uf
    import CPMO_parameters as p
    import CPMO_shared_functions as sf
    from qa_checks import qa_dataframe

    if p.CLIENT_LOB == 'CMK' and p.FULL_YEAR: 
        assert p.BRAND_SURPLUS_READ_CSV, "For full-year runs on BoB clients, p.BRAND_SURPLUS_READ_CSV should be set to True. This is to assume that brand performance is on-target in the coming year."

    if p.CLIENT_LOB == 'AETNA':
        if p.READ_FROM_BQ:
            brand_specialty_surplus = uf.read_BQ_data(
                BQ.gms_aetna_brand_specialty_surplus,
                project_id = p.BQ_INPUT_PROJECT_ID,
                dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
                table_id = 'brand_specialty_surplus',
                customer = ', '.join(sorted(p.CUSTOMER_ID)))
        else:
            assert False, 'Error: READ FROM BQ'

        brand_specialty_surplus = sf.standardize_df(brand_specialty_surplus)
        guarantee_category = brand_specialty_surplus.GUARANTEE_CATEGORY.unique()
        assert len(guarantee_category) == 1, "Multiple guarantee categories"

        brand_specialty_surplus['CHANNEL'] = brand_specialty_surplus['CHANNEL'].str.upper()
        brand_specialty_surplus['MAILIND'] = brand_specialty_surplus['CHANNEL'].str[0]
        
        #guarantee type that offsets between brand and generic witht specialty 
        if guarantee_category in ['Aetna MR Offsetting']:
            brand_specialty_surplus.loc[:,'BREAKOUT'] =  brand_specialty_surplus.loc[:,'CLIENT'] + '_RETAIL'
            brand_specialty_surplus.loc[brand_specialty_surplus.MAILIND == 'M','BREAKOUT'] =  brand_specialty_surplus.loc[:,'CLIENT'] + '_MAIL'
            brand_specialty_surplus_agg = brand_specialty_surplus\
                                        .groupby(['CLIENT','MEASUREMENT']).agg(SURPLUS = ('YTD_SURPLUS',sum)).reset_index()
        #guarantee types that offset between brand and generic without specialty 
        elif guarantee_category in ['Aetna BG Pure Vanilla','Aetna BG Offsetting']:
            brand_specialty_surplus.loc[:,'BREAKOUT'] =  brand_specialty_surplus.loc[:,'CLIENT'] + '_RETAIL'
            brand_specialty_surplus.loc[brand_specialty_surplus.MAILIND == 'M','BREAKOUT'] =  brand_specialty_surplus.loc[:,'CLIENT'] + '_MAIL'
            brand_specialty_surplus_agg = brand_specialty_surplus.loc[brand_specialty_surplus.MAILIND != 'S']\
                                        .groupby(['CLIENT','MEASUREMENT']).agg(SURPLUS = ('YTD_SURPLUS',sum)).reset_index()
        #guarantee types that offset between retail 30 and 90
        elif guarantee_category in ['Aetna Offsetting','Aetna NA Offsetting']:
            brand_specialty_surplus['BREAKOUT'] = np.where(
                brand_specialty_surplus['BUCKET'] == 'GER',
                brand_specialty_surplus[['CLIENT', 'CHANNEL']].agg('_'.join, axis=1),
                brand_specialty_surplus[['CLIENT', 'BUCKET', 'CHANNEL']].agg('_'.join, axis=1)
            )
            brand_specialty_surplus_agg = pd.DataFrame(columns = ['CLIENT','MEASUREMENT','SURPLUS'])
            
        #guarantee types that do not offset between brand/specialty and generic 
        elif guarantee_category in ['Aetna Pure Vanilla','Aetna NA Pure Vanilla']:
            brand_specialty_surplus['BREAKOUT'] = brand_specialty_surplus[['CLIENT', 'BUCKET', 'MAILIND', 'MEASUREMENT']].agg('_'.join, axis = 1) 
            brand_specialty_surplus_agg = pd.DataFrame(columns = ['CLIENT','MEASUREMENT','SURPLUS'])

        else: assert False, 'Incorrect Guarantee Cateogry'    

        contract_date_df = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.CONTRACT_DATE_FILE)
        soc = dt.datetime.strptime(contract_date_df['CONTRACT_EFF_DT'][0], '%Y-%m-%d')
        ytd = p.LAST_DATA
        go_live = p.GO_LIVE
        eoc = dt.datetime.strptime(contract_date_df['CONTRACT_EXPRN_DT'][0], '%Y-%m-%d')

        brand_specialty_surplus['FULLAWP_ADJ_PROJ_LAG'] =  (brand_specialty_surplus['FULLAWP_ADJ'] * ((go_live - ytd).days + 1) / ((ytd - soc).days + 1))
        brand_specialty_surplus['LAG_REIMB'] =  (brand_specialty_surplus['PRICE_REIMB'] * ((go_live - ytd).days + 1) / ((ytd - soc).days + 1))

        brand_specialty_surplus['FULLAWP_ADJ_PROJ_EOY'] =  (brand_specialty_surplus['FULLAWP_ADJ'] * ((eoc - go_live).days + 1) / ((ytd - soc).days + 1))
        brand_specialty_surplus['Old_Price_Effective_Reimb_Proj_EOY'] =  (brand_specialty_surplus['PRICE_REIMB'] * ((eoc - go_live).days + 1) / ((ytd - soc).days + 1))
        brand_specialty_surplus['Price_Effective_Reimb_Proj'] =  (brand_specialty_surplus['PRICE_REIMB'] * ((eoc - go_live).days + 1) / ((ytd - soc).days + 1))

        cols = ['CLIENT', 'REGION', 'BREAKOUT', 'MEASUREMENT', 'MAILIND', 
                'BUCKET', 'CHANNEL', 'RATE',
                'FULLAWP_ADJ', 'FULLAWP_ADJ_PROJ_LAG', 'FULLAWP_ADJ_PROJ_EOY',
                'PRICE_REIMB', 'LAG_REIMB', 'Old_Price_Effective_Reimb_Proj_EOY', 'Price_Effective_Reimb_Proj']
        brand_specialty_surplus[cols].to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.BRAND_SURPLUS_REPORT, index=False)
        brand_specialty_surplus_agg.to_csv(p.FILE_DYNAMIC_INPUT_PATH +  p.BRAND_SURPLUS_FILE, index=False)

    else:
        if p.BRAND_SURPLUS_READ_CSV: 
                #the local table path is given by p.FILE_INPUT_PATH + p.BER_GER_SURPLUS_FILE
                #NOTE that p.FILE_INPUT_PATH is already defined in CPMOP_parameters.py
                #BRAND_SURPLUS_FILE should be defined in CPMOP_parameters.py as the name of manual surplus table
                try:
                    brand_surplus_df = sf.standardize_df(pd.read_csv(p.FILE_INPUT_PATH + p.BRAND_SURPLUS_FILE))
                except:
                    print("WARNING: p.BRAND_SURPLUS_READ_CSV = True,\n but file {} does not exist.".format(p.BRAND_SURPLUS_FILE))
                    print("Brand surplus will be set to zero")
                    brand_surplus_df = pd.DataFrame(columns=['CLIENT', 'MEASUREMENT', 'SURPLUS'])
        else:
            if p.PROGRAM_INPUT_PATH[:3] == 'gs:':
                brand_surplus_df = uf.read_BQ_data(
                    project_id=p.BQ_INPUT_PROJECT_ID,
                    dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
                    table_id='brand_generic_custom',
                    query = BQ.brand_generic.format(**kwargs),
                    custom = True
                )
            else:
                assert False, "The p.PROGRAM_INPUT_PATH should be on GCP or p.BRAND_SURPLUS_READ_CSV = True"

            #filter for brand rows
            brand_surplus_df['MEASUREMENT'] = 'R30'
            brand_surplus_df = brand_surplus_df.loc[brand_surplus_df.BUCKET == 'BER'].reset_index(drop=True)
            brand_surplus_df = brand_surplus_df.loc[brand_surplus_df.OFFSET_GROUP.str.contains('_A_')].reset_index(drop=True)

            if len(brand_surplus_df) > 0: 
                #filter out for client names with additional suffixes other than those including '90' or '90 day' (or '90Day', '90day', etc.) indicating Measurement = R90; 
                #this excludes, e.g., RTE_90 or 90 Day_Preferred suffixes.
                ######NOTE: this is a 2-step hacky way to filter client names; 
                #1st step is to filter for client names that equal the shortest string assuming they do not have additional suffixes such as LTC, HIF, TER, Speciality, etc.
                #addtional step: since R90 measurement is not included in this df, add a measurement column knowing it's either M30 or R30 
                client_common_name_df = brand_surplus_df.loc[brand_surplus_df.CLNAME.isin(list(brand_surplus_df.CLNAME[brand_surplus_df.CLNAME.str.len().groupby(brand_surplus_df.CLCODE).idxmin()]))]
                client_common_name_df['MEASUREMENT'] = client_common_name_df.MAILIND + '30'
                #2nd step is to filter for client names that that have the 90, 90day, 90 day, 90 Day, or 90Day suffix attached to their name; excluding additional suffixes
                #this is done by replacing the common name with '' and checking wethere the remaining string.lower() (once spaces and special chars are removed) is in ['90' or '90day']
                #addtional step: since the following df has only R90 measurement group, add a measurement column
                client_90day_names_df = brand_surplus_df.loc[brand_surplus_df.apply(lambda row: ''.join(ch.lower() for ch in row.CLNAME.replace(client_common_name_df.loc[client_common_name_df.CLCODE == row.CLCODE, 'CLNAME'].values[0], '') if ch.isalnum()), axis=1).isin(['90', '90day'])]
                client_90day_names_df['MEASUREMENT'] = client_90day_names_df.MAILIND + '90'
                brand_surplus_df = client_common_name_df.append(client_90day_names_df)

            # Define 'CLIENT'
            brand_surplus_df['CLIENT'] = brand_surplus_df['CLCODE']

        #filter for necessary columns
        brand_surplus_df = brand_surplus_df[['CLIENT', 'MEASUREMENT', 'SURPLUS']]

        # generate a report showing which clients have BG offsets
        print('============= Brand Generic Offset Clients ================\n')
        if len(brand_surplus_df) == 0:
            if len(p.CUSTOMER_ID) == 1:
                print('Client {} does not have a BG offset'.format(p.CUSTOMER_ID))
            else:
                print('Clients, {}, do not have a BG offsets.'.format(p.CUSTOMER_ID))
        else:
            if len(p.CUSTOMER_ID) == 1:
                print('Client {} has a BG offset'.format(p.CUSTOMER_ID))
            else:
                print('The following clients have a brand-generic offset, this offset will be included in the optimization' )
                for client in brand_surplus_df.CLCODE.unique():
                    print('\t{}'.format(client))
        print('=========================================================')

        brand_surplus_df.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.BRAND_SURPLUS_FILE,index=False)

def populate_spclty_surplus_df():
    '''
    Reads in flat file containing expected specialty surplus for the whole contract period.
    
    Currently only receiving these data manually for Pacificsource customer IDs. But if this
    feature expands we'll use this function to pull data from BQ and preprocess it.
    
    Outputs CSV file to be used by specialty_surplus_dict_generator_ytd_lag_eoy(). If SPECIALTY_OFFSET is False or the customer ID does
    not have specialty offsetting data, then an intentionally empty file is saved to Dynamic Input.
    '''
    import CPMO_shared_functions as sf
    from qa_checks import qa_dataframe
    import pandas as pd
    import CPMO_parameters as p

    spclty_surplus_df = sf.standardize_df(pd.read_csv(p.FILE_INPUT_PATH + p.SPECIALTY_SURPLUS_DATA, dtype = p.VARIABLE_TYPE_DIC))
    qa_dataframe(spclty_surplus_df, dataset = p.SPECIALTY_SURPLUS_DATA)
    
    if p.SPECIALTY_OFFSET:
        cid_list = p.CUSTOMER_ID
    else:
        cid_list = []
        
    spclty_surplus_df = spclty_surplus_df[spclty_surplus_df['CUSTOMER_ID'].isin(cid_list)]
    
    spclty_surplus_df.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.SPECIALTY_SURPLUS_FILE, index=False)
    
    # Generate a report showing which clients have specialty offsets
    print('\n============= Specialty Offset Clients ================\n')
    if p.SPECIALTY_OFFSET == False:
        print('Specialty offsetting is turned off.')
    elif len(spclty_surplus_df) == 0:
        print('Client {} does not have a specialty offset'.format(p.CUSTOMER_ID))
    else:
        print('Client {} has a specialty offset'.format(p.CUSTOMER_ID))
    print('=========================================================')

def prepare_client_guarantees_premc():
    import BQ
    from CPMO_shared_functions import standardize_df, add_virtual_r90
    import pandas as pd
    import CPMO_parameters as p

    # Get client guarantees from before the market check occurred, if applicable
    if p.TRUECOST_CLIENT:
        client_guarantees = uf.read_BQ_data(
            project_id=p.BQ_INPUT_PROJECT_ID,
            dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
            table_id='combined_measurement_mapping' + p.WS_SUFFIX,
            client = ', '.join(sorted(p.CUSTOMER_ID)),
            query = BQ.client_guarantees_dummy_TC.format(
                _customer_id = uf.get_formatted_string(p.CUSTOMER_ID),
                _project = p.BQ_INPUT_PROJECT_ID, 
                _landing_dataset = p.BQ_INPUT_DATASET_DS_PRO_LP,
                _table_id = 'combined_measurement_mapping'+ p.WS_SUFFIX),
            custom = True)
    elif p.READ_FROM_BQ:
        client_guarantees = uf.read_BQ_data(
            BQ.client_guarantees,
            project_id = p.BQ_INPUT_PROJECT_ID,
            dataset_id = p.BQ_INPUT_DATASET_DS_PRO_LP,
            table_id='client_guarantees_premc' + p.WS_SUFFIX,
            client = ', '.join(sorted(p.CUSTOMER_ID)))
    else:
        client_guarantees = pd.read_csv(p.FILE_DYNAMIC_INPUT_PATH + p.CLIENT_GUARANTEE_PREMC_FILE, dtype = p.VARIABLE_TYPE_DIC)
    
    ## Modify client premac guarantee according to data readin
    if p.GENERIC_OPT and not p.BRAND_OPT:
        client_guarantees = client_guarantees[client_guarantees['BG_FLAG'] == 'G']
    elif not p.GENERIC_OPT and p.BRAND_OPT:
        client_guarantees = client_guarantees[client_guarantees['BG_FLAG'] == 'B']
    client_guarantees = standardize_df(client_guarantees)
    client_guarantees = add_virtual_r90(client_guarantees, update_guarantees = True)
    client_guarantees.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.CLIENT_GUARANTEE_PREMC_FILE, index = False)
    

def read_breakout_df(): 
    '''
    The purpose of this function is to read in the new breakout logic mapping and save it under dynamic input folder
    '''
    import pandas as pd
    import CPMO_parameters as p
    from CPMO_shared_functions import standardize_df
    from qa_checks import qa_dataframe
    import BQ
    import util_funcs as uf
    if p.READ_FROM_BQ:
        breakout_mapping = uf.read_BQ_data(
            BQ.breakout_mapping,
            project_id=p.BQ_INPUT_PROJECT_ID,
            dataset_id=p.BQ_INPUT_DATASET_DS_PRO_LP,
            table_id="breakout_mapping" + p.WS_SUFFIX,
            customer = ', '.join(sorted(p.CUSTOMER_ID))
        )
    else:
        breakout_mapping = pd.read_csv(p.FILE_INPUT_PATH + p.RAW_BREAKOUT_MAPPING_FILE, dtype = p.VARIABLE_TYPE_DIC)
    breakout_mapping = standardize_df(breakout_mapping)
    qa_dataframe(breakout_mapping, dataset = p.BREAKOUT_MAPPING_FILE)
    
    breakout_mapping.to_csv(p.FILE_DYNAMIC_INPUT_PATH + p.BREAKOUT_MAPPING_FILE, index = False)

    
if __name__ == '__main__':
    from CPMO_shared_functions import set_run_status, update_run_status, check_run_status
    if not( check_run_status(run_status="Started") or check_run_status(run_status="Failed")):
        # Table:set_run_status needs to be appended only with one row per run. The row can be updated multiple times, during the LP run
        # The new creation of the pre-processing pods need not run set_run_status, and append a new row to client_run_status
        # This check shall ensure we have only one row in the client_run_status per run_id
        set_run_status()
        update_run_status(i_error_type='Started Preprocessing') 
    try:
        from CPMO_shared_functions import check_and_create_folder
        import CPMO_parameters as p
        import util_funcs as uf
        check_and_create_folder(p.FILE_LOG_PATH)
        unique_id_for_production()
        read_breakout_df()
        check_start_date()
        
        if p.MARKET_CHECK: 
            prepare_client_guarantees_premc()

        prepare_price_overrides_file()

        prepare_mac_1026()
        prepare_generic_launches_file()
        prepare_gpi_change_exclusions()

        prepare_pref_pharm_list()
        prepare_mac_mapping()
        prepare_mac_constraints()

        prepare_pharmacy_guarantees()

        prepare_OC_pharm_pref_file()

        prepare_drug_mac_history()
        
        prepare_non_mac_rate()

        prepare_goodrx()
        if p.RMS_OPT or p.APPLY_GENERAL_MULTIPLIER:
            prepare_rms()
        prepare_leakage_gpis(leakage_rank = p.LEAKAGE_RANK)
        
        populate_brand_generic_df(_customer_id = uf.get_formatted_string(p.CUSTOMER_ID),
                                  _current_year=int(p.LAST_DATA.year),
                                  _project = p.BQ_INPUT_PROJECT_ID,
                                  _landing_dataset = p.BQ_INPUT_DATASET_ENT_CNFV_PROD,
                                  _table_id = "gms_gms_taxonomy" + p.WS_SUFFIX)
        populate_spclty_surplus_df()

    except Exception as e:
        import sys
        exc_type, exc_value, exc_traceback = sys.exc_info()
        fname = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
        error_loc = fname + ' - line ' + str(line_number)
        update_run_status('Failed', 'Preprocessing', repr(e), error_loc)
        raise e
