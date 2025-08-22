from google.cloud import bigquery

all_other_medd_client_perf = f"""
SELECT
  chain_group AS CHAIN_GROUP,
  surplus AS SURPLUS
"""


awp_history_table = f"""
SELECT
  CAST(ndc AS STRING) AS NDC,
  drug_price_at AS DRUG_PRICE_AT,
  FORMAT_DATE('%Y-%m-%d', eff_date) AS EFF_DATE,
  FORMAT_DATE('%Y-%m-%d', exp_date0) AS EXP_DATE,
  full_drug_name AS FULL_DRUG_NAME,
  CAST(gpi AS STRING) AS GPI,
  gpi_cls_nm AS GPI_CLS_NM,
  src_cd AS SRC_CD
"""

client_guarantees = f"""
SELECT
    CLIENT, 
    REGION, 
    BREAKOUT, 
    MEASUREMENT, 
    Rate, 
    PHARMACY_TYPE
"""

client_name_id_mapping = f"""
SELECT
  CAST(client_name AS STRING) AS Client_Name,
  CAST(customer_id AS STRING) AS CUSTOMER_ID,
  guarantee_category AS Guarantee_Category,
  no_of_vcmls AS No_of_Vcmls
"""

client_name_id_mapping_custom = """
SELECT
  CAST(client_name AS STRING) AS Client_Name,
  CAST(customer_id AS STRING) AS CUSTOMER_ID,
  guarantee_category AS Guarantee_Category,
  no_of_vcmls AS No_of_Vcmls
FROM {_project}.{_landing_dataset}.{_table_id}
WHERE Customer_ID in ({_customer_id})
"""

gen_launch_all = f"""
SELECT
  CAST(customer_id AS STRING) AS CUSTOMER_ID,
  CAST(client_name AS STRING) AS Client_Name,
  measurement AS MEASUREMENT,
  chain_group AS CHAIN_GROUP,
  month AS Month,
  CAST(qty AS FLOAT64) AS QTY,
  fullawp AS FULLAWP,
  ing_cost AS ING_COST,
"""

gen_launch_backout = f"""
SELECT
  CAST(gpi AS INT64) AS GPI
"""
# client param needed
ger_opt_mac_price_override = f"""
SELECT
  CAST(client AS STRING) AS CLIENT,
  CAST(region AS STRING) AS REGION,
  vcml_id AS VCML_ID,
  CAST(gpi AS STRING) AS GPI,
  CAST(ndc AS STRING) AS NDC,
  price_ovrd_amt AS PRICE_OVRD_AMT,
  date_added AS DATE_ADDED,
  source AS SOURCE,
  reason AS REASON,
"""

ger_opt_mac_price_override_custom = """
SELECT
  CLIENT,
  REGION,
  VCML_ID,
  GPI,
  NDC,
  PRICE_OVRD_AMT,
  DATE_ADDED,
  SOURCE,
  REASON,
FROM {_project}.{_landing_dataset}.{_table_id}
WHERE client in ({_customer_id})
"""


ger_opt_daily_total = f"""
SELECT
  CAST(client AS STRING) AS client,
  CAST(customer_id AS STRING) AS Customer_Id,
  CAST(client_name AS STRING) AS Client_Name,
  CAST(breakout AS STRING) AS BREAKOUT,
  CAST(region AS STRING) AS Region,
  measurement AS MEASUREMENT,
  preferred AS Preferred,
  chain_group AS CHAIN_GROUP,
  CAST(mlor_cd AS BOOL) AS MLOR_CD,
  gnrcind AS GNRCIND,
  CAST(gpi AS STRING) AS GPI,
  CAST(ndc AS STRING) AS NDC,
  claim_date AS CLAIM_DATE,
  claims AS CLAIMS,
  CAST(qty AS FLOAT64) AS QTY,
  awp AS AWP,
  CAST(spend AS float64) AS SPEND,
  CAST(member_cost AS float64) AS MEMBER_COST,
  CAST(disp_fee AS float64) AS DISP_FEE,
  CAST(ucamt_unit AS float64) AS UCAMT_UNIT,
  CAST(pct25_ucamt_unit AS float64) AS PCT25_UCAMT_UNIT,
  rec_curr_ind AS REC_CURR_IND,
  rec_add_user AS REC_ADD_USER,
  rec_add_ts AS REC_ADD_TS,
  rec_chg_user AS REC_CHG_USER,
  rec_chg_ts AS REC_CHG_TS
"""

daily_totals_pharm = f"""
SELECT
  CAST(client AS STRING) AS client,
  CAST(customer_id AS STRING) AS Customer_Id,
  CAST(client_name AS STRING) AS Client_Name,
  CAST(breakout AS STRING) AS BREAKOUT,
  CAST(region AS STRING) AS Region,
  measurement AS MEASUREMENT,
  preferred AS Preferred,
  chain_group AS CHAIN_GROUP,
  chain_subgroup as CHAIN_SUBGROUP,
  CAST(mlor_cd AS BOOL) AS MLOR_CD,
  gnrcind AS GNRCIND,
  CAST(gpi AS STRING) AS GPI,
  CAST(ndc AS STRING) AS NDC,
  claim_date AS CLAIM_DATE,
  CLAIMS_distinct AS CLAIMS,
  CAST(qty AS FLOAT64) AS QTY,
  awp AS AWP,
  CAST(spend AS float64) AS SPEND,
  CAST(member_cost AS float64) AS MEMBER_COST,
  CAST(disp_fee AS float64) AS DISP_FEE,
  CAST(PHARMACY_CLAIMS AS float64) AS PHARMACY_CLAIMS,
  CAST(PHARMACY_QTY AS float64) AS PHARMACY_QTY,
  CAST(PHARMACY_AWP AS float64) AS PHARMACY_AWP,
  CAST(PHARMACY_SPEND AS float64) AS PHARMACY_SPEND,
  CAST(PHARMACY_MEMBER_COST AS float64) AS PHARMACY_MEMBER_COST,
  CAST(PHARMACY_DISP_FEE AS float64) AS PHARMACY_DISP_FEE,
  uc_claims AS UC_CLAIMS,
  CAST(ucamt_unit AS float64) AS UCAMT_UNIT,
  CAST(pct25_ucamt_unit AS float64) AS PCT25_UCAMT_UNIT,
  CAST(CLIENT_PCT25_UCAMT_UNIT AS float64) AS CLIENT_PCT25_UCAMT_UNIT,
  CAST(CLIENT_PCT50_UCAMT_UNIT AS float64) AS CLIENT_PCT50_UCAMT_UNIT,
  CAST(PHARMACY_PCT25_UCAMT_UNIT AS float64) AS PHARMACY_PCT25_UCAMT_UNIT,
  CAST(PHARMACY_PCT50_UCAMT_UNIT AS float64) AS PHARMACY_PCT50_UCAMT_UNIT,
  rec_curr_ind AS REC_CURR_IND,
  rec_add_user AS REC_ADD_USER,
  rec_add_ts AS REC_ADD_TS,
  rec_chg_user AS REC_CHG_USER,
  rec_chg_ts AS REC_CHG_TS
"""

# needs client param
ger_opt_unc_daily_total = f"""
SELECT
  CAST(client AS STRING) AS client,
  CAST(customer_id AS STRING) AS Customer_Id,
  CAST(client_name AS STRING) AS Client_Name,
  CAST(breakout AS STRING) AS BREAKOUT,
  CAST(region AS STRING) AS Region,
  measurement AS MEASUREMENT,
  preferred AS Preferred,
  chain_group AS CHAIN_GROUP,
  chain_subgroup AS CHAIN_SUBGROUP,
  CAST(mlor_cd AS BOOL) AS MLOR_CD,
  gnrcind AS GNRCIND,
  CAST(gpi AS STRING) AS GPI,
  CAST(ndc AS STRING) AS NDC,
  claim_date AS CLAIM_DATE,
  CLAIMS_distinct AS CLAIMS,
  CAST(qty AS FLOAT64) AS QTY,
  awp AS AWP,
  CAST(spend AS float64) AS SPEND,
  CAST(member_cost AS float64) AS MEMBER_COST,
  CAST(disp_fee AS float64) AS DISP_FEE,
  CAST(PHARMACY_CLAIMS AS float64) AS PHARMACY_CLAIMS,
  CAST(PHARMACY_QTY AS float64) AS PHARMACY_QTY,
  CAST(PHARMACY_AWP AS float64) AS PHARMACY_AWP,
  CAST(PHARMACY_SPEND AS float64) AS PHARMACY_SPEND,
  CAST(PHARMACY_MEMBER_COST AS float64) AS PHARMACY_MEMBER_COST,
  CAST(PHARMACY_DISP_FEE AS float64) AS PHARMACY_DISP_FEE,
  uc_claims AS UC_CLAIMS,
  CAST(ucamt_unit AS float64) AS UCAMT_UNIT,
  CAST(pct25_ucamt_unit AS float64) AS PCT25_UCAMT_UNIT,
  CAST(CLIENT_PCT25_UCAMT_UNIT AS float64) AS CLIENT_PCT25_UCAMT_UNIT,
  CAST(CLIENT_PCT50_UCAMT_UNIT AS float64) AS CLIENT_PCT50_UCAMT_UNIT,
  CAST(PHARMACY_PCT25_UCAMT_UNIT AS float64) AS PHARMACY_PCT25_UCAMT_UNIT,
  CAST(PHARMACY_PCT50_UCAMT_UNIT AS float64) AS PHARMACY_PCT50_UCAMT_UNIT,
  rec_curr_ind AS REC_CURR_IND,
  rec_add_user AS REC_ADD_USER,
  rec_add_ts AS REC_ADD_TS,
  rec_chg_user AS REC_CHG_USER,
  rec_chg_ts AS REC_CHG_TS
"""

gpi_change_exclusion_ndc = f"""
SELECT
  CAST(gpi_cd AS STRING) AS GPI_CD,
  CAST(drug_id AS STRING) AS DRUG_ID
"""

mac_1026 = f"""
SELECT
  mac_list AS MAC_LIST,
  CAST(gpi AS STRING) AS GPI,
  CAST(ndc AS STRING) AS NDC,
  price AS PRICE,
  mac_eff_dt AS MAC_EFF_DT,
"""

mac_list = f"""
SELECT
  CAST(mac AS STRING) AS MAC,
  CAST(gpi AS STRING) AS GPI,
  CAST(ndc AS STRING) AS NDC,
  price AS PRICE,
  mac_list AS MAC_LIST
"""

ger_opt_msrmnt_map = f"""
SELECT
  CAST(client AS STRING) AS client,
  CAST(client_name AS STRING) AS Client_Name,
  CAST(customer_id AS STRING) AS Customer_Id,
  guarantee_category AS Guarantee_Category,
  no_of_vcmls AS No_of_VCMLs,
  MEASUREMENT,
  MEASUREMENT_CLEAN,
  BREAKOUT,
  CAST(region AS STRING) AS Region,
  preferred AS Preferred,
  network AS Network,
  mlor_cd AS MLOR_CD,
  raw_chain_group AS CHAIN_GROUP_TEMP,
  chain_group AS CHAIN_GROUP,
  chain_subgroup AS CHAIN_SUBGROUP,
  number_of_rows,
  distinct_claims as claims,
  fullawp AS FULLAWP,
  CAST(qty AS FLOAT64) AS QTY,
  spend AS SPEND,
  member_cost AS MEMBER_COST,
  disp_fee AS DISP_FEE,
  distinct_pstcosttype_macs
"""

package_size_to_ndc = f"""
SELECT
  drug_nme AS DRUG_NME,
  CAST(gpi AS STRING) AS GPI,
  CAST(ndc11 AS STRING) AS NDC11,
  pack_size AS PACK_SIZE,
  label_name AS LABEL_NAME
"""

commercial_pharm_guarantees = f"""
SELECT
  pharmacy_group AS PHARMACY_GROUP,
  CAST(target_rate AS FLOAT64) AS TARGET_RATE
"""

pharm_guarantees = f"""
SELECT
  pharmacy AS Pharmacy,
  CAST(client AS STRING) AS CLIENT,
  CAST(breakout STRING) AS BREAKOUT,
  CAST(region AS STRING) AS REGION,
  CAST(rate AS FLOAT64) AS Rate
"""

pref_pharm_list = f"""
SELECT
  CAST(client AS STRING) AS CLIENT,
  CAST(breakout AS STRING) AS BREAKOUT,
  CAST(region AS STRING) AS REGION,
  pref_pharm AS PREF_PHARM,
  psao_guarantee AS PSAO_GUARANTEE
"""

# needs client
vcml_reference = f"""
SELECT
  CAST(customer_id AS STRING) AS Customer_ID,
  chnl_ind AS CHNL_IND,
  base_mac_list_id AS Base_MAC_List_ID,
  vcml_id AS VCML_ID,
  scale_factor AS Scale_Factor,
  FORMAT_DATE('%m%b%y', rec_effective_date) AS Rec_Effective_Date,
  FORMAT_DATE('%m%b%y', rec_expiration_date) AS Rec_Expiration_Date,
  CAST(reporting_gid AS INT64) AS Reporting_GID,
  rec_curr_ind AS Rec_Curr_Ind,
  rec_add_user AS Rec_Add_USer,
  FORMAT_TIMESTAMP('%F %H:%M:%E3S', rec_add_ts) AS Rec_Add_TS,
  rec_chg_user AS Rec_Chg_User,
  FORMAT_TIMESTAMP('%F %H:%M:%E3S', rec_chg_ts) AS Rec_Chg_TS,
  CAST(unique_row_id AS INT64) AS Unique_Row_ID,
  CAST(base_factor AS FLOAT64) AS Base_Factor,
  CAST(vcml_reference_worktable_gid AS INT64) AS VCML_Reference_WorkTable_GID,
  floor_ind AS Floor_Ind,
  brand_generic_cd AS Brand_Generic_CD
"""

# intemediary tables
Mac_Mapping = f"""
SELECT
  CAST(CUSTOMER_ID AS STRING) AS CUSTOMER_ID,
  MEASUREMENT,
  CHAIN_GROUP,
  CAST(MAC_LIST AS STRING) AS MAC_LIST,
  CAST(REGION AS STRING) AS REGION
"""

Mac_Constraints = f"""
SELECT
  CAST(CLIENT AS STRING) AS CLIENT,
  CAST(BREAKOUT AS STRING) AS BREAKOUT,
  CAST(REGION AS STRING) AS REGION,
  MEASUREMENT,
  ELE,
  KRG,
  WAG,
  CAR,
  TPS,
  CVS,
  EPC,
  RAD,
  WMT,
  ABS,
  ACH,
  AHD,
  ART,
  GIE,
  KIN,
  NONPREF_OTH,
  PREF_OTH,
  MAIL,
  MCHOICE
"""

YTD_Pharmacy_Performance = f"""
SELECT
  CAST(CLIENT AS STRING) AS CLIENT,
  CAST(REGION AS STRING) AS REGION,
  CAST(BREAKOUT AS STRING) AS BREAKOUT,
  CHAIN_GROUP,
  DOF_MONTH,
  INGREDIENT_COST,
  AWP,
  CLAIM_COUNT
"""

Pharmacy_approx_coef = f"""
SELECT
  CAST(CLIENT AS STRING) AS CLIENT,
  CHAIN_GROUP,
  SLOPE,
  INTERCEPT
"""

Gen_Launch = f"""
SELECT
  CAST(CLIENT AS STRING) AS CLIENT,
  CAST(BREAKOUT AS STRING) AS BREAKOUT,
  CAST(REGION AS STRING) AS REGION,
  MEASUREMENT,
  CHAIN_GROUP,
  MONTH,
  CAST(qty AS FLOAT64) AS QTY,
  FULLAWP,
  ING_COST
"""

mac_lists = f"""
SELECT
  MAC,
  CAST(GPI AS STRING) AS GPI,
  CAST(NDC AS STRING) AS NDC,
  PRICE,
  GPI_NDC,
  MAC_LIST_ID,
  MAC_LIST,
  NDC_Count
"""

Pharmacy_approx_coef = f"""
SELECT
  CAST(CLIENT AS STRING) AS CLIENT,
  CHAIN_GROUP,
  SLOPE,
  INTERCEPT
"""
YTD_Pharmacy_Performance = f"""
SELECT
  CAST(CLIENT AS STRING) AS CLIENT,
  CAST(REGION AS STRING) AS REGION,
  CAST(BREAKOUT AS STRING) AS BREAKOUT,
  CHAIN_GROUP,
  DOF_MONTH,
  INGREDIENT_COST,
  AWP,
  CLAIM_COUNT
"""

Mac_Constraints = f"""
SELECT
  CAST(CLIENT AS STRING) AS CLIENT,
  CAST(BREAKOUT AS STRING) AS BREAKOUT,
  CAST(REGION AS STRING) AS REGION,
  MEASUREMENT,
  ELE,
  KRG,
  WAG,
  CAR,
  TPS,
  CVS,
  EPC,
  RAD,
  WMT,
  ABS,
  ACH,
  AHD,
  ART,
  GIE,
  KIN,
  NONPREF_OTH,
  PREF_OTH,
  MAIL,
  MCHOICE
"""

lp_data = f"""
SELECT
  CAST(CLIENT AS STRING) AS CLIENT,
  CAST(BREAKOUT AS STRING) AS BREAKOUT,
  CAST(REGION AS STRING) AS REGION,
  MEASUREMENT,
  CAST(GPI AS STRING) AS GPI,
  CHAIN_GROUP,
  GO_LIVE,
  MAC_LIST,
  CURRENT_MAC_PRICE,
  GPI_Only,
  CLAIMS,
  CAST(QTY AS FLOAT64) AS QTY,
  FULLAWP_ADJ,
  PRICE_REIMB,
  LM_CLAIMS,
  LM_QTY,
  LM_FULLAWP_ADJ,
  LM_PRICE_REIMB,
  CLAIMS_PROJ_LAG,
  QTY_PROJ_LAG,
  FULLAWP_ADJ_PROJ_LAG,
  CLAIMS_PROJ_EOY,
  QTY_PROJ_EOY,
  FULLAWP_ADJ_PROJ_EOY,
  uc_unit,
  uc_unit25,
  CURR_AWP,
  CURR_AWP_MIN,
  CURR_AWP_MAX,
  CAST(NDC AS STRING) AS NDC,
  GPI_NDC,
  PKG_SZ,
  avg_awp,
  BREAKOUT_AWP_MAX,
  num1026_NDC_PRICE,
  num1026_GPI_PRICE,
  MAC1026_unit_price,
  MAC1026_GPI_FLAG,
  Pharmacy_Type,
  Price_Mutable,
  PRICE_REIMB_UNIT,
  Eff_unit_price,
  MAC_PRICE_UNIT_Adj,
  Eff_capped_price,
  PRICE_REIMB_ADJ
"""

MedD_LP_Algorithm_Pharmacy_Output_Month = f"""
SELECT
  GPI_NDC,
  CAST(GPI AS STRING) AS GPI,
  CAST(NDC AS STRING) AS NDC,
  PKG_SZ,
  CAST(CLIENT AS STRING) AS CLIENT,
  CAST(BREAKOUT AS STRING) AS BREAKOUT,
  CAST(REGION AS STRING) AS REGION,
  MEASUREMENT,
  CHAIN_GROUP,
  CHAIN_SUBGROUP,
  MAC_LIST,
  PRICE_MUTABLE,
  CLAIMS_PROJ_EOY,
  QTY_PROJ_EOY,
  FULLAWP_ADJ_PROJ_EOY,
  OLD_MAC_PRICE,
  MAC1026_UNIT_PRICE,
  CAST(GPI_Strength AS INT64) AS GPI_Strength,
  New_Price,
  lb,
  ub,
  LM_CLAIMS,
  LM_QTY,
  LM_FULLAWP_ADJ,
  LM_PRICE_REIMB,
  PRICE_REIMB_CLAIM
"""

PHARM_AND_CLIENT_VCMLS = """
SELECT
  CAST(mac AS STRING) AS MAC,
  CAST(gpi AS STRING) AS GPI,
  CAST(ndc AS STRING) AS NDC,
  price AS PRICE,
  mac_list AS MAC_LIST
FROM 
  `anbc-prod.fdl_gdp_ae_ds_pro_lp_share_ent_prod.mac_list`
WHERE mac in (
  SELECT 
  CAST(client_mac_list as string) as MAC_LIST
  FROM `anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_CLIENT_PHARM_CLAIMS_STANDARD`
  WHERE customer_id IN ({_customer_id})
  AND client_mac_list LIKE 'MAC%'
  AND DOF BETWEEN '{_contract_eff_date}' AND '{_last_data}'
  UNION DISTINCT
  SELECT
  cast(pharmacy_mac_list as string) as MAC_LIST 
  FROM `anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_CLIENT_PHARM_CLAIMS_STANDARD`
  WHERE customer_id IN ({_customer_id})
  AND pharmacy_mac_list LIKE 'MAC%'
  AND DOF BETWEEN '{_contract_eff_date}' AND '{_last_data}'
)
"""

Price_Check_Output = f"""
SELECT
  CAST(CLIENT AS STRING) AS CLIENT,
  CAST(BREAKOUT AS STRING) AS BREAKOUT,
  CAST(REGION AS STRING) AS REGION,
  MEASUREMENT,
  CHAIN_GROUP,
  CHAIN_SUBGROUP,
  MAC_LIST,
  GPI_NDC,
  CAST(GPI AS STRING) AS GPI,
  CAST(NDC AS STRING) AS NDC,
  OLD_MAC_PRICE,
  New_Price,
  Final_Price,
  PKG_SZ,
  QTY_PROJ_EOY,
  GPI_CHANGE_EXCEPT,
  FULLAWP_ADJ_PROJ_EOY,
  CLAIMS_PROJ_EOY,
  PRICE_MUTABLE,
  MAC1026_UNIT_PRICE
"""
lp_total_output_df =f"""
SELECT * EXCEPT (client_name, timestamp, AT_RUN_ID, Dm_Begn_Dtm, RUN_ID)
"""

ger_opt_unc_ndc_percentiles_constrained = f"""
SELECT 
   CAST(customer_id AS STRING) AS customer_id
  ,CAST(client AS STRING) AS client
  ,CAST(breakout AS STRING) AS BREAKOUT
  ,CAST(region AS STRING) AS region
  ,measurement
  ,preferred
  ,chain_group
  ,chain_subgroup
  ,mlor_cd
  ,gnrcind
  ,CAST(gpi AS STRING) AS gpi
  ,CAST(ndc AS STRING) AS ndc
  ,CAST(min_ucamt_quantity AS float64) AS min_ucamt_quantity
  ,CAST(pct01_ucamt_unit AS float64) AS pct01_ucamt_unit
  ,CAST(pct03_ucamt_unit AS float64) AS pct03_ucamt_unit
  ,CAST(pct25_ucamt_unit AS float64) AS pct25_ucamt_unit
  ,CAST(pct50_ucamt_unit AS float64) AS pct50_ucamt_unit
  ,CAST(pct75_ucamt_unit AS float64) AS pct75_ucamt_unit
  ,CAST(pct90_ucamt_unit AS float64) AS pct90_ucamt_unit
  ,CAST(pct99_ucamt_unit AS float64) AS pct99_ucamt_unit
  ,CAST(max_ucamt_quantity AS float64) AS max_ucamt_quantity
  ,distinct_uc_prices
  ,claims_in_constraints
  ,CAST(claims_cnt AS int64) AS CLAIMS
  ,CAST(quantity_cnt AS float64) AS QTY_IN_CONSTRAINTS
  ,quantity_lt_pct00
  ,quantity_gt_pct00
  ,quantity_lt_pct01
  ,quantity_gt_pct01
  ,quantity_lt_pct03
  ,quantity_gt_pct03
  ,quantity_lt_pct25
  ,quantity_gt_pct25
  ,quantity_lt_pct50
  ,quantity_gt_pct50
  ,quantity_lt_pct75
  ,quantity_gt_pct75
  ,quantity_lt_pct90
  ,quantity_gt_pct90
  ,quantity_lt_pct99
  ,quantity_gt_pct99
  ,quantity_lt_pct100
  ,quantity_gt_pct100

"""

ger_opt_unc_gpi_percentiles_constrained = f"""
SELECT 
   CAST(customer_id AS STRING) AS customer_id
  ,CAST(client AS STRING) AS client
  ,CAST(breakout AS STRING) AS BREAKOUT
  ,CAST(region AS STRING) AS region
  ,measurement
  ,preferred
  ,chain_group
  ,chain_subgroup
  ,mlor_cd
  ,gnrcind
  ,CAST(gpi AS STRING) AS gpi
  ,CAST(ndc AS STRING) AS ndc
  ,CAST(min_ucamt_quantity AS float64) AS min_ucamt_quantity
  ,CAST(pct01_ucamt_unit AS float64) AS pct01_ucamt_unit
  ,CAST(pct03_ucamt_unit AS float64) AS pct03_ucamt_unit
  ,CAST(pct25_ucamt_unit AS float64) AS pct25_ucamt_unit
  ,CAST(pct50_ucamt_unit AS float64) AS pct50_ucamt_unit
  ,CAST(pct75_ucamt_unit AS float64) AS pct75_ucamt_unit
  ,CAST(pct90_ucamt_unit AS float64) AS pct90_ucamt_unit
  ,CAST(pct99_ucamt_unit AS float64) AS pct99_ucamt_unit
  ,CAST(max_ucamt_quantity AS float64) AS max_ucamt_quantity
  ,distinct_uc_prices
  ,claims_in_constraints
  ,CAST(claims_cnt AS int64) AS CLAIMS
  ,CAST(quantity_cnt AS float64) AS QTY_IN_CONSTRAINTS
  ,quantity_lt_pct00
  ,quantity_gt_pct00
  ,quantity_lt_pct01
  ,quantity_gt_pct01
  ,quantity_lt_pct03
  ,quantity_gt_pct03
  ,quantity_lt_pct25
  ,quantity_gt_pct25
  ,quantity_lt_pct50
  ,quantity_gt_pct50
  ,quantity_lt_pct75
  ,quantity_gt_pct75
  ,quantity_lt_pct90
  ,quantity_gt_pct90
  ,quantity_lt_pct99
  ,quantity_gt_pct99
  ,quantity_lt_pct100
  ,quantity_gt_pct100
"""

gen_launch_backout = f"""
SELECT 
 CAST(GPI AS STRING) AS GPI 
"""

performance_files = f"""
SELECT
  ENTITY
  ,PERFORMANCE
"""

awp_spend_total = f"""
SELECT
    CAST(CLIENT AS STRING) AS CLIENT,
    CAST(BREAKOUT AS STRING) AS BREAKOUT,
    CAST(REGION AS STRING) AS REGION,
    CAST(MEASUREMENT AS STRING) AS MEASUREMENT,
    CAST(PHARMACY_TYPE AS STRING) AS PHARMACY_TYPE,
    CAST(CHAIN_GROUP AS STRING) AS CHAIN_GROUP,	
    CAST(CHAIN_SUBGROUP AS STRING) AS CHAIN_SUBGROUP,	
    FULLAWP_ADJ,
    FULLAWP_ADJ_PROJ_LAG,
    FULLAWP_ADJ_PROJ_EOY,
    PRICE_REIMB,
    LAG_REIMB,
    Old_Price_Effective_Reimb_Proj_EOY,
    Price_Effective_Reimb_Proj
"""

awp_spend_perf = f"""
SELECT
    CAST(CUSTOMER_ID AS STRING) AS CUSTOMER_ID,
    CAST(CLIENT AS STRING) AS CLIENT,
    CAST(ENTITY AS STRING) AS ENTITY,
    ENTITY_TYPE,
    CAST(BREAKOUT AS STRING) AS BREAKOUT,
    CAST(CHAIN_GROUP AS STRING) AS CHAIN_GROUP,
    CLIENT_OR_PHARM,
    FULLAWP_ADJ,
    FULLAWP_ADJ_PROJ_LAG,
    FULLAWP_ADJ_PROJ_EOY,
    PRICE_REIMB,
    LAG_REIMB,
    Old_Price_Effective_Reimb_Proj_EOY,
    Price_Effective_Reimb_Proj,
    GEN_LAG_AWP,
    GEN_LAG_ING_COST,
    GEN_EOY_AWP,
    GEN_EOY_ING_COST,
    Pre_existing_Perf,
    Model_Perf,
    Pre_existing_Perf_Generic, 
    Model_Perf_Generic,
    YTD_Perf_Generic,
    Proj_Spend_Do_Nothing,
    Proj_Spend_Model,
    Increase_in_Spend,
    Increase_in_Reimb,
    Total_Ann_AWP,
    GER_Do_Nothing,
    GER_Model,
    GER_Target,
    CAST(ALGO_RUN_DATE AS STRING) AS ALGO_RUN_DATE,
    CLIENT_TYPE,
    UNC_OPT,
    GOODRX_OPT,
    CAST(GO_LIVE AS STRING) AS GO_LIVE,
    DATA_ID,
    TIERED_PRICE_LIM,
    RUN_TYPE,
    AT_RUN_ID,
    Run_rate_w_changes,
    Run_rate_do_nothing,
    IA_CODENAME,
    LEAKAGE_PRE,
    LEAKAGE_POST,
    LEAKAGE_AVOID
    
"""

contract_info_custom = """
SELECT DISTINCT contract_eff_dt, contract_exprn_dt
FROM {_project}.{_landing_dataset}.{_table_id}
WHERE customer_ID IN ({_customer_id})
    AND ('{_data_start}' BETWEEN contract_eff_dt AND contract_exprn_dt
        OR '{_last_data}' BETWEEN contract_eff_dt AND contract_exprn_dt)
    AND lower(client) NOT LIKE '%specialty%'
    AND upper(client) NOT LIKE '%LTC%'
    AND upper(client) NOT LIKE '%WRAP%'
    AND upper(client) NOT LIKE '%LDD%'
"""

brand_generic = """
SELECT CLCODE, ClientName as CLNAME, MAILIND, BUCKET, OFFSET_GROUP, RATE, NORMING, FULLAWP, SURPLUS,
SUM(SURPLUS) OVER (PARTITION BY OFFSET_GROUP) AS SURPLUS_BY_OFFSET
FROM (
SELECT t.CLCODE, s.ClientName, t.MAILIND, t.BUCKET, t.IA_CODENAME AS OFFSET_GROUP, s.TgtRate AS RATE,
SUM(s.ClmCount_MTH) AS FREQ, SUM(s.ICToUse_Mth_SUM) AS NORMING, SUM(s.FULLAWP_MTH_SUM) AS FULLAWP,
(1-s.TgtRate)*SUM(s.FullAWP_Mth_Sum) - SUM(s.ICToUse_Mth_SUM) AS SURPLUS
FROM {_project}.{_staging_dataset}.GMS_SUMMARYPROCESS_MONTHLYDATATBL_RXCLAIM  s
INNER JOIN {_project}.{_landing_dataset}.{_table_id} t
  ON upper(trim(t.CLNAME)) = upper(trim(s.ClientName)) AND t.MAILIND = s.MAILIND AND t.BUCKET = s.BUCKET
WHERE t.CLCODE IN ({_customer_id})
AND s.Year = {_current_year} 
AND lower(ClientName) NOT LIKE '%specialty%'
AND upper(ClientName) NOT LIKE '%LTC%'
AND upper(ClientName) NOT LIKE '%WRAP%'
GROUP BY t.CLCODE, s.ClientName, t.MAILIND, t.BUCKET, t.IA_CODENAME, s.TgtRate) t
"""


full_spend_data = f"""
SELECT
  CAST(CLIENT AS STRING) AS CLIENT,
  CAST(BREAKOUT AS STRING) AS BREAKOUT,
  CAST(REGION AS STRING) AS REGION,
  MEASUREMENT,
  CHAIN_GROUP,
  CLAIMS,
  CAST(QTY AS FLOAT64) AS QTY,
  AWP,
  PRICE_REIMB,
  PERIOD
"""

old_prices_monthly_projections = f"""
SELECT * EXCEPT (client_name, timestamp, AT_RUN_ID)"""

raw_goodrx_custom = """
SELECT 
    GPI_CD, 
    CONCAT(ndc_lablr_id,ndc_prod_cd,ndc_pkg_cd) AS NDC, 
    UPPER(store) AS CHAIN, 
    CAST(qty AS FLOAT64) AS QTY, 
    CAST(PRICE_UNIT_QTY AS FLOAT64) AS PRICE_UNIT_QTY
FROM {_project}.{_staging_dataset}.GOODRX_DRUG_PRICE
WHERE LOAD_DT = (
    SELECT MAX(LOAD_DT) 
    FROM {_project}.{_staging_dataset}.GOODRX_DRUG_PRICE
    )
"""

pharmacy_raw_claims_custom = """
SELECT
  CAST(customer_id AS STRING) AS Customer_ID,
  CAST(client_name AS STRING) AS Client_Name,
  guarantee_category AS Guarantee_Category,
  month AS Month,
  performance AS Performance,
  excl_flag AS EXCL_FLAG,
  network AS NETWORK,
  chain_group AS CHAIN_GROUP,
  channel AS CHANNEL,
  claim_count AS CLAIM_COUNT,
  awp AS AWP,
  ingredient_cost AS INGREDIENT_COST
FROM pbm-mac-lp-prod-de.ds_pro_lp.pharmacy_raw_claims
WHERE customer_id IN ({_customer_id})
    AND CASE WHEN {_start_year} = {_end_year} THEN
            CASE WHEN month >= {_start_month} AND month <= {_end_month} THEN 1 ELSE 0 END
        ELSE -- get tail ends if non calendar year
            CASE WHEN month >= {_start_month} OR month <= {_end_month} THEN 1 ELSE 0 END 
        END = 1
"""
# client_pharm claims

V_GER_OPT_CLIENT_PHARM_CLAIMS_STANDARD = """
SELECT 
cast(customer_id as string) as Customer_ID, 
dof, 
cast(fullawp as float64) as fullawp, 
cast(norming as float64) as norming, 
cast(rate as float64) as rate, 
cast(pharmacy_rate as float64) as pharmacy_rate,
client_guarantee,
pharmacy_guarantee,
chain_group_temp as chain_group,
network,
case when (a.MEASUREMENT_CLEAN ='R30P') then 'R30'
            when (a.MEASUREMENT_CLEAN ='R30N') then 'R30'
            when (a.MEASUREMENT_CLEAN ='R90P') then 'R90'
            when (a.MEASUREMENT_CLEAN ='R90N') then 'R90'
            else a.MEASUREMENT_CLEAN end as MEASUREMENT
FROM `anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_CLIENT_PHARM_CLAIMS_STANDARD` a
LEFT JOIN 
(SELECT
    NCPDP,
    CASE
    	WHEN Pharmacy_Group NOT IN {_pharm_list} THEN 'IND'
    	ELSE Pharmacy_Group
    END AS CHAIN_GROUP_TEMP
FROM `anbc-prod.fdl_ent_cnfv_prod.gms_ger_opt_vw_pharmacy_chain_mapping`) b
ON a.pharmacy_id = b.ncpdp
WHERE customer_id IN ({_customer_id})
AND dof <= DATE('{_last_data}')
AND dof >= DATE('{_contract_eff_date}')
AND GNRCIND = 'TRUE'
AND upper(Measurement_Clean) NOT LIKE '%LTC%'
AND upper(Measurement_Clean) NOT LIKE '%ALF%'
AND LOWER(Client_Name) NOT LIKE '%specialty%'
AND LOWER(Client_Name) NOT LIKE '%ltc%'
AND LOWER(Client_Name) NOT LIKE '%wrap%'
AND LOWER(Client_Name) NOT LIKE '%ldd%'
;
"""

pharmacy_claim_medd_custom = """
SELECT customer_id, FULLAWP,Quantity,NORMING, client_name, cast(carrier_id as string) as carrier_id, network, 
measurement_clean, cast(mlor_cd as bool) as mlor_cd, FORMAT_DATETIME("%m", DOF) as MONTH,
CASE
    WHEN CHAIN_GROUP NOT IN ('CVS', 'RAD', 'WAG', 'KRG', 'WMT','CST','HYV','ELE','MJR','ART') THEN 'IND'
    ELSE CHAIN_GROUP
    END AS CHAIN_GROUP_TEMP
FROM `anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_CLIENT_PHARM_CLAIMS_STANDARD` a 
WHERE DOF between '{_data_start}' AND '{_last_data}'
	AND  CUSTOMER_ID IN ({_customer_id})
    AND UPPER(Client_Name) NOT LIKE '%SPECIALTY%'
    AND UPPER(Client_Name) NOT LIKE '%LTC%'
    AND UPPER(Client_Name) NOT LIKE '%WRAP%'
    AND UPPER(Client_Name) NOT LIKE '%LDD%' 
    and Pharmacy_Guarantee like '%Y%'
"""

pharmacy_measurement_mapping_medd_custom = """
SELECT DISTINCT MEASUREMENT_CLEAN, NETWORK,
CASE
    WHEN CHAIN_GROUP NOT IN ('CVS', 'RAD', 'WAG', 'KRG', 'WMT','CST','HYV','ELE','MJR','ART') THEN 'IND'
    ELSE CHAIN_GROUP
    END AS CHAIN_GROUP_TEMP
FROM `{_project}.{_landing_dataset}.{_table_id}`
WHERE CHAIN_GROUP <> 'IND' and CUSTOMER_ID IN ({_customer_id})
and Pharmacy_Guarantee like '%Y%'
ORDER BY CHAIN_GROUP_TEMP, MEASUREMENT_CLEAN, NETWORK
"""

pharmacy_measurement_mappingNY_medd_custom = """
SELECT DISTINCT MEASUREMENT_CLEAN, NETWORK,
CASE
    WHEN CHAIN_GROUP NOT IN ('CVS', 'RAD', 'WAG', 'KRG', 'WMT') THEN 'IND'
    ELSE CHAIN_GROUP
    END AS CHAIN_GROUP_TEMP
FROM `{_project}.{_landing_dataset}.{_table_id}`
WHERE CHAIN_GROUP <> 'IND' and CUSTOMER_ID IN ({_customer_id})
and Pharmacy_Guarantee like '%Y%'
ORDER BY CHAIN_GROUP_TEMP, MEASUREMENT_CLEAN, NETWORK
"""

pharmacy_target_rate_medd_custom = """
SELECT DISTINCT NETWORK, CHAIN_GROUP, Rate, MEASUREMENT_CLEAN 
FROM
(SELECT sub1.*,
sum(AWP) over (partition by CHAIN_GROUP, MEASUREMENT_CLEAN) as chain_meas_awp,
(sum(AWP) over (partition by CHAIN_GROUP,MEASUREMENT_CLEAN, Rate))/(sum(AWP) over (partition by CHAIN_GROUP, MEASUREMENT_CLEAN)) as ratio
FROM
(SELECT trim(network) as NETWORK, trim(chain_group) as CHAIN_GROUP,Pharmacy_Rate as Rate, trim(Measurement_clean) as MEASUREMENT_CLEAN , sum(FULLAWP) as AWP
FROM `{_project}.{_landing_dataset}.{_table_id}` 
WHERE cast(CUSTOMER_ID as string) in ({_customer_id}) and Client_Name not like '%Specialty%' and Client_Name not like '%LTC%' and Client_Name not like '%WRAP%'
and Client_Name not like '%ALF%' and Client_Name not like '%HIF%' and trim(chain_group) in ('CVS', 'RAD', 'WAG', 'KRG', 'WMT','CST','HYV','ELE','MJR','ART')
and rate > .3 and dof>'{_data_start}' and trim(medd_clm_ind)= 'Y'
and trim(Pharmacy_Guarantee)='Y'
GROUP BY CHAIN_GROUP, NETWORK,MEASUREMENT_CLEAN, RATE
) sub1) sub2
WHERE ratio>0.95
"""

pharmacy_target_rateNY_medd_custom="""
SELECT DISTINCT NETWORK, CHAIN_GROUP, Rate, MEASUREMENT_CLEAN 
FROM
(SELECT sub1.*,
sum(AWP) over (partition by CHAIN_GROUP, MEASUREMENT_CLEAN) as chain_meas_awp,
(sum(AWP) over (partition by CHAIN_GROUP,MEASUREMENT_CLEAN, Rate))/(sum(AWP) over (partition by CHAIN_GROUP, MEASUREMENT_CLEAN)) as ratio
FROM
(SELECT trim(network) as NETWORK, trim(chain_group) as CHAIN_GROUP,Pharmacy_Rate as Rate, trim(Measurement_clean) as MEASUREMENT_CLEAN , sum(FULLAWP) as AWP
FROM `{_project}.{_landing_dataset}.{_table_id}` 
WHERE cast(CUSTOMER_ID as string) in ({_customer_id}) and Client_Name not like '%Specialty%' and Client_Name not like '%LTC%' and Client_Name not like '%WRAP%'
and Client_Name not like '%ALF%' and Client_Name not like '%HIF%' and trim(chain_group) in ('CVS', 'RAD', 'WAG', 'KRG', 'WMT')
and rate > .3 and dof>'{_data_start}' and trim(medd_clm_ind)= 'Y'
and trim(Pharmacy_Guarantee)='Y'
GROUP BY CHAIN_GROUP, NETWORK,MEASUREMENT_CLEAN, RATE
) sub1) sub2
WHERE ratio>0.95
"""

mcchoice_target_rate_medd_custom = """
select target_rate
from `{_project}.{_landing_dataset}.{_table_id}`
where MEDD_CLM_IND = 'Y' AND EFF_DATE >= '2021-01-01' AND GNRCIND = 'G' and CHAIN_GROUP <> 'LTC' and PHARMACY_GROUP IN ('CVS','KROGER','RAD','WAG','WALMART')
and (pharmacy_type like '%MEDD%' and pharmacy_type like '%Network%' or pharmacy_type like '%Maintenance%')
and pharmacy_type = 'Maintenance Choice'
ORDER BY PHARMACY_GROUP, PHARMACY_TYPE"""

##this query is used to get mac list for medd clients in <prepare_gpi_change_exclusions> function of <Pre_Processing.py>
##if DE updates <mac_list> table in <ds_pro_lp> data set according to the new <query_client_mac_list> in <sql_queries.py>
##the <mac_list> query in this script (above) would work for both commercial and medd clients
mac_list_for_medd_custom = """
SELECT DISTINCT CAST (MAC_GPI_CD AS STRING) AS GPI,
                CAST (DRUG_ID AS STRING) AS NDC,
                CAST (TRIM (CAST (MAC_COST_AMT AS STRING)) AS FLOAT64) AS PRICE,
                CONCAT (MAC_GPI_CD, '_', DRUG_ID) AS GPI_NDC,
                REPLACE (MAC_LIST_ID, 'MAC', '') AS MAC_LIST,
                CAST (MAC_LIST_ID AS STRING) AS MAC
FROM {_project}.{_landing_dataset}.{_table_id_base} 
WHERE MAC_LIST_ID IN
(SELECT DISTINCT VCML_ID
 FROM {_project}.{_landing_dataset}.{_table_id_vcml}
 WHERE customer_id IN ({_customer_id}) 
     AND Rec_curr_ind = 'Y' 
     AND CHNL_IND NOT IN ('LTC', 'ALF', 'HIF', 'IHS')
) 
AND MAC_COST_AMT > 0
AND curr_ind = 'Y'
AND CURRENT_DATE BETWEEN MAC_EFF_DT AND MAC_EXPRN_DT
AND VALID_GPPC = 'Y'
"""

commercial_FY_chain_network_uti = """
SELECT A1.*, awp_chain, CAST(awp_network_chain/awp_chain as float64) as RATIO FROM
(SELECT NETWORK, CHAIN_GROUP, sum(fullawp) as awp_network_chain
FROM `{_project}.{_landing_dataset}.{_table_id}`
WHERE CUSTOMER_ID IN ({_customer_id}) and DOF>='{_data_start}' and pharmacy_guarantee = 'Y' and measurement_clean <> 'M30'
GROUP BY NETWORK, CHAIN_GROUP) A1
LEFT JOIN
(SELECT CHAIN_GROUP, sum(fullawp) as awp_chain
FROM `{_project}.{_landing_dataset}.{_table_id}`
WHERE CUSTOMER_ID IN ({_customer_id}) and DOF>='{_data_start}' and pharmacy_guarantee = 'Y' and measurement_clean <> 'M30'
GROUP BY CHAIN_GROUP) A2
ON A1.CHAIN_GROUP = A2.CHAIN_GROUP
ORDER BY CHAIN_GROUP
"""


vcml_claims_custom = """
SELECT DISTINCT PSTCOSTTYPE
FROM {_project}.{_landing_dataset}.{_table_id}
WHERE Customer_ID IN ({_customer_id})
    AND DOF BETWEEN '{_data_start}' AND '{_last_data}'
    AND LOWER(Client_Name) NOT LIKE '%specialty%'
    AND LOWER(Client_Name) NOT LIKE '%ltc%'
    AND LOWER(Client_Name) NOT LIKE '%wrap%'
    AND LOWER(Client_Name) NOT LIKE '%ldd%'
"""

WTW_AON_beg_q_m_prices_by_cust_id_custom = """
SELECT
q.* EXCEPT(rn), m.BEG_M_PRICE
FROM (
  SELECT
  MAC_GPI_CD AS GPI,
  CONCAT(MAC_NDC_LBLR_ID, MAC_NDC_PROD_CD, MAC_NDC_PKG_CD) AS NDC,
  RIGHT(MAC_LIST_ID, LENGTH(MAC_LIST_ID)-3) as MAC_LIST,
  CAST(MAC_COST_AMT AS float64) AS BEG_Q_PRICE,
    ROW_NUMBER() OVER (PARTITION BY MAC_LIST_ID, MAC_GPI_CD ORDER BY MAC_EFF_DT DESC) AS rn
  FROM
    `{_project}.{_landing_dataset}.{_table}`
    WHERE right(left(MAC_LIST_ID, 7),{_customer_id_len}) IN ({_customer_id})
    AND MAC_EFF_DT <= DATE_TRUNC('{_go_live_date}', Quarter)
    AND MAC_STUS_CD = 'A') q
LEFT JOIN (
  SELECT
  MAC_GPI_CD AS GPI,
  CONCAT(MAC_NDC_LBLR_ID, MAC_NDC_PROD_CD, MAC_NDC_PKG_CD) AS NDC,
  RIGHT(MAC_LIST_ID, LENGTH(MAC_LIST_ID)-3) as MAC_LIST,
  CAST(MAC_COST_AMT AS float64) AS BEG_M_PRICE,
    ROW_NUMBER() OVER (PARTITION BY MAC_LIST_ID, MAC_GPI_CD ORDER BY MAC_EFF_DT DESC) AS rn
  FROM
    `{_project}.{_landing_dataset}.{_table}`
    WHERE right(left(MAC_LIST_ID, 7),{_customer_id_len}) IN ({_customer_id})
    AND MAC_EFF_DT <= DATE_TRUNC('{_go_live_date}', Month)
    AND MAC_STUS_CD = 'A') m
ON q.GPI = m.GPI
AND q.NDC = m.NDC
AND q.MAC_LIST = m.MAC_LIST
WHERE q.rn = 1 AND m.rn = 1
"""


GER_OPT_TAXONOMY_FINAL = f"""
SELECT
contract_exprn_dt as contract_expiry,
bg_offset as BG_Offset,
cast(customer_id as string) as Customer_ID, 
guarantee_category
"""

ia_codenames_custom = """
SELECT DISTINCT CAST(trim(mailind) AS STRING) as mailind, CAST(trim(UPPER(IA_CODENAME)) AS STRING) as IA_CODENAME FROM `{_project}.{_landing_dataset}.{_table_id}` 
WHERE trim(clcode) in ({_customer_id}) and trim(BUCKET)='GER' 
AND trim(UPPER(clname)) not like '%ALF%' AND trim(UPPER(clname)) not like '%LTC%' AND trim(UPPER(clname)) not like '%HIF%' AND trim(UPPER(clname)) not like '%\\\_TER%' AND trim(UPPER(clname)) not like '%IHS%' and trim(UPPER(clname)) not like '%WRAP%'
and trim(UPPER(clname)) not like '%LDD%' and trim(UPPER(clname)) not like '%SPECIALTY%'
and current_date between PARSE_DATE('%m/%d/%Y', contractstart) and PARSE_DATE('%m/%d/%Y', contractend)
"""

measurement_mapping_custom = """
select CLIENT, Customer_ID , Client_name, guarantee_category, no_of_vcmls, measurement_clean, breakout, region,
case when measurement_clean LIKE 'M%' then 'None'
else case when non_pref_sum > pref_sum then 'NONPREF' 
else 'PREF' end end as preferred, 
network, carrier_id, cast(mlor_cd as bool) as mlor_cd, 
chain_group_temp, chain_group, measurement, number_of_rows, claims, fullawp, qty, spend, member_cost, disp_fee, distinct_pstcosttype_macs  
from (
select CLIENT, Customer_ID , Client_name, guarantee_category, no_of_vcmls, measurement_clean, breakout, region, prplspreferredcustom, network, carrier_id, mlor_cd, 
chain_group_temp, chain_group, chain_subgroup, measurement, number_of_rows, claims, fullawp, qty, spend, member_cost, disp_fee, distinct_pstcosttype_macs,  
sum(case when preferred = 'NONPREF' then FULLAWP else 0 end) over (partition by CHAIN_GROUP, REGION) as non_pref_sum, 
sum(case when preferred = 'PREF' then FULLAWP else 0 end) over (partition by CHAIN_GROUP, REGION) as pref_sum 
from (
SELECT
    A1.Customer_ID as CLIENT,
    A1.Customer_ID,
    A1.Client_Name,
    A1c.Guarantee_Category, A1c.No_of_VCMLs,
    A1.Measurement_Clean,

        CASE WHEN A1c.Guarantee_Category in ('Pure Vanilla','MedD/EGWP Vanilla','MedD/EGWP NonOffsetting Complex','NonOffsetting Complex') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
             WHEN A1c.Guarantee_Category in ('Offsetting R30/R90','MedD/EGWP Offsetting R30/R90/LTC') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
             WHEN A1c.Guarantee_Category in ('NonOffsetting R30/R90','MedD/EGWP NonOffsetting R30/R90/LTC') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail')
                WHEN A1.Measurement_Clean LIKE '%R30%' THEN CONCAT(A1.Customer_ID,'_R30')
                ELSE CONCAT(A1.CUSTOMER_ID,'_R90') END
             WHEN A1c.Guarantee_Category in ('Offsetting Complex') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
             WHEN A1c.Guarantee_Category in ('MedD/EGWP Offsetting Complex') THEN
                CASE WHEN R30_Offset in ('R30/R30N/R90N') THEN
                    CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail')
                    WHEN A1.Measurement_Clean LIKE '%P%' THEN CONCAT(A1.Customer_ID,'_RPREF')
                    ELSE CONCAT(A1.CUSTOMER_ID,'_RNPREF') END
                ELSE
                    CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
                END
          END AS BREAKOUT,
    

       A1.Customer_ID AS Region,
       CASE WHEN A1d.PRPLSPREFERREDCUSTOM='P' THEN 'PREF' ELSE 
            CASE WHEN A1.MEASUREMENT_CLEAN LIKE '%M%' THEN 'None' ELSE 'NONPREF' END
            END AS PREFERRED,
       A1d.PRPLSPREFERREDCUSTOM,
       A1.NETWORK,
       A1.CARRIER_ID,
       A1.MLOR_CD,
       A2.CHAIN_GROUP_TEMP,
       case when ((A1.Measurement_Clean like 'M%') and (A1.NETWORK in ('MCHCE','CVSNPP','2RATHN','1CTEDS'))) then 'MCHOICE'
          when ((A1.measurement_clean like 'M%') and (A2.chain_group_temp = 'CVS')) then 'MAIL' 
          when ((A1.Measurement_Clean like 'M%') and (A2.CHAIN_GROUP_TEMP='IND')) then 'MAIL'
          when ((A1.Measurement_Clean not like 'M%') and (A2.CHAIN_GROUP_TEMP='IND')) then
            CASE WHEN A1d.PRPLSPREFERREDCUSTOM='P' THEN 'PREF_OTH' ELSE 'NONPREF_OTH' END         
          ELSE A2.CHAIN_GROUP_TEMP END AS CHAIN_GROUP,
        
        case when (A1.MEASUREMENT_CLEAN ='R30P') then 'R30'
            when (A1.MEASUREMENT_CLEAN ='R30N') then 'R30'
            when (A1.MEASUREMENT_CLEAN ='R90P') then 'R90'
            when (A1.MEASUREMENT_CLEAN ='R90N') then 'R90'
            else A1.MEASUREMENT_CLEAN end as MEASUREMENT
            
     ,count(*) as number_of_rows
     ,count(distinct A1.claim_id) as claims
     ,sum(FULLAWP) as FULLAWP
     ,sum(Quantity) as QTY
     ,sum(NORMING) as SPEND
     ,sum(PSTCOPAY) as MEMBER_COST
     ,sum(PSTFEE) as DISP_FEE

     ,count(distinct case when PSTCOSTTYPE like 'MAC%' then PSTCOSTTYPE end) as distinct_pstcosttype_macs


from 
          (
          select CLM1.*
            FROM anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_CLIENT_PHARM_CLAIMS_STANDARD CLM1  
            where GNRCIND = 'TRUE'
            AND Measurement_Clean NOT LIKE '%LTC%'
            AND Measurement_Clean NOT LIKE '%ALF%'
            AND network NOT IN ('2CAR3E', 'ALST90', 'UHGATE') --These networks are to not be included in reconciliation according to Geoff Lee
           -- AND Customer_ID in ('3775','4454','3731','4477','2011','2705','3061','4608','2113','2031','2042','2043' )
            union all 
            select CLM2.*
            from pbm-mac-lp-prod-de.staging.GER_OPT_CLAIMS_ALGORITHM_UNC CLM2 
            where GNRCIND = 'TRUE'
            AND Measurement_Clean NOT LIKE '%LTC%'
            AND Measurement_Clean NOT LIKE '%ALF%'
            AND network NOT IN ('2CAR3E', 'ALST90', 'UHGATE') --These networks are to not be included in reconciliation according to Geoff Lee
           -- AND Customer_ID in ('3775','4454','3731','4477','2011','2705','3061','4608','2113','2031','2042','2043' )
        )A1
INNER JOIN
        (
          SELECT Client_Name, Customer_ID, Guarantee_Category, No_of_VCMLs, R30_Offset, R90_Offset, R30P_Offset,R90P_Offset, R30N_offset, R90N_Offset, contract_eff_dt, contract_exprn_dt
          FROM anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_TAXONOMY_FINAL
          -- WHERE Customer_ID in ('3775','4454','3731','4477','2011','2705','3061','4608','2113','2031','2042','2043' )
        ) A1c
ON A1.Customer_ID = A1c.Customer_ID
and A1.DOF between A1c.contract_eff_dt and A1c.contract_exprn_dt

INNER JOIN
        (
            SELECT
            NCPDP,
            CASE WHEN ((MEDD>0) or (EGWP>0)) THEN
            CASE WHEN Pharmacy_Group NOT IN ('CVS', 'KRG', 'RAD', 'WAG', 'WMT')
                 THEN 'IND'
                 ELSE Pharmacy_Group
                 END 
            ELSE
                 CASE WHEN Pharmacy_Group NOT IN ('ABS', 'HMA', 'AHD', 'ART', 'CAR', 'CVS','ELE', 'EPC', 'GIE', 'KIN', 'KRG', 'RAD', 'TPS', 'WAG', 'WMT')
                 THEN 'IND'
                 ELSE Pharmacy_Group
                 END
            END AS CHAIN_GROUP_TEMP
            FROM (SELECT * FROM anbc-prod.fdl_ent_cnfv_prod.gms_ger_opt_vw_pharmacy_chain_mapping , (SELECT * FROM anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_TAXONOMY_FINAL WHERE Customer_ID in ({_customer_id}) )x) y
        ) AS A2
ON A1.Pharmacy_ID = A2.NCPDP

LEFT JOIN
        (
          SELECT DISTINCT NIFCARRIERID,PRPNETWORK,PRPLSPREFERREDCUSTOM
          FROM pbm-mac-lp-prod-ai.staging_ds.MSRMNT_MAP_PREF_PHARM 
        ) A1d
ON A1.Carrier_ID = A1d.NIFCARRIERID
AND A1.NETWORK = A1d.PRPNETWORK

WHERE   A1.Client_Name NOT LIKE '%Specialty%'
        and upper(A1.Client_Name) NOT LIKE '%LDD%' 
        and A1.Client_Name not like '%WRAP%' 
        and A1.Client_Name not like '%ALF%' and A1.Client_Name not like '%HIF%'
       -- AND DOF >= '2021-01-01' AND DOF <= '2021-09-30'
        GROUP BY
        CLIENT,
        A1.Customer_ID,
        A1.Client_Name,
        A1c.Guarantee_Category, A1c.No_of_VCMLs,
        A1.Measurement_Clean,

          CASE WHEN A1c.Guarantee_Category in ('Pure Vanilla','MedD/EGWP Vanilla','MedD/EGWP NonOffsetting Complex','NonOffsetting Complex') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
             WHEN A1c.Guarantee_Category in ('Offsetting R30/R90','MedD/EGWP Offsetting R30/R90/LTC') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
             WHEN A1c.Guarantee_Category in ('NonOffsetting R30/R90','MedD/EGWP NonOffsetting R30/R90/LTC') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail')
                WHEN A1.Measurement_Clean LIKE '%R30%' THEN CONCAT(A1.Customer_ID,'_R30')
                ELSE CONCAT(A1.CUSTOMER_ID,'_R90') END
             WHEN A1c.Guarantee_Category in ('Offsetting Complex') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
             WHEN A1c.Guarantee_Category in ('MedD/EGWP Offsetting Complex') THEN
                CASE WHEN R30_Offset in ('R30/R30N/R90N') THEN
                    CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail')
                    WHEN A1.Measurement_Clean LIKE '%P%' THEN CONCAT(A1.Customer_ID,'_RPREF')
                    ELSE CONCAT(A1.CUSTOMER_ID,'_RNPREF') END
                ELSE
                    CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
                END
          END,

        A1.NETWORK,
        A1.CARRIER_ID,
        A1.MLOR_CD,
        A2.CHAIN_GROUP_TEMP,
        A1d.PRPLSPREFERREDCUSTOM,
        chain_group,
        Measurement
) ttt
) tttt

WHERE customer_id IN ({_customer_id}) 
"""

daily_total_custom2 = """
  SELECT
  A1.CLIENT AS client,
  A1.CUSTOMER_ID as customer_id,
  A3.Client_Name as client_name,
  A3.BREAKOUT as breakout,
  A3.REGION as region,
  A3.Measurement_Clean AS measurement,
  A3.PREFERRED as preferred,
  A3.CHAIN_GROUP as chain_group,
  A1.MLOR_CD as mlor_cd,
  A1.GNRCIND as gnrcind,
  A1.GPI as gpi ,
  A1.NDC as ndc,
  A1.DOF AS claim_date,
  CAST(COUNT(*) AS FLOAT64)AS claims,
  SUM(A1.QUANTITY) AS qty,
  SUM(A1.FULLAWP) AS awp,
  SUM(A1.NORMING) AS spend,
  SUM(A1.PSTCOPAY) AS member_cost,
  SUM(A1.PSTFEE) AS disp_fee,
  A4.PCT50_UCAMT_UNIT AS ucamt_unit,
  A4.PCT25_UCAMT_UNIT AS pct25_ucamt_unit,
  'Y' AS rec_curr_ind,
  'Infoworks' AS rec_add_user,
  CURRENT_TIMESTAMP AS rec_add_ts,
  'Infoworks' AS rec_chg_user,
  CURRENT_TIMESTAMP as rec_chg_ts
FROM (
  SELECT
    client,
    customer_id,
    Client_Name,
	carrier_id,
    MLOR_CD,
    GNRCIND,
    GPI,
    NDC,
    QUANTITY,
    FULLAWP,
    PSTCOPAY,
    PSTFEE,
    UCAMT,
    CLAIM_ID,
    Pharmacy_ID,
    network,
    measurement_clean,
    norming,
    DOF
  FROM
    pbm-mac-lp-prod-de.ds_pro_lp_dev.ger_opt_unc_mac CLM1
  WHERE
    GNRCIND = 'TRUE'
    AND Measurement_Clean NOT LIKE '%LTC%'
    AND Measurement_Clean NOT LIKE '%ALF%'
    AND network NOT IN ('2CAR3E', 'ALST90', 'UHGATE') --These networks are to not be included in reconciliation according to Geoff Lee
    ) A1
INNER JOIN (
  SELECT
    NCPDP,
    CASE WHEN Pharmacy_Group NOT IN ('ABS', 'HMA', 'AHD', 'ART', 'CAR', 'CVS', 'ELE', 'EPC', 'GIE', 'KIN', 'KRG', 'RAD', 'TPS', 'WAG', 'WMT') THEN 'IND'
    ELSE Pharmacy_Group
    END AS CHAIN_GROUP_TEMP
  FROM anbc-prod.fdl_ent_cnfv_prod.gms_ger_opt_vw_pharmacy_chain_mapping ) A2
ON
  A1.Pharmacy_ID = A2.NCPDP
INNER JOIN
  (select CLIENT, Customer_ID , Client_name, guarantee_category, no_of_vcmls, measurement_clean, breakout, region,
case when measurement_clean LIKE 'M%' then 'None'
else case when non_pref_sum > pref_sum then 'NONPREF' 
else 'PREF' end end as preferred, 
network, carrier_id, mlor_cd, 
chain_group_temp, chain_group, measurement, number_of_rows, claims, fullawp, qty, spend, member_cost, disp_fee, distinct_pstcosttype_macs  
from (
select CLIENT, Customer_ID , Client_name, guarantee_category, no_of_vcmls, measurement_clean, breakout, region, prplspreferredcustom, network, carrier_id, mlor_cd, 
chain_group_temp, chain_group, measurement, number_of_rows, claims, fullawp, qty, spend, member_cost, disp_fee, distinct_pstcosttype_macs,  
sum(case when preferred = 'NONPREF' then FULLAWP else 0 end) over (partition by CHAIN_GROUP, REGION) as non_pref_sum, 
sum(case when preferred = 'PREF' then FULLAWP else 0 end) over (partition by CHAIN_GROUP, REGION) as pref_sum 
from (
SELECT
    A1.Customer_ID as CLIENT,
    A1.Customer_ID,
    A1.Client_Name,
    A1c.Guarantee_Category, A1c.No_of_VCMLs,
    A1.Measurement_Clean,

        CASE WHEN A1c.Guarantee_Category in ('Pure Vanilla','MedD/EGWP Vanilla','MedD/EGWP NonOffsetting Complex','NonOffsetting Complex') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
             WHEN A1c.Guarantee_Category in ('Offsetting R30/R90','MedD/EGWP Offsetting R30/R90/LTC') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
             WHEN A1c.Guarantee_Category in ('NonOffsetting R30/R90','MedD/EGWP NonOffsetting R30/R90/LTC') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail')
                WHEN A1.Measurement_Clean LIKE '%R30%' THEN CONCAT(A1.Customer_ID,'_R30')
                ELSE CONCAT(A1.CUSTOMER_ID,'_R90') END
             WHEN A1c.Guarantee_Category in ('Offsetting Complex') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
             WHEN A1c.Guarantee_Category in ('MedD/EGWP Offsetting Complex') THEN
                CASE WHEN R30_Offset in ('R30/R30N/R90N') THEN
                    CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail')
                    WHEN A1.Measurement_Clean LIKE '%P%' THEN CONCAT(A1.Customer_ID,'_RPREF')
                    ELSE CONCAT(A1.CUSTOMER_ID,'_RNPREF') END
                ELSE
                    CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
                END
          END AS BREAKOUT,
    

       A1.Customer_ID AS Region,
       CASE WHEN A1d.PRPLSPREFERREDCUSTOM='P' THEN 'PREF' ELSE 
            CASE WHEN A1.MEASUREMENT_CLEAN LIKE '%M%' THEN 'None' ELSE 'NONPREF' END
            END AS PREFERRED,
       A1d.PRPLSPREFERREDCUSTOM,
       A1.NETWORK,
       A1.CARRIER_ID,
       A1.MLOR_CD,
       A2.CHAIN_GROUP_TEMP,
       case when ((A1.Measurement_Clean like 'M%') and (A1.NETWORK in ('MCHCE','CVSNPP'))) then 'MCHOICE'
          when ((A1.Measurement_Clean like 'M%') and (A2.CHAIN_GROUP_TEMP='IND')) then 'MAIL'
          when ((A1.Measurement_Clean not like 'M%') and (A2.CHAIN_GROUP_TEMP='IND')) then
            CASE WHEN A1d.PRPLSPREFERREDCUSTOM='P' THEN 'PREF_OTH' ELSE 'NONPREF_OTH' END         
          ELSE A2.CHAIN_GROUP_TEMP END AS CHAIN_GROUP,
        
        case when (A1.MEASUREMENT_CLEAN ='R30P') then 'R30'
            when (A1.MEASUREMENT_CLEAN ='R30N') then 'R30'
            when (A1.MEASUREMENT_CLEAN ='R90P') then 'R90'
            when (A1.MEASUREMENT_CLEAN ='R90N') then 'R90'
            else A1.MEASUREMENT_CLEAN end as MEASUREMENT
            
     ,count(*) as number_of_rows
     ,count(distinct A1.claim_id) as claims
     ,sum(FULLAWP) as FULLAWP
     ,sum(Quantity) as QTY
     ,sum(NORMING) as SPEND
     ,sum(PSTCOPAY) as MEMBER_COST
     ,sum(PSTFEE) as DISP_FEE

     ,count(distinct case when PSTCOSTTYPE like 'MAC%' then PSTCOSTTYPE end) as distinct_pstcosttype_macs


from 
          (
          select CLM1.*
            FROM anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_CLIENT_PHARM_CLAIMS_STANDARD  CLM1  
            where GNRCIND = 'TRUE'
            AND Measurement_Clean NOT LIKE '%LTC%'
            AND Measurement_Clean NOT LIKE '%ALF%'
            AND network NOT IN ('2CAR3E', 'ALST90', 'UHGATE') --These networks are to not be included in reconciliation according to Geoff Lee
           -- AND Customer_ID in ('3775','4454','3731','4477','2011','2705','3061','4608','2113','2031','2042','2043' )
            union all 
            select CLM2.*
            from pbm-mac-lp-prod-de.staging.GER_OPT_CLAIMS_ALGORITHM_UNC CLM2 
            where GNRCIND = 'TRUE'
            AND Measurement_Clean NOT LIKE '%LTC%'
            AND Measurement_Clean NOT LIKE '%ALF%'
            AND network NOT IN ('2CAR3E', 'ALST90', 'UHGATE') --These networks are to not be included in reconciliation according to Geoff Lee
           -- AND Customer_ID in ('3775','4454','3731','4477','2011','2705','3061','4608','2113','2031','2042','2043' )
        )A1
INNER JOIN
        (
          SELECT Client_Name, Customer_ID, Guarantee_Category, No_of_VCMLs, R30_Offset, R90_Offset, R30P_Offset,R90P_Offset, R30N_offset, R90N_Offset, contract_eff_dt, contract_exprn_dt
          FROM anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_TAXONOMY_FINAL
          -- WHERE Customer_ID in ('3775','4454','3731','4477','2011','2705','3061','4608','2113','2031','2042','2043' )
        ) A1c
ON A1.Customer_ID = A1c.Customer_ID
and A1.DOF between A1c.contract_eff_dt and A1c.contract_exprn_dt

INNER JOIN
        (
            SELECT
            NCPDP,
            CASE WHEN Pharmacy_Group NOT IN ('ABS', 'HMA', 'AHD', 'ART', 'CAR', 'CVS','ELE', 'EPC', 'GIE', 'KIN', 'KRG', 'RAD', 'TPS', 'WAG', 'WMT')
                 THEN 'IND'
                 ELSE Pharmacy_Group
                 END AS CHAIN_GROUP_TEMP
            FROM anbc-prod.fdl_ent_cnfv_prod.gms_ger_opt_vw_pharmacy_chain_mapping
        ) AS A2
ON A1.Pharmacy_ID = A2.NCPDP

LEFT JOIN
        (
          SELECT DISTINCT NIFCARRIERID,PRPNETWORK,PRPLSPREFERREDCUSTOM
          FROM pbm-mac-lp-prod-ai.staging_ds.MSRMNT_MAP_PREF_PHARM 
        ) A1d
ON A1.Carrier_ID = A1d.NIFCARRIERID
AND A1.NETWORK = A1d.PRPNETWORK

WHERE   A1.Client_Name NOT LIKE '%Specialty%'
        and upper(A1.Client_Name) NOT LIKE '%LDD%' 
        and A1.Client_Name not like '%WRAP%' 
        and A1.Client_Name not like '%ALF%' and A1.Client_Name not like '%HIF%'
       -- AND DOF >= '2021-01-01' AND DOF <= '2021-09-30'
        GROUP BY
        CLIENT,
        A1.Customer_ID,
        A1.Client_Name,
        A1c.Guarantee_Category, A1c.No_of_VCMLs,
        A1.Measurement_Clean,

          CASE WHEN A1c.Guarantee_Category in ('Pure Vanilla','MedD/EGWP Vanilla','MedD/EGWP NonOffsetting Complex','NonOffsetting Complex') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
             WHEN A1c.Guarantee_Category in ('Offsetting R30/R90','MedD/EGWP Offsetting R30/R90/LTC') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
             WHEN A1c.Guarantee_Category in ('NonOffsetting R30/R90','MedD/EGWP NonOffsetting R30/R90/LTC') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail')
                WHEN A1.Measurement_Clean LIKE '%R30%' THEN CONCAT(A1.Customer_ID,'_R30')
                ELSE CONCAT(A1.CUSTOMER_ID,'_R90') END
             WHEN A1c.Guarantee_Category in ('Offsetting Complex') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
             WHEN A1c.Guarantee_Category in ('MedD/EGWP Offsetting Complex') THEN
                CASE WHEN R30_Offset in ('R30/R30N/R90N') THEN
                    CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail')
                    WHEN A1.Measurement_Clean LIKE '%P%' THEN CONCAT(A1.Customer_ID,'_RPREF')
                    ELSE CONCAT(A1.CUSTOMER_ID,'_RNPREF') END
                ELSE
                    CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
                END
          END,

        A1.NETWORK,
        A1.CARRIER_ID,
        A1.MLOR_CD,
        A2.CHAIN_GROUP_TEMP,
        A1d.PRPLSPREFERREDCUSTOM,
        chain_group,
        Measurement
) ttt
) tttt

WHERE customer_id IN ({_customer_id}) ) AS A3
ON
  A1.customer_id = A3.customer_id
  AND A1.MEASUREMENT_CLEAN = A3.MEASUREMENT_CLEAN
  AND A1.MLOR_CD = A3.MLOR_CD
  AND A2.CHAIN_GROUP_TEMP = A3.CHAIN_GROUP_TEMP
  AND A1.NETWORK=A3.NETWORK
  AND A1.carrier_id=A3.carrier_id
LEFT JOIN
  pbm-mac-lp-prod-de.ds_pro_lp_dev.GER_DA_percentile3 AS A4
ON
  A1.CUSTOMER_ID = A4.CUSTOMER_ID
  AND A3.BREAKOUT = A4.BREAKOUT
  AND A3.REGION = A4.REGION
  AND A3.Measurement_Clean = A4.MEASUREMENT
  AND A3.PREFERRED = A4.PREFERRED
  AND A3.Chain_group = A4.Chain_group
  AND A1.MLOR_CD = A4.MLOR_CD
  AND A1.GNRCIND = A4.GNRCIND
  AND A1.GPI = A4.GPI
  AND A1.NDC = A4.NDC

INNER JOIN anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_TAXONOMY_FINAL AS A5
on A1.customer_id=A5.customer_id
where A1.DOF between A5.contract_eff_dt and A5.contract_exprn_dt and 
A1.customer_id IN ({_customer_id}) and 
A1.DOF <= PARSE_DATE('%m/%d/%Y', '{_claim_date}')

GROUP BY
  A1.CLIENT,
  A1.CUSTOMER_ID,
  A3.Client_Name,
  A3.BREAKOUT,
  A3.REGION,
  A3.Measurement_Clean,
  A3.PREFERRED,
  A3.CHAIN_GROUP,
  A1.MLOR_CD,
  A1.GNRCIND,
  A1.GPI,
  A1.NDC,
  A1.DOF,
  A4.PCT50_UCAMT_UNIT,
  A4.PCT25_UCAMT_UNIT ;
  """

daily_total_custom = """
SELECT
  A1.CLIENT AS client,
  A1.CUSTOMER_ID as customer_id,
  A3.Client_Name as client_name,
  A3.BREAKOUT as breakout,
  A3.REGION as region,
  A3.Measurement AS measurement,
  A3.PREFERRED as preferred,
  A3.CHAIN_GROUP as chain_group,
  A1.MLOR_CD as mlor_cd,
  A1.GNRCIND as gnrcind,
  A1.GPI as gpi,
  A1.NDC as ndc,
  A1.DOF AS claim_date,
  CAST(COUNT(*) AS FLOAT64)AS claims,
  SUM(A1.QUANTITY) AS qty,
  SUM(A1.FULLAWP) AS awp,
  SUM(A1.NORMING) AS spend,
  CAST(SUM(A1.PSTCOPAY) AS FLOAT64) AS member_cost,
  CAST(SUM(A1.PSTFEE) AS FLOAT64) AS disp_fee,
  CAST(A4.PCT50_UCAMT_UNIT AS FLOAT64) AS ucamt_unit,
  CAST(A4.PCT25_UCAMT_UNIT AS FLOAT64) AS pct25_ucamt_unit,
  'Y' AS rec_curr_ind,
  'Infoworks' AS rec_add_user,
  CURRENT_TIMESTAMP AS rec_add_ts,
  'Infoworks' AS rec_chg_user,
  CURRENT_TIMESTAMP as rec_chg_ts
FROM (
  SELECT
    client,
    customer_id,
    Client_Name,
	carrier_id,
    MLOR_CD,
    GNRCIND,
    GPI,
    NDC,
    QUANTITY,
    FULLAWP,
    PSTCOPAY,
    PSTFEE,
    UCAMT,
    CLAIM_ID,
    Pharmacy_ID,
    network,
    measurement_clean,
    norming,
    DOF
  FROM
    pbm-mac-lp-prod-de.ds_pro_lp_dev.ger_opt_unc_mac CLM1
  WHERE
    GNRCIND = 'TRUE'
    AND Measurement_Clean NOT LIKE '%LTC%'
    AND Measurement_Clean NOT LIKE '%ALF%'
    AND network NOT IN ('2CAR3E', 'ALST90', 'UHGATE') --These networks are to not be included in reconciliation according to Geoff Lee
    ) A1
INNER JOIN (
 
            SELECT
            NCPDP,
            CASE WHEN ((MEDD>0) or (EGWP>0)) THEN
            CASE WHEN Pharmacy_Group NOT IN ('CVS', 'KRG', 'RAD', 'WAG', 'WMT')
                 THEN 'IND'
                 ELSE Pharmacy_Group
                 END 
            ELSE
                 CASE WHEN Pharmacy_Group NOT IN ('ABS', 'HMA', 'AHD', 'ART', 'CAR', 'CVS','ELE', 'EPC', 'GIE', 'KIN', 'KRG', 'RAD', 'TPS', 'WAG', 'WMT')
                 THEN 'IND'
                 ELSE Pharmacy_Group
                 END
            END AS CHAIN_GROUP_TEMP
            FROM (SELECT * FROM anbc-prod.fdl_ent_cnfv_prod.gms_ger_opt_vw_pharmacy_chain_mapping , (SELECT * FROM anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_TAXONOMY_FINAL WHERE Customer_ID in ({_customer_id}) )x) y
         ) A2
ON
  A1.Pharmacy_ID = A2.NCPDP
INNER JOIN
  (select CLIENT, Customer_ID , Client_name, guarantee_category, no_of_vcmls, measurement_clean, breakout, region,
case when measurement_clean LIKE 'M%' then 'None'
else case when non_pref_sum > pref_sum then 'NONPREF' 
else 'PREF' end end as preferred, 
network, carrier_id, mlor_cd, 
chain_group_temp, chain_group, measurement, number_of_rows, claims, fullawp, qty, spend, member_cost, disp_fee, distinct_pstcosttype_macs  
from (
select CLIENT, Customer_ID , Client_name, guarantee_category, no_of_vcmls, measurement_clean, breakout, region, prplspreferredcustom, network, carrier_id, mlor_cd, 
chain_group_temp, chain_group, measurement, number_of_rows, claims, fullawp, qty, spend, member_cost, disp_fee, distinct_pstcosttype_macs,  
sum(case when preferred = 'NONPREF' then FULLAWP else 0 end) over (partition by CHAIN_GROUP, REGION) as non_pref_sum, 
sum(case when preferred = 'PREF' then FULLAWP else 0 end) over (partition by CHAIN_GROUP, REGION) as pref_sum 
from (
SELECT
    A1.Customer_ID as CLIENT,
    A1.Customer_ID,
    A1.Client_Name,
    A1c.Guarantee_Category, A1c.No_of_VCMLs,
    A1.Measurement_Clean,

        CASE WHEN A1c.Guarantee_Category in ('Pure Vanilla','MedD/EGWP Vanilla','MedD/EGWP NonOffsetting Complex','NonOffsetting Complex') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
             WHEN A1c.Guarantee_Category in ('Offsetting R30/R90','MedD/EGWP Offsetting R30/R90/LTC') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
             WHEN A1c.Guarantee_Category in ('NonOffsetting R30/R90','MedD/EGWP NonOffsetting R30/R90/LTC') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail')
                WHEN A1.Measurement_Clean LIKE '%R30%' THEN CONCAT(A1.Customer_ID,'_R30')
                ELSE CONCAT(A1.CUSTOMER_ID,'_R90') END
             WHEN A1c.Guarantee_Category in ('Offsetting Complex') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
             WHEN A1c.Guarantee_Category in ('MedD/EGWP Offsetting Complex') THEN
                CASE WHEN R30_Offset in ('R30/R30N/R90N') THEN
                    CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail')
                    WHEN A1.Measurement_Clean LIKE '%P%' THEN CONCAT(A1.Customer_ID,'_RPREF')
                    ELSE CONCAT(A1.CUSTOMER_ID,'_RNPREF') END
                ELSE
                    CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
                END
          END AS BREAKOUT,
    

       A1.Customer_ID AS Region,
       CASE WHEN A1d.PRPLSPREFERREDCUSTOM='P' THEN 'PREF' ELSE 
            CASE WHEN A1.MEASUREMENT_CLEAN LIKE '%M%' THEN 'None' ELSE 'NONPREF' END
            END AS PREFERRED,
       A1d.PRPLSPREFERREDCUSTOM,
       A1.NETWORK,
       A1.CARRIER_ID,
       A1.MLOR_CD,
       A2.CHAIN_GROUP_TEMP,
       case when ((A1.Measurement_Clean like 'M%') and (A1.NETWORK in ('MCHCE','CVSNPP','2RATHN','1CTEDS'))) then 'MCHOICE'
            when ((A1.Measurement_Clean like 'M%') and (A2.CHAIN_GROUP_TEMP = 'CVS')) then 'MAIL'
            when ((A1.Measurement_Clean like 'M%') and (A2.CHAIN_GROUP_TEMP='IND')) then 'MAIL'
            when ((A1.Measurement_Clean not like 'M%') and (A2.CHAIN_GROUP_TEMP='IND')) then
            CASE WHEN A1d.PRPLSPREFERREDCUSTOM='P' THEN 'PREF_OTH' ELSE 'NONPREF_OTH' END         
          ELSE A2.CHAIN_GROUP_TEMP END AS CHAIN_GROUP,
        
        case when (A1.MEASUREMENT_CLEAN ='R30P') then 'R30'
            when (A1.MEASUREMENT_CLEAN ='R30N') then 'R30'
            when (A1.MEASUREMENT_CLEAN ='R90P') then 'R90'
            when (A1.MEASUREMENT_CLEAN ='R90N') then 'R90'
            else A1.MEASUREMENT_CLEAN end as MEASUREMENT
            
     ,count(*) as number_of_rows
     ,count(distinct A1.claim_id) as claims
     ,sum(FULLAWP) as FULLAWP
     ,sum(Quantity) as QTY
     ,sum(NORMING) as SPEND
     ,sum(PSTCOPAY) as MEMBER_COST
     ,sum(PSTFEE) as DISP_FEE

     ,count(distinct case when PSTCOSTTYPE like 'MAC%' then PSTCOSTTYPE end) as distinct_pstcosttype_macs


from 
          (
          select CLM1.*
            FROM anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_CLIENT_PHARM_CLAIMS_STANDARD  CLM1  
            where GNRCIND = 'TRUE'
            AND Measurement_Clean NOT LIKE '%LTC%'
            AND Measurement_Clean NOT LIKE '%ALF%'
            AND network NOT IN ('2CAR3E', 'ALST90', 'UHGATE') --These networks are to not be included in reconciliation according to Geoff Lee
           -- AND Customer_ID in ('3775','4454','3731','4477','2011','2705','3061','4608','2113','2031','2042','2043' )
            union all 
            select CLM2.*
            from pbm-mac-lp-prod-de.staging.GER_OPT_CLAIMS_ALGORITHM_UNC CLM2 
            where GNRCIND = 'TRUE'
            AND Measurement_Clean NOT LIKE '%LTC%'
            AND Measurement_Clean NOT LIKE '%ALF%'
            AND network NOT IN ('2CAR3E', 'ALST90', 'UHGATE') --These networks are to not be included in reconciliation according to Geoff Lee
           -- AND Customer_ID in ('3775','4454','3731','4477','2011','2705','3061','4608','2113','2031','2042','2043' )
        )A1
INNER JOIN
        (
          SELECT Client_Name, Customer_ID, Guarantee_Category, No_of_VCMLs, R30_Offset, R90_Offset, R30P_Offset,R90P_Offset, R30N_offset, R90N_Offset, contract_eff_dt, contract_exprn_dt
          FROM anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_TAXONOMY_FINAL
          -- WHERE Customer_ID in ('3775','4454','3731','4477','2011','2705','3061','4608','2113','2031','2042','2043' )
        ) A1c
ON A1.Customer_ID = A1c.Customer_ID
and A1.DOF between A1c.contract_eff_dt and A1c.contract_exprn_dt

INNER JOIN
        (
            SELECT
            NCPDP,
            CASE WHEN ((MEDD>0) or (EGWP>0)) THEN
            CASE WHEN Pharmacy_Group NOT IN ('CVS', 'KRG', 'RAD', 'WAG', 'WMT')
                 THEN 'IND'
                 ELSE Pharmacy_Group
                 END 
            ELSE
                 CASE WHEN Pharmacy_Group NOT IN ('ABS', 'HMA', 'AHD', 'ART', 'CAR', 'CVS','ELE', 'EPC', 'GIE', 'KIN', 'KRG', 'RAD', 'TPS', 'WAG', 'WMT')
                 THEN 'IND'
                 ELSE Pharmacy_Group
                 END
            END AS CHAIN_GROUP_TEMP
            FROM (SELECT * FROM anbc-prod.fdl_ent_cnfv_prod.gms_ger_opt_vw_pharmacy_chain_mapping , (SELECT * FROM anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_TAXONOMY_FINAL WHERE Customer_ID in ({_customer_id}) )x) y
        ) AS A2
ON A1.Pharmacy_ID = A2.NCPDP

LEFT JOIN
        (
          SELECT DISTINCT NIFCARRIERID,PRPNETWORK,PRPLSPREFERREDCUSTOM
          FROM pbm-mac-lp-prod-ai.staging_ds.MSRMNT_MAP_PREF_PHARM 
        ) A1d
ON A1.Carrier_ID = A1d.NIFCARRIERID
AND A1.NETWORK = A1d.PRPNETWORK

WHERE   A1.Client_Name NOT LIKE '%Specialty%'
        and upper(A1.Client_Name) NOT LIKE '%LDD%' 
        and A1.Client_Name not like '%WRAP%' 
        and A1.Client_Name not like '%ALF%' and A1.Client_Name not like '%HIF%'
       -- AND DOF >= '2021-01-01' AND DOF <= '2021-09-30'
        GROUP BY
        CLIENT,
        A1.Customer_ID,
        A1.Client_Name,
        A1c.Guarantee_Category, A1c.No_of_VCMLs,
        A1.Measurement_Clean,

          CASE WHEN A1c.Guarantee_Category in ('Pure Vanilla','MedD/EGWP Vanilla','MedD/EGWP NonOffsetting Complex','NonOffsetting Complex') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
             WHEN A1c.Guarantee_Category in ('Offsetting R30/R90','MedD/EGWP Offsetting R30/R90/LTC') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
             WHEN A1c.Guarantee_Category in ('NonOffsetting R30/R90','MedD/EGWP NonOffsetting R30/R90/LTC') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail')
                WHEN A1.Measurement_Clean LIKE '%R30%' THEN CONCAT(A1.Customer_ID,'_R30')
                ELSE CONCAT(A1.CUSTOMER_ID,'_R90') END
             WHEN A1c.Guarantee_Category in ('Offsetting Complex') THEN
                CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
             WHEN A1c.Guarantee_Category in ('MedD/EGWP Offsetting Complex') THEN
                CASE WHEN R30_Offset in ('R30/R30N/R90N') THEN
                    CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail')
                    WHEN A1.Measurement_Clean LIKE '%P%' THEN CONCAT(A1.Customer_ID,'_RPREF')
                    ELSE CONCAT(A1.CUSTOMER_ID,'_RNPREF') END
                ELSE
                    CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
                END
          END,

        A1.NETWORK,
        A1.CARRIER_ID,
        A1.MLOR_CD,
        A2.CHAIN_GROUP_TEMP,
        A1d.PRPLSPREFERREDCUSTOM,
        chain_group,
        Measurement
) ttt
) tttt

WHERE customer_id IN ({_customer_id}) ) AS A3
ON
  A1.customer_id = A3.customer_id
  AND A1.MEASUREMENT_CLEAN = A3.MEASUREMENT_CLEAN
  AND A1.MLOR_CD = A3.MLOR_CD
  AND A2.CHAIN_GROUP_TEMP = A3.CHAIN_GROUP_TEMP
  AND A1.NETWORK=A3.NETWORK
  AND A1.carrier_id=A3.carrier_id
LEFT JOIN
  (SELECT DISTINCT
  CLIENT,
  CUSTOMER_ID,
  BREAKOUT,
  REGION,
  MEASUREMENT,
  PREFERRED,
  CHAIN_GROUP,
  MLOR_CD,
  GNRCIND,
  GPI,
  NDC,
  PERCENTILE_DISC(ucamt_quantity,0.25) OVER(PARTITION BY CLIENT, customer_id, breakout, region, MEASUREMENT, preferred, chain_group, mlor_cd, gnrcind, gpi, ndc ) AS PCT25_UCAMT_UNIT,
  PERCENTILE_DISC(ucamt_quantity,0.50) OVER(PARTITION BY CLIENT, customer_id, breakout, region, MEASUREMENT, preferred, chain_group, mlor_cd, gnrcind, gpi, ndc ) AS PCT50_UCAMT_UNIT
FROM (
  SELECT
    CLIENT,
    customer_id,
    breakout,
    region,
    MEASUREMENT,
    preferred,
    chain_group,
    mlor_cd,
    gnrcind,
    gpi,
    ndc,
    claim_id,
    ucamt/quantity AS ucamt_quantity
  FROM
    (SELECT
  B1.CLIENT,
  B1.CUSTOMER_ID,
  B3.Client_Name,
  B3.BREAKOUT,
  B3.REGION,
  B3.Measurement AS MEASUREMENT,
  B3.PREFERRED,
  B3.CHAIN_GROUP,
  B1.MLOR_CD,
  B1.GNRCIND,
  B1.GPI,
  B1.NDC,
  B1.QUANTITY,
  B1.FULLAWP,
  B1.PSTCOPAY,
  B1.PSTFEE,
  B1.UCAMT,
  B1.CLAIM_ID
FROM (
  SELECT
    client,
    customer_id,
    Client_Name,
	carrier_id,
    MLOR_CD,
    GNRCIND,
    GPI,
    NDC,
    QUANTITY,
    FULLAWP,
    PSTCOPAY,
    PSTFEE,
    UCAMT,
    CLAIM_ID,
    Pharmacy_ID,
    network,
    measurement_clean
  FROM
    pbm-mac-lp-prod-de.ds_pro_lp_dev.ger_opt_unc_mac CLM1
  WHERE
    GNRCIND = 'TRUE'
    AND Measurement_Clean NOT LIKE '%LTC%'
    AND Measurement_Clean NOT LIKE '%ALF%' 
    AND network NOT IN ('2CAR3E', 'ALST90', 'UHGATE') --These networks are to not be included in reconciliation according to Geoff Lee
  UNION ALL 
    SELECT
    client,
    customer_id,
    Client_Name,
	carrier_id,
    MLOR_CD,
    GNRCIND,
    GPI,
    NDC,
    QUANTITY,
    FULLAWP,
    PSTCOPAY,
    PSTFEE,
    UCAMT,
    CLAIM_ID,
    Pharmacy_ID,
    network,
    measurement_clean
  FROM
    pbm-mac-lp-prod-de.ds_pro_lp_dev.ger_opt_unc_mac2 CLM2
  WHERE
    GNRCIND = 'TRUE'
    AND Measurement_Clean NOT LIKE '%LTC%'
    AND Measurement_Clean NOT LIKE '%ALF%' 
    AND network NOT IN ('2CAR3E', 'ALST90', 'UHGATE') --These networks are to not be included in reconciliation according to Geoff Lee
    ) B1
INNER JOIN (
                    SELECT
                    NCPDP,
                    CASE WHEN ((MEDD>0) or (EGWP>0)) THEN
                    CASE WHEN Pharmacy_Group NOT IN ('CVS', 'KRG', 'RAD', 'WAG', 'WMT')
                         THEN 'IND'
                         ELSE Pharmacy_Group
                         END 
                    ELSE
                         CASE WHEN Pharmacy_Group NOT IN ('ABS', 'HMA', 'AHD', 'ART', 'CAR', 'CVS','ELE', 'EPC', 'GIE', 'KIN', 'KRG', 'RAD', 'TPS', 'WAG', 'WMT')
                         THEN 'IND'
                         ELSE Pharmacy_Group
                         END
                    END AS CHAIN_GROUP_TEMP
                    FROM (SELECT * FROM anbc-prod.fdl_ent_cnfv_prod.gms_ger_opt_vw_pharmacy_chain_mapping , (SELECT * FROM anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_TAXONOMY_FINAL WHERE Customer_ID in ({_customer_id}) )x) y ) B2
ON
  B1.Pharmacy_ID = B2.NCPDP
INNER JOIN
 (select CLIENT, Customer_ID , Client_name, guarantee_category, no_of_vcmls, measurement_clean, breakout, region,
case when measurement_clean LIKE 'M%' then 'None'
else case when non_pref_sum > pref_sum then 'NONPREF' 
else 'PREF' end end as preferred, 
network, carrier_id, mlor_cd, 
chain_group_temp, chain_group, measurement, number_of_rows, claims, fullawp, qty, spend, member_cost, disp_fee, distinct_pstcosttype_macs  
from (
select CLIENT, Customer_ID , Client_name, guarantee_category, no_of_vcmls, measurement_clean, breakout, region, prplspreferredcustom, network, carrier_id, mlor_cd, 
chain_group_temp, chain_group, measurement, number_of_rows, claims, fullawp, qty, spend, member_cost, disp_fee, distinct_pstcosttype_macs,  
sum(case when preferred = 'NONPREF' then FULLAWP else 0 end) over (partition by CHAIN_GROUP, REGION) as non_pref_sum, 
sum(case when preferred = 'PREF' then FULLAWP else 0 end) over (partition by CHAIN_GROUP, REGION) as pref_sum 
from (
SELECT
    C1.Customer_ID as CLIENT,
    C1.Customer_ID,
    C1.Client_Name,
    C1c.Guarantee_Category, C1c.No_of_VCMLs,
    C1.Measurement_Clean,
        CASE WHEN C1c.Guarantee_Category in ('Pure Vanilla','MedD/EGWP Vanilla','MedD/EGWP NonOffsetting Complex','NonOffsetting Complex') THEN
                CASE WHEN C1.Measurement_Clean LIKE 'M%' THEN CONCAT(C1.CUSTOMER_ID,'_Mail') ELSE CONCAT(C1.CUSTOMER_ID,'_Retail') END
             WHEN C1c.Guarantee_Category in ('Offsetting R30/R90','MedD/EGWP Offsetting R30/R90/LTC') THEN
                CASE WHEN C1.Measurement_Clean LIKE 'M%' THEN CONCAT(C1.CUSTOMER_ID,'_Mail') ELSE CONCAT(C1.CUSTOMER_ID,'_Retail') END
             WHEN C1c.Guarantee_Category in ('NonOffsetting R30/R90','MedD/EGWP NonOffsetting R30/R90/LTC') THEN
                CASE WHEN C1.Measurement_Clean LIKE 'M%' THEN CONCAT(C1.CUSTOMER_ID,'_Mail')
                WHEN C1.Measurement_Clean LIKE '%R30%' THEN CONCAT(C1.Customer_ID,'_R30')
                ELSE CONCAT(C1.CUSTOMER_ID,'_R90') END
             WHEN C1c.Guarantee_Category in ('Offsetting Complex') THEN
                CASE WHEN C1.Measurement_Clean LIKE 'M%' THEN CONCAT(C1.CUSTOMER_ID,'_Mail') ELSE CONCAT(C1.CUSTOMER_ID,'_Retail') END
             WHEN C1c.Guarantee_Category in ('MedD/EGWP Offsetting Complex') THEN
                CASE WHEN R30_Offset in ('R30/R30N/R90N') THEN
                    CASE WHEN C1.Measurement_Clean LIKE 'M%' THEN CONCAT(C1.CUSTOMER_ID,'_Mail')
                    WHEN C1.Measurement_Clean LIKE '%P%' THEN CONCAT(C1.Customer_ID,'_RPREF')
                    ELSE CONCAT(C1.CUSTOMER_ID,'_RNPREF') END
                ELSE
                    CASE WHEN C1.Measurement_Clean LIKE 'M%' THEN CONCAT(C1.CUSTOMER_ID,'_Mail') ELSE CONCAT(C1.CUSTOMER_ID,'_Retail') END
                END
          END AS BREAKOUT,
       C1.Customer_ID AS Region,
       CASE WHEN C1d.PRPLSPREFERREDCUSTOM='P' THEN 'PREF' ELSE 
            CASE WHEN C1.MEASUREMENT_CLEAN LIKE '%M%' THEN 'None' ELSE 'NONPREF' END
            END AS PREFERRED,
       C1d.PRPLSPREFERREDCUSTOM,
       C1.NETWORK,
       C1.CARRIER_ID,
       C1.MLOR_CD,
       C2.CHAIN_GROUP_TEMP,
	   case when ((C1.Measurement_Clean like 'M%') and (C1.NETWORK in ('MCHCE','CVSNPP','2RATHN','1CTEDS'))) then 'MCHOICE'
          when ((C1.Measurement_Clean like 'M%') and (C2.CHAIN_GROUP_TEMP = 'CVS')) then 'MAIL'
          when ((C1.Measurement_Clean like 'M%') and (C2.CHAIN_GROUP_TEMP='IND')) then 'MAIL'
          when ((C1.Measurement_Clean not like 'M%') and (C2.CHAIN_GROUP_TEMP='IND')) then
            CASE WHEN C1d.PRPLSPREFERREDCUSTOM='P' THEN 'PREF_OTH' ELSE 'NONPREF_OTH' END         
          ELSE C2.CHAIN_GROUP_TEMP END AS CHAIN_GROUP,
        
        case when (C1.MEASUREMENT_CLEAN ='R30P') then 'R30'
            when (C1.MEASUREMENT_CLEAN ='R30N') then 'R30'
            when (C1.MEASUREMENT_CLEAN ='R90P') then 'R90'
            when (C1.MEASUREMENT_CLEAN ='R90N') then 'R90'
            else C1.MEASUREMENT_CLEAN end as MEASUREMENT
            
     ,count(*) as number_of_rows
     ,count(distinct C1.claim_id) as claims
     ,sum(FULLAWP) as FULLAWP
     ,sum(Quantity) as QTY
     ,sum(NORMING) as SPEND
     ,sum(PSTCOPAY) as MEMBER_COST
     ,sum(PSTFEE) as DISP_FEE

     ,count(distinct case when PSTCOSTTYPE like 'MAC%' then PSTCOSTTYPE end) as distinct_pstcosttype_macs


from 
          (
          select CLM1.*
            FROM anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_CLIENT_PHARM_CLAIMS_STANDARD  CLM1  
            where GNRCIND = 'TRUE'
            AND Measurement_Clean NOT LIKE '%LTC%'
            AND Measurement_Clean NOT LIKE '%ALF%'
            AND network NOT IN ('2CAR3E', 'ALST90', 'UHGATE') --These networks are to not be included in reconciliation according to Geoff Lee
           -- AND Customer_ID in ('3775','4454','3731','4477','2011','2705','3061','4608','2113','2031','2042','2043' )
            union all 
            select CLM2.*
            from pbm-mac-lp-prod-de.staging.GER_OPT_CLAIMS_ALGORITHM_UNC CLM2 
            where GNRCIND = 'TRUE'
            AND Measurement_Clean NOT LIKE '%LTC%'
            AND Measurement_Clean NOT LIKE '%ALF%'
            AND network NOT IN ('2CAR3E', 'ALST90', 'UHGATE') --These networks are to not be included in reconciliation according to Geoff Lee
           -- AND Customer_ID in ('3775','4454','3731','4477','2011','2705','3061','4608','2113','2031','2042','2043' )
        )C1
INNER JOIN
        (
          SELECT Client_Name, Customer_ID, Guarantee_Category, No_of_VCMLs, R30_Offset, R90_Offset, R30P_Offset,R90P_Offset, R30N_offset, R90N_Offset, contract_eff_dt, contract_exprn_dt
          FROM anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_TAXONOMY_FINAL
          -- WHERE Customer_ID in ('3775','4454','3731','4477','2011','2705','3061','4608','2113','2031','2042','2043' )
        ) C1c
ON C1.Customer_ID = C1c.Customer_ID
and C1.DOF between C1c.contract_eff_dt and C1c.contract_exprn_dt

INNER JOIN
        (
                SELECT
                NCPDP,
                CASE WHEN ((MEDD>0) or (EGWP>0)) THEN
                CASE WHEN Pharmacy_Group NOT IN ('CVS', 'KRG', 'RAD', 'WAG', 'WMT')
                     THEN 'IND'
                     ELSE Pharmacy_Group
                     END 
                ELSE
                     CASE WHEN Pharmacy_Group NOT IN ('ABS', 'HMA', 'AHD', 'ART', 'CAR', 'CVS','ELE', 'EPC', 'GIE', 'KIN', 'KRG', 'RAD', 'TPS', 'WAG', 'WMT')
                     THEN 'IND'
                     ELSE Pharmacy_Group
                     END
                END AS CHAIN_GROUP_TEMP
                FROM (SELECT * FROM anbc-prod.fdl_ent_cnfv_prod.gms_ger_opt_vw_pharmacy_chain_mapping , (SELECT * FROM anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_TAXONOMY_FINAL WHERE Customer_ID in ({_customer_id}) )x) y
                        ) AS C2
ON C1.Pharmacy_ID = C2.NCPDP

LEFT JOIN
        (
          SELECT DISTINCT NIFCARRIERID,PRPNETWORK,PRPLSPREFERREDCUSTOM
          FROM pbm-mac-lp-prod-ai.staging_ds.MSRMNT_MAP_PREF_PHARM 
        ) C1d
ON C1.Carrier_ID = C1d.NIFCARRIERID
AND C1.NETWORK = C1d.PRPNETWORK

WHERE   C1.Client_Name NOT LIKE '%Specialty%'
        and upper(C1.Client_Name) NOT LIKE '%LDD%' 
        and C1.Client_Name not like '%WRAP%' 
        and C1.Client_Name not like '%ALF%' and C1.Client_Name not like '%HIF%'
       -- AND DOF >= '2021-01-01' AND DOF <= '2021-09-30'
        GROUP BY
        CLIENT,
        C1.Customer_ID,
        C1.Client_Name,
        C1c.Guarantee_Category, C1c.No_of_VCMLs,
        C1.Measurement_Clean,

          CASE WHEN C1c.Guarantee_Category in ('Pure Vanilla','MedD/EGWP Vanilla','MedD/EGWP NonOffsetting Complex','NonOffsetting Complex') THEN
                CASE WHEN C1.Measurement_Clean LIKE 'M%' THEN CONCAT(C1.CUSTOMER_ID,'_Mail') ELSE CONCAT(C1.CUSTOMER_ID,'_Retail') END
             WHEN C1c.Guarantee_Category in ('Offsetting R30/R90','MedD/EGWP Offsetting R30/R90/LTC') THEN
                CASE WHEN C1.Measurement_Clean LIKE 'M%' THEN CONCAT(C1.CUSTOMER_ID,'_Mail') ELSE CONCAT(C1.CUSTOMER_ID,'_Retail') END
             WHEN C1c.Guarantee_Category in ('NonOffsetting R30/R90','MedD/EGWP NonOffsetting R30/R90/LTC') THEN
                CASE WHEN C1.Measurement_Clean LIKE 'M%' THEN CONCAT(C1.CUSTOMER_ID,'_Mail')
                WHEN C1.Measurement_Clean LIKE '%R30%' THEN CONCAT(C1.Customer_ID,'_R30')
                ELSE CONCAT(C1.CUSTOMER_ID,'_R90') END
             WHEN C1c.Guarantee_Category in ('Offsetting Complex') THEN
                CASE WHEN C1.Measurement_Clean LIKE 'M%' THEN CONCAT(C1.CUSTOMER_ID,'_Mail') ELSE CONCAT(C1.CUSTOMER_ID,'_Retail') END
             WHEN C1c.Guarantee_Category in ('MedD/EGWP Offsetting Complex') THEN
                CASE WHEN R30_Offset in ('R30/R30N/R90N') THEN
                    CASE WHEN C1.Measurement_Clean LIKE 'M%' THEN CONCAT(C1.CUSTOMER_ID,'_Mail')
                    WHEN C1.Measurement_Clean LIKE '%P%' THEN CONCAT(C1.Customer_ID,'_RPREF')
                    ELSE CONCAT(C1.CUSTOMER_ID,'_RNPREF') END
                ELSE
                    CASE WHEN C1.Measurement_Clean LIKE 'M%' THEN CONCAT(C1.CUSTOMER_ID,'_Mail') ELSE CONCAT(C1.CUSTOMER_ID,'_Retail') END
                END
          END,

        C1.NETWORK,
        C1.CARRIER_ID,
        C1.MLOR_CD,
        C2.CHAIN_GROUP_TEMP,
        C1d.PRPLSPREFERREDCUSTOM,
        chain_group,
        Measurement
) ttt
) tttt) AS B3
ON
  B1.customer_id = B3.customer_id
  AND B1.MEASUREMENT_CLEAN = B3.MEASUREMENT_CLEAN
  AND B1.MLOR_CD = B3.MLOR_CD
  AND B2.CHAIN_GROUP_TEMP = B3.CHAIN_GROUP_TEMP
  AND B1.NETWORK=B3.NETWORK
  AND B1.carrier_id=B3.carrier_id
GROUP BY
  B1.CLIENT,
  B1.CUSTOMER_ID,
  B3.Client_Name,
  B3.BREAKOUT,
  B3.REGION,
  B3.Measurement,
  B3.PREFERRED,
  B3.CHAIN_GROUP,
  B1.MLOR_CD,
  B1.GNRCIND,
  B1.GPI,
  B1.NDC,
  B1.QUANTITY,
  B1.FULLAWP,
  B1.PSTCOPAY,
  B1.PSTFEE,
  B1.UCAMT,
  B1.CLAIM_ID) X
  GROUP BY
    CLIENT,
    customer_id,
    breakout,
    region,
    MEASUREMENT,
    preferred,
    chain_group,
    mlor_cd,
    gnrcind,
    gpi,
    ndc,
    claim_id,
    ucamt_quantity )A) AS A4
ON
  A1.CUSTOMER_ID = A4.CUSTOMER_ID
  AND A3.BREAKOUT = A4.BREAKOUT
  AND A3.REGION = A4.REGION
  AND A3.Measurement = A4.MEASUREMENT
  AND A3.PREFERRED = A4.PREFERRED
  AND A3.Chain_group = A4.Chain_group
  AND A1.MLOR_CD = A4.MLOR_CD
  AND A1.GNRCIND = A4.GNRCIND
  AND A1.GPI = A4.GPI
  AND A1.NDC = A4.NDC

INNER JOIN anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_TAXONOMY_FINAL AS A5
on A1.customer_id=A5.customer_id
where A1.DOF between A5.contract_eff_dt and A5.contract_exprn_dt and 
A1.customer_id IN ({_customer_id}) and 
A1.DOF <= PARSE_DATE('%m/%d/%Y', '{_claim_date}')

GROUP BY
  A1.CLIENT,
  A1.CUSTOMER_ID,
  A3.Client_Name,
  A3.BREAKOUT,
  A3.REGION,
  A3.Measurement,
  A3.PREFERRED,
  A3.CHAIN_GROUP,
  A1.MLOR_CD,
  A1.GNRCIND,
  A1.GPI,
  A1.NDC,
  A1.DOF,
  A4.PCT50_UCAMT_UNIT,
  A4.PCT25_UCAMT_UNIT ;
  """

client_guarantees_custom="""
SELECT B1.CLIENT, B1.REGION, B1.BREAKOUT, B1.MEASUREMENT, CAST(B1.RATE AS FLOAT64) AS Rate, B2.PHARMACY_TYPE FROM (
SELECT C1.CLIENT, C1.REGION, C1.BREAKOUT, C1.MEASUREMENT, C1.RATE,
	ROW_NUMBER() over (partition by CLIENT, REGION, BREAKOUT, MEASUREMENT ORDER BY RATE ASC) as row_number_rates,
	count(*) over (partition by CLIENT, REGION, BREAKOUT, MEASUREMENT) as count_rates FROM
	(SELECT DISTINCT A1.CUSTOMER_ID as CLIENT, 
	    A1.CUSTOMER_ID as REGION,
	    CASE WHEN A2.Guarantee_Category in ('Pure Vanilla','MedD/EGWP Vanilla') THEN
	    		CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
	    	 WHEN A2.Guarantee_Category in ('Offsetting R30/R90','MedD/EGWP Offsetting R30/R90/LTC') THEN
	    	    CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
	         WHEN A2.Guarantee_Category in ('NonOffsetting R30/R90','MedD/EGWP NonOffsetting R30/R90/LTC','MedD/EGWP NonOffsetting Complex','NonOffsetting Complex') THEN
	    	    CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail')
	            WHEN A1.Measurement_Clean LIKE '%R30%' THEN CONCAT(A1.Customer_ID,'_R30')
	            ELSE CONCAT(A1.CUSTOMER_ID,'_R90') END
			 WHEN A2.Guarantee_Category in ('Offsetting Complex') THEN
			 	CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
			 WHEN A2.Guarantee_Category in ('MedD/EGWP Offsetting Complex') THEN
			 	CASE WHEN R30_Offset in ('R30/R30N/R90N') THEN
				 	CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail')
	                WHEN A1.Measurement_Clean LIKE '%P%' THEN CONCAT(A1.Customer_ID,'_RPREF')
	                ELSE CONCAT(A1.CUSTOMER_ID,'_RNPREF') END
				ELSE
					CASE WHEN A1.Measurement_Clean LIKE 'M%' THEN CONCAT(A1.CUSTOMER_ID,'_Mail') ELSE CONCAT(A1.CUSTOMER_ID,'_Retail') END
				END
	      END AS BREAKOUT,
		  
		case when (A1.MEASUREMENT_CLEAN ='R30P') then 'R30'
				when (A1.MEASUREMENT_CLEAN ='R30N') then 'R30'
				when (A1.MEASUREMENT_CLEAN ='R90P') then 'R90'
				when (A1.MEASUREMENT_CLEAN ='R90N') then 'R90'
				else A1.MEASUREMENT_CLEAN end as MEASUREMENT,
	   RATE
	   from {_project}.{_landing_dataset}.{_table_id_cust_info} as A1
	   left join {_project}.{_landing_dataset}.{_table_id_tax} AS A2
	   ON A1.CUSTOMER_ID = A2.CUSTOMER_ID
	        where A1.CUSTOMER_ID in ({_customer_id})
	        and CLIENT not like '%Specialty%'
	        and CLIENT not like '%client owned%'
	        and CLIENT not like '%LTC%'
	        and CLIENT not like '%WRAP%'
	        AND CLIENT NOT LIKE '%LDD%'
	        AND '{_last_data_date}' between A1.contract_eff_dt and A1.contract_exprn_dt
			
			)C1
			) B1
LEFT JOIN
(SELECT * FROM pbm-mac-lp-prod-ai.{_staging_ds_dataset}.{_table_id_pref_dtls}) B2
ON B1.row_number_rates=B2.row_number_rates
AND B1.count_rates=B2.count_rates
"""

disp_fee_pct_cust = """
WITH
  mac_maps AS (
  SELECT
    CUSTOMER_ID,
    MEASUREMENT,
    CHAIN_SUBGROUP,
    MAC_LIST
  FROM
    `{_output_project}.{_output_dataset}.mac_mapping_subgroup`
  WHERE
    CUSTOMER_ID in ({_customer_id})
    AND AT_RUN_ID IN ("{_run_id}")
    ),

  claims AS (
  SELECT
    *
  FROM (
    SELECT
      CASE
        WHEN CHAIN_SUBGROUP IN (
        SELECT DISTINCT CHAIN_SUBGROUP FROM mac_maps WHERE CHAIN_SUBGROUP NOT IN ('MAIL','MCHOICE')) 
        THEN CHAIN_SUBGROUP
      ELSE
      'NONPREF_OTH'
    END
      AS CHAIN_SUBGROUP,
      gpi AS GPI,
      MEASUREMENT,
      CASE
        WHEN qty = 0 THEN NULL
      ELSE
      CAST(DISP_FEE/QTY AS FLOAT64)
    END
      AS DISP_FEE_UNIT
    FROM
      `{_project}.{_dataset}.combined_daily_totals`
    WHERE
      CLAIM_DATE BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL {_time_lag} MONTH)
      AND CURRENT_DATE()
      AND CUSTOMER_ID IN (
      SELECT
        DISTINCT customer_id
      FROM
        `{_project}.{_landing_dataset}.GER_OPT_TAXONOMY_FINAL`
      WHERE
        guarantee_category NOT LIKE "%MedD%")
      AND MEASUREMENT IN ('R30',
        'R90')
      AND CHAIN_SUBGROUP NOT IN ('MCHOICE',
        'MAIL',
        'PREF_OTH' ))
  WHERE
    DISP_FEE_UNIT IS NOT NULL)

SELECT
  DISTINCT MAC_LIST,
  GPI,
  PERCENTILE_DISC(DISP_FEE_UNIT,
    0.25) OVER(PARTITION BY MAC_LIST, gpi) AS MIN_DISP_FEE_UNIT,
  PERCENTILE_DISC(DISP_FEE_UNIT,
    0.75) OVER(PARTITION BY MAC_LIST, gpi) AS MAX_DISP_FEE_UNIT
FROM
  claims a
LEFT JOIN
  mac_maps b
ON
  (a.CHAIN_SUBGROUP = b.CHAIN_SUBGROUP
    AND a.MEASUREMENT = b.MEASUREMENT)
WHERE
  a.MEASUREMENT IN (
  SELECT
    DISTINCT MEASUREMENT
  FROM
    mac_maps)
"""

disp_fee_pct = """
SELECT
    gpi AS GPI,
    CHAIN_GROUP,
    CAST(pct_25 as FLOAT64) AS MIN_DISP_FEE_UNIT,
    CAST(pct_75 as FLOAT64) AS MAX_DISP_FEE_UNIT
"""

zbd_gpi_custom = """
SELECT
  GPI
FROM (
  SELECT
    gpi AS GPI,
    SUM(ZBD_COUNT) AS ZBD_CLAIMS,
    COUNT(*) AS TOTAL_CLAIMS,
    ROUND(SUM(ZBD_COUNT)/COUNT(*),2) AS PERC_ZBD
  FROM (
    SELECT
      *,
      CASE
        WHEN ZERO_BALANCE_INDICATOR = 'TRUE' THEN 1
      ELSE
      0
    END
      AS ZBD_COUNT
    FROM
      `anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_CLIENT_PHARM_CLAIMS_STANDARD`
    WHERE customer_id IN ({_customer_id})
      AND measurement_clean != 'M30'
      AND dof BETWEEN DATE_SUB(current_date(), INTERVAL 1 MONTH) AND current_date()
      AND pharmacy_mac_list != 'NON-MAC'
      AND pharmacy_guarantee = 'Y')
  GROUP BY
    GPI)
WHERE
  PERC_ZBD >= 0.8
"""

goodrx_raw_data = """
SELECT
  GPI,
  NDC,
  CHAIN_GROUP,
  QTY,
  MIN_GRX_PRC_QTY,
  MAX_GRX_PRC_QTY
"""

ytd_combined_claims = """
SELECT
    CUSTOMER_ID,
    CLIENT_NAME,
    REGION,
    BREAKOUT,
    MEASUREMENT,
    PREFERRED,
    CHAIN_GROUP,
    GPI,
    NDC,
    TOTAL_CLAIMS,
    TOTAL_DISP_QTY,
    TOTAL_SPEND,
    TOTAL_MEMBER_COST_AMOUNT,
    TOTAL_CLIENT_BILL_AMOUNT,
    TOTAL_DISP_FEE,
    TOTAL_DISP_QTY/TOTAL_CLAIMS AS AVG_QTY_CLAIM
"""

non_mac_rate = """
SELECT
    VCML_ID,
    1-CAST(NMG_Percent/100. AS FLOAT64) AS NON_MAC_RATE
"""

base_mac_list = """
SELECT
  mac_list_id AS BASE_MAC,
  CAST(mac_gpi_cd AS STRING) AS GPI,
  CAST(drug_id AS STRING) AS NDC, 
FROM
  `{_project}.{_dataset}.{_table}`
WHERE
  mac_list_id IN ('{_base_mac}')
  AND mac_stus_cd = 'A'
  AND curr_ind = 'Y'
  AND valid_gppc = 'Y'
  AND mac_cost_amt > 0
  AND CURRENT_DATE() BETWEEN mac_eff_dt AND mac_exprn_dt
"""

costplus_data = """
SELECT
  GPI_CD AS GPI,
  MAX(UNIT_PRICE_WITH_SHIPPING) AS MCCP_UNIT_PRICE
  FROM
  `{_project}.{_dataset}.{_table}`
WHERE BRAND_GENERIC = 'generic'
AND LOAD_DT = (SELECT MAX(LOAD_DT) FROM  `{_project}.{_dataset}.{_table}`)
GROUP BY GPI_CD
"""

wmt_unc_override_custom = """
SELECT * 
FROM  `{_project}.{_dataset}.{_table}`
WHERE CLIENT IN ({_customer_id})
"""

begin_m_q_prices_custom = """
SELECT
q.* EXCEPT(rn), m.BEG_M_PRICE
FROM (
  SELECT
  MAC_GPI_CD AS GPI,
  CONCAT(MAC_NDC_LBLR_ID, MAC_NDC_PROD_CD, MAC_NDC_PKG_CD) AS NDC,
  MAC_LIST_ID AS VCML_ID,
  CAST (MAC_COST_AMT AS float64) AS BEG_Q_PRICE,
    ROW_NUMBER() OVER (PARTITION BY MAC_LIST_ID, MAC_GPI_CD ORDER BY MAC_EFF_DT DESC) AS rn
  FROM
    `{_project}.{_landing_dataset}.{_table}`
    WHERE MAC_EFF_DT <= DATE_TRUNC('{_go_live}', Quarter)
    AND MAC_STUS_CD = 'A'
    AND MAC_NDC_LBLR_ID LIKE "%*%"
    AND MAC_LIST_ID IN UNNEST({_macs})) q
LEFT JOIN (
SELECT
  MAC_GPI_CD AS GPI,
  CONCAT(MAC_NDC_LBLR_ID, MAC_NDC_PROD_CD, MAC_NDC_PKG_CD) AS NDC,
  MAC_LIST_ID AS VCML_ID,
  CAST (MAC_COST_AMT AS float64) AS BEG_M_PRICE,
  ROW_NUMBER() OVER (PARTITION BY MAC_LIST_ID, MAC_GPI_CD ORDER BY MAC_EFF_DT DESC) AS rn
  FROM
    `{_project}.{_landing_dataset}.{_table}`
    WHERE MAC_EFF_DT <= DATE_TRUNC('{_go_live}', Month)
    AND MAC_STUS_CD = 'A'
    AND MAC_NDC_LBLR_ID LIKE "%*%"
    AND MAC_LIST_ID IN UNNEST({_macs})) m
ON q.GPI = m.GPI
AND q.NDC = m.NDC
AND q.VCML_ID = m.VCML_ID
WHERE q.rn = 1
AND m.rn = 1
"""
  
Gen_Launch_schema = [
    bigquery.SchemaField("CLIENT", "STRING"),	
    bigquery.SchemaField("BREAKOUT", "STRING"),	
    bigquery.SchemaField("REGION", "STRING"),	
    bigquery.SchemaField("MEASUREMENT", "STRING"),	
    bigquery.SchemaField("CHAIN_GROUP", "STRING"),	
    bigquery.SchemaField("MONTH", "INTEGER"),	
    bigquery.SchemaField("QTY", "FLOAT"),	
    bigquery.SchemaField("FULLAWP", "FLOAT"),	
    bigquery.SchemaField("ING_COST", "FLOAT"),	
    bigquery.SchemaField("client_name", "STRING"),	
    bigquery.SchemaField("timestamp", "STRING")
]

Mac_Constraints_schema = [
    bigquery.SchemaField("CLIENT", "STRING"),	
    bigquery.SchemaField("BREAKOUT", "STRING"),	
    bigquery.SchemaField("REGION", "STRING"),	
    bigquery.SchemaField("MEASUREMENT", "STRING"),	
    bigquery.SchemaField("ELE", "INTEGER"),	
    bigquery.SchemaField("KRG", "INTEGER"),	
    bigquery.SchemaField("WAG", "INTEGER"),	
    bigquery.SchemaField("CAR", "INTEGER"),	
    bigquery.SchemaField("TPS", "INTEGER"),	
    bigquery.SchemaField("CVS", "INTEGER"),	
    bigquery.SchemaField("EPC", "INTEGER"),	
    bigquery.SchemaField("RAD", "INTEGER"),	
    bigquery.SchemaField("WMT", "INTEGER"),	
    bigquery.SchemaField("ABS", "INTEGER"),	
    bigquery.SchemaField("ACH", "INTEGER"),	
    bigquery.SchemaField("AHD", "INTEGER"),	
    bigquery.SchemaField("ART", "INTEGER"),	
    bigquery.SchemaField("GIE", "INTEGER"),	
    bigquery.SchemaField("KIN", "INTEGER"),	
    bigquery.SchemaField("NONPREF_OTH", "INTEGER"),	
    bigquery.SchemaField("PREF_OTH", "INTEGER"),	
    bigquery.SchemaField("MAIL", "INTEGER"),	
    bigquery.SchemaField("MCHOICE", "INTEGER"),	
    bigquery.SchemaField("client_name", "STRING"),	
    bigquery.SchemaField("timestamp", "STRING")	    
]

chain_region_mac_mapping_schema = [
    bigquery.SchemaField("CUSTOMER_ID", "STRING"),
    bigquery.SchemaField("REGION", "STRING"),
    bigquery.SchemaField("MEASUREMENT", "STRING"),
    bigquery.SchemaField("CHAIN_GROUP", "STRING"),
    bigquery.SchemaField("MAC_LIST", "STRING"),
    bigquery.SchemaField("client_name", "STRING"),
    bigquery.SchemaField("timestamp", "STRING"),
] # D-verify

lp_data_schema = [
    bigquery.SchemaField("CLIENT", "STRING"),
    bigquery.SchemaField("BREAKOUT", "STRING"),
    bigquery.SchemaField("REGION", "STRING"),
    bigquery.SchemaField("MEASUREMENT", "STRING"),
    bigquery.SchemaField("GPI", "STRING"),
    bigquery.SchemaField("CHAIN_GROUP", "STRING"),
    bigquery.SchemaField("GO_LIVE", "FLOAT"),
    bigquery.SchemaField("MAC_LIST", "STRING"),
    bigquery.SchemaField("CURRENT_MAC_PRICE", "FLOAT"),
    bigquery.SchemaField("GPI_ONLY", "INTEGER"),
    bigquery.SchemaField("CLAIMS", "FLOAT"),
    bigquery.SchemaField("QTY", "FLOAT"),
    bigquery.SchemaField("FULLAWP_ADJ", "FLOAT"),
    bigquery.SchemaField("PRICE_REIMB", "FLOAT"),
    bigquery.SchemaField("LM_CLAIMS", "FLOAT"),
    bigquery.SchemaField("LM_QTY", "FLOAT"),
    bigquery.SchemaField("LM_FULLAWP_ADJ", "FLOAT"),
    bigquery.SchemaField("LM_PRICE_REIMB", "FLOAT"),
    bigquery.SchemaField("UC_CLAIMS", "FLOAT"),
    bigquery.SchemaField("LM_UC_CLAIMS", "FLOAT"),
    bigquery.SchemaField("CLAIMS_PROJ_LAG", "FLOAT"),
    bigquery.SchemaField("QTY_PROJ_LAG", "FLOAT"),
    bigquery.SchemaField("FULLAWP_ADJ_PROJ_LAG", "FLOAT"),
    bigquery.SchemaField("CLAIMS_PROJ_EOY", "FLOAT"),
    bigquery.SchemaField("QTY_PROJ_EOY", "FLOAT"),
    bigquery.SchemaField("FULLAWP_ADJ_PROJ_EOY", "FLOAT"),
    bigquery.SchemaField("UC_UNIT", "FLOAT"),
    bigquery.SchemaField("UC_UNIT25", "FLOAT"),
    bigquery.SchemaField("CURR_AWP", "FLOAT"),
    bigquery.SchemaField("CURR_AWP_MIN", "FLOAT"),
    bigquery.SchemaField("CURR_AWP_MAX", "FLOAT"),
    bigquery.SchemaField("NDC", "STRING"),
    bigquery.SchemaField("GPI_NDC", "STRING"),
    bigquery.SchemaField("PKG_SZ", "STRING"),
    bigquery.SchemaField("AVG_AWP", "FLOAT"),
    bigquery.SchemaField("BREAKOUT_AWP_MAX", "FLOAT"),
    bigquery.SchemaField("num1026_NDC_PRICE", "FLOAT"),
    bigquery.SchemaField("num1026_GPI_PRICE", "FLOAT"),
    bigquery.SchemaField("MAC1026_UNIT_PRICE", "FLOAT"),
    bigquery.SchemaField("MAC1026_GPI_FLAG", "INTEGER"),
    bigquery.SchemaField("GOODRX_UPPER_LIMIT", "FLOAT"),
    bigquery.SchemaField("PHARMACY_TYPE", "STRING"),
    bigquery.SchemaField("PRICE_MUTABLE", "INTEGER"),
    bigquery.SchemaField("CLIENT_RATE", "FLOAT"),
    bigquery.SchemaField("PHARMACY_RATE", "FLOAT"),
    bigquery.SchemaField("CUSTOMER_ID", "STRING"),
    bigquery.SchemaField("VCML_ID", "STRING"),
    bigquery.SchemaField("MLOR_CD", "STRING"),
    bigquery.SchemaField("GNRCIND", "STRING"),
    bigquery.SchemaField("MIN_UCAMT_QUANTITY", "FLOAT"),
    bigquery.SchemaField("PCT25_UCAMT_UNIT", "FLOAT"),
    bigquery.SchemaField("PCT50_UCAMT_UNIT", "FLOAT"),
    bigquery.SchemaField("MAX_UCAMT_QUANTITY", "FLOAT"),
    bigquery.SchemaField("DISTINCT_UC_PRICES", "FLOAT"),
    bigquery.SchemaField("CLAIMS_IN_CONSTRAINTS", "FLOAT"),
    bigquery.SchemaField("UC_PERCENTILE_CLAIMS", "FLOAT"),
    bigquery.SchemaField("QTY_GT_PCT01", "FLOAT"),
    bigquery.SchemaField("QTY_GT_PCT25", "FLOAT"),
    bigquery.SchemaField("QTY_GT_PCT50", "FLOAT"),
    bigquery.SchemaField("QTY_GT_PCT90", "FLOAT"),
    bigquery.SchemaField("PCT01_UCAMT_UNIT", "FLOAT"),
    bigquery.SchemaField("PCT90_UCAMT_UNIT", "FLOAT"),
    bigquery.SchemaField("VCML_AVG_AWP", "FLOAT"),
    bigquery.SchemaField("VCML_AVG_CALIM_QTY", "FLOAT"),
    bigquery.SchemaField("PRICE_CHANGED_UC", "BOOL"),
    bigquery.SchemaField("MAC_PRICE_UPPER_LIMIT_UC ", "FLOAT"),
    bigquery.SchemaField("PRE_UC_MAC_PRICE", "FLOAT"),
    bigquery.SchemaField("RAISED_PRICE_UC", "BOOL"),
    bigquery.SchemaField("UNC_FRAC", "FLOAT"),
    bigquery.SchemaField("U&C_EBIT", "FLOAT"),
    bigquery.SchemaField("PRICE_REIMB_UNIT", "FLOAT"),
    bigquery.SchemaField("EFF_UNIT_PRICE", "FLOAT"),
    bigquery.SchemaField("MAC_PRICE_UNIT_ADJ", "FLOAT"),
    bigquery.SchemaField("EFF_CAPPED_PRICE", "FLOAT"),
    bigquery.SchemaField("PRICE_REIMB_ADJ", "FLOAT"),
    bigquery.SchemaField("UNC_FRAC_OLD", "FLOAT"),
    bigquery.SchemaField("LM_UNC_FRAC_OLD", "FLOAT"),
    bigquery.SchemaField("CLAIMS_PROJ_EOY_OLDUNC", "FLOAT"),
    bigquery.SchemaField("QTY_PROJ_EOY_OLDUNC", "FLOAT"),
    bigquery.SchemaField("FULLAWP_ADJ_PROJ_EOY_OLDUNC", "FLOAT"),
    bigquery.SchemaField("LM_CLAIMS_OLDUNC", "FLOAT"),
    bigquery.SchemaField("LM_QTY_OLDUNC", "FLOAT"),
    bigquery.SchemaField("LM_FULLAWP_ADJ_OLDUNC", "FLOAT"),
    bigquery.SchemaField("client_name", "STRING"),
    bigquery.SchemaField("timestamp", "STRING")
] #D-rewirtten

mac_lists_schema = [
    bigquery.SchemaField("MAC", "STRING"),
    bigquery.SchemaField("GPI", "STRING"),
    bigquery.SchemaField("NDC", "STRING"),
    bigquery.SchemaField("PRICE", "FLOAT"),
    bigquery.SchemaField("GPI_NDC", "STRING"),
    bigquery.SchemaField("MAC_LIST", "STRING"),
    bigquery.SchemaField("NDC_Count", "INTEGER"),
    bigquery.SchemaField("client_name", "STRING"),
    bigquery.SchemaField("timestamp", "STRING")
] # D-verify

Mac_Mapping_schema = [
    bigquery.SchemaField("CUSTOMER_ID", "STRING"),
    bigquery.SchemaField("MEASUREMENT", "STRING"),
    bigquery.SchemaField("CHAIN_GROUP", "STRING"),
    bigquery.SchemaField("MAC_LIST", "STRING"),
    bigquery.SchemaField("REGION", "STRING"),
    bigquery.SchemaField("client_name", "STRING"),
    bigquery.SchemaField("timestamp", "STRING")
]

Mac_Mapping_VCML_schema = [
    bigquery.SchemaField("Customer_ID", "STRING"),
    bigquery.SchemaField("CHNL_IND", "STRING"),
    bigquery.SchemaField("Base_MAC_List_ID", "STRING"),
    bigquery.SchemaField("VCML_ID", "STRING"),
    bigquery.SchemaField("Scale_Factor", "FLOAT"),
    bigquery.SchemaField("Rec_Effective_Date", "STRING"),
    bigquery.SchemaField("Rec_Expiration_Date", "STRING"),
    bigquery.SchemaField("Reporting_GID", "FLOAT"),
    bigquery.SchemaField("Rec_Curr_Ind", "STRING"),
    bigquery.SchemaField("Rec_Add_USer", "STRING"),
    bigquery.SchemaField("Rec_Add_TS", "STRING"),
    bigquery.SchemaField("Rec_Chg_User", "STRING"),
    bigquery.SchemaField("Rec_Chg_TS", "STRING"),
    bigquery.SchemaField("Unique_Row_ID", "INTEGER"),
    bigquery.SchemaField("Base_Factor", "FLOAT"),
    bigquery.SchemaField("VCML_Reference_WorkTable_GID", "NUMERIC"),
    bigquery.SchemaField("Floor_Ind", "STRING"),
    bigquery.SchemaField("Brand_Generic_CD", "STRING"),
    bigquery.SchemaField("CHNL_IND_NEW", "STRING"),
    bigquery.SchemaField("CHAIN_GROUP", "STRING"),
    bigquery.SchemaField("client_name", "STRING"),
    bigquery.SchemaField("timestamp", "STRING")
]

Pharmacy_approx_coef_schema = [
    bigquery.SchemaField("CLIENT", "STRING"),
    bigquery.SchemaField("CHAIN_GROUP", "STRING"),
    bigquery.SchemaField("SLOPE", "FLOAT"),
    bigquery.SchemaField("INTERCEPT", "FLOAT"),
    bigquery.SchemaField("client_name", "STRING"),
    bigquery.SchemaField("timestamp", "STRING")
]

Pharmacy_approx_XYdata_schema = [
    bigquery.SchemaField("CLIENT", "STRING"),
    bigquery.SchemaField("CHAIN_GROUP", "STRING"),
    bigquery.SchemaField("DOF_MONTH", "FLOAT"),
    bigquery.SchemaField("CLIENT_SPEND", "FLOAT"),
    bigquery.SchemaField("CLIENT_AWP", "FLOAT"),
    bigquery.SchemaField("BREAKOUT", "STRING"),
    bigquery.SchemaField("REGION", "STRING"),
    bigquery.SchemaField("CLIENT_RATE", "FLOAT"),
    bigquery.SchemaField("X_CLIENT_SURPLUS", "FLOAT"),
    bigquery.SchemaField("PHARM_INGREDIENT_COST", "FLOAT"),
    bigquery.SchemaField("PHARM_AWP", "FLOAT"),
    bigquery.SchemaField("PHARM_CLAIM_COUNT", "FLOAT"),
    bigquery.SchemaField("PHARM_RATE", "FLOAT"),
    bigquery.SchemaField("Y_PHARM_SURPLUS", "FLOAT"),
    bigquery.SchemaField("client_name", "STRING"),
    bigquery.SchemaField("timestamp", "STRING")
]

Price_Overrides_schema = [
    bigquery.SchemaField("CLIENT", "STRING"),
    bigquery.SchemaField("REGION", "STRING"),
    bigquery.SchemaField("GPI", "INTEGER"),
    bigquery.SchemaField("DATE_ADDED", "STRING"),
    bigquery.SchemaField("SOURCE", "STRING"),
    bigquery.SchemaField("client_name", "STRING"),
    bigquery.SchemaField("timestamp", "STRING")
]

YTD_Pharmacy_Performance_schema = [
    bigquery.SchemaField("CLIENT", "STRING"),
    bigquery.SchemaField("REGION", "STRING"),
    bigquery.SchemaField("BREAKOUT", "STRING"),
    bigquery.SchemaField("CHAIN_GROUP", "STRING"),
    bigquery.SchemaField("DOF_MONTH", "INTEGER"),
    bigquery.SchemaField("INGREDIENT_COST", "FLOAT"),
    bigquery.SchemaField("AWP", "FLOAT"),
    bigquery.SchemaField("CLAIM_COUNT", "FLOAT"),
    bigquery.SchemaField("client_name", "STRING"),
    bigquery.SchemaField("timestamp", "STRING")
]


awp_spend_perf_schema = [
    bigquery.SchemaField("CUSTOMER_ID", "STRING"),
    bigquery.SchemaField("CLIENT", "STRING"),
    bigquery.SchemaField("ENTITY", "STRING"),
    bigquery.SchemaField("ENTITY_TYPE", "STRING"),
    bigquery.SchemaField("BREAKOUT", "STRING"),
    bigquery.SchemaField("CHAIN_GROUP", "STRING"),
    bigquery.SchemaField("CLIENT_OR_PHARM", "STRING"),
    bigquery.SchemaField("FULLAWP_ADJ", "FLOAT64"),
    bigquery.SchemaField("FULLAWP_ADJ_PROJ_LAG", "FLOAT64"),
    bigquery.SchemaField("FULLAWP_ADJ_PROJ_EOY", "FLOAT64"),
    bigquery.SchemaField("PRICE_REIMB", "FLOAT64"),
    bigquery.SchemaField("LAG_REIMB", "FLOAT64"),
    bigquery.SchemaField("Old_Price_Effective_Reimb_Proj_EOY", "FLOAT64"),
    bigquery.SchemaField("Price_Effective_Reimb_Proj", "FLOAT64"),
    bigquery.SchemaField("GEN_LAG_AWP", "FLOAT64"),
    bigquery.SchemaField("GEN_LAG_ING_COST", "FLOAT64"),
    bigquery.SchemaField("GEN_EOY_AWP", "FLOAT64"),
    bigquery.SchemaField("GEN_EOY_ING_COST", "FLOAT64"),
    bigquery.SchemaField("Pre_existing_Perf", "FLOAT64"),
    bigquery.SchemaField("Model_Perf", "FLOAT64"),
    bigquery.SchemaField("Proj_Spend_Do_Nothing", "FLOAT64"),
    bigquery.SchemaField("Proj_Spend_Model", "FLOAT64"),
    bigquery.SchemaField("Increase_in_Spend", "FLOAT64"),
    bigquery.SchemaField("Increase_in_Reimb", "FLOAT64"),
    bigquery.SchemaField("Total_Ann_AWP", "FLOAT64"),
    bigquery.SchemaField("GER_Do_Nothing", "FLOAT64"),
    bigquery.SchemaField("GER_Model", "FLOAT64"),
    bigquery.SchemaField("GER_Target", "FLOAT64"),
    bigquery.SchemaField("ALGO_RUN_DATE", "TIMESTAMP"),
    #bigquery.SchemaField("REC_CURR_IND", "STRING"),
    bigquery.SchemaField("CLIENT_TYPE", "STRING"),
    bigquery.SchemaField("UNC_OPT", "BOOLEAN"),
    bigquery.SchemaField("GOODRX_OPT", "BOOLEAN"),
    bigquery.SchemaField("GO_LIVE", "DATE"),
    bigquery.SchemaField("DATA_ID", "STRING"),
    bigquery.SchemaField("TIERED_PRICE_LIM", "BOOLEAN"),
    bigquery.SchemaField("RUN_TYPE", "STRING"),
    bigquery.SchemaField("AT_RUN_ID", "STRING"),
    bigquery.SchemaField("Run_rate_do_nothing","FLOAT64"),
    bigquery.SchemaField("Run_rate_w_changes","FLOAT64"),
    bigquery.SchemaField("IA_CODENAME", "STRING"),
    bigquery.SchemaField("Pre_existing_Perf_Generic", "FLOAT64"),
    bigquery.SchemaField("Model_Perf_Generic", "FLOAT64"),
    bigquery.SchemaField("YTD_Perf_Generic", "FLOAT64"),
    bigquery.SchemaField("LEAKAGE_PRE", "FLOAT64"),
    bigquery.SchemaField("LEAKAGE_POST", "FLOAT64"),
    bigquery.SchemaField("LEAKAGE_AVOID", "FLOAT64")
]

ytd_surplus_monthly_schema = [
    bigquery.SchemaField("CUSTOMER_ID", "STRING"),
    bigquery.SchemaField("CLIENT", "STRING"),
    bigquery.SchemaField("BREAKOUT", "STRING"),
    bigquery.SchemaField("MONTH", "FLOAT64"),
    bigquery.SchemaField("AWP", "FLOAT64"),
    bigquery.SchemaField("SPEND", "FLOAT64"),
    bigquery.SchemaField("SURPLUS", "FLOAT64"),
    bigquery.SchemaField("ALGO_RUN_DATE", "TIMESTAMP"),
    bigquery.SchemaField("CLIENT_TYPE", "STRING"),
    bigquery.SchemaField("TIERED_PRICE_LIM", "BOOLEAN"),
    bigquery.SchemaField("UNC_OPT", "BOOLEAN"),
    bigquery.SchemaField("GOODRX_OPT", "BOOLEAN"),
    bigquery.SchemaField("GO_LIVE", "DATE"),
    bigquery.SchemaField("DATA_ID", "STRING"),
    bigquery.SchemaField("RUN_TYPE", "STRING"),
    bigquery.SchemaField("AT_RUN_ID", "INTEGER"),
    bigquery.SchemaField("client_name", "STRING"),
    bigquery.SchemaField("timestamp", "STRING")
]

psot_schema = [
    bigquery.SchemaField("RUN_ID", "STRING", mode="REQUIRED"),
    bigquery.SchemaField("SIX_TO_FIFTEEN_DOLLARS", "FLOAT"),
    bigquery.SchemaField("FIFTEEN_TO_TWENTYFIVE_DOLLARS", "FLOAT"),
    bigquery.SchemaField("TWENTYFIVE_TO_FIFTY_DOLLARS", "FLOAT"),
    bigquery.SchemaField("FIFTY_TO_HUNDRED_DOLLARS", "FLOAT"),
    bigquery.SchemaField("HUNDRED_DOLLARS_AND_ABOVE", "FLOAT"),
    bigquery.SchemaField("CHANGE_IN_CAPPED", "FLOAT"),
    bigquery.SchemaField("CHANGE_IN_PSAO", "FLOAT"),
    bigquery.SchemaField("CPI_CVS", "FLOAT"),
    bigquery.SchemaField("CPI_KRG", "FLOAT"),
    bigquery.SchemaField("CPI_RAD", "FLOAT"),
    bigquery.SchemaField("CPI_WAG", "FLOAT"),
    bigquery.SchemaField("CPI_WMT", "FLOAT"),
    bigquery.SchemaField("CPI_TOTAL", "FLOAT"),
    bigquery.SchemaField("OFFSET", "BOOLEAN"),
    bigquery.SchemaField("RETAIL_BLENDED_SURPLUS_AMT", "FLOAT"),
    bigquery.SchemaField("client_name", "STRING"),
    bigquery.SchemaField("timestamp", "STRING"),
    bigquery.SchemaField("Dm_Begn_Dtm", "TIMESTAMP"),
    bigquery.SchemaField("AT_RUN_ID", "STRING"),
    bigquery.SchemaField("MAIL_BLENDED_SURPLUS_AMT", "FLOAT"),
    bigquery.SchemaField("RETAIL_BLENDED_SURPLUS_DO_NOTHING", "FLOAT"),
    bigquery.SchemaField("MAIL_BLENDED_SURPLUS_DO_NOTHING", "FLOAT"),
    bigquery.SchemaField("CPI_CVS_SP", "FLOAT"),
    bigquery.SchemaField("CHANGE_IN_NON_CAPPED", "FLOAT")
]
