CREATE OR REPLACE TABLE `pbm-mac-lp-prod-ai.ds_sandbox.nadac_test_eva` AS (
WITH
  MAC_MAPPING AS (
  SELECT
    *
  FROM
    `pbm-mac-lp-prod-ai.ds_sandbox.grx_mac_mapping`
  WHERE RUN_DATE = (SELECT MAX(RUN_DATE) FROM `pbm-mac-lp-prod-ai.ds_sandbox.grx_mac_mapping`)
),

NONSPCLT_CLAIMS AS (
  SELECT
    *
  FROM (
    SELECT
      CUSTOMER_ID,
      GPI,
      NDC
    FROM
      (SELECT * FROM `pbm-mac-lp-prod-de.ds_pro_lp.combined_daily_totals`
       UNION DISTINCT 
       SELECT * FROM `anbc-pss-prod.de_gms_enrv_pss_prod.gms_aetna_standard_combined_daily_totals`)
    WHERE
      CUSTOMER_ID IN (
      SELECT
        DISTINCT CUSTOMER_ID
      FROM
        MAC_MAPPING)
        AND GNRCIND = 'TRUE'
        AND customer_id not in (SELECT DISTINCT CUSTOMER_ID FROM `anbc-pss-prod.de_gms_enrv_pss_prod.gms_truecost_taxonomy_final` WHERE current_date() BETWEEN Contract_Eff_Dt and Original_Contract_Exprn_Dt)
    GROUP BY
      CUSTOMER_ID,
      GPI,
      NDC)),

TRUECOST_CLAIMS AS (
  SELECT
    *
  FROM (
    SELECT
      CUSTOMER_ID,
      GPI,
      NDC
    FROM
      `pbm-mac-lp-prod-de.ds_pro_lp.combined_daily_totals_TC`
    WHERE
      CUSTOMER_ID IN (
      SELECT
        DISTINCT CUSTOMER_ID
      FROM
        MAC_MAPPING)
    AND BG_FLAG = 'G'
    AND customer_id in (SELECT DISTINCT CUSTOMER_ID FROM `anbc-pss-prod.de_gms_enrv_pss_prod.gms_truecost_taxonomy_final` WHERE current_date() BETWEEN Contract_Eff_Dt and Original_Contract_Exprn_Dt)
    GROUP BY
      CUSTOMER_ID,
      GPI,
      NDC)),

SPCLT_CLAIMS AS (SELECT 
  CLCODE AS CUSTOMER_ID, 
  clms.GPI, 
  clms.NDC, 
FROM `anbc-pss-prod.de_gms_enrv_pss_prod.gms_spec_client_pharma_clms` clms
WHERE
      CLCODE IN (
      SELECT
        DISTINCT CUSTOMER_ID
      FROM
        MAC_MAPPING)
AND CLCODE not in (SELECT DISTINCT CUSTOMER_ID FROM `anbc-pss-prod.de_gms_enrv_pss_prod.gms_truecost_taxonomy_final` WHERE current_date() BETWEEN Contract_Eff_Dt and Original_Contract_Exprn_Dt)

GROUP BY 1,2,3
),

CLAIMS AS (SELECT * FROM NONSPCLT_CLAIMS UNION ALL SELECT * FROM SPCLT_CLAIMS UNION ALL SELECT * FROM TRUECOST_CLAIMS),

claims_mapped AS (
         SELECT * FROM
            CLAIMS),


NADAC_WAC AS (
  SELECT CUSTOMER_ID, GPI, NDC, COALESCE(NADAC, WAC) AS NADAC_UNIT_PRICE FROM
(SELECT * FROM `pbm-mac-lp-prod-de.ds_pro_lp.nadac_wac_price`
where CUSTOMER_ID not in (SELECT DISTINCT CUSTOMER_ID FROM `anbc-pss-prod.de_gms_enrv_pss_prod.gms_truecost_taxonomy_final` WHERE current_date() BETWEEN Contract_Eff_Dt and Original_Contract_Exprn_Dt) 
UNION ALL
SELECT * FROM `pbm-mac-lp-prod-de.ds_pro_lp.nadac_wac_price_TC`
where CUSTOMER_ID in (SELECT DISTINCT CUSTOMER_ID FROM `anbc-pss-prod.de_gms_enrv_pss_prod.gms_truecost_taxonomy_final` WHERE current_date() BETWEEN Contract_Eff_Dt and Original_Contract_Exprn_Dt) )
WHERE BRND_GNRC_CD = 'GNRC'
),

NADAC_ALL_CLIENTS AS (
SELECT distinct GPI, NDC,
    PERCENTILE_DISC(NADAC_UNIT_PRICE, 0.50) OVER(PARTITION BY GPI, NDC) AS NADAC_UNIT_PRICE
FROM NADAC_WAC
)

-- NADAC_TABLE AS (
  
SELECT claims_filtered.*, 
  COALESCE(NADAC_NDC.NADAC_UNIT_PRICE, NADAC_GPI.NADAC_UNIT_PRICE, NADAC_NDC_ALL.NADAC_UNIT_PRICE, NADAC_GPI_ALL.NADAC_UNIT_PRICE) AS NADAC_UNIT_PRICE 
  -- NADAC_NDC.NADAC_UNIT_PRICE,
  -- NADAC_GPI.NADAC_UNIT_PRICE,
  FROM (SELECT DISTINCT CUSTOMER_ID, GPI, NDC FROM claims_mapped) claims_filtered
LEFT JOIN (SELECT CUSTOMER_ID, GPI, NDC, NADAC_UNIT_PRICE AS NADAC_UNIT_PRICE
              FROM NADAC_WAC
              WHERE NDC not like '%*%') NADAC_NDC
    ON claims_filtered.GPI = NADAC_NDC.GPI
    AND claims_filtered.NDC = NADAC_NDC.NDC
    AND claims_filtered.CUSTOMER_ID = NADAC_NDC.CUSTOMER_ID
  LEFT JOIN (SELECT CUSTOMER_ID, GPI, NADAC_UNIT_PRICE AS NADAC_UNIT_PRICE
              FROM NADAC_WAC
              WHERE NDC like '%*%') NADAC_GPI
    ON claims_filtered.GPI = NADAC_GPI.GPI
    AND claims_filtered.CUSTOMER_ID = NADAC_GPI.CUSTOMER_ID
  LEFT JOIN (SELECT GPI, NDC, NADAC_UNIT_PRICE
              FROM NADAC_ALL_CLIENTS
              WHERE NDC not like '%*%') NADAC_NDC_ALL
    ON claims_filtered.GPI = NADAC_NDC_ALL.GPI
    AND claims_filtered.NDC = NADAC_NDC_ALL.NDC
  LEFT JOIN (SELECT GPI, NADAC_UNIT_PRICE
              FROM NADAC_ALL_CLIENTS
              WHERE NDC like '%*%') NADAC_GPI_ALL
    ON claims_filtered.GPI = NADAC_GPI_ALL.GPI
    );
-- INSERT INTO `pbm-mac-lp-prod-ai.ds_sandbox.vcml_grx_mac_archive` 
-- (
WITH
  MAC_MAPPING AS (
  SELECT
    *
  FROM
    `pbm-mac-lp-prod-ai.ds_sandbox.grx_mac_mapping`
  WHERE RUN_DATE = (SELECT MAX(RUN_DATE) FROM `pbm-mac-lp-prod-ai.ds_sandbox.grx_mac_mapping`)
),

NONSPCLT_CLAIMS AS (
  SELECT
    *,
    CASE
      WHEN TOT_QTY > 0 THEN TOT_AWP/TOT_QTY
    ELSE
    0
  END
    AS AWP_UNIT,
  FROM (
    SELECT
      CUSTOMER_ID,
      REGION,
      MEASUREMENT,
      CHAIN_SUBGROUP,
      CHAIN_GROUP,
      GPI,
      NDC,
      -- CASE WHEN GNRCIND = 'TRUE' THEN 'G' ELSE 'B' END AS BG_FLAG,
      SUM(CLAIMS_distinct) AS CLAIMS,
      SUM(AWP) AS TOT_AWP,
      SUM(QTY) AS TOT_QTY,
      SUM(SPEND) AS TOT_SPEND,
      SUM(DISP_FEE) AS TOT_DISP_FEE,
      SUM(MEMBER_COST) AS TOT_MBR_COST
    FROM
      (SELECT * FROM `pbm-mac-lp-prod-de.ds_pro_lp.combined_daily_totals`
       UNION DISTINCT 
       SELECT * FROM `anbc-pss-prod.de_gms_enrv_pss_prod.gms_aetna_standard_combined_daily_totals`)
    WHERE
      CUSTOMER_ID IN (
      SELECT
        DISTINCT CUSTOMER_ID
      FROM
        MAC_MAPPING)
        AND GNRCIND = 'TRUE'
        AND customer_id not in (SELECT DISTINCT CUSTOMER_ID FROM `anbc-pss-prod.de_gms_enrv_pss_prod.gms_truecost_taxonomy_final` WHERE current_date() BETWEEN Contract_Eff_Dt and Original_Contract_Exprn_Dt)
    GROUP BY
      CUSTOMER_ID,
      REGION, 
      MEASUREMENT,
      CHAIN_SUBGROUP,
      CHAIN_GROUP,
      GPI,
      NDC)
      
  WHERE
    TOT_AWP > 0
    ),

TRUECOST_CLAIMS AS (
  SELECT
    *,
    CASE
      WHEN TOT_QTY > 0 THEN TOT_AWP/TOT_QTY
    ELSE
    0
  END
    AS AWP_UNIT,
  FROM (
    SELECT
      CUSTOMER_ID,
      REGION,
      MEASUREMENT,
      CHAIN_SUBGROUP,
      CHAIN_GROUP,
      GPI,
      NDC,
      -- BG_FLAG,
      SUM(CLAIMS_distinct) AS CLAIMS,
      SUM(AWP) AS TOT_AWP,
      SUM(QTY) AS TOT_QTY,
      SUM(SPEND) AS TOT_SPEND,
      SUM(DISP_FEE) AS TOT_DISP_FEE,
      SUM(MEMBER_COST) AS TOT_MBR_COST
    FROM
      `pbm-mac-lp-prod-de.ds_pro_lp.combined_daily_totals_TC`
    WHERE
      CUSTOMER_ID IN (
      SELECT
        DISTINCT CUSTOMER_ID
      FROM
        MAC_MAPPING)
    AND BG_FLAG = 'G'
    AND customer_id in (SELECT DISTINCT CUSTOMER_ID FROM `anbc-pss-prod.de_gms_enrv_pss_prod.gms_truecost_taxonomy_final` WHERE current_date() BETWEEN Contract_Eff_Dt and Original_Contract_Exprn_Dt)
    GROUP BY
      CUSTOMER_ID,
      REGION, 
      MEASUREMENT,
      CHAIN_SUBGROUP,
      CHAIN_GROUP,
      GPI,
      NDC)
  WHERE
    TOT_AWP > 0),

awp_hist AS (SELECT GPI, NDC, MIN(DRUG_PRICE_AT) AS AWP_UNIT
FROM `pbm-mac-lp-prod-de.ds_pro_lp.awp_history_table`
GROUP BY 1,2),


SPCLT_CLAIMS AS (SELECT 
  CLCODE AS CUSTOMER_ID, 
  CLCODE AS REGION,
  'SPCLT' AS MEASUREMENT,
  'CMK_SPECIALTY' AS CHAIN_SUBGROUP,
  'CMK_SPECIALTY' AS CHAIN_GROUP,
  clms.GPI, 
  clms.NDC, 
  -- 'G' AS BG_FLAG,
  SUM(ROUND(CAST(claim_count AS FLOAT64),0)) AS CLAIMS,
  SUM(ROUND(CAST(AWP AS FLOAT64),4)) AS TOT_AWP,
  SUM(ROUND(CAST(QUANTITY AS FLOAT64),4)) AS TOT_QTY,
  SUM(ROUND(CAST(INGRED_COST AS FLOAT64),4)) AS TOT_SPEND,
  0 AS TOT_DISP_FEE,
  SUM(ROUND(CAST(member_cost AS FLOAT64),4)) AS TOT_MBR_COST,
  MIN(AWP_UNIT) AS AWP_UNIT
FROM `anbc-pss-prod.de_gms_enrv_pss_prod.gms_spec_client_pharma_clms` clms
LEFT JOIN awp_hist 
 ON awp_hist.GPI = clms.GPI
 AND awp_hist.NDC = clms.NDC
WHERE
      CLCODE IN (
      SELECT
        DISTINCT CUSTOMER_ID
      FROM
        MAC_MAPPING)
AND CLCODE not in (SELECT DISTINCT CUSTOMER_ID FROM `anbc-pss-prod.de_gms_enrv_pss_prod.gms_truecost_taxonomy_final` WHERE current_date() BETWEEN Contract_Eff_Dt and Original_Contract_Exprn_Dt)

GROUP BY 1,2,3,4,5,6,7
),

CLAIMS AS (SELECT * FROM NONSPCLT_CLAIMS UNION ALL SELECT * FROM SPCLT_CLAIMS UNION ALL SELECT * FROM TRUECOST_CLAIMS),

client_info AS (
  SELECT
      CUSTOMER_ID,
      CLIENT_NAME,
      CLIENT_TYPE,
      guarantee_category,
      CASE
        WHEN guarantee_category LIKE "MedD%" THEN "MEDD"
      ELSE
      'COMMERCIAL'
    END
      AS MEDD_COMM
    FROM
      (SELECT       CUSTOMER_ID,
      CLIENT_NAME,
      CLIENT_TYPE,
      guarantee_category FROM (SELECT CUSTOMER_ID,
            CLIENT_NAME,
            CLIENT_TYPE,
            'TrueCost' AS guarantee_category
            FROM  `anbc-pss-prod.de_gms_enrv_pss_prod.gms_truecost_taxonomy_final`
            WHERE current_date() BETWEEN Contract_Eff_Dt and Original_Contract_Exprn_Dt
            UNION DISTINCT 
            SELECT
            CUSTOMER_ID,
                  CLIENT_NAME,
                  CLIENT_TYPE,
                  guarantee_category FROM `pbm-mac-lp-prod-de.landing.GER_OPT_TAXONOMY_FINAL`
                  WHERE CUSTOMER_ID not in (SELECT DISTINCT CUSTOMER_ID FROM `anbc-pss-prod.de_gms_enrv_pss_prod.gms_truecost_taxonomy_final` WHERE current_date() BETWEEN Contract_Eff_Dt and Original_Contract_Exprn_Dt))
      UNION DISTINCT 
      SELECT       CUSTOMER_ID,
      CLIENT_NAME,
      CLIENT_TYPE,
      guarantee_category FROM `anbc-pss-prod.de_gms_enrv_pss_prod.gms_aetna_ger_opt_taxonomy_final`)
    WHERE
      CUSTOMER_ID IN (
      SELECT
        DISTINCT CUSTOMER_ID
      FROM
        MAC_MAPPING)
),

full_mac AS (
  SELECT GPI,
      NDC,
      MAC_LIST,
      MAX(MAC) AS MAC,
    AVG(PRICE) AS PRICE 
    FROM (SELECT
    DISTINCT
      GPI,
      NDC,
      MAC_LIST,
      MAC,
      -- 'G' AS BG_FLAG,
      PRICE
      FROM
      (SELECT * FROM `pbm-mac-lp-prod-de.ds_pro_lp.mac_list_TC` 
        UNION DISTINCT 
        SELECT * FROM `anbc-pss-prod.de_gms_enrv_pss_prod.gms_aetna_mac_list`
        )
    UNION DISTINCT
    SELECT
      GPI,
      NDC,
      MAC_LIST,
      MAC,
      -- BG_FLAG,
      PRICE
    FROM
      (SELECT GPI,
      NDC,
      MAC_LIST,
      MAC,
      -- 'G' AS BG_FLAG,
      GENERIC_PRICE AS PRICE FROM `pbm-mac-lp-prod-de.ds_pro_lp.tru_price_list_TC` 
      where GENERIC_PRICE > 0
        -- UNION DISTINCT 
--         SELECT GPI,
--       NDC,
--       MAC_LIST,
--       MAC,
--       'B' AS BG_FLAG,
--       BRAND_PRICE AS PRICE FROM `pbm-mac-lp-prod-de.ds_pro_lp.tru_price_list_TC` 
--       where BRAND_PRICE >0
        )
    )
    GROUP BY GPI, NDC, MAC_LIST
),

mac_ndc AS (SELECT
            GPI,
            NDC,
            MAC_LIST,
            PRICE
          FROM
            full_mac),

mac_gpi AS (SELECT
          GPI,
          NDC,
          MAC_LIST,
          PRICE
        FROM
          full_mac
        WHERE
          NDC LIKE "%*%"),

claims_mapped AS (
         SELECT * EXCEPT(RUN_DATE), MAX(RUN_DATE) OVER() AS RUN_DATE,
         FROM( 
          SELECT
            CLAIMS.*,
            MAC_LIST,
            RUN_DATE
          FROM
            CLAIMS
          LEFT JOIN
            MAC_MAPPING
          ON
            CLAIMS.CUSTOMER_ID = MAC_MAPPING.CUSTOMER_ID
            AND CLAIMS.MEASUREMENT = MAC_MAPPING.MEASUREMENT
            AND CLAIMS.CHAIN_SUBGROUP = MAC_MAPPING.CHAIN_SUBGROUP
          WHERE
            (MAC_MAPPING.CLIENT_VCML_ID IS NOT NULL OR CLAIMS.CHAIN_GROUP = 'CMK_SPECIALTY' or claims.MEASUREMENT like 'S%'))),

drug_data_ndc AS (
  SELECT
      gpi_nm,
      GPI,
      NDC,
      drug_multi_src_cd,
      IFNULL(spclt_cd, 'N') AS spclt_drug_ind,
      drg.eff_dt,
      drg.exprn_dt
  FROM (
      SELECT
        gpi_nm,
        gpi_cd AS GPI,
        drug_id AS NDC,
        drug_multi_src_cd,
        eff_dt,
        exprn_dt
      FROM `anbc-pss-prod.fdm_cnfv_pss_prod.gms_v_drug_denorm_gcp`
      WHERE gpi_cd  NOT LIKE "% %"
        AND gpi_nm != " "
        ) drg
  LEFT JOIN (
      SELECT
        gpi_cd, 
        drug_id,
        eff_dt,
        exprn_dt,
        'Y' AS spclt_cd
      FROM `anbc-pss-prod.fdm_cnfv_pss_prod.gms_v_drug_denorm_gcp`
      WHERE spclt_ctgry_cd IN ('B','L','H','T','V','R','P')
        AND CONCAT(spclt_offr_cd, spclt_chrct_cd) IN ('101','001','002','003','004','007','008','009','011')
        AND UPPER(spclt_drug_ind) = 'Y'
        AND gpi_cd NOT IN ('',
                      '6240502500E520',
                      '21101040102110',
                      '86101070002005',
                      '12353077100120',
                      '3950004010E520')) spl
  ON drg.gpi = spl.gpi_cd
      AND drg.ndc = spl.drug_id
      AND drg.eff_dt = spl.eff_dt
      AND drg.exprn_dt = spl.exprn_dt
),

spclt AS (
  SELECT
      GPI,
      MIN(gpi_nm) AS gpi_nm,
      MAX(spclt_drug_ind) AS SPCLT_DRUG_IND,
    FROM drug_data_ndc
    WHERE CURRENT_DATE('America/New_York') BETWEEN eff_dt AND exprn_dt
    GROUP BY 1
),

vcml_ref_spec AS (
SELECT CUSTOMER_ID, vcml_id AS MAC_LIST ,SUBSTRING(VCML_ID, 4+LENGTH(CUSTOMER_ID)) AS VCML_IND
FROM (SELECT customer_id, vcml_id  
FROM `anbc-pss-prod.fdm_cnfv_pss_prod.gms_v_ger_opt_vcml_reference_lp`
WHERE CURRENT_DATE() BETWEEN rec_effective_date AND rec_expiration_date
AND rec_curr_ind = 'Y'
UNION DISTINCT 
SELECT customer_id, vcml_id  
FROM  `anbc-pss-prod.de_gms_enrv_pss_prod.gms_aetna_vcml_reference`
WHERE CURRENT_DATE() BETWEEN rec_effective_date AND rec_expiration_date
AND rec_curr_ind = 'Y')
WHERE SUBSTRING(VCML_ID, 4+LENGTH(CUSTOMER_ID)) IN ('41','4','1','SX','S3','S9')),

allmacs_spec AS (
SELECT * FROM (
SELECT CUSTOMER_ID, GPI, NDC, vcml_ref_spec.VCML_IND, PRICE AS CURRENT_MAC_PRICE 
FROM full_mac
INNER JOIN vcml_ref_spec
ON full_mac.MAC = vcml_ref_spec.MAC_LIST)
PIVOT(MIN(CURRENT_MAC_PRICE) FOR VCML_IND IN ('SX','S3','S9','41','4','1'))),

curr_mac_spec AS (
SELECT Spec_CLAIMS.*, 
COALESCE(allmacs_spec.SX,allmacs_spec.S3,allmacs_spec.S9,LEAST(allmacs_spec.41,allmacs_spec.4,allmacs_spec.1),allmacs_gpi.SX,allmacs_gpi.S3,allmacs_gpi.S9,LEAST(allmacs_gpi.41,allmacs_gpi.4,allmacs_gpi.1)) AS CURR_MAC_PRC,
FROM (SELECT * FROM SPCLT_CLAIMS
    ) Spec_CLAIMS
LEFT JOIN allmacs_spec
ON Spec_CLAIMS.CUSTOMER_ID = allmacs_spec.CUSTOMER_ID
AND Spec_CLAIMS.GPI = allmacs_spec.GPI
AND Spec_CLAIMS.NDC = allmacs_spec.NDC
LEFT JOIN (SELECT * FROM allmacs_spec WHERE NDC LIKE "%*%") allmacs_gpi
ON Spec_CLAIMS.CUSTOMER_ID = allmacs_gpi.CUSTOMER_ID
AND Spec_CLAIMS.GPI = allmacs_gpi.GPI),

COST_PLUS_TABLE AS (
SELECT
  GPI_CD AS GPI,
  ROUND(MAX(CAST(REPLACE(REPLACE(TRIM(_unit_price_with_shipping_), '$', ''), ',', '') AS FLOAT64)),4) AS MCCP_UNIT_PRICE,
  -- CASE WHEN BRND_GNRC_CD = 'GNRC' THEN 'G'
  -- ELSE 'B' END AS BG_FLAG
  FROM
  `pbm-mac-lp-prod-ai.ds_sandbox.costplus_jan_2025`
    WHERE BRND_GNRC_CD = 'GNRC'
-- AND LOAD_DT = (SELECT MAX(LOAD_DT) FROM  `pbm-mac-lp-prod-de.publish.COSTPLUS_DRUG_PRICE`)
GROUP BY GPI_CD
),

NADAC_TABLE AS (
SELECT * FROM `pbm-mac-lp-prod-ai.ds_sandbox.nadac_test_eva`
    ),

COST_VANTAGE_TABLE AS (
SELECT 
GPI14 AS GPI,
ROUND(CAST(Cost_Fee_per_Unit AS FLOAT64),4) AS CV_UNIT_PRICE 
FROM 
  `anbc-pss-prod.fdm_cnfv_pss_prod.gms_gms_pgm_prm_68_cst_vantage_cost_file`
WHERE
CURRENT_DATE() BETWEEN eff_dt AND thru_dt
QUALIFY ROW_NUMBER() OVER (PARTITION BY GPI14 ORDER BY EFF_DT DESC) = 1
),

final_table AS (
SELECT * FROM (SELECT
  claims_mac.CUSTOMER_ID,
  claims_mac.CLIENT_NAME,
  claims_mac.CLIENT_TYPE,
  claims_mac.guarantee_category,
  claims_mac.MEDD_COMM,
  claims_mac.REGION,
  claims_mac.MEASUREMENT,
  claims_mac.CHAIN_SUBGROUP AS CHAIN_GROUP,
  claims_mac.GPI,
  claims_mac.NDC,
  claims_mac.CLAIMS,
  CAST(claims_mac.TOT_AWP AS NUMERIC) AS TOT_AWP,
  CAST(claims_mac.TOT_QTY AS NUMERIC) AS TOT_QTY,
  CASE WHEN TRIM(claims_mac.guarantee_category) = 'TrueCost'
    THEN CAST(claims_mac.TOT_SPEND AS NUMERIC) + CAST(claims_mac.TOT_DISP_FEE AS NUMERIC) 
    ELSE CAST(claims_mac.TOT_SPEND AS NUMERIC) END AS TOT_SPEND,
  CAST(claims_mac.TOT_MBR_COST AS NUMERIC) AS TOT_MBR_COST,
  CAST(claims_mac.AWP_UNIT AS NUMERIC) AS AWP_UNIT,
  CASE WHEN claims_mac.CHAIN_GROUP = 'CMK_SPECIALTY' or claims_mac.MEASUREMENT = 'S%' THEN CONCAT(claims_mac.CUSTOMER_ID,"SX") ELSE claims_mac.MAC_LIST END AS MAC_LIST, 
  ROUND(COALESCE(grx_raw.weighted_mean,GRX_PRICE_BOB_CHAIN_MEAS, GRX_PRICE_BOB_CHAIN, 
                  grx_old.GOODRX_UNIT_PRICE, grx_old_bob.GOODRX_PRICE_BOB_CHAIN, 
                  grx_spec.GOODRX_UNIT_PRICE,grx_spec_bob.GOODRX_PRICE_BOB,
                  goodrx_cost_unit, goodrx_cost_unit_bob),4) AS VCML_GRX_PRICE,
  ROUND(CASE WHEN TRIM(claims_mac.guarantee_category) = 'TrueCost'
          THEN (COALESCE(claims_mac.CURR_MAC_PRC, 
          CASE WHEN claims_mac.CURR_MAC_PRC IS NULL THEN
            CASE WHEN claims_mac.MEASUREMENT NOT LIKE 'S%' THEN 
              CASE WHEN claims_mac.MEDD_COMM = "MEDD" THEN (1-0.55)*AWP_UNIT ELSE (1-0.43)*AWP_UNIT  END
            WHEN claims_mac.MEASUREMENT LIKE 'S%' THEN CAST(claims_mac.TOT_SPEND AS NUMERIC)*CAST(claims_mac.AWP_UNIT AS NUMERIC)/CAST(claims_mac.TOT_AWP AS NUMERIC) END END) * claims_mac.TOT_QTY + claims_mac.TOT_DISP_FEE)/claims_mac.TOT_QTY
        ELSE COALESCE(claims_mac.CURR_MAC_PRC,
        CASE WHEN claims_mac.CURR_MAC_PRC IS NULL THEN 
          CASE WHEN claims_mac.MEASUREMENT NOT LIKE 'S%' THEN
            CASE WHEN claims_mac.MEDD_COMM = "MEDD" THEN (1-0.55)*AWP_UNIT ELSE (1-0.43)*AWP_UNIT END
          WHEN claims_mac.MEASUREMENT LIKE 'S%' THEN CAST(claims_mac.TOT_SPEND AS NUMERIC)*CAST(claims_mac.AWP_UNIT AS NUMERIC)/CAST(claims_mac.TOT_AWP AS NUMERIC) END END)
    END ,4) AS CURR_MAC_PRC,
  spclt.SPCLT_DRUG_IND,
  claims_mac.RUN_DATE,
  claims_mac.PRICE_MUTABLE,
  COST_PLUS_TABLE.MCCP_UNIT_PRICE,
  spclt.gpi_nm AS GPI_NM,
  COALESCE(clms.COSTSAVER_CLIENT, "N") AS COSTSAVER_CLIENT,
  CASE WHEN claims_mac.CURR_MAC_PRC IS NULL THEN "NON_MAC" ELSE "MAC" END AS MAC_NONMAC,
  CASE WHEN claims_mac.MEASUREMENT = 'R30' THEN (NADAC_UNIT_PRICE * claims_mac.TOT_QTY + (8-1.85) * claims_mac.CLAIMS)/claims_mac.TOT_QTY
       WHEN claims_mac.MEASUREMENT = 'R90' THEN (NADAC_UNIT_PRICE * claims_mac.TOT_QTY + (11.95-1.85) * claims_mac.CLAIMS)/claims_mac.TOT_QTY
       WHEN claims_mac.MEASUREMENT = 'M30' THEN (NADAC_UNIT_PRICE * claims_mac.TOT_QTY + (18.46-1.85) * claims_mac.CLAIMS)/claims_mac.TOT_QTY
       ELSE NADAC_UNIT_PRICE END AS NADAC_UNIT_PRICE,
  CASE WHEN claims_mac.MEASUREMENT = 'R30' THEN (COST_VANTAGE_TABLE.CV_UNIT_PRICE * claims_mac.TOT_QTY + (8-1.85) * claims_mac.CLAIMS)/claims_mac.TOT_QTY
       WHEN claims_mac.MEASUREMENT = 'R90' THEN (COST_VANTAGE_TABLE.CV_UNIT_PRICE * claims_mac.TOT_QTY + (11.95-1.85) * claims_mac.CLAIMS)/claims_mac.TOT_QTY
       WHEN claims_mac.MEASUREMENT = 'M30' THEN (COST_VANTAGE_TABLE.CV_UNIT_PRICE * claims_mac.TOT_QTY + (18.46-1.85) * claims_mac.CLAIMS)/claims_mac.TOT_QTY
       ELSE COST_VANTAGE_TABLE.CV_UNIT_PRICE END AS CV_UNIT_PRICE,
       CAST(claims_mac.TOT_DISP_FEE AS NUMERIC) AS TOT_DISP_FEE,
       CAST(claims_mac.TOT_SPEND AS NUMERIC) AS TOT_ING_COST
FROM (
  SELECT
    client_info.*,
    claims_agg.* EXCEPT(CUSTOMER_ID,
      NDC_PRICE,
      GPI_PRICE),
      CASE WHEN (client_info.guarantee_category LIKE '%Vanilla%' AND CHAIN_SUBGROUP LIKE "%R90OK%") THEN 'R30'
           WHEN CHAIN_SUBGROUP = 'MCHOICE' THEN 'R30'
           WHEN CHAIN_SUBGROUP = 'CMK_SPECIALTY' or MEASUREMENT like 'S%' THEN 'R30'
      ELSE MEASUREMENT 
      END AS DUMMY_MEASUREMENT
  FROM (
    SELECT
      *,
      CASE
        WHEN CONCAT(CUSTOMER_ID, GPI) IN (SELECT DISTINCT CONCAT(CLIENT, GPI) FROM `pbm-mac-lp-prod-de.ds_pro_lp.ger_opt_mac_price_override` WHERE NDC LIKE "%*%") THEN "GPI OVERRIDE LIST"
        WHEN CONCAT(CUSTOMER_ID, NDC) IN (SELECT DISTINCT CONCAT(CLIENT, NDC) FROM `pbm-mac-lp-prod-de.ds_pro_lp.ger_opt_mac_price_override` WHERE NDC NOT LIKE "%*%") THEN "NDC OVERRIDE LIST"
        WHEN NDC IN (SELECT DISTINCT drug_id AS NDC FROM `pbm-mac-lp-prod-de.ds_pro_lp.gpi_change_exclusion_ndc`) THEN "EXCLUSION LIST"
        WHEN CONCAT(MAC_LIST, NDC) IN (SELECT DISTINCT CONCAT(MAC_LIST, NDC)FROM 
            full_mac WHERE NDC NOT LIKE "%*%") THEN "NDC PRICING"
      ELSE "PRICE MUTABLE"
    END AS PRICE_MUTABLE,
      CASE
        WHEN CHAIN_SUBGROUP IN ("CVSSP_R90OK",'CVS_R90OK','CVSSP','MCHOICE',"CMK_SPECIALTY") THEN "CVS"
        WHEN CHAIN_SUBGROUP IN ("WAG_R90OK") THEN "WAG"
        WHEN CHAIN_SUBGROUP IN ("NONPREF_OTH_R90OK","PREF_OTH") THEN "NONPREF_OTH"
        WHEN CHAIN_SUBGROUP IN ("WMT_R90OK") THEN "WMT"
        WHEN CHAIN_SUBGROUP IN ("HMA_R90OK") THEN "HMA"
        WHEN CHAIN_SUBGROUP IN ("CAR_R90OK") THEN "CAR"
        WHEN CHAIN_SUBGROUP IN ("ELE_R90OK") THEN "ELE"
        WHEN CHAIN_SUBGROUP IN ("EPC_R90OK") THEN "EPC"
        WHEN CHAIN_SUBGROUP IN ("GEN_R90OK") THEN "GEN"
      ELSE CHAIN_SUBGROUP
    END AS DUMMY_CHAIN_GROUP
    FROM (
      SELECT
        claims.*,
        mac_gpi.PRICE AS GPI_PRICE,
      CASE 
        WHEN NDC_PRICE IS NOT NULL THEN NDC_PRICE
        WHEN claims.MEASUREMENT LIKE "SPCLT" THEN curr_mac_spec.CURR_MAC_PRC 
        WHEN curr_mac_spec.CURR_MAC_PRC  IS NULL AND mac_gpi.PRICE IS NULL AND NDC_PRICE IS NULL THEN NULL
        ELSE mac_gpi.PRICE
      END  AS CURR_MAC_PRC
      FROM (
        SELECT
          claims_mapped.*,
          mac_ndc.PRICE AS NDC_PRICE
        FROM  claims_mapped
        LEFT JOIN mac_ndc
          ON
          claims_mapped.MAC_LIST = mac_ndc.MAC_LIST
          AND claims_mapped.GPI = mac_ndc.GPI
          AND claims_mapped.NDC = mac_ndc.NDC) claims
      LEFT JOIN  mac_gpi
      ON
        claims.MAC_LIST = mac_gpi.MAC_LIST
        AND claims.GPI = mac_gpi.GPI
      LEFT JOIN curr_mac_spec
      ON claims.CUSTOMER_ID = curr_mac_spec.CUSTOMER_ID
        AND claims.GPI = curr_mac_spec.GPI
        AND claims.NDC = curr_mac_spec.NDC)) claims_agg
  LEFT JOIN client_info
    ON claims_agg.CUSTOMER_ID = client_info.CUSTOMER_ID) claims_mac
  LEFT JOIN (SELECT * FROM `pbm-mac-lp-prod-de.ds_pro_lp.client_goodrx_data_gpi` 
             UNION DISTINCT 
             SELECT * FROM `anbc-pss-prod.de_gms_enrv_pss_prod.gms_aetna_client_goodrx_data_gpi`
            )
             grx_raw
    ON claims_mac.GPI = grx_raw.GPI
    AND claims_mac.CUSTOMER_ID = grx_raw.CUSTOMER_ID
    AND claims_mac.DUMMY_MEASUREMENT = grx_raw.MEASUREMENT
    AND claims_mac.DUMMY_CHAIN_GROUP = grx_raw.CHAIN_GROUP
  LEFT JOIN (SELECT GPI, MEASUREMENT, CHAIN_GROUP, AVG(weighted_mean) AS GRX_PRICE_BOB_CHAIN_MEAS FROM
            (SELECT * FROM `pbm-mac-lp-prod-de.ds_pro_lp.client_goodrx_data_gpi` 
             UNION DISTINCT 
             SELECT * FROM `anbc-pss-prod.de_gms_enrv_pss_prod.gms_aetna_client_goodrx_data_gpi`
            ) GROUP BY 1,2,3) grx_chain_meas
    ON claims_mac.GPI = grx_chain_meas.GPI
    AND claims_mac.DUMMY_MEASUREMENT = grx_chain_meas.MEASUREMENT
    AND claims_mac.DUMMY_CHAIN_GROUP = grx_chain_meas.CHAIN_GROUP
  LEFT JOIN (SELECT GPI, CHAIN_GROUP, AVG(weighted_mean) AS GRX_PRICE_BOB_CHAIN FROM
            (SELECT * FROM `pbm-mac-lp-prod-de.ds_pro_lp.client_goodrx_data_gpi` 
             UNION DISTINCT 
             SELECT * FROM `anbc-pss-prod.de_gms_enrv_pss_prod.gms_aetna_client_goodrx_data_gpi`
            ) GROUP BY 1,2) grx_chain
    ON claims_mac.GPI = grx_chain.GPI
    AND claims_mac.DUMMY_CHAIN_GROUP = grx_chain.CHAIN_GROUP
  LEFT JOIN `pbm-mac-lp-prod-ai.ds_sandbox.grx_raw_df` grx_old
    ON claims_mac.GPI = grx_old.GPI
    AND claims_mac.DUMMY_CHAIN_GROUP = grx_old.CHAIN_GROUP
    AND claims_mac.MEASUREMENT = grx_old.MEASUREMENT
    AND claims_mac.RUN_DATE = grx_old.RUN_DATE
  LEFT JOIN (SELECT GPI, CHAIN_GROUP, RUN_DATE, MIN(GOODRX_UNIT_PRICE) AS GOODRX_PRICE_BOB_CHAIN FROM `pbm-mac-lp-prod-ai.ds_sandbox.grx_raw_df` GROUP  BY 1,2,3) grx_old_bob
    ON claims_mac.GPI = grx_old_bob.GPI
    AND claims_mac.DUMMY_CHAIN_GROUP = grx_old_bob.CHAIN_GROUP
    AND claims_mac.RUN_DATE = grx_old_bob.RUN_DATE
  LEFT JOIN `pbm-mac-lp-prod-ai.ds_sandbox.grx_raw_df` grx_spec
    ON claims_mac.GPI = grx_spec.GPI
    AND claims_mac.DUMMY_CHAIN_GROUP = grx_spec.CHAIN_GROUP
    AND claims_mac.DUMMY_MEASUREMENT = grx_spec.MEASUREMENT
    AND claims_mac.RUN_DATE = grx_spec.RUN_DATE
  LEFT JOIN (SELECT GPI, CHAIN_GROUP, RUN_DATE, MIN(GOODRX_UNIT_PRICE) AS GOODRX_PRICE_BOB FROM `pbm-mac-lp-prod-ai.ds_sandbox.grx_raw_df` GROUP  BY 1,2,3) grx_spec_bob
    ON claims_mac.GPI = grx_spec_bob.GPI
    AND claims_mac.DUMMY_CHAIN_GROUP = grx_spec_bob.CHAIN_GROUP
    AND claims_mac.RUN_DATE = grx_spec_bob.RUN_DATE
  LEFT JOIN `pbm-mac-lp-prod-ai.ds_sandbox.grx_flat_file` grx_flat_file
    ON claims_mac.GPI = grx_flat_file.GPI
    AND claims_mac.DUMMY_CHAIN_GROUP = grx_flat_file.CHAIN_GROUP
    AND claims_mac.DUMMY_MEASUREMENT = grx_flat_file.MEASUREMENT
  LEFT JOIN (SELECT GPI, CHAIN_GROUP, MIN(goodrx_cost_unit) AS goodrx_cost_unit_bob FROM `pbm-mac-lp-prod-ai.ds_sandbox.grx_flat_file` GROUP  BY 1,2) grx_flat_file_bob
    ON claims_mac.GPI = grx_flat_file_bob.GPI
    AND claims_mac.DUMMY_CHAIN_GROUP = grx_flat_file_bob.CHAIN_GROUP
  LEFT JOIN spclt
    ON claims_mac.GPI = spclt.GPI
  LEFT JOIN COST_PLUS_TABLE
    ON claims_mac.GPI = COST_PLUS_TABLE.GPI
  LEFT JOIN NADAC_TABLE
    ON claims_mac.GPI = NADAC_TABLE.GPI
    AND claims_mac.NDC = NADAC_TABLE.NDC
    AND claims_mac.CUSTOMER_ID = NADAC_TABLE.CUSTOMER_ID
  LEFT JOIN COST_VANTAGE_TABLE
    ON claims_mac.GPI = COST_VANTAGE_TABLE.GPI
  LEFT JOIN (SELECT
    DISTINCT CUSTOMER_ID,
    "Y" AS COSTSAVER_CLIENT
    FROM
    `anbc-pss-prod.de_gms_enrv_pss_prod.clms_survey_summary`) clms
    ON claims_mac.CUSTOMER_ID = clms.CUSTOMER_ID
WHERE claims_mac.CUSTOMER_ID IS NOT NULL
))


SELECT * FROM final_table
WHERE CURR_MAC_PRC IS NOT NULL
AND SPCLT_DRUG_IND IS NOT NULL
-- )