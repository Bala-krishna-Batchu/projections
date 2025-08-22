with
date_filter_table as (
    SELECT
    Customer_Id,
    -- DATE_SUB(last_day(max(dof), week), interval 1 week) as date_filter
    last_day(max(dof), week) as date_filter
    FROM `pbm-mac-lp-prod-de.landing.GER_OPT_CLIENT_PHARM_CLAIMS_STANDARD`
    WHERE CLIENT_GUARANTEE = 'Y'
    group by 1
),base_table as(
    SELECT
        tax.client_name,
        tax.customer_id,
        tax.client_type,
        tax.guarantee_category,
        tax.contract_eff_dt,
        tax.contract_exprn_dt,
        d.date_filter,
        GREATEST(tax.contract_eff_dt,DATE_SUB(d.date_filter,INTERVAL 91 DAY)) AS proj_back_date,
        case
            when tax.medd = 1 then 1 
            when tax.egwp = 1 then 1
            when tax.wrap = 1 then 1
            else 0
        end as MedD,
    FROM `anbc-pss-prod.fdm_cnfv_pss_prod.gms_ger_opt_taxonomy_final` as tax
    left JOIN date_filter_table d
    on tax.customer_id = d.customer_id
), base_pharmacy_rate_table as (
    SELECT
        claims.CUSTOMER_ID,
        trim(chain_group) as CHAIN_GROUP,
        Pharmacy_Rate, 
        trim(Measurement_clean) as MEASUREMENT_CLEAN,
        sum(FULLAWP) as AWP
    from `pbm-mac-lp-prod-de.landing.GER_OPT_CLIENT_PHARM_CLAIMS_STANDARD` as claims
    inner join base_table as tax
        on claims.customer_id = tax.customer_id
    WHERE claims.dof BETWEEN tax.contract_eff_dt AND tax.contract_exprn_dt
    AND NOT REGEXP_CONTAINS(LOWER(claims.Client_Name), r'(specialty|wrap|ldd|nadac|_hif|_ter|_ihs|_hiv)')
    and trim(Pharmacy_Guarantee)='Y'
    GROUP BY claims.CUSTOMER_ID,CHAIN_GROUP,MEASUREMENT_CLEAN, Pharmacy_Rate
),
pharmacy_rates_table as (
    select base_pharmacy_rate_table.*
    from base_pharmacy_rate_table
    inner join (
    select
        customer_id,
        base_pharmacy_rate_table.CHAIN_GROUP,
        base_pharmacy_rate_table.MEASUREMENT_CLEAN,
        max(awp) as max_awp
    from base_pharmacy_rate_table
    group by customer_id, CHAIN_GROUP, MEASUREMENT_CLEAN
    ) as sub
    on base_pharmacy_rate_table.customer_id = sub.customer_id
    and base_pharmacy_rate_table.CHAIN_GROUP = sub.CHAIN_GROUP
    and base_pharmacy_rate_table.MEASUREMENT_CLEAN = sub.MEASUREMENT_CLEAN
    and base_pharmacy_rate_table.awp = sub.max_awp
),

individual_claims as (
    select
    	claims.CUSTOMER_ID,
       	tax.Client_Name,
    	claims.Measurement_Clean,
        LAST_DAY(claims.dof, WEEK) AS Week,
        claims.rate,
        tax.client_type,
        tax.guarantee_category,
        tax.MedD,
        pharmacy_rates_table.pharmacy_rate,
        tax.contract_eff_dt,
        tax.contract_exprn_dt,
        tax.proj_back_date,
        tax.date_filter,
        CLIENT_GUARANTEE,
        PHARMACY_GUARANTEE,
        claims.CLAIM_ID,
        claims.FULLAWP,
        claims.NORMING,
        claims.PSTINGPD,
        claims.appingpd,
        claims.PSTCOSTTYPE,
        claims.QUANTITY,
        claims.UCAMT,
        claims.GPI,
        claims.NDC,
        claims.NETWORK,
        claims.mlor_cd,
        claims.dof,
        case
            when claims.Measurement_Clean like 'R3%' then 'R30'
            when claims.Measurement_Clean like 'R9%' then 'R90'
            when claims.Measurement_Clean like 'M3%' then 'M30'
            else claims.Measurement_Clean
        end as MEASUREMENT,

        CASE
            WHEN tax.Guarantee_Category in ('Pure Vanilla','MedD/EGWP Vanilla') THEN
                CASE
                    WHEN claims.Measurement_Clean LIKE 'M%' THEN 'Mail'
                    WHEN claims.Measurement_Clean LIKE '%LTC%' THEN 'LTC'
                    WHEN claims.Measurement_Clean LIKE '%ALF%' THEN 'ALF'
                    WHEN claims.Measurement_Clean LIKE '%ECN%' THEN 'ECN'
                    ELSE 'Retail'
                END
            WHEN tax.Guarantee_Category in ('Offsetting R30/R90','MedD/EGWP Offsetting R30/R90/LTC','MedD/EGWP Offsetting Complex','Offsetting Complex') THEN
                CASE
                    WHEN claims.Measurement_Clean LIKE 'M%' THEN 'Mail'
                    WHEN claims.Measurement_Clean LIKE '%LTC%' THEN 'LTC'
                    WHEN claims.Measurement_Clean LIKE '%ALF%' THEN 'ALF'
                    WHEN claims.Measurement_Clean LIKE '%ECN%' THEN 'ECN'
                    ELSE 'Retail'
                END
            WHEN tax.Guarantee_Category in ('NonOffsetting R30/R90', 'NonOffsetting Complex') THEN
                CASE
                    WHEN claims.Measurement_Clean LIKE 'M%' THEN 'Mail'
                    WHEN claims.Measurement_Clean LIKE '%R30%' THEN 'R30'
                    WHEN claims.Measurement_Clean LIKE '%R90%' THEN 'R90'
                    WHEN claims.Measurement_Clean LIKE '%LTC%' THEN 'LTC'
                    WHEN claims.Measurement_Clean LIKE '%ALF%' THEN 'ALF'
                    WHEN claims.Measurement_Clean LIKE '%ECN%' THEN 'ECN'
                    ELSE claims.Measurement_Clean
                END
              WHEN tax.Guarantee_Category in ('MedD/EGWP NonOffsetting R30/R90/LTC', 'MedD/EGWP NonOffsetting Complex') THEN
                CASE
                    WHEN claims.Measurement_Clean LIKE 'M%' THEN 'Mail'
                    WHEN claims.Measurement_Clean LIKE '%R30P' THEN 'R30P'
                    WHEN claims.Measurement_Clean LIKE '%R30%' THEN 'R30'
                    WHEN claims.Measurement_Clean LIKE '%R90P%' THEN 'R90P'
                    WHEN claims.Measurement_Clean LIKE '%R90%' THEN 'R90'
                    WHEN claims.Measurement_Clean LIKE '%LTC%' THEN 'LTC'
                    WHEN claims.Measurement_Clean LIKE '%ALF%' THEN 'ALF'
                    WHEN claims.Measurement_Clean LIKE '%ECN%' THEN 'ECN'
                    WHEN LOWER(claims.Measurement_Clean) LIKE '%90%preferred%' THEN 'R90P'
                    WHEN LOWER(claims.Measurement_Clean) LIKE '%preferred90%' THEN 'R90P'
                    WHEN LOWER(claims.Measurement_Clean) LIKE '%30%preferred%' THEN 'R30P'
                    WHEN LOWER(claims.Measurement_Clean) LIKE '%preferred30%' THEN 'R30P'
                    ELSE claims.Measurement_Clean
                END
            ELSE claims.Measurement_Clean
        END as BREAKOUT,

        CASE
            WHEN ((claims.Measurement_Clean LIKE 'M%') AND (claims.NETWORK IN ('MCHCE', 'CVSNPP'))) THEN 'MCHOICE'
            WHEN ((claims.Measurement_Clean LIKE 'M%') AND (claims.chain_group ='IND')) THEN 'MAIL'
            ELSE claims.chain_group
        END AS CHAIN_GROUP_TEMP,

        from `pbm-mac-lp-prod-de.landing.GER_OPT_CLIENT_PHARM_CLAIMS_STANDARD` as claims
        inner join base_table as tax
            on claims.customer_id = tax.customer_id
        left join pharmacy_rates_table
            on pharmacy_rates_table.customer_id = claims.customer_id
            and pharmacy_rates_table.CHAIN_GROUP = claims.CHAIN_GROUP
            and pharmacy_rates_table.MEASUREMENT_CLEAN = claims.MEASUREMENT_CLEAN

		WHERE (client_guarantee = 'Y' or pharmacy_guarantee = 'Y')
        AND claims.dof BETWEEN tax.contract_eff_dt AND tax.contract_exprn_dt
        and claims.dof <= tax.date_filter
        AND NOT REGEXP_CONTAINS(LOWER(claims.Client_Name), r'(specialty|wrap|ldd|nadac|_hif|_ter|_ihs|_hiv)')

-- pulling in current prices
    ), MAC_GPI AS (
        SELECT
        CAST(MAC_LIST_ID AS STRING) AS MAC_LIST_ID,
        CAST(MAC_GPI_CD AS STRING) AS GPI,
        CAST(MAC_COST_AMT AS FLOAT64) AS PRICE
        FROM `anbc-pss-prod.fdm_cnfv_pss_prod.gms_ger_opt_base_mac_lists`
        WHERE VALID_GPPC = 'Y'
        AND CURRENT_DATE < MAC_EXPRN_DT
        AND DRUG_ID LIKE '%*%'
    ), MAC_NDC AS (
        SELECT
        CAST(MAC_LIST_ID AS STRING) AS MAC_LIST_ID,
        CAST(DRUG_ID AS STRING) AS NDC,
        CAST(MAC_COST_AMT AS FLOAT64) AS PRICE
        FROM `anbc-pss-prod.fdm_cnfv_pss_prod.gms_ger_opt_base_mac_lists`
        WHERE VALID_GPPC = 'Y'
        AND CURRENT_DATE < MAC_EXPRN_DT
        AND DRUG_ID NOT LIKE '%*%'
-- this is how we determine the AWP discount by Network across the board so we can set a cap on prices
), FREQ_NETWORK AS (
  SELECT customer_id, mlor_cd, network, CAST(100-ROUND(NORMING/FULLAWP*20)*9 AS INT64) AS DISCOUNT,
  ROW_NUMBER() OVER (
      PARTITION BY customer_id, mlor_cd, network 
      ORDER BY COUNT(CLAIM_ID) DESC) AS RN, 
  COUNT(CLAIM_ID) AS N
  FROM individual_claims
  WHERE PSTCOSTTYPE = 'AWP'
  GROUP BY 1, 2, 3, 4
), FREQ_CHNL AS (
    SELECT customer_id, mlor_cd, CAST(100-ROUND(NORMING/FULLAWP*20)*9 AS INT64) AS DISCOUNT,
  ROW_NUMBER() OVER (
      PARTITION BY customer_id, mlor_cd 
      ORDER BY COUNT(CLAIM_ID) DESC) AS RN, 
  COUNT(CLAIM_ID) AS N
  FROM individual_claims
  WHERE PSTCOSTTYPE = 'AWP'
  GROUP BY 1, 2, 3
), DISCOUNTS AS (
  SELECT A.customer_id, A.mlor_cd, A.network, 
  A.discount AS NETWORK_DISCOUNT,
  B.discount AS CHNL_DISCOUNT
  FROM FREQ_NETWORK A
  JOIN FREQ_CHNL B
  ON A.customer_id = B.customer_id
  AND A.mlor_cd = B.mlor_cd
  WHERE A.RN = 1 AND B.RN = 1
), RAW_PROJ_CLAIMS AS (
    SELECT
    c.CUSTOMER_ID,
    c.CLIENT_NAME,
    c.MEASUREMENT,
    c.MEASUREMENT_CLEAN,
    c.CONTRACT_EFF_DT,
    c.CONTRACT_EXPRN_DT,
    c.proj_back_date,
    c.date_filter,
    c.client_type,
    c.medd,
    c.guarantee_category,
    c.CHAIN_GROUP_TEMP,
    c.RATE,
    c.PHARMACY_RATE,
    c.BREAKOUT,
    c.DOF,
    c.GPI,
    c.NDC,
    c.QUANTITY,
    c.NORMING,
    c.APPINGPD,
    c.FULLAWP,
    c.PSTCOSTTYPE,
    c.UCAMT,
    c.NETWORK,
    c.CLIENT_GUARANTEE,
    c.PHARMACY_GUARANTEE,
    COALESCE(MAC_NDC.PRICE,MAC_GPI.PRICE) AS MAC_PRICE,
    CASE WHEN COALESCE(D.NETWORK_DISCOUNT,D.CHNL_DISCOUNT) IS NULL AND c.mlor_cd = 'TRUE' THEN CASE WHEN UPPER(c.client_name) LIKE '%MEDICAID%' OR (c.medd = 0) THEN 0.43 ELSE 0.55 END
         WHEN COALESCE(D.NETWORK_DISCOUNT,D.CHNL_DISCOUNT) IS NULL AND c.mlor_cd = 'FALSE' THEN CASE WHEN UPPER(c.client_name) LIKE '%MEDICAID%' OR (c.medd = 0) THEN 0.43 ELSE 0.55 END
         ELSE COALESCE(D.NETWORK_DISCOUNT,D.CHNL_DISCOUNT) END AS DISCOUNT
    FROM individual_claims c
    LEFT JOIN MAC_GPI
    ON c.PSTCOSTTYPE = MAC_GPI.MAC_LIST_ID
    AND c.GPI = MAC_GPI.GPI
    LEFT JOIN MAC_NDC
    ON c.PSTCOSTTYPE = MAC_NDC.MAC_LIST_ID
    AND c.NDC = MAC_NDC.NDC
    LEFT JOIN DISCOUNTS D
    ON c.customer_id = D.customer_id
    AND c.mlor_cd = d.mlor_cd
    AND c.network = d.network
    LEFT JOIN `pbm-mac-lp-prod-ai.pricing_management.clnt_params` clnt_param
    ON CONCAT("['",c.customer_id,"']") = clnt_param.customer_id
    -- the raw claims used for projection only go back about a month
    WHERE c.dof >= c.proj_back_date
), daily_projections as (
SELECT
    	CUSTOMER_ID,
       	Client_Name,
        MEASUREMENT,
    	Measurement_Clean,
        RATE,
        client_type,
        guarantee_category,
        MedD,
        pharmacy_rate,
        BREAKOUT,
        CHAIN_GROUP_TEMP,
        contract_eff_dt,
        contract_exprn_dt,
        date_filter,
-- Assume floor prices and price overrides have already been applied to the VCML, then the
-- cap ends up being either the AWP discount or the UCAMT on the claim
SUM(CASE WHEN CLIENT_GUARANTEE = 'Y' THEN 1/(DATE_DIFF(date_filter,proj_back_date,DAY)+1) ELSE 0 END) AS CLIENT_PROJ_CLAIMS_DAILY,
SUM(CASE WHEN  CLIENT_GUARANTEE = 'N' THEN 0
    WHEN MAC_PRICE IS NOT NULL THEN LEAST(MAC_PRICE*QUANTITY,FULLAWP*(1-DISCOUNT/100),UCAMT)/(DATE_DIFF(date_filter,proj_back_date,DAY)+1)
    ELSE NORMING/(DATE_DIFF(date_filter,proj_back_date,DAY)+1) END) AS CLIENT_PROJ_SPEND_DAILY,
SUM(CASE WHEN CLIENT_GUARANTEE = 'Y' THEN FULLAWP/(DATE_DIFF(date_filter,proj_back_date,DAY)+1) ELSE 0 END) AS CLIENT_PROJ_AWP_DAILY,
SUM(CASE WHEN PHARMACY_GUARANTEE = 'Y' THEN 1/(DATE_DIFF(date_filter,proj_back_date,DAY)+1) ELSE 0 END) AS PHARMACY_PROJ_CLAIMS_DAILY,
SUM(CASE WHEN  PHARMACY_GUARANTEE = 'N' THEN 0
    WHEN MAC_PRICE IS NOT NULL THEN LEAST(MAC_PRICE*QUANTITY,FULLAWP*(1-DISCOUNT/100),UCAMT)/(DATE_DIFF(date_filter,proj_back_date,DAY)+1)
    ELSE APPINGPD/(DATE_DIFF(date_filter,proj_back_date,DAY)+1) END) AS PHARMACY_PROJ_SPEND_DAILY,
------------------------------------------
------------------------------------------
SUM(CASE WHEN PHARMACY_GUARANTEE = 'Y' THEN FULLAWP/(DATE_DIFF(date_filter,proj_back_date,DAY)+1) ELSE 0 END) AS PHARMACY_PROJ_AWP_DAILY
FROM RAW_PROJ_CLAIMS
GROUP BY 
CUSTOMER_ID,
Client_Name,
MEASUREMENT,
Measurement_Clean,
RATE,
client_type,
guarantee_category,
MedD,
pharmacy_rate,
BREAKOUT,
CHAIN_GROUP_TEMP,
contract_eff_dt,
contract_exprn_dt,
date_filter
----- generate dates for each week after we have claims out until end of contract
----- for every single breakout
), W AS (
    SELECT *,
    GENERATE_DATE_ARRAY(DATE_ADD(date_filter,INTERVAL 7 DAY), contract_exprn_dt, INTERVAL 1 WEEK) AS contract_weeks
    FROM daily_projections
), weekly_periods as ( 
    SELECT *,
    LAST_DAY(wk,WEEK) AS week
    FROM W, W.contract_weeks as wk
), grouped_claims_with_projections as (
    --- this part is the projections (ACTUAL_FLAG = 0)
    SELECT 
    	CUSTOMER_ID,
       	Client_Name,
    	Measurement_Clean,
        MEASUREMENT,
        week as week,
        rate,
        client_type,
        guarantee_category,
        MedD,
        BREAKOUT,
        CHAIN_GROUP_TEMP,
        pharmacy_rate,
        contract_eff_dt,
        contract_exprn_dt,
        0 AS actual_flag,
    CLIENT_PROJ_CLAIMS_DAILY * 7 AS CLIENT_CLAIMS,
    CLIENT_PROJ_AWP_DAILY * 7 AS CLIENT_AWP,
    CLIENT_PROJ_SPEND_DAILY * 7 AS CLIENT_SPEND,
    PHARMACY_PROJ_CLAIMS_DAILY * 7 AS PHARMACY_CLAIMS,
    PHARMACY_PROJ_AWP_DAILY * 7 AS PHARMACY_AWP,
    PHARMACY_PROJ_SPEND_DAILY * 7 AS PHARMACY_SPEND
    FROM weekly_periods
UNION ALL
    --- this part is the actuals (ACTUAL_FLAG = 1)
    select
    	CUSTOMER_ID,
       	Client_Name,
    	Measurement_Clean,
        MEASUREMENT,
        Week,
        rate,
        client_type,
        guarantee_category,
        MedD,
        BREAKOUT,
        CHAIN_GROUP_TEMP,
        pharmacy_rate,
        contract_eff_dt,
        contract_exprn_dt,
        1 AS actual_flag,
        CAST(COUNT(DISTINCT CASE WHEN CLIENT_GUARANTEE = 'Y' THEN CLAIM_ID END) AS FLOAT64) AS CLIENT_CLAIMS,
    	SUM(CASE WHEN CLIENT_GUARANTEE = 'Y' THEN FULLAWP ELSE 0 END) AS CLIENT_AWP,
    	SUM(CASE WHEN CLIENT_GUARANTEE = 'Y' THEN NORMING ELSE 0 END) AS CLIENT_SPEND,

		CAST(COUNT(DISTINCT CASE WHEN PHARMACY_GUARANTEE = 'Y' THEN CLAIM_ID END) AS FLOAT64) AS PHARMACY_CLAIMS,
    	SUM(CASE WHEN PHARMACY_GUARANTEE = 'Y' THEN FULLAWP ELSE 0 END) AS PHARMACY_AWP,
        SUM(CASE WHEN PHARMACY_GUARANTEE = 'Y' THEN appingpd ELSE 0 END) AS PHARMACY_SPEND,
        from individual_claims
	GROUP BY
    	CUSTOMER_ID,
       	Client_Name,
        MEASUREMENT,
    	Measurement_Clean,
		week,
        RATE,
        client_type,
        guarantee_category,
        MedD,
        pharmacy_rate,
        BREAKOUT,
        CHAIN_GROUP_TEMP,
        contract_eff_dt,
        contract_exprn_dt
), clean_grouped_claims_with_projections as (
    select *,
        sum((1-RATE) * CLIENT_AWP - CLIENT_SPEND) OVER (PARTITION BY CUSTOMER_ID, Client_Name, BREAKOUT,Week,CHAIN_GROUP_TEMP) AS Performance,
        CASE WHEN CLIENT_AWP <> 0 THEN ROUND((1-CLIENT_SPEND/CLIENT_AWP)*100, 2) ELSE 0 END AS Effective_GER,
        ROUND((1-RATE) * CLIENT_AWP - CLIENT_SPEND, 2) AS Measurement_Performance,
        CASE WHEN PHARMACY_AWP <> 0 THEN ROUND((1-PHARMACY_SPEND/PHARMACY_AWP)*100, 2) ELSE 0 END AS PHARMACY_Effective_GER,
    from grouped_claims_with_projections
        where (CLIENT_CLAIMS > 0 OR PHARMACY_CLAIMS > 0)
),
  ds_updates AS ( (
    SELECT
      hist.client_name AS Customer_id,
      PARSE_DATE("%y%m%d", SUBSTRING(fromdt,2,6)) AS PRICE_CHANGE_DATE,
      run_id AS RunID,
      COUNT(DISTINCT GPI) AS Number_GPIs,
      COUNT(DISTINCT MACLIST) AS Number_VCMLs
    FROM
      `pbm-mac-lp-prod-de.publish.LP_PRICE_RECOMMENDATION_HIST` hist
    INNER JOIN
      base_table
    ON
      hist.client_name = base_table.customer_id
    WHERE
      PARSE_DATE("%y%m%d", SUBSTRING(fromdt,2,6)) > contract_eff_dt
    GROUP BY
      1,
      2,
      3)
  UNION ALL (
    SELECT
      hist.client_name AS Customer_id,
      PARSE_DATE("%y%m%d", SUBSTRING(fromdt,2,6)) AS PRICE_CHANGE_DATE,
      run_id AS RunID,
      COUNT(DISTINCT GPI) AS Number_GPIs,
      COUNT(DISTINCT MACLIST) AS Number_VCMLs
    FROM
      `pbm-mac-lp-prod-de.publish.CP_PRICE_RECOMMENDATION_HIST` hist
    INNER JOIN
      base_table
    ON
      hist.client_name = base_table.customer_id
    WHERE
      PARSE_DATE("%y%m%d", SUBSTRING(fromdt,2,6)) > contract_eff_dt
    GROUP BY
      1,
      2,
      3 ) 
      ),
      full_updates AS (
      SELECT
        vcml.customer_id,
        mac_eff_dt AS PRICE_CHANGE_DATE,
        COUNT(DISTINCT mac_gpi_cd) AS Number_GPIs,
        COUNT(DISTINCT mac_list_id) AS Number_VCMLs
      FROM
        `anbc-pss-prod.fdm_cnfv_pss_prod.gms_v_drug_mac_hist` mac
      INNER JOIN
        `anbc-pss-prod.fdm_cnfv_pss_prod.gms_ger_opt_vcml_reference` AS vcml
      ON
        mac.mac_list_id = vcml.vcml_id
        AND rec_curr_ind = 'Y'
      INNER JOIN
        base_table
      ON
        vcml.customer_id = base_table.customer_id
        AND mac_eff_dt > contract_eff_dt
      GROUP BY
        1,
        2 ),
    detail_price_updates as (
    SELECT
      full_updates.customer_id,
      full_updates.PRICE_CHANGE_DATE,
      COALESCE(ds_updates.Number_GPIs,full_updates.Number_GPIs) AS Number_GPIs,
      COALESCE(ds_updates.Number_VCMLs,full_updates.Number_VCMLs) AS Number_VCMLs,
      COALESCE(ds_updates.RunID, 'OTHER') AS RunID,
    FROM
      full_updates
    LEFT JOIN
      ds_updates
    ON
      TRIM(ds_updates.customer_id) = TRIM(full_updates.customer_id)
      AND DATE(ds_updates.PRICE_CHANGE_DATE) = DATE(full_updates.PRICE_CHANGE_DATE)
    WHERE
      (runID != 'OTHER'
        or (COALESCE(ds_updates.Number_GPIs,full_updates.Number_GPIs)/COALESCE(ds_updates.Number_VCMLs,full_updates.Number_VCMLs)> 2
        and COALESCE(ds_updates.Number_VCMLs,full_updates.Number_VCMLs) > 1))
    ),
price_updates as (
    select
        customer_id as Customer_id,
        max(PRICE_CHANGE_DATE) as Last_Go_Live
    from detail_price_updates
    group by 1
),
lp_client_table as (
    select
        distinct
        CLCODE,
        1 as LP_Client
    from (
        select
            distinct
            CLCODE
        from `pbm-mac-lp-prod-ai.ds_production.GER_OPT_Monthly_Algo_Exclusion_List_Table`
        where current_date() between START_DATE and END_DATE
        and REC_CURR_IND = 'Y'
    )
)
SELECT
    clean_grouped_claims_with_projections.*,
    case when (lp_client_table.LP_Client is null or lp_client_table.LP_Client = 0) then 0 else 1 end as LP_Client,
    price_updates.Last_Go_Live,
from clean_grouped_claims_with_projections
left join lp_client_table
    on clean_grouped_claims_with_projections.customer_id = lp_client_table.CLCODE
left join price_updates
    on clean_grouped_claims_with_projections.customer_id = price_updates.customer_id
order by 1,3