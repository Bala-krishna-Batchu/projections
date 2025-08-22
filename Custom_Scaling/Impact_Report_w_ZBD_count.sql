SELECT DISTINCT 
    Customer_Id, 
    GPI, 
    GPI_NM, 
    CLIENT_MAC_LIST, 
    N_CLM_ALL,
    N_ZBD_CLM_ALL,

    COUNT(*) AS N_CLM_MACLIST,
    
    COUNT(CASE WHEN zero_balance_indicator = 'TRUE' THEN 1 END) AS N_ZBD_CLM_MACLIST,  -- New column added

    ROUND(AVG(CurrentMAC), 4) AS UNIT_SPEND_OLD,
    ROUND(AVG(MACPRC), 4) AS UNIT_SPEND_NEW,
    ROUND(SUM(MBR_COST) / SUM(ALLOWED), 4) AS PCT_MBRCOST_AS_ALLOWED,
    ROUND(MIN(Floor_Price), 4) AS FLOOR_PRICE
 
FROM (
    SELECT CLM.*,

    COUNT(*) OVER(PARTITION BY GPI) AS N_CLM_ALL,
    SUM(CASE WHEN zero_balance_indicator = 'TRUE' THEN 1 ELSE 0 END) OVER(PARTITION BY GPI) AS N_ZBD_CLM_ALL
 
    FROM (
        SELECT A.*,
            C.GPI_NM,
            B.MACPRC,
            B.CurrentMAC,
            CASE WHEN UCAMT = 0 THEN 1000000 ELSE UCAMT END AS UC_AMT,
            CASE WHEN FULLAWP = 0 THEN 1000000 ELSE FULLAWP * (1 - 0.43) END AS AWP_AMT,
            PSTCOPAY AS MBR_COST,
            NORMING + PSTFEE AS ALLOWED,
            mac1026.MAC_COST_AMT AS Floor_Price

        FROM (
            SELECT *
            FROM `anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_CLIENT_PHARM_CLAIMS_STANDARD`
            WHERE Customer_Id = '4667'                            ---change customer_ID
            AND DOF BETWEEN '2025-01-01' AND '2025-02-20'         ---change this to include the most recent claims since this year
            AND NORMING > 0
        ) A

        LEFT JOIN (
            SELECT *
            FROM `pbm-mac-lp-prod-ai.ds_production_lp.LP_Price_Recomendations`
            WHERE AT_RUN_ID = 'LP20250221113033460888816'         ---change this to match the custom scaling run ID
        ) B
        ON A.CLIENT_MAC_LIST = B.MACLIST
        AND A.GPI = B.GPI

        LEFT JOIN (
            SELECT DISTINCT DRUG_ID, GPI_CD, GPI_NM
            FROM `anbc-pss-prod.fdm_cnfv_pss_prod.V_DRUG_DENORM`
            WHERE BRND_GNRC_CD = 'GNRC'
        ) C
        ON A.GPI = C.GPI_CD
        AND A.NDC = C.DRUG_ID

        LEFT JOIN (
            select DISTINCT MAC_LIST as MAC_LIST_ID, GPI as MAC_GPI_CD, price as MAC_COST_AMT 
            from `anbc-prod.fdl_gdp_ae_ds_pro_lp_share_ent_prod.mac_1026`
        ) mac1026
        ON A.GPI = mac1026.MAC_GPI_CD

        WHERE B.MACPRC IS NOT NULL
    ) CLM
)
GROUP BY 1, 2, 3, 4, 5, 6
ORDER BY Customer_Id, N_CLM_ALL DESC, N_CLM_MACLIST DESC;