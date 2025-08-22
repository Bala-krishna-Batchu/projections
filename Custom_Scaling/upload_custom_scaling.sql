
-- First check that the prices look okay
select *
from `pbm-mac-lp-prod-ai.ds_production_lp.LP_Price_Recomendations_Custom`
--CHANGE BOTH SUFFIX AND PREFIX BELOW
WHERE AT_RUN_ID IN ('CS20231012000000000004011') 
--('CS20230913000000000015283', 'CS20230913000000000014651', 'CS20230913000000000015582', 'CS20230913000000000014164');

-- Then add the run to the custom scaling status table
INSERT INTO `pbm-mac-lp-prod-ai.pricing_management.pm_run_review_status_custom`
    (Run_id, Run_Date, User_id, Client_Id_List, Client_Name, Status, Dm_Begn_Dtm)
SELECT 
    AT_RUN_ID AS Run_id,
    Dm_Begn_Dtm AS Run_Date,
    'C626272' AS User_id,   -- Replace with your CID if needed
    customer_id AS Client_Id_List,
    client_name AS Client_Name,
    'Pending' AS Status,
    CURRENT_TIMESTAMP() AS Dm_Begn_Dtm
FROM (
    SELECT DISTINCT 
        tax.client_name,
        customer_id,
        Dm_Begn_Dtm,
        AT_RUN_ID
    FROM `pbm-mac-lp-prod-ai.ds_production_lp.LP_Price_Recomendations_Custom` lpc
    LEFT JOIN (
        SELECT customer_id, client_name 
        FROM anbc-prod.fdl_gdp_ae_ent_enrv_prod.GER_OPT_TAXONOMY_FINAL
    ) tax 
    ON lpc.client_name = tax.customer_id
    WHERE AT_RUN_ID IN (
        'CS20231012000000000004011'
    )
);

-- Check that this looks okay too
select *
from `pbm-mac-lp-prod-ai.pricing_management.pm_run_review_status_custom`
WHERE RUN_ID IN ('CS20231012000000000004011') --('CS20230913000000000015283', 'CS20230913000000000014651', 'CS20230913000000000015582', 'CS20230913000000000014164');

-- Update status to "Approved"
UPDATE `pbm-mac-lp-prod-ai.pricing_management.pm_run_review_status_custom` 
SET Status = 'Approved' 
WHERE RUN_ID IN ('CS20231012000000000004011') -- ('CS20230913000000000015283', 'CS20230913000000000014651', 'CS20230913000000000015582', 'CS20230913000000000014164');