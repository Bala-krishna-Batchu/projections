query_data_quality = """
SELECT
    customer_id,
    CONCAT
    (
        CASE WHEN vcml_cnt_chk = 0 THEN '' ELSE 'vcml_cnt_chk;' END,
        CASE WHEN cust_id_clnm_chk = 0 THEN '' ELSE 'cust_id_clnm_chk;' END,
        CASE WHEN tax_cust_id_chk = 0 THEN '' ELSE 'tax_cust_id_chk;' END,
        CASE WHEN ref_dup_chk = 0 THEN '' ELSE 'ref_dup_chk;' END,
        CASE WHEN null_chk_sf_vcml_ref_tbl = 0 THEN '' ELSE 'null_chk_sf_vcml_ref_tbl;' END,
        CASE WHEN tax_dup_chk = 0 THEN '' ELSE 'tax_dup_chk;' END,
        CASE WHEN null_chk_tax_tbl = 0 THEN '' ELSE 'null_chk_tax_tbl;' END,
        CASE WHEN price_chk = 0 THEN '' ELSE 'price_chk;' END,
        CASE WHEN price_override_inactive_chk = 0 THEN '' ELSE 'price_override_inactive_chk;' END,
        CASE WHEN miss_clnt_chk = 0 THEN '' ELSE 'miss_clnt_chk;' END,
        CASE WHEN logical_values_chk_awp_qty = 0 THEN '' ELSE 'logical_values_chk_awp_qty;' END,
        CASE WHEN format_chk_clms_tbl = 0 THEN '' ELSE 'format_chk_clms_tbl;' END,
        CASE WHEN custom_vcml_chk = 0 THEN '' ELSE 'custom_vcml_chk;' END,
        CASE WHEN tax_auto30_client_type_chk = 0 THEN '' ELSE 'tax_auto30_client_type_chk;' END,
        CASE WHEN vcml_guar_chk = 0 THEN '' ELSE 'vcml_guar_chk;' END,
        CASE WHEN phmcy_clms_cid_carrier_mapping_chk = 0 THEN '' ELSE 'phmcy_clms_cid_carrier_mapping_chk;' END,
        CASE WHEN pharm_present_chk = 0 THEN '' ELSE 'pharm_present_chk;' END,
        CASE WHEN dup_chk_cust_info_chk = 0 THEN '' ELSE 'dup_chk_cust_info_chk;' END,
        CASE WHEN dup_chk_pharm_guar = 0 THEN '' ELSE 'dup_chk_pharm_guar;' END,
        CASE WHEN pharm_guar_chk = 0 THEN '' ELSE 'pharm_guar_chk;' END,
        CASE WHEN claim_to_pharm_map_chk = 0 THEN '' ELSE 'claim_to_pharm_map_chk;' END,
        CASE WHEN meas_to_pharm_map_chk = 0 THEN '' ELSE 'meas_to_pharm_map_chk;' END,
        CASE WHEN vcml_id_chk = 0 THEN '' ELSE 'vcml_id_chk;' END,
        CASE WHEN zero_clnt_guar_chk = 0 THEN '' ELSE 'zero_clnt_guar_chk;' END,
        CASE WHEN zero_pharm_guar_chk = 0 THEN '' ELSE 'zero_pharm_guar_chk;' END,
        CASE WHEN inclusion_list_chk = 0 THEN '' ELSE 'inclusion_list_chk;' END,
        CASE WHEN max_claim_date_chk = 0 THEN '' ELSE 'max_claim_date_chk;' END,
        CASE WHEN coins_chk = 0 THEN '' ELSE 'coins_chk;' END,
        CASE WHEN anom_awp_vol_chk_tot_tbl = 0 THEN '' ELSE 'anom_awp_vol_chk_tot_tbl;' END,
        CASE WHEN anom_awp_vol_pharm_chk_tot_tbl = 0 THEN '' ELSE 'anom_awp_vol_pharm_chk_tot_tbl;' END,
        CASE WHEN anom_meas_vol_chk_tot_tbl = 0 THEN '' ELSE 'anom_meas_vol_chk_tot_tbl;' END,
        CASE WHEN client_pharm_awp_amb_chk = 0 THEN '' ELSE 'client_pharm_awp_amb_chk;' END,
        CASE WHEN anom_last_data_ck_tbl = 0 THEN '' ELSE 'anom_last_data_ck_tbl;' END,
        CASE WHEN customer_id_table_chk = 0 THEN '' ELSE 'customer_id_table_chk;' END,
        CASE WHEN r90_claims_vcml_chk = 0 THEN '' ELSE 'r90_claims_vcml_chk;' END,
        CASE WHEN null_tic_chk = 0 THEN '' ELSE 'null_tic_chk;' END,
        CASE WHEN null_nadac_chk = 0 THEN '' ELSE 'null_nadac_chk;' END,
        CASE WHEN null_grte_chk = 0 THEN '' ELSE 'null_grte_chk;' END
    ) AS error_message
FROM anbc-prod.fdl_gdp_ae_ent_enrv_prod.e2r_wide
WHERE 
    customer_id IN ({customer_id_str})
    AND NOT
    (
        vcml_cnt_chk = 0
        AND cust_id_clnm_chk = 0
        AND tax_cust_id_chk = 0
        AND ref_dup_chk = 0
        AND null_chk_sf_vcml_ref_tbl = 0
        AND tax_dup_chk = 0
        AND null_chk_tax_tbl = 0
        AND price_chk = 0
        AND price_override_inactive_chk = 0
        AND miss_clnt_chk = 0
        AND logical_values_chk_awp_qty = 0
        AND format_chk_clms_tbl = 0
        AND custom_vcml_chk = 0
        AND tax_auto30_client_type_chk = 0
        AND vcml_guar_chk = 0
        AND phmcy_clms_cid_carrier_mapping_chk = 0
        AND pharm_present_chk = 0
        AND dup_chk_cust_info_chk = 0
        AND dup_chk_pharm_guar = 0
        AND pharm_guar_chk = 0
        AND claim_to_pharm_map_chk = 0
        AND meas_to_pharm_map_chk = 0
        AND vcml_id_chk = 0
        AND zero_clnt_guar_chk = 0
        AND zero_pharm_guar_chk = 0
        AND inclusion_list_chk = 0
        AND max_claim_date_chk = 0
        AND coins_chk = 0
        AND anom_awp_vol_pharm_chk_tot_tbl = 0
        AND anom_awp_vol_chk_tot_tbl = 0
        AND anom_meas_vol_chk_tot_tbl = 0
        AND client_pharm_awp_amb_chk = 0
        AND anom_last_data_ck_tbl = 0
        AND customer_id_table_chk = 0
        AND r90_claims_vcml_chk = 0
        AND null_tic_chk = 0
        AND null_nadac_chk = 0
        AND null_grte_chk = 0   
    )
"""

query_data_quality_ws = """
SELECT
    customer_id,
    CONCAT
    (
        CASE WHEN vcml_cnt_chk = 0 THEN '' ELSE 'vcml_cnt_chk;' END,
        CASE WHEN cust_id_clnm_chk = 0 THEN '' ELSE 'cust_id_clnm_chk;' END,
        CASE WHEN tax_cust_id_chk = 0 THEN '' ELSE 'tax_cust_id_chk;' END,
        CASE WHEN ref_dup_chk = 0 THEN '' ELSE 'ref_dup_chk;' END,
        CASE WHEN null_chk_sf_vcml_ref_tbl = 0 THEN '' ELSE 'null_chk_sf_vcml_ref_tbl;' END,
        CASE WHEN tax_dup_chk = 0 THEN '' ELSE 'tax_dup_chk;' END,
        CASE WHEN null_chk_tax_tbl = 0 THEN '' ELSE 'null_chk_tax_tbl;' END,
        CASE WHEN price_chk = 0 THEN '' ELSE 'price_chk;' END,
        CASE WHEN price_override_inactive_chk = 0 THEN '' ELSE 'price_override_inactive_chk;' END,
        CASE WHEN miss_clnt_chk = 0 THEN '' ELSE 'miss_clnt_chk;' END,
        CASE WHEN logical_values_chk_awp_qty = 0 THEN '' ELSE 'logical_values_chk_awp_qty;' END,
        CASE WHEN format_chk_clms_tbl = 0 THEN '' ELSE 'format_chk_clms_tbl;' END,
        CASE WHEN custom_vcml_chk = 0 THEN '' ELSE 'custom_vcml_chk;' END,
        CASE WHEN tax_auto30_client_type_chk = 0 THEN '' ELSE 'tax_auto30_client_type_chk;' END,
        CASE WHEN vcml_guar_chk = 0 THEN '' ELSE 'vcml_guar_chk;' END,
        CASE WHEN phmcy_clms_cid_carrier_mapping_chk = 0 THEN '' ELSE 'phmcy_clms_cid_carrier_mapping_chk;' END,
        CASE WHEN pharm_present_chk = 0 THEN '' ELSE 'pharm_present_chk;' END,
        CASE WHEN dup_chk_cust_info_chk = 0 THEN '' ELSE 'dup_chk_cust_info_chk;' END,
        CASE WHEN dup_chk_pharm_guar = 0 THEN '' ELSE 'dup_chk_pharm_guar;' END,
        --- CASE WHEN pharm_guar_chk = 0 THEN '' ELSE 'pharm_guar_chk;' END, 
        --- pharm_guar_chk is commented out because next year CVS does not have guarantees but different types of reconciliation 
        CASE WHEN claim_to_pharm_map_chk = 0 THEN '' ELSE 'claim_to_pharm_map_chk;' END,
        CASE WHEN meas_to_pharm_map_chk = 0 THEN '' ELSE 'meas_to_pharm_map_chk;' END,
        CASE WHEN vcml_id_chk = 0 THEN '' ELSE 'vcml_id_chk;' END,
        CASE WHEN zero_clnt_guar_chk = 0 THEN '' ELSE 'zero_clnt_guar_chk;' END,
        CASE WHEN zero_pharm_guar_chk = 0 THEN '' ELSE 'zero_pharm_guar_chk;' END,
        CASE WHEN inclusion_list_chk = 0 THEN '' ELSE 'inclusion_list_chk;' END,
        CASE WHEN max_claim_date_chk = 0 THEN '' ELSE 'max_claim_date_chk;' END,
        CASE WHEN coins_chk = 0 THEN '' ELSE 'coins_chk;' END,
        CASE WHEN anom_awp_vol_chk_tot_tbl = 0 THEN '' ELSE 'anom_awp_vol_chk_tot_tbl;' END,
        CASE WHEN anom_awp_vol_pharm_chk_tot_tbl = 0 THEN '' ELSE 'anom_awp_vol_pharm_chk_tot_tbl;' END,
        CASE WHEN anom_meas_vol_chk_tot_tbl = 0 THEN '' ELSE 'anom_meas_vol_chk_tot_tbl;' END,
        CASE WHEN client_pharm_awp_amb_chk = 0 THEN '' ELSE 'client_pharm_awp_amb_chk;' END
        --- CASE WHEN anom_last_data_ck_tbl = 0 THEN '' ELSE 'anom_last_data_ck_tbl;' END,
        --- anom_last_data_ck_tbl because we do not need to check if the client was run in the last two weeks for the WS
        --- CASE WHEN customer_id_table_chk = 0 THEN '' ELSE 'customer_id_table_chk;' END,
        --- customer_id_table_chk is commented out because the e2r welcome season table does not have the customer_id_table_chk column yet
        --- CASE WHEN r90_claims_vcml_chk = 0 THEN '' ELSE 'r90_claims_vcml_chk;' END
        --- r90_claims_vcml_chk is commented out out because the e2r welcome season table does not have the r90_claims_vcml_chk column yet
    ) AS error_message
FROM anbc-prod.fdl_gdp_ae_ent_enrv_prod.e2r_wide_ws
WHERE 
    customer_id IN ({customer_id_str})
    AND NOT
    (
        vcml_cnt_chk = 0
        AND cust_id_clnm_chk = 0
        AND tax_cust_id_chk = 0
        AND ref_dup_chk = 0
        AND null_chk_sf_vcml_ref_tbl = 0
        AND tax_dup_chk = 0
        AND null_chk_tax_tbl = 0
        AND price_chk = 0
        AND price_override_inactive_chk = 0
        AND miss_clnt_chk = 0
        AND logical_values_chk_awp_qty = 0
        AND format_chk_clms_tbl = 0
        AND custom_vcml_chk = 0
        AND tax_auto30_client_type_chk = 0
        AND vcml_guar_chk = 0
        AND phmcy_clms_cid_carrier_mapping_chk = 0
        AND pharm_present_chk = 0
        AND dup_chk_cust_info_chk = 0
        AND dup_chk_pharm_guar = 0
        --- AND pharm_guar_chk = 0
        AND claim_to_pharm_map_chk = 0
        AND meas_to_pharm_map_chk = 0
        AND vcml_id_chk = 0
        AND zero_clnt_guar_chk = 0
        AND zero_pharm_guar_chk = 0
        AND inclusion_list_chk = 0
        AND max_claim_date_chk = 0
        AND coins_chk = 0
        AND anom_awp_vol_pharm_chk_tot_tbl = 0
        AND anom_awp_vol_chk_tot_tbl = 0
        AND anom_meas_vol_chk_tot_tbl = 0
        AND client_pharm_awp_amb_chk = 0
        -- AND anom_last_data_ck_tbl = 0
        -- AND customer_id_table_chk = 0
        -- AND r90_claims_vcml_chk = 0
    )
"""


query_client_tracking = """
INSERT INTO pbm-mac-lp-prod-ai.pricing_management.client_run_status 
(
    customer_id, client_name, client_type, at_run_id, run_type, 
    program_output_path, run_timestamp, run_status, 
    error_type, error_message, batch_id
)
VALUES 
{dq_values}
"""
