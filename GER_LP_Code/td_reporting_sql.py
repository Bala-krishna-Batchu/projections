# -*- coding: utf-8 -*-
"""
"""

update_rec_curr_ind_query = '''
   
    UPDATE {_table_name} SET REC_CURR_IND = 'N'
    WHERE CUSTOMER_ID = {_customer_id} and TIERED_PRICE_LIM = '{_tiered_price_lim}' and ALGO_RUN_DATE BETWEEN '{_algo_run_date1}' AND '{_algo_run_date2}'
    '''

create_table_price_change = ''' CREATE MULTISET TABLE SB_FINANCE_G2_GER_OPT.GER_LP_OUT_PRICE_CHANGE,
NO FALLBACK, NO BEFORE JOURNAL, NO AFTER JOURNAL
(MACLIST  VARCHAR(50),
GPI VARCHAR(50),
GPPC  VARCHAR(50),
NDC11  VARCHAR(50),
EFFDATE DATE,
TERMDATE  DATE,
MACPRC  DECIMAL(9,4),
Current_MAC DECIMAL(10,6),
Quantity  DECIMAL(20 , 9),
CUSTOMER_ID  VARCHAR(50),
MEASUREMENT  VARCHAR(50),
CHAIN_GROUP VARCHAR(50),
MAC_LIST VARCHAR(50),
REGION VARCHAR(50),
GOODRX_PRICE DECIMAL(11,9), 
MAC1026_PRICE DECIMAL(38,5), 
MAC_EFF_DT DATE FORMAT 'yyyy/mm/dd',
GPI_CLS_NM VARCHAR(50),
GPI_CTGRY_NM VARCHAR(50), 
ALGO_RUN_DATE DATE FORMAT 'yyyy/mm/dd', 
CLIENT VARCHAR(50),
ALGO_RUN_OWNER VARCHAR(50), REC_CURR_IND VARCHAR(50),
CLIENT_TYPE VARCHAR(11),
UNC_OPT VARCHAR(6),
GOODRX_OPT VARCHAR(6), 
GO_LIVE DATE,
DATA_ID VARCHAR(50),
TIERED_PRICE_LIM VARCHAR(50))

PRIMARY INDEX (CUSTOMER_ID, MACLIST, GPI) '''


create_table_price_dist = ''' CREATE MULTISET  TABLE SB_Finance_G2_GER_OPT.GER_LP_OUT_PRICE_DIST ( 

        CUSTOMER_ID VARCHAR(5), 

        REGION VARCHAR(5), 

        BREAKOUT VARCHAR(15), 

        SCRIPT_BUCKET VARCHAR(25), 

        script_bucket_cat VARCHAR(25), 

        CLAIMS_PROJ_EOY DECIMAL(30 , 14), 

        CLIENT_NAME VARCHAR(20), 

        ALGO_RUN_DATE DATE FORMAT 'YYYY-MM-DD', 

        ALGO_RUN_OWNER VARCHAR(20), 

        REC_CURR_IND VARCHAR(1),
        CLIENT_TYPE VARCHAR(11),
        UNC_OPT VARCHAR(6),
        GOODRX_OPT VARCHAR(6), 
        GO_LIVE DATE FORMAT 'yyyy/mm/dd',
        DATA_ID VARCHAR(50),
        TIERED_PRICE_LIM VARCHAR(50)

    ) 

    PRIMARY INDEX (CUSTOMER_ID,  BREAKOUT,  SCRIPT_BUCKET,  SCRIPT_BUCKET_CAT);   '''
    
    
create_table_awp_spend = ''' CREATE MULTISET  TABLE SB_Finance_G2_GER_OPT.GER_LP_OUT_AWP_SPEND  ( 

    CUSTOMER_ID VARCHAR(8), 
    BREAKOUT VARCHAR(11), 
    CHAIN_GROUP VARCHAR(11), 
    FULLAWP_ADJ DECIMAL(25 , 8), 
    FULLAWP_ADJ_PROJ_LAG DECIMAL(25 , 8), 
    FULLAWP_ADJ_PROJ_EOY DECIMAL(25 , 8), 
    PRICE_REIMB DECIMAL(25 , 8), 
    LAG_REIMB DECIMAL(25 ,8), 
    Old_Price_Effective_Reimb_Proj_EOY DECIMAL(25 , 8), 
    Price_Effective_Reimb_Proj DECIMAL(25 , 8), 
    GEN_LAG_AWP DECIMAL(25 , 8), 
    GEN_LAG_ING_COST DECIMAL(25 , 8), 
    GEN_EOY_AWP DECIMAL(25 , 8), 
    GEN_EOY_ING_COST DECIMAL(25 , 8), 
    CLIENT_NAME VARCHAR(50), 
    ALGO_RUN_DATE DATE FORMAT 'MM/DD/YYYY', 
    ALGO_RUN_OWNER VARCHAR(20), 
    ENTITY_TYPE VARCHAR(20), 
    Rate DECIMAL(6 , 5), 
    REC_CURR_IND VARCHAR(1),
    CLIENT_TYPE VARCHAR(11),
    UNC_OPT VARCHAR(6),
    GOODRX_OPT VARCHAR(6),
    GO_LIVE DATE FORMAT 'MM/DD/YYYY',
    DATA_ID VARCHAR(50),
    TIERED_PRICE_LIM VARCHAR(50)
    ) 

    PRIMARY INDEX (CUSTOMER_ID, CHAIN_GROUP);   '''
    
    
create_table_ytd_surplus = ''' CREATE MULTISET  TABLE SB_Finance_G2_GER_OPT.GER_LP_OUT_YTD_SURPLUS  (
        CUSTOMER_ID VARCHAR(8),
        MONTH_VALUE BYTEINT,
        MEASUREMENT VARCHAR(8),
        AWP DECIMAL(16 , 4),
        SPEND DECIMAL(16 , 2),
        Rate DECIMAL(5 , 4),
        SURPLUS DECIMAL(16 , 7),
        CLIENT_NAME VARCHAR (20),
        ALGO_RUN_DATE DATE FORMAT 'MM/DD/YYYY',
        ALGO_RUN_OWNER VARCHAR(20),
        REC_CURR_IND VARCHAR(1),
        CLIENT_TYPE VARCHAR(11),
        UNC_OPT VARCHAR(6),
        GOODRX_OPT VARCHAR(6),
        GO_LIVE DATE FORMAT 'yyyy/mm/dd',
        DATA_ID VARCHAR(50),
        TIERED_PRICE_LIM VARCHAR(50)
    )
    PRIMARY INDEX (CUSTOMER_ID, MONTH_VALUE, MEASUREMENT);
 '''

create_table_performance_summary = ''' CREATE MULTISET  TABLE SB_Finance_G2_GER_OPT.GER_LP_OUT_PERFORMANCE_SUM  (
        ENTITY VARCHAR(11),
        Prexisting DECIMAL(30 , 14),
        Model DECIMAL(30 , 14),
        CHAIN_GROUP VARCHAR(11),
        Pre_spend DECIMAL(30 , 14),
        Model_spend DECIMAL(30 , 14),
        SF_client_spend DECIMAL(30 , 14),
        Client_spend_algo_sf DECIMAL(30 , 14),
        Pharm_spend_algo_sf DECIMAL(30 , 14),
        Client_spend_sf_pre DECIMAL(30 , 14),
        Client_spend_model_pre DECIMAL(30 , 14),
        Pharm_spend_pre DECIMAL(30 , 14),
        Pharm_spend_algo DECIMAL(30 , 14),
        Pharm_spend_SF DECIMAL(30 , 14),
        Pharm_spend_algo_pre DECIMAL(30 , 14),
        Pharm_spend_SF_pre DECIMAL(30 , 14),
        ENTITY_TYPE VARCHAR(12),
        CUSTOMER_ID VARCHAR(4),
        CLIENT_NAME VARCHAR(20),
        ALGO_RUN_DATE DATE FORMAT 'MM/DD/YYYY',
        ALGO_RUN_OWNER VARCHAR(20),
        AMOUNT_TYPE VARCHAR(7),
        REC_CURR_IND VARCHAR(1),
        CLIENT_TYPE VARCHAR(11),
        UNC_OPT VARCHAR(6),
        GOODRX_OPT VARCHAR(6),
        GO_LIVE DATE FORMAT 'yyyy/mm/dd',
        DATA_ID VARCHAR(50),
        TIERED_PRICE_LIM VARCHAR(50)
    )
    PRIMARY INDEX (CUSTOMER_ID, ENTITY);
 '''
 
 
create_table_performance = '''CREATE MULTISET  TABLE SB_Finance_G2_GER_OPT.GER_LP_OUT_PRE_MOD_PERF  (
        ENTITY VARCHAR(15),
        Prexisting DECIMAL(30 , 14),
        Model DECIMAL(30 , 14),
        CUSTOMER_ID VARCHAR(4),
        CLIENT_NAME VARCHAR(20),
        ALGO_RUN_DATE DATE FORMAT 'MM/DD/YYYY',
        ALGO_RUN_OWNER VARCHAR(20),
        REC_CURR_IND VARCHAR(1),
        CLIENT_TYPE VARCHAR(11),
        UNC_OPT VARCHAR(6),
        GOODRX_OPT VARCHAR(6),
        GO_LIVE DATE FORMAT 'yyyy/mm/dd',
        DATA_ID VARCHAR(50),
        TIERED_PRICE_LIM VARCHAR(50)
    )
    PRIMARY INDEX (CUSTOMER_ID, ENTITY)'''

query_check_table_exist = """
        Select count(*) from dbc.tables 
        where databasename = 'SB_FINANCE_G2_GER_OPT' 
        and TableName in ('{_table}')
        """

####################################################################
#### Value Reporting Code for uploading MAC Mapping Value file #####
####################################################################

query_create_mac_mapping_table='''
    CREATE  MULTISET TABLE SB_FINANCE_G2_GER_OPT.GER_OPT_MAC_MAPPING,
    NO FALLBACK, NO BEFORE JOURNAL, NO AFTER JOURNAL
    (
    CUSTOMER_ID varchar(30),	
    MEASUREMENT varchar(30),	
    CHAIN_GROUP	varchar(30),
    MAC_LIST varchar(30),	
    REGION varchar(150),
    REC_CURR_IND varchar(30)
    ) 
    Primary index (customer_id, measurement, chain_group,region)
'''    
query_update_rec_curr_ind_mac_mapping ='''
     UPDATE SB_FINANCE_G2_GER_OPT.GER_OPT_MAC_MAPPING SET REC_CURR_IND ='N'
     WHERE CUSTOMER_ID IN ({_customer_id})
     '''

###### New Reporting Script ##################
create_table_awp_spend_perf = '''
CREATE MULTISET TABLE SB_Finance_G2_GER_OPT.GER_LP_OUT_AWP_SPEND_PERF
(
CUSTOMER_ID VARCHAR(8),
CLIENT VARCHAR(50),
ENTITY VARCHAR(20),
ENTITY_TYPE VARCHAR(20),
BREAKOUT VARCHAR(11),
CHAIN_GROUP VARCHAR(11),
CLIENT_OR_PHARM VARCHAR(8),
FULLAWP_ADJ DECIMAL(25 , 8),
FULLAWP_ADJ_PROJ_LAG DECIMAL(25 , 8),
FULLAWP_ADJ_PROJ_EOY DECIMAL(25 , 8),
PRICE_REIMB DECIMAL(25 , 8),
LAG_REIMB DECIMAL(25 , 8),
Old_Price_Effective_Reimb_Proj_EOY DECIMAL(25 , 8),
Price_Effective_Reimb_Proj DECIMAL(25 , 8),
GEN_LAG_AWP DECIMAL(25 , 8),
GEN_LAG_ING_COST DECIMAL(25 , 8),
GEN_EOY_AWP DECIMAL(25 , 8),
GEN_EOY_ING_COST DECIMAL(25 , 8),
Pre_existing_Perf DECIMAL(25 , 8),
Model_Perf DECIMAL(25 , 8),
Proj_Spend_Do_Nothing DECIMAL(25 , 8),
Proj_Spend_Model DECIMAL(25 , 8),
Increase_in_Spend DECIMAL(25 , 8),
Increase_in_Reimb DECIMAL(25 , 8),
Total_Ann_AWP DECIMAL(25 , 8),
GER_Do_Nothing DECIMAL(25 , 8),
GER_Model DECIMAL(25 , 8),
GER_Target DECIMAL(25 , 8),
ALGO_RUN_DATE TIMESTAMP(4) FORMAT 'YYYY-MM-DDBHH:MI:SS',
REC_CURR_IND CHAR(1),
CLIENT_TYPE VARCHAR(11),
UNC_OPT VARCHAR(6),
GOODRX_OPT VARCHAR(6),
GO_LIVE DATE FORMAT 'YYYY-MM-DD',
DATA_ID VARCHAR(50),
TIERED_PRICE_LIM VARCHAR(6),
RUN_TYPE VARCHAR(20)
)
PRIMARY INDEX (CUSTOMER_ID, CLIENT, ENTITY, BREAKOUT, CHAIN_GROUP, CLIENT_OR_PHARM);
'''

# Create table GER_LP_OUT_YTD_SURPLUS_MONTHLY
create_table_YTD_surplus_monthly = '''
CREATE MULTISET TABLE SB_Finance_G2_GER_OPT.GER_LP_OUT_YTD_SURPLUS_MONTHLY
(
CUSTOMER_ID VARCHAR(8),
CLIENT VARCHAR(50),
BREAKOUT VARCHAR(11),
"MONTH" INT,
AWP DECIMAL(25 , 8),
SPEND DECIMAL(25 , 8),
SURPLUS DECIMAL(25 , 8),
ALGO_RUN_DATE TIMESTAMP(4) FORMAT 'YYYY-MM-DDBHH:MI:SS',
CLIENT_TYPE VARCHAR(11),
TIERED_PRICE_LIM VARCHAR(6),
UNC_OPT VARCHAR(6),
GOODRX_OPT VARCHAR(6),
GO_LIVE DATE FORMAT 'YYYY-MM-DD',
DATA_ID VARCHAR(50),
REC_CURR_IND CHAR(1),
RUN_TYPE VARCHAR(20)
)
PRIMARY INDEX (CUSTOMER_ID, CLIENT, BREAKOUT, "MONTH");
'''

# update record current indicator by dimensions of customer_id, tiered_price_lim, go_live, Run_type
update_rec_curr_ind_query_run_type = '''
    UPDATE {_table_name} SET REC_CURR_IND = 'N'
    WHERE CUSTOMER_ID = {_customer_id} 
    and TIERED_PRICE_LIM = '{_tiered_price_lim}' 
    and GO_LIVE = '{_go_live}'
    and RUN_TYPE = '{_run_type}'
    '''
