# -*- coding: utf-8 -*-
"""CPMO_plan_liability

This module contains the functions that are used to create and calculate the
plan liablity of a Med-D plan based on cost shares and copays.

This script requires pandas and PuLP be installed within the Python
environment you are running these functions in

This module contains the following functions:
      *
"""

# Import required parameters file and packages
import CPMO_parameters as p
import pandas as pd
import numpy as np
from pulp import *

###################################################### PLAN LAMBDAS FUNCTIONS ##################################################
"""
CPMO_plan_liability: Create Plan Lambdas Functions

Plan Liability function for creating the plan lambdas and pulp variables

"""
def genPlanRegString(row):
    '''
    This function takes a row that contains the CLIENT BREAKOUT and REGION combination and attaches the
    _plan_lambda string.
    Inputs:
        df - dataframe row that contains the CLIENT BREAKOUT and REGION fields, etc.
    Output:
        string containing unique CLIENT BREAKOUT and REGION combination with _plan_lambda.
    '''
    plan_lambda_string = row['CLIENT'] + '_' +\
                         row['BREAKOUT'] + '_' +\
                         row['REGION']
    plan_lambda_string += '_pl_lambda'

    return plan_lambda_string

def genPlanLambda(row):
    '''
    This function takes a row that contains the string of CLIENT BREAKOUT REGION, and _plan_lambda and creates the pulp variable.
    Inputs:
        df - dataframe row that contains the string of CLIENT, BREAKOUT, REGION and _plan_lambda.
    Output:
        A pulp variable consiting of the plan lambdas.
    '''
    return pulp.LpVariable(row['Plan_Liab_Lambda_String'])


def getPlanLambdas(lp_data_df, client_list=p.PLAN_LIAB_CLIENTS):
    '''
    This function takes the genPlanRegString function and the genPlanLambda function and applies them to the dataframe
    on each unique combination of CLIENT, BREAKOUT and REGION based of selected Clients from parameters script.
    Inputs:
        df - dataframe and client_list from the parameters script.
    Output:
        df - dataframe of unique combinations of CLIENT, BREAKOUT and REGION including the plan lambda pulp variables.
    '''
    client_list =  ['WELLCARE']
    lp_data_pl_clients = lp_data_df.loc[lp_data_df.CLIENT.isin(client_list),
                                        ['CLIENT', 'BREAKOUT', 'REGION']]
    distinct_regions = lp_data_pl_clients.drop_duplicates().reset_index()
    distinct_regions['Plan_Liab_Lambda_String'] = distinct_regions.apply(genPlanRegString,
                                                                         axis=1)
    distinct_regions['Plan_Liab_Lambda'] = distinct_regions.apply(genPlanLambda,
                                                                         axis=1)
    return distinct_regions


def calcPlanCost(lp_data_df, mac_prc_col='New_Price', calc_30_daysup=True):
    '''
    This function creates a list and dataframe of plan costs for either New Price, Effective Capped Price New, or
    Effective Capped Price. The function filters on projected quantity of greater than zero.
    Inputs:
        df - dataframe lp_data_df, mac_prc_col setting, and calc_30_daysup parameter.
    Output:
        list of plan costs by CLIENT, BREAOUT and REGION, and dataframe containing the sum of ICL, GAP, and CAT costs.
    '''
    if calc_30_daysup:
        lp_data_df['PER_UNIT_30DAYS'] = (lp_data_df.QTY_PROJ_EOY/lp_data_df.DAYSSUP_PROJ_EOY) * 30
        lp_data_df['NUM30DAYS'] = lp_data_df.DAYSSUP_PROJ_EOY  / 30

    lp_data_df['Allowed_Cost'] = lp_data_df[mac_prc_col] * lp_data_df['PER_UNIT_30DAYS'] + lp_data_df['DISP_FEE']

    #Will need to parameterize the 75% for Gap and 15% for Catastrophic
    #Allowed Cost modifier if this does not hold for other clients
    lp_data_df['ICL_COPAY'] = lp_data_df['COPAY']
    lp_data_df.loc[lp_data_df.COPAY_RATE > 0.00, 'ICL_COPAY'] = lp_data_df.loc[lp_data_df.COPAY_RATE > 0.00, 'Allowed_Cost'] * lp_data_df.loc[lp_data_df.COPAY_RATE > 0.00, 'COPAY_RATE']
    lp_data_df['ICL_MAX'] = np.maximum(lp_data_df['Allowed_Cost'] - lp_data_df['ICL_COPAY'],0)
    lp_data_df['ICL_Cost'] = lp_data_df['PROP_ICL'] * lp_data_df['ICL_MAX'] * lp_data_df['NUM30DAYS']

    lp_data_df['GAP_Cost'] = lp_data_df['PROP_GAP'] * (lp_data_df['Allowed_Cost'] * (1- lp_data_df['PROP_LIS']) * .75) * lp_data_df['NUM30DAYS']
    lp_data_df['CAT_Cost'] = lp_data_df['PROP_CAT'] * (lp_data_df['Allowed_Cost'] * .15) * lp_data_df['NUM30DAYS']

    plan_costs = dict()

    for client in lp_data_df.CLIENT.unique():
        breakout_list = lp_data_df.loc[lp_data_df.CLIENT==client, 'BREAKOUT'].unique()
        for breakout in breakout_list:
            reg_list = lp_data_df.loc[(lp_data_df.CLIENT==client) &
                                      (lp_data_df.BREAKOUT==breakout), 'REGION'].unique()
            for reg in reg_list:
                dict_key = client + '_' + breakout + '_' + reg
                reg_data = lp_data_df.loc[(lp_data_df.CLIENT==client) &
                                          (lp_data_df.BREAKOUT==breakout) &
                                          (lp_data_df.REGION==reg)]

                cost = reg_data['ICL_Cost'].sum() + reg_data['GAP_Cost'].sum() + reg_data['CAT_Cost'].sum()
                plan_costs[dict_key] = cost

    return plan_costs, lp_data_df



###################################################### ICL COST FUNCTIONS ##################################################
"""
CPMO_plan_liability: Create ICLCosts Functions

Plan Liability function for creating the ICL Cost Over decision
variable and its constraints

"""
def createICLDecisionVariable(lp_data_df):
    '''
    This function creates an ICL pulp decision variable for use in the LP based on the GPI_NDC,
    Client, Breakout, Measurement, Region, & Pharmacy.
    Inputs:
        df - dataframe with ICL decision variable fields.
    Output:
        pulp variable for ICL Cost Over with lower bound of 0.
    '''
    ICL_var = str('ICL_'+str(lp_data_df.GPI_NDC) + '_' + str(lp_data_df.CLIENT) + '_' + str(lp_data_df.BREAKOUT) +
                            '_' + str(lp_data_df.MEASUREMENT) + '_' + str(lp_data_df.REGION) +'_' + str(lp_data_df.CHAIN_GROUP) +'_' + str(lp_data_df.CHAIN_SUBGROUP))

    return pulp.LpVariable(ICL_var, lowBound=0)


def generateICLCostDecisionVariables_andConstraints(lp_data_df):
    '''
    This function generates the ICL decision variable name as well as calls the function to create the ICL decision PuLP variable.
    Inputs:
        df - dataframe with Client, Breakout, Region, Pharmacy, GPI-NDC, MAC Price, Pricing Bounds, etc.
    Output:
        df - dataframe with ICL Cost Over variable names and pulp variables. AA list of ICL Cost Over constraints.
    '''
    # Calculate the 30 per unit quantity - and ICLcost decision varialbe nam, and dispensingfee and copay variable
    lp_data_df['PER_UNIT_30DAYS'] = np.where(lp_data_df['DAYSSUP_PROJ_EOY'] == 0, 0,
                                          (lp_data_df['QTY_PROJ_EOY']/lp_data_df['DAYSSUP_PROJ_EOY'])*30)
    lp_data_df['DISP_FEE_COPAY'] = lp_data_df.DISP_FEE - lp_data_df.COPAY
    # Create the ICL pulp decision variable and decision variable name for each row
    lp_data_df['ICL_Cost_Over'] = lp_data_df.apply(createICLDecisionVariable, axis=1)
    lp_data_df['ICL_Dec_Var_Name'] = lp_data_df.apply(lambda lp_data_df: 'ICL_'+str(lp_data_df.GPI_NDC) + '_' + str(lp_data_df.CLIENT) +
                                                                         '_' + str(lp_data_df.BREAKOUT) + '_' + str(lp_data_df.MEASUREMENT) +
                                                                         '_' + str(lp_data_df.REGION) +'_' + str(lp_data_df.CHAIN_GROUP) + 
                                                                         '_' + str(lp_data_df.CHAIN_SUBGROUP), axis=1)
    # Create ICLCost contraints, for every row create a linear equation and store in list
    # Create list of ICLCost constraints
    icl_cons_list = []
    idx_dec_var = lp_data_df.columns.get_loc("Price_Decision_Var")
    idx_per_unit = lp_data_df.columns.get_loc("PER_UNIT_30DAYS")
    idx_disp_copay = lp_data_df.columns.get_loc("DISP_FEE_COPAY")
    idx_copay_rate = lp_data_df.columns.get_loc('COPAY_RATE')
    idx_disp_fee = lp_data_df.columns.get_loc('DISP_FEE')
    idx_icl_over = lp_data_df.columns.get_loc("ICL_Cost_Over")

    # Create linear equation by row for ICLCost constraints: AllowedCost - Copay <= 0
    for i in range(1, lp_data_df.shape[0]):
        icl_cons = ""
        if lp_data_df.iloc[i, idx_copay_rate] > 0.00:
            icl_cons += (1- lp_data_df.iloc[i,idx_copay_rate]) * lp_data_df.iloc[i,idx_per_unit] * lp_data_df.iloc[i,idx_dec_var] + (1- lp_data_df.iloc[i,idx_copay_rate]) * lp_data_df.iloc[i,idx_disp_fee] - lp_data_df.iloc[i,idx_icl_over]
        else:
            icl_cons += lp_data_df.iloc[i,idx_per_unit]*lp_data_df.iloc[i,idx_dec_var] + lp_data_df.iloc[i,idx_disp_copay] - lp_data_df.iloc[i,idx_icl_over]
        # Create list of ICLCost Constraints
        icl_cons_list.append(icl_cons)

    return lp_data_df, icl_cons_list



###################################################### PLAN COST CONSTRAINT FUNCTIONS ##################################################
"""
CPMO_plan_liability: Create Plan Cost Constraints

Plan Liability function for creating plan cost constraints for ICL, GAP, and CAT
The plan cost constraints are at the CLIENT, BREAKOUT and REGION level

"""

def generate_constraint_plancost(lp_data_df, plan_lambda_df):
    '''
    This function generates the pulp LpAffinneExpression plan cost constraints.
    Inputs:
        df - dataframe with Client, Breakout, Region, Pharmacy, GPI-NDC, MAC Price, Pricing Bounds, etc.
        plan lambda dataframe of plan lambda decision variables filtered for CLIENT, BREAKOUT and REGION.
    Output:
        pulp LpAffineExpressions creating a list of plan cost contraints.
    '''
    # Step 1: Create the calculated value for NUM30DAYS supply
    lp_data_df['NUM30DAYS'] = lp_data_df.DAYSSUP_PROJ_EOY/30
    lp_data_df['PER_UNIT_30DAYS'] = np.where(lp_data_df['DAYSSUP_PROJ_EOY'] == 0, 0,
                                          (lp_data_df['QTY_PROJ_EOY']/lp_data_df['DAYSSUP_PROJ_EOY'])*30)

    lp_data_df['ICL_WT'] = lp_data_df.PROP_ICL * lp_data_df.NUM30DAYS * lp_data_df.MOV_FAC
    lp_data_df['GAP_WT'] = ((1-lp_data_df.PROP_LIS) * lp_data_df.NUM30DAYS * lp_data_df.PROP_GAP) * 0.75 * lp_data_df.MOV_FAC
    lp_data_df['CAT_WT'] = (0.15) * lp_data_df.NUM30DAYS * lp_data_df.PROP_CAT * lp_data_df.MOV_FAC

   # Step 2: Create the LpAffineExpression
    plan_cost_cons = []
    client_list = lp_data_df.CLIENT.unique()
    idx_dec_var = lp_data_df.columns.get_loc("Price_Decision_Var")
    idx_per_unit = lp_data_df.columns.get_loc("PER_UNIT_30DAYS")
    idx_disp = lp_data_df.columns.get_loc("DISP_FEE")
    idx_gapwt = lp_data_df.columns.get_loc("GAP_WT")
    idx_catwt = lp_data_df.columns.get_loc("CAT_WT")

   # Step 3: Create DOT Product of the ICL Cost Var * Prop ICL * Num30daysSupply summed by CLIENT, BREAKOUT, REGION
   # Create scalar of Dispensing Fee * GAP Weight/CAT Weight for the LpAffineExpression functions
    lp_data_df['DISP_GAP'] = lp_data_df['DISP_FEE'] * lp_data_df['GAP_WT']
    lp_data_df['DISP_CAT'] = lp_data_df['DISP_FEE'] * lp_data_df['CAT_WT']
    cons_ = LpAffineExpression()

    for client in client_list:
        breakout_list = lp_data_df.loc[(lp_data_df.CLIENT==client), 'BREAKOUT'].unique()
        for breakout in breakout_list:
            reg_list = lp_data_df.loc[(lp_data_df.CLIENT==client) & (lp_data_df.BREAKOUT==breakout), 'REGION'].unique()
            for reg in reg_list:
                reg_data = lp_data_df.loc[(lp_data_df.CLIENT==client) &
                                          (lp_data_df.BREAKOUT==breakout) &
                                          (lp_data_df.REGION==reg)]

                plan_lambda = plan_lambda_df.loc[(plan_lambda_df.CLIENT==client) & (plan_lambda_df.BREAKOUT==breakout) & (plan_lambda_df.REGION==reg)].Plan_Liab_Lambda.values[0]

                cons_ = ""

                cons_ += LpAffineExpression([(reg_data.ICL_Cost_Over.values[i],
                                              reg_data.ICL_WT.values[i])
                                             for i in range(reg_data.shape[0])])

                cons_ += LpAffineExpression([(reg_data.Price_Decision_Var.values[i],
                                               reg_data.PER_UNIT_30DAYS.values[i] * reg_data.GAP_WT.values[i])
                                               for i in range(reg_data.shape[0])]) + reg_data.DISP_GAP.sum()

                cons_ += LpAffineExpression([(reg_data.Price_Decision_Var.values[i],
                                               reg_data.PER_UNIT_30DAYS.values[i] * reg_data.CAT_WT.values[i])
                                               for i in range(reg_data.shape[0])]) + reg_data.DISP_CAT.sum()

                cons_ += -plan_lambda
                plan_cost_cons.append(cons_)

    return plan_cost_cons


###################################################### PLAN LIABILITY OUTPUT FUNCTION ##################################################
"""
CPMO_plan_liability: Create Plan Liability Output Files

Plan Liability function for creating ouptut generated by the plan lambdas, ICL, and plan cost constraint
functions used in the overall mac opt script

"""

def generatePlanLiabilityOutput(lp_data_df, new_old_price, lag_price_col, temp_work_dir=''):
    '''
    This function generates the ouptut excel file for the client (WELLCARE), which consists of the old and new prices, AWP,
    old and new ingredient cost, old and new plan lambdas, etc.
    Inputs:
        df - dataframe with CLIENT, BREAKOUT, REGION, Pharmacy, GPI-NDC, MAC Price, Pricing Bounds, etc.
        df - datframe of merged new old price columns from output dictionary calcPlanCost (for retreiving the lambdas).
    Output:
        df - dataframe and writer creating the excel workbook with multiple sheets (on MAC List, Not On MAC List).
    '''
    # Filter on QTY_PROJ_EOY > 0 and MEASUREMENT not equal to MAIL30
    lp_data_df = lp_data_df.loc[(lp_data_df.QTY_PROJ_EOY > 0) & (lp_data_df.MEASUREMENT != 'M30')]

    # Create flag for if on MAC List
    lp_data_df['On_MAC_List'] = np.where(lp_data_df['CURRENT_MAC_PRICE']> 0, 1, 0)
    lp_data_df['30-day Scripts'] = lp_data_df.DAYSSUP_PROJ_EOY/30

    # Rename Columns
    lp_data_df.rename(columns={'PKG_SZ':'GPPC','NDC':'NDC11'}, inplace=True)
    reg_columns1 = ['REGION','On_MAC_List', 'MEASUREMENT',
                    lag_price_col, 'Eff_capped_price_new',
                    #'Eff_capped_price', 'Eff_capped_price_new',
                    'FULLAWP_ADJ_PROJ_EOY',
                    #'Price_Reimb_Proj', 'Old_Price_Effective_Reimb_Proj_EOY',
                    'OLD_INGREDIENT_COST', 'NEW_INGREDIENT_COST',
                    'CHAIN_GROUP', 'CHAIN_SUBGROUP', 'GPI', 'NDC11', 'Pharmacy_Type', '30-day Scripts', 'QTY_PROJ_EOY', 'TIER']
    reg_full_df1 = lp_data_df[reg_columns1]
    reg_full_df1.rename(columns = {lag_price_col: 'Old Effective Price',
                                   'Eff_capped_price_new': 'New Effective Price',
                                   #'Eff_capped_price': 'Old Price',
                                   #'Eff_capped_price_new': 'New Price',
                                   'OLD_INGREDIENT_COST': 'Old Ingredient Cost',
                                   'NEW_INGREDIENT_COST': 'New Ingredient Cost',
                                   'FULLAWP_ADJ_PROJ_EOY': 'AWP',
                                   'QTY_PROJ_EOY': 'QTY'}, inplace=True)

    reg_full_df1 = pd.merge(reg_full_df1, new_old_price[['PLAN_COST_NEW','PLAN_COST_OLD','ICL_Cost_x', 'ICL_Cost_y', 'GAP_Cost_x', 'GAP_Cost_y', 'CAT_Cost_x', 'CAT_Cost_y']], how='left', on=None, left_index=True, right_index=True)
    reg_full_df1.rename(columns={'PLAN_COST_NEW': 'New Plan Liability', 'PLAN_COST_OLD': 'Old Plan Liability',
                                 'ICL_Cost_x': 'New ICL Cost', 'ICL_Cost_y': 'Old ICL Cost',
                                 'GAP_Cost_x': 'New GAP Cost', 'GAP_Cost_y': 'Old GAP Cost',
                                 'CAT_Cost_x': 'New CAT Cost', 'CAT_Cost_y': 'Old CAT Cost'}, inplace=True)

    # reg_full_agg = reg_full_df1.groupby(by = ['REGION','On_MAC_List', 'CHAIN_GROUP', 'GPI', 'NDC11', 'Pharmacy_Type', 'TIER'])
    # reg_full_df = reg_full_agg['Old Ingredient Cost', 'New Ingredient Cost', 'AWP', 'New Plan Liability', 'Old Plan Liability','New ICL Cost','Old ICL Cost','New GAP Cost', 'Old GAP Cost','New CAT Cost','Old CAT Cost','30-day Scripts', 'QTY'].agg(sum)
    # reg_full_df = pd.concat([reg_full_df, reg_full_agg['Old Price', 'New Price'].agg(np.nanmean)], axis=1).reset_index()
    reg_full_df = reg_full_df1#.loc[reg_full_df1.MEASUREMENT == 'R30']

    reg_output_On_MAC_List = pd.pivot_table(reg_full_df.loc[reg_full_df.On_MAC_List ==1], index=['REGION','On_MAC_List','MEASUREMENT','GPI', 'NDC11'],
                                            columns=['Pharmacy_Type','CHAIN_GROUP', 'CHAIN_SUBGROUP'], values=['Old Effective Price','AWP','New Ingredient Cost','Old Ingredient Cost',
                                                    'New Effective Price','New Plan Liability','Old Plan Liability','New ICL Cost','Old ICL Cost','New GAP Cost', 'Old GAP Cost','New CAT Cost','Old CAT Cost', 'QTY', '30-day Scripts', 'TIER'], aggfunc=np.nanmax, fill_value = 0)
    # reg_output_Not_On_MAC_List = pd.pivot_table(reg_full_df.loc[reg_full_df.On_MAC_List ==0], index=['REGION','On_MAC_List','GPI', 'NDC11'],
    #                                         columns=['Pharmacy_Type','CHAIN_GROUP'], values=['Old Price','AWP','New Ingredient Cost','Old Ingredient Cost',
    #                                                 'New Price','New Plan Liability','Old Plan Liability'], aggfunc=np.nanmax)
    reg_output_On_MAC_List.reset_index(level=0, inplace=True)
    region_list = {}

    fname = p.TIMESTAMP + 'PL_Output_Full.xlsx'
    if 'gs://' in p.FILE_OUTPUT_PATH:
        temp_fpath = os.path.join(temp_work_dir, fname)
        cloud_path = os.path.join(p.FILE_OUTPUT_PATH, fname)
        writer = pd.ExcelWriter(temp_path, engine='xlsxwriter')
    else:
        writer = pd.ExcelWriter(os.path.join(p.FILE_OUTPUT_PATH, fname), engine='xlsxwriter')

    reg_output_On_MAC_List.to_excel(writer, sheet_name='ON_MAC_LIST', index=True)
    #reg_output_Not_On_MAC_List.to_excel(writer, sheet_name='NOT_ON_MAC_LIST', index=True)

    writer.save()
    writer.close()
    if 'gs://' in p.FILE_OUTPUT_PATH:
        uf.upload_blob(p.BUCKET, temp_fpath, cloud_path)

    return lp_data_df

    for region in reg_output_On_MAC_List['REGION'].unique():
        reg_df = reg_output_On_MAC_List[reg_output_On_MAC_List['REGION'] == region]
        region_list[region] = reg_df

    def save_xlxs(region_list, path):
        '''
        Save a dictionary of dataframes to an exel file, with each dataframe as a seperate page
        '''
        with pd.ExcelWriter(path) as writer:
            for key in region_list:
                region_list[key].to_excel(writer, key, index=True)
        writer.save()
        writer.close()

    save_xlxs(region_list, p.FILE_OUTPUT_PATH + 'Drug_List.xlsx')
    return lp_data_df


###################################################### CALL FUNCTIONS ##################################################
"""
CPMO_plan_liability: Create function that returns outputs for each function built

Plan Liability function for creating ouptut pf specific functions in the plan liability module to be used in the
overall mac opt script

"""

def createPlanCostObj(lp_data_df):
    '''
    This function calls the function in the CPMO plan liability script to run and create output in the overall mac opt script.
    Inputs:
        df - dataframe with Client, Breakout, Region, Pharmacy, GPI-NDC, MAC Price, Pricing Bounds, etc.
    Output:
        df - dataframes and lists of constraints from called functions within function.
    '''
    plan_lambdas = getPlanLambdas(lp_data_df)

    lp_data_df, icl_cons = generateICLCostDecisionVariables_andConstraints(lp_data_df)

    plan_cost_cons = generate_constraint_plancost(lp_data_df, plan_lambdas)

    return lp_data_df, plan_lambdas, icl_cons, plan_cost_cons
