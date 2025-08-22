# -*- coding: utf-8 -*-
"""
Shared functions (non-LP) file for CLIENT_PHARMACY_MAC_OPTIMIZATION

@author: JOHN WALKER
@version: 1.1.0, 01.27.2020
"""
import CPMO_parameters as p
import pandas as pd
import numpy as np
import copy
import logging


def standardize_df(df):
    '''
    This is a series of common steps to get all dataframes and data inputs into the same
    general format.
    Inputs:
        df - a dataframe that needs to be standardized
    Outputs:
        df - the standardized dataframe
    '''

    # GIT - Yiwei: Captialize column names
    # Add captialization to all names to standardize the input for all clients
    df.columns = map(str.upper, df.columns)

    # Captilaize everything in the following columns to make sure that we can use string matching
    for column in ['CLIENT', 'BREAKOUT', 'REGION', 'MEASUREMENT', 'CHAIN_GROUP']:
        if column in df.columns:
            df[column] = df[column].apply(lambda x: x.upper())

    # Different data sources use different names for Wellcare so this standarized them to "WELLCARE"
    if 'CLIENT' in df.columns:
        df.loc[df.CLIENT == 'WC', 'CLIENT'] = 'WELLCARE'

    # Ensures that no GPI loses its leading 0
    if 'GPI' in df.columns:
        if df.GPI.dtype == float:
            df.GPI = df.GPI.astype('int64')

        df.GPI = df.GPI.astype(str).apply(lambda x: x.split('.')[0])
        df.GPI = ('00' + df.GPI).str[-14:]

    # Resolves naming differences of NDC columns between different data sources
    if 'NDC11' in df.columns:
        df.rename(columns={'NDC11': 'NDC'}, inplace=True)

    # Ensures that NDCs do not lose leading 0
    if 'NDC' in df.columns:
        if df.NDC.dtype == float:
            df.NDC = df.NDC.astype('int64')

        df.NDC = df.NDC.astype(str).apply(lambda x: x.split('.')[0])
        df.NDC = ('0000' + df.NDC).str[-11:]

    # Creates a common column of GPI and NDC for string matching
    if ('GPI' in df.columns) & ('NDC' in df.columns):
        df['GPI_NDC'] = df.GPI + '_' + df.NDC

    return df


def dict_to_df(dictionary, column_names):
    '''
    Simple helper function that takes a dictionary and turns it into a dataframe with two columns:
    one for keys and one for values.
    Inputs:
        dictionary: the dictionary that you want to change to a dataframe
        column_names: the column names that will be used in the dataframe for the key and values, repectively
    Outputs:
        df: the dataframe of the dictionary keys and values
    '''
    df = pd.DataFrame(columns=column_names)
    for key in dictionary:
        df = df.append({column_names[0]: key,
                        column_names[1]: dictionary[key]}, ignore_index=True)
    return df


def df_to_dict(df, column_names):
    '''
    Simple helper function that takes a pandas dataframe and turns it into a dictionary with key values pairings for
    every row in the dataframe.  This only works if there are unique keys in the dataframe
    Inputs:
        df: the dataframe that you want to change to a dictionary
        column_names: the column names that are used in the dataframe for the key and values, repectively
    Outputs:
        temp_dict: the dictionary keys and values from each row of the dataframe
    '''
    df = df.reset_index(drop=True)
    temp_dict = dict()
    for i in range(len(df)):
        temp_dict[df[column_names[0]][i]] = df[column_names[1]][i]

    return temp_dict


def calculatePerformance(data_df, client_guarantees, pharmacy_guarantees, client_list, pharmacy_list, oc_pharm_perf,
                         gen_launch_perf, pharm_approx, days=0, reimb_column='PRICE_REIMB_ADJ',
                         client_AWP_column='FULLAWP_ADJ', pharm_AWP_column='FULLAWP_ADJ', AWP_column='',
                         restriction='none', other=False):  # Still needs work
    '''
    Input:
        data_df: DataFrame which has the data for months that we have data for all the Pharmacies and regions
    Output:
        Dictionary of performances for CVS, Non Preferred Capped Chains, SSI, & Walgreens
    '''
    #    cvs = 'CVS'
    #    non_pref_capped = ['RAD', 'KRG', 'WAG', 'WMT']

    # client_list = data_df.CLIENT.unique()
    if len(AWP_column) > 0:
        client_AWP_column = AWP_column
        pharm_AWP_column = AWP_column

    perf = dict()
    if restriction != 'pharm':
        for client in client_list:  # loop through clients
            #            logging.debug(client)
            breakout_list = data_df.loc[data_df['CLIENT'] == client, 'BREAKOUT'].unique()
            for breakout in breakout_list:  # loop through breakouts
                #                logging.debug(breakout)
                perf_temp = 0
                region_list = data_df.loc[
                    (data_df['CLIENT'] == client) & (data_df['BREAKOUT'] == breakout), 'REGION'].unique()
                spend = 0
                target = 0
                awp = 0
                for region in region_list:
                    #                    logging.debug(region)
                    measurement_list = data_df.loc[(data_df['CLIENT'] == client) & (data_df['BREAKOUT'] == breakout) & (
                                data_df['REGION'] == region), 'MEASUREMENT'].unique()
                    for measurement in measurement_list:
                        #                        logging.debug(measurement)
                        pref_perf = 0
                        npref_perf = 0
                        if data_df.loc[(data_df['CLIENT'] == client) & (data_df['BREAKOUT'] == breakout) & (
                                data_df['REGION'] == region) & (data_df['MEASUREMENT'] == measurement) & (
                                               data_df['PHARMACY_TYPE'] == 'Preferred'), 'PHARMACY_TYPE'].any():
                            #                            logging.debug('Preferred Exists')
                            preferred_df = data_df.loc[
                                (data_df['CLIENT'] == client) & (data_df['BREAKOUT'] == breakout) & (
                                            data_df['REGION'] == region) & (data_df['MEASUREMENT'] == measurement) & (
                                            data_df['PHARMACY_TYPE'] == 'Preferred')]
                            perf_guarantee = client_guarantees.loc[(client_guarantees['CLIENT'] == client) & (
                                        client_guarantees['BREAKOUT'] == breakout) & (client_guarantees[
                                                                                          'REGION'] == region) & (
                                                                               client_guarantees[
                                                                                   'MEASUREMENT'] == measurement) & (
                                                                               client_guarantees[
                                                                                   'PHARMACY_TYPE'] == 'Preferred'), 'RATE'].values[
                                0]
                            #                            logging.debug(perf_guarantee)
                            pref_perf = (1 - perf_guarantee) * preferred_df[client_AWP_column].sum() - preferred_df[
                                reimb_column].sum()
                            #                            logging.debug('Preferred AWP: ', preferred_df[client_AWP_column].sum())
                            awp += preferred_df[client_AWP_column].sum()
                            #                            logging.debug('Preferred Target: ', (1-perf_guarantee)*preferred_df[client_AWP_column].sum())
                            target += (1 - perf_guarantee) * preferred_df[client_AWP_column].sum()
                            #                            logging.debug('Preferred Spend: ', preferred_df[reimb_column].sum())
                            spend += preferred_df[reimb_column].sum()
                        #                            logging.debug('Preferred: ' + str(pref_perf))

                        if data_df.loc[(data_df['CLIENT'] == client) & (data_df['BREAKOUT'] == breakout) & (
                                data_df['REGION'] == region) & (data_df['MEASUREMENT'] == measurement) & (
                                               data_df['PHARMACY_TYPE'] == 'Non_Preferred'), 'PHARMACY_TYPE'].any():
                            npreferred_df = data_df.loc[
                                (data_df['CLIENT'] == client) & (data_df['BREAKOUT'] == breakout) & (
                                            data_df['REGION'] == region) & (data_df['MEASUREMENT'] == measurement) & (
                                            data_df['PHARMACY_TYPE'] == 'Non_Preferred')]
                            npref_guarantee = client_guarantees.loc[(client_guarantees['CLIENT'] == client) & (
                                        client_guarantees['BREAKOUT'] == breakout) & (client_guarantees[
                                                                                          'REGION'] == region) & (
                                                                                client_guarantees[
                                                                                    'MEASUREMENT'] == measurement) & (
                                                                                client_guarantees[
                                                                                    'PHARMACY_TYPE'] == 'Non_Preferred'), 'RATE'].values[
                                0]

                            npref_perf = (1 - npref_guarantee) * npreferred_df[client_AWP_column].sum() - npreferred_df[
                                reimb_column].sum()
                            #                            logging.debug(npref_guarantee)
                            #                            logging.debug('NPreferred AWP: ', npreferred_df[client_AWP_column].sum())
                            awp += npreferred_df[client_AWP_column].sum()
                            #                            logging.debug('NPreferred Target: ', (1-npref_guarantee)*npreferred_df[client_AWP_column].sum())
                            target += (1 - npref_guarantee) * npreferred_df[client_AWP_column].sum()
                            #                            logging.debug('NPreferred Spend: ', npreferred_df[reimb_column].sum())
                            spend += npreferred_df[reimb_column].sum()
                        #                            logging.debug('Non Preferred: ' + str(npref_perf))

                        perf_temp += (pref_perf + npref_perf)

                logging.debug(client + '_' + breakout)
                logging.debug('AWP: %f', awp)
                logging.debug('Target: %f', target)
                logging.debug('Spend: %f', spend)
                logging.debug('Perf: %f', perf_temp)
                perf[client + '_' + breakout] = perf_temp + gen_launch_perf[client + '_' + breakout]

    if restriction != 'client':
        pharm_clients = copy.deepcopy(client_list)
        if other:
            pharm_clients = np.append(pharm_clients, 'OTHER')
        for pharmacy in pharmacy_list:
            logging.debug(pharmacy)
            perf[pharmacy] = 0

            for client in pharm_clients:

                pharmacy_data = data_df.loc[(data_df.CHAIN_GROUP == pharmacy) & (data_df.CLIENT == client)]
                pharmacy_guarantee = pharmacy_guarantees.loc[(pharmacy_guarantees['PHARMACY'] == pharmacy) & (
                            pharmacy_guarantees['CLIENT'] == client), 'RATE'].values[0]
                if client == 'OTHER':
                    scale_factor = 1
                    intercept = 0
                else:
                    scale_factor = pharm_approx.loc[
                        (pharm_approx.CLIENT == client) & (pharm_approx.CHAIN_GROUP == pharmacy), 'SLOPE'].values[0]
                    intercept = pharm_approx.loc[(pharm_approx.CLIENT == client) & (
                                pharm_approx.CHAIN_GROUP == pharmacy), 'INTERCEPT'].values[0] * days
                logging.debug(client)
                logging.debug('Guarantee: %f', pharmacy_guarantee)
                logging.debug('AWP: %f', pharmacy_data[pharm_AWP_column].sum())
                logging.debug('Target: %f', (1 - pharmacy_guarantee) * pharmacy_data[pharm_AWP_column].sum())
                logging.debug('Spend: %f', pharmacy_data[reimb_column].sum())
                logging.debug('Perf: %f',
                              (1 - pharmacy_guarantee) * pharmacy_data[pharm_AWP_column].sum() - pharmacy_data[
                                  reimb_column].sum())

                raw_performance = (1 - pharmacy_guarantee) * pharmacy_data[pharm_AWP_column].sum() - pharmacy_data[
                    reimb_column].sum()
                perf[pharmacy] += scale_factor * raw_performance + intercept
                logging.debug('Scale Factor: %f', scale_factor)
                logging.debug('Intercept: %f', intercept)
                logging.debug('New Perf: %f', scale_factor * raw_performance + intercept)
            perf[pharmacy] += oc_pharm_perf[pharmacy]
            perf[pharmacy] += gen_launch_perf[pharmacy]
    return perf


def calculatePerformance2(data_df, client_guarantees, pharmacy_guarantees, client_list, pharmacy_list, oc_pharm_perf,
                          gen_launch_perf, pharm_approx, days=0, reimb_column='PRICE_REIMB_ADJ',
                          client_AWP_column='FULLAWP_ADJ', pharm_AWP_column='FULLAWP_ADJ', AWP_column='',
                          restriction='none', other=False, qty_column=False, full_disp=False):  # Still needs work
    '''
    Calculates the performance of the clients and pharmacies as a dollar surplus given all of the inputs and then outputs the client and pharmacy performances
    Input:
        data_df: DataFrame which has the data for months that we have data for all the Pharmacies and regions
    Output:
        Dictionary of performances for CVS, Non Preferred Capped Chains, SSI, & Walgreens
    '''
    #    cvs = 'CVS'
    #    non_pref_capped = ['RAD', 'KRG', 'WAG', 'WMT']

    # Check if certain columns are in data dataframe
    if qty_column:
        disp_qty = True
    else:
        disp_qty = False

    # client_list = data_df.CLIENT.unique()
    if len(AWP_column) > 0:
        client_AWP_column = AWP_column
        pharm_AWP_column = AWP_column

    perf = dict()
    if restriction != 'pharm':
        for client in client_list:  # loop through clients
            # logging.debug(client)
            breakout_list = data_df.loc[data_df['CLIENT'] == client, 'BREAKOUT'].unique()
            for breakout in breakout_list:  # loop through breakouts
                # logging.debug(breakout)
                perf_temp = 0
                region_list = data_df.loc[
                    (data_df['CLIENT'] == client) & (data_df['BREAKOUT'] == breakout), 'REGION'].unique()
                spend = 0
                target = 0
                awp = 0
                for region in region_list:
                    logging.debug(region)
                    measurement_list = data_df.loc[(data_df['CLIENT'] == client) & (data_df['BREAKOUT'] == breakout) & (
                                data_df['REGION'] == region), 'MEASUREMENT'].unique()
                    for measurement in measurement_list:
                        logging.debug(measurement)
                        pref_perf = 0
                        npref_perf = 0
                        if data_df.loc[(data_df['CLIENT'] == client) & (data_df['BREAKOUT'] == breakout) & (
                                data_df['REGION'] == region) & (data_df['MEASUREMENT'] == measurement) & (
                                               data_df['PHARMACY_TYPE'] == 'Preferred'), 'PHARMACY_TYPE'].any():
                            #                            logging.debug('Preferred Exists')
                            preferred_df = data_df.loc[
                                (data_df['CLIENT'] == client) & (data_df['BREAKOUT'] == breakout) & (
                                            data_df['REGION'] == region) & (data_df['MEASUREMENT'] == measurement) & (
                                            data_df['PHARMACY_TYPE'] == 'Preferred')]
                            perf_guarantee = client_guarantees.loc[(client_guarantees['CLIENT'] == client) & (
                                        client_guarantees['BREAKOUT'] == breakout) & (client_guarantees[
                                                                                          'REGION'] == region) & (
                                                                               client_guarantees[
                                                                                   'MEASUREMENT'] == measurement) & (
                                                                               client_guarantees[
                                                                                   'PHARMACY_TYPE'] == 'Preferred'), 'RATE'].values[
                                0]
                            #                            logging.debug(perf_guarantee)
                            pref_perf = (1 - perf_guarantee) * preferred_df[client_AWP_column].sum() - preferred_df[
                                reimb_column].sum()
                            #                            logging.debug('Preferred AWP: ', preferred_df[client_AWP_column].sum())
                            awp += preferred_df[client_AWP_column].sum()
                            #                            logging.debug('Preferred Target: ', (1-perf_guarantee)*preferred_df[client_AWP_column].sum())
                            target += (1 - perf_guarantee) * preferred_df[client_AWP_column].sum()
                            #                            logging.debug('Preferred Spend: ', preferred_df[reimb_column].sum())
                            spend += preferred_df[reimb_column].sum()
                            #                            logging.debug('Preferred: ' + str(pref_perf))

                            # print the extra information if full_disp is on
                            if full_disp:
                                if disp_qty:
                                    print('Preferred MAC Qty: ',
                                          preferred_df.loc[preferred_df['CURRENT_MAC_PRICE'] > 0, qty_column].sum())
                                    print('Preferred NonMAC Qty: ',
                                          preferred_df.loc[preferred_df['CURRENT_MAC_PRICE'] <= 0, qty_column].sum())

                                print('Preferred MAC AWP: ',
                                      preferred_df.loc[preferred_df['CURRENT_MAC_PRICE'] > 0, client_AWP_column].sum())
                                print('Preferred NonMAC AWP: ',
                                      preferred_df.loc[preferred_df['CURRENT_MAC_PRICE'] <= 0, client_AWP_column].sum())

                                print('Preferred MAC Spend: ',
                                      preferred_df.loc[preferred_df['CURRENT_MAC_PRICE'] > 0, reimb_column].sum())
                                print('Preferred NonMAC Spend: ',
                                      preferred_df.loc[preferred_df['CURRENT_MAC_PRICE'] <= 0, reimb_column].sum())

                        if data_df.loc[(data_df['CLIENT'] == client) & (data_df['BREAKOUT'] == breakout) & (
                                data_df['REGION'] == region) & (data_df['MEASUREMENT'] == measurement) & (
                                               data_df['PHARMACY_TYPE'] == 'Non_Preferred'), 'PHARMACY_TYPE'].any():
                            npreferred_df = data_df.loc[
                                (data_df['CLIENT'] == client) & (data_df['BREAKOUT'] == breakout) & (
                                            data_df['REGION'] == region) & (data_df['MEASUREMENT'] == measurement) & (
                                            data_df['PHARMACY_TYPE'] == 'Non_Preferred')]
                            print(client, breakout, region, measurement)
                            npref_guarantee = client_guarantees.loc[(client_guarantees['CLIENT'] == client) & (
                                        client_guarantees['BREAKOUT'] == breakout) & (client_guarantees[
                                                                                          'REGION'] == region) & (
                                                                                client_guarantees[
                                                                                    'MEASUREMENT'] == measurement) & (
                                                                                client_guarantees[
                                                                                    'PHARMACY_TYPE'] == 'Non_Preferred'), 'RATE'].values[
                                0]

                            npref_perf = (1 - npref_guarantee) * npreferred_df[client_AWP_column].sum() - npreferred_df[
                                reimb_column].sum()
                            #                            logging.debug(npref_guarantee)
                            #                            logging.debug('NPreferred AWP: ', npreferred_df[client_AWP_column].sum())
                            awp += npreferred_df[client_AWP_column].sum()
                            #                            logging.debug('NPreferred Target: ', (1-npref_guarantee)*npreferred_df[client_AWP_column].sum())
                            target += (1 - npref_guarantee) * npreferred_df[client_AWP_column].sum()
                            #                            logging.debug('NPreferred Spend: ', npreferred_df[reimb_column].sum())
                            spend += npreferred_df[reimb_column].sum()
                            #                            logging.debug('Non Preferred: ' + str(npref_perf))

                            logging.debug('Perf: %f', pref_perf + npref_perf)

                            if full_disp:
                                if disp_qty:
                                    print('NPreferred MAC Qty: ',
                                          npreferred_df.loc[npreferred_df['CURRENT_MAC_PRICE'] > 0, qty_column].sum())
                                    print('NPreferred NonMAC Qty: ',
                                          npreferred_df.loc[npreferred_df['CURRENT_MAC_PRICE'] <= 0, qty_column].sum())

                                print('NPreferred MAC AWP: ', npreferred_df.loc[
                                    npreferred_df['CURRENT_MAC_PRICE'] > 0, client_AWP_column].sum())
                                print('NPreferred NonMAC AWP: ', npreferred_df.loc[
                                    npreferred_df['CURRENT_MAC_PRICE'] <= 0, client_AWP_column].sum())

                                print('NPreferred MAC Spend: ',
                                      npreferred_df.loc[npreferred_df['CURRENT_MAC_PRICE'] > 0, reimb_column].sum())
                                print('NPreferred NonMAC Spend: ',
                                      npreferred_df.loc[npreferred_df['CURRENT_MAC_PRICE'] <= 0, reimb_column].sum())

                        perf_temp += (pref_perf + npref_perf)

                logging.debug(client + '_' + breakout)
                logging.debug('AWP: %f', awp)
                logging.debug('Target: %f', target)
                logging.debug('Spend: %f', spend)
                logging.debug('Perf: %f', perf_temp)
                perf[client + '_' + breakout] = perf_temp + gen_launch_perf[client + '_' + breakout]

    if restriction != 'client':
        pharm_clients = copy.deepcopy(client_list)
        if other:
            pharm_clients = np.append(pharm_clients, 'OTHER')
        for pharmacy in pharmacy_list:
            logging.debug(pharmacy)
            perf[pharmacy] = 0

            for client in pharm_clients:
                logging.debug(client)
                raw_performance = 0

                if client == 'OTHER':
                    scale_factor = 1
                    intercept = 0
                else:
                    print(client,pharmacy)
                    scale_factor = pharm_approx.loc[
                        (pharm_approx.CLIENT == client) & (pharm_approx.CHAIN_GROUP == pharmacy), 'SLOPE'].values[0]
                    intercept = pharm_approx.loc[(pharm_approx.CLIENT == client) & (
                            pharm_approx.CHAIN_GROUP == pharmacy), 'INTERCEPT'].values[0] * days

                # HACK: Took out breakout != Mail and added data_df['CHAIN_GROUP'] == pharmacy
                breakout_list = data_df.loc[
                    (data_df['CLIENT'] == client) & (data_df['CHAIN_GROUP'] == pharmacy), 'BREAKOUT'].unique()
                if other:
                    breakout_list = ['OTHER']

                for breakout in breakout_list:  # loop through breakouts
                    # logging.debug(breakout)

                    ### GIT: Yiwei - Added CHAIN_GROUP for MCHOICE, works for all clients
                    region_list = data_df.loc[(data_df['CLIENT'] == client) & (data_df['BREAKOUT'] == breakout) & (
                                data_df['CHAIN_GROUP'] == pharmacy), 'REGION'].unique()
                    if other:
                        region_list = ['OTHER']
                    for region in region_list:
                        # logging.debug(region)
                        pharmacy_data = data_df.loc[(data_df.CHAIN_GROUP == pharmacy) &
                                                    (data_df.CLIENT == client) &
                                                    (data_df.BREAKOUT == breakout) &
                                                    (data_df.REGION == region)]

                        pharmacy_guarantee = pharmacy_guarantees.loc[(pharmacy_guarantees['PHARMACY'] == pharmacy) &
                                                                     (pharmacy_guarantees['CLIENT'] == client) &
                                                                     (pharmacy_guarantees['BREAKOUT'] == breakout) &
                                                                     (pharmacy_guarantees[
                                                                          'REGION'] == region), 'RATE'].values[0]

                        # logging.debug('Guarantee: %f', pharmacy_guarantee)
                        # logging.debug('AWP: %f', pharmacy_data[pharm_AWP_column].sum())
                        # logging.debug('Target: %f', (1-pharmacy_guarantee) * pharmacy_data[pharm_AWP_column].sum())
                        # logging.debug('Spend: %f', pharmacy_data[reimb_column].sum())
                        # logging.debug('Perf: %f', (1-pharmacy_guarantee) * pharmacy_data[pharm_AWP_column].sum() - pharmacy_data[reimb_column].sum())
                        raw_performance += (1 - pharmacy_guarantee) * pharmacy_data[pharm_AWP_column].sum() - \
                                           pharmacy_data[reimb_column].sum()

                        if full_disp:
                            if disp_qty:
                                print('Pharm MAC Qty: ',
                                      pharmacy_data.loc[pharmacy_data['CURRENT_MAC_PRICE'] > 0, qty_column].sum())
                                print('Pharm NonMAC Qty: ',
                                      pharmacy_data.loc[pharmacy_data['CURRENT_MAC_PRICE'] <= 0, qty_column].sum())

                            print('Pharm MAC AWP: ',
                                  pharmacy_data.loc[pharmacy_data['CURRENT_MAC_PRICE'] > 0, client_AWP_column].sum())
                            print('Pharm NonMAC AWP: ',
                                  pharmacy_data.loc[pharmacy_data['CURRENT_MAC_PRICE'] <= 0, client_AWP_column].sum())

                            print('Pharm MAC Spend: ',
                                  pharmacy_data.loc[pharmacy_data['CURRENT_MAC_PRICE'] > 0, reimb_column].sum())
                            print('Pharm NonMAC Spend: ',
                                  pharmacy_data.loc[pharmacy_data['CURRENT_MAC_PRICE'] <= 0, reimb_column].sum())
                perf[pharmacy] += scale_factor * raw_performance + intercept
                logging.debug('Performance: %f', raw_performance)
                logging.debug('Scale Factor: %f', scale_factor)
                logging.debug('Intercept: %f', intercept)
                logging.debug('New Perf: %f', scale_factor * raw_performance + intercept)
            perf[pharmacy] += oc_pharm_perf[pharmacy]
            perf[pharmacy] += gen_launch_perf[pharmacy]
    return perf


def clean_mac_1026(mac1026):
    mac1026all = mac1026.copy(deep=True)
    mac1026all = mac1026all.groupby(['gpi'], as_index=False)['mac_cost_amt'].max()
    mac1026 = mac1026.loc[mac1026['ndc'] == "***********", :]
    mac1026all = mac1026all.loc[~mac1026all['gpi'].isin(mac1026['gpi']), :]
    mac1026.drop(["mac_list_id", "ndc"], axis=1, inplace=True)
    mac1026 = pd.concat([mac1026, mac1026all]).reset_index(drop=True)
    mac1026.rename(index=str, columns={"mac_cost_amt": "MAC1026_unit_price", "gpi": "GPI"}, inplace=True)

    return mac1026


def clean_mac_1026_NDC(mac1026):
    '''
    Renames columns and can provide gpi level 1026 floors if first several lines are uncommented
    '''
    #    mac1026all = mac1026.copy(deep=True)
    #    mac1026all = mac1026all.groupby(['gpi'],as_index=False)['mac_cost_amt'].max()
    #    mac1026all['ndc'] = '***********'
    #    mac1026_gpi = mac1026.loc[mac1026['ndc'] == "***********",:]
    #    mac1026all = mac1026all.loc[~mac1026all['gpi'].isin(mac1026_gpi['gpi']),:]
    #    mac1026.drop(["mac_list_id"],axis=1,inplace=True)
    #    mac1026 = pd.concat([mac1026,mac1026all]).reset_index(drop=True)
    mac1026.rename(index=str, columns={"mac_cost_amt": "MAC1026_unit_price",
                                       "gpi": "GPI",
                                       "ndc": "NDC"}, inplace=True)

    return mac1026


def getLowerBound(df):
    lb, _ = df.Price_Bounds
    return lb


def getUpperBound(df):
    _, ub = df.Price_Bounds
    return ub


def check_price_increase_decrease_initial(df, month):
    rules_violated = 0

    up_fac = df.New_Price / df.MAC_PRICE_UNIT_ADJ

    if p.TIERED_PRICE_LIM and (df.CLIENT in p.TIERED_PRICE_CLIENT):
        if (df.PRICE_REIMB_CLAIM <= 100) and (up_fac > 1.5):
            rules_violated = 1

        if (df.PRICE_REIMB_CLAIM <= 6) and (up_fac > 2):
            rules_violated = 1

        if (df.PRICE_REIMB_CLAIM <= 3) and (up_fac > 201):
            rules_violated = 1

    else:
        if up_fac > (1 + p.GPI_UP_FAC):
            rules_violated = 1

    #    if rules_violated == 1:
    #        logging.debug('GPI pricing error at: '+ df.Dec_Var_Name)
    #        logging.debug('Orig price: ' + str(df.MAC_PRICE_UNIT_ADJ))
    #        logging.debug('New price: ' + str(df.New_Price))

    return rules_violated


def check_agg_price_cons(df, month):
    price_constraints_df = df.loc[(df.Price_Mutable == 1), :]

    rules_violated = 0
    for client in price_constraints_df.CLIENT.unique():
        if (not p.TIERED_PRICE_LIM) or (not df.CLIENT in p.TIERED_PRICE_CLIENT):
            breakout_list = price_constraints_df.loc[price_constraints_df.CLIENT == client, 'BREAKOUT'].unique()
            for breakout in breakout_list:
                reg_list = price_constraints_df.loc[(price_constraints_df.CLIENT == client) &
                                                    (price_constraints_df.BREAKOUT == breakout), 'REGION'].unique()
                for reg in reg_list:
                    mes_list = price_constraints_df.loc[(price_constraints_df.CLIENT == client) &
                                                        (price_constraints_df.BREAKOUT == breakout) &
                                                        (price_constraints_df.REGION == reg), 'MEASUREMENT'].unique()
                    for mes in mes_list:
                        chain_list = price_constraints_df.loc[(price_constraints_df.CLIENT == client) &
                                                              (price_constraints_df.BREAKOUT == breakout) &
                                                              (price_constraints_df.REGION == reg) &
                                                              (
                                                                          price_constraints_df.MEASUREMENT == mes), 'CHAIN_GROUP'].unique()
                        for chain in chain_list:
                            p_v_df = price_constraints_df.loc[(price_constraints_df.CLIENT == client) &
                                                              (price_constraints_df.BREAKOUT == breakout) &
                                                              (price_constraints_df.REGION == reg) &
                                                              (price_constraints_df.MEASUREMENT == mes) &
                                                              (price_constraints_df.CHAIN_GROUP == chain)]

                            old_ing_cost = p_v_df['MAC_PRICE_UNIT_ADJ'] * p_v_df['QTY']
                            lower_bound = old_ing_cost.sum() * (
                                        1 - p.AGG_LOW_FAC) - 1  # due to rounding errors this function will flag if off by $.20 +/- $1 is done to keep from flagging rounding errors
                            upper_bound = old_ing_cost.sum() * (1 + p.AGG_UP_FAC) + 1

                            new_ing_cost = (p_v_df['New_Price'] * p_v_df['QTY']).sum()

                            if new_ing_cost > upper_bound:
                                rules_violated += 1
                                logging.debug(
                                    'Agg price upper bound violation at {}_{}_{}_{}_{}'.format(client, breakout, reg,
                                                                                               mes, chain))
                                logging.debug('Lower bound: ', lower_bound)
                                logging.debug('Upper bound: ', upper_bound)
                                logging.debug('Actual: ', new_ing_cost)


                            elif new_ing_cost < lower_bound:
                                rules_violated += 1
                                logging.debug(
                                    'Agg price lower bound violation at {}_{}_{}_{}_{}'.format(client, breakout, reg,
                                                                                               mes, chain))
                                logging.debug('Lower bound: %f', lower_bound)
                                logging.debug('Upper bound: %f', upper_bound)
                                logging.debug('Actual: %f', new_ing_cost)

    return rules_violated


### GIT: Yiwei - Added function below for PSAO calculations
def is_column_unique(column):
    return (column[0] == column).all()  # True if all values in a dataframe volumn are the same, False if otherwise
