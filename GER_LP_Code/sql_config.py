# -*- coding: utf-8 -*-
"""
"""

#import pyodbc
import sys
import CPMO_parameters as p
import teradata

#######Append the path with your credentials.py path#######

#TODO:move to parameters
sys.path.append(p.CREDENTIALS_PATH)
#sys.path.append('C:/Users/z295114/Desktop')
import credentials as cd


def get_teradata_connection():
    #database = "SB_Finance_G2_GER_OPT"
    user = cd.TERADATA_UID
    password = cd.TERADATA_PWD
    host = 'prdedw1.caremarkrx.net'
    udaExec = teradata.UdaExec (appName="test", version="1.0", logConsole=False)

    conn =  udaExec.connect(method="odbc",system=host, username=user, password=password, driver=cd.TERADATA_DRIVER)
    return conn
