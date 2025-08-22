# -*- coding: utf-8 -*-
"""
Parameters file for CLIENT_PHARMACY_MAC_OPTIMIZATION

@author: JOHN WALKER
@version: 1.1.0, 01.27.2020
"""
import datetime as dt

# Client Name
client_name = 'Clearstone'  # Dell Inc, United Airlines, Southern Company
client_type = 'MEDD'  # or MEDD
pilot = True

increase = '/Flat Increase/'

# Input files
PROGRAM_INPUT_PATH = 'C:/Users/c247920/OneDrive - CVS Health/Custom 2022/Clearstone/'
FILE_INPUT_PATH = PROGRAM_INPUT_PATH + 'Inputs/Input/'
FILE_OUTPUT_PATH = PROGRAM_INPUT_PATH + 'Outputs/'
FILE_DYNAMIC_INPUT_PATH = PROGRAM_INPUT_PATH + 'Dynamic_Inputs/'
FILE_LOG_PATH = PROGRAM_INPUT_PATH + 'Logs/'
MAC1026_FILE = 'MAC1026.csv'
MAC_LIST_FILE = '20220419_' + client_name + '_mac_list.csv'
REGION_MAC_FILE = '20220419_' + client_name + '_Region_Mac_Mapping.csv'
CLIENT_GUARANTEE_FILE = '20220419_' + client_name + '_client_guarantee.csv'
PHARM_GUARANTEE_FILE = '20220419_' + client_name + '_Pharm_Guarantees.csv'
GEN_LAUNCH_FILE = '20220419_' + client_name + '_GEN_LAUNCH.csv'
BACKOUT_GEN = '20220419_BackoutGenericLaunch_' + client_name + '.csv'
MAC_CONSTRAINT_FILE = '20220419_' + client_name + '_MAC_constraints.csv'
PACKAGE_SIZE_FILE = '20220419_packagesize_to_ndc.csv'
SPECIALTY_EXCLUSION_FILE = '20220419_GPI_exclusion_NDC.csv'  # '20200214_GPI_change_exclusion_NDC_GR.csv' #
PHARMACY_YTD_FILE = '20220419_' + client_name + '_Pharm_YTD.csv'
OC_PHARM_PERF_FILE = '20220419_all_other_client_perf.csv'
PHARMACY_APPROX_FILE = 'Clearstone_Pharmacy_coef.csv'
PREFERRED_PHARM_FILE = '20220419_Pref_Pharm_List.csv'
# WC_SUGGESTED_GUARDRAILS = '20200220_WCGuardRailsLax.csv'
UC_ADJUSTMENT = '20220419_' + client_name + '_UC_Adjustment_flat.csv'  # '20200629_' + client_name + '_UC_Adjustment.csv'
NEW_MAC_FILE = ''
FLOOR_GPI_LIST = '20201209_Floor_GPIs.csv'
# CLIENT_SUGGESTED_TIERS = '20190919_WC_Tiers.csv'
LAG_YTD_Override_File = '20210308_'+client_name+'_YTD_LAG_Override.csv'


CLIENT_TARGET_BUFFER = 0
PHARMACY_TARGET_BUFFER = 0

# It modifies the freedom that zero utilization drugs have. The business impact should still be understood and analyzed further.
# To be use at the beginning of the year (Q1/Q2) when there are a lot of drugs with zero utilization.
ZERO_QTY_TIGHT_BOUNDS = True

# The weight of the soft constraint on the zero utilization drugs.
ZERO_QTY_WEIGHT = 10

FLOOR_PRICE = True  # Allows for GPI on the FLOOR_GPI_LIST file to get set to the Mac 1026 floor.
UC_OPT = False
# Output Mapping Files
TMAC_DRUG_FILE = '20200619_TMAC_Drug_Info.csv'
TMAC_MAC_MAP_FILE = '20200619_TMAC_MAC_Mapping.csv'

# Set options that may or may not need to be changed depending on run
CAPPED_PHARMACY_LIST = ['CVS', 'RAD', 'WAG', 'KRG', 'WMT']  # , 'CVS_ALT', 'WAG_ALT', 'KRG_ALT', 'WMT_ALT',]#, 'ABS', 'AHD', 'GIE', 'KIN' ]
OTHER_CAPPED_PHARMACY_LIST = ['CST','ELE','HYV']  # 'ABS', 'AHD', 'GIE', 'KIN', 'MCHOICE']#, 'ABS', 'AHD', 'GIE', 'KIN' ]
NON_CAPPED_PHARMACY_LIST = ['NONPREF_OTH', 'PREF_OTH']  # , 'ART', 'CAR']
OVER_REIMB_CHAINS = []

PSAO_LIST = []  # 'ACH', 'ELE', 'EPC', 'TPS']
AGREEMENT_PHARMACY_LIST = CAPPED_PHARMACY_LIST + OTHER_CAPPED_PHARMACY_LIST

# HACK: added full year
FULL_YEAR = False
GO_LIVE = dt.datetime.strptime('04/29/2022', '%m/%d/%Y')
LAST_DATA = dt.datetime.strptime('03/31/2022', '%m/%d/%Y')  # Last day of pulled data (could maybe be inferred in future)
DATA_ID = client_name + '_Sept_OctGL_uc25'  # '_May_JulGL_uc25' The label for the preprocessed daily totals file.  This allows you to have different preprocessed files for testing

TIERED_PRICE_LIM = False  # Provides option for clients to get tiered price limits as opposed to single price
TIERED_PRICE_CLIENT = [client_name.upper()]  # all caps

SIM = False  # Set true if you want to go into simulation mode, which allows you to run an LP up to once a month to simulate out results for the rest of the year
SIM_MONTHS = [4]  # [7]
LP_RUN = [4]  # [7] The months that the lp will run.  If it is not a month the LP will run, then the performance will be calculated with no price changes

READ_IN_NEW_MACS = False  # If you want to read in a set of MACs that are different from the pre-processing
LAG_YTD_OVERRIDE = False  # If you want to read in an outside performance file
YTD_OVERRIDE = False
NDC_UPDATE = False  # As we are changing NDCs outside of this algorithm in 2019, this allows us to have different pricing for the LAG period and Implementation period
STRENGTH_PRICE_CHANGE_EXCEPTION = False  # This allows a list of GPI12s to have tiered pricing changes.  This was originally developed to allow some of the prices to get into strength order
NO_MAIL = False  # If you do not want to consider mail
LIMITED_BO = False  # Limits breakouts considered to only those in the BO_LIST
MONTHLY_PHARM_YTD = True  # The standard input for pharmacy ytd is daily totals. If you want to only provide monthly totals, set this to true.
PRICE_ZERO_PROJ_QTY = True  # This allows the pricing of drugs that are projected to have no utilization.  This allows all prices on MAC list to be in line.
BO_LIST = ['MEDD', 'REGIONS']  # Breakouts to be considered in LP if LIMITED_BO = True

READ_DT = False  # Set True to read in and process daily totals.  Set False to read in an already processed daily totals
SAVE_DT = True  # Set True to save daily totals after preprocessing them with the label DATA_ID (specified below).  Note: daily totals are only saved if they are processed during the run (ie READ_DT = True)

TIMESTAMP = client_name + dt.date.today().strftime("%m%d%Y")  # The label given to all output files for this set of runs
PERF_FILE = 'PERF' + TIMESTAMP  # This is the old performance file.  This file has been depricated.
WRITE_OUTPUT = False # This determines whether or not the output of the lp is written
OUTPUT_FULL_MAC = False  # This determines whether or not the full mac is output or only the changes
WRITE_LP_FILES = False  #

CAPPED_OPT = True  # Sets it so that all capped pharmacies get over_reimb_gamma in objective function
SSI_FIXED = False  # Set it so that SSI prices do not change
WC_FIXED = False  # Set it so that WellCare Prices do not change
CLIENT_TIERS = False
CLIENT_GR = False  # the client supplies guard rails set this to true
GR_SCALE = False  # If you want to loosen up the supplied scale factor, set to true
GR_SF = 1.0  # This is how much the scale factor will be loosened both up and down (max 1)
LIM_TIER = False  # This is if we only want to implement certain tiers of guard rails
TIER_LIST = ['1', '2']  # These are the only guard rails that will be implemented if LIM_TIER = True
CAPPED_ONLY = True

UC_ADJUST = False  # Adds an adjustment to account for known overperformance due to U&C
GPI_UP_FAC = 0.2  # Proportion that the current price can increase at a single GPI-NDC in a single MAC list
GPI_LOW_FAC = .6 # Proportion that the current price can decrease at a single GPI-NDC in a single MAC list
AGG_UP_FAC = -1  # Amount that an entire MAC list can increase.  This is weighted by utilization.  -1 will turn this off
AGG_LOW_FAC = -1  # Amount that an entire MAC list can increase.  This is weighted by utilization.
MAINTENANCE = False  # If you want months following first run through in the simulation to have different GPI up and GPI down factors, set to True
MAINTENANCE_GPI_UP_FAC = .2  # GPI_UP_FAC for all months after first month if MAINTENANCE is True
MAINTENANCE_GPI_LOW_FAC = .8  # GPI_LOW_FAC for all months after first month if MAINTENANCE is True
CLIENT_OVRPERF_PEN = 0.0  # Penalty for overperformance even though overperformance is not itself a liability
OVER_REIMB_GAMMA = 0.1
COST_GAMMA = 1.0

PREF_OTHER_FACTOR = 1.0 #Factor of CVS pricing that independents cannot go below
PROJ_ALPHA = 0.7 #Smoothing factor for Exponential Smoothing
WAG_KRG_UC_UNIT_LIM = True #Since WAG and KRG exclude U&C claims this allows a lower ceiling on WAG and KRG prices
UC_UNIT_FACTOR = 0.9 #Multiplyer of KRG and WAG U&C ceiling. Set to less than 1 for a lower cieling and greater than 1 for a higher cieling
REMOVE_KRG_WMT_UC = True  #This multiplies the Kroger and Walmart U&Cs by 500, effectively eliminating them

NDC_MAC_LISTS = []
