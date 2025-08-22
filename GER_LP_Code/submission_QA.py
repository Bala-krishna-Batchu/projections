# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import CPMO_parameters as p

from QA import test_price_changes_file

'''
Place the two files you get from data engineering into the
folder that p.PROGRAM_INPUT_PATH points to.
'''

# File from data engineering containing floors used
FLOORED_FILE = "lp_floors_overrides_used.txt"
# File with the modified price changes
FNL_FILE = "GER_CLIENT-NAME_LOB_DATE"

# Read in data
submitted_prices = pd.read_csv(p.FILE_OUTPUT_PATH + p.PRICE_CHANGE_FILE, sep = '\t', dtype = p.VARIABLE_TYPE_DIC)
test_price_changes_file()
floored_prices = pd.read_csv(p.PROGRAM_INPUT_PATH + FLOORED_FILE, sep = '|', dtype = p.VARIABLE_TYPE_DIC)
# Double check the below file is read in correctly. It's a fixed width format file, rather than tab or comma separated
# colspecs currently defines the withs of the columns read in. 
colspecs = [(9,18), (19,52), (67,83)]
final_prices = pd.read_fwf(p.PROGRAM_INPUT_PATH + FNL_FILE, header = None, colspecs=colspecs, names=['MAC_LIST', 'GPINDC', 'RAW_PRICE'], dtype = {'RAW_PRICE':str,'GPINDC':str, 'MAC_LIST':str})

# Make more human readable columns in final_prices
submitted_prices['GPI_NDC'] = submitted_prices.GPI + "_" + submitted_prices.NDC11
floored_prices['GPI_NDC'] = floored_prices.GPI + "_" + floored_prices.NDC_LABEL + floored_prices.NDCPROD + floored_prices.NDCPKG

final_prices['GPI_NDC'] = final_prices['GPINDC'].str[:14] + "_" + final_prices['GPINDC'].str[-11:]
final_prices['Price'] = final_prices['RAW_PRICE'].str[-13:]
final_prices['Price'] = final_prices['Price'].astype(int) / 100000

final_prices['NDC'] = final_prices['GPI_NDC'].str[-11:]

# Test NDC11 are all stars
if all(final_prices['NDC'].astype(str).str.match(pat = '\*{11}')):
    print('\n\n All NDC are *s ')
else:
    assert False, "*ERROR: Some NDC are not *, implying that we are recomending NDC level prices."

# Assert that the file we submitted and the final prices have the same number of rows
assert len(final_prices) == len(submitted_prices)
print("Check this number against the records received count emailed to you: " + str(len(submitted_prices)))
print("Check this number against the floors used count emailed to you: " + str(len(floored_prices[~pd.notna(floored_prices['OVRD_COST_AMT'])])))
print("Check this number against the overrides used count emailed to you: " + str(len(floored_prices[pd.notna(floored_prices['OVRD_COST_AMT'])])))

# Check if unchanged prices are still unchanged
unchanged = submitted_prices[~submitted_prices.GPI_NDC.isin(floored_prices.GPI_NDC)]

assert len(unchanged.GPI_NDC.unique()) == (len(submitted_prices.GPI_NDC.unique()) - len(floored_prices.GPI_NDC.unique()))
assert len(unchanged.GPI_NDC.unique()) == (len(final_prices.GPI_NDC.unique()) - len(floored_prices.GPI_NDC.unique()))

m1 = pd.merge(unchanged, final_prices[['GPI_NDC', 'Price', 'MAC_LIST']], how = 'inner', left_on = ['MACLIST','GPI_NDC'], right_on = ['MAC_LIST','GPI_NDC'])

assert len(m1.GPI_NDC.unique()) == (len(submitted_prices.GPI_NDC.unique()) - len(floored_prices.GPI_NDC.unique()))
assert m1['MACPRC'].round(3).equals(m1['Price'].round(3))

# Check if floored prices were changed correctly
changed = final_prices[final_prices.GPI_NDC.isin(floored_prices.GPI_NDC)]

assert len(changed.GPI_NDC.unique()) == len(floored_prices.GPI_NDC.unique())

m2 = pd.merge(changed, floored_prices[["#MACLIST","GPI_NDC","Floor_Price", 'OVRD_COST_AMT']], how = 'inner', left_on = ['MAC_LIST','GPI_NDC'], right_on = ['#MACLIST','GPI_NDC'])
m2['Updated_Price'] = np.where(m2['OVRD_COST_AMT'].isna(),m2['Floor_Price'],m2['OVRD_COST_AMT'])

assert len(m2.GPI_NDC.unique()) == len(floored_prices.GPI_NDC.unique())
assert m2['Price'].round(3).equals(m2['Updated_Price'].round(3))

# Check to make sure the MAIL VCML is not in the floored prices file
assert np.all(floored_prices['#MACLIST'][-1:] != 2)
