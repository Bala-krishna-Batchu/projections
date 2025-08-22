# Fix for the CUSTOMER_ID parameter issue

# INCORRECT - This causes the BigQuery syntax error:
custom_params_incorrect = {
    "CUSTOMER_ID": "['4590']",  # This is a string, not a list
    # ... other parameters
}

# CORRECT - This should fix the issue:
custom_params_correct = {
    "CUSTOMER_ID": ['4590'],  # This is an actual list
    # ... other parameters
}

# The issue occurs because:
# 1. When CUSTOMER_ID = "['4590']" (string), get_formatted_string() treats it as a single element
# 2. This results in SQL like: WHERE customer_ID IN ('['4590']') which is invalid syntax
# 3. When CUSTOMER_ID = ['4590'] (list), get_formatted_string() processes it correctly
# 4. This results in SQL like: WHERE customer_ID IN ('4590') which is valid syntax

print("Fix your custom_params by changing:")
print('"CUSTOMER_ID": "[\'4590\']"')
print("to:")
print('"CUSTOMER_ID": [\'4590\']')