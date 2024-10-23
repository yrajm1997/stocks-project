# Import Required Packages
import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


#########################  Load Constituent Stock Prices data  #########################

# Nifty 50 Constituent Price data
csp = pickle.load(open('./data/constituent_stock_prices.pkl', 'rb'))

# Combine 50 dataframes into one, Add a column 'Symbol' to distinguish that the row data is for that particular company

comb_df = pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol'])

for key in csp.keys():
    tmp_df = csp[key].copy()
    tmp_df.reset_index(inplace=True)
    tmp_df['Symbol'] = key.split('.')[0]
    comb_df = pd.concat([comb_df, tmp_df], ignore_index=True)

#print(comb_df.head())

# Remove any old 'stock_db.sqlite' file
def check_and_delete_file(file_path):
    # Check if the file exists
    if os.path.isfile(file_path):
        print(f"File '{file_path}' found. Deleting...")
        os.remove(file_path)
        print("File deleted.")
    else:
        print(f"File '{file_path}' does not exist. Creating new...")

check_and_delete_file('stock_db.sqlite')

### Create a SQLite Database (in-memory)
# Connect to a sqlite DB (It will create it if it doesn't exists)
conn = sqlite3.connect('stock_db.sqlite')
print("Created DB successfully!")

# Create a table 'stock_prices' in DB
conn.execute('''
CREATE TABLE IF NOT EXISTS stock_prices(
                      date DATE,
                      open DOUBLE,
                      high DOUBLE,
                      low DOUBLE,
                      close DOUBLE,
                      volume INT,
                      symbol VARCHAR(20));''')

conn.commit()
print("\nTable 'stock_prices' created successfully!")

# Function to convert the 'date' from Timestamp to String yyyy-mm-dd
def convert_date(date):
    yyyy = date.year
    mm = date.month
    dd = date.day
    if mm<10:
        mm = '0' + str(mm)
    if dd<10:
        dd = '0' + str(dd)
    return f"{yyyy}-{mm}-{dd}"

comb_df['Date'] = comb_df['Date'].apply(convert_date)

# Insert data into stock_prices table
conn.executemany('''
INSERT INTO stock_prices (date, open, high, low, close, volume, symbol) VALUES (?, ?, ?, ?, ?, ?, ?)
''', comb_df.values)
conn.commit()
print("Data inserted into 'stock_prices' successfully!")

### Query the Database # Show table content
# cursor = conn.execute('''
# SELECT * from stock_prices limit 10;
# ''')
# for row in cursor:
#     print(row)


#########################  Load Constituent Stock Fundamentals data  #########################

# Load second pickle file
csf = pickle.load(open('./data/constituent_stock_fundamentals.pkl', 'rb'))

# Select any one Stock. Each stock is a dictionary
d1 = csf['ADANIENT.NS']

# print(d1.keys())   =>  dict_keys(['income_statement', 'balancesheet_statement', 'cashflow_statement'])

# Each stock has 3 different dataframes
dd1 = pd.DataFrame(d1['income_statement'])            # dd1.shape => (51, 38)
dd2 = pd.DataFrame(d1['balancesheet_statement'])      # dd2.shape => (49, 54)
dd3 = pd.DataFrame(d1['cashflow_statement'])          # dd3.shape => (83, 40)


# Combine 'income_statement' data for all stocks
colm_list = pd.DataFrame(csf['ADANIENT.NS']['income_statement']).columns.tolist()

income_df = pd.DataFrame(columns=colm_list)

for key in csf.keys():
    tmp_df = pd.DataFrame(csf[key]['income_statement']).copy()
    income_df = pd.concat([income_df, tmp_df], ignore_index=True)

income_df['symbol'] = income_df['symbol'].apply(lambda x: x.split('.')[0])

# print(income_df.shape)  =>  (2430, 38)

# Correct datatype for columns after combine operation
for i in range(len(income_df.columns)):
    if income_df.dtypes.values[i] !=  dd1.dtypes.values[i]:
        income_df[income_df.columns[i]] = income_df[income_df.columns[i]].astype(dd1.dtypes.values[i])


# Combine 'balancesheet_statement' data for all stocks
colm_list = pd.DataFrame(csf['ADANIENT.NS']['balancesheet_statement']).columns.tolist()

balancesheet_df = pd.DataFrame(columns=colm_list)

for key in csf.keys():
    tmp_df = pd.DataFrame(csf[key]['balancesheet_statement']).copy()
    balancesheet_df = pd.concat([balancesheet_df, tmp_df], ignore_index=True)

balancesheet_df['symbol'] = balancesheet_df['symbol'].apply(lambda x: x.split('.')[0])

# Correct datatype for columns after combine operation
for i in range(len(income_df.columns)):
    if balancesheet_df.dtypes.values[i] !=  dd2.dtypes.values[i]:
        balancesheet_df[balancesheet_df.columns[i]] = balancesheet_df[balancesheet_df.columns[i]].astype(dd2.dtypes.values[i])


# Combine 'cashflow_statement' data
colm_list = pd.DataFrame(csf['ADANIENT.NS']['cashflow_statement']).columns.tolist()

cashflow_df = pd.DataFrame(columns=colm_list)

for key in csf.keys():
    tmp_df = pd.DataFrame(csf[key]['cashflow_statement']).copy()
    cashflow_df = pd.concat([cashflow_df, tmp_df], ignore_index=True)

cashflow_df['symbol'] = cashflow_df['symbol'].apply(lambda x: x.split('.')[0])

# Correct datatype for columns after combine operation
for i in range(len(income_df.columns)):
    if cashflow_df.dtypes.values[i] !=  dd3.dtypes.values[i]:
        cashflow_df[cashflow_df.columns[i]] = cashflow_df[cashflow_df.columns[i]].astype(dd3.dtypes.values[i])


# Remove unnecessary columns
income_df.drop(columns=['link', 'finalLink'], inplace=True)
balancesheet_df.drop(columns=['link', 'finalLink'], inplace=True)
cashflow_df.drop(columns=['link', 'finalLink'], inplace=True)


### Insert 'income_statement' data into SQLite DBstock prices

# Create sql query to create 'income_statement' table with reqd cols and datatype
income_table_template = '''
CREATE TABLE IF NOT EXISTS income_statement(\n'''

for i in range(len(income_df.columns)):
    if 'date' in str.lower(income_df.columns[i]):
        datatype = 'DATE'
    elif 'year' in str.lower(income_df.columns[i]):
        datatype = 'INTEGER'
    elif income_df.dtypes.values[i] == 'object':
        datatype = 'VARCHAR(20)'
    elif income_df.dtypes.values[i] == 'int64':
        datatype = 'INTEGER'
    elif income_df.dtypes.values[i] == 'float64':
        datatype = 'DOUBLE'
    else:
        datatype = 'VARCHAR(20)'
    income_table_template += f"      {income_df.columns[i]} {datatype}, \n"

income_table_template = income_table_template[:-3] + f"\n      );"

# print(income_table_template)  =>  "CREATE TABLE IF NOT EXISTS income_statement(..."

# Create table 'income_statement' in DB
conn.execute(income_table_template)
conn.commit()
print("\nTable 'income_statement' created successfully")


# Function to convert acceptedDate
def convert_accepted_date(date):
    return date[:10]

income_df['acceptedDate'] = income_df['acceptedDate'].apply(convert_accepted_date)

# Form a Insert query for income_statement
income_insert_template = "\nINSERT INTO income_statement ("

for i in range(len(income_df.columns)):
    income_insert_template += f"{income_df.columns[i]}, "

income_insert_template = income_insert_template[:-2] + f") VALUES (?{', ?'*(len(income_df.columns)-1)})\n"
# print(income_insert_template)  =>  "INSERT INTO income_statement (date, symbol, ..."

# Insert income_statement data
conn.executemany(income_insert_template, income_df.values)
conn.commit()
print("Data inserted into 'income_statement' successfully!")


### Insert 'balancesheet_statement' data into SQLite DB

# Create sql query to create 'balancesheet_statement' table with reqd cols and datatype
balancesheet_table_template = '''
CREATE TABLE IF NOT EXISTS balancesheet_statement(\n'''

for i in range(len(balancesheet_df.columns)):
    if 'date' in str.lower(balancesheet_df.columns[i]):
        datatype = 'DATE'
    elif 'year' in str.lower(balancesheet_df.columns[i]):
        datatype = 'INTEGER'
    elif balancesheet_df.dtypes.values[i] == 'object':
        datatype = 'VARCHAR(20)'
    elif balancesheet_df.dtypes.values[i] == 'int64':
        datatype = 'INTEGER'
    elif balancesheet_df.dtypes.values[i] == 'float64':
        datatype = 'DOUBLE'
    else:
        datatype = 'VARCHAR(20)'
    balancesheet_table_template += f"      {balancesheet_df.columns[i]} {datatype}, \n"

balancesheet_table_template = balancesheet_table_template[:-3] + f"\n      );"

# print(balancesheet_table_template)  =>  "CREATE TABLE IF NOT EXISTS balancesheet_statement(..."

# Create table 'balancesheet_statement' in DB
conn.execute(balancesheet_table_template)
conn.commit()
print("\nTable 'balancesheet_statement' created successfully")

balancesheet_df['acceptedDate'] = balancesheet_df['acceptedDate'].apply(convert_accepted_date)

# Form a Insert query for balancesheet_statement
balancesheet_insert_template = "\nINSERT INTO balancesheet_statement ("

for i in range(len(balancesheet_df.columns)):
    balancesheet_insert_template += f"{balancesheet_df.columns[i]}, "

balancesheet_insert_template = balancesheet_insert_template[:-2] + f") VALUES (?{', ?'*(len(balancesheet_df.columns)-1)})\n"
# print(balancesheet_insert_template)  =>  "INSERT INTO balancesheet_statement (date, symbol, ..."

# Insert balancesheet_statement data
conn.executemany(balancesheet_insert_template, balancesheet_df.values)
conn.commit()
print("Data inserted into 'balancesheet_statement' successfully!")


### Insert 'cashflow_statement' data into SQLite DB

# Create sql query to create 'balancesheet_statement' table with reqd cols and datatype
cashflow_table_template = '''
CREATE TABLE IF NOT EXISTS cashflow_statement(\n'''

for i in range(len(cashflow_df.columns)):
    if 'date' in str.lower(cashflow_df.columns[i]):
        datatype = 'DATE'
    elif 'year' in str.lower(cashflow_df.columns[i]):
        datatype = 'INTEGER'
    elif cashflow_df.dtypes.values[i] == 'object':
        datatype = 'VARCHAR(20)'
    elif cashflow_df.dtypes.values[i] == 'int64':
        datatype = 'INTEGER'
    elif cashflow_df.dtypes.values[i] == 'float64':
        datatype = 'DOUBLE'
    else:
        datatype = 'VARCHAR(20)'
    cashflow_table_template += f"      {cashflow_df.columns[i]} {datatype}, \n"

cashflow_table_template = cashflow_table_template[:-3] + f"\n      );"

# print(cashflow_table_template)  =>  "CREATE TABLE IF NOT EXISTS cashflow_statement(..."

# Create table 'cashflow_statement' in DB
conn.execute(cashflow_table_template)
conn.commit()
print("\nTable 'cashflow_statement' created successfully")

cashflow_df['acceptedDate'] = cashflow_df['acceptedDate'].apply(convert_accepted_date)

# Form a Insert query for balancesheet_statement
cashflow_insert_template = "\nINSERT INTO cashflow_statement ("

for i in range(len(cashflow_df.columns)):
    cashflow_insert_template += f"{cashflow_df.columns[i]}, "

cashflow_insert_template = cashflow_insert_template[:-2] + f") VALUES (?{', ?'*(len(cashflow_df.columns)-1)})\n"
# print(cashflow_insert_template)  =>  "INSERT INTO cashflow_statement (date, symbol, ..."

# Insert cashflow_statement data
conn.executemany(cashflow_insert_template, cashflow_df.values)
conn.commit()
print("Data inserted into 'cashflow_statement' successfully!")


### Close the DB connection
conn.close()
print("DB connection closed!")
