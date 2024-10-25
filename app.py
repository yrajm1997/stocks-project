### Import Required Packages
import os
import openai
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI


# Read OpenAI key from Codespaces Secrets
api_key = os.environ['OPENAI_KEY']             # <-- change this as per your Codespaces secret's name
os.environ['OPENAI_API_KEY'] = api_key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load Model
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)


### Create Chain for Classifying user request as `Need SQL`, `Non SQL`, or `Other`
# Build prompt
template0 = """
Given the user request below, classify it as either being about `Need SQL`, `Non SQL`, or `Other`.
If the user request requires some data to be fetched from the database tables then classify it as `Need SQL`.
If the user request does not requires any data to be fetched from the database but instead need information on what type of data is present inside the database tables then classify it as `Non SQL`.
If the user request looks out of the context then classify it as `Other`.

The databse has following tables:
CREATE TABLE IF NOT EXISTS stock_prices (date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume INT, symbol VARCHAR(20));
CREATE TABLE IF NOT EXISTS income_statement( date DATE, symbol VARCHAR(20), reportedCurrency VARCHAR(20), cik VARCHAR(20), fillingDate DATE, acceptedDate DATE, calendarYear INTEGER, period VARCHAR(20), revenue INTEGER, costOfRevenue INTEGER, grossProfit INTEGER, grossProfitRatio DOUBLE, researchAndDevelopmentExpenses INTEGER, generalAndAdministrativeExpenses INTEGER, sellingAndMarketingExpenses INTEGER, sellingGeneralAndAdministrativeExpenses INTEGER, otherExpenses INTEGER, operatingExpenses INTEGER, costAndExpenses INTEGER, interestIncome INTEGER, interestExpense INTEGER, depreciationAndAmortization INTEGER, ebitda INTEGER, ebitdaratio DOUBLE, operatingIncome INTEGER, operatingIncomeRatio DOUBLE, totalOtherIncomeExpensesNet INTEGER, incomeBeforeTax INTEGER, incomeBeforeTaxRatio DOUBLE, incomeTaxExpense INTEGER, netIncome INTEGER, netIncomeRatio DOUBLE, eps DOUBLE, epsdiluted DOUBLE, weightedAverageShsOut INTEGER, weightedAverageShsOutDil INTEGER );
CREATE TABLE IF NOT EXISTS balancesheet_statement( date DATE, symbol VARCHAR(20), reportedCurrency VARCHAR(20), cik VARCHAR(20), fillingDate DATE, acceptedDate DATE, calendarYear INTEGER, period VARCHAR(20), cashAndCashEquivalents INTEGER, shortTermInvestments INTEGER, cashAndShortTermInvestments INTEGER, netReceivables INTEGER, inventory INTEGER, otherCurrentAssets INTEGER, totalCurrentAssets INTEGER, propertyPlantEquipmentNet INTEGER, goodwill INTEGER, intangibleAssets INTEGER, goodwillAndIntangibleAssets INTEGER, longTermInvestments INTEGER, taxAssets INTEGER, otherNonCurrentAssets INTEGER, totalNonCurrentAssets INTEGER, otherAssets INTEGER, totalAssets INTEGER, accountPayables INTEGER, shortTermDebt INTEGER, taxPayables INTEGER, deferredRevenue INTEGER, otherCurrentLiabilities INTEGER, totalCurrentLiabilities INTEGER, longTermDebt INTEGER, deferredRevenueNonCurrent INTEGER, deferredTaxLiabilitiesNonCurrent INTEGER, otherNonCurrentLiabilities INTEGER, totalNonCurrentLiabilities INTEGER, otherLiabilities INTEGER, capitalLeaseObligations INTEGER, totalLiabilities VARCHAR(20), preferredStock VARCHAR(20), commonStock VARCHAR(20), retainedEarnings VARCHAR(20), accumulatedOtherComprehensiveIncomeLoss VARCHAR(20), othertotalStockholdersEquity VARCHAR(20), totalStockholdersEquity VARCHAR(20), totalEquity VARCHAR(20), totalLiabilitiesAndStockholdersEquity VARCHAR(20), minorityInterest VARCHAR(20), totalLiabilitiesAndTotalEquity VARCHAR(20), totalInvestments VARCHAR(20), totalDebt VARCHAR(20), netDebt VARCHAR(20) ); 
CREATE TABLE IF NOT EXISTS cashflow_statement( date DATE, symbol VARCHAR(20), reportedCurrency VARCHAR(20), cik VARCHAR(20), fillingDate DATE, acceptedDate DATE, calendarYear INTEGER, period VARCHAR(20), netIncome INTEGER, depreciationAndAmortization INTEGER, deferredIncomeTax INTEGER, stockBasedCompensation INTEGER, changeInWorkingCapital INTEGER, accountsReceivables INTEGER, inventory INTEGER, accountsPayables INTEGER, otherWorkingCapital INTEGER, otherNonCashItems INTEGER, netCashProvidedByOperatingActivities INTEGER, investmentsInPropertyPlantAndEquipment INTEGER, acquisitionsNet INTEGER, purchasesOfInvestments INTEGER, salesMaturitiesOfInvestments INTEGER, otherInvestingActivites INTEGER, netCashUsedForInvestingActivites INTEGER, debtRepayment INTEGER, commonStockIssued INTEGER, commonStockRepurchased INTEGER, dividendsPaid INTEGER, otherFinancingActivites INTEGER, netCashUsedProvidedByFinancingActivities INTEGER, effectOfForexChangesOnCash INTEGER, netChangeInCash INTEGER, cashAtEndOfPeriod INTEGER, cashAtBeginningOfPeriod INTEGER, operatingCashFlow INTEGER, capitalExpenditure INTEGER, freeCashFlow INTEGER ); 
The 'symbol' column in each table contains the companies names in capital letters.

Do not respond with more than two words.

Request: {request}
Classification:
"""

PROMPT0 = PromptTemplate(input_variables=["request"], template=template0)

# Classification Chain
clf_chain = (PROMPT0
             | llm
             | StrOutputParser()       # to get output in a more usable format
             )

# # Classify query
# response0 = clf_chain.invoke({"request": "Need open, high prices for any ten Wipro records"})
# print(response0)


### Create Chain for SQL Query Generation
# Build prompt
template1 = """
You are a SQLite expert. Given an input request, return a syntactically correct SQLite query to run.
Unless the user specifies in the question a specific number of examples to obtain, query for at most 10 results using the LIMIT clause as per SQLite. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".
Do not return any new columns nor perform aggregation on columns. Return only the columns present in tables and further aggregations will be done by python code in later steps.
The databse currently have data of Nifty 50 constituents, and the symbol column includes the ticker representing the stock.
Once you start generating sql queries make sure that you only use correct ticker symbol for filtering and restrict yourself to use only current Nifty 50 constituents.

Use the following format:

Request: Request here
SQLQuery: Generated SQL Query here

Only use the following tables:
CREATE TABLE IF NOT EXISTS stock_prices (date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume INT, symbol VARCHAR(20));
CREATE TABLE IF NOT EXISTS income_statement( date DATE, symbol VARCHAR(20), reportedCurrency VARCHAR(20), cik VARCHAR(20), fillingDate DATE, acceptedDate DATE, calendarYear INTEGER, period VARCHAR(20), revenue INTEGER, costOfRevenue INTEGER, grossProfit INTEGER, grossProfitRatio DOUBLE, researchAndDevelopmentExpenses INTEGER, generalAndAdministrativeExpenses INTEGER, sellingAndMarketingExpenses INTEGER, sellingGeneralAndAdministrativeExpenses INTEGER, otherExpenses INTEGER, operatingExpenses INTEGER, costAndExpenses INTEGER, interestIncome INTEGER, interestExpense INTEGER, depreciationAndAmortization INTEGER, ebitda INTEGER, ebitdaratio DOUBLE, operatingIncome INTEGER, operatingIncomeRatio DOUBLE, totalOtherIncomeExpensesNet INTEGER, incomeBeforeTax INTEGER, incomeBeforeTaxRatio DOUBLE, incomeTaxExpense INTEGER, netIncome INTEGER, netIncomeRatio DOUBLE, eps DOUBLE, epsdiluted DOUBLE, weightedAverageShsOut INTEGER, weightedAverageShsOutDil INTEGER );
CREATE TABLE IF NOT EXISTS balancesheet_statement( date DATE, symbol VARCHAR(20), reportedCurrency VARCHAR(20), cik VARCHAR(20), fillingDate DATE, acceptedDate DATE, calendarYear INTEGER, period VARCHAR(20), cashAndCashEquivalents INTEGER, shortTermInvestments INTEGER, cashAndShortTermInvestments INTEGER, netReceivables INTEGER, inventory INTEGER, otherCurrentAssets INTEGER, totalCurrentAssets INTEGER, propertyPlantEquipmentNet INTEGER, goodwill INTEGER, intangibleAssets INTEGER, goodwillAndIntangibleAssets INTEGER, longTermInvestments INTEGER, taxAssets INTEGER, otherNonCurrentAssets INTEGER, totalNonCurrentAssets INTEGER, otherAssets INTEGER, totalAssets INTEGER, accountPayables INTEGER, shortTermDebt INTEGER, taxPayables INTEGER, deferredRevenue INTEGER, otherCurrentLiabilities INTEGER, totalCurrentLiabilities INTEGER, longTermDebt INTEGER, deferredRevenueNonCurrent INTEGER, deferredTaxLiabilitiesNonCurrent INTEGER, otherNonCurrentLiabilities INTEGER, totalNonCurrentLiabilities INTEGER, otherLiabilities INTEGER, capitalLeaseObligations INTEGER, totalLiabilities VARCHAR(20), preferredStock VARCHAR(20), commonStock VARCHAR(20), retainedEarnings VARCHAR(20), accumulatedOtherComprehensiveIncomeLoss VARCHAR(20), othertotalStockholdersEquity VARCHAR(20), totalStockholdersEquity VARCHAR(20), totalEquity VARCHAR(20), totalLiabilitiesAndStockholdersEquity VARCHAR(20), minorityInterest VARCHAR(20), totalLiabilitiesAndTotalEquity VARCHAR(20), totalInvestments VARCHAR(20), totalDebt VARCHAR(20), netDebt VARCHAR(20) ); 
CREATE TABLE IF NOT EXISTS cashflow_statement( date DATE, symbol VARCHAR(20), reportedCurrency VARCHAR(20), cik VARCHAR(20), fillingDate DATE, acceptedDate DATE, calendarYear INTEGER, period VARCHAR(20), netIncome INTEGER, depreciationAndAmortization INTEGER, deferredIncomeTax INTEGER, stockBasedCompensation INTEGER, changeInWorkingCapital INTEGER, accountsReceivables INTEGER, inventory INTEGER, accountsPayables INTEGER, otherWorkingCapital INTEGER, otherNonCashItems INTEGER, netCashProvidedByOperatingActivities INTEGER, investmentsInPropertyPlantAndEquipment INTEGER, acquisitionsNet INTEGER, purchasesOfInvestments INTEGER, salesMaturitiesOfInvestments INTEGER, otherInvestingActivites INTEGER, netCashUsedForInvestingActivites INTEGER, debtRepayment INTEGER, commonStockIssued INTEGER, commonStockRepurchased INTEGER, dividendsPaid INTEGER, otherFinancingActivites INTEGER, netCashUsedProvidedByFinancingActivities INTEGER, effectOfForexChangesOnCash INTEGER, netChangeInCash INTEGER, cashAtEndOfPeriod INTEGER, cashAtBeginningOfPeriod INTEGER, operatingCashFlow INTEGER, capitalExpenditure INTEGER, freeCashFlow INTEGER ); 
The 'symbol' column in each table contains the companies names in capital letters.

Request: {request}
SQLQuery:
"""

PROMPT1 = PromptTemplate(input_variables=["request"], template=template1)

# SQL Query Generation Chain
sql_chain = (PROMPT1
             | llm
             | StrOutputParser()       # to get output in a more usable format
             )

## Generate sql query
# response1 = sql_chain.invoke({"request": "Need open, high prices for any ten Wipro records"})
# print(response1)


### Python tool for code execution
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)
repl_tool.run("1+1")


### Create Chain for Insights Generation
# Build prompt
template2 = """
Use the following pieces of user request and sql query to generate python code that should first load the required data from 'stock_db.sqlite' database and 
then show insights related to that data. If the generated insights contains a figure or plot then that should be saved inside the 'figures' directory.
If there is some tables or numerical values as insights then those should be printed out explicitely using print statement along with their description.
Generate and return python code only, no additional text.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

{request_plus_sqlquery}

Generate code:
"""

PROMPT2 = PromptTemplate(input_variables=["request_plus_sqlquery"], template=template2)

# Code Generation Chain
code_chain = (PROMPT2
              | llm
              | StrOutputParser()       # to get output in a more usable format
              )


### Create Chain for Generating Suggestions
# Build prompt
template3 = """
Use the following pieces of user request and database details to generate suggestions for user to ask for useful insights from the database.
Suggestion should not be more than 4 lines.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

SQLite database has following tables:
CREATE TABLE IF NOT EXISTS stock_prices (date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume INT, symbol VARCHAR(20));
CREATE TABLE IF NOT EXISTS income_statement( date DATE, symbol VARCHAR(20), reportedCurrency VARCHAR(20), cik VARCHAR(20), fillingDate DATE, acceptedDate DATE, calendarYear INTEGER, period VARCHAR(20), revenue INTEGER, costOfRevenue INTEGER, grossProfit INTEGER, grossProfitRatio DOUBLE, researchAndDevelopmentExpenses INTEGER, generalAndAdministrativeExpenses INTEGER, sellingAndMarketingExpenses INTEGER, sellingGeneralAndAdministrativeExpenses INTEGER, otherExpenses INTEGER, operatingExpenses INTEGER, costAndExpenses INTEGER, interestIncome INTEGER, interestExpense INTEGER, depreciationAndAmortization INTEGER, ebitda INTEGER, ebitdaratio DOUBLE, operatingIncome INTEGER, operatingIncomeRatio DOUBLE, totalOtherIncomeExpensesNet INTEGER, incomeBeforeTax INTEGER, incomeBeforeTaxRatio DOUBLE, incomeTaxExpense INTEGER, netIncome INTEGER, netIncomeRatio DOUBLE, eps DOUBLE, epsdiluted DOUBLE, weightedAverageShsOut INTEGER, weightedAverageShsOutDil INTEGER );
CREATE TABLE IF NOT EXISTS balancesheet_statement( date DATE, symbol VARCHAR(20), reportedCurrency VARCHAR(20), cik VARCHAR(20), fillingDate DATE, acceptedDate DATE, calendarYear INTEGER, period VARCHAR(20), cashAndCashEquivalents INTEGER, shortTermInvestments INTEGER, cashAndShortTermInvestments INTEGER, netReceivables INTEGER, inventory INTEGER, otherCurrentAssets INTEGER, totalCurrentAssets INTEGER, propertyPlantEquipmentNet INTEGER, goodwill INTEGER, intangibleAssets INTEGER, goodwillAndIntangibleAssets INTEGER, longTermInvestments INTEGER, taxAssets INTEGER, otherNonCurrentAssets INTEGER, totalNonCurrentAssets INTEGER, otherAssets INTEGER, totalAssets INTEGER, accountPayables INTEGER, shortTermDebt INTEGER, taxPayables INTEGER, deferredRevenue INTEGER, otherCurrentLiabilities INTEGER, totalCurrentLiabilities INTEGER, longTermDebt INTEGER, deferredRevenueNonCurrent INTEGER, deferredTaxLiabilitiesNonCurrent INTEGER, otherNonCurrentLiabilities INTEGER, totalNonCurrentLiabilities INTEGER, otherLiabilities INTEGER, capitalLeaseObligations INTEGER, totalLiabilities VARCHAR(20), preferredStock VARCHAR(20), commonStock VARCHAR(20), retainedEarnings VARCHAR(20), accumulatedOtherComprehensiveIncomeLoss VARCHAR(20), othertotalStockholdersEquity VARCHAR(20), totalStockholdersEquity VARCHAR(20), totalEquity VARCHAR(20), totalLiabilitiesAndStockholdersEquity VARCHAR(20), minorityInterest VARCHAR(20), totalLiabilitiesAndTotalEquity VARCHAR(20), totalInvestments VARCHAR(20), totalDebt VARCHAR(20), netDebt VARCHAR(20) ); 
CREATE TABLE IF NOT EXISTS cashflow_statement( date DATE, symbol VARCHAR(20), reportedCurrency VARCHAR(20), cik VARCHAR(20), fillingDate DATE, acceptedDate DATE, calendarYear INTEGER, period VARCHAR(20), netIncome INTEGER, depreciationAndAmortization INTEGER, deferredIncomeTax INTEGER, stockBasedCompensation INTEGER, changeInWorkingCapital INTEGER, accountsReceivables INTEGER, inventory INTEGER, accountsPayables INTEGER, otherWorkingCapital INTEGER, otherNonCashItems INTEGER, netCashProvidedByOperatingActivities INTEGER, investmentsInPropertyPlantAndEquipment INTEGER, acquisitionsNet INTEGER, purchasesOfInvestments INTEGER, salesMaturitiesOfInvestments INTEGER, otherInvestingActivites INTEGER, netCashUsedForInvestingActivites INTEGER, debtRepayment INTEGER, commonStockIssued INTEGER, commonStockRepurchased INTEGER, dividendsPaid INTEGER, otherFinancingActivites INTEGER, netCashUsedProvidedByFinancingActivities INTEGER, effectOfForexChangesOnCash INTEGER, netChangeInCash INTEGER, cashAtEndOfPeriod INTEGER, cashAtBeginningOfPeriod INTEGER, operatingCashFlow INTEGER, capitalExpenditure INTEGER, freeCashFlow INTEGER ); 
The 'symbol' column in each table contains the companies names in capital letters.

{request}

Generate suggestion:
"""

PROMPT3 = PromptTemplate(input_variables=["request"], template=template3)

# Suggestion Generation Chain
sug_chain = (PROMPT3
             | llm
             | StrOutputParser()       # to get output in a more usable format
             )


### Create Chain for Generating Response for General queries about the data stored in DB
# Build prompt
template4 = """
Use the following user request and database details to generate appropriate response describing the data stored inside the database.
Response should not be more than 10 lines.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

SQLite database has following tables:
CREATE TABLE IF NOT EXISTS stock_prices (date DATE, open DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE, volume INT, symbol VARCHAR(20));
CREATE TABLE IF NOT EXISTS income_statement( date DATE, symbol VARCHAR(20), reportedCurrency VARCHAR(20), cik VARCHAR(20), fillingDate DATE, acceptedDate DATE, calendarYear INTEGER, period VARCHAR(20), revenue INTEGER, costOfRevenue INTEGER, grossProfit INTEGER, grossProfitRatio DOUBLE, researchAndDevelopmentExpenses INTEGER, generalAndAdministrativeExpenses INTEGER, sellingAndMarketingExpenses INTEGER, sellingGeneralAndAdministrativeExpenses INTEGER, otherExpenses INTEGER, operatingExpenses INTEGER, costAndExpenses INTEGER, interestIncome INTEGER, interestExpense INTEGER, depreciationAndAmortization INTEGER, ebitda INTEGER, ebitdaratio DOUBLE, operatingIncome INTEGER, operatingIncomeRatio DOUBLE, totalOtherIncomeExpensesNet INTEGER, incomeBeforeTax INTEGER, incomeBeforeTaxRatio DOUBLE, incomeTaxExpense INTEGER, netIncome INTEGER, netIncomeRatio DOUBLE, eps DOUBLE, epsdiluted DOUBLE, weightedAverageShsOut INTEGER, weightedAverageShsOutDil INTEGER );
CREATE TABLE IF NOT EXISTS balancesheet_statement( date DATE, symbol VARCHAR(20), reportedCurrency VARCHAR(20), cik VARCHAR(20), fillingDate DATE, acceptedDate DATE, calendarYear INTEGER, period VARCHAR(20), cashAndCashEquivalents INTEGER, shortTermInvestments INTEGER, cashAndShortTermInvestments INTEGER, netReceivables INTEGER, inventory INTEGER, otherCurrentAssets INTEGER, totalCurrentAssets INTEGER, propertyPlantEquipmentNet INTEGER, goodwill INTEGER, intangibleAssets INTEGER, goodwillAndIntangibleAssets INTEGER, longTermInvestments INTEGER, taxAssets INTEGER, otherNonCurrentAssets INTEGER, totalNonCurrentAssets INTEGER, otherAssets INTEGER, totalAssets INTEGER, accountPayables INTEGER, shortTermDebt INTEGER, taxPayables INTEGER, deferredRevenue INTEGER, otherCurrentLiabilities INTEGER, totalCurrentLiabilities INTEGER, longTermDebt INTEGER, deferredRevenueNonCurrent INTEGER, deferredTaxLiabilitiesNonCurrent INTEGER, otherNonCurrentLiabilities INTEGER, totalNonCurrentLiabilities INTEGER, otherLiabilities INTEGER, capitalLeaseObligations INTEGER, totalLiabilities VARCHAR(20), preferredStock VARCHAR(20), commonStock VARCHAR(20), retainedEarnings VARCHAR(20), accumulatedOtherComprehensiveIncomeLoss VARCHAR(20), othertotalStockholdersEquity VARCHAR(20), totalStockholdersEquity VARCHAR(20), totalEquity VARCHAR(20), totalLiabilitiesAndStockholdersEquity VARCHAR(20), minorityInterest VARCHAR(20), totalLiabilitiesAndTotalEquity VARCHAR(20), totalInvestments VARCHAR(20), totalDebt VARCHAR(20), netDebt VARCHAR(20) ); 
CREATE TABLE IF NOT EXISTS cashflow_statement( date DATE, symbol VARCHAR(20), reportedCurrency VARCHAR(20), cik VARCHAR(20), fillingDate DATE, acceptedDate DATE, calendarYear INTEGER, period VARCHAR(20), netIncome INTEGER, depreciationAndAmortization INTEGER, deferredIncomeTax INTEGER, stockBasedCompensation INTEGER, changeInWorkingCapital INTEGER, accountsReceivables INTEGER, inventory INTEGER, accountsPayables INTEGER, otherWorkingCapital INTEGER, otherNonCashItems INTEGER, netCashProvidedByOperatingActivities INTEGER, investmentsInPropertyPlantAndEquipment INTEGER, acquisitionsNet INTEGER, purchasesOfInvestments INTEGER, salesMaturitiesOfInvestments INTEGER, otherInvestingActivites INTEGER, netCashUsedForInvestingActivites INTEGER, debtRepayment INTEGER, commonStockIssued INTEGER, commonStockRepurchased INTEGER, dividendsPaid INTEGER, otherFinancingActivites INTEGER, netCashUsedProvidedByFinancingActivities INTEGER, effectOfForexChangesOnCash INTEGER, netChangeInCash INTEGER, cashAtEndOfPeriod INTEGER, cashAtBeginningOfPeriod INTEGER, operatingCashFlow INTEGER, capitalExpenditure INTEGER, freeCashFlow INTEGER ); 
The 'symbol' column in each table contains the companies names in capital letters.

{request}

Generate response:
"""

PROMPT4 = PromptTemplate(input_variables=["request"], template=template4)

# General Response Chain
gnrl_chain = (PROMPT4
              | llm
              | StrOutputParser()       # to get output in a more usable format
              )


# Club SQL + Code generation chains
sql_code_chain = sql_chain | code_chain


### Create UI using Chainlit
import chainlit as cl

@cl.on_chat_start            # for actions to happen when the chat starts
async def main():
    await cl.Message(content=f"Welcome to Stock Prices Insights Application").send()

@cl.on_message               # for actions to happen whenever user enters a message
async def main(message: cl.Message):

    # print(f"User Input: {message.content}")

    # Remove any files from 'figures' directory
    for filename in os.listdir('figures'):
        file_path = os.path.join('figures', filename)
        if os.path.isfile(file_path) and filename != '__init__.py':
            os.remove(file_path)  # Remove file

    # Route the user request to necessary chain
    clf_label = clf_chain.invoke({"request": message.content})

    if "need sql" in clf_label.lower():
        ## Generate code for insights
        code_response = sql_code_chain.invoke({"request": message.content})
        # print(code_response)
        ## Execute code
        output = repl_tool.run(code_response)
        # print(output)
    elif "non sql" in clf_label.lower():
        output = gnrl_chain.invoke({"request": message.content})
    else:
        output = "The request is out of context."

    # Generate suggestions
    suggest = sug_chain.invoke({"request": message.content})
    
    # If image is present inside 'figures' directory then Send the plot image to the chat
    if len(os.listdir('figures')) > 1:
        for imgfile in os.listdir('figures'): 
            print(imgfile)
            if os.path.isfile(os.path.join('figures', imgfile)) and imgfile != '__init__.py':
                # Attach the image to the message and send response, image, and suggestions
                image = cl.Image(path="./figures/"+imgfile, name="image1", size="large", display="inline")
                await cl.Message(
                    content=f"Response: \n{output}", 
                    elements=[image]
                    ).send()
                await cl.Message(content=f"Further suggestions: \n{suggest}",).send()
    else:
        # Send response and suggestions
        await cl.Message(content=f"Response: \n{output}",).send()
        await cl.Message(content=f"Further suggestions: \n{suggest}",).send()
