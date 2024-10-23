# Import Required Packages
import os
import openai
import sqlite3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

##### SQL Query Generation using an LLM

# Read OpenAI key from Codespaces Secrets
api_key = os.environ['OPENAI_KEY']             # <-- change this as per your Codespaces secret's name
os.environ['OPENAI_API_KEY'] = api_key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load Model
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

# Create Chain for Query Generation using LCEL (LangChain Expression Language)
# Build prompt
template1 = """
You are a SQLite expert. Given an input request, return a syntactically correct SQLite query to run.
Unless the user specifies in the question a specific number of examples to obtain, query for at most 10 results using the LIMIT clause as per SQLite. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".

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

# Query Generation Chain - created using LCEL (LangChain Expression Language)
chain1 = (PROMPT1
          | llm
          | StrOutputParser()       # to get output in a more usable format
          )

## Generate sql query
# response1 = chain1.invoke({"request": "Need open, high prices for any ten Wipro records"})
# print(response1)


### Python Code generation using an LLM
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)
repl_tool.run("1+1")


# Create Chain for Insights Generation
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

# Query Generation Chain - created using LCEL (LangChain Expression Language)
chain2 = (PROMPT2
          | llm
          | StrOutputParser()       # to get output in a more usable format
          )

# Final Chain
final_chain = chain1 | chain2


# Create UI using Chainlit
import chainlit as cl

@cl.on_message
async def main(message: cl.Message):

    # print(f"User Input: {message.content}")

    # Remove any files from 'figures' directory
    for filename in os.listdir('figures'):
        file_path = os.path.join('figures', filename)
        if os.path.isfile(file_path) and filename != '__init__.py':
            os.remove(file_path)  # Remove file

    # Generate code for insights
    code_response = final_chain.invoke({"request": message.content})
    # # print(code_response)
    # # print("="*50)
    
    # Execute code
    output = repl_tool.run(code_response)
    #print(output)

    # Send the plot to the chat
    if len(os.listdir('figures')) != 0:
        for imgfile in os.listdir('figures'): 
            print(imgfile)
            if os.path.isfile(os.path.join('figures', imgfile)) and imgfile != '__init__.py':
                # Attach the image to the message
                image = cl.Image(path="./figures/"+imgfile, name="image1", size="large", display="inline")
                await cl.Message(
                    content=f"Response: \n{output}",
                    elements=[image],
                ).send()
    else:
        # Send response
        await cl.Message(content=f"Response: \n{output}",).send()
