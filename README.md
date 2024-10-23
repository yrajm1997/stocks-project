# stocks-project
This is a repo for GenAI M5 AST2

## Steps to Follow

1. Add your OpenAI key to your repository Secrets. 
   
   Go to Settings -> Secrets and Variables -> Codespaces -> New repository secret -> Give Name(eg. OPENAI_KEY) and paste Secret Value

2. Start a Codespace by going to `Code` dropdown > Select `Codespaces` tab > Click on `Create codespace on main`

3. Install requirement:
   ```
   pip install -r requirements.txt
   ```

4. Read data from pickle files and create a database:
   ```
   python main.py
   ```

5. Start application:
   ```
   chainlit run app.py
   ```

6. Once the application is running, access it in browser
