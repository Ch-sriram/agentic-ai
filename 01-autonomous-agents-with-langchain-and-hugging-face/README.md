# Build Autonomous Agents w/ LangChain & Hugging Face

- You'll be building small-scale agents that will help you grasp the concept of how LangChain and Hugging Face libraries can be utilized with Python3, to host an agent in your local machine.

## Requirements

1. Install [Python3](https://www.python.org/downloads/): Preferably 3.13+
   - Check if you've python already installed using the command: `python3 --version` in your terminal/cmd.
2. Create a virtual environment using python as follows:

   ```sh
   mkdir agentic-ai
   # Navigate to the created directory using: `cd agentic-ai`
   
   # From inside the agentic-ai directory, create a virtual environment and name it as .venv
   python -m venv .venv

   # Activate the virtual environment
   source .venv/bin/activate

   # To deactivate the virtual environment, just do the following: source .venv/bin/deactivate
   ```

   You should something as follows in your terminal

   ```terminal
   (.venv) (base) Personal-MacBook-Pro ~/Documents/repos/personal/agentic-ai:(~|git@main!)
   ```

   The `(.venv)` in front of the terminal prompt is an indicator that the virtual environment is active. To deactivate the virtual environment, just run `source .venv/bin/deactivate`.

3. Install LangChain, Hugging Face, DuckDuckGo Search, Beautiful Soup, and Python Dotenv using the following command:

   ```sh
   pip install langchain langchain-huggingface huggingface-hub torch duckduckgo-search python-dotenv beautifulsoup4

   # install transformers@4.55.4
   pip install "transformers==4.55.4"

   # install ddgs (Dux Distributed Global Search)
   pip install ddgs
   ```

   NOTE: You should see some packages being installed, on your terminal/cmd.

   **What are the packages related to?**
   - `pip`: PIP Installs Packages is a python package installer
   - `langchain`: python package for reasoning
   - `langchain-huggingface`: used for accessing LLM models
   - `huggingface-hub`: where all the open source LLMs are hosted
   - `torch`: PyTorch package contains all the deeplearning and machine learning models
   - `transformers`: acts as model's engine &mdash; core computation package for LLMs
   - `duckduckgo-search`: for live web results
   - `python-dotenv`: for accessing environment/system tokens
   - `beautifulsoup4`: for scraping and cleaning web-page results
   - `ddgs`: Dux Distributed Global Search &mdash; <https://pypi.org/project/ddgs/>

4. Open your browser and naviagte to <https://huggingface.co> \[or search for hugging face in Google\], and login/signup to your account.
   1. Click on your avatar (top-right)
   2. Navigate to `Access Tokens`
   3. Click on `+ Create new token`
   4. Give a name to the token in the `Token name` field
   5. Click on `Custom` preset
   6. In the `Inference` label, check on `Make calls to Inference Providers` checkbox
   7. Click the `Create token` button
   8. Copy the token presented there (it won't be shown again, ever, once you close the dialog/modal)
   9. Then, in your terminal, type in the following, and save it to `.env` file as follows:

      ```sh
      echo "HF_TOKEN=<secret-token>" > .env
      ```

      NOTE: `<secret-token>` is the hugging face token you copied in step 8.

5. If all the steps are followed correctly from pt. 1 to 4, you should be able to run the following command in terminal:

   ```sh
   python -c "import langchain, transformers; print('✅ Setup successful!')"
   ```

   and see something similar to the following output:

   ```terminal
   [transformers] PyTorch was not found. Models won't be available and only tokenizers, configuration and file/data utilities can be used.
   ✅ Setup successful!
   ```

## Run Basic Agent

- Run the basic agent at [`agent_basic.py`](./agent_basic.py) using `python`:

   ```sh
   python agent_basic.py
   ```

   you should see the following output:

   ```terminal
   Note: Environment variable`HF_TOKEN` is set and is the current active token independently from the token you've just configured.
   Device set to use cpu
   An AI agent is a computer programmed to perform a task on a computer.
   ```

## Other Agents

- There are 5 agents located in this directory itself.
- To run each of them, just use their file names with the python command.
- The agents are the following:
  1. Basic Agent: run using `python agent_basic.py`
  2. Joke Teller Agent: run using `python agent_joketeller.py`
  3. Multi Step Agent: run using `python agent_multistep.py`
  4. Agent with Memory and Context: run using `agent_memory.py`
  5. Agent with Tools like Calculator: run using `python agent_tools.py`
  6. Mini Study Buddy Agent (Contains concepts from 1-5): run using `python mini_study_buddy.py`
