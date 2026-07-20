# Agentic AI

- This repository contains the courses I've taken from LinkedIn Learning from various instructors on Agentic AI.
- The courses, along with the code and instructions to run the code can be found in their respective directories:
  1. [Autonomous Agents with LangChain & Hugging Face](./01-autonomous-agents-with-langchain-and-hugging-face/)
  2. [Building Agentic SaaS Workflows with AutoGen Studio & OpenAI APIs](./02-building-agentic-saas-worflows-with-autogen-studio/)

## Run Individual Projects

1. Install Python3:
   1. macOS: [`homebrew` installation](https://docs.brew.sh/Language-Runtimes-and-Packages)
   2. Windows: [`winget` installation](https://learn.microsoft.com/en-us/windows/dev-environment/python?tabs=winget).
   3. Linux: [`apt` / `dnf` installation](https://docs.python-guide.org/starting/install3/linux/)
2. Navigate into one of the course directories, and then activate the virtual environment using:

   ```sh
   source .venv/bin/activate

   # OR simply run:
   # ./.venv/bin/activate
   ```

   > NOTE: To deactivate the virtual environment, just run the following command once in an activated venv:
   >
   > ```sh
   > deactivate
   > ```

3. Then, if you find a `requirements.txt` file, just install the packages required using:

   ```sh
   pip install -r requirements.txt

   # In some cases, the default python interpreter is set to Python 2, for that, explicitly 
   # mention pip3 as follows (this case only happens if you aren't using a venv or pip refers
   # to python 2. Check python --version and set python 3 to be default python):
   # pip3 install -r requirements.txt
   ```

4. Run the scripts inside the course using the following command:

   ```sh
   python <script-name>.py
   ```
