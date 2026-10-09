---
description: Create a Python virtual environment or conda environment
---

Create a Python virtual environment for the project.

Steps:
1. Ask the user which type of environment they want to create:
   - Python venv (virtualenv)
   - Conda environment

2. Ask for the environment name (default: "venv" for venv, or project name for conda)

3. For Python venv:
   - Create the virtual environment using `python -m venv <name>`
   - Provide activation instructions for the user's OS
   - If a `requirements.txt` exists, ask if they want to install dependencies

4. For Conda environment:
   - Ask for Python version (default: latest)
   - Create the environment using `conda create -n <name> python=<version>`
   - Provide activation instructions
   - If a `environment.yml` or `requirements.txt` exists, ask if they want to install dependencies

5. After creation, display:
   - Activation command for the user's OS
   - Next steps for installing dependencies
   - How to verify the environment is working
