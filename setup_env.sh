#!/bin/bash

# 1. Define the environment name/folder
ENV_NAME=".venv"

# 2. Create the virtual environment in the current directory
echo "Creating virtual environment in $ENV_NAME..."
python3 -m venv $ENV_NAME

# 3. Activate the environment
# For bash/zsh
source $ENV_NAME/bin/activate

# 4. Upgrade pip for stability
pip install --upgrade pip

# 5. Install requirements if the file exists
if [ -f "ASL/requirements.txt" ]; then
    echo "Installing packages from requirements.txt..."
    pip install -r ASL/requirements.txt
    echo "Installation complete."
else
    echo "Error: requirements.txt not found!"
fi