Prerequisites
Python 3.8+ installed
Git installed
Kaggle account
model_data.pt file from training
Step 1: Initial Setup
BASH

# 1. Create project directory
mkdir landcover_project
cd landcover_project

# 2. Create virtual environment
python -m venv landcover_env

# 3. Activate virtual environment
# On Windows:
.\landcover_env\Scripts\activate
# On Linux/Mac:
# source landcover_env/bin/activate
Step 2: Install Required Packages
BASH

# Install required packages
pip install torch torchvision flask opencv-python numpy matplotlib pillow requests tqdm kaggle
Step 3: Kaggle Setup
Go to kaggle.com → Your Account → Settings → API → Create New API Token
Download kaggle.json
Create .kaggle directory and move kaggle.json:
BASH

# On Windows:
mkdir %USERPROFILE%\.kaggle
move kaggle.json %USERPROFILE%\.kaggle\

# On Linux/Mac:
# mkdir ~/.kaggle
# mv kaggle.json ~/.kaggle/
Step 4: Create Project Files
Create server.py, client.py, and test_setup.py using the code provided earlier
Place your model_data.pt in the project directory
Step 5: Test Model Loading
BASH

# Run test_setup.py to verify model loads correctly
python test_setup.py

# Expected output:
# ✓ Model file found
# PyTorch version: 2.6.0+cpu
# Using device: cpu
# ✓ Model loaded successfully
# Model classes: 5
# Model channels: 3
# ✓ Model successfully switched to eval mode
Step 6: Start the Server
BASH

# In terminal 1 (keep this running)
python server.py

# Expected output:
# Loading model...
# Model loaded successfully
# * Running on http://0.0.0.0:5000
Step 7: Run the Client
BASH

# In terminal 2 (new terminal, activate environment first)
.\landcover_env\Scripts\activate
python client.py
Complete Directory Structure

landcover_project/
├── landcover_env/
├── model_data.pt
├── server.py
├── client.py
├── test_setup.py
└── landcoverai/  (will be created when client runs)
    ├── images/
    └── masks/
Troubleshooting Guide
If model doesn't load:
BASH

# Check if model_data.pt exists
dir model_data.pt
If server won't start:
BASH

# Check if port 5000 is already in use
# On Windows:
netstat -ano | findstr :5000
# Kill the process if needed
taskkill /PID <PID> /F
If client can't connect:
BASH

# Verify server is running
curl http://localhost:5000
If dataset won't download:
BASH

# Check Kaggle credentials
dir %USERPROFILE%\.kaggle\kaggle.json
Expected Workflow
First Run:
BASH

# Terminal 1
.\landcover_env\Scripts\activate
python server.py

# Terminal 2
.\landcover_env\Scripts\activate
python client.py
The client will:

Download the dataset (first time only)
Select random images
Send them to server for processing
Display results
Common Issues and Solutions
"ModuleNotFoundError":
BASH

pip install <missing_module>
"Port already in use":
BASH

# Change port in server.py:
app.run(host='0.0.0.0', port=5001)  # Try different port
"Model loading failed":
BASH

# Verify model file:
python test_setup.py
"Kaggle API not working":
BASH

# Check permissions:
icacls %USERPROFILE%\.kaggle\kaggle.json
# Should be only readable by your user
To Stop Everything
Close visualization windows: Close any matplotlib windows
Stop the client: Ctrl+C in client terminal
Stop the server: Ctrl+C in server terminal
Deactivate environment:
BASH

deactivate
To Run Again Later
Navigate to project directory
Activate environment
Start server
Run client
Remember:

Keep server running while using client
Dataset downloads only on first run
Check both terminals for error messages
model_data.pt must be in project directory
