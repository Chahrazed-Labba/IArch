# IArch

## Presentation :

IArch is a tool that allows archaeologists to make Explainable Artificial Intelligence (XAI) data analyses without having specific programming expertise. The platform covers the complete ML workflow, from data processing and feature selection to applying the ML models and explaining the predictions using the SHapley Additive exPlanations (SHAP). The last version (2025) includes now Social Network Analyses (SNA). 

## Notes : 
 - This project is still under development, and further experiments with archaeologists are planned for the near future.
- If this is you first installation, it could take some time and storage into your device. We recommand to have an internet connection and sufficient free storage. 

## Requirements : 
- Python version should be at least python 3.9.0 (we recommand this version)
- The rest of the requirements are expressed within the file requirements.txt
- To avoid conflicts with existing python configurations, we recommend using a virtual environment. 

# Installation for Windows
## This is the first time you install python on your computer
### First install python 3.9.0 properly, without affecting system. 
Go to the official Python website:
👉 https://www.python.org/downloads/release/python-390/

Download the Windows Installer (64-bit) or Windows Installer (32-bit) depending on your system architecture.

Open the downloaded file (python-3.9.x-amd64.exe).

Check the box: "Add Python 3.9 to PATH" (very important for using it in the command line).

Click "Install Now" and wait for the installation to complete.

Verify the installation:
```bash
python --version
```
Now, verify pip is correctly install :
```bash
pip --version
```
If it's not installed, run:
```bash
python -m ensurepip
```

### Then create a virtual environment
Verify if venv is installed
```bash
python -m venv --help
```
Then, navigate to the folder where you want to create the environment (for example, C:\my_project):
```bash
cd C:\mon_projet
```
And create a new virtual environment (let's name it env):
```bash
python -m venv env
```
After creating it, you need to activate it:
```bash
env\Scripts\activate
```
When trying to activate the virtual environment in PowerShell, you may encounter an error like this:
```bash
venv\Scripts\activate : File ... cannot be loaded because running scripts is disabled on this system.
```
A simple solution is to temporarily allow script execution only for the current PowerShell session. Run the following command directly in the your current terminal:
```bash
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
After running the command above, activate the virtual environment:
```bash
.\venv\Scripts\Activate.ps1
```
If everything works correctly, your PowerShell prompt will change and display the virtual environment name, for example:
```bash
(venv) PS C:\Users\YourName\MyProject>
```

### Install the requirements file
You can now install the requirements file
```bash
pip install -r requirements.txt
``` 

### Finally run the App:
```bash
streamlit run app.py
```


# Installation for MacOS
## This is the first time you install python on your computer
### First install python 3.9.0 properly, without affecting system. 
Copy/Past each line of code into the terminal or command invite

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
You will have to type your passeword. It's the admin password of the device. 
Then you will be ask to press **enter**. Do it. 

Then install **pyenv**

```bash
brew install pyenv
```
Wait for the installation to finish

Now add **pyenv** to PATH
```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init --path)"' >> ~/.zshrc
source ~/.zshrc
```

Finally install Python 3.9.0 and verify the installation
```bash
pyenv install 3.9.0
```
```bash
pyenv versions
```

### Now you have to determine python 3.9.0 as the actual version into the app folder
```bash
cd my_project
```
If the app folder is on an external device as external hard drive, you should enter:
```bash
cd /Volumes/name_of_external_device/name_of_app_folder
```

After, load python 3.9.0, create a virtual environment and activate it
```bash
pyenv local 3.9.0
```
```bash
python3 -m venv my_venv
```
```bash
source my_venv/bin/activate
```


### And finally, you need to install all the necessary dependencies for the application to work properly (this may take some time). But first, on MacOS, GDAL and Rust must be installed manually before installing certain dependencies. Here's how to do it :
```bash
brew install gdal
```
Verify that gdal is correctly configure
```bash
which gdal-config
```
And add it to your virtual environnement
```bash
export GDAL_CONFIG=$(which gdal-config)
export GDAL_VERSION=$(gdal-config --version)
```
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
```bash
source $HOME/.cargo/env
```


You can now install the requirements file
```bash
pip install -r requirements_for_mac.txt
``` 

### Final step. When the installation of the requirements.txt file is complete, you can run the app with this command :
```bash
streamlit run app.py
```
The application will then open in your default browser. You can now use it without an internet connection.

## The next time you use IArch, you won't have to reinstall all the dependencies. You just need to specify the application folder and the virtual environment. 
Navigate to the project directory (where you created the virtual environment):
```bash
cd /path/to/IArch
```
Activate the virtual environment
```bash
source my_venv/bin/activate
```
Run IArch:
```bash
streamlit run app.py
```
