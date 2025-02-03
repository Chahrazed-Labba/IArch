# IArch

## Presentation :

IArch is a tool that allows archaeologists to make Explainable Artificial Intelligence (XAI) data analyses without having specific programming expertise. The platform covers the complete ML workflow, from data processing and feature selection to applying the ML models and explaining the predictions using the SHapley Additive exPlanations (SHAP). 

## Notes : 
 - This project is still under development, and further experiments with archaeologists are planned for the near future.
 - In its current version the tool supports only the analysis  of numerical categorical data.

## Environment : 
- Python version should be at least python 3.9
- The rest of the requirements are expressed within the file requirements.txt
- To avoid conflicts with existing pyton configurations, we recommend using a virtual environment. 

Create the environment.
```bash
python -m venv venv
```
Then activate it.

On windows
```bash
venv\Scripts\activate
```

On Mac
```bash
source venv/bin/activate
```

## Location file : 
Provide information on the direction of the folder containing the application 
```bash
cd D:\IArch-Visual studio - streamlit
```
This is an example, change your information. 

## Requirements :
And finally install the requirements
```bash
pip install -r requirements.txt
```

## Finally run the App : 
```bash
streamlit run app.py
```
