import pandas as pd
import csv
import tempfile
import os
from ydata_profiling import ProfileReport
import numpy as np
import shap
from statistics import mean
import matplotlib.pyplot as plt
import pdfkit
import time



# Create a fonction to detect the delimiter and dialect of the dataset

def detect_delimiter(csv_file):
	with open(csv_file, 'r') as file:
		dialect = csv.Sniffer().sniff(file.read(100000))
		print(dialect)
		return dialect.delimiter




def upload_data(file):
	temp_file = tempfile.NamedTemporaryFile(delete=False)
	temp_file.write(file.read())
	temp_file.close()
	file_path = temp_file.name
	delimiter = detect_delimiter(file_path)
	df = pd.read_csv(file_path, delimiter = delimiter)
	os.remove(file_path)
	return df

def profiling(df):
    # Chemin du répertoire et du fichier
    profile_dir = "Data_profile"
    profile_html_path = os.path.join(profile_dir, "data_profile.html")

    # Vérifier si le répertoire existe, sinon le créer
    if not os.path.exists(profile_dir):
        os.makedirs(profile_dir)

    # Créer le profil et le sauvegarder dans le fichier HTML
    profile = ProfileReport(df, title="Data Profile", explorative=True)
    profile.to_file(profile_html_path)

    # Lire le fichier HTML et le retourner
    with open(profile_html_path, "r") as f:
        profile_html = f.read()
    return profile_html
    
