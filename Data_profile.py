import pandas as pd
import csv
import tempfile
import os
from pandas_profiling import ProfileReport


def detect_delimiter(csv_file):
	with open(csv_file, 'r') as file:
		dialect = csv.Sniffer().sniff(file.read(100000))
		print(dialect)
		return dialect.delimiter


def load_data(file):
	temp_file = tempfile.NamedTemporaryFile(delete=False)
	temp_file.write(file.read())
	temp_file.close()
	file_path = temp_file.name
	delimiter = detect_delimiter(file_path)
	df = pd.read_csv(file_path, delimiter=delimiter)
	os.remove(file_path)
	return df

def profiling(df): 
	profile = ProfileReport(df, title="Data Profile", explorative=True)
	profile_path = "data_profile.html"
	profile.to_file(profile_path)
	# Read the HTML file
	with open(profile_path, "r") as f:
		profile_html = f.read()

	return profile_html