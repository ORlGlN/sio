import csv
import os
from datetime import datetime

HISTORY_CSV_PATH = './history.csv'

def initialize_history_file(hist_path=HISTORY_CSV_PATH):
    if not os.path.exists(hist_path):
        headers = ["Date", "Model ID", "Retraining Type", "Remarks"]
        
        with open(hist_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(headers)

def append_to_hist(hist_path, model_id, remarks, is_auto=True):
    with open(hist_path, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        dt = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        csvwriter.writerow([dt, model_id, 'Auto' if is_auto else 'Manual', remarks])

initialize_history_file()