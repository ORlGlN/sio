import streamlit as st
import pandas as pd
import os
import json
import threading
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import time
from datetime import datetime
from backend.model import run_retraining, save_model_and_config
from backend.history import append_to_hist
import copy


CONFIG_PATH = './monitoring_config.json'
HISTORY_CSV_PATH = './history.csv'

def initialize_config():
    if not os.path.exists(CONFIG_PATH):
        default_config = {
            "filtering_vars": [],
            "conditions": {},
        }
        for key, value in default_config.items():
            st.session_state[key] = value

        save_config(default_config)
    else:
        with open(CONFIG_PATH, 'r') as f:
            default_config = json.load(f)
        for key, value in default_config.items():
            st.session_state[key] = value

    return default_config

def save_config(config):
    with open(CONFIG_PATH, 'w') as f:
        json.dump(config, f, indent=4)

default_config = copy.deepcopy(initialize_config())

# Set the page layout to wide mode
st.set_page_config(layout="wide")

st.session_state.sim_started = False

def load_data():
    df = pd.read_excel('./input/Y2 NTA fuel (raw data).xlsx', skiprows=2)
    df = df.drop(columns=['Unnamed: 0'])
    df = df.rename(columns={df.columns[0]: 'Datetime'})
    return df

if 'data' not in st.session_state:
    st.session_state.data = load_data()

data = st.session_state.data

def display_statistics(data):
    st.write(data.describe().round(2))

def filter_data(data, conditions):
    for variable, condition in conditions.items():
        if condition:
            try:
                condition = condition.replace('{var}', f"data['{variable}']")
                condition_expr = eval(condition)
                data = data[condition_expr]
            except Exception as e:
                st.error(f"Error in filtering data for variable {variable} with condition {condition}: {str(e)}")
                return pd.DataFrame()  # Return an empty DataFrame
    return data

# Automatic Re-Training Criteria section
st.markdown(f"## Monitoring Criteria")

# Variables selection and filtering
variables = [c for c in data.columns if 'date' not in c.lower()]
filtering_vars = st.multiselect("Choose Variable(s) for Data Filtering", variables, default=st.session_state.get("filtering_vars", []), key='monitoring_filtering_vars')

conditions = st.session_state.conditions

for var in list(conditions.keys()):  # Use list() to avoid 'dictionary size changed during iteration' error
    if var not in filtering_vars:
        del conditions[var]

for var in filtering_vars:
    condition = st.text_input(f"Condition for {var}" + ' (==, >, <, >=, <=, &, |, e.g., {var} > 3000), ({var} >= 15000) & ({var} <= 30000)', value=conditions.get(var, ''))
    conditions[var] = condition

st.session_state["conditions"] = conditions

config = {
    "filtering_vars": filtering_vars,
    "conditions": conditions,
}

if st.button('Save Configuration'):    
    changes = []

    for k, v in config.items():
        if k in default_config:
            prev_v = default_config[k]
        else:
            prev_v = ''

        if isinstance(v, dict): 
            v = str(v)
            prev_v = str(prev_v)

        if str(v).strip() != str(prev_v).strip():
            changes.append(f'{k} ({prev_v} to {v})')

    if len(changes) > 0:
        append_to_hist(HISTORY_CSV_PATH, None, f'Monitoring settings changed: {", ".join(changes)}')

    save_config(config)

    st.success("Configuration saved successfully!")
