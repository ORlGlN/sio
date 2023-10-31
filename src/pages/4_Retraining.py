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


CONFIG_PATH = './retraining_config.json'
HISTORY_CSV_PATH = './history.csv'

st.session_state.sim_started = False

def initialize_config():
    if not os.path.exists(CONFIG_PATH):
        default_config = {
            "error_threshold": 10,
            "target_var": "NTA Fuel",
            "selected_vars": [],
            "filtering_vars": [],
            "conditions": {},
            "auto_or_custom": "Auto",
            "min_sample_size": 600,
            "auto_when_conditions_fail": "Fallback up to 12 months",
            "retraining_cooldown_hr": 72,
            "algorithm": "Linear Regression",
            "manual_date_range": "Last 3 months",
            "target_var_manual": "NTA Fuel",
            "selected_vars_manual": [],
            "filtering_vars_manual": [],
            "conditions_manual": {},
            "algorithm_manual": "Linear Regression"
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

# Reading the dates from the config during initialization
if "auto_start_date" in st.session_state:
    if isinstance(st.session_state["auto_start_date"], str):
        st.session_state.auto_start_date = datetime.strptime(st.session_state["auto_start_date"], '%Y-%m-%d').date()

if "auto_end_date" in st.session_state:
    if isinstance(st.session_state["auto_end_date"], str):
        st.session_state.auto_end_date = datetime.strptime(st.session_state["auto_end_date"], '%Y-%m-%d').date()

if "manual_start_date" in st.session_state:
    if isinstance(st.session_state["manual_start_date"], str):
        st.session_state.manual_start_date = datetime.strptime(st.session_state["manual_start_date"], '%Y-%m-%d').date()

if "manual_end_date" in st.session_state:
    if isinstance(st.session_state["manual_end_date"], str):
        st.session_state.manual_end_date = datetime.strptime(st.session_state["manual_end_date"], '%Y-%m-%d').date()

# Set the page layout to wide mode
st.set_page_config(layout="wide")

# Load the data
@st.cache_data
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
st.markdown(f"## Automatic Re-Training Criteria")

# Error threshold slider
error_threshold = st.slider("Threshold for Moving Avg. of % Absolute Error", min_value=0, max_value=100, value=st.session_state.get("error_threshold", 10), format="%d%%")

# Model Algorithm Selection
algorithms = ["Linear Regression", "Lasso Regression", "Ridge Regression", "Elastic Regression", "Auto (Optimal model based on Adj-R2)"]
algorithm = st.selectbox("Select the model algorithm", algorithms, index=algorithms.index(st.session_state.get("algorithm", "Linear Regression")), key='algo_sel_box')

# Variables selection and filtering
variables = [c for c in data.columns if 'date' not in c.lower()]
target_var = st.selectbox("Choose Variable for Re-training Target Y", variables, index=variables.index(st.session_state.get("target_var", "NTA Fuel")), key='y_sel_box')
inp_vars = [v for v in variables if v != target_var]
selected_vars = st.multiselect("Choose Variable(s) for Re-training Input", inp_vars, default=st.session_state.get("selected_vars", []), key='auto_sel_vars')
filtering_vars = st.multiselect("Choose Variable(s) for Data Filtering", inp_vars, default=st.session_state.get("filtering_vars", []), key='auto_filtering_vars')

conditions = st.session_state.conditions

for var in list(conditions.keys()):  # Use list() to avoid 'dictionary size changed during iteration' error
    if var not in filtering_vars:
        del conditions[var]

for var in filtering_vars:
    condition = st.text_input(f"Condition for {var}" + ' (==, >, <, >=, <=, &, |, e.g., {var} > 3000), ({var} >= 15000) & ({var} <= 30000)', value=conditions.get(var, ''))
    conditions[var] = condition

st.session_state["conditions"] = conditions

# Date range selection
auto_or_custom = st.selectbox("Select Date Range Mode", ["Auto", "Custom Range"], index=0 if st.session_state.get("auto_or_custom", "Auto") == "Auto" else 1, key='aut_cus_sel_box')
end_date = data['Datetime'].max()

min_sample_size = st.session_state.get("min_sample_size", "3000")
retraining_cd_hr = st.session_state.get("retraining_cooldown_hr", 1)

auto_when_conditions_fail = st.session_state.auto_when_conditions_fail
earliest_date = pd.to_datetime(st.session_state.data['Datetime']).min()
latest_date = pd.to_datetime(st.session_state.data['Datetime']).max()

if auto_or_custom == "Auto":
    start_date = st.session_state.get("auto_start_date", earliest_date)
    end_date = st.session_state.get("auto_end_date", latest_date)

    # Allow user to configure the minimum sample size
    min_sample_size = st.text_input("Minimum sample size", value=min_sample_size)
    try:
        min_sample_size = int(min_sample_size)
    except ValueError:
        st.error("Please enter a valid integer for the minimum sample size.")
        min_sample_size = 3000  # Default value in case of an invalid input

    fallback_message = ""
    
    auto_when_conditions_fail = st.selectbox("When conditions failed to meet", ["Skip", "Fallback up to 12 months"], index=0 if st.session_state.get("auto_when_conditions_failed") == "Skip" else 1, key='aut_fallback_sel_box')

    # Use number_input to let the user set the cooldown period in hours
    retraining_cd_hr = st.number_input("Re-training Cooldown (hours)", min_value=1, max_value=999999, value=retraining_cd_hr)
    st.session_state["retraining_cooldown_hr"] = retraining_cd_hr

else:
    start_date = pd.Timestamp(st.date_input("Start Date", value=st.session_state.get("auto_start_date", earliest_date), key='auto_start_date_picker'))
    end_date = pd.Timestamp(st.date_input("End Date", value=st.session_state.get("auto_end_date", latest_date), key='auto_end_date_picker'))
    subset_data = data[(data['Datetime'] >= start_date) & (data['Datetime'] <= end_date)]

    st.markdown(f"### Statistics for Selected Date Range")
    display_statistics(subset_data)

    subset_data = filter_data(subset_data, conditions)

    if filtering_vars:
        st.markdown(f"### Statistics for Filtered Data")
        st.write(subset_data[selected_vars + [target_var]].describe().round(2))

# Manual Retraining section
with st.expander("Manual Re-Training"):
    date_ranges = ["Last 3 months", "Last 6 months", "Last 9 months", "Last 12 months", "Custom Range"]
    date_range_manual = st.selectbox("Choose Date Range for Manual Re-Training", date_ranges, index=date_ranges.index(st.session_state.get("manual_date_range", "Last 3 months")), key='manual_sel_box')
    
    if date_range_manual == "Custom Range":
        start_date_manual = pd.Timestamp(st.date_input("Start Date", value=st.session_state.get("manual_start_date", earliest_date), key='man_start_date_picker'))
        end_date_manual = pd.Timestamp(st.date_input("End Date", value=st.session_state.get("manual_end_date", latest_date), key='man_end_date_picker'))
    else:
        months = int(date_range_manual.split()[1])
        end_date_manual = data['Datetime'].max()
        start_date_manual = end_date_manual - pd.DateOffset(months=months)

    subset_data_manual = data[(data['Datetime'] >= start_date_manual) & (data['Datetime'] <= end_date_manual)]
    display_statistics(subset_data_manual)

    target_var_manual = st.selectbox("Choose Variable for Re-training Target Y", variables, index=variables.index(st.session_state.get("target_var_manual", "NTA Fuel")), key='y_sel_box_manual')
    inp_vars_manual = [v for v in variables if v != target_var_manual]
    selected_vars_manual = st.multiselect("Choose Variable(s) for Re-training Input", inp_vars_manual, default=st.session_state.get("selected_vars_manual", []), key='manual_sel_vars')
    filtering_vars_manual = st.multiselect("Choose Variable(s) for Data Filtering", inp_vars_manual, default=st.session_state.get("filtering_vars_manual", []), key='filtering_sel_vars')
    conditions_manual = st.session_state.conditions_manual

    for var in list(conditions_manual.keys()):  # Use list() to avoid 'dictionary size changed during iteration' error
        if var not in filtering_vars_manual:
            del conditions_manual[var]
            
    for var in filtering_vars_manual:
        condition_manual = st.text_input(f"Condition for {var}" + ' (==, >, <, >=, <=, &, |, e.g., {var} > 3000), ({var} >= 15000) & ({var} <= 30000)', value=conditions_manual.get(var, ''), key=f'manual_cond_{var}')
        conditions_manual[var] = condition_manual
    
    st.session_state[f"conditions_manual"] = conditions_manual

    subset_data_manual = filter_data(subset_data_manual, conditions_manual)
    if filtering_vars_manual:
        st.markdown(f"### Filtered Statistics for Selected Variables")
        st.write(subset_data_manual[selected_vars_manual + [target_var_manual]].describe().round(2))

    algorithm_manual = st.selectbox("Select the model algorithm", algorithms, index=algorithms.index(st.session_state.get("algorithm_manual", "Linear Regression")), key='manual_algo_sel_box')

    # Remarks input
    manual_remarks = st.text_area("Remarks", value="", key='manual_remarks')

    if st.button("Start Manual Retraining"):
        # Call retraining function
        weights, y_intercept, model, r2, adj_r2, algo_type = run_retraining(subset_data_manual, target_var_manual, selected_vars_manual, algorithm_manual)

        date_created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        config = {
            "retraining_type": "Manual",
            "algorithm": f'{algo_type} (Auto)' if algorithm_manual.startswith('Auto ') else algorithm_manual,
            "algo_type": algo_type,
            "target_var": target_var_manual,
            "selected_vars": selected_vars_manual,
            "filtering_vars": filtering_vars_manual,
            "conditions": conditions_manual,
            "date_range": date_range_manual,
            "start_date": start_date_manual.strftime('%Y-%m-%d'),
            "end_date": end_date_manual.strftime('%Y-%m-%d'),
            "ori_coeffs": weights.tolist(),
            "coeffs": weights.tolist(),
            "ori_y_intercept": y_intercept,
            "y_intercept": y_intercept,
            "r2": r2,
            "adj_r2": adj_r2,
            "remarks": manual_remarks,
            "created_on": date_created,
            "updated_on": date_created
        }
        
        filename, new_model_id = save_model_and_config(model, config, algorithm_manual)
        append_to_hist(HISTORY_CSV_PATH, new_model_id, f'Manual re-training completed with adj. R2 of {adj_r2:.3f}. {manual_remarks}', is_auto=False)
        st.success(f"Model re-trained successfully and saved as {filename}!")

# Saving the state to the configuration file

config = {
    "error_threshold": error_threshold,
    "target_var": target_var,
    "selected_vars": selected_vars,
    "filtering_vars": filtering_vars,
    "conditions": conditions,
    "auto_or_custom": auto_or_custom,
    "auto_start_date": start_date.strftime('%Y-%m-%d') if start_date is not None else start_date,
    "auto_end_date": end_date.strftime('%Y-%m-%d') if end_date is not None else end_date, 
    "auto_when_conditions_fail": auto_when_conditions_fail, 
    "min_sample_size": min_sample_size,
    "retraining_cooldown_hr": retraining_cd_hr,
    "algorithm": algorithm,
    "manual_date_range": date_range_manual,
    "manual_start_date": start_date_manual.strftime('%Y-%m-%d'),
    "manual_end_date": end_date_manual.strftime('%Y-%m-%d'),
    "target_var_manual": target_var_manual,
    "selected_vars_manual": selected_vars_manual,
    "filtering_vars_manual": filtering_vars_manual,
    "conditions_manual": conditions_manual,
    "algorithm_manual": algorithm_manual
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

        if isinstance(v, datetime):
            v = str(v.strftime('%Y-%m-%d'))
            prev_v = str(prev_v.strftime('%Y-%m-%d'))

        if str(v).strip() != str(prev_v).strip():
            changes.append(f'{k} ({prev_v} to {v})')

    if len(changes) > 0:
        append_to_hist(HISTORY_CSV_PATH, None, f'Re-training settings changed: {", ".join(changes)}')

    save_config(config)

    st.success("Configuration saved successfully!")
