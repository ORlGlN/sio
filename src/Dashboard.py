import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
from sklearn.linear_model import LinearRegression
import numpy as np
import json
import joblib
import os
from glob import glob
from backend.model import run_retraining, save_model_and_config
from backend.history import initialize_history_file, append_to_hist
import threading
from datetime import datetime
import warnings
import math

warnings.simplefilter("ignore")

stt = st.session_state

# Set the page layout to wide mode
st.set_page_config(layout="wide")

MONITORING_CONFIG_PATH = './monitoring_config.json'
RETRAINING_CONFIG_PATH = './retraining_config.json'
MODEL_CONFIG_PATH = './model_config.json'
MODEL_VAULT_DIR = './model_vault/'
HISTORY_CSV_PATH = './history.csv'
DEFAULT_DATA_PATH = './input/Y2 NTA fuel (raw data).xlsx'

initialize_history_file(HISTORY_CSV_PATH)

if not os.path.exists(RETRAINING_CONFIG_PATH):
    st.warning('WARNING: No retraining_config.json found, please navigate to Retraining page to initiate it.')
    time.sleep(999999)

with open(RETRAINING_CONFIG_PATH, 'r') as f:
    retraining_config = json.load(f)
    if len(retraining_config['selected_vars']) == 0 or 'auto_start_date' not in retraining_config or 'auto_end_date' not in retraining_config:
        st.warning('WARNING: No variable is selected to be included for auto re-training, please navigate to Retraining page to change it.')
        time.sleep(999999)      

if not os.path.exists(MONITORING_CONFIG_PATH):
    st.warning('WARNING: No monitoring_config.json found, please navigate to Monitoring page to initiate it.')
    time.sleep(999999)

with open(MONITORING_CONFIG_PATH, 'r') as f:
    monitoring_config = json.load(f)

if not os.path.exists(MODEL_CONFIG_PATH):
    st.warning('WARNING: No model_config.json found, please navigate to Models page to initiate it or Re-training page to manually create a model first.')
    time.sleep(999999)

with open(MODEL_CONFIG_PATH, 'r') as f:
    model_config = json.load(f)

if len(model_config['deployed_models']) == 0:
    st.warning('WARNING: No model is currently deployed, please navigate to Models page to initiate it.')
    time.sleep(999999)

def get_curr_time_str():
    # return datetime.fromisoformat(datetime.now().astimezone().isoformat()).strftime('%Y-%m-%d %H:%M:%S %z')
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class NList:
    def __init__(self, max_size=5, default=''):
        self.max_size = max_size
        self._items = [default]

    def add(self, item):
        if len(self._items) >= self.max_size:
            self._items.pop(-1)  # Remove the oldest/first item
        self._items = [f'<b>{get_curr_time_str()}:</b><br/>{item}'] + self._items

    def get_last(self):
        return self._items[0]

    def get_items(self):
        return self._items

    def __len__(self):
        return len(self._items)

    def __str__(self):
        return str(self._items)
    
def load_monitoring_models(config):
    selected_model_ids = config['monitoring_models']

    monitoring_models = []

    for model_id in selected_model_ids:
        with open(glob(f"{MODEL_VAULT_DIR}/SIO-{model_id}-*.pkl")[0], 'rb') as f:
            model = joblib.load(f)
        with open(glob(f"{MODEL_VAULT_DIR}/SIO-{model_id}-*.json")[0], 'r') as f:
            metadata = json.load(f)

        # Load edited coeffs and y-intercepts
        if isinstance(metadata['coeffs'], list) and len(metadata['coeffs']) > 0:
            model.coef_ = np.asarray(metadata['coeffs'], dtype=np.float32) 
        
        if isinstance(metadata['y_intercept'], float):
            model.intercept_ = metadata['y_intercept']

        monitoring_models.append({
            'id': model_id,
            'model': model,
            'metadata': metadata
        })
    return monitoring_models

def load_data():
    df = pd.read_excel(DEFAULT_DATA_PATH, skiprows=2)
    df = df.drop(columns=['Unnamed: 0'])
    df = df.rename(columns={df.columns[0]: 'Datetime'})
    data_name = DEFAULT_DATA_PATH.split('/')[-1]
    return df, data_name

# def calc_errs(df):
#     # Train a simple regression model
#     X = df[["H2", "CO", "CO2 Recycle", "H2 Rich Fuel", "S/C"]]
#     y = df["NTA Fuel"]
#     model = LinearRegression().fit(X, y)
#     df["Predicted NTA Fuel"] = model.predict(X)

#     df["Absolute Error"] = abs(df["NTA Fuel"] - df["Predicted NTA Fuel"])

#     df["Error"] = df["Predicted NTA Fuel"] - df["NTA Fuel"]
#     df["% Error"] = (df["Error"] / df["NTA Fuel"]) * 100

#     return df

def calc_errs(df, models):
    for m in models:
        model_id = m['id']
        metadata = m['metadata']
        model = m['model']
        X = df[metadata['selected_vars']]
        df[f"Predicted NTA Fuel SIO-{model_id}"] = model.predict(X.fillna(0))

        df[f"Absolute Error SIO-{model_id}"] = abs(df["NTA Fuel"] - df[f"Predicted NTA Fuel SIO-{model_id}"])
        df[f"Error SIO-{model_id}"] = df[f"Predicted NTA Fuel SIO-{model_id}"] - df["NTA Fuel"]
        df[f"% Error SIO-{model_id}"] = (df[f"Error SIO-{model_id}"] / df["NTA Fuel"]) * 100
    return df

def filter_monitoring(data, conditions):
    for variable, condition in conditions.items():
        if condition:
            try:
                condition = condition.replace('{var}', f"data['{variable}']")
                condition_expr = eval(condition)
                data.loc[~condition_expr, [c for c in data.columns if 'date' not in c.lower()]] = np.nan
            except Exception as e:
                st.error(f"Error in filtering data for variable {variable} with condition {condition}: {str(e)}")
                return pd.DataFrame()  # Return an empty DataFrame
    return data

if 'data' not in stt or 'data_name' not in stt:
    stt.data, stt.data_name = load_data()

# stt.data = calc_errs(stt.data)

data = stt.data
monitoring_models = load_monitoring_models(model_config)
deployed_model = model_config["deployed_models"][0]

try: 
    data = calc_errs(data, monitoring_models)
except KeyError as e:
    st.error(f'Please check to make sure your data source contains these columns: {e}')
    time.sleep(999999)

err_thres = retraining_config['error_threshold']
retrain_cd_hr = retraining_config['retraining_cooldown_hr']

retrain_msgs = NList(5)

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

def retrain_model_with_status(model_id, data, retraining_conf, curr_err, err_thres, cd_hr):

    ## preprocess data
    target_var = retraining_conf['target_var']
    selected_vars = retraining_conf['selected_vars']
    filtering_vars = retraining_conf['filtering_vars']
    conditions = retraining_conf['conditions']
    auto_or_custom = retraining_conf['auto_or_custom']
    auto_start_date = retraining_conf['auto_start_date']
    auto_end_date = retraining_conf['auto_end_date']
    min_sample_size = retraining_conf['min_sample_size']
    algorithm = retraining_conf['algorithm']
    when_conditions_fail = retraining_conf['auto_when_conditions_fail']

    # Determine the maximum available months in the data
    max_available_months = math.ceil((data['Datetime'].max() - data['Datetime'].min()).days / 30)

    # Determine the appropriate date range
    are_conditions_met = False

    if auto_or_custom == "Custom Range" and auto_start_date is not None and auto_end_date is not None:
        end_date = auto_end_date
        start_date = auto_start_date
        subset_data = data[(data['Datetime'] >= start_date) & (data['Datetime'] <= end_date)]
        subset_data_filtered = filter_data(subset_data, conditions)

        months = (end_date.year - start_date.year) * 12 + end_date.month - start_date.month

        if len(subset_data_filtered) >= min_sample_size:
            are_conditions_met = True
        
    else:
        end_date = data['Datetime'].max()

        # Generate the months intervals starting from the max_available_months and decreasing by 3
        months_intervals = list(range(max_available_months, 0, -3))

        for months in months_intervals:
            start_date = end_date - pd.DateOffset(months=months)
            subset_data = data[(data['Datetime'] >= start_date) & (data['Datetime'] <= end_date)]
            subset_data_filtered = filter_data(subset_data, conditions)

            if len(subset_data_filtered) >= min_sample_size:
                are_conditions_met = True
                break


    if not are_conditions_met and when_conditions_fail == "Skip":
        retrain_msgs.add(f'Last re-training for SIO-{model_id} is skipped due to unmet conditions.')
        append_to_hist(HISTORY_CSV_PATH, model_id, 'Re-training skipped due to unmet conditions')
        return

    condition_str = 'All conditions met' if are_conditions_met else f'Proceeding with last {months} months data although conditions unmet.'

    remarks = f'Run re-training for SIO-{model_id} due to moving avg abs % err of {curr_err:.2f} % (> {err_thres}%). ' + condition_str
    append_to_hist(HISTORY_CSV_PATH, model_id, f'Retraining due to moving avg abs % err of {curr_err:.2f} % (> {err_thres}%). ' + condition_str)
    retrain_msgs.add(remarks)


    ## run training
    try:
        weights, y_intercept, model, r2, adj_r2, algo_type = run_retraining(subset_data_filtered, target_var, selected_vars, algorithm)
        
        date_created = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        config = {
            "retraining_type": "Auto",
            "algorithm": f'{algo_type} (Auto)' if algorithm.startswith('Auto ') else algorithm,
            "algo_type": algo_type,
            "target_var": target_var,
            "selected_vars": selected_vars,
            "filtering_vars": filtering_vars,
            "conditions": conditions,
            "start_date": start_date.strftime('%Y-%m-%d'),
            "end_date": end_date.strftime('%Y-%m-%d'),
            "ori_coeffs": weights.tolist(),
            "coeffs": weights.tolist(),
            "ori_y_intercept": y_intercept,
            "y_intercept": y_intercept,
            "r2": r2,
            "adj_r2": adj_r2,
            "remarks": remarks,
            "created_on": date_created,
            "updated_on": date_created
        }
        
        filename, new_model_id = save_model_and_config(model, config, algorithm)

        retrain_msgs.add(f'Last re-training for SIO-{model_id} completed. SIO-{new_model_id} with adj. R2 of {adj_r2:.2f}. Cooldown: {cd_hr} hrs.')
        append_to_hist(HISTORY_CSV_PATH, model_id, f'Re-training completed. New model SIO-{new_model_id} with adj. R2 of {adj_r2:.2f}.')

    except Exception as e:
        if subset_data_filtered.shape[0] == 0:
            retrain_msgs.add(f'Re-training for SIO-{model_id} skipped due to empty data after filtering.')
            append_to_hist(HISTORY_CSV_PATH, model_id, 'Re-training skipped due to empty data after filtering with specified conditions.')
        else:
            retrain_msgs.add(f'Re-training for SIO-{model_id} skipped due to unexpected errors.')
            append_to_hist(HISTORY_CSV_PATH, model_id, f'Re-training skipped due to unexpected errors: {e}')


if 'sim_started' not in stt:
    stt.sim_started = False

speed_labels = ["1x (real-time)", "2x", "3x", "4x", "5x (No simulated delay, up to server spec)", "50x (Graph update at N interval)", "Max (Graph update after simulation done)"]
delay_values = [3600, 1, 0.5, 0.25, 0, 0, 0]
speed_dict = dict(zip(speed_labels, delay_values))
selected_speed_label = st.sidebar.selectbox("Speed (Slowest to Fastest)", options=speed_labels, index=5, disabled=stt.sim_started)
delay = speed_dict[selected_speed_label]

is_normal_spds = selected_speed_label not in [speed_labels[-1], speed_labels[-2]]
is_max_spd = selected_speed_label == speed_labels[-1]
is_graph_n_interval = selected_speed_label == speed_labels[-2]

if is_graph_n_interval:
    graph_update_int = st.sidebar.slider('Graph Update Interval (every N samples, 1 sample = 1 hr)', 1, 2160, 720, disabled=stt.sim_started)

variables = [c for c in data.columns if 'date' not in c.lower() or 'error' not in c.lower()]
color_map = dict(zip(variables, px.colors.qualitative.Plotly))

ts_cols = st.sidebar.multiselect("Select Variables for Time Series", options=variables, disabled=stt.sim_started)
hist_cols = st.sidebar.multiselect("Select Variables for Histogram", options=variables, disabled=stt.sim_started)

error_placeholder = st.empty()
error_diff_moving_avg_placeholder = st.empty()

error_diff_placeholder = st.empty()
ts_placeholder = st.empty()

col1, col2 = st.columns(2)
hist_placeholders = {hist_cols[i]: (col1.empty() if i % 2 == 0 else col2.empty()) for i in range(len(hist_cols))}


def on_start_sim_click():
    stt.sim_started = not stt.sim_started

def on_restart_sim_click():
    stt.sim_started = False
    last_retrain_hrs = -retrain_cd_hr*2 # placeholder to make sure it is larger than cd to run at first time

if not stt.sim_started:
    st.sidebar.button("Start Simulation", on_click=on_start_sim_click)
else:
    st.sidebar.button('Reset Simulation', on_click=on_restart_sim_click)

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()

st.sidebar.markdown('## Notifications')
sidebar_retrain_status = st.sidebar.empty()
if len(retrain_msgs) == 1 and retrain_msgs.get_last() == '': sidebar_retrain_status.markdown('No notification at the moment')
st.sidebar.markdown('---')


st.sidebar.markdown(f'Data Source: {stt.data_name}')
st.sidebar.markdown(f'Deployed Model: SIO-{deployed_model}')

try:
    earliest_date = pd.to_datetime(stt.data['Datetime']).min()
    latest_date = pd.to_datetime(stt.data['Datetime']).max()
except KeyError as e:
    st.error(f'Please check to make sure your data source contains these columns: {e}')
    time.sleep(999999)

if 'retrain_msg' not in stt:
    stt['retrain_msg'] = ''

last_retrain_hrs = -retrain_cd_hr*2 # placeholder to make sure it is larger than cd to run at first time

start_date = pd.Timestamp(st.sidebar.date_input("Start Date", value=earliest_date, key='auto_start_date_picker', disabled=stt.sim_started))
end_date = pd.Timestamp(st.sidebar.date_input("End Date", value=latest_date, key='auto_end_date_picker', disabled=stt.sim_started))

if stt.sim_started:    

    last_index = len(data) - 1
    
    unfiltered_data = data[(data['Datetime'] >= start_date) & (data['Datetime'] <= end_date)].copy()
    filtered_data = filter_monitoring(data[(data['Datetime'] >= start_date) & (data['Datetime'] <= end_date)].copy(), monitoring_config['conditions'])


    for index in range(0, last_index):
        subset_data = filtered_data.iloc[:index+1]
        subset_data_unfiltered = unfiltered_data.iloc[:index+1]

        window_size = 7 * 24  # Assuming the data is hourly, so 7 days would be 7*24 hours

        if stt['retrain_msg'] != retrain_msgs.get_last():
            stt['retrain_msg'] = retrain_msgs.get_last()
            stat_str = ''
            for tt in retrain_msgs.get_items():
                if tt != '':
                    stat_str += f'🔔 {tt}\n\n'
            sidebar_retrain_status.markdown(stat_str, unsafe_allow_html=True)

        for m in monitoring_models:
            model_id = m['id']

            subset_data[f"Absolute % Error SIO-{model_id}"] = (subset_data[f"Absolute Error SIO-{model_id}"] / subset_data["NTA Fuel"]) * 100
            subset_data[f"Moving Avg Abs % Error SIO-{model_id}"] = subset_data[f"Absolute % Error SIO-{model_id}"].rolling(window=window_size).mean()
            # subset_data[f"Moving Avg % Error SIO-{model_id}"] = subset_data[f"% Error SIO-{model_id}"].rolling(window=window_size).mean()

            if model_id == deployed_model and subset_data[f"Moving Avg Abs % Error SIO-{model_id}"].iloc[-1] != np.nan and subset_data[f"Moving Avg Abs % Error SIO-{model_id}"].iloc[-1] > err_thres:
                # print('EXCEEED!!', subset_data[f"Moving Avg Abs % Error SIO-{model_id}"])
                hrs_since_last_retrain = index + 1 - last_retrain_hrs

                if hrs_since_last_retrain > retrain_cd_hr:
                    threading.Thread(target=retrain_model_with_status, args=(model_id, subset_data, retraining_config, subset_data[f"Moving Avg Abs % Error SIO-{model_id}"].iloc[-1], err_thres, retrain_cd_hr)).start()
                    last_retrain_hrs = index + 1

        if is_normal_spds or (is_max_spd and index == (last_index - 1)) or (is_graph_n_interval and (index % graph_update_int == 0 or index == (last_index - 1))):

            fig_error = px.line(subset_data, x='Datetime', y=[f'Moving Avg Abs % Error SIO-{m["id"]}' for m in monitoring_models], title="Moving Average of Prediction Absolute % Error", template="plotly_white")
            fig_error.add_shape(go.layout.Shape(type="rect", x0=subset_data["Datetime"].min(), x1=subset_data["Datetime"].max(), y0=0, y1=err_thres, fillcolor="lightgreen", opacity=0.5, line=dict(width=0)))
            
            darker_pastel_green = "#4CAF75"
            for y_val in [err_thres, 0]:
                fig_error.add_shape(go.layout.Shape(type="line", x0=subset_data["Datetime"].min(), x1=subset_data["Datetime"].max(), y0=y_val, y1=y_val, line=dict(color=darker_pastel_green, dash="dot")))

            fig_error.update_layout(legend=dict(y=1.1, x=0.5, xanchor='center', orientation='h', title=None))
            # fig_error.update_layout(yaxis_range=y_axis_limit_abs)
            error_placeholder.plotly_chart(fig_error, use_container_width=True)

            if len(ts_cols) > 0:
                fig_ts = px.line(subset_data_unfiltered, x='Datetime', y=ts_cols, title="Time Series (Unfiltered)", template="plotly_white", color_discrete_map=color_map)
                fig_ts.update_layout(legend=dict(y=1.1, x=0.5, xanchor='center', orientation='h', title=None))
                ts_placeholder.plotly_chart(fig_ts, use_container_width=True)

            ### Error Diff Chart
            fig_error_diff = px.line(subset_data, x='Datetime', y=[f"Predicted NTA Fuel SIO-{m['id']}" for m in monitoring_models] + ['NTA Fuel'], title="Raw Error Diff (Pred - y)", template="plotly_white")
            fig_error_diff.update_layout(legend=dict(y=1.1, x=0.5, xanchor='center', orientation='h', title=None))
            error_diff_placeholder.plotly_chart(fig_error_diff, use_container_width=True)

            ### Moving Avg Error Diff ChartW
            fig_moving_avg_percent_error_diff = px.line(subset_data, x='Datetime', y=[f"% Error SIO-{m['id']}" for m in monitoring_models], title="% Error Diff (Pred - y) / y", template="plotly_white")
            fig_moving_avg_percent_error_diff.add_shape(go.layout.Shape(type="rect", x0=subset_data["Datetime"].min(), x1=subset_data["Datetime"].max(), y0=-err_thres, y1=err_thres, fillcolor="lightgreen", opacity=0.5, line=dict(width=0)))
            # fig_moving_avg_percent_error_diff.update_layout(yaxis_range=y_axis_limit_diff)
            darker_pastel_green = "#4CAF75"
            for y_val in [-err_thres, err_thres]:
                fig_moving_avg_percent_error_diff.add_shape(go.layout.Shape(type="line", x0=subset_data["Datetime"].min(), x1=subset_data["Datetime"].max(), y0=y_val, y1=y_val, line=dict(color=darker_pastel_green, dash="dot")))
            
            fig_moving_avg_percent_error_diff.update_layout(legend=dict(y=1.1, x=0.5, xanchor='center', orientation='h', title=None))
            error_diff_moving_avg_placeholder.plotly_chart(fig_moving_avg_percent_error_diff, use_container_width=True)

            for col in hist_cols:
                fig_hist = px.histogram(subset_data, x=col, title=f"{col} Distribution", marginal="box", nbins=50, template="plotly_white", color_discrete_sequence=[color_map[col]])
                hist_placeholders[col].plotly_chart(fig_hist)

        progress_bar.progress(index/last_index)
        status_text.text(f"Simulating {index}/{last_index}")

        if delay > 0: time.sleep(delay)

    status_text.text("Simulation Completed!")

    last_retrain_hrs = -retrain_cd_hr*2 # placeholder to make sure it is larger than cd to run at first time

else:
    st.title('SIO Regression Auto-Retraining Prototype Tool')
    st.info("Use the sidebar to get started.")
