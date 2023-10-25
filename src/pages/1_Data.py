import streamlit as st
from streamlit_elements import elements, mui, html, dashboard, sync, lazy
import datetime
import pandas as pd

st.set_page_config(layout="wide")

stt = st.session_state

stt.sim_started = False

st.title("Upload Custom Excel Data")

uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
if uploaded_file is not None:
    stt.uploaded_file = uploaded_file

if 'uploaded_file' in stt:
    uploaded_file = stt.uploaded_file

if 'skip_rows' not in stt:
    stt.skip_rows = 0
    stt.skip_cols = 0
    stt.rename_dict = {}

if uploaded_file:
    st.write(f'Uploaded file: {uploaded_file.name}')

    # Choose the number of rows and columns to skip
    stt.skip_rows = st.slider("Number of rows to skip:", 0, 50, stt.skip_rows)
    stt.skip_cols = st.slider("Number of columns to skip from the start:", 0, 50, stt.skip_cols)

    # Read the Excel file with the specified number of skipped rows and columns
    data = pd.read_excel(uploaded_file, skiprows=stt.skip_rows)
    if stt.skip_cols:
        data = data.iloc[:, stt.skip_cols:]

    # Rename columns
    for col in data.columns:
        new_name = st.text_input(f"Rename column '{col}':", stt.rename_dict[col] if col in stt.rename_dict else col)
        if new_name != col:
            stt.rename_dict[col] = new_name
    data = data.rename(columns=stt.rename_dict)

    if st.button('Apply'):
        st.success('Data source changes applied.')

    # Display the preview of the data
    st.subheader("Preview of the data:")
    st.write(data.head())  # Display the top 5 rows for a quick preview
    
    stt.data = data
    stt.data_name = uploaded_file.name

with elements("models"):
    with mui.Stack(spacing=2):
        with mui.Paper(key='paper-1', sx={ 'padding': 2}, elevation=0):
            mui.Typography('PI Data API Endpoint', variant='h5')
            with mui.Stack(sx={'marginTop': 4}):
                mui.InputLabel('Specify the API endpoint url to ingest the input data automatically', sx={'marginBottom': 2})
                mui.TextField(label="Endpoint URL", defaultValue='https://api-endpoint/pi/data', onChange=lazy(sync('api_endpoint_url')))
       
