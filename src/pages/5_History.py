import streamlit as st
from streamlit_elements import elements, mui
import pandas as pd
from backend.history import initialize_history_file
import io

st.set_page_config(layout="wide")

stt = st.session_state

st.session_state.sim_started = False

HISTORY_CSV_PATH = './history.csv'

initialize_history_file()

history_df = pd.read_csv(HISTORY_CSV_PATH)
history_df['id'] = range(history_df.shape[0])
history_df = history_df.set_index('id').reset_index()
history_df = history_df.sort_values(by='Date', ascending=False)
stt.history_df = history_df
stt.rows = history_df.to_dict(orient='records')

stt.columns = []
for col in history_df.columns:
    stt.columns.append({'field': col, 'headerName': col, 'flex': 1})
stt.columns[0]['flex'] = 0
stt.columns[1]['flex'] = 1.5
stt.columns[-1]['flex'] = 7

def delete_history():
    stt.history_df = pd.DataFrame(columns=stt.history_df.columns)
    stt.history_df.drop(columns=['id']).to_csv(HISTORY_CSV_PATH, index=False)
    return stt.history_df


_, col1, col2 = st.columns([7, 1, 1])

def download_history(cont):
    if stt.history_df.shape[0] > 0:
        start_date = pd.to_datetime(stt.history_df['Date']).min().strftime('%Y-%m-%d')
        end_date = pd.to_datetime(stt.history_df['Date']).max().strftime('%Y-%m-%d')

        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            stt.history_df.drop(columns=['id']).to_excel(writer, sheet_name=f"{start_date} to {end_date}", index=False)
        excel_bytes = output.getvalue()

        cont.download_button(
            label="Export History",
            data=excel_bytes,
            file_name=f"history_{start_date}_to_{end_date}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

download_history(col1)

if col2.button("Clear History"):
    history_df = delete_history()
    stt.rows = history_df.to_dict(orient='records')


def on_row_click(event):
    st.session_state.remarks = f'### Remarks for id {event["row"]["id"]}:\n\n{event["row"]["Remarks"]}'

if 'remarks' in st.session_state:
    st.sidebar.markdown(st.session_state.remarks)

with elements("update-history"):
    with mui.Stack(spacing=2):
        with mui.Paper(key='paper-1', sx={ "height": '800px', 'padding': 2, 'paddingBottom': 8}, elevation=0):
            mui.DataGrid(
                columns=stt.columns,
                rows=stt.rows,
                onRowClick=on_row_click
            )
