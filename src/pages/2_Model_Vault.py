import streamlit as st
import os
import json
from datetime import datetime
from backend.history import append_to_hist
import pandas as pd
import io
import time


MODEL_METADATA_PATH = './model_metadata.json'
MODEL_VAULT_DIR = './model_vault/'
MODEL_CONFIG_PATH = './model_config.json'
HISTORY_CSV_PATH = './history.csv'

# Set the page layout to wide mode
st.set_page_config(layout="wide")

st.session_state.sim_started = False

def read_model_config():
    """Read the model_config.json and return selected models or an empty list if not available."""
    if os.path.exists('model_config.json'):
        with open('model_config.json', 'r') as f:
            config = json.load(f)
    else:
        config = {
            "max_monitoring": 5,
            "monitoring_models": [],
            "deployed_models": [],
        }
    return config

# def get_deployed_model_id():
#     if os.path.exists(MODEL_METADATA_PATH):
#         with open(MODEL_METADATA_PATH, 'r') as f:
#             metadata = json.load(f)
#         return metadata.get('deployed_model_id', None)
#     return None

st.session_state.is_drawing_table = False

def load_models_from_vault():
    os.makedirs(MODEL_VAULT_DIR, exist_ok=True)

    model_files = [f for f in os.listdir(MODEL_VAULT_DIR) if f.endswith('.pkl')][::-1]
    models = []
    for model_file in model_files:
        with open(os.path.join(MODEL_VAULT_DIR, model_file.replace('.pkl', '.json')), 'r') as f:
            model_data = json.load(f)
        model_id = model_file.split('-')[1]  # Assuming ID is the second part after splitting by '-'
        models.append((model_id, model_data))
    return models

def on_mode_change(model_id):
    st.session_state[f"advanced_mode_{model_id}"] = not st.session_state[f"advanced_mode_{model_id}"]

def on_coeff_slider_change(multiplier):
    st.session_state['edited_coeffs_simple'] = [c * multiplier for c in st.session_state['edited_coeffs_simple']]
    st.session_state['edited_y_intercept_simple'] = st.session_state['edited_y_intercept_simple'] * multiplier

def on_expand_click(model_id, idx):
    st.session_state[f"expand_{model_id}_{idx}"] = not st.session_state[f"expand_{model_id}_{idx}"]

def on_delete_click(model_id):
    st.session_state['confirm_delete'] = model_id

def on_delete_cancel_click():
    del st.session_state['confirm_delete']  # reset the session state

def on_delete_confirm_click(model_id, algo_type):
    # Code to delete the model from the vault
    model_path = os.path.join(MODEL_VAULT_DIR, f"SIO-{model_id}-{algo_type.replace(' ', '_')}.pkl")
    try:
        os.remove(model_path)
        os.remove(model_path.replace('.pkl', '.json'))
        st.success(f"Model SIO-{model_id} deleted successfully!")
    except Exception as e:
        st.error(f"Error deleting model: {e}")
    del st.session_state['confirm_delete']  # reset the session state after either action

def confirm_delete(model_id, algo_type):
    if st.session_state.get('confirm_delete') == model_id:
        col1, col2 = st.columns(2)
        with col1:
            st.button("Confirm", on_click=on_delete_confirm_click, args=(model_id, algo_type))
        with col2:
            st.button("Cancel", on_click=on_delete_cancel_click)

def display_model_table(models):
    if 'model_config' not in st.session_state:
        model_config = read_model_config()
        st.session_state['model_config'] = model_config
    else:
        model_config = st.session_state['model_config']

    st.info(f"Select up to {model_config['max_monitoring']} models to be added to dashboard and 1 model for deployment at a time. Only model added to dashboard can be deployed.")

    col1, col2 = st.columns(2)
    sort_by = col1.selectbox("Sort by", ["Model ID", "R2", "Adj. R2", "Updated On"], index=0)
    order = col2.selectbox("Order", ["Ascending", "Descending"], index=1)

    cc1, cc2 = st.columns(2)
    only_monitoring_checked = cc1.checkbox('Only show added to dashboard models')
    only_deploy_checked = cc2.checkbox('Only show the deployed model')
    
    if only_monitoring_checked:
        models = [m for m in models if m[0] in model_config['monitoring_models']]

    if only_deploy_checked:
        models = [m for m in models if m[0] in model_config['deployed_models']]

    # Sort the models
    if sort_by == "R2":
        sorted_models = sorted(models, key=lambda item: item[1]['r2'], reverse=(order == "Descending"))
    elif sort_by == "Adj. R2":
        sorted_models = sorted(models, key=lambda item: item[1]['adj_r2'], reverse=(order == "Descending"))
    else:  # sort by Model ID
        sorted_models = sorted(models, key=lambda item: int(item[0]), reverse=(order == "Descending"))

    items_per_page = 20
    total_items = len(sorted_models)
    max_pages = -(-total_items // items_per_page)  # Ceiling division

    # Initialize or update the current page in the session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1

    def on_prev_page_click():
        if st.session_state.current_page > 1 and not st.session_state.is_drawing_table:
            st.session_state.current_page -= 1
    
    def on_next_page_click():
        if st.session_state.current_page < max_pages and not st.session_state.is_drawing_table:
            st.session_state.current_page += 1

    # Buttons for pagination control
    col1, _, col2 = st.columns([1, 9, 1])
    if not st.session_state.is_drawing_table:
        col1.button("← Previous", on_click=on_prev_page_click, disabled=st.session_state.is_drawing_table)
        col2.button("Next →", on_click=on_next_page_click, disabled=st.session_state.is_drawing_table)
            

    start_idx = (st.session_state.current_page - 1) * items_per_page
    end_idx = st.session_state.current_page * items_per_page

    col_layout = [1, 2, 1, 1, 2, 2, 1, 1, 1, 1, 1]
    headers = ["Model ID", "Algorithm", "R2", "Adj. R2", "Input Date Range", "Updated On", "Type", "More Details", "Display in Dashboard", "Deploy", "Delete"]
    header_columns = st.columns(col_layout)

    st.session_state.is_drawing_table = True

    for i, header in enumerate(headers):
        header_columns[i].markdown(f"**{header}**")
   
    for j, (model_id, model_data) in enumerate(sorted_models[start_idx: end_idx]):
        cols = st.columns(col_layout)
        
        cols[0].write(f"{int(model_id)}")
        cols[1].write(f"{model_data.get('algorithm')}")
        cols[2].write(f"{round(model_data.get('r2'), 3)}")
        cols[3].write(f"{round(model_data.get('adj_r2'), 3)}")
        cols[4].write(f"{model_data.get('start_date')} to {model_data.get('end_date')}")
        cols[5].write(f"{model_data.get('updated_on')}")
        cols[6].write(f"{model_data.get('retraining_type')}")

        if f"expand_{model_id}_{j}" not in st.session_state:
            st.session_state[f"expand_{model_id}_{j}"] = False

        if st.session_state[f"expand_{model_id}_{j}"]:
            display_model_details(model_id, model_data)

        expand_state = st.session_state[f"expand_{model_id}_{j}"]
        icon = "▼" if not expand_state else "▲"
        cols[7].button(icon, key=f"btn_{model_id}_{j}", on_click=on_expand_click, args=(model_id, j))

        # Monitoring Checkbox
        is_selected_mon = model_id in model_config['monitoring_models']
        if is_selected_mon:
            cols[8].button("✔️", disabled=(not is_selected_mon and len(model_config['monitoring_models']) >= model_config['max_monitoring']) or (model_id in model_config['deployed_models']), key=f"dashboard_{j}_{model_id}", on_click=on_monitoring_change, args=(model_id, model_config['max_monitoring']))
        else:
            cols[8].button("➕", disabled=(not is_selected_mon and len(model_config['monitoring_models']) >= model_config['max_monitoring']) or (model_id in model_config['deployed_models']), key=f"dashboard_{j}_{model_id}", on_click=on_monitoring_change, args=(model_id, model_config['max_monitoring']))            
        # cols[8].checkbox("", value=is_selected_mon, disabled=(not is_selected_mon and len(model_config['monitoring_models']) >= model_config['max_monitoring']) or (model_id in model_config['deployed_models']), key=f"dashboard_{j}_{model_id}", on_change=on_monitoring_change, args=(model_id, model_config['max_monitoring']))
        
        # Auto Re-train Checkbox
        is_deployed = model_id in model_config['deployed_models']
        if is_deployed:
            cols[9].button("✔️", disabled=(not is_selected_mon) or (not is_deployed and len(model_config['deployed_models']) >= 1), key=f"retrain_{j}_{model_id}", on_click=on_deploy_change, args=(model_id,))
        else:
            cols[9].button("➕", disabled=(not is_selected_mon) or (not is_deployed and len(model_config['deployed_models']) >= 1), key=f"retrain_{j}_{model_id}", on_click=on_deploy_change, args=(model_id,))

        if cols[10].button(f"🗑️", key=f"del_{model_id}"):
            on_delete_click(model_id)
        
        confirm_delete(model_id, model_data.get('algorithm'))

        st.markdown("<hr style='height:1px;border:none;color:#333;background-color:#333;margin:5px 0;'/>", unsafe_allow_html=True)

    if 'model_config' in st.session_state:
        if len(st.session_state['model_config']['deployed_models']) > 0:
            st.sidebar.markdown(f"### Deployed Model:")
            for mid in st.session_state['model_config']['deployed_models']:
                st.sidebar.markdown(f'- SIO-{mid}')

        if len(st.session_state['model_config']['monitoring_models']) > 0:
            st.sidebar.markdown(f"### Display in Dashbaord:")
            for mid in st.session_state['model_config']['monitoring_models']:
                st.sidebar.markdown(f'- SIO-{mid}')
                
    time.sleep(1.0)
    st.session_state.is_drawing_table = False

def save_model_config():
    with open(MODEL_CONFIG_PATH, 'w') as f:
        json.dump(st.session_state["model_config"], f)

def on_deploy_change(model_id):
    """Toggle the auto-retrain state for the model."""
    if model_id in st.session_state['model_config']['deployed_models']:
        st.session_state['model_config']['deployed_models'].remove(model_id)
    else:
        st.session_state['model_config']['deployed_models'].append(model_id)
    save_model_config()

def on_monitoring_change(model_id, max_monitoring):
    current_selected = st.session_state["model_config"]['monitoring_models']
    if model_id in current_selected:
        current_selected.remove(model_id)
    else:
        if len(current_selected) < max_monitoring:
            current_selected.append(model_id)
        else:
            return
    save_model_config()

def save_changes_to_file(model_id, model_data):
    """Save the updated model data to its respective file."""
    model_file_path = os.path.join(MODEL_VAULT_DIR, f"SIO-{model_id.zfill(9)}-{model_data['algorithm'].replace(' ', '_')}.json")
    with open(model_file_path, 'w') as f:
        json.dump(model_data, f, indent=4)

def download_coefficients_to_excel(model_id, model_data, cont):
    # Convert the model coefficients to a pandas DataFrame
    df = pd.DataFrame({
        'Variable': model_data.get('selected_vars', []),
        'Coefficient': model_data.get('coeffs', model_data.get('ori_coeffs', []))
    })

    # Convert the DataFrame to Excel format
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name=f"SIO-{model_id}", index=False)
    excel_bytes = output.getvalue()

    # Use Streamlit's download button to allow downloading of the Excel file
    cont.download_button(
        label="Download Coefficients",
        data=excel_bytes,
        file_name=f"SIO-{model_id}_coefficients.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

def display_model_details(model_id, model_data):
    st.subheader("Model Details", divider='blue')
    
    # Displaying model configuration details
    cols = st.columns([2, 1])
    cols2 = st.columns([2, 1])
    cols[0].write("Selected Vars:")
    cols2[0].text(', '.join(model_data.get('selected_vars', [])))
    cols[1].write("Target Var:")
    cols2[1].text(model_data.get('target_var'))
    st.write("Conditions:", model_data.get('conditions'))

    # Editable coefficients
    st.subheader("Coefficients:")
    ori_coeffs = model_data.get('ori_coeffs', [])
    coeffs = model_data.get('coeffs', [])
    selected_vars = model_data.get('selected_vars', [])
    # edited_coeffs = []
    # for var, coeff, ori_coeff in zip(selected_vars, coeffs, ori_coeffs):
    #     edited_value = st.number_input(f"{var} (default: {round(ori_coeff, 7)})", value=coeff, format='%.7f')
    #     edited_coeffs.append(edited_value)

    # Coefficient editing mode selection
    if f"advanced_mode_{model_id}" not in st.session_state:
        st.session_state[f"advanced_mode_{model_id}"] = False

    advanced_mode = st.session_state.get(f"advanced_mode_{model_id}", False)
    adv_checkbox = st.checkbox("Advanced Mode", value=advanced_mode, key=f"advanced_mode_checkbox_{model_id}", on_change=on_mode_change, args=(model_id,))
    
    # Editable coefficients
    ori_coeffs = model_data.get('ori_coeffs', [])
    coeffs = model_data.get('coeffs', [])
    selected_vars = model_data.get('selected_vars', [])
    ori_y_intercept = model_data.get('ori_y_intercept', 0.0)
    y_intercept = model_data.get('y_intercept', 0.0)
    
    st.session_state['edited_coeffs_simple'] = coeffs if len(coeffs) != 0 else ori_coeff
    st.session_state['edited_y_intercept_simple'] = y_intercept if y_intercept is not None else ori_y_intercept

    # Check if we're in advanced mode or simple mode
    if st.session_state[f"advanced_mode_{model_id}"]:
        edited_coeffs = []
        for var, coeff, ori_coeff in zip(selected_vars, coeffs, ori_coeffs):
            edited_value = st.number_input(f"{var} (default: {round(ori_coeff, 7)})", value=coeff, format='%.7f')
            edited_coeffs.append(edited_value)
        
        edited_y_intercept = st.number_input(f"y-intercept (default: {round(ori_y_intercept, 7)})", value=y_intercept, format='%.7f')
    else:
        multiplier = st.slider("Coefficient Multiplier", -10.0, 10.0, 1.0, key=f'coeff_simp_slider_{model_id}')
        on_coeff_slider_change(multiplier)

        for var, coeff, ori_coeff in zip(selected_vars, st.session_state['edited_coeffs_simple'], ori_coeffs):
            st.text(f"{var} (default: {round(ori_coeff, 7)}): {round(coeff, 7)}")

        st.text(f"y-intercept (default: {round(ori_y_intercept, 7)}): {round(st.session_state['edited_y_intercept_simple'], 7)}")

    # # Editable remarks
    remarks = model_data.get('remarks', "")
    edited_remarks = st.text_area("Remarks:", value=remarks)

    coeffs_act_col = st.columns(2)
    if coeffs_act_col[0].button("Save Changes", key=f'save_chg_butt_{model_id}'):

        changed_coeffs = edited_coeffs if st.session_state[f"advanced_mode_{model_id}"] else st.session_state['edited_coeffs_simple']
        changed_y_intercept = edited_y_intercept if st.session_state[f"advanced_mode_{model_id}"] else st.session_state['edited_y_intercept_simple']

        changes = []

        for var, coeff, ori_coeff in zip(selected_vars, changed_coeffs, coeffs):
            if coeff != ori_coeff:
                changes.append(f'{var} ({round(ori_coeff, 3)} to {round(coeff, 3)})')

        if changed_y_intercept != y_intercept:
            changes.append(f'y-intercept ({round(ori_y_intercept, 3)} to {round(changed_y_intercept, 3)})')

        if remarks != edited_remarks:
            changes.append(edited_remarks)

        if len(changes) > 0:
            append_to_hist(HISTORY_CSV_PATH, None, f'Model SIO-{model_id} changed: {", ".join(changes)}', is_auto=model_data['retraining_type'].lower() == 'auto')

        model_data['coeffs'] = changed_coeffs
        model_data['y_intercept'] = changed_y_intercept
        model_data['remarks'] = edited_remarks
        model_data['updated_on'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save the edited model data back to the file
        save_changes_to_file(model_id, model_data)
        st.success("Changes saved successfully!")

    download_coefficients_to_excel(model_id, model_data, coeffs_act_col[1])

# deployed_model_id = get_deployed_model_id()
models = load_models_from_vault()
# deployed_model = models.get(deployed_model_id, {})

display_model_table(models)