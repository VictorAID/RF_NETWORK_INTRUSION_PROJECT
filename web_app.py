import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define feature columns
feature_columns = [
    'ifInOctets11', 'ifOutOctets11', 'ifoutDiscards11', 'ifInUcastPkts11', 
    'ifInNUcastPkts11', 'ifInDiscards11', 'ifOutUcastPkts11', 'ifOutNUcastPkts11', 
    'tcpOutRsts', 'tcpInSegs', 'tcpOutSegs', 'tcpPassiveOpens', 'tcpRetransSegs', 
    'tcpCurrEstab', 'tcpEstabResets', 'tcp?ActiveOpens', 'udpInDatagrams', 
    'udpOutDatagrams', 'udpInErrors', 'udpNoPorts', 'ipInReceives', 'ipInDelivers', 
    'ipOutRequests', 'ipOutDiscards', 'ipInDiscards', 'ipForwDatagrams', 'ipOutNoRoutes', 
    'ipInAddrErrors', 'icmpInMsgs', 'icmpInDestUnreachs', 'icmpOutMsgs', 
    'icmpOutDestUnreachs', 'icmpInEchos', 'icmpOutEchoReps'
]

# Streamlit app
st.title("Network Intrusion Detection")

# Input options
st.sidebar.title("Input Options")
input_type = st.sidebar.selectbox("Choose input type", ("Manual Input", "Upload CSV/Excel"))

if input_type == "Manual Input":
    user_input = {}
    for feature in feature_columns:
        user_input[feature] = st.sidebar.number_input(f"{feature}", value=0)
    user_data = pd.DataFrame(user_input, index=[0])
    
    # Process and predict
    if st.button("Predict"):
        # Predict
        prediction = model.predict(user_data)
        st.write(f"Prediction: {'Anomaly' if prediction[0] == 1 else 'Normal'}")

elif input_type == "Upload CSV/Excel":
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            user_data = pd.read_csv(uploaded_file)
        else:
            user_data = pd.read_excel(uploaded_file)

        if st.button("Predict for Uploaded File"):
            # Ensure the correct columns are present
            if all(col in user_data.columns for col in feature_columns):
                # Predict
                predictions = model.predict(user_data[feature_columns])
                user_data['Prediction'] = ['Anomaly' if p == 1 else 'Normal' for p in predictions]
                st.write(user_data)
                st.write("### Summary")
                st.write(user_data['Prediction'].value_counts())
                
                # Save the file to be downloaded
                st.download_button(
                    label="Download Predictions",
                    data=user_data.to_csv(index=False).encode('utf-8'),
                    file_name='predictions.csv',
                    mime='text/csv'
                )
            else:
                st.error("Uploaded file is missing required columns.")
