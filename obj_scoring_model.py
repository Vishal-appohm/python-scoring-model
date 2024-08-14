import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from simple_salesforce import Salesforce
import streamlit as st

# Streamlit Widgets for input
st.title("Lead Scoring Model")
username = st.text_input("Username")
password = st.text_password("Password")
security_token = st.text_input("Security Token")
consumer_key = st.text_input("Consumer Key")
consumer_secret = st.text_input("Consumer Secret")

if st.button("Run Lead Scoring"):
    if not (username and password and security_token and consumer_key and consumer_secret):
        st.error("Please provide all required Salesforce credentials.")
    else:
        try:
            # Connect to Salesforce
            sf = Salesforce(username=username, password=password, security_token=security_token,
                            consumer_key=consumer_key, consumer_secret=consumer_secret, domain='test')

            # Query lead data from Salesforce
            def query_lead_data(sf, query):
                leads = sf.query(query)
                lead_data_list = []

                while True:
                    for lead in leads['records']:
                        lead_data_list.append({
                            'Id': lead['Id'],
                            'LeadSource': lead['LeadSource'],
                            'Status': lead['Status'],
                            'Industry': lead['Industry'],
                            'AnnualRevenue': lead['AnnualRevenue'],
                            'NumberOfEmployees': lead['NumberOfEmployees'],
                            'Rating': lead['Rating'],
                            'Converted': lead['IsConverted'],
                            'LeadScore': lead['Lead_Score__c'],
                            'Country': lead['Country']
                        })

                    if not leads['done']:
                        leads = sf.query_more(leads['nextRecordsUrl'], True)
                    else:
                        break

                return lead_data_list

            query = "SELECT Id, Lead_Score__c, Name, LeadSource, Status, Industry, AnnualRevenue, NumberOfEmployees, Rating, IsConverted, Country FROM Lead WHERE Status IN ('Closed - Not Converted', 'Closed - Converted')"
            lead_data_list = query_lead_data(sf, query)

            if not lead_data_list:
                st.warning("No lead data available for training the model")
            else:
                lead_data_df = pd.DataFrame(lead_data_list)

                # Define features and target column
                X = lead_data_df.drop(['Id', 'Converted', 'Status', 'NumberOfEmployees', 'Converted', 'LeadScore'], axis=1)
                y = lead_data_df['Converted']

                # Dummy encode categorical variables
                X = pd.get_dummies(X, columns=['LeadSource', 'Industry', 'Rating', 'Country'])

                # Scale features
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)

                # Train Random Forest Regressor
                rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_regressor.fit(X_scaled, y)
                st.success("Model Trained Successfully!")

                # Score lead data
                def score_leads(lead_data_list, X_columns, scaler, model):
                    lead_scores = {}
                    for lead_data in lead_data_list:
                        lead_df = pd.DataFrame([lead_data])
                        lead_df = pd.get_dummies(lead_df, columns=['LeadSource', 'Industry', 'Rating', 'Country'])
                        lead_df = lead_df.reindex(columns=X_columns, fill_value=0)
                        lead_scaled = scaler.transform(lead_df)
                        score = model.predict(lead_scaled)[0]
                        lead_scores[lead_data['Id']] = score * 100

                    return lead_scores

                query2 = "SELECT Id, Lead_Score__c, Name, LeadSource, Status, Industry, AnnualRevenue, NumberOfEmployees, Rating, IsConverted, Country FROM Lead WHERE IsConverted = False AND Status != 'Closed - Not Converted'"
                lead_data_list = query_lead_data(sf, query2)

                lead_scores = score_leads(lead_data_list, X.columns, scaler, rf_regressor)
                st.write("Lead Scores generated")

                # Update lead records with lead scores
                def update_lead_scores(sf, lead_scores):
                    records_to_update = []
                    for lead_id, lead_score in lead_scores.items():
                        records_to_update.append({'Id': lead_id, 'Lead_Score__c': lead_score})

                    if records_to_update:
                        sf.bulk.Lead.update(records_to_update, batch_size=10000, use_serial=True)
                        st.write(len(lead_scores), "Lead scores updated successfully.")
                    else:
                        st.warning("No lead records to update.")

                update_lead_scores(sf, lead_scores)
                st.success("Lead scoring process completed successfully!")

        except Exception as e:
            st.error(f"An error occurred: {e}")
