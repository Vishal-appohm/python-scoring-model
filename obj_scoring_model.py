import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
from simple_salesforce import Salesforce

# Salesforce credentials
USERNAME = os.getenv('USERNAME')
PASSWORD = os.getenv('PASSWORD')
SECURITY_TOKEN = os.getenv('SECURITY_TOKEN')
CONSUMER_KEY = os.getenv('CONSUMER_KEY')
CONSUMER_SECRET = os.getenv('CONSUMER_SECRET')



# Connect to Salesforce
sf = Salesforce(username=USERNAME, password=PASSWORD, security_token=SECURITY_TOKEN, consumer_key=CONSUMER_KEY, consumer_secret=CONSUMER_SECRET, domain='test')
print('SF Connected: ', sf)

# Query lead data from Salesforce in batch
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
        
        # Count the number of leads retrieved in this batch
        num_leads_in_batch = len(leads['records'])
        print("Number of leads in batch:", num_leads_in_batch)
        
        if not leads['done']:
            # If there are more records, get next batch
            leads = sf.query_more(leads['nextRecordsUrl'], True)
        else:
            break
            
    return lead_data_list

# Query Salesforce for lead data AND Lead_Score__c != null
query = "SELECT Id, Lead_Score__c, Name, LeadSource, Status, Industry, AnnualRevenue, NumberOfEmployees, Rating, IsConverted, Country FROM Lead WHERE Status IN ('Closed - Not Converted' , 'Closed - Converted' )"
lead_data_list = query_lead_data(sf, query)

if not lead_data_list:
    print("No lead data available for training the model")
    os._exit(0)
else:
    print(len(lead_data_list), " will be used for training the model")
    
    
# Convert queried data to DataFrame
lead_data_df = pd.DataFrame(lead_data_list)

# Drop rows with missing lead scores
#lead_data_df.dropna(subset=['LeadScore'], inplace=True)

# Define features (columns to use for prediction) and target column
X = lead_data_df.drop(['Id', 'Converted', 'Status','NumberOfEmployees','Converted','LeadScore'], axis=1) #drop non required columns
y = lead_data_df['Converted']

# Dummy encode categorical variables 

X = pd.get_dummies(X, columns=['LeadSource', 'Industry', 'Rating', 'Country']) #convert non-numerical data into numerical data 

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_scaled, y)
print("Model Trained Successfully!")



# Score lead data
def score_leads(lead_data_list, X_columns, scaler, model):
    lead_scores = {}
    for lead_data in lead_data_list:
        lead_df = pd.DataFrame([lead_data])
        # Dummy encode categorical variables
        lead_df = pd.get_dummies(lead_df, columns=['LeadSource', 'Industry', 'Rating', 'Country'])#must be present in X feature
        # Align the columns of lead_df with X
        lead_df = lead_df.reindex(columns=X_columns, fill_value=0)
        lead_scaled = scaler.transform(lead_df)
        score = model.predict(lead_scaled)[0]
        lead_scores[lead_data['Id']] = score * 100
        
    return lead_scores

# Query Salesforce for new lead data
query2 = "SELECT Id, Lead_Score__c, Name, LeadSource, Status, Industry, AnnualRevenue, NumberOfEmployees, Rating, IsConverted, Country FROM Lead WHERE IsConverted = False AND Status != 'Closed - Not Converted'"
lead_data_list = query_lead_data(sf, query2)
leads = sf.query(query2)
# Score new lead data
lead_scores = score_leads(lead_data_list, X.columns, scaler, rf_regressor)
print("Lead Scores generated")

# Update lead records with lead scores
def update_lead_scores(sf, lead_scores):
    records_to_update = []
    for lead_id, lead_score in lead_scores.items():
        records_to_update.append({'Id': lead_id, 'Lead_Score__c': lead_score})
    
    if records_to_update:
        print("Id, lead_scores:",lead_scores)
        print("Updating the records")
        sf.bulk.Lead.update(records_to_update, batch_size=10000, use_serial=True)
        print(len(lead_scores),"Lead scores updated successfully.")
    else:
        print("No lead records to update.")

update_lead_scores(sf, lead_scores)