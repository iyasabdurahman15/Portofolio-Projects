import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

class absenteeism_model():
    def __init__(self, model_file, scaler_file):
        with open('model_12', 'rb') as model_file, open('scaler_12', 'rb') as scaler_file:
            self.reg = pickle.load(model_file)
            self.scaler = pickle.load(scaler_file)
            self.data = None

    def load_and_clean_data(self, data_file):
        df = pd.read_csv(data_file, delimiter = ',')
        self.df_with_predictions = df.copy()

        df = df.drop(['ID'], axis=1)

        df['Absenteeism Time in Hours'] = 'Nan'

        reason_columns = pd.get_dummies(df['Reason for Absence'], drop_first = True, dtype='int')

        df = df.drop(['Reason for Absence'], axis=1)

        reason_type_1 = reason_columns.loc[:, 1:14].max(axis=1)
        reason_type_2 = reason_columns.loc[:, 15:17].max(axis=1)
        reason_type_3 = reason_columns.loc[:, 18:21].max(axis=1)
        reason_type_4 = reason_columns.loc[:, 22:].max(axis=1)

        df = pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4], axis=1)

        df = df.rename(columns = {0:'Reason_1', 1:'Reason_2', 2:'Reason_3', 3:'Reason_4'})

        df_reason_mod = df.rename(columns = {0:'Reason_1', 1:'Reason_2', 2:'Reason_3', 3:'Reason_4'})

        df_reason_mod = df_reason_mod[['Reason_1', 'Reason_2', 'Reason_3',
       'Reason_4', 'Date', 'Transportation Expense', 'Distance to Work', 'Age',
       'Daily Work Load Average', 'Body Mass Index', 'Education', 'Children',
       'Pets', 'Absenteeism Time in Hours']]

        df_reason_mod['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')

        list_month = [df_reason_mod['Date'][i].month for i in range (df_reason_mod.shape[0])]

        df_reason_mod['Month Value'] = list_month

        def date_to_weekday(date_value):
            return date_value.weekday()

        df_reason_mod['Day of the Week'] = df_reason_mod['Date'].apply(date_to_weekday)

        df_reason_mod['Education'] = df_reason_mod['Education'].map({1:0, 2:1, 3:1, 4:1})

        df_reason_mod = df_reason_mod.drop(['Date'], axis=1)

        df_reason_mod = df_reason_mod.fillna(value=0)

        df_after = df_reason_mod.drop(['Absenteeism Time in Hours', 'Daily Work Load Average', 'Day of the Week', 'Distance to Work'], axis=1)

        self.data_preprocessed = df_after.copy()
        
        df_after_without_dummies = df_after.drop(['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Education'], axis=1)
        dummies = df_after[['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4', 'Education']]
        preprocessed = self.scaler.transform(df_after_without_dummies)
        preprocessed = pd.DataFrame(preprocessed, columns=df_after_without_dummies.columns)
        self.data = pd.concat([preprocessed, dummies], axis=1)

    def predicted_probability(self):
        if(self.data is not None):
            pred = self.reg.predict_proba(self.data)[:,1]
            return pred

    def predicted_output_category(self):
        if(self.data is not None):
            pred_outputs = self.reg.predict(self.data)
            return pred_outputs

    def predicted_output(self):
        if(self.data is not None):
            self.data_preprocessed['Probability'] = self.reg.predict_proba(self.data)[:,1]
            self.data_preprocessed['Predictions'] = self.reg.predict(self.data)
            return self.data_preprocessed