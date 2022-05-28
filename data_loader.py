import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats

class DataLoader:

    chosen_inputs = {'velocity_smooth', 'grade_smooth', 'cadence', 'heartrate'}
    data = pd.DataFrame
    processed_data = pd.DataFrame

    def __init__(self):
        
        self.data = pd.DataFrame()
        self.processed_data = pd.DataFrame()

    def load_from_save(self):
        self.processed_data = pd.read_csv('processed_data.csv') 

    def load_data(self):
        
        if os.path.isfile('processed_data.csv'):
            self.load_from_save()
            return
        
        dataset = {i: [] for i in self.chosen_inputs}
        
        for folder_name in os.listdir('data'):
            if folder_name.isnumeric() and os.path.isfile(f'data/{folder_name}/streams/data.json'):
                
                with open(f'data/{folder_name}/streams/data.json') as f:
                    data = json.load(f)
                
                if self.chosen_inputs.issubset(set(data.keys())):
                    
                    
                    for key in self.chosen_inputs:
                        dataset[key] += data[key]["data"]

        self.data = pd.DataFrame(data=dataset)   
        self.process_data()    

    def process_data(self):

        self.processed_data = self.data

        for col in self.processed_data:

            if col != 'heartrate':
                self.processed_data[col] = ((self.processed_data[col]-self.processed_data[col].min())/(self.processed_data[col].max() - self.processed_data[col].min()))

        self.processed_data.to_csv('processed_data.csv', index=False)

    def get_processed_data(self):

        self.load_data()
        return self.processed_data.to_numpy()

    def get_testing_data(self):

        df = pd.read_json('test_data.json')   
        df  = df[list(self.chosen_inputs)]  
        
        new_df = pd.DataFrame()

        for col in df:
            new_df[col] = np.array(df[col]['data'])

        for col in new_df:
            if col != 'heartrate':
                new_df[col] = ((new_df[col]-new_df[col].min())/(new_df[col].max() - new_df[col].min()))

        return df.to_numpy()

def main():

    dl = DataLoader()
    dl.get_testing_data()

if __name__ == '__main__':
    main()