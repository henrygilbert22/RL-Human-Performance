import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats
import shutil

class DataLoader:

    chosen_inputs = {'velocity_smooth', 'grade_smooth', 'cadence', 'heartrate'}
    data = pd.DataFrame
    processed_data = pd.DataFrame

    def __init__(self):
        
        self.data = pd.DataFrame()
        self.processed_data = pd.DataFrame()

    def load_from_save(self):
        self.processed_data = pd.read_csv('processed_data.csv') 

    def get_valid_folders(self):
        
        for folder_name in os.listdir('data'):
            if folder_name.isnumeric() and os.path.isfile(f'data/{folder_name}/streams/data.json'):
                
                with open(f'data/{folder_name}/streams/data.json') as f:
                    data = json.load(f)
                
                if not self.chosen_inputs.issubset(set(data.keys())):
                    print(f'{folder_name} is not a valid folder')
                    shutil.rmtree(f'data/{folder_name}')

    def load_data(self):
        
        if os.path.isfile('processed_data.csv'):
            self.load_from_save()
            return
        
        dataset = {i: [] for i in self.chosen_inputs}
        valid_names = []

        for folder_name in os.listdir('data'):
            if folder_name.isnumeric() and os.path.isfile(f'data/{folder_name}/streams/data.json'):
                
                with open(f'data/{folder_name}/streams/data.json') as f:
                    data = json.load(f)
                
                if self.chosen_inputs.issubset(set(data.keys())):
                    valid_names = []
                    for key in self.chosen_inputs:
                        dataset[key] += data[key]["data"]

        self.data = pd.DataFrame(data=dataset)   
        self.process_data()    

    def process_data(self):

        self.processed_data = self.data
        self.processed_data["test_heartrate"] = self.processed_data["heartrate"]

        for col in self.processed_data:

            if col != 'test_heartrate':
                self.processed_data[col] = ((self.processed_data[col]-self.processed_data[col].min())/(self.processed_data[col].max() - self.processed_data[col].min()))

        self.processed_data.to_csv('processed_data.csv', index=False)

    def get_processed_data(self):

        self.load_data()
        return self.processed_data.to_dict()

    def get_testing_data(self):

        with open('test_data.json') as f:
            
            data = json.load(f)
            data_subset = {i: data[i]["data"] for i in self.chosen_inputs}

        df  = pd.DataFrame(data=data_subset)
        df["test_heartrate"] = df["heartrate"]

        for col in df:
            if col != 'test_heartrate':
                df[col] = ((df[col]-df[col].min())/(df[col].max() - df[col].min()))

        return df.to_dict()

def main():

    dl = DataLoader()
    dl.get_valid_folders()

if __name__ == '__main__':
    main()