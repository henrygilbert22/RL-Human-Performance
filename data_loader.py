import os
import sys
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from scipy import stats
import shutil
import random

class DataLoader:

    chosen_inputs = {'velocity_smooth', 'grade_smooth', 'cadence', 'heartrate'}
    data = pd.DataFrame
    processed_data = pd.DataFrame

    def __init__(self):
        
        self.data = pd.DataFrame()
        self.processed_data = pd.DataFrame()

    def load_from_save(self, name: str):
        self.processed_data = pd.read_csv(f'{name}_data.csv') 

    def load_data(self, train: bool = True):
        
        if train:
            name = "train"
        else:
            name = "test"

        if os.path.isfile(f'{name}_data.csv'):
            self.load_from_save(name)
            return
        
        dataset = {i: [] for i in self.chosen_inputs}

        for folder_name in os.listdir(f'{name}_data'):
            if folder_name.isnumeric() and os.path.isfile(f'{name}_data/{folder_name}/streams/data.json'):
                
                with open(f'{name}_data/{folder_name}/streams/data.json') as f:
                    data = json.load(f)
                
                if self.chosen_inputs.issubset(set(data.keys())):

                    for key in self.chosen_inputs:
                        dataset[key] += data[key]["data"]

        self.data = pd.DataFrame(data=dataset)   
        self.process_data(name)    

    def process_data(self, name: str):

        self.processed_data = self.data
        self.processed_data["test_heartrate"] = self.processed_data["heartrate"]

        for col in self.processed_data:

            if col != 'test_heartrate':
                self.processed_data[col] = ((self.processed_data[col]-self.processed_data[col].min())/(self.processed_data[col].max() - self.processed_data[col].min()))

        self.processed_data.to_csv(f'{name}_data.csv', index=False)

    def get_processed_data(self, train: bool = True):

        self.load_data(train)
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