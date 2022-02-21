from asyncio import SendfileNotAvailableError
import os
import sys
import pandas as pd
import numpy as np
import json

from sklearn.preprocessing import OneHotEncoder

    
# LatLnd: [0,1]
# Moving: bool
# Time is second increments (I think)


class Environment:
    
    chosen_inputs = {'altitude', 'distance', 'velocity_smooth', 'latlng', 'time', 'grade_smooth', 'moving', 'heartrate'}
    data = pd.DataFrame
    processed_data = pd.DataFrame
    
    actions: list = []
    
    def __init__(self) -> None:
        
        self.data = pd.DataFrame()
        self.processed_data = pd.DataFrame()
        
        self.load_data()
    
    def load_from_save(self):
        
        self.processed_data = pd.read_csv('env.csv')   
        
    def load_data(self):
        
        if os.path.isfile('env.csv'):
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
                
        df = pd.DataFrame(data=dataset)
        df[['lat','lng']] = pd.DataFrame(df.latlng.tolist(), index=df.index)
        df.drop('latlng', axis=1, inplace=True)
        
        self.data = df
        
        self.infer_actions()
        self.process_data()
    
    def infer_actions(self):
        
        actions = []
        velocities = list(self.data['velocity_smooth'])
        gradients = list(self.data['grade_smooth'])
        
        for i in range(len(velocities)):
            
            if i + 1 < len(velocities):
                
                action = 0
                speed_change = velocities[i+1] - velocities[i]
                upcomming_grad = gradients[i+1]
                
                if upcomming_grad > 0:
                    upcomming_grad += 1
                    action = upcomming_grad + speed_change
                    
                else:
                    upcomming_grad -= 1
                    action = upcomming_grad + speed_change

                actions.append(action)
        
        actions.append(actions[-1])         # Artificially pad 
        actions = np.array(actions)
        
        actions = (((actions - min(actions)) / (max(actions) - min(actions))) * (10 - 0)) + 0       # Transforming to scale of 0 to 10
        actions = [round(i) for i in actions]
        
        self.actions = actions      
                
    def process_data(self):
        
        self.processed_data = pd.DataFrame(self.data)
                
        numerical_processing = ['altitude', 'distance', 'velocity_smooth', 'lat', 'lng', 'grade_smooth', 'heartrate']
        
        for col in numerical_processing:
            self.processed_data[col] = ((self.processed_data[col]-self.processed_data[col].mean())/self.processed_data[col].std())
            
        self.processed_data['time'] = self.processed_data['time'].apply(lambda x: int(x / 1800))
        enc = OneHotEncoder(sparse=False)
        
        time_onehot = enc.fit_transform(self.processed_data[['time']])
        t_hot = pd.DataFrame(time_onehot, columns=list(enc.categories_[0]))
        self.processed_data.drop('time', axis=1, inplace=True)
        
        moving_onehot = enc.fit_transform(self.processed_data[['moving']])
        moving_hot = pd.DataFrame(moving_onehot, columns=list(enc.categories_[0]))
        self.processed_data.drop('moving', axis=1, inplace=True)
        
        for col in t_hot:
            self.processed_data[col] = t_hot[col]
        
        for col in moving_hot:
            self.processed_data[col] = moving_hot[col]
        
        self.processed_data['Actions'] = pd.Series(self.actions)
        
        self.processed_data.to_csv('env.csv')
    
    def get_dataset(self):
        
        return self.processed_data.to_numpy()  
             
    def analytics(self):
        
        chosen_set = {'altitude', 'distance', 'velocity_smooth', 'latlng', 'time', 'grade_smooth', 'moving', 'heartrate'}
        num_available = 0
        total_docs = 0
        
        total_keys = {'heartrate':0, 'cadence':0, 'distance':0, 'moving':0, 'altitude':0, 'watts':0, 'velocity_smooth':0, 'time':0, 'grade_smooth':0, 'latlng':0, 'temp':0}
        sets_of_keys = {}
        
        for folder_name in os.listdir('data'):
            
            if folder_name.isnumeric() and os.path.isfile(f'data/{folder_name}/streams/data.json'):
                with open(f'data/{folder_name}/streams/data.json') as f:
                    self.data = json.load(f)
            
            keys = frozenset(self.data.keys())
            if chosen_set.issubset(keys): num_available += 1
            total_docs += 1
            
            if keys in sets_of_keys:
                sets_of_keys[keys] += 1
            else:
                sets_of_keys[keys] = 1
        
        total = set(list(sets_of_keys.keys())[0])
        
        sets_of_keys = {k: v for k, v in sorted(sets_of_keys.items(), key=lambda item: item[1], reverse=True)}
        
        i = 0
        for keys in sets_of_keys:
            
            if i<4:
                sorted_keys = list(keys)
                sorted_keys.sort()
                
                print(f'{sorted_keys} -> {sets_of_keys[keys]}')
                total = total.intersection(set(keys))
            
            i += 1
            
        
        print(f"Intersection of all: {total}")
        
        print(f"Docs available: {num_available} out of: {total_docs}")
        print(total_keys.keys() - chosen_set)
   
        print(keys)
    


def main():
    
    e = Environment()
    e.load_data()
    # e.process_data()
    # e.get_dataset()
    e.analytics()
    

if __name__ == '__main__':
    main()