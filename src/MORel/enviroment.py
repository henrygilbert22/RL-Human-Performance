from asyncio import SendfileNotAvailableError
import os
import sys
import pandas as pd
import json

from dataclasses import dataclass

    
# LatLnd: [0,1]
# Moving: bool
# Time is second increments (I think)


class Enviroment:
    
    chosen_inputs = {'altitude', 'distance', 'velocity_smooth', 'latlng', 'time', 'grade_smooth', 'moving', 'heartrate'}
    data = pd.DataFrame
    
    def __init__(self) -> None:
        
        self.load_data()
    
    def load_data(self):
        
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
        
        self.data = df
        
             
    def analytics(self):
        
        chosen_set = {'altitude', 'distance', 'velocity_smooth', 'latlng', 'time', 'grade_smooth', 'moving', 'heartrate'}
        num_available = 0
        total_docs = 0
        
        total_keys = {'heartrate':0, 'cadence':0, 'distance':0, 'moving':0, 'altitude':0, 'watts':0, 'velocity_smooth':0, 'time':0, 'grade_smooth':0, 'latlng':0, 'temp':0}
        sets_of_keys = {}
        
        for folder_name in os.listdir('data'):
            
            if folder_name.isnumeric() and os.path.isfile(f'data/{folder_name}/streams/data.json'):
                with open(f'data/{folder_name}/streams/data.json') as f:
                    data = json.load(f)
            
            keys = frozenset(data.keys())
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
    e = Enviroment()
    

if __name__ == '__main__':
    main()