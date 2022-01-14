import os
import sys
import pandas as pd
import json

from dataclasses import dataclass

    
# LatLnd: [0,1]
# Moving: bool
# Time is second increments (I think)


class Enviroment:
    
    def __init__(self) -> None:
        
        self.load_data()
    
    def load_data(self):
        
        dataset = {
            "altitude": [],
            "latlng": [],
            "velocity_smooth": [],
            "grade_smooth": [],
            "temp" : [],
            "distance": [],
            "moving": [],
            "time": []
            }
        
        for folder_name in os.listdir('data'):
            with open(f'data/{folder_name}/streams/data.json') as f:
                data = json.load(f)
            
            for key in data:
                dataset[key] += data[key]["data"]
        
        df = pd.DataFrame(data=dataset)
        
        print(df)
        print(len(df.index))
                
                
                
    def analytics(self):
        
        
        with open('data/2721064100/streams/data.json') as f:
            data = json.load(f)
        
        print()
        for key in data:
            print(key)
    


def main():
    e = Enviroment()
    

if __name__ == '__main__':
    main()