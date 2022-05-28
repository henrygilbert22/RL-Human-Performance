import pickle
import random
import copy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import mlflow
from tensorflow.keras.layers import LSTM

import data_loader

class LSTMModel:

    config: dict
    model: object
    data: list
    d_lodaer: data_loader.DataLoader

    def __init__(self, config: dict) -> None:
        
        self.config = config
        self.d_lodaer = data_loader.DataLoader()
        self.data = self.d_lodaer.get_processed_data()

    def process_data(self):
        
        heartrate = self.data[:,0]
        grade_smooth = self.data[:,1]
        velocity_smooth = self.data[:,2]
        cadence = self.data[:,3]

        



def main():

    model = LSTMModel({})
    model.process_data()

if __name__ == '__main__':
    main()