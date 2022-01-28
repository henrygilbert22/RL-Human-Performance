import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import LSTM
import numpy as np

from enviroment import Enviroment

class Dynamics:
    
    look_behind: int
    shift_amount: int
    env: Enviroment
    data: list
    
    X: list
    Y: list
    
    def __init__(self, look_behind: int, shift_amount: int) -> None:
        
        self.look_behind = look_behind
        self.shift_amount = shift_amount
        self.env = Enviroment()
        
        self.data = self.env.get_dataset()
        
    def shift_data(self):
        
        for i in range(self.look_behind, len(self.data) - self.shift_amount, 1):
            self.X.append(self.data[i-self.look_behind: self.look_behind])          # Need to also account for adding in actions, do that somewhere else
            self.Y.append(self.data[i+self.shift_amount])
    
    def build_model(self):
        
        model = keras.Sequential()

        model.add(LSTM(50, input_shape=(1,len(np.array(self.X[0]).flatten())), activation='relu', return_sequences=True, name="digits"))
        model.add(keras.layers.Dropout(0.25))

        model.add(LSTM(50, activation='relu', return_sequences=True, name="sequence1"))
        model.add(keras.layers.Dropout(0.2))

        model.add(LSTM(50, activation='relu', return_sequences=True, name="sequence2"))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense((len(np.array(self.X[0]).flatten())), name='finale'))

        model.compile(loss='mean_squared_error', optimizer='adam')
        
        
        
def main():
    d = Dynamics()

if __name__ == '__main__':
    main()