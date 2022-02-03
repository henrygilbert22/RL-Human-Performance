import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.layers import LSTM
from keras import callbacks
import numpy as np
import sys

from enviroment import Enviroment

class Dynamics:
    
    look_behind: int
    shift_amount: int
    env: Enviroment
    data: list
    
    X: list = []
    Y: list = []
    
    model: object
    
    def __init__(self, look_behind: int, shift_amount: int) -> None:
        
        self.look_behind = look_behind
        self.shift_amount = shift_amount
        self.env = Enviroment()
        
        self.data = self.env.get_dataset()
        self.shift_data()
        self.build_model()
        
        self.train_model()
                
    def shift_data(self):
        
        for i in range(self.look_behind, len(self.data) - self.shift_amount, 1):
            
            self.X.append(np.reshape(self.data[i-self.look_behind: i], (1, 5, 17)))    
            self.Y.append(np.reshape(self.data[i+self.shift_amount-1][:-1], (1, 1, 16)))      # To remove the action in the output, we don't need to predict the next action, just next state

     
    def build_model(self):
        
        model = keras.Sequential()
        
        model.add(LSTM(50, input_shape=(self.look_behind, 17 ), activation='relu', return_sequences=True, name="digits"))
        model.add(keras.layers.Dropout(0.25))

        model.add(LSTM(50, activation='relu', return_sequences=True, name="sequence1"))
        model.add(keras.layers.Dropout(0.2))

        model.add(LSTM(50, activation='relu', return_sequences=True, name="sequence2"))
        model.add(keras.layers.Dropout(0.2))

        model.add(keras.layers.Dense((len(np.array(self.Y[0]).flatten())), name='finale'))

        model.compile(loss='mean_squared_error', optimizer='adam')
        
        self.model = model
    
    def train_model(self):
                
        earlystopping = callbacks.EarlyStopping(monitor ="loss", mode ="min", patience = 10, restore_best_weights = True)
        train_data = tf.data.Dataset.from_tensor_slices((self.X, self.Y))
            
        self.model.fit(train_data, batch_size=256, epochs=100, callbacks=[earlystopping])
        
def main():
    d = Dynamics(5, 1)

if __name__ == '__main__':
    main()