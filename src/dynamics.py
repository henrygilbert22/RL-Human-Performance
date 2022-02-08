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
    layer_size: int
    env: Enviroment
    data: list
    
    X: list = []
    Y: list = []
    
    model: object
    
    def __init__(self, look_behind: int, shift_amount: int, layer_size: int) -> None:
        
        gpus = tf.config.experimental.list_physical_devices('GPU')
        
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

        self.strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])

        self.look_behind = look_behind
        self.shift_amount = shift_amount
        self.layer_size = layer_size
        self.env = Enviroment()
        
        self.data = self.env.get_dataset()
        self.shift_data()
        self.build_model()
        
        self.train_model()
                
    def shift_data(self):
        
        for i in range(self.look_behind, len(self.data) - self.shift_amount, 1):
            
            self.X.append(np.reshape(self.data[i-self.look_behind: i], ( self.look_behind, 17)))    
            self.Y.append(np.reshape(self.data[i+self.shift_amount-1][:-1], ( 1, 16)))      # To remove the action in the output, we don't need to predict the next action, just next state

     
    def build_model(self):
        
        with self.strategy.scope():

            model = keras.Sequential()
            
            model.add(LSTM(self.layer_size, input_shape=(self.look_behind, 17 ), activation='relu', return_sequences=True, name="digits"))
            model.add(keras.layers.Dropout(0.25))

            model.add(LSTM(self.layer_size, activation='relu', return_sequences=True, name="sequence1"))
            model.add(keras.layers.Dropout(0.2))

            model.add(LSTM(self.layer_size, activation='relu', return_sequences=True, name="sequence2"))
            model.add(keras.layers.Dropout(0.2))

            model.add(LSTM(self.layer_size, activation='relu', return_sequences=True, name="sequence3"))
            model.add(keras.layers.Dropout(0.2))

            model.add(LSTM(self.layer_size, activation='relu', return_sequences=True, name="sequence4"))
            model.add(keras.layers.Dropout(0.2))
            
            model.add(LSTM(self.layer_size, activation='relu', return_sequences=True, name="sequence5"))
            model.add(keras.layers.Dropout(0.2))

            model.add(keras.layers.Dense((len(np.array(self.Y[0]).flatten())), name='finale'))

            model.compile(loss='mean_squared_error', optimizer='adam')
            
            self.model = model
    
    def train_model(self):

        earlystopping = callbacks.EarlyStopping(monitor ="loss", mode ="min", patience = 5, restore_best_weights = True)
        
        train_data = tf.data.Dataset.from_tensor_slices((self.X[0:int(len(self.X)*0.8)], self.Y[0:int(len(self.X)*0.8)]))
        train_data = train_data.batch(64)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        train_data = train_data.with_options(options)
        
        test_X = self.X[int(len(self.X)*0.8):]
        test_Y = self.Y[int(len(self.X)*0.8):]
        test_data = tf.data.Dataset.from_tensor_slices((test_X, test_Y))
        test_data = test_data.batch(64)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        test_data = test_data.with_options(options)
        
        
            
        self.model.fit(train_data, epochs=100, callbacks=[earlystopping], validation_data=test_data)
        
        results = self.model.evaluate(test_data, batch_size=256)
        print("results")
        print(results)
            

        
        
def main():
    
    d = Dynamics(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))

if __name__ == '__main__':
    main()