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
import matplotlib.pyplot as plt

import data_loader

class LSTMModel:

    config: dict
    model: object
    data: list
    d_lodaer: data_loader.DataLoader

    X: list = []
    Y: list = []

    train_data: tf.data.Dataset
    test_data: tf.data.Dataset

    strategy: tf.distribute.MirroredStrategy

    def __init__(self, config: dict) -> None:
        
        self.config = config
        self.d_lodaer = data_loader.DataLoader()
        self.data = self.d_lodaer.get_processed_data()

        self.configure_tf_strat()

    def configure_tf_strat(self):

        physical_devices = tf.config.list_physical_devices('GPU')
        
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        self.strategy = tf.distribute.MirroredStrategy(["/gpu:0", "/gpu:1"])
        mlflow.tensorflow.autolog() 
    
    def build_dataset(self):

        X_train = [[x] for x in np.array(self.X[0:int(len(self.X) * 0.8)])]
        X_test = [[x] for x in np.array(self.X[int(len(self.X) * 0.8):])]
        Y_train = [[x] for x in np.array(self.Y[0:int(len(self.Y) * 0.8)])]
        Y_test = [[x] for x in np.array(self.Y[int(len(self.Y) * 0.8):])]

        train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_data = train_data.batch(self.config['batch_size'], drop_remainder=True)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        train_data = train_data.with_options(options).prefetch(tf.data.AUTOTUNE)
        
        test_data = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        test_data = test_data.batch(self.config['batch_size'], drop_remainder=True)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        test_data = test_data.with_options(options).prefetch(tf.data.AUTOTUNE)

        self.train_data = train_data
        self.test_data = test_data

    def process_data(self):
        
        heartrate = self.data[:,0]
        grade_smooth = self.data[:,3]
        velocity_smooth = self.data[:,2]
        cadence = self.data[:,1]

        h_segements = []
        g_segements = []
        v_segements = []
        c_segements = []

        for i in range(len(heartrate) - (self.config['sequence_length'] + self.config['step_length'])):

            c_segements.append(cadence[i:i+self.config['sequence_length']])
            v_segements.append(velocity_smooth[i:i+self.config['sequence_length']])
            g_segements.append(grade_smooth[i+self.config['sequence_length']: i+self.config['sequence_length'] + self.config['step_length']])
            h_segements.append(heartrate[i + self.config['sequence_length'] + self.config['step_length']])
        
        for i in range(len(h_segements)):

            self.X.append(list(c_segements[i]) + list(v_segements[i]) + list(g_segements[i]))
            self.Y.append(h_segements[i])

    def build_model(self):

        with self.strategy.scope():

            init = tf.keras.initializers.HeUniform()        
            model = keras.Sequential()      

            model.add(LSTM(50, input_shape=(1, self.config['sequence_length']*2  + self.config['step_length']), return_sequences=True, activation='relu'))
            model.add(LSTM(50, return_sequences=False, activation='relu'))
            model.add(keras.layers.Dense(1, activation='linear', kernel_initializer=init))

            model.compile(
                loss='mean_squared_error', 
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']), 
                metrics=['mean_absolute_error']
            )
            self.model = model

    def train_model(self):

        es = EarlyStopping(monitor='val_mean_absolute_error', mode='min', min_delta=0.0001, patience=10, restore_best_weights=True)

        self.model.fit(
            self.train_data, 
            epochs=3, 
            callbacks=[es], 
            validation_data=self.test_data, 
            batch_size=self.config['batch_size']
        )

    def test_model(self):

        Y_test = [[x] for x in np.array(self.Y[int(len(self.Y) * 0.8):])]
        y_pred = self.model.predict(self.test_data)

        plt.plot(y_pred, label='predicted')
        plt.plot(Y_test, label='actual')
        plt.legend()
        plt.savefig('figures/prediction1.png')


    def run_experiment(self):

        self.process_data()
        self.build_dataset()
        self.build_model()
        self.train_model()
        self.test_model()

def main():

    config = {
        'sequence_length': 3,
        'step_length': 5,
        'batch_size': 64,
        'learning_rate': 0.005,
    }

    model = LSTMModel(config)
    model.run_experiment()

if __name__ == '__main__':
    main()