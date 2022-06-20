import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os
import mlflow
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt
import sys

import data_loader

class LSTMModel:

    config: dict
    model: object
    train_data: list
    test_data: list
    d_loader: data_loader.DataLoader

    train_X: list = []
    train_Y: list = []
    test_X: list = []
    test_Y: list = []

    train_data: tf.data.Dataset
    test_data: tf.data.Dataset

    strategy: tf.distribute.MirroredStrategy

    def __init__(self, config: dict) -> None:
        
        self.config = config
        self.d_loader = data_loader.DataLoader()
        
        self.train_data = self.d_loader.get_processed_data(True)
        self.test_data = self.d_loader.get_processed_data(False)

        self.configure_mlflow()
        self.configure_tf_strat()

    def configure_mlflow(self):

        mlflow.set_experiment('hr_forecasting')
        mlflow.start_run()

    def configure_tf_strat(self):

        physical_devices = tf.config.list_physical_devices('GPU')
        
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        self.strategy = tf.distribute.MirroredStrategy(["/gpu:0", "/gpu:1"])

    def build_dataset(self):

        X_train = [[x] for x in np.array(self.train_X)]
        X_test = [[x] for x in np.array(self.test_X)]
        Y_train = [[y] for y in np.array(self.train_Y)]
        Y_test = [[y] for y in np.array(self.test_Y)]

        train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_data = train_data.batch(self.config['batch_size'], drop_remainder=True)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        train_data = train_data.with_options(options)
        
        test_data = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
        test_data = test_data.batch(self.config['batch_size'], drop_remainder=True)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        test_data = test_data.with_options(options)

        self.train_data = train_data
        self.test_data = test_data

    def create_data(self):

        self.train_X, self.train_Y = self.process_data(self.train_data)
        self.test_X, self.test_Y = self.process_data(self.test_data)
        
    def process_data(self, data: dict):
        
        heartrate = list(data['heartrate'].values())
        test_heartrate = list(data['test_heartrate'].values())
        grade_smooth = list(data['grade_smooth'].values())
        velocity_smooth = list(data['velocity_smooth'].values())
        cadence = list(data['cadence'].values())

        h_segements = []
        h_test_segements = []
        g_segements = []
        v_segements = []
        c_segements = []

        X = []
        Y = []

        for i in range(len(heartrate) - (self.config['sequence_length'] + self.config['step_length'])):

            c_segements.append(cadence[i:i+self.config['sequence_length']])
            v_segements.append(velocity_smooth[i:i+self.config['sequence_length']])
            g_segements.append(grade_smooth[i+self.config['sequence_length']: i+self.config['sequence_length'] + self.config['step_length']])
            h_segements.append(heartrate[i:i+self.config['sequence_length']])
            h_test_segements.append(test_heartrate[i + self.config['sequence_length'] + self.config['step_length']])
        
        for i in range(len(h_segements)):

            X.append(list(c_segements[i]) + list(v_segements[i]) + list(g_segements[i]) + list(h_segements[i]))
            Y.append(h_test_segements[i])

        return X, Y

    def build_model(self):

        with self.strategy.scope():

            init = tf.keras.initializers.HeUniform()        
            model = keras.Sequential()      

            model.add(LSTM(100, input_shape=(1, self.config['sequence_length']*3  + self.config['step_length']), return_sequences=True, activation='relu'))
            model.add(LSTM(100, return_sequences=True, activation='relu'))
            model.add(LSTM(100, return_sequences=True, activation='relu'))
            model.add(LSTM(100, return_sequences=False, activation='relu'))
            model.add(keras.layers.Dense(1, activation='linear', kernel_initializer=init))

            model.compile(
                loss='mean_squared_error', 
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.config['learning_rate']), 
                metrics=['mean_absolute_error']
            )
            self.model = model

    def train_model(self):

        es = EarlyStopping(monitor='val_mean_absolute_error', mode='min', min_delta=0.01, patience=5, restore_best_weights=True)

        history = self.model.fit(
            self.train_data, 
            epochs=100, 
            callbacks=[es], 
            validation_data=self.test_data, 
            batch_size=self.config['batch_size'],
            shuffle=True
        )

        self.model.save(f'models/model.h5')
        return history.history

    def log_experiment(self, history: dict):

        mlflow.log_artifact('lstm_model.py')
        mlflow.log_artifact('figures')
        mlflow.log_artifact('models')

        for key, value in history.items():
            [mlflow.log_metric(key, value[i]) for i in range(len(value))]

        for key, value in self.config.items():
            mlflow.log_param(key, value)

    def calculate_relative_zone_error(self, pred: list, actual: list):

        relative_zone_error = []
        for i in range(len(pred)):
            relative_zone_error.append(abs(pred[i] - actual[i]) / 20)

        return sum(relative_zone_error) / len(relative_zone_error)

    def test_model(self):
        
        test_runs = self.d_loader.load_individual_test_rides()
        mae_comp = tf.keras.losses.MeanAbsoluteError()
        maes = []
        rzes = []

        for i in range(len(test_runs)):
            
            self.test_X, self.test_Y = self.process_data(test_runs[i])
            self.build_dataset()

            Y_test = [x for x in np.array(self.test_Y)]
            y_pred = self.model.predict(self.test_data)

            mae = float(mae_comp(y_pred, Y_test).numpy())
            relative_zone_error = self.calculate_relative_zone_error(y_pred, Y_test)[0]
            maes.append(mae)
            rzes.append(relative_zone_error)

            plt.plot(y_pred, label='predicted')
            plt.plot(Y_test, label='actual')
            plt.title(f'MAE: {round(mae, 2)} - Relative-Zone-Error: {round(relative_zone_error, 2)}')
            plt.legend()
            plt.savefig(f'figures/prediction_{i}.png')
            plt.clf()
        
        mlflow.log_metric('avg_test_mae', sum(maes) / len(maes))
        mlflow.log_metric('avg_test_rze', sum(rzes) / len(rzes))


    def run_experiment(self):

        self.create_data()
        self.build_dataset()
        self.build_model()
        history = self.train_model()
        self.test_model()
        self.log_experiment(history)

def main():

    config = {
        'sequence_length': 10,
        'step_length': 120,
        'batch_size': 64,
        'learning_rate': 0.005,
    }

    model = LSTMModel(config)
    model.run_experiment()

if __name__ == '__main__':
    main()