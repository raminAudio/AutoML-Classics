'''
Author: Ramin Anushiravani
Date: April 12th/23
Feed Forward Neural Network Helper
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from utils.model_utils import ModelUtil
import keras


class FFNN(ModelUtil):
    '''
    This class implements 5 different feedforward neural networks and inherits from the ModelUtil class.
    '''

    def __init__(self, verbose=0):

        self.verbose = verbose

    def create_model_1(self, lr=0.001, inp_shape=[]):
        '''
        architecture 1 - deep with the most params - mainly to test for raw data
        '''
        model = Sequential()
        model.add(BatchNormalization())
        model.add(Dense(1024, input_shape=(inp_shape,), activation='relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.1))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))
        opt = keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])
        return model

    def create_model_2(self, lr=0.001, inp_shape=[]):
        '''
        architecture 2 - not as many params as 1
        '''

        model = Sequential()
        model.add(BatchNormalization())
        model.add(Dense(256, input_shape=(inp_shape,), activation='relu'))
        model.add(Dropout(0.4))
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))
        opt = keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])
        return model

    def create_model_3(self, lr=0.001, inp_shape=[]):
        '''
            architecture 3 - less neurons to avoid overfitting for PCA features
        '''
        model = Sequential()
        model.add(BatchNormalization())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))
        opt = keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])
        return model

    def create_model_4(self, lr=0.0001, inp_shape=[]):
        '''
            architecture 3 - less neurons to avoid overfitting for PCA features
        '''
        model = Sequential()
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(16, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))
        opt = keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])
        return model

    def create_model_5(self, lr=0.0001, inp_shape=[]):
        '''
            architecture 3 - less neurons to avoid overfitting for PCA features
        '''
        model = Sequential()
        model.add(BatchNormalization())
        model.add(Dense(64, activation='linear'))
        model.add(Dropout(0.3))
        model.add(Dense(32, activation='linear'))
        model.add(Dropout(0.3))
        model.add(Dense(16, activation='linear'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(Dense(8, activation='linear'))
        model.add(BatchNormalization())
        model.add(Dense(1, activation='sigmoid'))
        opt = keras.optimizers.Adam(learning_rate=lr)
        model.compile(
            loss='binary_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])
        return model

    def train_FFNN(
            self,
            X_train,
            y_train,
            model_num=1,
            lr=0.001,
            num_folds=5,
            epochs=5,
            batch_size=32):
        '''
        Training a FFNN -
        Args:
            X_train : training matrix
            y_train : training labels
            model_num : which model arch to use
            lr : learning rate
            num_folds : number of cross validation folds
            epochs : number of epochs
            batch_size : batch size
        Returns:
            model : trained model
        '''

        inp_shape = X_train.shape[1]
        if model_num == 1:
            model = self.create_model_1(lr, inp_shape)
        elif model_num == 2:
            model = self.create_model_2(lr, inp_shape)
        elif model_num == 3:
            model = self.create_model_3(lr, inp_shape)
        elif model_num == 4:
            model = self.create_model_4(lr, inp_shape)
        elif model_num == 5:
            model = self.create_model_5(lr, inp_shape)

        if num_folds == 1:
            model.fit(
                X_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=self.verbose)
            return model

        GEN = self.gen_data(X_train, y_train, num_folds=num_folds)
        while True:
            try:
                training_data, training_label, validation_data, validation_label = next(
                    GEN)
                if model_num == 1:
                    model = self.create_model_1(lr, inp_shape)
                elif model_num == 2:
                    model = self.create_model_2(lr, inp_shape)
                elif model_num == 3:
                    model = self.create_model_3(lr, inp_shape)
                elif model_num == 4:
                    model = self.create_model_4(lr, inp_shape)
                elif model_num == 4:
                    model = self.create_model_5(lr, inp_shape)

                model.fit(
                    training_data,
                    training_label,
                    validation_data=(
                        validation_data,
                        validation_label),
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=self.verbose)
            except StopIteration:
                break
        return model
