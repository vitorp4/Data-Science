from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from processing import inverse_scaling
import numpy as np
import pandas as pd

class MLP:

    models = {}

    def __init__(self, input_size=4, horizons=12, inits=1):
        self.input_size = input_size
        self.horizons = horizons
        self.inits = inits

    def build(self, hidden_layers=[10,5], activation='relu', optimizer='adam', loss='mse'):
        for h in range(1, self.horizons+1):
            model = Sequential()
        
            for i, neurons in enumerate(hidden_layers):
                if i == 0:
                    model.add(Dense(neurons, input_shape=(self.input_size,), activation=activation))
                else:
                    model.add(Dense(neurons, activation=activation))

            model.add(Dense(1))
            model.compile(optimizer=optimizer, loss=loss)
            self.models[f't+{h}'] = model

    def train(self, X, Y, validation_split=0.2, epochs=200):
        if type(X) == pd.DataFrame:
            X = X.values
        if type(Y) == pd.DataFrame:
            Y = Y.values

        stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        for h in range(1, self.horizons+1):
            best_val_loss = np.inf
            best_model = None
    
            for _ in range(self.inits):
                model = self.models[f't+{h}']
                history = model.fit(X, Y[:,h-1], epochs=epochs, validation_split=validation_split, callbacks=[stop])
                val_loss = history.history['val_loss'][-1]

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
     
            self.models[f't+{h}'] = best_model
    
    def predict(self, X, scaler, index=None):
        if type(X) == pd.DataFrame:
            X = X.values

        pred = {}
        for h in range(1, self.horizons+1):
            model = self.models[f't+{h}']
            pred[f't+{h}'] = inverse_scaling(model.predict(X), scaler, h)
        pred = pd.DataFrame(data=pred)

        if index is not None:
            pred.index= index

        return pred