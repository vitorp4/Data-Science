from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
from processing import shift
import numpy as np
import pandas as pd

class MLP:

    models = {}

    def __init__(self, n_lags, n_horizons, n_inits):
        self.n_lags = n_lags
        self.n_horizons = n_horizons
        self.n_inits = n_inits

    def build(self, hidden_layers, activation, optimizer, loss):
        for h in range(1, self.n_horizons+1):
            model = Sequential()
        
            for i, neurons in enumerate(hidden_layers):
                if i == 0:
                    model.add(Dense(neurons, input_shape=(self.n_lags,), activation=activation))
                else:
                    model.add(Dense(neurons, activation=activation))

            model.add(Dense(1))
            model.compile(optimizer=optimizer, loss=loss)
            self.models[f't+{h}'] = model

    def train(self, X, y, validation_split, epochs):
        stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        for h in range(1, self.n_horizons+1):
            best_val_loss = np.inf
            best_model = None
    
            for _ in range(self.n_inits):
                model = self.models[f't+{h}']
                history = model.fit(X, y[:,h-1], epochs=epochs, validation_split=validation_split, callbacks=[stop])
                val_loss = history.history['val_loss'][-1]

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
     
            self.models[f't+{h}'] = best_model
    
    def predict(self, X):
        pred = np.empty((X.shape[0], 0))

        for h in range(1, self.n_horizons+1):
            model = self.models[f't+{h}']
            pred = np.append(pred, model.predict(X), 1)

        pred = pd.DataFrame(data={f't+{h}':shift(pred[:,h-1], h) for h in range(1, self.n_horizons+1)})
        return pred