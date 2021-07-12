import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from mlp import MLP
from processing import patterns, persistence
from metrics import all_metrics_from_dataframe
from taylor_diagram import diagram

df = pd.read_csv('australia.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp').rolling(12).mean().iloc[12::12]
df.index.name = None
df = df.interpolate(method='polynomial', order=5, axis=0).clip(lower=0)

INPUT_SIZE = 4
HORIZONS = 12
INITS = 1 
HIDDEN_LAYERS = [10]
ACTIVATION = 'relu'
OPTIMIZER = 'adam'
LOSS = 'mse'
VALIDATION_SPLIT = 0.2
EPOCHS = 200

CENTRAL = 'KIATAWF1'

serie = df[CENTRAL]
index = serie.index
serie.describe()

train, test = train_test_split(serie, test_size=.33, shuffle=False)
train_idx, test_idx = train_test_split(index, test_size=.33, shuffle=False)

pers = persistence(test, 12, index=test_idx)

for TIMELAG in range(1,20):

    X_train, Y_train = patterns(train, INPUT_SIZE, TIMELAG, HORIZONS, dropnan=True, index=train_idx)
    X_test = patterns(test, INPUT_SIZE, TIMELAG, dropnan=False, index=test_idx)

    model = MLP(input_size=INPUT_SIZE, horizons=HORIZONS, inits=INITS)
    model.build(hidden_layers=HIDDEN_LAYERS, activation=ACTIVATION, optimizer=OPTIMIZER, loss=LOSS)
    model.train(X_train.values, Y_train.values, validation_split=VALIDATION_SPLIT, epochs=EPOCHS)
    pred = model.predict(X_test.values)
    pred.index = test_idx

    metrics = [all_metrics_from_dataframe(pers, test), all_metrics_from_dataframe(pred, test)]
    diagram(std_obs=np.nanstd(test), metrics=metrics, names=['P','M'], savename=CENTRAL+'_'+str(TIMELAG))
