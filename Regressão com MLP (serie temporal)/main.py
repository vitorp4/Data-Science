#%%
import pandas as pd
from mlp import MLP
from persistence import Persistence
from sklearn.model_selection import train_test_split
from processing import supervised_patterns, scaling, inverse_scaling
from metrics import std, metrics_df
from taylor_diagram import taylor_diagram
from iop import iop

if __name__ == '__main__':

    df = pd.read_csv('data/australia.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').rolling(6).mean().iloc[6::6]
    df.index.name = None
    df = df.interpolate(method='polynomial', order=5, axis=0).clip(lower=0)

    for CENTRAL in ['KIATAWF1', 'BOCORWF1','MACARTH1','STARHLWF']:

        serie = df[CENTRAL]

        train, test = train_test_split(serie, test_size=.33, shuffle=False)
        train, test, scaler = scaling(train, test, feature_range=(0,1))

        params = {
            'input_size': 5,
            'timelag': 1,
            'horizons': 12,
            'inits': 1,
            'hidden_layers': [10],
            'activation': 'sigmoid',
            'optimizer': 'sgd',
            'loss': 'mse',
            'validation_split': 0.1,
            'epochs': 200
        }

        pers = Persistence(horizons=params['horizons']).predict(test, scaler=scaler)

        X_train, Y_train = supervised_patterns(train, params['input_size'], params['timelag'], params['horizons'], dropnan=True)
        X_test = supervised_patterns(test, params['input_size'], params['timelag'], dropnan=False)

        model = MLP(input_size=params['input_size'], horizons=params['horizons'], inits=params['inits'])
        model.build(hidden_layers=params['hidden_layers'], activation=params['activation'], optimizer=params['optimizer'], loss=params['loss'])
        model.train(X_train, Y_train, validation_split=params['validation_split'], epochs=params['epochs'])
        pred = model.predict(X_test, scaler)

        metrics = {
            'P': metrics_df(pers, test),
            'M': metrics_df(pred, test)
        }
        taylor_diagram(std_obs=std(inverse_scaling(test, scaler)), metrics=metrics, title=CENTRAL, savename=f"taylor_{CENTRAL}")
        iop(metrics=metrics, stat='rmse', central=CENTRAL, model='MLP', savename=f"iop_mse_{CENTRAL}")
