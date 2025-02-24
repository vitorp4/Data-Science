import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def supervised_patterns(serie, input_size=4, timelag=1, horizons=None, dropnan=False, index=None):
    XY = {}
    for l in range(input_size-1, -1, -1):
        XY[f't-{l}'] = shift(serie, l*timelag)

    if horizons != None:
        for h in range(1, horizons+1):
            XY[f't+{h}'] = shift(serie, -h)

    XY = pd.DataFrame(data=XY)
    if index is not None:
        XY.index = index

    if dropnan:
        XY = XY.dropna()

    X = XY.iloc[:,:input_size]
    if horizons != None:
        Y = XY.iloc[:,-horizons:]
        return X, Y
    else:
        return X

def shift(arr, num, fill=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

def scaling(train, test=None, feature_range=None):
    if type(train) == pd.Series:
        train = train.values
    if test is not None:
        if type(test) == pd.Series:
            test = test.values

    if feature_range is not None:
        scaler = MinMaxScaler(feature_range=feature_range)
    else:
        scaler = MinMaxScaler()

    scaler.fit(train.reshape(-1, 1))
    train_scaled = scaler.transform(train.reshape(-1, 1)).reshape(1,-1)[0]

    if test is not None:
        test_scaled = scaler.transform(test.reshape(-1, 1)).reshape(1,-1)[0]
        return train_scaled, test_scaled, scaler
    else:
        return train_scaled, scaler

def inverse_scaling(test, scaler, shifts=0):
    if len(test.shape) == 1:
        return scaler.inverse_transform(shift(test.reshape(-1, 1), shifts)).reshape(1, -1)[0]
    elif len(test.shape) == 2:
        return scaler.inverse_transform(shift(test, shifts)).reshape(1, -1)[0]