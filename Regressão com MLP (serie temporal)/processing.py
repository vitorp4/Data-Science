import pandas as pd
import numpy as np

def patterns(serie, lags, horizons=None, dropnan=False):
    XY = {}
    for l in range(lags-1, -1, -1):
        XY[f't-{l}'] = serie.shift(l)
    if horizons != None:
        for h in range(1, horizons+1):
            XY[f't+{h}'] = serie.shift(-h)
    XY = pd.DataFrame(data=XY)
    if dropnan:
        XY = XY.dropna()
    X = XY.iloc[:,:lags]
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