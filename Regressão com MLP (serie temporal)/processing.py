import pandas as pd
import numpy as np

def patterns(serie, lags, horizons=None, dropnan=False, index=None):
    XY = {}
    for l in range(lags-1, -1, -1):
        XY[f't-{l}'] = shift(serie, l)
    if horizons != None:
        for h in range(1, horizons+1):
            XY[f't+{h}'] = shift(serie, -h)
    XY = pd.DataFrame(data=XY)
    if index is not None:
        XY.index = index
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

def persistance(serie, horizons, index=None):
    persist = {}
    for h in range(1, horizons+1):
        persist[f't+{h}'] = shift(serie, h)
    persist = pd.DataFrame(data=persist)
    if index is not None:
        persist.index = index
    return persist