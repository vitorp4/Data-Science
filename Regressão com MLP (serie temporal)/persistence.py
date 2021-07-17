from processing import inverse_scaling
import pandas as pd

class Persistence:
    def __init__(self, horizons):
        self.horizons = horizons

    def predict(self, test, scaler, index=None):
        persist = {}
        for h in range(1, self.horizons+1):
            persist[f't+{h}'] = inverse_scaling(test, scaler, h)
        persist = pd.DataFrame(data=persist)

        if index is not None:
            persist.index = index

        return persist