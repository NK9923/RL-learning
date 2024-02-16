import numpy as np
import pandas as pd

# verwendete Version   
def calculate_ex_ante_volatility(data, target_vola):
    delta = 60 / 61
    data_vol = data.dropna(subset='RoR')['RoR']

    ex_ante = np.zeros(len(data))

    for t in range(1, len(data)):
        total_sum = 0
        for i in range(t - 1):
            total_sum += (1 - delta) * (delta ** i) * (data_vol[t - 1 - i] - np.mean(data_vol[:t])) ** 2

        ex_ante[t - 1] = np.sqrt(total_sum)

    position_sizing = pd.DataFrame({'Index': data.index, 'ex_ante': ex_ante, 'Target_vola' : target_vola})
    position_sizing['Position_Size'] = position_sizing['Target_vola'] / position_sizing['ex_ante']    
    position_sizing['Position_Size'].replace([np.inf, -np.inf], np.nan, inplace=True)

    # Fill NaN values with the last available value (forward fill) and then the next available value (backward fill)
    position_sizing['Position_Size'].fillna(method='ffill', inplace=True)
    position_sizing['Position_Size'].fillna(method='bfill', inplace=True)

    position_sizing['Position_Size'].fillna(method='ffill', inplace=True)
    fill = position_sizing['Position_Size'].iloc[11]
    position_sizing.loc[position_sizing.index[0:10], 'Position_Size'] = fill
    position_sizing['Position_Size'] = round(position_sizing['Position_Size'])        
    
    return position_sizing


# eigentlich richtige Variante aber zu rechenintensiv
# TODO mit pybind11 C++ Code wrapppen
def calculate_ex_ante_volatility1(data, target_vola=0.4, span=30):
    delta = 60 / 61

    ex_ante = np.zeros(len(data))
    returns = pd.Series(data)

    for t in range(1, len(data)):
        if t > span:
            returns_ema = returns.iloc[(t - span):t].ewm(span=span).mean()
        else:
            returns_ema = returns.iloc[:t].mean()

        total_sum = 0
        for i in range(t - 1):
            total_sum += (1 - delta) * (delta ** i) * (data[t - 1 - i] - returns_ema.iloc[i]) ** 2

        ex_ante[t - 1] = np.sqrt(total_sum)

    return ex_ante
