"""ALTERNATIVE PATH - time series (e.g. Colombo daily/monthly rainfall): RF + ARIMA hybrid.
CSV needs columns: date, value  (e.g. rainfall_mm). Time-based split, NO shuffling.
Hybrid = ARIMA forecast + Random Forest forecast of ARIMA's residuals (from lag features)."""
import sys, numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

path = sys.argv[1] if len(sys.argv) > 1 else "data/rainfall.csv"
s = pd.read_csv(path, parse_dates=["date"]).set_index("date")["value"].asfreq("MS").interpolate()
split = int(len(s) * 0.8); train, test = s[:split], s[split:]
SEASON, LAGS, SEED = 12, 12, 42

arima = SARIMAX(train, order=(1, 0, 1), seasonal_order=(1, 1, 1, SEASON)).fit(disp=False)
fit_res = (train - arima.fittedvalues).iloc[SEASON:]                      # in-sample residuals

def lag_frame(res):
    return pd.concat({f"lag{i}": res.shift(i) for i in range(1, LAGS + 1)}, axis=1).dropna()

Xr = lag_frame(fit_res); yr = fit_res.loc[Xr.index]
rf = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(Xr, yr)

arima_fc = arima.forecast(len(test))
hist = list(fit_res.values); res_fc = []                                   # recursive residual forecast
for _ in range(len(test)):
    x = np.array(hist[-LAGS:][::-1]).reshape(1, -1)
    r = rf.predict(x)[0]; res_fc.append(r); hist.append(r)
hybrid_fc = arima_fc.values + np.array(res_fc)

def report(name, p):
    k = test.values > 1                                  # MAPE is meaningless when rainfall is ~0 mm
    mape = np.mean(np.abs((test.values[k] - p[k]) / test.values[k])) * 100
    print(f"{name:14s} MAE={mean_absolute_error(test, p):8.3f} RMSE={mean_squared_error(test, p) ** .5:8.3f} MAPE={mape:7.2f}%")
report("SARIMA", arima_fc.values); report("RF+SARIMA", hybrid_fc)
