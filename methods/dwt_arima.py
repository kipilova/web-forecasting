from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import mean_squared_error
import time
from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pywt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0  # чтобы избежать деления на ноль
    return np.mean(diff) * 100

# === Применение DWT === #
def apply_dwt(data, wavelet='db1'):
    coeffs = pywt.dwt(data.flatten(), wavelet)
    A, D = coeffs[0], coeffs[1]
    return A, D

# Восстановление данных
def reconstruct_dwt(A, D, wavelet='db1'):
    reconstructed = pywt.idwt(A, D, wavelet)
    return reconstructed

def get_dwt_arima_metrics(df_train, df_test):

    forecast_horizon = 180

    # === Логарифмирование ===
    df_train['views'] = np.log1p(df_train['views'])
    df_test['views'] = np.log1p(df_test['views'])

    # --- Удаление выбросов по IQR ---
    Q1 = df_train['views'].quantile(0.25)
    Q3 = df_train['views'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # только "нормальные" значения
    df_train = df_train[(df_train['views'] >= lower_bound) & (df_train['views'] <= upper_bound)]

    train_values = df_train['views'].values

    # === DWT разложение ===
    wavelet = 'dmey' # db1, db4, sym5, coif3, bior3.5, dmey
    A, D = pywt.dwt(train_values, wavelet)

    forecast_horizon = min(180, len(A), len(D))

    start_fit = time.perf_counter()
    model_A = auto_arima(A, seasonal=False, stepwise=True, suppress_warnings=True)
    model_D = auto_arima(D, seasonal=False, stepwise=True, suppress_warnings=True)
    end_fit = time.perf_counter()
    fit_time = end_fit - start_fit

    start_pred = time.perf_counter()
    dwt_steps = forecast_horizon // 2
    A_forecast = model_A.predict(n_periods=dwt_steps)
    D_forecast = model_D.predict(n_periods=dwt_steps)
    forecast_log = pywt.idwt(A_forecast, D_forecast, wavelet)
    forecast_final = np.expm1(forecast_log)
    end_pred = time.perf_counter()
    predict_time = end_pred - start_pred

    y_pred = forecast_final
    y_true = np.expm1(df_test['views'].values)

    # Синхронизация длин
    min_len = min(len(y_true), len(y_pred))
    y_true = y_true[:min_len]
    y_pred = y_pred[:min_len]

    # === Метрики ===
    residuals = y_true - y_pred
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nonzero_mask = y_true != 0
    # mape = np.mean(np.abs(residuals[nonzero_mask] / y_true[nonzero_mask])) * 100
    mape = smape(y_true, y_pred)
    bias = np.mean(residuals)

    # === Вывод ===
    print("DWT + ARIMA")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Bias: {bias}")
    print(f"Время обучения (fit): {fit_time:.4f} сек")
    print(f"Время прогноза (predict): {predict_time:.4f} сек")

    forecast_index = df_test.index[:len(y_pred)]
    y_pred_series = pd.Series(y_pred, index=forecast_index)
    y_true_series = pd.Series(y_true, index=forecast_index)

    # plt.figure(figsize=(12, 6))
    # plt.plot(y_true_series, label="Реальные данные", color='blue')
    # plt.plot(y_pred_series, label="Прогноз DWT+ARIMA", color='red')
    # plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.xticks(rotation=45)
    # plt.title("Сравнение прогноза DWT+ARIMA и реальных данных")
    # plt.xlabel("Дата")
    # plt.ylabel("Просмотры")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    return mse, mae, r2, mape, bias, fit_time, predict_time