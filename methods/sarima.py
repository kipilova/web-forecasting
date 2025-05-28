from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
import time
from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0  # чтобы избежать деления на ноль
    return np.mean(diff) * 100

def get_sarima_metrics(df_train, df_test):
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

    y_train = df_train['views'].values
    y_test = df_test['views'].values

    # Применение auto_arima для автоматического подбора параметров
    model_auto = auto_arima(
        y_train,
        seasonal=True,
        m=7,  # недельная сезонность
        stepwise=True,
        trace=True,
        suppress_warnings=True,
        start_p=0, start_q=0, start_P=0, start_Q=0,
        max_p=3, max_q=3, max_d=2,
        max_P=2, max_Q=2, max_D=1,
        error_action='ignore',
    )
    
    # Обучение
    start_fit = time.perf_counter()
    model_fit = model_auto.fit(y_train)
    end_fit = time.perf_counter()
    fit_time = end_fit - start_fit
    
    # Прогнозирование
    start_pred = time.perf_counter()
    forecast = model_fit.predict(n_periods=len(y_test))
    end_pred = time.perf_counter()
    predict_time = end_pred - start_pred

    # Обратное логарифмирование
    y_pred = np.expm1(forecast)
    y_true = np.expm1(y_test)

    y_pred = pd.Series(y_pred, index=df_test.index)
    y_true = pd.Series(y_true, index=df_test.index)

    residuals = y_true - y_pred

    # plt.figure(figsize=(6, 4))
    # sns.histplot(residuals, bins=30, kde=True)
    # plt.title('Гистограмма остатков')
    # plt.xlabel('Остаток')
    # plt.ylabel('Частота')
    # plt.tight_layout()
    # plt.show()
    
    # Метрики
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    nonzero_mask = y_true != 0
    # mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
    mape = smape(y_true, y_pred)
    bias = np.mean(y_pred - y_true)

    print("SARIMA")
    print("MSE: ", mse)
    print("MAE: ", mae)
    print("R2: ", r2)
    print("mape: ", mape)
    print("bias: ", bias)
    print(f"Время обучения (fit): {fit_time:.4f} сек")
    print(f"Время прогноза (predict): {predict_time:.4f} сек")

    # === Визуализация прогноза и реальных данных === #
    # plt.figure(figsize=(12, 6))
    # plt.plot(y_true.index, y_true, label="Реальные данные", color='blue')
    # plt.plot(y_pred.index, y_pred, label="Прогноз ARIMA", color='red')
    # plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.xticks(rotation=45)
    # plt.title("Сравнение прогноза ARIMA и реальных данных")
    # plt.xlabel("Дата")
    # plt.ylabel("Просмотры")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    return mse, mae, r2, mape, bias, fit_time, predict_time
