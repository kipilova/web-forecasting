import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

def get_analog_long_metrics(df_train, df_test):

    df_train.index = pd.to_datetime(df_train.index)
    df_test.index = pd.to_datetime(df_test.index)

    # Логарифмирование для сглаживания выбросов
    df_train['views'] = np.log1p(df_train['views'])
    df_test['views'] = np.log1p(df_test['views'])

    start_fit = time.perf_counter()

    predictions = []

    for current_date in df_test.index:
        weekday = current_date.weekday()
        month = current_date.month
        day = current_date.day

        # Поиск аналогичных дней за прошлые годы: тот же месяц, день и день недели
        analog_days = df_train[
            (df_train.index.month == month) &
            (df_train.index.day == day) &
            (df_train.index.weekday == weekday)
        ]

        # такие же дни недели в этом месяце
        if analog_days.empty:
            analog_days = df_train[
                (df_train.index.month == month) &
                (df_train.index.weekday == weekday)
            ]

        # все дни с таким же днём недели
        if analog_days.empty:
            analog_days = df_train[df_train.index.weekday == weekday]

        pred = analog_days['views'].mean()
        predictions.append(pred)

    end_fit = time.perf_counter()
    fit_time = end_fit - start_fit

    start_pred = time.perf_counter()
    y_pred = np.array(predictions)
    y_true = df_test['views'].values
    end_pred = time.perf_counter()
    predict_time = end_pred - start_pred

    # Обратное логарифмирование
    y_pred_final = np.expm1(y_pred)
    y_true_final = np.expm1(y_true)

    residuals = y_true_final - y_pred_final

    # plt.figure(figsize=(6, 4))
    # sns.histplot(residuals, bins=30, kde=True)
    # plt.title('Гистограмма остатков')
    # plt.xlabel('Остаток')
    # plt.ylabel('Частота')
    # plt.tight_layout()
    # plt.show()

    # Метрики
    mse = mean_squared_error(y_true_final, y_pred_final)
    mae = mean_absolute_error(y_true_final, y_pred_final)
    r2 = r2_score(y_true_final, y_pred_final)
    nonzero_mask = y_true_final != 0
    mape = np.mean(np.abs((y_true_final[nonzero_mask] - y_pred_final[nonzero_mask]) / y_true_final[nonzero_mask])) * 100
    bias = np.mean(y_pred_final - y_true_final)

    print("Метрики модели (многолетняя аналогия):")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Bias: {bias}")
    print(f"Время генерации прогноза (fit): {fit_time:.4f} сек")
    print(f"Время прогноза (predict): {predict_time:.6f} сек")

    # Визуализация
    # plt.figure(figsize=(12,6))
    # plt.plot(df_test.index, y_true_final, label='Реальные данные', color='blue')
    # plt.plot(df_test.index, y_pred_final, label='Прогноз', color='red')
    # plt.xticks(rotation=45)
    # plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.title('Прогнозирование по аналогии (несколько лет)')
    # plt.xlabel('Дата')
    # plt.ylabel('Просмотры')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    return mse, mae, r2, mape, bias, fit_time, predict_time
