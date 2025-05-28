from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
import pandas as pd
import seaborn as sns

def create_sequences_multifeature(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length][0])
        return np.array(X), np.array(y)

def get_lstm_metrics(df_train, df_test, use_day_of_week=True):

    # === Добавление признака день недели ===
    if use_day_of_week:
        df_train['day_of_week'] = df_train.index.dayofweek
        df_test['day_of_week'] = df_test.index.dayofweek

    # === Логарифмирование ===
    df_train['views'] = np.log1p(df_train['views'])
    df_test['views'] = np.log1p(df_test['views'])

    # === Масштабирование ===
    scaler_views = MinMaxScaler()
    views_train_scaled = scaler_views.fit_transform(df_train['views'].values.reshape(-1, 1))
    views_test_scaled = scaler_views.transform(df_test['views'].values.reshape(-1, 1))

    if use_day_of_week:
        scaler_dow = MinMaxScaler()
        dow_train_scaled = scaler_dow.fit_transform(df_train['day_of_week'].values.reshape(-1, 1))
        dow_test_scaled = scaler_dow.transform(df_test['day_of_week'].values.reshape(-1, 1))
        train_scaled = np.hstack((views_train_scaled, dow_train_scaled))
        test_scaled = np.hstack((views_test_scaled, dow_test_scaled))
        input_shape = (30, 2)
    else:
        train_scaled = views_train_scaled
        test_scaled = views_test_scaled
        input_shape = (30, 1)

    # === Формирование последовательностей ===
    seq_length = 30
    if len(train_scaled) <= seq_length or len(test_scaled) <= seq_length:
        print("Недостаточно данных для создания последовательностей.")
        return None

    X_train, y_train = create_sequences_multifeature(train_scaled, seq_length)
    X_test, y_test = create_sequences_multifeature(test_scaled, seq_length)

    # === Модель ===
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # === Обучение ===
    start_fit = time.perf_counter()
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test),
              callbacks=[early_stop], verbose=0)
    end_fit = time.perf_counter()
    fit_time = end_fit - start_fit

    # === Прогноз ===
    start_pred = time.perf_counter()
    y_pred = model.predict(X_test)
    end_pred = time.perf_counter()
    predict_time = end_pred - start_pred

    # === Обратное преобразование ===
    y_pred_rescaled = scaler_views.inverse_transform(y_pred)
    y_test_rescaled = scaler_views.inverse_transform(y_test.reshape(-1, 1))

    y_pred_final = np.expm1(y_pred_rescaled).flatten()
    y_true_final = np.expm1(y_test_rescaled).flatten()

    # === Метрики ===
    residuals = y_true_final - y_pred_final
    mse = mean_squared_error(y_true_final, y_pred_final)
    mae = mean_absolute_error(y_true_final, y_pred_final)
    r2 = r2_score(y_true_final, y_pred_final)
    nonzero_mask = y_true_final != 0
    mape = np.mean(np.abs(residuals[nonzero_mask] / y_true_final[nonzero_mask])) * 100
    bias = np.mean(residuals)

    # plt.figure(figsize=(6, 4))
    # sns.histplot(residuals, bins=30, kde=True)
    # plt.title('Гистограмма остатков')
    # plt.xlabel('Остаток')
    # plt.ylabel('Частота')
    # plt.tight_layout()
    # plt.show()

    label = "с day_of_week" if use_day_of_week else "без day_of_week"
    print(f"LSTM ({label})")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Bias: {bias}")
    print(f"Время обучения (fit): {fit_time:.4f} сек")
    print(f"Время прогноза (predict): {predict_time:.4f} сек")

    # === Визуализация ===
    forecast_index = df_test.index[seq_length:seq_length + len(y_pred_final)]
    y_pred_series = pd.Series(y_pred_final, index=forecast_index)
    y_true_series = pd.Series(y_true_final, index=forecast_index)

    # plt.figure(figsize=(12, 6))
    # plt.plot(y_true_series, label="Реальные данные", color='blue')
    # plt.plot(y_pred_series, label="Прогноз LSTM", color='red')
    # plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.xticks(rotation=45)
    # plt.title(f"Сравнение прогноза LSTM и реальных данных ({label})")
    # plt.xlabel("Дата")
    # plt.ylabel("Просмотры")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    return mse, mae, r2, mape, bias, fit_time, predict_time
