from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pywt
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

def create_sequences_multifeature(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])
    return np.array(X), np.array(y)

def train_and_forecast_lstm(X_train, y_train, X_test, y_test, epochs=20):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_test, y_test),
              callbacks=[early_stop], verbose=0)

    y_pred = model.predict(X_test)
    return model, y_pred.flatten()

def get_dwt_lstm_metrics(df_train, df_test, seq_length=10):

    df_train.index = pd.date_range(start="2023-01-01", periods=len(df_train), freq='D')
    df_test.index = pd.date_range(start="2024-01-01", periods=len(df_test), freq='D')

    # Добавление day_of_week
    df_train['day_of_week'] = df_train.index.dayofweek
    df_test['day_of_week'] = df_test.index.dayofweek

    # Логарифмирование
    df_train['views'] = np.log1p(df_train['views'])
    df_test['views'] = np.log1p(df_test['views'])

    wavelet = 'dmey' # db1, db4, sym5, coif3, bior3.5, dmey
    A_train, D_train = pywt.dwt(df_train['views'].values, wavelet)
    A_test, D_test = pywt.dwt(df_test['views'].values, wavelet)

    dow_train = df_train['day_of_week'].values[:len(A_train)]
    dow_test = df_test['day_of_week'].values[:len(A_test)]

    scaler_A = MinMaxScaler()
    scaler_D = MinMaxScaler()
    scaler_dow = MinMaxScaler()

    A_train_scaled = scaler_A.fit_transform(A_train.reshape(-1,1))
    D_train_scaled = scaler_D.fit_transform(D_train.reshape(-1,1))
    dow_train_scaled = scaler_dow.fit_transform(dow_train.reshape(-1,1))

    A_test_scaled = scaler_A.transform(A_test.reshape(-1,1))
    D_test_scaled = scaler_D.transform(D_test.reshape(-1,1))
    dow_test_scaled = scaler_dow.transform(dow_test.reshape(-1,1))

    # Объединение 3 признаков в один массив
    train_combined = np.hstack((A_train_scaled, D_train_scaled, dow_train_scaled))
    test_combined = np.hstack((A_test_scaled, D_test_scaled, dow_test_scaled))

    # Создание последовательностей
    X_train, y_train = create_sequences_multifeature(train_combined, seq_length)
    X_test, y_test = create_sequences_multifeature(test_combined, seq_length)

    # Обучение и прогноз
    start_fit = time.perf_counter()
    model, y_pred_scaled = train_and_forecast_lstm(X_train, y_train, X_test, y_test, epochs=50)
    end_fit = time.perf_counter()
    fit_time = end_fit - start_fit

    # Обратное масштабирование только для 'views'
    y_pred = scaler_A.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
    y_true = scaler_A.inverse_transform(y_test.reshape(-1,1)).flatten()

    # Обратное логарифмирование
    y_pred_final = np.expm1(y_pred)
    y_true_final = np.expm1(y_true)

    # Метрики
    residuals = y_true_final - y_pred_final
    mse = mean_squared_error(y_true_final, y_pred_final)
    mae = mean_absolute_error(y_true_final, y_pred_final)
    r2 = r2_score(y_true_final, y_pred_final)
    nonzero_mask = y_true_final != 0
    mape = np.mean(np.abs(residuals[nonzero_mask] / y_true_final[nonzero_mask])) * 100
    bias = np.mean(residuals)

    print("DWT + LSTM + day_of_week")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"R²: {r2}")
    print(f"MAPE: {mape:.2f}%")
    print(f"Bias: {bias}")
    print(f"Время обучения (fit): {fit_time:.4f} сек")
    # print(f"Время прогноза (predict): {predict_time:.4f} сек")

    forecast_index = df_test.index[seq_length:seq_length+len(y_pred_final)]
    y_pred_series = pd.Series(y_pred_final, index=forecast_index)
    y_true_series = pd.Series(y_true_final, index=forecast_index)

    # plt.figure(figsize=(12, 6))
    # plt.plot(y_true_series, label="Реальные данные", color='blue')
    # plt.plot(y_pred_series, label="Прогноз DWT+LSTM+day_of_week", color='red')
    # plt.gca().xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.xticks(rotation=45)
    # plt.title("Сравнение прогноза DWT+LSTM с day_of_week и реальных данных")
    # plt.xlabel("Дата")
    # plt.ylabel("Просмотры")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    return mse, mae, r2, mape, bias, fit_time, fit_time
