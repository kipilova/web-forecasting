import pandas as pd
from methods.arima import get_arima_metrics
from methods.sarima import get_sarima_metrics
from methods.lstm import get_lstm_metrics
from methods.gru import get_gru_metrics
from methods.dwt_arima import get_dwt_arima_metrics
from methods.dwt_lstm import get_dwt_lstm_metrics
from methods.analog_short import get_analog_short_metrics
from methods.analog_long import get_analog_long_metrics
import time
import numpy as np
from data.load_datasets import get_wikipedia_pageviews
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def main():

    with open('data/articles_list.txt', 'r', encoding='utf-8') as f:
        articles = [line.strip() for line in f]

    metrics_dict = {'arima': [], 'sarima': [], 'lstm': [], 'gru': [], 'dwt + arima': [], 'dwt + lstm': [], 'analog_short': [], 'analog_long': []}
    avg_metrics = {}

    article_num = 0

    for article in articles:
        if article_num % 200 == 0 and article_num != 0:
            print("Пауза для снижения нагрузки...")
            time.sleep(15)
        article_num += 1
        print(article)

        # Получение данных для обучения
        start_date_train = "20210101"
        end_date_train = "20231231"
        df_train = get_wikipedia_pageviews(article, start_date_train, end_date_train)

        # Получение данных для тестирования
        start_date_test = "20240101"
        end_date_test = "20240630"
        df_test = get_wikipedia_pageviews(article, start_date_test, end_date_test)

        # Отображение даты в графиках в правильном формате
        df_train['date'] = pd.to_datetime(df_train['date'])
        df_train.set_index('date', inplace=True)

        df_test['date'] = pd.to_datetime(df_test['date'])
        df_test.set_index('date', inplace=True)

        # plt.figure(figsize=(14, 6))
        # plt.plot(df_train.index, df_train['views'], label="Просмотры", linewidth=1.2)
        # plt.xticks(rotation=45)
        # ax = plt.gca()
        # ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # plt.title("Динамика просмотров статьи (2021–2023)", fontsize=14)
        # plt.xlabel("Дата")
        # plt.ylabel("Количество просмотров")
        # plt.grid(True)
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        if df_train is not None:

            # Модель ARIMA
            # arima_metrics = get_arima_metrics(df_train, df_test)
            # metrics_dict['arima'].append(arima_metrics)

            # Модель SARIMA
            # sarima_metrics = get_sarima_metrics(df_train, df_test)
            # metrics_dict['sarima'].append(sarima_metrics)

            # Модель LSTM
            # lstm_metrics = get_lstm_metrics(df_train, df_test)
            # metrics_dict['lstm'].append(lstm_metrics)

            # Модель GRU
            # gru_metrics = get_gru_metrics(df_train, df_test)
            # metrics_dict['gru'].append(gru_metrics)

            # Модель DWT + ARIMA
            # dwt_arima_metrics = get_dwt_arima_metrics(df_train, df_test)
            # metrics_dict['dwt + arima'].append(dwt_arima_metrics)

            # Модель DWT + LSTM
            # dwt_lstm_metrics = get_dwt_lstm_metrics(df_train, df_test)
            # metrics_dict['dwt + lstm'].append(dwt_lstm_metrics)

            # # Модель analog
            # analog_short_metrics = get_analog_short_metrics(df_train, df_test)
            # metrics_dict['analog_short'].append(analog_short_metrics)

            analog_long_metrics = get_analog_long_metrics(df_train, df_test)
            metrics_dict['analog_long'].append(analog_long_metrics)

        else:
            print(f"Skipping article {article} due to missing data.")

        time.sleep(1)

    # Среднее значение метрик
    avg_metrics = {}
    for method, metrics in metrics_dict.items():
        avg_metrics[method] = {
        'MSE': np.mean([m[0] for m in metrics]),
        'MAE': np.mean([m[1] for m in metrics]),
        'R2': np.mean([m[2] for m in metrics]),
        'MAPE': np.mean([m[3] for m in metrics]),
        'Bias': np.mean([m[4] for m in metrics]),
        'время обучения(сек)': np.mean([m[5] for m in metrics]),
        'время прогноза(сек)': np.mean([m[6] for m in metrics]),
    }

    print("Average metrics for each method:")
    for method, metrics in avg_metrics.items():
        print(f"\nMethod: {method}")
        for metric, value in metrics.items():
            print(f"{metric}: {value}") 

if __name__ == "__main__":
    main()
