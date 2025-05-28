import requests
import pandas as pd

# === Функция для получения данных из API Wikipedia === #
def get_wikipedia_pageviews(article, start_date, end_date, project='en.wikipedia.org'):
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{project}/all-access/user/{article}/daily/{start_date}/{end_date}"
    headers = {'User-Agent': 'WikipediaPageviewsCollector/1.0 (akipilova@gmail.com)'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        items = data['items']
        df = pd.DataFrame({
            'date': [item['timestamp'][:8] for item in items],
            'views': [item['views'] for item in items]
        })
        df['date'] = pd.to_datetime(df['date'])
        return df
    else:
        print(f"Ошибка: {response.status_code}, {response.text}")
        return None
    
