import json

with open('data/topviews-2023.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract article titles
articles = [item['article'] for item in data]

# Optionally, save to a text file
with open('data/articles_list.txt', 'w', encoding='utf-8') as f:
    for title in articles:
        f.write(title + '\n')

print(f"Extracted {len(articles)} article titles.")