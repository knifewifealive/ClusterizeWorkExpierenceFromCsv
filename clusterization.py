import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter

def clusterize(filename='buffer.csv', out_filename='clustered_output.csv'):
    df = pd.read_csv(filename)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Опыт работы'])

    kmeans = KMeans(n_clusters=300)
    df['Кластер индекс'] = kmeans.fit_predict(X)

    # Находим самое частое слово в каждом кластере
    cluster_words = {}
    word_pattern = re.compile(r'\b[А-Яа-яA-Za-zЁё]+\b')
    for cluster in df['Кластер индекс'].unique():
        cluster_texts = df[df['Кластер индекс'] == cluster]['Опыт работы']
        words = []
        for text in cluster_texts:
            words.extend(word for word in word_pattern.findall(text))
        if words:
            counter = Counter(words).most_common(3)
            most_common_word, second_common_word, third_common_word = counter[0][0], counter[1][0], counter[2][0]
        else:
            most_common_word = ''
        cluster_words[cluster] = most_common_word, second_common_word, third_common_word

    df['Топ 3 слова в кластере по частотности'] = df['Кластер индекс'].map(cluster_words)

    df = df.sort_values('Кластер индекс')
    print(df.head())
    df[['Кластер индекс', 'Топ 3 слова в кластере по частотности', 'Опыт работы']].to_csv(out_filename, index=False)