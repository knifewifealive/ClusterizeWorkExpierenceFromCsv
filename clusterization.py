import matplotlib.pyplot as plt
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from collections import Counter


def clusterize(filename='buffer.csv', out_filename='clustered_output.csv') -> None:
    """
    Функция для кластеризации данных из файла CSV.
    Кластеризует текстовые данные и сохраняет результаты в новый CSV файл.
    :param filename: Путь к входному файлу CSV.
    :param out_filename: Путь к выходному файлу CSV.
    """
    df = pd.read_csv(filename)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Опыт работы'])

    kmeans = KMeans(n_clusters=300)
    df['Кластер индекс'] = kmeans.fit_predict(X)

    # Находим самое частое слово в каждом кластере
    cluster_words = {}
    word_pattern = re.compile(r'\b[А-Яа-яA-Za-zЁё]{4,}\b')
    for cluster in df['Кластер индекс'].unique():
        cluster_texts = df[df['Кластер индекс'] == cluster]['Опыт работы']
        words = []
        for text in cluster_texts:
            words.extend(word for word in word_pattern.findall(text))
        if words:
            counter = Counter(words).most_common(2)
            try:
                common_word_1st, common_word_2nd = counter[0][0], counter[1][0]
            except IndexError:
                common_word_1st = counter[0][0]
                common_word_2nd = ''
        else:
            common_word_1st, common_word_2nd = ''
        cluster_words[cluster] = common_word_1st, common_word_2nd

    df['Самое популярное слово / словосочетание кластера'] = df['Кластер индекс'].map(
        cluster_words)

    df = df.sort_values('Кластер индекс')
    df[['Кластер индекс', 'Самое популярное слово / словосочетание кластера','Опыт работы']].to_csv(out_filename, index=False)
