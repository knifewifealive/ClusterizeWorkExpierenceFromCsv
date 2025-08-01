import pandas as pd
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from collections import Counter

nltk.download('stopwords')
from nltk.corpus import stopwords

filename = 'buffer.csv'

def find_optimal_eps(X_reduced, min_samples=2):
    """
    Построение k-distance графика для выбора eps.
    """
    neighbors = NearestNeighbors(n_neighbors=min_samples, metric='cosine')
    neighbors_fit = neighbors.fit(X_reduced)
    distances, indices = neighbors_fit.kneighbors(X_reduced)
    distances = np.sort(distances[:, min_samples-1], axis=0)
    
    plt.figure(figsize=(8, 6))
    plt.plot(distances)
    plt.xlabel('Точки')
    plt.ylabel(f'Расстояние до {min_samples}-го ближайшего соседа')
    plt.title('K-Distance график для выбора eps')
    plt.show()
    
    return 0.25

def preprocess_text(df: pd.DataFrame) -> tuple[pd.DataFrame, any]:
    """
    Предобработка текста: удаление пустых строк и пробелов, векторизация.
    param df: DataFrame с колонкой 'Опыт работы кандидата'
    return: кортеж из DataFrame и векторизованных данных (Столбец 'Опыт работы кандидата')
    """
    df = df.dropna(subset=['Опыт работы кандидата'])
    df['Опыт работы кандидата'] = df['Опыт работы кандидата'].astype(str).str.strip()
    df = df[df['Опыт работы кандидата'] != '']

    vectorizer = TfidfVectorizer(stop_words=stopwords.words('russian'), max_features=1000)
    X = vectorizer.fit_transform(df['Опыт работы кандидата'])
    return df, X

# ----------------------------
# KMEANS CLUSTERING
# ----------------------------
def cluster_with_kmeans(df: pd.DataFrame, X, n_clusters=250) -> pd.DataFrame:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Кластер индекс'] = kmeans.fit_predict(X)
    return df

# ----------------------------
# AGGLOMERATIVE CLUSTERING
# ----------------------------
def cluster_with_agglomerative(df: pd.DataFrame, X, n_clusters=250) -> pd.DataFrame:
    model = AgglomerativeClustering(n_clusters=n_clusters)
    df['Кластер индекс'] = model.fit_predict(X.toarray())
    return df

# ----------------------------
# DBSCAN CLUSTERING
# ----------------------------
def cluster_with_dbscan(df: pd.DataFrame, X, eps=0.25, min_samples=5) -> pd.DataFrame:
    model = DBSCAN(eps=eps, min_samples=min_samples)
    df['Кластер индекс'] = model.fit_predict(X)
    return df

# ----------------------------
# LABEL CLUSTERS
# ----------------------------
def label_clusters(df: pd.DataFrame) -> pd.DataFrame:
    word_pattern = re.compile(r'\b[А-Яа-яA-Za-zЁё]{4,}\b')
    cluster_words = {}

    for cluster_label in sorted(df['Кластер индекс'].unique()):
        cluster_texts = df[df['Кластер индекс'] == cluster_label]['Опыт работы кандидата']
        words = []
        for text in cluster_texts:
            words.extend(word.lower() for word in word_pattern.findall(text))
        if words:
            most_common = Counter(words).most_common(2)
            top1 = most_common[0][0] if len(most_common) > 0 else ''
            top2 = most_common[1][0] if len(most_common) > 1 else ''
            label = f"{top1}, {top2}" if top2 else top1
        else:
            label = ''
        cluster_words[cluster_label] = label

    df['Самое популярное слово / словосочетание кластера'] = df['Кластер индекс'].map(cluster_words)
    return df

# Определение приоритетных секторов и ключевых слов
priority_sectors = {
    "Недвижимость": ["риэлтор", "агент", "недвижимост", "ипотек", "жилье", "риелтор"],
    "Розничные продажи": ["продавец", "торговля"],
    "Банковские специалисты": ["банк", "кредит", "финанс", "страхов", "инвестиц"]
}

# Функция для проверки, является ли кластер рекомендованным
def is_recommended(cluster_words, priority_sectors):
    # Разбиваем строку самых частых слов на список и приводим к нижнему регистру
    cluster_words_list = [word.lower() for word in cluster_words.split(', ')]
    # Проверяем наличие ключевых слов из приоритетных секторов
    for sector, keywords in priority_sectors.items():
        for keyword in keywords:
            if any(keyword in word for word in cluster_words_list):
                return "Да"
    return "Нет"




# ----------------------------
# SAVE CLUSTERED DATA
# ----------------------------
def save_clustered(df: pd.DataFrame, out_filename='clustered_output.csv'):
    df = df.sort_values('Кластер индекс')
    df.to_csv(out_filename, index=False, sep=';')
    print(f"Готово! Результат сохранён в файл: {out_filename}")

def cluster_file_data(filename='buffer.csv'):

    df_raw = pd.read_csv(filename, delimiter=';')
    df_processed, X = preprocess_text(df_raw)

    # Выбор метода кластеризации:
    #df_clustered = cluster_with_kmeans(df_processed.copy(), X, n_clusters=100)
    #df_clustered = cluster_with_agglomerative(df_processed.copy(), X, n_clusters=100)
    # Определение оптимального eps
    eps = find_optimal_eps(X, min_samples=2)
    print(f"Рекомендуемое значение eps: {eps}")

    df_clustered = cluster_with_dbscan(df_processed.copy(), X, eps=eps, min_samples=4)

    df_labeled = label_clusters(df_clustered)
    df_labeled['Рекомендован'] = df_labeled['Опыт работы кандидата'].apply(
        lambda x: is_recommended(x, priority_sectors)
    )
    save_clustered(df_labeled, out_filename='clustered_output.csv')

