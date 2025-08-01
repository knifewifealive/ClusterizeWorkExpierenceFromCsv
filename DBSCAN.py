import pandas as pd
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from natasha import MorphVocab, NewsMorphTagger, Segmenter, NewsEmbedding, Doc

nltk.download('stopwords')
from nltk.corpus import stopwords

###################################
# Лемматизация слишком долгая
###################################

filename = 'buffer.csv'

def preprocess_text(df: pd.DataFrame) -> tuple[pd.DataFrame, any, any]:
    """
    Предобработка текста: лемматизация с natasha, векторизация столбца 'Опыт работы кандидата'.
    param df: DataFrame с колонкой 'Опыт работы кандидата'
    return: кортеж из DataFrame, векторизованных данных и векторизатора
    """
    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    emb = NewsEmbedding()  # Добавляем эмбеддинги
    morph_tagger = NewsMorphTagger(emb)  # Передаем эмбеддинги в NewsMorphTagger
    
    def lemmatize(text):
        """Лемматизация текста с помощью natasha"""
        doc = Doc(text.lower())
        doc.segment(segmenter)
        doc.tag_morph(morph_tagger)
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
        words = [token.lemma for token in doc.tokens if token.lemma and len(token.lemma) >= 3]
        return ' '.join(word for word in words if word not in russian_stopwords)

    russian_stopwords = stopwords.words('russian')

    # Заполняем пропуски
    df['Опыт работы кандидата'] = df['Опыт работы кандидата'].fillna('не указано').astype(str).str.strip()
    df = df[df['Опыт работы кандидата'] != '']

    # Лемматизация
    df['Processed_Text'] = df['Опыт работы кандидата'].apply(lemmatize)

    # Векторизация
    vectorizer = TfidfVectorizer(max_features=500, stop_words=russian_stopwords)
    X = vectorizer.fit_transform(df['Processed_Text'])

    # Снижение размерности
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_reduced = svd.fit_transform(X)

    return df, X_reduced, vectorizer

def find_optimal_eps(X_reduced, min_samples=2):
    """
    Построение k-distance графика для выбора eps.
    """
    from sklearn.neighbors import NearestNeighbors
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

def cluster_with_dbscan(df: pd.DataFrame, X, eps=0.3, min_samples=2) -> pd.DataFrame:
    """
    Кластеризация с DBSCAN с использованием косинусного расстояния.
    """
    distance_matrix = cosine_distances(X)
    model = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    df['Кластер индекс'] = model.fit_predict(distance_matrix)

    # Оценка качества
    if len(set(df['Кластер индекс'])) > 1:
        score = silhouette_score(X, df['Кластер индекс'], metric='cosine')
        print(f"Силуэтный коэффициент: {score:.4f}")
    else:
        print("Кластеризация не удалась: только один кластер или все точки - шум")

    print(f"Доля шума (кластер -1): {100 * (df['Кластер индекс'] == -1).mean():.2f}%")
    print(f"Количество кластеров: {len(set(df['Кластер индекс'])) - (1 if -1 in df['Кластер индекс'].values else 0)}")

    return df

def label_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Присвоение меток кластерам на основе самых частых слов.
    """
    word_pattern = re.compile(r'\b[А-Яа-яA-Za-zЁё]{3,}\b')
    cluster_words = {}

    for cluster_label in sorted(df['Кластер индекс'].unique()):
        cluster_texts = df[df['Кластер индекс'] == cluster_label]['Processed_Text']
        words = []
        for text in cluster_texts:
            words.extend(word.lower() for word in word_pattern.findall(text))
        if words:
            most_common = Counter(words).most_common(2)
            top1 = most_common[0][0] if len(most_common) > 0 else ''
            top2 = most_common[1][0] if len(most_common) > 1 else ''
            label = f"{top1}, {top2}" if top2 else top1
        else:
            label = 'не указано'
        cluster_words[cluster_label] = label

    df['Самое популярное слово / словосочетание кластера'] = df['Кластер индекс'].map(cluster_words)
    return df

def save_clustered(df: pd.DataFrame, out_filename='clustered_output.csv'):
    """
    Сохранение результатов.
    """
    df = df.sort_values('Кластер индекс')
    df.to_csv(out_filename, index=False, sep=';')
    print(f"Готово! Результат сохранён в файл: {out_filename}")

def cluster_file_data(filename='buffer.csv'):
    """
    Основная функция для кластеризации данных.
    """
    df_raw = pd.read_csv(filename, delimiter=';')
    df_processed, X_reduced, vectorizer = preprocess_text(df_raw)

    # Определение оптимального eps
    eps = find_optimal_eps(X_reduced, min_samples=2)
    print(f"Рекомендуемое значение eps: {eps}")

    # Кластеризация
    df_clustered = cluster_with_dbscan(df_processed.copy(), X_reduced, eps=eps, min_samples=2)

    # Присвоение меток
    df_labeled = label_clusters(df_clustered)
    save_clustered(df_labeled, out_filename='clustered_output.csv')

    # Статистика
    print(df_labeled['Кластер индекс'].value_counts())

if __name__ == '__main__':
    cluster_file_data()