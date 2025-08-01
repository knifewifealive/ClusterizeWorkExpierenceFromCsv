import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def elbow_method(filename='buffer.csv'):
    """
    Метод локтя для определения оптимального количества кластеров.
    """

    df = pd.read_csv(filename, delimiter=';')

    # Удаляем строки, где опыт работы отсутствует
    df = df.dropna(subset=['Опыт работы кандидата'])

    # Удаляем строки, где в опыте только пробелы или пусто
    df['Опыт работы кандидата'] = df['Опыт работы кандидата'].astype(
        str).str.strip()
    df = df[df['Опыт работы кандидата'] != '']

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Опыт работы кандидата'])

    inertia = []
    K = range(1, 301, 10)

    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    plt.plot(K, inertia, 'bx-')
    plt.xlabel('Количество кластеров')
    plt.ylabel('Inertia')
    plt.title('Метод локтя')
    plt.show()


elbow_method()
# Оптимально 250
