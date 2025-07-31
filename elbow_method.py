import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def elbow_method(filename='buffer.csv'):
    """
    Метод локтя для определения оптимального количества кластеров.
    """

    df = pd.read_csv(filename)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Опыт работы'])
    inertia = []
    K = range(1,500, 10)  # Проверяем количество кластеров от 300 до 1000 с шагом 100
    # Метод локтя для определения оптимального количества кластеров
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