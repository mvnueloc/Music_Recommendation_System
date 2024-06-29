import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

print("Hola, bienvenido a nuestro recomendador de canciones con IA.")

# <-- leer el csv -->
file_path = './songs_normalize.csv'
songs_df = pd.read_csv(file_path)
# print(songs_df.head())

# <-- elegir las características para el clustering -->
features = ['popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
            'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# <-- normalizar las características -->
scaler = StandardScaler()
songs_normalized = scaler.fit_transform(songs_df[features])

# <-- calcular el número óptimo de clusters -->
subset_df = songs_df.sample(frac=0.1, random_state=1)
subset_normalized = scaler.transform(subset_df[features])

wcss_subset = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=100, n_init=5, random_state=0)
    kmeans.fit(subset_normalized)
    wcss_subset.append(kmeans.inertia_)

def find_optimal_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 10, wcss[-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i + 1
        y0 = wcss[i]
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2*y1 - y2*x1)
        denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator / denominator)
    
    return distances.index(max(distances)) + 1

optimal_clusters = find_optimal_clusters(wcss_subset)
# print(f'Optimal number of clusters: {optimal_clusters}')

# <-- graficar el método del codo -->
plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss_subset, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters (Subset Data)')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# <--aplicar k-means con el número óptimo de clusters-->
optimal_clusters = optimal_clusters
kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
songs_df['cluster'] = kmeans.fit_predict(songs_normalized)

# print(songs_df.head())cle

# <-- graficar los datos clasificados-->
pca = PCA(n_components=2) # redudir las dimensiones a 2
songs_pca = pca.fit_transform(songs_normalized)

pca_df = pd.DataFrame(data=songs_pca, columns=['principal_component_1', 'principal_component_2'])
pca_df['cluster'] = songs_df['cluster']

plt.figure(figsize=(10, 7))
for cluster in range(optimal_clusters):
    clustered_data = pca_df[pca_df['cluster'] == cluster]
    plt.scatter(clustered_data['principal_component_1'], clustered_data['principal_component_2'], label=f'Cluster {cluster}', s=50)

plt.title('K-means Clusters (PCA Reduced Data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# <-- recomendar canciones -->
def classify_and_recommend(new_song_params, num_recommendations=5):

    new_song_df = pd.DataFrame([new_song_params])
    new_song_normalized = scaler.transform(new_song_df[features])
    
    cluster_label = kmeans.predict(new_song_normalized)[0]
    
    cluster_songs = songs_df[songs_df['cluster'] == cluster_label]
    
    recommendations = cluster_songs.sample(n=num_recommendations, random_state=1)
    
    new_song_pca = pca.transform(new_song_normalized)
    
    plt.figure(figsize=(10, 7))
    for cluster in range(optimal_clusters):
        clustered_data = pca_df[pca_df['cluster'] == cluster]
        plt.scatter(clustered_data['principal_component_1'], clustered_data['principal_component_2'], label=f'Cluster {cluster}', s=50)
    
    plt.scatter(new_song_pca[0, 0], new_song_pca[0, 1], c='red', label='New Song', s=200, marker='X')
    plt.title('K-means Clusters (PCA Reduced Data) with New Song Prediction')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.show()
    
    return recommendations[['artist', 'song', 'duration_ms', 'popularity', 'danceability', 'energy', 'key', 
                            'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 
                            'liveness', 'valence', 'tempo', 'genre']]


# <-- solicitar los datos al usuario  -->
print("Por favor, introduce los siguientes datos para recomendarte una cancion:")

new_song_example = {
    'popularity': 70,
    'danceability': 0.75,
    'energy': 0.8,
    'key': 5,
    'loudness': -6.0,
    'mode': 1,
    'speechiness': 0.05,
    'acousticness': 0.1,
    'instrumentalness': 0.0,
    'liveness': 0.2,
    'valence': 0.6,
    'tempo': 120.0
}
# def solicitar_datos_cancion():
#     print("Por favor, introduce los siguientes datos de la canción:")
#     new_song_example = {
#         'popularity': int(input("Popularidad (0-100): ")),
#         'danceability': float(input("Bailable (0.0-1.0): ")),
#         'energy': float(input("Energía (0.0-1.0): ")),
#         'key': int(input("Clave musical (0-11 donde 0 es Do y 11 es Si): ")),
#         'loudness': float(input("Volumen (dB): ")),
#         'mode': int(input("Modo (0 = menor, 1 = mayor): ")),
#         'speechiness': float(input("Habla (0.0-1.0): ")),
#         'acousticness': float(input("Acústica (0.0-1.0): ")),
#         'instrumentalness': float(input("Instrumentalidad (0.0-1.0): ")),
#         'liveness': float(input("Vivacidad (0.0-1.0): ")),
#         'valence': float(input("Valencia (0.0-1.0): ")),
#         'tempo': float(input("Tempo (BPM): "))
#     }
#     return new_song_example

# new_song_example = solicitar_datos_cancion()

# <-- mostrar las recomendaciones -->
recommendations = classify_and_recommend(new_song_example)
# print(recommendations)

recommendations.to_csv('song_recommendations.csv', index=False)
print("Las recomendaciones se han guardado en 'song_recommendations.csv'")




