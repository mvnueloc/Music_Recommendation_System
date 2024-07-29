
# Music Recommendation System

Music recommendation system based on the analysis and classification of song metadata, such as rhythm, tempo, popularity, and musical key.

<img width="998" alt="Spotify for developers" src="[https://github.com/user-attachments/assets/b38b5b79-e631-4247-8159-335fea4e603c](https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.paeg.uz%2Fposts%2F299-sohaning-muhim-korsatgichlari%2F%3Fu%3Dspotify-api-making-my-first-call-mark-needham-jj-v2Ae7ASN&psig=AOvVaw2bZsY1hrnFTvLnCDqH5g9n&ust=1722371506162000&source=images&cd=vfe&opi=89978449&ved=0CBEQjRxqFwoTCKiokoOMzYcDFQAAAAAdAAAAABAQ)![image](https://github.com/user-attachments/assets/31623128-1e04-48f8-b6f7-f76b8a28ac37)
">


## Project Summary

To recommend music, we used the K-means algorithm to classify songs. We determined the optimal number of clusters using a dataset containing song metadata. Each resulting group shares similar characteristics, enabling more accurate and personalized recommendations.

<img width="998" alt="Captura de pantalla 2024-07-29 a la(s) 2 05 07 p m" src="https://github.com/user-attachments/assets/b38b5b79-e631-4247-8159-335fea4e603c">

<img width="999" alt="Captura de pantalla 2024-07-29 a la(s) 2 05 16 p m" src="https://github.com/user-attachments/assets/f257079a-ccae-4067-badc-52aeac00646a">

To recommend songs that match users' tastes, you need to enter the following values. These can be obtained from an average of your favorite playlist or your favorite song:

- Popularity: 73
- Danceability: 0.676
- Energy: 0.4610
- Musical Key: 1
- Volume (dB): -6.746
- Mode: 0
- Speechiness: 0.1430
- Acousticness: 0.0322
- Instrumentalness: 0.000001
- Liveness: 0.3580
- Valence: 0.7150
- Tempo (BPM): 87.917

This value is compared to the previous dataset, the algorithm displays the 5 songs with the most similar features, and saves all the details in a JSON file.

<img width="702" alt="Captura de pantalla 2024-07-29 a la(s) 2 29 34 p m" src="https://github.com/user-attachments/assets/35d32ce8-ec3f-47e8-a2e0-798f2f5bbee2">

<img width="913" alt="Captura de pantalla 2024-07-29 a la(s) 2 30 11 p m" src="https://github.com/user-attachments/assets/a1d0058c-db7c-44dc-bf5a-b70fc61e4b78">

## Run Locally

Clone the project

```bash
  git clone https://github.com/mvnueloc/Music_Recommendation_System
```

Go to the project directory

```bash
  cd Music_Recommendation_System
```

Install dependences

```bash
  pip install -r requirements.txt
```


Run the script

```bash
  python app.py
```


## Credits and Acknowledgments

To conduct this analysis, the Kaggle dataset of [Kaggle dataset of Spotify top hits from 2000 to 2019](https://www.kaggle.com/datasets/paradisejoy/top-hits-spotify-from-20002019) was utilized. This dataset provides a detailed view of the most popular songs on Spotify over nearly two decades, offering a solid foundation for an in-depth analysis of musical trends and listener behavior.

## Authors

- [@mvnueloc](https://github.com/mvnueloc)
- [@danielctecla](https://github.com/danielctecla)
- [@Ergusha11](https://github.com/Ergusha11)

