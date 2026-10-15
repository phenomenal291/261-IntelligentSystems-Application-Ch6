"""
Data Preparation Script for Spotify Songs Dataset
HK261 Intelligent Systems - Chapter 6 Application
Generates a curated dataset of popular global tracks with realistic Spotify audio features.
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "spotify_songs.json")

# Curated Tracks with authentic Spotify audio characteristics
TRACKS = [
    # --- Pop & Dance-Pop ---
    {"id": "sp_01", "title": "Blinding Lights", "artist": "The Weeknd", "genre": "Synthwave / Pop", "year": 2019,
     "danceability": 0.51, "energy": 0.73, "valence": 0.33, "tempo": 171.0, "acousticness": 0.001, "instrumentalness": 0.00, "speechiness": 0.06, "loudness": -5.9, "popularity": 96, "curated": True},
    {"id": "sp_02", "title": "Shape of You", "artist": "Ed Sheeran", "genre": "Pop", "year": 2017,
     "danceability": 0.83, "energy": 0.65, "valence": 0.93, "tempo": 96.0, "acousticness": 0.58, "instrumentalness": 0.00, "speechiness": 0.08, "loudness": -3.2, "popularity": 92, "curated": True},
    {"id": "sp_03", "title": "Levitating", "artist": "Dua Lipa", "genre": "Disco-Pop", "year": 2020,
     "danceability": 0.70, "energy": 0.83, "valence": 0.91, "tempo": 103.0, "acousticness": 0.01, "instrumentalness": 0.00, "speechiness": 0.06, "loudness": -3.8, "popularity": 90, "curated": True},
    {"id": "sp_04", "title": "Stay", "artist": "The Kid LAROI & Justin Bieber", "genre": "Pop / Trap", "year": 2021,
     "danceability": 0.59, "energy": 0.76, "valence": 0.48, "tempo": 170.0, "acousticness": 0.04, "instrumentalness": 0.00, "speechiness": 0.05, "loudness": -5.5, "popularity": 89, "curated": True},
    {"id": "sp_05", "title": "As It Was", "artist": "Harry Styles", "genre": "Indie Pop", "year": 2022,
     "danceability": 0.52, "energy": 0.73, "valence": 0.66, "tempo": 174.0, "acousticness": 0.34, "instrumentalness": 0.001, "speechiness": 0.06, "loudness": -5.3, "popularity": 93, "curated": True},
    {"id": "sp_06", "title": "Bad Guy", "artist": "Billie Eilish", "genre": "Electropop", "year": 2019,
     "danceability": 0.70, "energy": 0.43, "valence": 0.56, "tempo": 135.0, "acousticness": 0.33, "instrumentalness": 0.13, "speechiness": 0.38, "loudness": -10.9, "popularity": 88, "curated": True},
    {"id": "sp_07", "title": "Cruel Summer", "artist": "Taylor Swift", "genre": "Synth-Pop", "year": 2019,
     "danceability": 0.55, "energy": 0.70, "valence": 0.56, "tempo": 170.0, "acousticness": 0.12, "instrumentalness": 0.00, "speechiness": 0.16, "loudness": -5.7, "popularity": 95, "curated": False},
    {"id": "sp_08", "title": "Blank Space", "artist": "Taylor Swift", "genre": "Electropop", "year": 2014,
     "danceability": 0.76, "energy": 0.70, "valence": 0.58, "tempo": 96.0, "acousticness": 0.10, "instrumentalness": 0.00, "speechiness": 0.05, "loudness": -5.4, "popularity": 89, "curated": False},
    {"id": "sp_09", "title": "Flowers", "artist": "Miley Cyrus", "genre": "Disco-Pop", "year": 2023,
     "danceability": 0.71, "energy": 0.68, "valence": 0.65, "tempo": 118.0, "acousticness": 0.06, "instrumentalness": 0.00, "speechiness": 0.07, "loudness": -4.3, "popularity": 91, "curated": False},
    {"id": "sp_10", "title": "Starboy", "artist": "The Weeknd ft. Daft Punk", "genre": "R&B / Electropop", "year": 2016,
     "danceability": 0.68, "energy": 0.59, "valence": 0.49, "tempo": 186.0, "acousticness": 0.14, "instrumentalness": 0.00, "speechiness": 0.28, "loudness": -7.0, "popularity": 94, "curated": False},
    {"id": "sp_11", "title": "Don't Start Now", "artist": "Dua Lipa", "genre": "Nu-Disco", "year": 2019,
     "danceability": 0.79, "energy": 0.79, "valence": 0.68, "tempo": 124.0, "acousticness": 0.01, "instrumentalness": 0.00, "speechiness": 0.08, "loudness": -4.5, "popularity": 87, "curated": False},
    {"id": "sp_12", "title": "Dynamite", "artist": "BTS", "genre": "Disco-Pop", "year": 2020,
     "danceability": 0.75, "energy": 0.77, "valence": 0.74, "tempo": 114.0, "acousticness": 0.01, "instrumentalness": 0.00, "speechiness": 0.09, "loudness": -4.9, "popularity": 85, "curated": False},
    {"id": "sp_13", "title": "Watermelon Sugar", "artist": "Harry Styles", "genre": "Funk-Pop", "year": 2019,
     "danceability": 0.55, "energy": 0.82, "valence": 0.56, "tempo": 95.0, "acousticness": 0.12, "instrumentalness": 0.00, "speechiness": 0.05, "loudness": -4.2, "popularity": 88, "curated": False},

    # --- EDM / Dance / Workout ---
    {"id": "sp_14", "title": "Wake Me Up", "artist": "Avicii", "genre": "EDM / Dance", "year": 2013,
     "danceability": 0.53, "energy": 0.78, "valence": 0.64, "tempo": 124.0, "acousticness": 0.003, "instrumentalness": 0.002, "speechiness": 0.05, "loudness": -5.7, "popularity": 87, "curated": True},
    {"id": "sp_15", "title": "Get Lucky", "artist": "Daft Punk ft. Pharrell", "genre": "Nu-Disco / Funk", "year": 2013,
     "danceability": 0.79, "energy": 0.81, "valence": 0.86, "tempo": 116.0, "acousticness": 0.04, "instrumentalness": 0.00, "speechiness": 0.04, "loudness": -9.3, "popularity": 86, "curated": True},
    {"id": "sp_16", "title": "Levels", "artist": "Avicii", "genre": "Progressive House", "year": 2011,
     "danceability": 0.58, "energy": 0.89, "valence": 0.45, "tempo": 126.0, "acousticness": 0.05, "instrumentalness": 0.01, "speechiness": 0.03, "loudness": -3.1, "popularity": 84, "curated": False},
    {"id": "sp_17", "title": "Closer", "artist": "The Chainsmokers ft. Halsey", "genre": "Electropop / Future Bass", "year": 2016,
     "danceability": 0.75, "energy": 0.52, "valence": 0.66, "tempo": 95.0, "acousticness": 0.41, "instrumentalness": 0.00, "speechiness": 0.03, "loudness": -5.6, "popularity": 87, "curated": False},
    {"id": "sp_18", "title": "Summer", "artist": "Calvin Harris", "genre": "Dance / EDM", "year": 2014,
     "danceability": 0.60, "energy": 0.86, "valence": 0.74, "tempo": 128.0, "acousticness": 0.02, "instrumentalness": 0.01, "speechiness": 0.03, "loudness": -3.6, "popularity": 85, "curated": False},
    {"id": "sp_19", "title": "One Kiss", "artist": "Calvin Harris & Dua Lipa", "genre": "Deep House", "year": 2018,
     "danceability": 0.79, "energy": 0.86, "valence": 0.59, "tempo": 124.0, "acousticness": 0.04, "instrumentalness": 0.00, "speechiness": 0.11, "loudness": -3.2, "popularity": 86, "curated": False},
    {"id": "sp_20", "title": "Animals", "artist": "Martin Garrix", "genre": "Big Room House", "year": 2013,
     "danceability": 0.61, "energy": 0.94, "valence": 0.32, "tempo": 128.0, "acousticness": 0.001, "instrumentalness": 0.78, "speechiness": 0.06, "loudness": -4.8, "popularity": 80, "curated": False},
    {"id": "sp_21", "title": "Titanium", "artist": "David Guetta ft. Sia", "genre": "Dance-Pop / Electro", "year": 2011,
     "danceability": 0.60, "energy": 0.79, "valence": 0.30, "tempo": 126.0, "acousticness": 0.07, "instrumentalness": 0.15, "speechiness": 0.10, "loudness": -3.7, "popularity": 85, "curated": False},
    {"id": "sp_22", "title": "Lean On", "artist": "Major Lazer & DJ Snake", "genre": "Moombahton / Dance", "year": 2015,
     "danceability": 0.72, "energy": 0.81, "valence": 0.52, "tempo": 98.0, "acousticness": 0.003, "instrumentalness": 0.001, "speechiness": 0.06, "loudness": -3.1, "popularity": 84, "curated": False},
    {"id": "sp_23", "title": "Faded", "artist": "Alan Walker", "genre": "Electro House", "year": 2015,
     "danceability": 0.60, "energy": 0.65, "valence": 0.17, "tempo": 90.0, "acousticness": 0.03, "instrumentalness": 0.001, "speechiness": 0.05, "loudness": -5.1, "popularity": 83, "curated": False},

    # --- Acoustic / Folk / Melancholic Ballads ---
    {"id": "sp_24", "title": "Someone Like You", "artist": "Adele", "genre": "Ballad / Acoustic", "year": 2011,
     "danceability": 0.38, "energy": 0.33, "valence": 0.28, "tempo": 135.0, "acousticness": 0.89, "instrumentalness": 0.00, "speechiness": 0.03, "loudness": -8.2, "popularity": 86, "curated": True},
    {"id": "sp_25", "title": "Drivers License", "artist": "Olivia Rodrigo", "genre": "Pop Ballad", "year": 2021,
     "danceability": 0.59, "energy": 0.43, "valence": 0.13, "tempo": 144.0, "acousticness": 0.72, "instrumentalness": 0.00, "speechiness": 0.06, "loudness": -8.8, "popularity": 87, "curated": True},
    {"id": "sp_26", "title": "Someone You Loved", "artist": "Lewis Capaldi", "genre": "Pop Ballad", "year": 2019,
     "danceability": 0.50, "energy": 0.41, "valence": 0.45, "tempo": 110.0, "acousticness": 0.75, "instrumentalness": 0.00, "speechiness": 0.03, "loudness": -5.7, "popularity": 89, "curated": False},
    {"id": "sp_27", "title": "All of Me", "artist": "John Legend", "genre": "R&B / Soul Ballad", "year": 2013,
     "danceability": 0.42, "energy": 0.26, "valence": 0.33, "tempo": 120.0, "acousticness": 0.92, "instrumentalness": 0.00, "speechiness": 0.03, "loudness": -7.0, "popularity": 88, "curated": False},
    {"id": "sp_28", "title": "Easy On Me", "artist": "Adele", "genre": "Piano Ballad", "year": 2021,
     "danceability": 0.60, "energy": 0.37, "valence": 0.13, "tempo": 142.0, "acousticness": 0.84, "instrumentalness": 0.00, "speechiness": 0.04, "loudness": -7.5, "popularity": 85, "curated": False},
    {"id": "sp_29", "title": "Say You Won't Let Go", "artist": "James Arthur", "genre": "Acoustic Pop", "year": 2016,
     "danceability": 0.36, "energy": 0.56, "valence": 0.49, "tempo": 85.0, "acousticness": 0.69, "instrumentalness": 0.00, "speechiness": 0.06, "loudness": -7.4, "popularity": 86, "curated": False},
    {"id": "sp_30", "title": "Photograph", "artist": "Ed Sheeran", "genre": "Acoustic Pop", "year": 2014,
     "danceability": 0.61, "energy": 0.38, "valence": 0.20, "tempo": 108.0, "acousticness": 0.61, "instrumentalness": 0.00, "speechiness": 0.05, "loudness": -10.5, "popularity": 86, "curated": False},
    {"id": "sp_31", "title": "Lovely", "artist": "Billie Eilish & Khalid", "genre": "Chamber Pop", "year": 2018,
     "danceability": 0.35, "energy": 0.30, "valence": 0.12, "tempo": 115.0, "acousticness": 0.93, "instrumentalness": 0.00, "speechiness": 0.03, "loudness": -10.1, "popularity": 89, "curated": False},
    {"id": "sp_32", "title": "Skinny Love", "artist": "Bon Iver", "genre": "Indie Folk", "year": 2007,
     "danceability": 0.54, "energy": 0.26, "valence": 0.22, "tempo": 76.0, "acousticness": 0.87, "instrumentalness": 0.00, "speechiness": 0.05, "loudness": -13.2, "popularity": 79, "curated": False},
    {"id": "sp_33", "title": "Fast Car", "artist": "Tracy Chapman", "genre": "Folk Rock", "year": 1988,
     "danceability": 0.71, "energy": 0.34, "valence": 0.19, "tempo": 104.0, "acousticness": 0.31, "instrumentalness": 0.00, "speechiness": 0.04, "loudness": -15.5, "popularity": 84, "curated": False},

    # --- Hip-Hop & Rap ---
    {"id": "sp_34", "title": "HUMBLE.", "artist": "Kendrick Lamar", "genre": "Hip-Hop / Rap", "year": 2017,
     "danceability": 0.90, "energy": 0.62, "valence": 0.42, "tempo": 150.0, "acousticness": 0.0003, "instrumentalness": 0.00, "speechiness": 0.10, "loudness": -6.6, "popularity": 89, "curated": True},
    {"id": "sp_35", "title": "Lose Yourself", "artist": "Eminem", "genre": "Hardcore Hip-Hop", "year": 2002,
     "danceability": 0.69, "energy": 0.74, "valence": 0.06, "tempo": 171.0, "acousticness": 0.01, "instrumentalness": 0.00, "speechiness": 0.26, "loudness": -4.5, "popularity": 88, "curated": True},
    {"id": "sp_36", "title": "Sunflower", "artist": "Post Malone & Swae Lee", "genre": "Melodic Rap / Pop", "year": 2018,
     "danceability": 0.76, "energy": 0.48, "valence": 0.91, "tempo": 90.0, "acousticness": 0.56, "instrumentalness": 0.00, "speechiness": 0.05, "loudness": -5.6, "popularity": 91, "curated": True},
    {"id": "sp_37", "title": "God's Plan", "artist": "Drake", "genre": "Hip-Hop / Trap", "year": 2018,
     "danceability": 0.75, "energy": 0.45, "valence": 0.36, "tempo": 77.0, "acousticness": 0.03, "instrumentalness": 0.00, "speechiness": 0.11, "loudness": -9.2, "popularity": 87, "curated": False},
    {"id": "sp_38", "title": "SICKO MODE", "artist": "Travis Scott", "genre": "Southern Hip-Hop / Trap", "year": 2018,
     "danceability": 0.83, "energy": 0.73, "valence": 0.45, "tempo": 155.0, "acousticness": 0.005, "instrumentalness": 0.00, "speechiness": 0.22, "loudness": -3.7, "popularity": 88, "curated": False},
    {"id": "sp_39", "title": "Circles", "artist": "Post Malone", "genre": "Pop Rap / Alt-Pop", "year": 2019,
     "danceability": 0.69, "energy": 0.76, "valence": 0.55, "tempo": 120.0, "acousticness": 0.19, "instrumentalness": 0.002, "speechiness": 0.04, "loudness": -3.5, "popularity": 89, "curated": False},
    {"id": "sp_40", "title": "Without Me", "artist": "Eminem", "genre": "Hip-Hop", "year": 2002,
     "danceability": 0.91, "energy": 0.67, "valence": 0.66, "tempo": 112.0, "acousticness": 0.003, "instrumentalness": 0.00, "speechiness": 0.07, "loudness": -2.8, "popularity": 87, "curated": False},
    {"id": "sp_41", "title": "Lucid Dreams", "artist": "Juice WRLD", "genre": "Emo Rap", "year": 2018,
     "danceability": 0.51, "energy": 0.57, "valence": 0.22, "tempo": 84.0, "acousticness": 0.35, "instrumentalness": 0.00, "speechiness": 0.20, "loudness": -7.2, "popularity": 86, "curated": False},
    {"id": "sp_42", "title": "Rockstar", "artist": "Post Malone ft. 21 Savage", "genre": "Trap", "year": 2017,
     "danceability": 0.58, "energy": 0.52, "valence": 0.13, "tempo": 160.0, "acousticness": 0.12, "instrumentalness": 0.00, "speechiness": 0.07, "loudness": -6.1, "popularity": 86, "curated": False},
    {"id": "sp_43", "title": "MONTERO", "artist": "Lil Nas X", "genre": "Pop Rap", "year": 2021,
     "danceability": 0.61, "energy": 0.50, "valence": 0.66, "tempo": 179.0, "acousticness": 0.29, "instrumentalness": 0.00, "speechiness": 0.22, "loudness": -6.7, "popularity": 83, "curated": False},

    # --- Rock & Alternative ---
    {"id": "sp_44", "title": "Bohemian Rhapsody", "artist": "Queen", "genre": "Classic Rock / Progressive", "year": 1975,
     "danceability": 0.41, "energy": 0.40, "valence": 0.22, "tempo": 144.0, "acousticness": 0.27, "instrumentalness": 0.00, "speechiness": 0.05, "loudness": -10.0, "popularity": 86, "curated": True},
    {"id": "sp_45", "title": "Smells Like Teen Spirit", "artist": "Nirvana", "genre": "Grunge / Rock", "year": 1991,
     "danceability": 0.50, "energy": 0.91, "valence": 0.72, "tempo": 117.0, "acousticness": 0.00002, "instrumentalness": 0.002, "speechiness": 0.07, "loudness": -4.5, "popularity": 88, "curated": True},
    {"id": "sp_46", "title": "Believer", "artist": "Imagine Dragons", "genre": "Alt-Rock / Pop-Rock", "year": 2017,
     "danceability": 0.77, "energy": 0.78, "valence": 0.67, "tempo": 125.0, "acousticness": 0.06, "instrumentalness": 0.00, "speechiness": 0.13, "loudness": -4.4, "popularity": 90, "curated": True},
    {"id": "sp_47", "title": "Do I Wanna Know?", "artist": "Arctic Monkeys", "genre": "Indie Rock / Stoner Rock", "year": 2013,
     "danceability": 0.55, "energy": 0.53, "valence": 0.41, "tempo": 85.0, "acousticness": 0.19, "instrumentalness": 0.001, "speechiness": 0.03, "loudness": -7.5, "popularity": 89, "curated": False},
    {"id": "sp_48", "title": "Mr. Brightside", "artist": "The Killers", "genre": "Indie Rock / Post-Punk Revival", "year": 2004,
     "danceability": 0.35, "energy": 0.91, "valence": 0.24, "tempo": 148.0, "acousticness": 0.001, "instrumentalness": 0.00, "speechiness": 0.07, "loudness": -5.2, "popularity": 87, "curated": False},
    {"id": "sp_49", "title": "Yellow", "artist": "Coldplay", "genre": "Alt-Rock / Post-Britpop", "year": 2000,
     "danceability": 0.43, "energy": 0.70, "valence": 0.29, "tempo": 173.0, "acousticness": 0.003, "instrumentalness": 0.00, "speechiness": 0.03, "loudness": -7.2, "popularity": 90, "curated": False},
    {"id": "sp_50", "title": "In the End", "artist": "Linkin Park", "genre": "Nu-Metal / Alt-Rock", "year": 2000,
     "danceability": 0.56, "energy": 0.86, "valence": 0.40, "tempo": 105.0, "acousticness": 0.01, "instrumentalness": 0.00, "speechiness": 0.06, "loudness": -5.9, "popularity": 87, "curated": False},
    {"id": "sp_51", "title": "Viva La Vida", "artist": "Coldplay", "genre": "Baroque Pop / Alt-Rock", "year": 2008,
     "danceability": 0.49, "energy": 0.62, "valence": 0.42, "tempo": 138.0, "acousticness": 0.09, "instrumentalness": 0.00, "speechiness": 0.03, "loudness": -7.3, "popularity": 87, "curated": False},
    {"id": "sp_52", "title": "Radioactive", "artist": "Imagine Dragons", "genre": "Electronic Rock", "year": 2012,
     "danceability": 0.44, "energy": 0.78, "valence": 0.21, "tempo": 136.0, "acousticness": 0.11, "instrumentalness": 0.00, "speechiness": 0.06, "loudness": -3.7, "popularity": 84, "curated": False},
    {"id": "sp_53", "title": "Hotel California", "artist": "Eagles", "genre": "Classic Rock", "year": 1976,
     "danceability": 0.58, "energy": 0.51, "valence": 0.74, "tempo": 147.0, "acousticness": 0.006, "instrumentalness": 0.005, "speechiness": 0.03, "loudness": -9.3, "popularity": 86, "curated": False},

    # --- R&B, Soul & Chill Lofi ---
    {"id": "sp_54", "title": "Kill Bill", "artist": "SZA", "genre": "Contemporary R&B / Pop", "year": 2022,
     "danceability": 0.64, "energy": 0.73, "valence": 0.42, "tempo": 89.0, "acousticness": 0.05, "instrumentalness": 0.14, "speechiness": 0.04, "loudness": -5.7, "popularity": 92, "curated": False},
    {"id": "sp_55", "title": "Die For You", "artist": "The Weeknd", "genre": "R&B / Soul", "year": 2016,
     "danceability": 0.58, "energy": 0.53, "valence": 0.51, "tempo": 134.0, "acousticness": 0.11, "instrumentalness": 0.00, "speechiness": 0.06, "loudness": -7.2, "popularity": 89, "curated": False},
    {"id": "sp_56", "title": "From the Start", "artist": "Laufey", "genre": "Bossa Nova / Jazz Pop", "year": 2023,
     "danceability": 0.72, "energy": 0.42, "valence": 0.74, "tempo": 82.0, "acousticness": 0.81, "instrumentalness": 0.00, "speechiness": 0.05, "loudness": -11.5, "popularity": 87, "curated": False},
    {"id": "sp_57", "title": "Redbone", "artist": "Childish Gambino", "genre": "Psychedelic Soul / Funk", "year": 2016,
     "danceability": 0.74, "energy": 0.35, "valence": 0.58, "tempo": 116.0, "acousticness": 0.17, "instrumentalness": 0.003, "speechiness": 0.10, "loudness": -8.3, "popularity": 86, "curated": False},
    {"id": "sp_58", "title": "Peaches", "artist": "Justin Bieber ft. Daniel Caesar", "genre": "Pop / R&B", "year": 2021,
     "danceability": 0.68, "energy": 0.70, "valence": 0.46, "tempo": 90.0, "acousticness": 0.32, "instrumentalness": 0.00, "speechiness": 0.12, "loudness": -6.2, "popularity": 85, "curated": False},
    {"id": "sp_59", "title": "Don't Know Why", "artist": "Norah Jones", "genre": "Vocal Jazz / Acoustic", "year": 2002,
     "danceability": 0.65, "energy": 0.20, "valence": 0.62, "tempo": 88.0, "acousticness": 0.88, "instrumentalness": 0.00, "speechiness": 0.03, "loudness": -15.4, "popularity": 78, "curated": False},
    {"id": "sp_60", "title": "Rainy Cafe Study", "artist": "Lofi Fruit Music", "genre": "Lofi Hip-Hop / Chill", "year": 2022,
     "danceability": 0.66, "energy": 0.22, "valence": 0.35, "tempo": 78.0, "acousticness": 0.78, "instrumentalness": 0.82, "speechiness": 0.04, "loudness": -16.0, "popularity": 74, "curated": False},

    # --- Ambient & Classical Study (Distinct / Outlier Cluster for DBSCAN demo) ---
    {"id": "sp_61", "title": "Nuvole Bianche", "artist": "Ludovico Einaudi", "genre": "Modern Classical / Piano", "year": 2004,
     "danceability": 0.34, "energy": 0.15, "valence": 0.10, "tempo": 139.0, "acousticness": 0.99, "instrumentalness": 0.89, "speechiness": 0.04, "loudness": -21.4, "popularity": 80, "curated": False},
    {"id": "sp_62", "title": "Clair de Lune", "artist": "Claude Debussy", "genre": "Impressionist Classical", "year": 1905,
     "danceability": 0.25, "energy": 0.08, "valence": 0.08, "tempo": 65.0, "acousticness": 0.99, "instrumentalness": 0.92, "speechiness": 0.03, "loudness": -26.0, "popularity": 76, "curated": False},
    {"id": "sp_63", "title": "Gymnopédie No. 1", "artist": "Erik Satie", "genre": "Minimalist Classical", "year": 1888,
     "danceability": 0.38, "energy": 0.02, "valence": 0.21, "tempo": 74.0, "acousticness": 0.99, "instrumentalness": 0.91, "speechiness": 0.04, "loudness": -28.0, "popularity": 75, "curated": False},
    {"id": "sp_64", "title": "Weightless", "artist": "Marconi Union", "genre": "Ambient Soundscape", "year": 2011,
     "danceability": 0.18, "energy": 0.05, "valence": 0.04, "tempo": 60.0, "acousticness": 0.95, "instrumentalness": 0.97, "speechiness": 0.04, "loudness": -24.5, "popularity": 72, "curated": False},

    # --- Latin / Reggaeton ---
    {"id": "sp_65", "title": "Despacito", "artist": "Luis Fonsi & Daddy Yankee", "genre": "Latin Pop / Reggaeton", "year": 2017,
     "danceability": 0.65, "energy": 0.80, "valence": 0.84, "tempo": 178.0, "acousticness": 0.21, "instrumentalness": 0.00, "speechiness": 0.15, "loudness": -4.8, "popularity": 86, "curated": False},
    {"id": "sp_66", "title": "Tití Me Preguntó", "artist": "Bad Bunny", "genre": "Urbano Latino / Dembow", "year": 2022,
     "danceability": 0.65, "energy": 0.72, "valence": 0.19, "tempo": 107.0, "acousticness": 0.10, "instrumentalness": 0.00, "speechiness": 0.25, "loudness": -5.2, "popularity": 89, "curated": False},
    {"id": "sp_67", "title": "Dakiti", "artist": "Bad Bunny & Jhayco", "genre": "Reggaeton", "year": 2020,
     "danceability": 0.73, "energy": 0.57, "valence": 0.15, "tempo": 150.0, "acousticness": 0.40, "instrumentalness": 0.00, "speechiness": 0.06, "loudness": -6.9, "popularity": 85, "curated": False},
    {"id": "sp_68", "title": "Bailando", "artist": "Enrique Iglesias", "genre": "Latin Pop", "year": 2014,
     "danceability": 0.73, "energy": 0.81, "valence": 0.89, "tempo": 91.0, "acousticness": 0.05, "instrumentalness": 0.00, "speechiness": 0.09, "loudness": -5.4, "popularity": 81, "curated": False},
    {"id": "sp_69", "title": "Mi Gente", "artist": "J Balvin & Willy William", "genre": "Reggaeton / Dance", "year": 2017,
     "danceability": 0.76, "energy": 0.70, "valence": 0.47, "tempo": 105.0, "acousticness": 0.01, "instrumentalness": 0.00, "speechiness": 0.09, "loudness": -6.3, "popularity": 82, "curated": False},
    {"id": "sp_70", "title": "Havana", "artist": "Camila Cabello ft. Young Thug", "genre": "Latin Pop", "year": 2017,
     "danceability": 0.77, "energy": 0.52, "valence": 0.39, "tempo": 105.0, "acousticness": 0.18, "instrumentalness": 0.00, "speechiness": 0.03, "loudness": -4.3, "popularity": 85, "curated": False},

    # --- Additional Diverse Benchmark Tracks (filling to ~100) ---
    {"id": "sp_71", "title": "Glimpse of Us", "artist": "Joji", "genre": "Lo-Fi / Sadcore", "year": 2022,
     "danceability": 0.44, "energy": 0.32, "valence": 0.27, "tempo": 170.0, "acousticness": 0.89, "instrumentalness": 0.00, "speechiness": 0.05, "loudness": -9.3, "popularity": 87, "curated": False},
    {"id": "sp_72", "title": "Die With A Smile", "artist": "Lady Gaga & Bruno Mars", "genre": "Pop Soul / Ballad", "year": 2024,
     "danceability": 0.54, "energy": 0.55, "valence": 0.53, "tempo": 105.0, "acousticness": 0.32, "instrumentalness": 0.00, "speechiness": 0.03, "loudness": -6.5, "popularity": 97, "curated": False},
    {"id": "sp_73", "title": "Espresso", "artist": "Sabrina Carpenter", "genre": "Nu-Disco / Pop", "year": 2024,
     "danceability": 0.70, "energy": 0.76, "valence": 0.69, "tempo": 104.0, "acousticness": 0.11, "instrumentalness": 0.00, "speechiness": 0.03, "loudness": -5.5, "popularity": 98, "curated": False},
    {"id": "sp_74", "title": "Birds of a Feather", "artist": "Billie Eilish", "genre": "Indie Pop", "year": 2024,
     "danceability": 0.75, "energy": 0.51, "valence": 0.44, "tempo": 105.0, "acousticness": 0.20, "instrumentalness": 0.05, "speechiness": 0.04, "loudness": -10.2, "popularity": 96, "curated": False},
    {"id": "sp_75", "title": "Not Like Us", "artist": "Kendrick Lamar", "genre": "West Coast Hip-Hop", "year": 2024,
     "danceability": 0.89, "energy": 0.47, "valence": 0.78, "tempo": 101.0, "acousticness": 0.01, "instrumentalness": 0.00, "speechiness": 0.32, "loudness": -7.0, "popularity": 94, "curated": False},
    {"id": "sp_76", "title": "Good Luck, Babe!", "artist": "Chappell Roan", "genre": "Synth-Pop", "year": 2024,
     "danceability": 0.71, "energy": 0.73, "valence": 0.78, "tempo": 117.0, "acousticness": 0.04, "instrumentalness": 0.00, "speechiness": 0.03, "loudness": -5.9, "popularity": 93, "curated": False},
    {"id": "sp_77", "title": "Too Sweet", "artist": "Hozier", "genre": "Indie Rock / Blues", "year": 2024,
     "danceability": 0.74, "energy": 0.62, "valence": 0.83, "tempo": 117.0, "acousticness": 0.03, "instrumentalness": 0.00, "speechiness": 0.04, "loudness": -6.3, "popularity": 90, "curated": False},
    {"id": "sp_78", "title": "Take Me to Church", "artist": "Hozier", "genre": "Indie Rock / Soul", "year": 2013,
     "danceability": 0.57, "energy": 0.66, "valence": 0.44, "tempo": 129.0, "acousticness": 0.63, "instrumentalness": 0.00, "speechiness": 0.05, "loudness": -5.3, "popularity": 87, "curated": False},
    {"id": "sp_79", "title": "Counting Stars", "artist": "OneRepublic", "genre": "Pop-Rock / Folk-Pop", "year": 2013,
     "danceability": 0.66, "energy": 0.70, "valence": 0.48, "tempo": 122.0, "acousticness": 0.06, "instrumentalness": 0.00, "speechiness": 0.04, "loudness": -5.0, "popularity": 85, "curated": False},
    {"id": "sp_80", "title": "Riptide", "artist": "Vance Joy", "genre": "Indie Folk / Pop", "year": 2013,
     "danceability": 0.48, "energy": 0.73, "valence": 0.51, "tempo": 102.0, "acousticness": 0.43, "instrumentalness": 0.00, "speechiness": 0.09, "loudness": -6.9, "popularity": 84, "curated": False},
    {"id": "sp_81", "title": "Stressed Out", "artist": "Twenty One Pilots", "genre": "Rap Rock / Alt-Pop", "year": 2015,
     "danceability": 0.73, "energy": 0.64, "valence": 0.65, "tempo": 170.0, "acousticness": 0.05, "instrumentalness": 0.00, "speechiness": 0.14, "loudness": -5.7, "popularity": 86, "curated": False},
    {"id": "sp_82", "title": "Heat Waves", "artist": "Glass Animals", "genre": "Psychedelic Pop / Indie", "year": 2020,
     "danceability": 0.76, "energy": 0.53, "valence": 0.53, "tempo": 81.0, "acousticness": 0.44, "instrumentalness": 0.00, "speechiness": 0.09, "loudness": -6.9, "popularity": 88, "curated": False},
    {"id": "sp_83", "title": "Industry Baby", "artist": "Lil Nas X & Jack Harlow", "genre": "Pop Rap", "year": 2021,
     "danceability": 0.74, "energy": 0.70, "valence": 0.89, "tempo": 150.0, "acousticness": 0.02, "instrumentalness": 0.00, "speechiness": 0.06, "loudness": -7.4, "popularity": 85, "curated": False},
    {"id": "sp_84", "title": "Can't Hold Us", "artist": "Macklemore & Ryan Lewis", "genre": "Hip-Hop / Dance", "year": 2011,
     "danceability": 0.64, "energy": 0.92, "valence": 0.85, "tempo": 146.0, "acousticness": 0.03, "instrumentalness": 0.00, "speechiness": 0.08, "loudness": -4.5, "popularity": 86, "curated": False},
    {"id": "sp_85", "title": "Wake Me Up When September Ends", "artist": "Green Day", "genre": "Punk Rock / Alt-Rock", "year": 2004,
     "danceability": 0.49, "energy": 0.74, "valence": 0.35, "tempo": 105.0, "acousticness": 0.03, "instrumentalness": 0.00, "speechiness": 0.03, "loudness": -4.2, "popularity": 84, "curated": False},
    {"id": "sp_86", "title": "Seven Nation Army", "artist": "The White Stripes", "genre": "Garage Rock / Blues Rock", "year": 2003,
     "danceability": 0.74, "energy": 0.46, "valence": 0.30, "tempo": 124.0, "acousticness": 0.01, "instrumentalness": 0.45, "speechiness": 0.08, "loudness": -7.6, "popularity": 85, "curated": False},
    {"id": "sp_87", "title": "Sweet Child O' Mine", "artist": "Guns N' Roses", "genre": "Hard Rock", "year": 1987,
     "danceability": 0.45, "energy": 0.96, "valence": 0.62, "tempo": 125.0, "acousticness": 0.09, "instrumentalness": 0.01, "speechiness": 0.06, "loudness": -7.7, "popularity": 86, "curated": False},
    {"id": "sp_88", "title": "Chandelier", "artist": "Sia", "genre": "Electropop", "year": 2014,
     "danceability": 0.40, "energy": 0.79, "valence": 0.57, "tempo": 117.0, "acousticness": 0.02, "instrumentalness": 0.00, "speechiness": 0.05, "loudness": -3.0, "popularity": 83, "curated": False},
    {"id": "sp_89", "title": "Rolling in the Deep", "artist": "Adele", "genre": "Soul / Pop Rock", "year": 2010,
     "danceability": 0.73, "energy": 0.76, "valence": 0.52, "tempo": 105.0, "acousticness": 0.13, "instrumentalness": 0.00, "speechiness": 0.03, "loudness": -4.6, "popularity": 85, "curated": False},
    {"id": "sp_90", "title": "Thinking Out Loud", "artist": "Ed Sheeran", "genre": "Blue-Eyed Soul", "year": 2014,
     "danceability": 0.78, "energy": 0.45, "valence": 0.59, "tempo": 79.0, "acousticness": 0.47, "instrumentalness": 0.00, "speechiness": 0.03, "loudness": -6.1, "popularity": 85, "curated": False},
    {"id": "sp_91", "title": "Shallow", "artist": "Lady Gaga & Bradley Cooper", "genre": "Country / Folk Pop", "year": 2018,
     "danceability": 0.57, "energy": 0.39, "valence": 0.32, "tempo": 96.0, "acousticness": 0.37, "instrumentalness": 0.00, "speechiness": 0.03, "loudness": -6.4, "popularity": 87, "curated": False},
    {"id": "sp_92", "title": "Wonderwall", "artist": "Oasis", "genre": "Britpop / Rock", "year": 1995,
     "danceability": 0.40, "energy": 0.89, "valence": 0.47, "tempo": 174.0, "acousticness": 0.0002, "instrumentalness": 0.00, "speechiness": 0.04, "loudness": -4.1, "popularity": 85, "curated": False},
    {"id": "sp_93", "title": "Careless Whisper", "artist": "George Michael", "genre": "Pop / Smooth Soul", "year": 1984,
     "danceability": 0.63, "energy": 0.59, "valence": 0.72, "tempo": 153.0, "acousticness": 0.38, "instrumentalness": 0.00, "speechiness": 0.03, "loudness": -9.8, "popularity": 84, "curated": False},
    {"id": "sp_94", "title": "Billie Jean", "artist": "Michael Jackson", "genre": "Post-Disco / Dance-Pop", "year": 1982,
     "danceability": 0.92, "energy": 0.65, "valence": 0.84, "tempo": 117.0, "acousticness": 0.02, "instrumentalness": 0.00, "speechiness": 0.08, "loudness": -3.1, "popularity": 88, "curated": False},
    {"id": "sp_95", "title": "Beat It", "artist": "Michael Jackson", "genre": "Dance-Rock / Pop", "year": 1982,
     "danceability": 0.81, "energy": 0.92, "valence": 0.92, "tempo": 139.0, "acousticness": 0.17, "instrumentalness": 0.00, "speechiness": 0.07, "loudness": -4.2, "popularity": 84, "curated": False},
    {"id": "sp_96", "title": "Take On Me", "artist": "a-ha", "genre": "Synth-Pop", "year": 1985,
     "danceability": 0.57, "energy": 0.90, "valence": 0.86, "tempo": 169.0, "acousticness": 0.02, "instrumentalness": 0.00, "speechiness": 0.05, "loudness": -3.8, "popularity": 85, "curated": False},
    {"id": "sp_97", "title": "Dreams", "artist": "Fleetwood Mac", "genre": "Soft Rock / Pop", "year": 1977,
     "danceability": 0.83, "energy": 0.49, "valence": 0.79, "tempo": 120.0, "acousticness": 0.06, "instrumentalness": 0.005, "speechiness": 0.03, "loudness": -9.7, "popularity": 88, "curated": False},
    {"id": "sp_98", "title": "Stairway to Heaven", "artist": "Led Zeppelin", "genre": "Hard Rock / Folk Rock", "year": 1971,
     "danceability": 0.34, "energy": 0.34, "valence": 0.20, "tempo": 82.0, "acousticness": 0.58, "instrumentalness": 0.004, "speechiness": 0.03, "loudness": -12.2, "popularity": 82, "curated": False},
    {"id": "sp_99", "title": "Experience", "artist": "Ludovico Einaudi", "genre": "Contemporary Classical", "year": 2013,
     "danceability": 0.42, "energy": 0.37, "valence": 0.16, "tempo": 134.0, "acousticness": 0.97, "instrumentalness": 0.95, "speechiness": 0.04, "loudness": -15.8, "popularity": 82, "curated": False},
    {"id": "sp_100", "title": "Midnight City", "artist": "M83", "genre": "Synth-Pop / Dream Pop", "year": 2011,
     "danceability": 0.52, "energy": 0.78, "valence": 0.33, "tempo": 105.0, "acousticness": 0.0003, "instrumentalness": 0.01, "speechiness": 0.04, "loudness": -5.6, "popularity": 84, "curated": False}
]

def main():
    # Feature columns used for clustering and recommendation
    feature_cols = [
        "danceability", "energy", "valence", "tempo",
        "acousticness", "instrumentalness", "speechiness", "loudness"
    ]

    # Convert to DataFrame
    df = pd.DataFrame(TRACKS)
    X = df[feature_cols].values

    # Normalize features using StandardScaler (zero mean, unit variance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute 2D PCA projection for canvas visualization
    pca_2d = PCA(n_components=2, random_state=42)
    X_pca2d = pca_2d.fit_transform(X_scaled)

    # Add 2D PCA coordinates and standardized vector to track metadata
    for i, t in enumerate(TRACKS):
        t["pca_x"] = round(float(X_pca2d[i, 0]), 4)
        t["pca_y"] = round(float(X_pca2d[i, 1]), 4)
        t["feature_vector"] = [round(float(v), 4) for v in X_scaled[i]]

    # Save to JSON
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "tracks": TRACKS,
            "feature_columns": feature_cols,
            "pca_explained_variance_ratio": [round(float(v), 4) for v in pca_2d.explained_variance_ratio_],
            "scaler_mean": [round(float(m), 4) for m in scaler.mean_],
            "scaler_scale": [round(float(s), 4) for s in scaler.scale_]
        }, f, indent=2, ensure_ascii=False)

    print(f"Dataset successfully compiled! {len(TRACKS)} tracks written to {OUTPUT_PATH}")
    print(f"2D PCA Explained Variance: {pca_2d.explained_variance_ratio_} (Total {sum(pca_2d.explained_variance_ratio_):.2%})")

if __name__ == "__main__":
    main()

