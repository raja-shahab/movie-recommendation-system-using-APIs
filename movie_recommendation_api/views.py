from django.shortcuts import render
from django.http import JsonResponse
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.views.decorators.csrf import csrf_exempt
import json

# Your API details
IMDB_API_KEY = "" # Enter your API
IMDB_BASE_URL = "https://imdb8.p.rapidapi.com/auto-complete"
IMDB_HEADERS = {
    'x-rapidapi-host': "imdb8.p.rapidapi.com",
    'x-rapidapi-key': IMDB_API_KEY
}

TMDB_API_KEY = ""  # Replace with your TMDB API key
TMDB_BASE_URL = "https://api.themoviedb.org/3"

def get_movie_data(title):
    response = requests.get(IMDB_BASE_URL, headers=IMDB_HEADERS, params={"q": title})
    if response.status_code == 200:
        return response.json().get('d', [])
    return []

def get_tmdb_data(movie_id):
    url = f"{TMDB_BASE_URL}/movie/{movie_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "append_to_response": "credits"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        return response.json()
    return {}

def home(request):
    return render(request, 'index.html')

@csrf_exempt
def recommend(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            search_movie = data.get('movie_name', '')
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON data."})

        movie_data = get_movie_data(search_movie)

        if not movie_data:
            return JsonResponse({"error": "No data found for the movie."})

        movies_list = []
        for item in movie_data:
            tmdb_data = get_tmdb_data(item.get('id', ''))

            title = item.get('l', 'N/A')
            year = item.get('y', 'N/A')
            image_url = item.get('i', {}).get('imageUrl', 'N/A')
            description = item.get('s', 'N/A')
            if not isinstance(description, str):
                description = ''

            genres = ", ".join([genre['name'] for genre in tmdb_data.get('genres', [])])
            director = ", ".join([crew['name'] for crew in tmdb_data.get('credits', {}).get('crew', []) if crew['job'] == 'Director'])
            writers = ", ".join([crew['name'] for crew in tmdb_data.get('credits', {}).get('crew', []) if crew['job'] in ['Writer', 'Screenplay']])

            movies_list.append({
                'movie_id': item.get('id', 'N/A'),
                'title': title,
                'year': year,
                'image_url': image_url,
                'description': description,
                'genres': genres,
                'director': director,
                'writers': writers
            })

        movies_df = pd.DataFrame(movies_list)
        movies_df['description'] = movies_df['description'].fillna('')
        movies_df['genres'] = movies_df['genres'].fillna('')
        movies_df['director'] = movies_df['director'].fillna('')
        movies_df['writers'] = movies_df['writers'].fillna('')

        combined_features = movies_df.apply(lambda x: f"{x['description']} {x['genres']} {x['director']} {x['writers']}", axis=1)
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(combined_features)

        try:
            idx = movies_df.index[movies_df['title'].str.lower() == search_movie.lower()].tolist()[0]
            cosine_sim = cosine_similarity(tfidf_matrix[idx:idx+1], tfidf_matrix).flatten()
            similar_indices = cosine_sim.argsort()[-7:-1][::-1]
            recommendations = movies_df.iloc[similar_indices]
        except IndexError:
            return JsonResponse({"error": "Movie not found in the dataset."})

        detailed_recommendations = []
        for _, row in recommendations.iterrows():
            tmdb_data = get_tmdb_data(row['movie_id'])
            if tmdb_data:
                detailed_recommendations.append({
                    'title': row['title'],
                    'year': row['year'],
                    'image_url': f"https://image.tmdb.org/t/p/w500{tmdb_data.get('poster_path', '')}",
                    'description': ' '.join(tmdb_data.get('overview', row['description']).split()[:53]),
                    'rating': tmdb_data.get('vote_average', 'N/A'),
                    'genres': row['genres'],
                    'director': row['director'],
                    'writers': row['writers']
                })
            else:
                detailed_recommendations.append({
                    'title': row['title'],
                    'year': row['year'],
                    'image_url': row['image_url'],
                    'description': row['description'],
                    'rating': 'N/A',
                    'genres': row['genres'],
                    'director': row['director'],
                    'writers': row['writers']
                })

        if not detailed_recommendations:
            return JsonResponse({"error": "No recommendations found."})

        return JsonResponse({"recommendations": detailed_recommendations})

    return JsonResponse({"error": "Invalid request method."})
