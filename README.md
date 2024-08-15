# 🎥 Movie Recommendation System

A powerful and intelligent Movie Recommendation System that combines real-time data retrieval with advanced content analysis to provide personalized movie suggestions. This project leverages IMDb and TMDB APIs to ensure that users receive the most accurate and up-to-date movie recommendations based on their preferences.

## 🚀 Features

### 1. Real-Time Data from IMDb API
- The system pulls up-to-date movie data directly from the IMDb API based on user searches.
- Ensures that the recommendations are always fresh and relevant.

### 2. Advanced Recommendation Model
- Utilizes content-based filtering to analyze movie features like descriptions, genres, and cast.
- Employs **TF-IDF vectorization** and **cosine similarity** to match user preferences with the best possible movie recommendations.

### 3. Fetch Detailed Information via TMDB API
- After identifying top recommendations, the system fetches detailed movie information from the TMDB API.
- Provides comprehensive insights including description, rating, genres, director, and writers, enhancing the user experience.

### 4. Personalized User Experience
- The system is designed to deliver highly personalized recommendations, making it an essential tool for movie enthusiasts looking to discover their next favorite film.

## 🛠️ Technologies Used

- **Django:** A high-level Python web framework for rapid development.
- **IMDb API:** For fetching real-time movie data.
- **TMDB API:** For retrieving detailed movie information.
- **TF-IDF Vectorization:** To convert textual data into numerical features.
- **Cosine Similarity:** To compute similarity between movies based on user preferences.

## 🌟 Project Highlights

- **Real-Time Recommendations:** Always up-to-date, thanks to the IMDb API.
- **Content-Based Filtering:** Ensures recommendations are closely aligned with user preferences.
- **Detailed Movie Information:** Fetches and displays comprehensive movie data, enriching the user experience.
