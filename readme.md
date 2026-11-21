# 🎬 Movie Recommendation System

A content-based movie recommendation system built with **FastAPI** (backend) and **Streamlit** (frontend), combining a locally trained **TF-IDF** model with live data from **The Movie Database (TMDB) API**.

Given a movie, the system recommends similar titles using cosine similarity over TF-IDF vectors — and gracefully falls back to on-the-fly text similarity for movies outside the training dataset (cold-start handling), so even brand-new releases get relevant recommendations.

---

## ✨ Features

- **Home feed** — Trending, Popular, Top Rated, Upcoming, and Now Playing movies, pulled live from TMDB.
- **Search** — Type a title and get live TMDB search suggestions, resolved to a details page.
- **Movie details** — Poster, backdrop, overview, genres, and release date.
- **Content-based recommendations (TF-IDF)** — Cosine similarity over a 45,000+ movie dataset (titles, overviews, genres, tags).
- **Cold-start fallback** — If a searched movie isn't in the trained dataset, its live overview + genre text is transformed through the fitted vectorizer and ranked against the existing matrix, so recommendations still work for movies released after the dataset was built.
- **Genre-based recommendations** — A secondary, TMDB-powered "more like this" rail using genre discovery.
- **Shareable routing** — The frontend uses query parameters (`?view=details&id=...`) so movie pages can be bookmarked or shared directly.

---

## 🧠 How the recommendation engine works

1. A **TF-IDF vectorizer** (`scikit-learn`) is pre-trained on a local dataset of ~45,000 movies, combining title, overview, genre, and tag text into a single feature space (`tfidf_matrix.pkl`, shape `45447 × 50000`).
2. For a known movie, its row vector is pulled from the matrix and compared against every other row via **cosine similarity** (via sparse matrix multiplication — vectors are already L2-normalized by `TfidfVectorizer`).
3. For an **unknown movie** (not in the training set), its live `overview + genres` text is passed through the *same fitted vectorizer* (`tfidf_obj.transform(...)`) to produce a comparable vector on the fly, which is then compared against the full matrix the same way.
4. Results are ranked by similarity score and returned with TMDB poster/metadata attached for each recommended title.
5. In parallel, a **genre-based recommendation list** is built using TMDB's `/discover/movie` endpoint, filtered by the primary genre of the selected movie.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI, Pydantic, httpx (async) |
| Recommendation engine | scikit-learn (TF-IDF), scipy (sparse matrices), NumPy, pandas |
| Frontend | Streamlit |
| External data | TMDB API |
| Data persistence | Pickle (`df.pkl`, `indices.pkl`, `tfidf_matrix.pkl`, `tfidf.pkl`) |

---

## 📂 Project Structure

```
movie-recommendation-system/
├── main.py              # FastAPI backend (routes, TMDB integration, TF-IDF engine)
├── app.py               # Streamlit frontend
├── df.pkl               # Movie metadata (title, overview, genres, tags, etc.)
├── indices.pkl          # Title → row-index lookup for the TF-IDF matrix
├── tfidf_matrix.pkl     # Precomputed TF-IDF sparse matrix (scipy csr_matrix)
├── tfidf.pkl            # Fitted TfidfVectorizer (used for cold-start fallback)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/movie-recommendation-system.git
cd movie-recommendation-system
```

### 2. Create a virtual environment & install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Add your TMDB API key

Create a `.env` file in the project root:

```
TMDB_API_KEY=your_tmdb_api_key_here
```

Get a free API key at [themoviedb.org](https://www.themoviedb.org/settings/api).

### 4. Add the dataset files

Place `df.pkl`, `indices.pkl`, `tfidf_matrix.pkl`, and `tfidf.pkl` in the project root, alongside `main.py`. (These are generated from a movie dataset such as [The Movies Dataset on Kaggle](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) — see Training Your Own Model below.)

### 5. Run the backend

```bash
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000` — interactive docs at `http://127.0.0.1:8000/docs`.

### 6. Run the frontend

In a separate terminal:

```bash
streamlit run app.py
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| GET | `/home` | Home feed by category (`trending`, `popular`, `top_rated`, `upcoming`, `now_playing`) |
| GET | `/tmdb/search` | Raw TMDB keyword search (multiple results) |
| GET | `/movie/id/{tmdb_id}` | Movie details by TMDB ID |
| GET | `/recommend/genre` | Genre-based recommendations for a TMDB movie ID |
| GET | `/recommend/tfidf` | TF-IDF recommendations for a known local title |
| GET | `/movie/search` | Full bundle: details + TF-IDF recs + genre recs for a search query |

---

## 🧪 Training Your Own Model

If you want to rebuild the pickle files from scratch using your own dataset:

1. Load your movie dataset into a `pandas.DataFrame` with at least a `title` column (plus `overview`, `genres`, `tags` if available).
2. Fit a `TfidfVectorizer` over the combined text fields and save it (`tfidf.pkl`).
3. Transform the full dataset into a sparse matrix and save it (`tfidf_matrix.pkl`).
4. Build a `title → row index` mapping (`pandas.Series` or `dict`) and save it (`indices.pkl`).
5. Save the DataFrame itself (`df.pkl`).

> ⚠️ Note: if your dataset has duplicate titles, the current title-lookup logic resolves to whichever duplicate was processed last when building the index map. Consider deduplicating by keeping the highest-`popularity` row per title before saving `indices.pkl`.

---

## 🔮 Possible Improvements

- Hybrid scoring: blend TF-IDF similarity with TMDB `vote_average`/`popularity`.
- Embedding-based similarity (e.g. sentence transformers) instead of / alongside TF-IDF.
- Fuzzy title matching to handle near-miss searches (punctuation, alternate titles).
- Deduplicate titles in the index map by popularity.
- Persistent caching layer (Redis) for TMDB responses to reduce API calls.
- Deploy backend + frontend (e.g. Render/Railway for FastAPI, Streamlit Community Cloud for the UI).

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙋 About

Built as a hands-on project to go beyond tutorial-level ML — covering vector space models, cold-start handling, async API design, and serving a recommendation model as an actual product rather than a notebook.