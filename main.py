
import os
import pickle
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import httpx

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv


# =====================================================
# ENVIRONMENT
# =====================================================

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")

TMDB_BASE = "https://api.themoviedb.org/3"
TMDB_IMG_500 = "https://image.tmdb.org/t/p/w500"

if not TMDB_API_KEY:
    raise RuntimeError(
        "TMDB_API_KEY missing. Add it to your .env file."
    )


# =====================================================
# FASTAPI APP
# =====================================================

app = FastAPI(
    title="Movie Recommender API",
    version="4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# PICKLE FILE PATHS
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DF_PATH = os.path.join(BASE_DIR, "df.pkl")
INDICES_PATH = os.path.join(BASE_DIR, "indices.pkl")
TFIDF_MATRIX_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.pkl")


# =====================================================
# GLOBAL VARIABLES
# =====================================================

df: Optional[pd.DataFrame] = None

indices_obj: Any = None

tfidf_matrix: Any = None

tfidf_obj: Any = None

TITLE_TO_IDX: Optional[Dict[str, int]] = None


# =====================================================
# PYDANTIC MODELS
# =====================================================

class TMDBMovieCard(BaseModel):
    tmdb_id: int
    title: str
    poster_url: Optional[str] = None
    release_date: Optional[str] = None
    vote_average: Optional[float] = None


class TMDBMovieDetails(BaseModel):
    tmdb_id: int
    title: str
    overview: Optional[str] = None
    release_date: Optional[str] = None
    poster_url: Optional[str] = None
    backdrop_url: Optional[str] = None
    genres: List[dict] = []


class TFIDFRecItem(BaseModel):
    title: str
    score: float
    tmdb: Optional[TMDBMovieCard] = None


class SearchBundleResponse(BaseModel):
    query: str
    movie_details: TMDBMovieDetails
    tfidf_recommendations: List[TFIDFRecItem]
    genre_recommendations: List[TMDBMovieCard]


# =====================================================
# GENERAL UTILITIES
# =====================================================

def _norm_title(title: str) -> str:
    return str(title).strip().lower()


def make_img_url(path: Optional[str]) -> Optional[str]:

    if not path:
        return None

    return f"{TMDB_IMG_500}{path}"


# =====================================================
# TMDB REQUEST FUNCTION
# =====================================================

async def tmdb_get(
    path: str,
    params: Dict[str, Any]
) -> Dict[str, Any]:

    query_params = dict(params)

    query_params["api_key"] = TMDB_API_KEY

    try:

        async with httpx.AsyncClient(timeout=20) as client:

            response = await client.get(
                f"{TMDB_BASE}{path}",
                params=query_params
            )

    except httpx.RequestError as e:

        raise HTTPException(
            status_code=502,
            detail=f"TMDB request error: {type(e).__name__} | {repr(e)}"
        )

    if response.status_code != 200:

        raise HTTPException(
            status_code=502,
            detail=f"TMDB error {response.status_code}: {response.text}"
        )

    return response.json()


# =====================================================
# CONVERT TMDB RESULTS INTO MOVIE CARDS
# =====================================================

async def tmdb_cards_from_results(
    results: List[dict],
    limit: int = 20
) -> List[TMDBMovieCard]:

    output = []

    for movie in (results or [])[:limit]:

        output.append(
            TMDBMovieCard(
                tmdb_id=int(movie["id"]),
                title=movie.get("title") or movie.get("name") or "",
                poster_url=make_img_url(movie.get("poster_path")),
                release_date=movie.get("release_date"),
                vote_average=movie.get("vote_average")
            )
        )

    return output


# =====================================================
# TMDB MOVIE DETAILS
# =====================================================

async def tmdb_movie_details(
    movie_id: int
) -> TMDBMovieDetails:

    data = await tmdb_get(
        f"/movie/{movie_id}",
        {"language": "en-US"}
    )

    return TMDBMovieDetails(
        tmdb_id=int(data["id"]),
        title=data.get("title") or "",
        overview=data.get("overview"),
        release_date=data.get("release_date"),
        poster_url=make_img_url(data.get("poster_path")),
        backdrop_url=make_img_url(data.get("backdrop_path")),
        genres=data.get("genres", []) or []
    )


# =====================================================
# TMDB SEARCH
# =====================================================

async def tmdb_search_movies(
    query: str,
    page: int = 1
) -> Dict[str, Any]:

    return await tmdb_get(
        "/search/movie",
        {
            "query": query,
            "include_adult": "false",
            "language": "en-US",
            "page": page
        }
    )


async def tmdb_search_first(
    query: str
) -> Optional[dict]:

    data = await tmdb_search_movies(
        query=query,
        page=1
    )

    results = data.get("results", [])

    return results[0] if results else None


# =====================================================
# TF-IDF: BUILD TITLE INDEX
# =====================================================

def build_title_to_idx_map(
    indices: Any
) -> Dict[str, int]:

    title_to_idx = {}

    try:

        for key, value in indices.items():

            title_to_idx[_norm_title(key)] = int(value)

        return title_to_idx

    except Exception:

        raise RuntimeError(
            "indices.pkl must be a dict or pandas Series."
        )


# =====================================================
# TF-IDF: GET LOCAL MOVIE INDEX
# =====================================================

def get_local_idx_by_title(
    title: str
) -> int:

    global TITLE_TO_IDX

    if TITLE_TO_IDX is None:

        raise HTTPException(
            status_code=500,
            detail="TF-IDF index map not initialized"
        )

    key = _norm_title(title)

    if key in TITLE_TO_IDX:

        return int(TITLE_TO_IDX[key])

    raise HTTPException(
        status_code=404,
        detail=f"Title not found in local dataset: '{title}'"
    )


# =====================================================
# TF-IDF: ORIGINAL LOCAL RECOMMENDATION
# =====================================================

def tfidf_recommend_titles(
    query_title: str,
    top_n: int = 10
) -> List[Tuple[str, float]]:

    global df, tfidf_matrix

    if df is None or tfidf_matrix is None:

        raise HTTPException(
            status_code=500,
            detail="TF-IDF resources not loaded"
        )

    idx = get_local_idx_by_title(query_title)

    query_vector = tfidf_matrix[idx]

    scores = (
        tfidf_matrix @ query_vector.T
    ).toarray().ravel()

    order = np.argsort(-scores)

    output = []

    for i in order:

        if int(i) == int(idx):
            continue

        try:

            movie_title = str(
                df.iloc[int(i)]["title"]
            )

        except Exception:
            continue

        output.append(
            (movie_title, float(scores[int(i)]))
        )

        if len(output) >= top_n:
            break

    return output


# =====================================================
# TF-IDF: NEW MOVIE FALLBACK
# =====================================================

def tfidf_recommend_from_text(
    query_text: str,
    top_n: int = 10,
    exclude_title: Optional[str] = None
) -> List[Tuple[str, float]]:

    global df, tfidf_matrix, tfidf_obj

    if (
        df is None
        or tfidf_matrix is None
        or tfidf_obj is None
    ):

        raise HTTPException(
            status_code=500,
            detail="TF-IDF resources not loaded"
        )

    # Transform query using the existing fitted vectorizer
    query_vector = tfidf_obj.transform(
        [query_text]
    )

    # No matching vocabulary
    if query_vector.nnz == 0:
        return []

    # Calculate cosine similarity
    # Both vectors are L2 normalized by TfidfVectorizer
    scores = (
        tfidf_matrix @ query_vector.T
    ).toarray().ravel()

    # Sort by highest similarity
    order = np.argsort(-scores)

    recommendations = []

    for i in order:

        score = float(scores[int(i)])

        if score <= 0:
            break

        try:

            movie_title = str(
                df.iloc[int(i)]["title"]
            )

        except Exception:
            continue

        # Exclude the selected movie
        if (
            exclude_title
            and _norm_title(movie_title) == _norm_title(exclude_title)
        ):
            continue

        recommendations.append(
            (movie_title, score)
        )

        if len(recommendations) >= top_n:
            break

    return recommendations


# =====================================================
# ATTACH TMDB POSTER TO LOCAL MOVIE
# =====================================================

async def attach_tmdb_card_by_title(
    title: str
) -> Optional[TMDBMovieCard]:

    try:

        movie = await tmdb_search_first(title)

        if not movie:
            return None

        return TMDBMovieCard(
            tmdb_id=int(movie["id"]),
            title=movie.get("title") or title,
            poster_url=make_img_url(movie.get("poster_path")),
            release_date=movie.get("release_date"),
            vote_average=movie.get("vote_average")
        )

    except Exception:

        return None


# =====================================================
# STARTUP: LOAD PICKLE FILES
# =====================================================

@app.on_event("startup")
def load_pickles():

    global df
    global indices_obj
    global tfidf_matrix
    global tfidf_obj
    global TITLE_TO_IDX

    # Load dataframe
    with open(DF_PATH, "rb") as file:

        df = pickle.load(file)

    # Load indices
    with open(INDICES_PATH, "rb") as file:

        indices_obj = pickle.load(file)

    # Load TF-IDF matrix
    with open(TFIDF_MATRIX_PATH, "rb") as file:

        tfidf_matrix = pickle.load(file)

    # Load vectorizer
    with open(TFIDF_PATH, "rb") as file:

        tfidf_obj = pickle.load(file)

    # Build title index
    TITLE_TO_IDX = build_title_to_idx_map(
        indices_obj
    )

    # Validate dataframe
    if df is None or "title" not in df.columns:

        raise RuntimeError(
            "df.pkl must contain a title column."
        )

    print("TF-IDF resources loaded successfully.")
    print("Movies loaded:", len(df))
    print("TF-IDF matrix shape:", tfidf_matrix.shape)


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# =====================================================
# HOME FEED
# =====================================================

@app.get(
    "/home",
    response_model=List[TMDBMovieCard]
)
async def home(
    category: str = Query("popular"),
    limit: int = Query(24, ge=1, le=50)
):

    try:

        if category == "trending":

            data = await tmdb_get(
                "/trending/movie/day",
                {"language": "en-US"}
            )

            return await tmdb_cards_from_results(
                data.get("results", []),
                limit=limit
            )

        if category not in {
            "popular",
            "top_rated",
            "upcoming",
            "now_playing"
        }:

            raise HTTPException(
                status_code=400,
                detail="Invalid category"
            )

        data = await tmdb_get(
            f"/movie/{category}",
            {
                "language": "en-US",
                "page": 1
            }
        )

        return await tmdb_cards_from_results(
            data.get("results", []),
            limit=limit
        )

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Home route failed: {e}"
        )


# =====================================================
# TMDB SEARCH ROUTE
# =====================================================

@app.get("/tmdb/search")
async def tmdb_search(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1, le=10)
):

    return await tmdb_search_movies(
        query=query,
        page=page
    )


# =====================================================
# MOVIE DETAILS ROUTE
# =====================================================

@app.get(
    "/movie/id/{tmdb_id}",
    response_model=TMDBMovieDetails
)
async def movie_details_route(
    tmdb_id: int
):

    return await tmdb_movie_details(
        tmdb_id
    )


# =====================================================
# GENRE RECOMMENDATIONS
# =====================================================

@app.get(
    "/recommend/genre",
    response_model=List[TMDBMovieCard]
)
async def recommend_genre(
    tmdb_id: int = Query(...),
    limit: int = Query(18, ge=1, le=50)
):

    details = await tmdb_movie_details(
        tmdb_id
    )

    if not details.genres:
        return []

    genre_id = details.genres[0]["id"]

    discover = await tmdb_get(
        "/discover/movie",
        {
            "with_genres": genre_id,
            "language": "en-US",
            "sort_by": "popularity.desc",
            "page": 1
        }
    )

    cards = await tmdb_cards_from_results(
        discover.get("results", []),
        limit=limit
    )

    return [
        card
        for card in cards
        if card.tmdb_id != tmdb_id
    ]


# =====================================================
# TF-IDF ONLY ROUTE
# =====================================================

@app.get("/recommend/tfidf")
async def recommend_tfidf(
    title: str = Query(..., min_length=1),
    top_n: int = Query(10, ge=1, le=50)
):

    recs = tfidf_recommend_titles(
        title,
        top_n=top_n
    )

    return [
        {
            "title": title,
            "score": score
        }
        for title, score in recs
    ]


# =====================================================
# MAIN MOVIE SEARCH + RECOMMENDATION BUNDLE
# =====================================================

@app.get(
    "/movie/search",
    response_model=SearchBundleResponse
)
async def search_bundle(
    query: str = Query(..., min_length=1),
    tfidf_top_n: int = Query(12, ge=1, le=30),
    genre_limit: int = Query(12, ge=1, le=30)
):

    # Search selected movie on TMDB
    best = await tmdb_search_first(query)

    if not best:

        raise HTTPException(
            status_code=404,
            detail=f"No TMDB movie found for query: {query}"
        )

    tmdb_id = int(best["id"])

    # Fetch movie details
    details = await tmdb_movie_details(
        tmdb_id
    )

    # =================================================
    # 1. TF-IDF RECOMMENDATIONS
    # =================================================

    tfidf_items: List[TFIDFRecItem] = []

    recs: List[Tuple[str, float]] = []

    # First try original local recommendation
    try:

        recs = tfidf_recommend_titles(
            details.title,
            top_n=tfidf_top_n
        )

    except HTTPException as e:

        if e.status_code != 404:
            raise

        recs = []

    # Fallback for movies missing from local dataset
    if not recs:

        genre_text = " ".join(
            genre.get("name", "")
            for genre in details.genres
        )

        query_text = " ".join([
            details.title,
            details.overview or "",
            genre_text
        ])

        try:

            recs = tfidf_recommend_from_text(
                query_text=query_text,
                top_n=tfidf_top_n,
                exclude_title=details.title
            )

            print(
                f"Using text-based TF-IDF fallback for: {details.title}"
            )

        except Exception as e:

            print(
                "TF-IDF fallback error:",
                repr(e)
            )

            recs = []

    # Attach TMDB movie cards
    for title, score in recs:

        card = await attach_tmdb_card_by_title(
            title
        )

        tfidf_items.append(
            TFIDFRecItem(
                title=title,
                score=score,
                tmdb=card
            )
        )

    # =================================================
    # 2. GENRE RECOMMENDATIONS
    # =================================================

    genre_recs: List[TMDBMovieCard] = []

    if details.genres:

        genre_id = details.genres[0]["id"]

        discover = await tmdb_get(
            "/discover/movie",
            {
                "with_genres": genre_id,
                "language": "en-US",
                "sort_by": "popularity.desc",
                "page": 1
            }
        )

        cards = await tmdb_cards_from_results(
            discover.get("results", []),
            limit=genre_limit
        )

        genre_recs = [
            card
            for card in cards
            if card.tmdb_id != details.tmdb_id
        ]

    # =================================================
    # 3. FINAL RESPONSE
    # =================================================

    return SearchBundleResponse(
        query=query,
        movie_details=details,
        tfidf_recommendations=tfidf_items,
        genre_recommendations=genre_recs
    )
