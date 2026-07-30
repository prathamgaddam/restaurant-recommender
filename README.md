# Philadelphia Restaurant Recommender

A multi-level restaurant recommendation system trained on 1M+ Yelp reviews, served via a FastAPI REST API with a content-based fallback for new users.

## Approach
- Level 1: Popularity-based (Bayesian weighted average)
- Level 2: Content-based filtering (cosine similarity on cuisine features)
- Level 3: SVD collaborative filtering with mean-centering bias fix
- Level 4: Neural Collaborative Filtering (NCF) in PyTorch

## Key Findings
-Zero-filling bias in SVD : missing ratings treated as implicit zeros caused RMSE of 4.49 and predictions outside the 1-5 scale. Fixed with mean-centering → RMSE dropped to 1.43
-Cross-city noise : training across all Yelp cities introduced noise since users never share restaurant experiences across cities. Filtering to Philadelphia increased matrix density from 0.27% to 2.12%
-RMSE vs ranking metrics : RMSE of 0.97 appeared strong but Hit Rate@10 of 6.2% revealed the model was barely personalizing. RMSE measures rating prediction accuracy, not recommendation quality
-Popularity bias : item embeddings for highly reviewed restaurants dominated over sparse user embeddings, causing the same restaurants to appear for every user regardless of taste
-Cold start problem : NCF has fixed embeddings for trained users only. Addressed with a content-based fallback endpoint in the API for new users
-Location context matters : users only share behavioral signal within the same city, making geographic filtering a prerequisite for meaningful collaborative filtering
## Results
| Metric | Score |
|--------|-------|
| RMSE | 0.97 |
| Hit Rate@10 | 6.2% |
| NDCG@10 | 1.2% |

## Tech Stack
PyTorch · FastAPI · scikit-learn · pandas · Yelp Open Dataset

## Setup
```bash
pip install -r requirements.txt
uvicorn app:app --reload
```
API docs at `http://localhost:8000/docs`

## Future Work

- Negative sampling — train the model to rank relevant restaurants above irrelevant ones, directly improving Hit Rate and NDCG
- Two Tower architecture — separate user and item networks to reduce popularity bias and enable scalable inference
- BERT4Rec — treat user interaction history as a sequence, applying self-attention to address cold start naturally
- Sentiment analysis — use review text as a richer signal than star ratings alone, surfacing hidden gems where sentiment diverges from aggregate rating