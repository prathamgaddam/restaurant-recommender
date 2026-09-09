import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import torch.nn as nn
import pandas as pd
import pickle
import numpy as np

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('user_encoder.pkl', 'rb') as f:
    le_userid = pickle.load(f)

with open('business_encoder.pkl', 'rb') as f:
    le_bid = pickle.load(f)

df_philly = pd.read_csv('df_philly.csv')
restaurants = pd.read_csv('restaurants_clean.csv')

class NCF(nn.Module):
    def __init__(self, num_users, num_restaurants, embedding_dim=16):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.rest_embedding = nn.Embedding(num_restaurants, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        user_emb = self.user_embedding(x[:, 0])
        rest_emb = self.rest_embedding(x[:, 1])
        combined = torch.cat([user_emb, rest_emb], dim=1)
        out = self.dropout(self.relu(self.fc1(combined)))
        out = self.dropout(self.relu(self.fc2(out)))
        out = self.dropout(self.relu(self.fc3(out)))
        out = self.fc4(out)
        out = 1 + 4 * torch.sigmoid(out)
        return out.squeeze()

num_users = df_philly['user_idx'].nunique()
num_restaurants = df_philly['business_idx'].nunique()

model = NCF(num_users, num_restaurants)
model.load_state_dict(torch.load('best_bpr_model.pth', map_location='cpu'))
model.eval()

class RecommendRequest(BaseModel):
    user_idx: int
    top_n: int = 10

class ColdStartRequest(BaseModel):
    cuisines: List[str]
    max_price: Optional[float] = None
    top_n: int = 10

@app.get("/")
def root():
    return {"message": "Philadelphia Restaurant Recommender API"}

@app.get("/restaurants")
def get_restaurants(limit: int = 20):
    philly_biz_ids = df_philly['business_id'].unique()
    philly_restaurants = restaurants[restaurants['business_id'].isin(philly_biz_ids)]
    return {"restaurants": philly_restaurants[['business_id', 'name', 'categories', 'stars']].head(limit).to_dict('records')}

@app.get("/cuisines")
def get_cuisines():
    philly_biz_ids = df_philly['business_id'].unique()
    philly_restaurants = restaurants[restaurants['business_id'].isin(philly_biz_ids)]
    all_cuisines = philly_restaurants['categories'].dropna().str.split(', ').explode().unique().tolist()
    return {"cuisines": sorted(all_cuisines)}

@app.post("/recommend")
def recommend(request: RecommendRequest):
    if request.user_idx < 0 or request.user_idx >= num_users:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_idx = request.user_idx
    
    with torch.no_grad():
        user_tensor = torch.tensor([user_idx] * num_restaurants, dtype=torch.long)
        rest_tensor = torch.tensor(list(range(num_restaurants)), dtype=torch.long)
        predictions = model(torch.stack([user_tensor, rest_tensor], dim=1)).numpy()
    
    top_indices = predictions.argsort()[::-1][:request.top_n]
    top_business_ids = le_bid.inverse_transform(top_indices)
    top_scores = predictions[top_indices]
    
    recs = restaurants[restaurants['business_id'].isin(top_business_ids)][['business_id', 'name']]
    recs = recs.set_index('business_id')
    
    results = [
        {"name": recs.loc[bid, 'name'], "score": float(score)}
        for bid, score in zip(top_business_ids, top_scores)
        if bid in recs.index
    ]
    
    return {"recommendations": results}

@app.post("/recommend/coldstart")
def recommend_coldstart(request: ColdStartRequest):
    filtered = restaurants[restaurants['categories'].apply(
        lambda x: any(c in str(x) for c in request.cuisines)
    )]
    
    top = filtered.nlargest(request.top_n, 'stars')[['name', 'stars', 'categories']]
    
    return {"recommendations": top.to_dict('records')}