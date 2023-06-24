import pandas as pd
from sklearn.metrics.pairwise import linear_kernel, pairwise_distances
from sqlalchemy import create_engine, text
from fastapi import FastAPI
from dotenv import load_dotenv
import os
# Load environment variables from .env file
load_dotenv()
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")

db_url = f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(db_url)
app = FastAPI()
# engine = create_engine('postgresql+psycopg2://ayderbek:password@localhost:5432/netflix')

reviews = pd.read_sql_query("SELECT * FROM review", con=engine)
users = pd.read_sql_query("SELECT * FROM _user", con=engine)
video_views = pd.read_sql_query("SELECT * FROM video_view", con=engine)

# Step 2: Create user-item matrix
user_item_matrix = reviews.pivot_table(index=['user_id'], columns=['video_id'], values='rating')

# Step 3: Fill NA/NaN values
user_item_matrix.fillna(0, inplace=True)

# Step 4: Compute user-user similarity matrix
user_similarity = 1 - pairwise_distances(user_item_matrix.values, metric="cosine")
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)


# Step 5: Define a function to get top N similar users and recommend videos
# def collaborative_filtering_recommendations(user_id, N=5):
#     similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:N + 1].index
#     recommendations = user_item_matrix.loc[similar_users].mean().sort_values(ascending=False).index
#     return recommendations


# Step 6: Test the function
@app.get('/recommendations/{user_id}')
async def recommendations(user_id: int, N: int = 5):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:N + 1].index
    recommendations = user_item_matrix.loc[similar_users].mean().sort_values(ascending=False).index.tolist()
    return {"recommended_videos": recommendations}



# print(collaborative_filtering_recommendations(144))
# Step 6: Test the function
# print(content_based_recommendations('Barbie 2023'))
# df = pd.read_sql_query('SELECT * FROM video LIMIT 5', con=engine)
# print(df)
