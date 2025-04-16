from flask import Flask, render_template, request
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

app = Flask(__name__)

# Load data
anime_df = pd.read_csv('anime.csv')
rating_df = pd.read_csv('rating.csv')

# Remove -1 ratings (indicates "not rated")
rating_df = rating_df[rating_df['rating'] != -1]

# Prepare dataset for Surprise
reader = Reader(rating_scale=(1, 10))
data = Dataset.load_from_df(rating_df[['user_id', 'anime_id', 'rating']], reader)

# Train SVD model
trainset, testset = train_test_split(data, test_size=0.2)
model = SVD()
model.fit(trainset)

# Create anime ID to name mapping
anime_dict = anime_df.set_index('anime_id')['name'].to_dict()

# Recommendation function
def recommend_for_user(user_id, n=5):
    user_rated = rating_df[rating_df['user_id'] == user_id]['anime_id'].tolist()
    all_anime_ids = anime_df['anime_id'].unique()
    unseen_anime_ids = [aid for aid in all_anime_ids if aid not in user_rated]

    predictions = [(aid, model.predict(user_id, aid).est) for aid in unseen_anime_ids]
    top_n = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]
    
    recommendations = [(anime_dict.get(aid, "Unknown"), round(score, 2)) for aid, score in top_n]
    return recommendations

@app.route('/')
def index():
    return render_template('user_input.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    recommendations = recommend_for_user(user_id)
    return render_template('user_recommendations.html', recommendations=recommendations, user_id=user_id)

if __name__ == '__main__':
    app.run(debug=True)
