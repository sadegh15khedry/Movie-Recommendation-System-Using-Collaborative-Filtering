import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import numpy as np

#loading the datasets
movie_df = pd.read_csv('drive/MyDrive/movies.csv')
tag_df = pd.read_csv('drive/MyDrive/tags.csv')
rating_df = pd.read_csv('drive/MyDrive/ratings.csv')

#selecting proper column
tag_df = tag_df[["userId", "movieId", "tag"]]
rating_df = rating_df[["userId", "movieId", "rating"]]
movie_df = movie_df[["title","movieId", "genres"]]

# #Exploratory data analysis
print("movie shape: "+ str(movie_df.shape))
print("tag shape: "+str(tag_df.shape))
print("tag shape: "+str(rating_df.shape))
print("\n\n")

print(movie_df.head())
print(tag_df.head())
print(rating_df.head())
print("\n\n")

print(movie_df.describe())
print(tag_df.describe())
print(rating_df.describe())
print("\n\n")

print("info")
print("movie info: "+str(movie_df.info()))
print("tag info: "+str(tag_df.info()))
print("rating info: "+str(rating_df.info()))
print("\n\n")

print("missing values")
print("movie missing values: "+str(movie_df.isnull().sum()))
print("tag missing values: "+str(tag_df.isnull().sum()))
print("rating missing values: "+str(rating_df.isnull().sum()))
print("\n\n")

print("duplicates")
print("movie duplicates: "+str(movie_df.duplicated().sum()))
print("tag duplicates: "+str(tag_df.duplicated().sum()))
print("rating duplicates: "+str(rating_df.duplicated().sum()))
print("\n\n")

rating_df["rating"].value_counts().plot(kind='bar')


print(tag_df.corr())
print(rating_df.corr())


#Finding and Handling missing values
movie_df = movie_df.dropna()
tag_df = tag_df.dropna()
rating_df = rating_df.dropna()


#joining movies and rating dataset on movieId
df = pd.merge(rating_df, movie_df, on="movieId", how="inner")
print(df.head())
print(df.shape)

#removing movies with less than 100 reviews due to the lack of processing power
agg_df = df.groupby('movieId').agg(mean_rating= ('rating', 'mean'), number_of_rating=('rating', 'count')).reset_index()
agg_df = agg_df[agg_df['number_of_rating']>100]
print(agg_df.shape)
print(agg_df.value_counts)

# sns.jointplot(x='mean_rating', y='number_of_rating', data=agg_df)

df_gt100 = pd.merge(df, agg_df, on='movieId', how='inner')
print(df_gt100.info)

#Creating user-movie matrix
user_movie_matrix = df_gt100.pivot_table(index='userId', columns='title', values='rating')
print(user_movie_matrix.head())
print(user_movie_matrix.shape)


#Normalizeing user item rating
matrix_norm = user_movie_matrix.subtract(user_movie_matrix.mean(axis=1), axis='rows')
print(matrix_norm.head())

#finding similar users
user_similarity = cosine_similarity(matrix_norm.fillna(0))
print(user_similarity.shape)

#finding similar movies
movie_similarity = cosine_similarity(matrix_norm.fillna(0))
print(movie_similarity.shape)

#picking a users based on a userId
picked_userId = 1

#converting the data back into a dataframe
movie_similarity = pd.DataFrame(StandardScaler().fit_transform(movie_similarity))
user_similarity = pd.DataFrame(StandardScaler().fit_transform(user_similarity))

#removing the selected user from the matrix
user_similarity.drop(index=picked_userId, inplace=True)
print(user_similarity.shape)

#setting up variables
number_of_simlar_users = 10
user_similarity_threshold = 0.3

#finding similar users
similar_users = user_similarity[user_similarity[picked_userId]>user_similarity_threshold].index
print("similar_users")
print(similar_users)

#finding the movies watched by the user
picked_user_watched = matrix_norm[matrix_norm.index==picked_userId].dropna(axis=1, how='all')
print(picked_user_watched.shape)

#keeping movies watched by similar users
similar_users_movies = matrix_norm[matrix_norm.index.isin(similar_users)].dropna(axis=1, how='all')
print(similar_users_movies.shape)

#removing the wwatched movies from the list
similar_users_movies.drop(picked_user_watched.columns, axis=1, inplace=True, errors='ignore')
print(similar_users_movies.shape)

#calculating item scores
item_score = {}
for i in similar_users_movies.columns:
  movie_rating = similar_users_movies[i]
  total = 0
  count = 0
  for u in similar_users:
    if u in movie_rating.index and pd.notna(movie_rating.loc[u]):
      score = user_similarity.loc[u, picked_userId] * movie_rating.loc[u]
      total += score
      count += 1
  item_score[i] = total/count

print(item_score)


#sorting the scores and printing the answer
ranked_item_score = pd.DataFrame(item_score.items(), columns=['movieId', 'movie_score']).sort_values(by='movie_score', ascending=False)
answer =ranked_item_score.head(number_of_simlar_users) 
print(answer)





