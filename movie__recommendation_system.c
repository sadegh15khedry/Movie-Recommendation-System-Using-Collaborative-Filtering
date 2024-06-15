import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats

#loading the datasets
movie_df = pd.read_csv('drive/MyDrive/movies.csv')
tag_df = pd.read_csv('drive/MyDrive/tags.csv')
rating_df = pd.read_csv('drive/MyDrive/ratings.csv')

#selecting proper column
tag_df = tag_df[["userId", "movieId", "tag"]]
rating_df = rating_df[["userId", "movieId", "rating"]]
movie_df = movie_df[["movieId", "genres"]]

#Exploratory data analysis
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

print(movie_df.corr())
print(tag_df.corr())
print(rating_df.corr())


#Finding and Handling missing values
movie_df = movie_df.dropna()
tag_df = tag_df.dropna()
rating_df = rating_df.dropna()

#Categorical Encoding
# for column in ['tag']:
#     tag_df[column] = label_encoder.fit_transform(tag_df[column])


#joining movies and rating dataset on movieId
df = pd.merge(rating_df, movie_df, on="movieId", how="inner")
print(df.head())
print(df.shape)

#removing movies with less than 100 reviews due to the lack of processing power
agg_df = df.groupby('movieId').agg(mean_rating= ('rating', 'mean'), number_of_rating=('rating', 'count')).reset_index()
agg_df = agg_df[agg_df['number_of_rating']>100]
print(agg_df.shape)
print(agg_df.value_counts)

sns.jointplot(x='mean_rating', y='number_of_rating', data=agg_df)










