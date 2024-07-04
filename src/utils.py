import pandas as pd

def load_datasets(dataset_dirctory_path):
    #loading the datasets
    movie_df = pd.read_csv(dataset_dirctory_path+'movies.csv')
    tag_df = pd.read_csv(dataset_dirctory_path+'tags.csv')
    rating_df = pd.read_csv(dataset_dirctory_path+'ratings.csv')
    
    return movie_df, tag_df, rating_df


def join_dataset(movie_df, rating_df):
    df = pd.merge(rating_df, movie_df, on="movieId", how="inner")
    return df