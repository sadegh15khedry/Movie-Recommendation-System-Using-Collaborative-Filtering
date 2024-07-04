import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from utils import join_dataset



def remove_moveies_with_less_number_of_ratings(df, threshold):
    agg_df = df.groupby('movieId').agg(mean_rating= ('rating', 'mean'), number_of_rating=('rating', 'count')).reset_index()
    agg_df = agg_df[agg_df['number_of_rating']>threshold]
    return agg_df

def get_similar_results_using_collaborative_filtering(movie_df, rating_df,ratings_count_threshold):
    #joining movies and rating dataset on movieId
    df = join_dataset(movie_df, rating_df)

    #removing movies with less than threshhold reviews due to the lack of processing power
    agg_df = remove_moveies_with_less_number_of_ratings(df, ratings_count_threshold)

    df_gt100 = pd.merge(df, agg_df, on='movieId', how='inner')
    print(df_gt100.info)
    
    #Creating user-movie matrix
    user_movie_matrix = df_gt100.pivot_table(index='userId', columns='title', values='rating')

    #Normalizeing user item rating
    matrix_norm = user_movie_matrix.subtract(user_movie_matrix.mean(axis=1), axis='rows')

    #finding similar users
    user_similarity = cosine_similarity(matrix_norm.fillna(0))


    #finding similar movies
    movie_similarity = cosine_similarity(matrix_norm.fillna(0))

    #picking a users based on a userId
    picked_userId = 1

    #converting the data back into a dataframe
    movie_similarity = pd.DataFrame(StandardScaler().fit_transform(movie_similarity))
    user_similarity = pd.DataFrame(StandardScaler().fit_transform(user_similarity))

    #removing the selected user from the matrix
    user_similarity.drop(index=picked_userId, inplace=True)

    #setting up variables
    number_of_simlar_users = 10
    user_similarity_threshold = 0.3

    #finding similar users
    similar_users = user_similarity[user_similarity[picked_userId]>user_similarity_threshold].index

    #finding the movies watched by the user
    picked_user_watched = matrix_norm[matrix_norm.index==picked_userId].dropna(axis=1, how='all')
    print(picked_user_watched.shape)

    #keeping movies watched by similar users
    similar_users_movies = matrix_norm[matrix_norm.index.isin(similar_users)].dropna(axis=1, how='all')

    #removing the wwatched movies from the list
    similar_users_movies.drop(picked_user_watched.columns, axis=1, inplace=True, errors='ignore')

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
    return answer

