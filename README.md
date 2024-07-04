# Movie Recommendation System using Collaborative Filtering

This repository contains Python code for building a movie recommendation system using collaborative filtering techniques. Below is a breakdown of the files and functionalities included:

## Files Included

- `recommendation_system.py`: Python script containing the implementation of the recommendation system.


## Prerequisites

- Python 3.x
- Required libraries: pandas, numpy, sklearn, matplotlib, seaborn

## Getting Started


1. Clone this repository:
   ```bash
   git clone https://github.com/sadegh15khedry/MovieRecommendationSystem.git
   cd Movie-Recommendation-System-Using-Collaborative-Filtering
   ```

2. Install the required libraries using the environment.yml file using conda:
   ```bash
   conda env create -f environment.yml
   ```

3. Download the movieLens datasets (`movies.csv`, `tags.csv`, `ratings.csv`) and update the path to them in the code.

4. Run the `recommendation_system.ipynb` notebook to generate movie recommendations:


## Code Description

### 1. Data Loading and Preprocessing

- Load the datasets (`movies.csv`, `tags.csv`, `ratings.csv`) using pandas.
- Select relevant columns (`tag_df`, `rating_df`, `movie_df`) for further analysis.
- Perform exploratory data analysis (EDA) to understand data shapes, missing values, duplicates, and basic statistics.

### 2. Data Aggregation

- Merge `rating_df` and `movie_df` on `movieId` to create a combined DataFrame (`df`).
- Aggregate ratings to find movies with more than 100 ratings (`agg_df`).
- Merge `df` with `agg_df` to filter out less popular movies (`df_gt100`).

### 3. User-Movie Matrix

- Create a user-movie matrix (`user_movie_matrix`) using `pivot_table`, where rows represent users, columns represent movies, and values represent ratings.

### 4. Normalization and Similarity Calculation

- Normalize `user_movie_matrix` (`matrix_norm`) by subtracting the mean rating of each user.
- Calculate cosine similarities (`user_similarity` and `movie_similarity`) based on `matrix_norm` to find similar users and movies.

### 5. Recommendation Process

- Select a user (`picked_userId`) and set up variables (`number_of_simlar_users`, `user_similarity_threshold`).
- Find similar users (`similar_users`) based on a similarity threshold.
- Identify movies watched by the selected user (`picked_user_watched`) and similar users (`similar_users_movies`).
- Calculate item scores (`item_score`) based on weighted sums of ratings from similar users.
- Sort and print top recommended movies (`ranked_item_score`) based on their scores.

## Results and Outputs

- The script outputs top recommended movies for a selected user (`picked_userId`) based on collaborative filtering.
- Evaluation metrics (e.g., precision, recall) and visualizations (e.g., heatmap of similarity matrices) can be added for performance analysis.

## Further Improvements

- Implement evaluation metrics to quantify the performance of the recommendation system.
- Optimize code efficiency for larger datasets and real-time recommendations.
- Incorporate content-based filtering or hybrid approaches for improved recommendation accuracy.

## Author

- Sadegh Khedry
  
## License

This project is licensed under the Apache-2.0 License - see the LICENSE.md file for details.
