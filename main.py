import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sqlalchemy import create_engine


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Database connection setup
    username = 'postgres'
    password = 'root'
    host_url = 'localhost'
    port = 5432
    database_name = 'postgres'
    app_schema = 'fantasyfootball'
    connection_string = f"postgresql+psycopg2://{username}:{password}@{host_url}:{port}/{database_name}"
    engine = create_engine(connection_string,
                                    connect_args={"options": f"-csearch_path={app_schema}"})

    # Load the data for the most recent season
    recent_season_query = """
    SELECT
        concat(first_name, ' ', last_name) as player_name,
        g.season_id,
        SUM(passing_attempts) as total_pass_atts,
        SUM(passing_completions) as total_pass_comps,
        SUM(passing_yards) as total_pass_yards,
        SUM(passing_touchdowns) as total_pass_tds,
        SUM(rushing_attempts) as total_rush_atts,
        SUM(rushing_yards) as total_rush_yards,
        SUM(rushing_touchdowns) as total_rush_tds,
        SUM(receptions) as total_recs,
        SUM(receiving_yards) as total_rec_yards,
        SUM(receiving_touchdowns) as total_rec_tds,
        SUM(fumbles) as total_fumbles,
        SUM(interceptions) as total_ints
    FROM player
        INNER JOIN stats s on player.id = s.player_id
        INNER JOIN game g on g.game_id = s.game_id
        INNER JOIN passing p on p.pass_id = s.pass_id
        INNER JOIN receiving r on s.reception_id = r.reception_id
        INNER JOIN rushing r2 on r2.rush_id = s.rush_id
    WHERE position = 'Running Back'
    AND g.season_id = 15
    GROUP BY first_name, last_name, player.id, g.season_id
    """
    # Load and preprocess the training data from the PostgreSQL database
    train_query = """
    SELECT
        concat(first_name, ' ', last_name) as player_name,
        g.season_id,
        SUM(total_points) as season_points,
        SUM(passing_attempts) as total_pass_atts,
        SUM(passing_completions) as total_pass_comps,
        SUM(passing_yards) as total_pass_yards,
        SUM(passing_touchdowns) as total_pass_tds,
        SUM(rushing_attempts) as total_rush_atts,
        SUM(rushing_yards) as total_rush_yards,
        SUM(rushing_touchdowns) as total_rush_tds,
        SUM(receptions) as total_recs,
        SUM(receiving_yards) as total_rec_yards,
        SUM(receiving_touchdowns) as total_rec_tds,
        SUM(fumbles) as total_fumbles,
        SUM(interceptions) as total_ints
    FROM player
        INNER JOIN stats s on player.id = s.player_id
        INNER JOIN game g on g.game_id = s.game_id
        INNER JOIN passing p on p.pass_id = s.pass_id
        INNER JOIN receiving r on s.reception_id = r.reception_id
        INNER JOIN rushing r2 on r2.rush_id = s.rush_id
    WHERE position = 'Running Back'
    AND g.season_id != 15
    GROUP BY first_name, last_name, player.id, g.season_id
    ORDER BY season_points DESC
    """
    train_data = pd.read_sql(train_query, engine)

    # Assign weights to each season
    season_weights = {
        1: 0.1,  # 2010
        2: 0.2,  # 2011
        3: 0.3,  # 2012
        4: 0.4,  # 2013
        5: 0.5,  # 2014
        6: 0.6,  # 2015
        7: 0.7,  # 2016
        8: 0.8,  # 2017
        9: 0.9,  # 2018
        10: 1.0, # 2019
        11: 1.1,  # 2020
        12: 1.2,  # 2021
        13: 1.3, # 2022
        14: 1.4,  # 2023
        15: 1.5   # 2024
    }
    # Preprocess the training data
    train_data['weight'] = train_data['season_id'].apply(lambda x: season_weights.get(x, 1.0))

    # Preprocess the training data
    X_train = train_data.drop(columns=['season_points', 'player_name', 'season_id', 'weight'])
    y_train = train_data['season_points']
    sample_weight = train_data['weight']
    # Extract player names from the recent season data
    player_names_recent = recent_season_data['player_name']
    X_recent = recent_season_data.drop(columns=['player_name', 'season_id'])

    # Model: Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    # Hyperparameter tuning using Grid Search
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train, sample_weight=sample_weight)

    # Best model
    best_model = grid_search.best_estimator_

    # Predictions for the upcoming season
    y_pred_recent = best_model.predict(X_recent)

    # Combine predictions with player names
    predictions_df = pd.DataFrame({
        'player_name': player_names_recent,
        'predicted_stats': y_pred_recent
    })

    print(predictions_df.head())

    # Save the predictions DataFrame to the PostgreSQL database
    submission_df = pd.DataFrame(
        {'predicted_stats': predictions_df['predicted_stats'], 'player_name': predictions_df['player_name']})
    submission_df.to_sql('running_back_predictions_2025', engine, if_exists='replace', index=False)

