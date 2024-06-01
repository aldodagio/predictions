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

    # Load and preprocess the training data from the PostgreSQL database
    train_query = """SELECT concat(first_name, ' ', last_name) as player_name, total_points, passing_attempts, passing_completions, passing_yards, passing_touchdowns, interceptions,
    receptions, receiving_yards, receiving_touchdowns, rushing_attempts, rushing_yards, rushing_touchdowns FROM player
    inner join stats s on player.id = s.player_id
    inner join game g on g.game_id = s.game_id
    inner join passing p on p.pass_id = s.pass_id
    inner join receiving r on s.reception_id = r.reception_id
    inner join rushing r2 on r2.rush_id = s.rush_id
    WHERE position = 'Quarterback' AND season_id != 14"""
    train_data = pd.read_sql(train_query, engine)
    X = train_data.drop(columns=['total_points', 'player_name'])
    y = train_data['total_points']
    player_names = train_data['player_name']
    # Split the data
    # Split the data
    X_train, X_test, y_train, y_test, player_train, player_test = train_test_split(X, y, player_names, test_size=0.2,
                                                                                   random_state=42)

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
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Predictions
    y_pred = best_model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    print(f'Mean Absolute Error: {mae}')
    print(f'Mean Squared Error: {mse}')

    # Combine predictions with player names
    predictions_df = pd.DataFrame({
        'player_name': player_test,
        'predicted_stats': y_pred
    })

    print(predictions_df.head())

    # Create a DataFrame for submission
    submission_df = pd.DataFrame({'predicted_stats': predictions_df['predicted_stats'], 'player_name': predictions_df['player_name']})

    # Save the submission DataFrame to the PostgreSQL database
    submission_df.to_sql('submission_table', engine, if_exists='replace', index=False)

    #y_train = train_data['player_name'].values # where total_points is the label
    #X_train = train_data.iloc[:, 1:].values

    # Encode player names as integer labels
    #label_encoder = LabelEncoder()
    #y_train_encoded = label_encoder.fit_transform(y_train)

    # Split the data into training and validation sets
    #X_train, X_val, y_train_encoded, y_val_encoded = train_test_split(X_train, y_train_encoded, test_size=0.2,
    #                                                                  random_state=42)

    # Standardize/normalize the features
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train)
    #X_val = scaler.transform(X_val)

    # Define the model
    #num_classes = len(label_encoder.classes_)
    #model = Sequential([
    #    Dense(units=64, activation="relu", input_shape=(X_train.shape[1],)),
    #    Dense(units=32, activation="relu"),
    #    Dense(units=num_classes, activation="softmax")
    #])

    # Compile the model with a classification loss function
    #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='sparse_categorical_crossentropy',
    #              metrics=['accuracy'])

    # Fit the model
    #model.fit(X_train, y_train_encoded, epochs=100, validation_data=(X_val, y_val_encoded))

    # Load and preprocess the test data from the PostgreSQL database
    #test_query = """SELECT concat(first_name, ' ', last_name) as player_name, SUM(total_points) as total_points,
#SUM(passing_attempts), SUM(passing_completions), SUM(passing_yards), SUM(passing_touchdowns),
#SUM(interceptions), SUM(receptions), SUM(receiving_yards), SUM(receiving_touchdowns),
#SUM(rushing_attempts), SUM(rushing_yards), SUM(rushing_touchdowns) FROM player
#     inner join stats s on player.id = s.player_id
#     inner join game g on g.game_id = s.game_id
#     inner join passing p on p.pass_id = s.pass_id
#     inner join receiving r on s.reception_id = r.reception_id
#     inner join rushing r2 on r2.rush_id = s.rush_id
# WHERE position = 'Wide Receiver' and g.season_id = 14
# group by last_name, first_name
# order by sum(total_points) desc"""
#     test_data = pd.read_sql(test_query, engine)
#     X_test = scaler.transform(test_data.iloc[:, 1:].values)
#
#     # Make predictions on the test data
#     predictions_encoded = model.predict(X_test)
#     predictions = np.argmax(predictions_encoded, axis=1)
#     predicted_player_names = label_encoder.inverse_transform(predictions)

    # Create a DataFrame for submission
    #submission_df = pd.DataFrame({'total_points': test_data['total_points'], 'label': predicted_player_names.flatten()})

    # Save the submission DataFrame to the PostgreSQL database
    #submission_df.to_sql('submission_table', engine, if_exists='replace', index=False)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
