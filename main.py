import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# -----------------------------
# DB CONNECTION
# -----------------------------
username = 'postgres'
password = 'root'
host_url = 'localhost'
port = 5432
database_name = 'postgres'
app_schema = 'fantasyfootball'

connection_string = (
    f"postgresql+psycopg2://{username}:{password}@{host_url}:{port}/{database_name}"
)
engine = create_engine(
    connection_string,
    connect_args={"options": f"-csearch_path={app_schema}"}
)

# -----------------------------
# MODEL CONFIG
# -----------------------------
QUANTILES = [0.25, 0.50, 0.75]

# 🔑 NEW TARGET: next-season fantasy PPG
TARGET = "next_season_fantasy_ppg"

TRAIN_CUTOFF = 12
VAL_SEASON = 13
TEST_SEASON = 14
PREDICT_SEASON = 16


# -----------------------------
# RB
# -----------------------------
def rb_query():
    query = """
    WITH season_stats AS (
        SELECT
            p.id AS player_id,
            CONCAT(p.first_name, ' ', p.last_name) AS player_name,
            g.season_id,
            COUNT(DISTINCT g.game_id) AS games_played,

            SUM(r2.rushing_attempts) AS rush_atts,
            SUM(r2.rushing_yards) AS rush_yards,
            SUM(r2.rushing_touchdowns) AS rush_tds,

            SUM(rec.receptions) AS receptions,
            SUM(rec.receiving_yards) AS rec_yards,
            SUM(rec.receiving_touchdowns) AS rec_tds,

            SUM(s.total_points) AS fantasy_points
        FROM postgres.fantasyfootball.player p
        JOIN postgres.fantasyfootball.stats s ON p.id = s.player_id
        JOIN postgres.fantasyfootball.game g ON g.game_id = s.game_id
        JOIN postgres.fantasyfootball.rushing r2 ON r2.rush_id = s.rush_id
        JOIN postgres.fantasyfootball.receiving rec ON rec.reception_id = s.reception_id
        WHERE p.position = 'Running Back'
          AND g.season_id >= 6
        GROUP BY p.id, p.first_name, p.last_name, g.season_id
    ),
    base AS (
        SELECT
            prev.player_id,
            prev.player_name,
            prev.season_id AS feature_season,

            prev.games_played,
            prev.rush_atts,
            prev.rush_yards,
            prev.rush_tds,
            prev.receptions,
            prev.rec_yards,
            prev.rec_tds,

            CASE WHEN prev.games_played > 0
                 THEN prev.rush_atts::float / prev.games_played ELSE 0 END AS rush_atts_pg,

            CASE WHEN prev.games_played > 0
                 THEN prev.fantasy_points::float / prev.games_played ELSE 0 END AS fantasy_ppg,

            CASE WHEN next.games_played > 0
                 THEN next.fantasy_points::float / next.games_played
                 ELSE NULL END AS next_season_fantasy_ppg
        FROM season_stats prev
        LEFT JOIN season_stats next
          ON prev.player_id = next.player_id
         AND next.season_id = prev.season_id + 1
    )
    SELECT
        b.*,

        AVG(b.fantasy_ppg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS fantasy_ppg_2yr_avg,

        AVG(b.rush_atts_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS rush_atts_pg_2yr_avg,

        AVG(b.games_played) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS games_played_2yr_avg,

        b.fantasy_ppg -
        LAG(b.fantasy_ppg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
        ) AS fantasy_ppg_delta,

        b.rush_atts_pg -
        LAG(b.rush_atts_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
        ) AS rush_atts_pg_delta
    FROM base b;
    """
    return pd.read_sql(query, engine)


def rb_features():
    return [
        "games_played",
        "rush_atts",
        "rush_yards",
        "rush_tds",
        "receptions",
        "rec_yards",
        "rec_tds",
        "rush_atts_pg",
        "fantasy_ppg",
        "fantasy_ppg_2yr_avg",
        "rush_atts_pg_2yr_avg",
        "games_played_2yr_avg",
        "fantasy_ppg_delta",
        "rush_atts_pg_delta",
    ]


# -----------------------------
# WR / TE (identical logic)
# -----------------------------
def wr_query(position="Wide Receiver"):
    query = f"""
    WITH season_stats AS (
        SELECT
            p.id AS player_id,
            CONCAT(p.first_name, ' ', p.last_name) AS player_name,
            g.season_id,
            COUNT(DISTINCT g.game_id) AS games_played,

            SUM(rec.receptions) AS receptions,
            SUM(rec.receiving_yards) AS rec_yards,
            SUM(rec.receiving_touchdowns) AS rec_tds,

            SUM(s.total_points) AS fantasy_points
        FROM postgres.fantasyfootball.player p
        JOIN postgres.fantasyfootball.stats s ON p.id = s.player_id
        JOIN postgres.fantasyfootball.game g ON g.game_id = s.game_id
        JOIN postgres.fantasyfootball.receiving rec ON rec.reception_id = s.reception_id
        WHERE p.position = '{position}'
          AND g.season_id >= 6
        GROUP BY p.id, p.first_name, p.last_name, g.season_id
    ),
    base AS (
        SELECT
            prev.player_id,
            prev.player_name,
            prev.season_id AS feature_season,

            prev.games_played,
            prev.receptions,
            prev.rec_yards,
            prev.rec_tds,

            CASE WHEN prev.games_played > 0
                 THEN prev.receptions::float / prev.games_played ELSE 0 END AS receptions_pg,

            CASE WHEN prev.games_played > 0
                 THEN prev.fantasy_points::float / prev.games_played ELSE 0 END AS fantasy_ppg,

            CASE WHEN next.games_played > 0
                 THEN next.fantasy_points::float / next.games_played
                 ELSE NULL END AS next_season_fantasy_ppg
        FROM season_stats prev
        LEFT JOIN season_stats next
          ON prev.player_id = next.player_id
         AND next.season_id = prev.season_id + 1
    )
    SELECT
        b.*,

        AVG(b.fantasy_ppg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS fantasy_ppg_2yr_avg,

        AVG(b.receptions_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS receptions_pg_2yr_avg,

        b.fantasy_ppg -
        LAG(b.fantasy_ppg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
        ) AS fantasy_ppg_delta
    FROM base b;
    """
    return pd.read_sql(query, engine)


def wr_features():
    return [
        "games_played",
        "receptions",
        "rec_yards",
        "rec_tds",
        "receptions_pg",
        "fantasy_ppg",
        "fantasy_ppg_2yr_avg",
        "receptions_pg_2yr_avg",
        "fantasy_ppg_delta",
    ]


def te_query():
    return wr_query("Tight End")


def te_features():
    return wr_features()


# -----------------------------
# QB
# -----------------------------
def qb_query():
    query = """
    WITH season_stats AS (
        SELECT
            p.id AS player_id,
            CONCAT(p.first_name, ' ', p.last_name) AS player_name,
            g.season_id,
            COUNT(DISTINCT g.game_id) AS games_played,

            SUM(pas.passing_attempts) AS pass_atts,
            SUM(pas.passing_completions) AS pass_completions,
            SUM(pas.passing_yards) AS pass_yards,
            SUM(pas.passing_touchdowns) AS pass_tds,
            SUM(pas.interceptions) AS pass_ints,

            SUM(r2.rushing_attempts) AS rush_atts,
            SUM(r2.rushing_yards) AS rush_yards,
            SUM(r2.rushing_touchdowns) AS rush_tds,

            SUM(s.total_points) AS fantasy_points
        FROM postgres.fantasyfootball.player p
        JOIN postgres.fantasyfootball.stats s ON p.id = s.player_id
        JOIN postgres.fantasyfootball.game g ON g.game_id = s.game_id
        LEFT JOIN postgres.fantasyfootball.passing pas ON pas.pass_id = s.pass_id
        LEFT JOIN postgres.fantasyfootball.rushing r2 ON r2.rush_id = s.rush_id
        WHERE p.position = 'Quarterback'
          AND g.season_id >= 6
        GROUP BY p.id, p.first_name, p.last_name, g.season_id
    ),
    base AS (
        SELECT
            prev.*,

            CASE WHEN prev.games_played > 0
                 THEN prev.fantasy_points::float / prev.games_played ELSE 0 END AS fantasy_ppg,

            CASE WHEN next.games_played > 0
                 THEN next.fantasy_points::float / next.games_played
                 ELSE NULL END AS next_season_fantasy_ppg
        FROM season_stats prev
        LEFT JOIN season_stats next
          ON prev.player_id = next.player_id
         AND next.season_id = prev.season_id + 1
        WHERE prev.games_played >= 8
    )
    SELECT
        b.season_id AS feature_season,
        b.player_id,
        b.player_name,
        b.games_played,
        b.pass_atts,
        b.pass_completions,
        b.pass_yards,
        b.pass_tds,
        b.pass_ints,
        b.rush_atts,
        b.rush_yards,
        b.rush_tds,
        b.fantasy_ppg,
        b.next_season_fantasy_ppg,

        AVG(b.pass_yards::float / b.games_played) OVER (
            PARTITION BY b.player_id
            ORDER BY b.season_id
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS pass_yards_pg_2yr_avg,

        AVG(b.pass_tds::float / b.games_played) OVER (
            PARTITION BY b.player_id
            ORDER BY b.season_id
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS pass_tds_pg_2yr_avg,

        AVG(b.rush_yards::float / b.games_played) OVER (
            PARTITION BY b.player_id
            ORDER BY b.season_id
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS rush_yards_pg_2yr_avg
    FROM base b;
    """
    return pd.read_sql(query, engine)


def qb_features():
    return [
        "games_played",
        "pass_atts",
        "pass_completions",
        "pass_yards",
        "pass_tds",
        "pass_ints",
        "rush_atts",
        "rush_yards",
        "rush_tds",
        "fantasy_ppg",
        "pass_yards_pg_2yr_avg",
        "pass_tds_pg_2yr_avg",
        "rush_yards_pg_2yr_avg",
    ]

def print_top_50(df, models, features, position_name):
    predict_df = df[df.feature_season == PREDICT_SEASON - 1].copy()

    for q in QUANTILES:
        predict_df[f"pred_{int(q * 100)}"] = models[q].predict(
            predict_df[features]
        )

    top50 = predict_df[
        ["player_name", "pred_25", "pred_50", "pred_75"]
    ].sort_values("pred_75", ascending=False).head(50)

    print(f"\nTop 50 {position_name}s – {PREDICT_SEASON} Projection (PPG)")
    print(top50)

# -----------------------------
# TRAIN / EVAL
# -----------------------------
def train_position_model(df, features, position_name):
    df = df[df[TARGET].notna()].copy()

    train = df[df.feature_season <= TRAIN_CUTOFF]
    val = df[df.feature_season == VAL_SEASON]
    test = df[df.feature_season == TEST_SEASON]

    X_train, y_train = train[features], train[TARGET]
    X_val, y_val = val[features], val[TARGET]
    X_test, y_test = test[features], test[TARGET]

    baseline = X_val["fantasy_ppg"]
    baseline_rmse = mean_squared_error(y_val, baseline, squared=False)

    models = {}
    for q in QUANTILES:
        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=q,
            max_depth=5,
            learning_rate=0.05,
            max_iter=300,
            random_state=42,
        )
        model.fit(X_train, y_train)
        models[q] = model

    val_rmse = mean_squared_error(
        y_val, models[0.50].predict(X_val), squared=False
    )
    test_rmse = mean_squared_error(
        y_test, models[0.50].predict(X_test), squared=False
    )

    print(f"\n{position_name}")
    print(f"Baseline RMSE: {baseline_rmse:.2f}")
    print(f"Validation RMSE: {val_rmse:.2f}")
    print(f"Test RMSE: {test_rmse:.2f}")

    return models


def train_weekly_position_model(df: pd.DataFrame, position: str, target_col='points'):
    """
    Trains a regression model for a given position on weekly in-season data.

    df: DataFrame with historical weekly data
    position: 'RB', 'WR', 'TE', 'QB'
    target_col: the column to predict (weekly fantasy points)
    """
    # Filter for position
    df_pos = df[df['position'] == position].copy()

    # Features: exclude target, player info, week info
    exclude_cols = ['player_name', 'team', 'position', target_col]
    feature_cols = [c for c in df_pos.columns if c not in exclude_cols]

    X = df_pos[feature_cols]
    y = df_pos[target_col]

    # Train/test split using historical weeks
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # Optional: validation performance
    val_preds = model.predict(X_val)
    rmse = ((val_preds - y_val) ** 2).mean() ** 0.5
    print(f"{position} weekly model validation RMSE: {rmse:.2f}")

    return model, feature_cols


def get_week_features(df: pd.DataFrame, current_week: int, position: str):
    """
    Prepares features for weekly predictions for a specific week.
    """
    df_week = df[(df['position'] == position) & (df['week'] <= current_week)].copy()

    # Example: 3-week rolling average of points
    df_week['rolling_3wk_ppg'] = df_week.groupby('player_name')['points'].rolling(3, min_periods=1).mean().reset_index(
        0, drop=True)

    # You can add opponent-adjusted stats here
    # df_week['opp_def_rank'] = ...

    # Only keep last week per player for prediction
    week_features = df_week[df_week['week'] == current_week].copy()
    return week_features

def predict_week(model, feature_cols, week_features: pd.DataFrame):
    X = week_features[feature_cols]
    week_features['pred_points'] = model.predict(X)
    return week_features[['player_name', 'team', 'pred_points']]

def rb_weekly_query():
    query = """
    SELECT
        p.id AS player_id,
        CONCAT(p.first_name, ' ', p.last_name) AS player_name,
        p.position,
        g.season_id,
        t.name AS team,
        g.week,
        s.total_points AS points,

        r2.rushing_attempts,
        r2.rushing_yards,
        r2.rushing_touchdowns,

        rec.receptions,
        rec.receiving_yards,
        rec.receiving_touchdowns
    FROM fantasyfootball.player p
    JOIN fantasyfootball.stats s ON p.id = s.player_id
    JOIN fantasyfootball.game g ON g.game_id = s.game_id
    LEFT JOIN fantasyfootball.rushing r2 ON r2.rush_id = s.rush_id
    LEFT JOIN fantasyfootball.receiving rec ON rec.reception_id = s.reception_id
    JOIN fantasyfootball.team t ON t.id = p.team_id
    WHERE p.position = 'Running Back'
      AND g.season_id >= 6
    """
    return pd.read_sql(query, engine)

def rookie_wr_features():
    return [
        "draft_round",
        "draft_pick",
        "college_games",
        "receptions_pg",
        "yards_pg",
        "tds_pg",
        "yards_per_reception",
    ]

def train_rookie_wr_model(df):
    df = df[df["rookie_fantasy_ppg"].notna()].copy()

    train = df[df.rookie_season <= 12]
    val   = df[df.rookie_season == 13]
    test  = df[df.rookie_season == 14]

    X_train = train[rookie_wr_features()]
    y_train = train["rookie_fantasy_ppg"]

    X_val = val[rookie_wr_features()]
    y_val = val["rookie_fantasy_ppg"]

    X_test = test[rookie_wr_features()]
    y_test = test["rookie_fantasy_ppg"]

    model = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=400,
        random_state=42,
    )

    model.fit(X_train, y_train)

    val_rmse = mean_squared_error(
        y_val, model.predict(X_val), squared=False
    )
    test_rmse = mean_squared_error(
        y_test, model.predict(X_test), squared=False
    )

    print(f"Rookie WR Validation RMSE: {val_rmse:.2f}")
    print(f"Rookie WR Test RMSE: {test_rmse:.2f}")

    return model

def rookie_wr_query():
    query = """
    WITH drafted_wr AS (
    SELECT
        nd.player_id,
        nd.season_id AS rookie_season,
        nd.draft_round,
        nd.draft_pick
    FROM fantasyfootball.nfl_draft nd
    JOIN fantasyfootball.player p
        ON p.id = nd.player_id
    WHERE p.position = 'Wide Receiver'
    ),
    
    rookie_nfl AS (
        SELECT
            s.player_id,
            g.season_id,
            COUNT(DISTINCT g.game_id) AS games_played,
            SUM(s.total_points) AS total_points
        FROM fantasyfootball.stats s
        JOIN fantasyfootball.game g
            ON g.game_id = s.game_id
        GROUP BY s.player_id, g.season_id
    ),
    
    college_agg AS (
        SELECT
            cp.id AS college_player_id,
            SUM(cr.games_played) AS college_games,
            SUM(cr.receptions) AS total_receptions,
            SUM(cr.receiving_yards) AS total_yards,
            SUM(cr.receiving_touchdowns) AS total_tds
        FROM fantasyfootball.college_player cp
        JOIN fantasyfootball.college_stats cs
            ON cs.player_id = cp.id
        JOIN fantasyfootball.college_receiving cr
            ON cs.reception_id = cr.reception_id
        GROUP BY cp.id
    )
    
    SELECT
        d.player_id,
        d.rookie_season,
    
        -- Draft features
        d.draft_round,
        d.draft_pick,
    
        -- College features
        c.college_games,
        c.total_receptions::float / NULLIF(c.college_games,0) AS receptions_pg,
        c.total_yards::float / NULLIF(c.college_games,0) AS yards_pg,
        c.total_tds::float / NULLIF(c.college_games,0) AS tds_pg,
        c.total_yards::float / NULLIF(c.total_receptions,0) AS yards_per_reception,
    
        -- TARGET
        CASE WHEN r.games_played > 0
             THEN r.total_points::float / r.games_played
             ELSE NULL END AS rookie_fantasy_ppg
    
    FROM drafted_wr d
    LEFT JOIN rookie_nfl r
      ON r.player_id = d.player_id
     AND r.season_id = d.rookie_season
    
    LEFT JOIN college_agg c
      ON c.college_player_id = d.player_id;  -- ideally mapped properly
    """
    return pd.read_sql(query, engine)

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":

    # 1️⃣ Load rookie dataset
    df = rookie_wr_query()

    # 2️⃣ Train model (pass df in!)
    rookie_model = train_rookie_wr_model(df)

    # 3️⃣ Get incoming rookies
    incoming = df[df.rookie_season == 16].copy()

    # 4️⃣ Predict
    incoming["predicted_ppg"] = rookie_model.predict(
        incoming[rookie_wr_features()]
    )

    # 5️⃣ Rank
    print(
        incoming[["player_id", "predicted_ppg"]]
        .sort_values("predicted_ppg", ascending=False)
    )
    # rb_df = rb_query()
    # rb_models = train_position_model(rb_df, rb_features(), "Running Back")
    #
    # wr_df = wr_query()
    # wr_models = train_position_model(wr_df, wr_features(), "Wide Receiver")
    #
    # te_df = te_query()
    # te_models = train_position_model(te_df, te_features(), "Tight End")
    #
    # qb_df = qb_query()
    # qb_models = train_position_model(qb_df, qb_features(), "Quarterback")
    #
    # # Print rankings
    # print_top_50(rb_df, rb_models, rb_features(), "Running Back")
    # print_top_50(wr_df, wr_models, wr_features(), "Wide Receiver")
    # print_top_50(te_df, te_models, te_features(), "Tight End")
    # print_top_50(qb_df, qb_models, qb_features(), "Quarterback")
    #
    # current_week = 3
    # df_weekly = rb_weekly_query()  # or combine RB, WR, TE, QB into one df
    #
    # # Example for RB
    # weekly_model, feature_cols = train_weekly_position_model(df_weekly, 'Running Back')
    # week_feats = get_week_features(df_weekly, current_week, 'Running Back')
    # week_preds = predict_week(weekly_model, feature_cols, week_feats)
    # print(week_preds.sort_values('pred_points', ascending=False).head(50))

