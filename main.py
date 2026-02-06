import pandas as pd
from sqlalchemy import create_engine
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import HistGradientBoostingRegressor

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

TRAIN_CUTOFF = 11
VAL_SEASON = 12
TEST_SEASON = 13
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


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    rb_df = rb_query()
    rb_models = train_position_model(rb_df, rb_features(), "Running Back")

    wr_df = wr_query()
    wr_models = train_position_model(wr_df, wr_features(), "Wide Receiver")

    te_df = te_query()
    te_models = train_position_model(te_df, te_features(), "Tight End")

    qb_df = qb_query()
    qb_models = train_position_model(qb_df, qb_features(), "Quarterback")

    # Print rankings
    print_top_50(rb_df, rb_models, rb_features(), "Running Back")
    print_top_50(wr_df, wr_models, wr_features(), "Wide Receiver")
    print_top_50(te_df, te_models, te_features(), "Tight End")
    print_top_50(qb_df, qb_models, qb_features(), "Quarterback")

