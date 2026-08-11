import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

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

TRAIN_CUTOFF = 13
VAL_SEASON = 14
TEST_SEASON = 15
PREDICT_SEASON = 17

TOP_N_BY_POSITION = {
    "QB": 12,
    "RB": 24,
    "WR": 24,
    "TE": 12,
    "K": 12,
    "DST": 12,
}


def get_candidate_models():
    return {
        "HistGradientBoosting": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="squared_error",
                    max_iter=300,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    min_samples_leaf=10,
                    l2_regularization=0.1,
                    random_state=42
                )
            )
        ]),

        "RandomForest": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]),

        "ExtraTrees": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=500,
                    max_depth=None,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    max_features=0.8,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]),

        "GradientBoosting": Pipeline([
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "model",
                GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=0.03,
                    max_depth=3,
                    min_samples_leaf=5,
                    loss="huber",
                    random_state=42
                )
            )
        ])
    }

def spearman_rank_correlation(y_true, y_pred):
    """
    Measures how well the predicted ordering matches the actual ordering.

    1.0  = perfect ranking
    0.0  = no ranking relationship
    -1.0 = completely reversed ranking
    """

    actual_ranks = pd.Series(y_true).rank(
        ascending=False,
        method="average"
    )

    predicted_ranks = pd.Series(y_pred).rank(
        ascending=False,
        method="average"
    )

    correlation = actual_ranks.corr(
        predicted_ranks,
        method="pearson"
    )

    if pd.isna(correlation):
        return 0.0

    return correlation

def top_n_hit_rate(y_true, y_pred, top_n):
    """
    What percentage of the actual top-N players
    did the model correctly identify as top-N?
    """

    results = pd.DataFrame({
        "actual": np.asarray(y_true),
        "predicted": np.asarray(y_pred)
    })

    top_n = min(top_n, len(results))

    if top_n == 0:
        return 0.0

    actual_top = set(
        results.nlargest(top_n, "actual").index
    )

    predicted_top = set(
        results.nlargest(top_n, "predicted").index
    )

    correct = len(actual_top & predicted_top)

    return correct / top_n

def mean_rank_error(y_true, y_pred):
    """
    Average number of ranking positions the prediction missed by.
    Lower is better.
    """

    results = pd.DataFrame({
        "actual": np.asarray(y_true),
        "predicted": np.asarray(y_pred)
    })

    results["actual_rank"] = (
        results["actual"]
        .rank(ascending=False, method="average")
    )

    results["predicted_rank"] = (
        results["predicted"]
        .rank(ascending=False, method="average")
    )

    return (
        results["actual_rank"] -
        results["predicted_rank"]
    ).abs().mean()

def save_predictions(df, table_name):

    df.to_sql(
        table_name,
        engine,
        schema="fantasyfootball",
        if_exists="replace",
        index=False
    )

def rookie_receiver_query(position):
    query = f"""
    WITH drafted_receivers AS (
        SELECT
            nd.player_id AS nfl_player_id,
            cp.id AS college_player_id,

            p.first_name,
            p.last_name,

            nd.season_id AS rookie_season,
            nd.draft_round,
            nd.draft_pick

        FROM fantasyfootball.nfl_draft nd

        JOIN fantasyfootball.player p
            ON p.id = nd.player_id

        JOIN fantasyfootball.college_player cp
            ON LOWER(TRIM(cp.first_name)) = LOWER(TRIM(p.first_name))
           AND LOWER(TRIM(cp.last_name)) = LOWER(TRIM(p.last_name))

        WHERE p.position = '{position}'
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

        GROUP BY
            s.player_id,
            g.season_id
    ),

    college_agg AS (
        SELECT
            cp.id AS college_player_id,

            COALESCE(
                MAX(rec.games_played),
                MAX(rush.games_played),
                0
            ) AS college_games,

            COALESCE(SUM(rec.receptions), 0) AS receptions,
            COALESCE(SUM(rec.receiving_yards), 0) AS rec_yards,
            COALESCE(SUM(rec.receiving_touchdowns), 0) AS rec_tds,

            COALESCE(SUM(rush.rushing_attempts), 0) AS rush_atts,
            COALESCE(SUM(rush.rushing_yards), 0) AS rush_yards,
            COALESCE(SUM(rush.rushing_touchdowns), 0) AS rush_tds

        FROM fantasyfootball.college_player cp

        JOIN fantasyfootball.college_stats cs
            ON cs.player_id = cp.id

        LEFT JOIN fantasyfootball.college_receiving rec
            ON cs.reception_id = rec.reception_id

        LEFT JOIN fantasyfootball.college_rushing rush
            ON cs.rush_id = rush.rush_id

        GROUP BY cp.id
    )

    SELECT
        d.nfl_player_id AS player_id,
        d.college_player_id,

        d.first_name,
        d.last_name,

        d.rookie_season,
        d.draft_round,
        d.draft_pick,

        c.college_games,

        c.receptions::float
            / NULLIF(c.college_games, 0)
            AS receptions_pg,

        c.rec_yards::float
            / NULLIF(c.college_games, 0)
            AS receiving_yards_pg,

        c.rec_tds::float
            / NULLIF(c.college_games, 0)
            AS receiving_tds_pg,

        c.rec_yards::float
            / NULLIF(c.receptions, 0)
            AS yards_per_reception,

        c.rush_yards::float
            / NULLIF(c.college_games, 0)
            AS rush_yards_pg,

        c.rush_tds::float
            / NULLIF(c.college_games, 0)
            AS rush_tds_pg,

        c.rush_yards::float
            / NULLIF(c.rush_atts, 0)
            AS rush_yards_per_attempt,

        CASE
            WHEN r.games_played > 0
            THEN r.total_points::float / r.games_played
            ELSE NULL
        END AS rookie_fantasy_ppg

    FROM drafted_receivers d

    LEFT JOIN rookie_nfl r
        ON r.player_id = d.nfl_player_id
       AND r.season_id = d.rookie_season

    LEFT JOIN college_agg c
        ON c.college_player_id = d.college_player_id

    ORDER BY
        d.rookie_season,
        d.draft_round,
        d.draft_pick;
    """

    return pd.read_sql(query, engine)

# -----------------------------
# DEFENSE / SPECIAL TEAMS
# -----------------------------
def dst_query():
    query = """
    WITH season_stats AS (
        SELECT
            p.id AS player_id,
            CONCAT(p.first_name, ' ', p.last_name) AS player_name,
            g.season_id,

            COUNT(DISTINCT g.game_id) AS games_played,

            COALESCE(SUM(d.sacks), 0) AS sacks,
            COALESCE(SUM(d.interceptions), 0) AS interceptions,
            COALESCE(SUM(d.safeties), 0) AS safeties,
            COALESCE(SUM(d.fumble_recoveries), 0) AS fumble_recoveries,
            COALESCE(SUM(d.blocked_kicks), 0) AS blocked_kicks,
            COALESCE(SUM(d.touchdowns), 0) AS touchdowns,

            COALESCE(SUM(d.points_allowed), 0) AS points_allowed,
            COALESCE(SUM(d.pass_yards_allowed), 0) AS pass_yards_allowed,
            COALESCE(SUM(d.rush_yards_allowed), 0) AS rush_yards_allowed,
            COALESCE(SUM(d.total_yards_allowed), 0) AS total_yards_allowed,

            COALESCE(SUM(s.total_points), 0) AS fantasy_points

        FROM fantasyfootball.player p

        JOIN fantasyfootball.stats s
            ON p.id = s.player_id

        JOIN fantasyfootball.game g
            ON g.game_id = s.game_id

        JOIN fantasyfootball.defense d
            ON d.defense_id = s.defense_id

        WHERE p.position = 'Defense/Special Teams'
          AND g.season_id >= 6

        GROUP BY
            p.id,
            p.first_name,
            p.last_name,
            g.season_id
    ),

    base AS (
        SELECT
            prev.player_id,
            prev.player_name,
            prev.season_id AS feature_season,

            prev.games_played,

            prev.sacks,
            prev.interceptions,
            prev.safeties,
            prev.fumble_recoveries,
            prev.blocked_kicks,
            prev.touchdowns,

            prev.points_allowed,
            prev.pass_yards_allowed,
            prev.rush_yards_allowed,
            prev.total_yards_allowed,

            CASE
                WHEN prev.games_played > 0
                THEN prev.sacks::float / prev.games_played
                ELSE 0
            END AS sacks_pg,

            CASE
                WHEN prev.games_played > 0
                THEN prev.interceptions::float / prev.games_played
                ELSE 0
            END AS interceptions_pg,

            CASE
                WHEN prev.games_played > 0
                THEN prev.fumble_recoveries::float / prev.games_played
                ELSE 0
            END AS fumble_recoveries_pg,

            CASE
                WHEN prev.games_played > 0
                THEN (
                    prev.interceptions
                    + prev.fumble_recoveries
                )::float / prev.games_played
                ELSE 0
            END AS takeaways_pg,

            CASE
                WHEN prev.games_played > 0
                THEN prev.touchdowns::float / prev.games_played
                ELSE 0
            END AS touchdowns_pg,

            CASE
                WHEN prev.games_played > 0
                THEN prev.points_allowed::float / prev.games_played
                ELSE 0
            END AS points_allowed_pg,

            CASE
                WHEN prev.games_played > 0
                THEN prev.pass_yards_allowed::float / prev.games_played
                ELSE 0
            END AS pass_yards_allowed_pg,

            CASE
                WHEN prev.games_played > 0
                THEN prev.rush_yards_allowed::float / prev.games_played
                ELSE 0
            END AS rush_yards_allowed_pg,

            CASE
                WHEN prev.games_played > 0
                THEN prev.total_yards_allowed::float / prev.games_played
                ELSE 0
            END AS total_yards_allowed_pg,

            CASE
                WHEN prev.games_played > 0
                THEN prev.fantasy_points::float / prev.games_played
                ELSE 0
            END AS fantasy_ppg,

            CASE
                WHEN next.games_played > 0
                THEN next.fantasy_points::float / next.games_played
                ELSE NULL
            END AS next_season_fantasy_ppg

        FROM season_stats prev

        LEFT JOIN season_stats next
            ON next.player_id = prev.player_id
           AND next.season_id = prev.season_id + 1
    )

    SELECT
        b.*,

        AVG(b.fantasy_ppg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS fantasy_ppg_2yr_avg,

        AVG(b.sacks_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS sacks_pg_2yr_avg,

        AVG(b.takeaways_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS takeaways_pg_2yr_avg,

        AVG(b.points_allowed_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS points_allowed_pg_2yr_avg,

        AVG(b.total_yards_allowed_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS total_yards_allowed_pg_2yr_avg,

        AVG(b.games_played) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS games_played_2yr_avg,

        b.fantasy_ppg
        - LAG(b.fantasy_ppg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
        ) AS fantasy_ppg_delta,

        b.sacks_pg
        - LAG(b.sacks_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
        ) AS sacks_pg_delta,

        b.takeaways_pg
        - LAG(b.takeaways_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
        ) AS takeaways_pg_delta,

        b.points_allowed_pg
        - LAG(b.points_allowed_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
        ) AS points_allowed_pg_delta,

        b.total_yards_allowed_pg
        - LAG(b.total_yards_allowed_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
        ) AS total_yards_allowed_pg_delta

    FROM base b;
    """

    return pd.read_sql(query, engine)

def dst_features():
    return [
        "games_played",

        "sacks",
        "interceptions",
        "safeties",
        "fumble_recoveries",
        "blocked_kicks",
        "touchdowns",

        "points_allowed",
        "pass_yards_allowed",
        "rush_yards_allowed",
        "total_yards_allowed",

        "sacks_pg",
        "interceptions_pg",
        "fumble_recoveries_pg",
        "takeaways_pg",
        "touchdowns_pg",

        "points_allowed_pg",
        "pass_yards_allowed_pg",
        "rush_yards_allowed_pg",
        "total_yards_allowed_pg",

        "fantasy_ppg",

        "fantasy_ppg_2yr_avg",
        "sacks_pg_2yr_avg",
        "takeaways_pg_2yr_avg",
        "points_allowed_pg_2yr_avg",
        "total_yards_allowed_pg_2yr_avg",
        "games_played_2yr_avg",

        "fantasy_ppg_delta",
        "sacks_pg_delta",
        "takeaways_pg_delta",
        "points_allowed_pg_delta",
        "total_yards_allowed_pg_delta",
    ]

# -----------------------------
# KICKER
# -----------------------------
def k_query():
    query = """
    WITH season_stats AS (
        SELECT
            p.id AS player_id,
            CONCAT(p.first_name, ' ', p.last_name) AS player_name,
            g.season_id,

            COUNT(DISTINCT g.game_id) AS games_played,

            COALESCE(SUM(k.extra_point_attempts), 0) AS xp_attempts,
            COALESCE(SUM(k.extra_points_made), 0) AS xp_made,

            COALESCE(SUM(k.field_goal_attempts), 0) AS fg_attempts,
            COALESCE(SUM(k.field_goals_made), 0) AS fg_made,

            COALESCE(
                SUM(k.fifty_yard_field_goals_made),
                0
            ) AS fg_50_made,

            COALESCE(SUM(s.total_points), 0) AS fantasy_points

        FROM fantasyfootball.player p

        JOIN fantasyfootball.stats s
            ON p.id = s.player_id

        JOIN fantasyfootball.game g
            ON g.game_id = s.game_id

        JOIN fantasyfootball.kicking k
            ON k.kicker_id = s.kicker_id

        WHERE p.position = 'Kicker'
          AND g.season_id >= 6

        GROUP BY
            p.id,
            p.first_name,
            p.last_name,
            g.season_id
    ),

    base AS (
        SELECT
            prev.player_id,
            prev.player_name,
            prev.season_id AS feature_season,

            prev.games_played,

            prev.xp_attempts,
            prev.xp_made,

            prev.fg_attempts,
            prev.fg_made,
            prev.fg_50_made,

            CASE
                WHEN prev.games_played > 0
                THEN prev.xp_attempts::float / prev.games_played
                ELSE 0
            END AS xp_attempts_pg,

            CASE
                WHEN prev.games_played > 0
                THEN prev.fg_attempts::float / prev.games_played
                ELSE 0
            END AS fg_attempts_pg,

            CASE
                WHEN prev.xp_attempts > 0
                THEN prev.xp_made::float / prev.xp_attempts
                ELSE 0
            END AS xp_pct,

            CASE
                WHEN prev.fg_attempts > 0
                THEN prev.fg_made::float / prev.fg_attempts
                ELSE 0
            END AS fg_pct,

            CASE
                WHEN prev.games_played > 0
                THEN prev.fantasy_points::float / prev.games_played
                ELSE 0
            END AS fantasy_ppg,

            CASE
                WHEN next.games_played > 0
                THEN next.fantasy_points::float / next.games_played
                ELSE NULL
            END AS next_season_fantasy_ppg

        FROM season_stats prev

        LEFT JOIN season_stats next
            ON next.player_id = prev.player_id
           AND next.season_id = prev.season_id + 1
    )

    SELECT
        b.*,

        AVG(b.fantasy_ppg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS fantasy_ppg_2yr_avg,

        AVG(b.fg_attempts_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS fg_attempts_pg_2yr_avg,

        AVG(b.xp_attempts_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS xp_attempts_pg_2yr_avg,

        AVG(b.games_played) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS games_played_2yr_avg,

        b.fantasy_ppg
        - LAG(b.fantasy_ppg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
        ) AS fantasy_ppg_delta,

        b.fg_attempts_pg
        - LAG(b.fg_attempts_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
        ) AS fg_attempts_pg_delta,

        b.xp_attempts_pg
        - LAG(b.xp_attempts_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
        ) AS xp_attempts_pg_delta

    FROM base b;
    """

    return pd.read_sql(query, engine)

def k_features():
    return [
        "games_played",

        "xp_attempts",
        "xp_made",

        "fg_attempts",
        "fg_made",
        "fg_50_made",

        "xp_attempts_pg",
        "fg_attempts_pg",

        "xp_pct",
        "fg_pct",

        "fantasy_ppg",

        "fantasy_ppg_2yr_avg",
        "fg_attempts_pg_2yr_avg",
        "xp_attempts_pg_2yr_avg",
        "games_played_2yr_avg",

        "fantasy_ppg_delta",
        "fg_attempts_pg_delta",
        "xp_attempts_pg_delta",
    ]
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

        JOIN postgres.fantasyfootball.stats s
            ON p.id = s.player_id

        JOIN postgres.fantasyfootball.game g
            ON g.game_id = s.game_id

        JOIN postgres.fantasyfootball.receiving rec
            ON rec.reception_id = s.reception_id

        WHERE p.position = '{position}'
          AND g.season_id >= 6

        GROUP BY
            p.id,
            p.first_name,
            p.last_name,
            g.season_id
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

            -- -------------------------------------------------
            -- PER-GAME FEATURES
            -- -------------------------------------------------
            CASE
                WHEN prev.games_played > 0
                THEN prev.receptions::float / prev.games_played
                ELSE 0
            END AS receptions_pg,

            CASE
                WHEN prev.games_played > 0
                THEN prev.rec_yards::float / prev.games_played
                ELSE 0
            END AS rec_yards_pg,

            CASE
                WHEN prev.games_played > 0
                THEN prev.rec_tds::float / prev.games_played
                ELSE 0
            END AS rec_tds_pg,

            -- -------------------------------------------------
            -- EFFICIENCY
            -- -------------------------------------------------
            CASE
                WHEN prev.receptions > 0
                THEN prev.rec_yards::float / prev.receptions
                ELSE 0
            END AS yards_per_reception,

            -- -------------------------------------------------
            -- FANTASY PRODUCTION
            -- -------------------------------------------------
            CASE
                WHEN prev.games_played > 0
                THEN prev.fantasy_points::float / prev.games_played
                ELSE 0
            END AS fantasy_ppg,

            -- -------------------------------------------------
            -- TARGET
            -- -------------------------------------------------
            CASE
                WHEN next.games_played > 0
                THEN next.fantasy_points::float / next.games_played
                ELSE NULL
            END AS next_season_fantasy_ppg

        FROM season_stats prev

        LEFT JOIN season_stats next
            ON prev.player_id = next.player_id
           AND next.season_id = prev.season_id + 1
    )

    SELECT
        b.*,

        -- -----------------------------------------------------
        -- TWO-YEAR AVERAGES
        -- -----------------------------------------------------
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

        AVG(b.rec_yards_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS rec_yards_pg_2yr_avg,

        AVG(b.rec_tds_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS rec_tds_pg_2yr_avg,

        AVG(b.games_played) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
            ROWS BETWEEN 1 PRECEDING AND CURRENT ROW
        ) AS games_played_2yr_avg,

        -- -----------------------------------------------------
        -- YEAR-OVER-YEAR TRENDS
        -- -----------------------------------------------------
        b.fantasy_ppg -
        LAG(b.fantasy_ppg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
        ) AS fantasy_ppg_delta,

        b.receptions_pg -
        LAG(b.receptions_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
        ) AS receptions_pg_delta,

        b.rec_yards_pg -
        LAG(b.rec_yards_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
        ) AS rec_yards_pg_delta,

        b.rec_tds_pg -
        LAG(b.rec_tds_pg) OVER (
            PARTITION BY b.player_id
            ORDER BY b.feature_season
        ) AS rec_tds_pg_delta

    FROM base b;
    """

    return pd.read_sql(query, engine)


def wr_features():
    return [
        # Availability / season volume
        "games_played",
        "receptions",
        "rec_yards",
        "rec_tds",

        # Per-game production
        "receptions_pg",
        "rec_yards_pg",
        "rec_tds_pg",

        # Efficiency
        "yards_per_reception",

        # Fantasy production
        "fantasy_ppg",

        # Historical stability
        "fantasy_ppg_2yr_avg",
        "receptions_pg_2yr_avg",
        "rec_yards_pg_2yr_avg",
        "rec_tds_pg_2yr_avg",
        "games_played_2yr_avg",

        # Trajectory / breakout indicators
        "fantasy_ppg_delta",
        "receptions_pg_delta",
        "rec_yards_pg_delta",
        "rec_tds_pg_delta",
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
def train_position_model(
        df,
        features,
        position_name,
        target_col=TARGET
):
    """
    Runs a ranking-first model tournament using walk-forward validation.

    Model selection priority:
        1. Top-N hit rate
        2. Top-N rank error
        3. Overall rank correlation
        4. MAE

    Also compares every position against a simple baseline:
        next year's ranking = this year's fantasy PPG ranking

    TEST_SEASON is kept untouched during model selection.
    After choosing a winner, the winner is evaluated on TEST_SEASON,
    then retrained on all available labeled historical data.
    """

    print()
    print("=" * 80)
    print(f"{position_name} MODEL TOURNAMENT")
    print("=" * 80)

    # ---------------------------------------------------------
    # Only rows with a known next-season result can be used
    # for training / evaluation.
    # ---------------------------------------------------------
    labeled_df = df[
        df[target_col].notna()
    ].copy()

    candidate_models = get_candidate_models()

    top_n = TOP_N_BY_POSITION.get(
        position_name,
        12
    )

    # ---------------------------------------------------------
    # Determine available walk-forward validation seasons.
    #
    # TEST_SEASON remains untouched.
    # ---------------------------------------------------------
    available_seasons = sorted(
        labeled_df["feature_season"]
        .dropna()
        .unique()
    )

    validation_seasons = [
        season
        for season in available_seasons
        if season < TEST_SEASON
    ]

    # Require several prior seasons before beginning validation.
    if len(validation_seasons) > 3:
        validation_seasons = validation_seasons[3:]

    # =========================================================
    # BASELINE
    # =========================================================
    print()
    print("-" * 80)
    print("LAST YEAR PPG BASELINE")
    print("-" * 80)

    baseline_fold_results = []

    for validation_season in validation_seasons:

        validation = labeled_df[
            labeled_df.feature_season == validation_season
        ].copy()

        if validation.empty:
            continue

        y_val = (
            validation[target_col]
            .reset_index(drop=True)
        )

        baseline_predictions = (
            validation["fantasy_ppg"]
            .reset_index(drop=True)
            .values
        )

        metrics = evaluate_predictions(
            y_val,
            baseline_predictions,
            top_n
        )

        metrics["season"] = validation_season

        baseline_fold_results.append(metrics)

        print(
            f"Season {validation_season}: "
            f"Rank Corr={metrics['rank_correlation']:.3f} | "
            f"Top {top_n}={metrics['top_n_hit_rate']:.1%} | "
            f"Top {top_n} Rank Error={metrics['top_n_rank_error']:.2f} | "
            f"RMSE={metrics['rmse']:.2f}"
        )

    baseline_df = pd.DataFrame(
        baseline_fold_results
    )

    if not baseline_df.empty:
        print()
        print("BASELINE AVERAGES")
        print(
            f"Rank Correlation : "
            f"{baseline_df['rank_correlation'].mean():.3f}"
        )

        print(
            f"Top {top_n} Hit Rate : "
            f"{baseline_df['top_n_hit_rate'].mean():.1%}"
        )

        print(
            f"Top {top_n} Rank Error : "
            f"{baseline_df['top_n_rank_error'].mean():.2f}"
        )

        print(
            f"Mean Rank Error : "
            f"{baseline_df['mean_rank_error'].mean():.2f}"
        )

        print(
            f"RMSE : "
            f"{baseline_df['rmse'].mean():.2f}"
        )

        print(
            f"MAE : "
            f"{baseline_df['mae'].mean():.2f}"
        )

    # =========================================================
    # MODEL TOURNAMENT
    # =========================================================
    tournament_results = []

    for model_name, model in candidate_models.items():

        fold_results = []

        print()
        print("-" * 80)
        print(model_name)
        print("-" * 80)

        for validation_season in validation_seasons:

            # -------------------------------------------------
            # Walk-forward:
            #
            # Train only on seasons BEFORE validation season.
            # -------------------------------------------------
            train = labeled_df[
                labeled_df.feature_season < validation_season
            ].copy()

            validation = labeled_df[
                labeled_df.feature_season == validation_season
            ].copy()

            if train.empty or validation.empty:
                continue

            X_train = train[features]
            y_train = train[target_col]

            X_val = validation[features]

            y_val = (
                validation[target_col]
                .reset_index(drop=True)
            )

            # -------------------------------------------------
            # Train / predict
            # -------------------------------------------------
            model.fit(
                X_train,
                y_train
            )

            predictions = model.predict(
                X_val
            )

            # -------------------------------------------------
            # Ranking-first evaluation
            # -------------------------------------------------
            metrics = evaluate_predictions(
                y_val,
                predictions,
                top_n
            )

            metrics["season"] = validation_season

            fold_results.append(metrics)

            print(
                f"Season {validation_season}: "
                f"Rank Corr={metrics['rank_correlation']:.3f} | "
                f"Top {top_n}={metrics['top_n_hit_rate']:.1%} | "
                f"Top {top_n} Rank Error="
                f"{metrics['top_n_rank_error']:.2f} | "
                f"Rank Error={metrics['mean_rank_error']:.2f} | "
                f"RMSE={metrics['rmse']:.2f}"
            )

        if not fold_results:
            continue

        fold_df = pd.DataFrame(
            fold_results
        )

        tournament_results.append({
            "model": model_name,

            "avg_top_n_hit_rate":
                fold_df["top_n_hit_rate"].mean(),

            "avg_top_n_rank_error":
                fold_df["top_n_rank_error"].mean(),

            "avg_rank_correlation":
                fold_df["rank_correlation"].mean(),

            "avg_rank_error":
                fold_df["mean_rank_error"].mean(),

            "avg_rmse":
                fold_df["rmse"].mean(),

            "avg_mae":
                fold_df["mae"].mean(),

            "folds":
                len(fold_df)
        })

    # =========================================================
    # TOURNAMENT RESULTS
    # =========================================================
    results_df = pd.DataFrame(
        tournament_results
    )

    if results_df.empty:
        raise ValueError(
            f"No tournament results generated for "
            f"{position_name}"
        )

    # ---------------------------------------------------------
    # Ranking-first winner selection
    # ---------------------------------------------------------
    results_df = (
        results_df
        .sort_values(
            by=[
                "avg_top_n_hit_rate",
                "avg_top_n_rank_error",
                "avg_rank_correlation",
                "avg_mae"
            ],
            ascending=[
                False,
                True,
                False,
                True
            ]
        )
        .reset_index(drop=True)
    )

    print()
    print("=" * 80)
    print(f"{position_name} TOURNAMENT RESULTS")
    print("=" * 80)

    print(
        results_df[
            [
                "model",
                "avg_top_n_hit_rate",
                "avg_top_n_rank_error",
                "avg_rank_correlation",
                "avg_rank_error",
                "avg_rmse",
                "avg_mae",
                "folds"
            ]
        ].to_string(index=False)
    )

    # =========================================================
    # WINNER
    # =========================================================
    winner_name = results_df.iloc[0]["model"]

    print()
    print(
        f"🏆 {position_name} WINNER: "
        f"{winner_name}"
    )

    # Get a fresh copy of the winning model.
    winner_model = get_candidate_models()[
        winner_name
    ]

    # =========================================================
    # FINAL UNTOUCHED TEST
    # =========================================================
    final_train = labeled_df[
        labeled_df.feature_season < TEST_SEASON
    ].copy()

    final_test = labeled_df[
        labeled_df.feature_season == TEST_SEASON
    ].copy()

    if not final_test.empty:

        winner_model.fit(
            final_train[features],
            final_train[target_col]
        )

        test_predictions = winner_model.predict(
            final_test[features]
        )

        test_y = (
            final_test[target_col]
            .reset_index(drop=True)
        )

        test_metrics = evaluate_predictions(
            test_y,
            test_predictions,
            top_n
        )

        # -----------------------------------------------------
        # Baseline on the exact same final test season
        # -----------------------------------------------------
        baseline_test_predictions = (
            final_test["fantasy_ppg"]
            .reset_index(drop=True)
            .values
        )

        baseline_test_metrics = evaluate_predictions(
            test_y,
            baseline_test_predictions,
            top_n
        )

        print()
        print("=" * 80)
        print(
            f"{position_name} FINAL TEST "
            f"(Season {TEST_SEASON})"
        )
        print("=" * 80)

        print()
        print("MODEL")
        print(
            f"Rank Correlation : "
            f"{test_metrics['rank_correlation']:.3f}"
        )

        print(
            f"Top {top_n} Hit Rate : "
            f"{test_metrics['top_n_hit_rate']:.1%}"
        )

        print(
            f"Top {top_n} Rank Error : "
            f"{test_metrics['top_n_rank_error']:.2f}"
        )

        print(
            f"Mean Rank Error : "
            f"{test_metrics['mean_rank_error']:.2f}"
        )

        print(
            f"RMSE : "
            f"{test_metrics['rmse']:.2f}"
        )

        print(
            f"MAE : "
            f"{test_metrics['mae']:.2f}"
        )

        print()
        print("LAST YEAR PPG BASELINE")

        print(
            f"Rank Correlation : "
            f"{baseline_test_metrics['rank_correlation']:.3f}"
        )

        print(
            f"Top {top_n} Hit Rate : "
            f"{baseline_test_metrics['top_n_hit_rate']:.1%}"
        )

        print(
            f"Top {top_n} Rank Error : "
            f"{baseline_test_metrics['top_n_rank_error']:.2f}"
        )

        print(
            f"Mean Rank Error : "
            f"{baseline_test_metrics['mean_rank_error']:.2f}"
        )

        print(
            f"RMSE : "
            f"{baseline_test_metrics['rmse']:.2f}"
        )

        print(
            f"MAE : "
            f"{baseline_test_metrics['mae']:.2f}"
        )

        # -----------------------------------------------------
        # Show the mistakes we actually care about.
        # -----------------------------------------------------
        print_biggest_ranking_misses(
            final_test,
            test_predictions,
            top_n
        )

    # =========================================================
    # FINAL PRODUCTION TRAINING
    #
    # Testing is finished.
    # Train the winner on every completed labeled season.
    # =========================================================
    winner_model.fit(
        labeled_df[features],
        labeled_df[target_col]
    )

    return winner_model, results_df

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
        "receiving_yards_pg",
        "receiving_tds_pg",
        "yards_per_reception",
        "rush_yards_pg",
        "rush_tds_pg",
        "rush_yards_per_attempt",
    ]

def train_rookie_model(df, features, label="Rookie"):
    df = df[df["rookie_fantasy_ppg"].notna()].copy()

    train = df[df.rookie_season <= 12]
    val   = df[df.rookie_season == 13]
    test  = df[df.rookie_season == 14]

    X_train = train[features]
    y_train = train["rookie_fantasy_ppg"]

    X_val = val[features]
    y_val = val["rookie_fantasy_ppg"]

    X_test = test[features]
    y_test = test["rookie_fantasy_ppg"]

    model = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=400,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # -----------------------------
    # Evaluation
    # -----------------------------
    val_rmse = mean_squared_error(
        y_val,
        model.predict(X_val),
        squared=False
    )

    test_rmse = mean_squared_error(
        y_test,
        model.predict(X_test),
        squared=False
    )

    print(f"{label} Validation RMSE: {val_rmse:.2f}")
    print(f"{label} Test RMSE: {test_rmse:.2f}")

    # -----------------------------
    # 🔎 Permutation Importance
    # -----------------------------
    # -----------------------------
    # 🔎 Permutation Importance (Train + Val)
    # -----------------------------
    print(f"\n{label} Permutation Importance (Train + Val):")

    X_all = pd.concat([X_train, X_val])
    y_all = pd.concat([y_train, y_val])

    perm = permutation_importance(
        model,
        X_all,
        y_all,
        n_repeats=20,
        random_state=42,
        scoring="neg_root_mean_squared_error"
    )

    importances = pd.DataFrame({
        "feature": features,
        "importance": perm.importances_mean
    }).sort_values("importance", ascending=False)

    print(importances)

    return model

def rookie_wr_query():
    return rookie_receiver_query("Wide Receiver")


def rookie_te_query():
    return rookie_receiver_query("Tight End")

def rookie_te_features():
    return rookie_wr_features()


def rookie_rb_features():
    return [
        "draft_round",
        "draft_pick",
        "college_games",
        "rush_yards_pg",
        "rush_tds_pg",
        "rush_yards_per_attempt",
        "receptions_pg",
        "receiving_yards_pg",
        "receiving_tds_pg",
    ]

def rookie_rb_query():
    query = """
WITH drafted_rb AS (
        SELECT
            nd.player_id,
            p.first_name,
            p.last_name,
            nd.season_id AS rookie_season,
            nd.draft_round,
            nd.draft_pick
        FROM fantasyfootball.nfl_draft nd
        JOIN fantasyfootball.player p
            ON p.id = nd.player_id
        WHERE p.position = 'Running Back'
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
            
            SUM(rush.games_played) AS college_games,
            
            -- Rushing
            SUM(rush.rushing_attempts) AS rush_atts,
            SUM(rush.rushing_yards) AS rush_yards,
            SUM(rush.rushing_touchdowns) AS rush_tds,
            
            -- Receiving
            SUM(rec.receptions) AS receptions,
            SUM(rec.receiving_yards) AS rec_yards,
            SUM(rec.receiving_touchdowns) AS rec_tds
            
        FROM fantasyfootball.college_player cp
        JOIN fantasyfootball.college_stats cs
            ON cs.player_id = cp.id
        LEFT JOIN fantasyfootball.college_rushing rush
            ON cs.rush_id = rush.rush_id
        LEFT JOIN fantasyfootball.college_receiving rec
            ON cs.reception_id = rec.reception_id
        GROUP BY cp.id
    )

    SELECT
        d.player_id,
        d.first_name,
        d.last_name,
        d.rookie_season,
        d.draft_round,
        d.draft_pick,

        c.college_games,

        -- Rushing per game
        c.rush_yards::float / NULLIF(c.college_games,0) AS rush_yards_pg,
        c.rush_tds::float / NULLIF(c.college_games,0) AS rush_tds_pg,
        c.rush_yards::float / NULLIF(c.rush_atts,0) AS rush_yards_per_attempt,

        -- Receiving per game
        c.receptions::float / NULLIF(c.college_games,0) AS receptions_pg,
        c.rec_yards::float / NULLIF(c.college_games,0) AS receiving_yards_pg,
        c.rec_tds::float / NULLIF(c.college_games,0) AS receiving_tds_pg,

        CASE 
            WHEN r.games_played > 0
            THEN r.total_points::float / r.games_played
            ELSE NULL
        END AS rookie_fantasy_ppg

    FROM drafted_rb d
    LEFT JOIN rookie_nfl r
        ON r.player_id = d.player_id
       AND r.season_id = d.rookie_season
    LEFT JOIN college_agg c
        ON c.college_player_id = d.player_id
    """
    return pd.read_sql(query, engine)


def rookie_qb_features():
    return [
        "draft_round",
        "draft_pick",
        "college_games",
        "pass_yards_pg",
        "pass_tds_pg",
        "completion_pct",
        "yards_per_attempt",
        "td_rate",
        "int_rate",
        "rush_yards_pg",
        "rush_tds_pg",
        "rush_yards_per_attempt",
    ]

def rookie_qb_query():
    query = """
    WITH drafted_qb AS (
        SELECT
            nd.player_id,
            p.first_name,
            p.last_name,
            nd.season_id AS rookie_season,
            nd.draft_round,
            nd.draft_pick
        FROM fantasyfootball.nfl_draft nd
        JOIN fantasyfootball.player p
            ON p.id = nd.player_id
        WHERE p.position = 'Quarterback'
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

            -- Games
            SUM(pg.games_played) AS college_games,

            -- Passing totals
            SUM(pg.passing_attempts) AS pass_att,
            SUM(pg.passing_completions) AS pass_comp,
            SUM(pg.passing_yards) AS pass_yards,
            SUM(pg.passing_touchdowns) AS pass_tds,
            SUM(pg.interceptions) AS interceptions,

            -- Rushing totals
            SUM(rg.rushing_attempts) AS rush_att,
            SUM(rg.rushing_yards) AS rush_yards,
            SUM(rg.rushing_touchdowns) AS rush_tds

        FROM fantasyfootball.college_player cp
        JOIN fantasyfootball.college_stats cs
            ON cs.player_id = cp.id
        LEFT JOIN fantasyfootball.college_passing pg
            ON cs.pass_id = pg.pass_id
        LEFT JOIN fantasyfootball.college_rushing rg
            ON cs.rush_id = rg.rush_id

        GROUP BY cp.id
    )

    SELECT
        d.*,
        c.college_games,

        -- Per-game passing
        c.pass_yards::float / NULLIF(c.college_games,0) AS pass_yards_pg,
        c.pass_tds::float / NULLIF(c.college_games,0) AS pass_tds_pg,

        -- Efficiency
        c.pass_comp::float / NULLIF(c.pass_att,0) AS completion_pct,
        c.pass_yards::float / NULLIF(c.pass_att,0) AS yards_per_attempt,
        c.pass_tds::float / NULLIF(c.pass_att,0) AS td_rate,
        c.interceptions::float / NULLIF(c.pass_att,0) AS int_rate,

        -- Per-game rushing
        c.rush_yards::float / NULLIF(c.college_games,0) AS rush_yards_pg,
        c.rush_tds::float / NULLIF(c.college_games,0) AS rush_tds_pg,
        c.rush_yards::float / NULLIF(c.rush_att,0) AS rush_yards_per_attempt,

        -- Rookie fantasy output
        CASE 
            WHEN r.games_played > 0
            THEN r.total_points::float / r.games_played
            ELSE NULL
        END AS rookie_fantasy_ppg

    FROM drafted_qb d
    LEFT JOIN rookie_nfl r
        ON r.player_id = d.player_id
       AND r.season_id = d.rookie_season
    LEFT JOIN college_agg c
        ON c.college_player_id = d.player_id   
    """
    return pd.read_sql(query, engine)

def evaluate_predictions(
        y_true,
        y_pred,
        top_n
):
    return {
        "rank_correlation": spearman_rank_correlation(
            y_true,
            y_pred
        ),

        "top_n_hit_rate": top_n_hit_rate(
            y_true,
            y_pred,
            top_n
        ),

        "top_n_rank_error": top_n_rank_error(
            y_true,
            y_pred,
            top_n
        ),

        "mean_rank_error": mean_rank_error(
            y_true,
            y_pred
        ),

        "rmse": mean_squared_error(
            y_true,
            y_pred,
            squared=False
        ),

        "mae": mean_absolute_error(
            y_true,
            y_pred
        )
    }

def top_n_rank_error(y_true, y_pred, top_n):
    """
    Average ranking error specifically for players
    who ACTUALLY finished in the top N.

    Lower is better.
    """

    results = pd.DataFrame({
        "actual": np.asarray(y_true),
        "predicted": np.asarray(y_pred)
    })

    results["actual_rank"] = (
        results["actual"]
        .rank(
            ascending=False,
            method="average"
        )
    )

    results["predicted_rank"] = (
        results["predicted"]
        .rank(
            ascending=False,
            method="average"
        )
    )

    actual_top_n = results[
        results["actual_rank"] <= top_n
    ].copy()

    if actual_top_n.empty:
        return np.nan

    return (
        actual_top_n["actual_rank"]
        - actual_top_n["predicted_rank"]
    ).abs().mean()

def print_biggest_ranking_misses(
        test_df,
        predictions,
        top_n,
        limit=15
):
    results = test_df[
        [
            "player_name",
            TARGET
        ]
    ].copy()

    results["prediction"] = predictions

    results["actual_rank"] = (
        results[TARGET]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )

    results["predicted_rank"] = (
        results["prediction"]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )

    results["rank_error"] = (
        results["actual_rank"]
        - results["predicted_rank"]
    ).abs()

    important_misses = results[
        results["actual_rank"] <= top_n
    ]

    important_misses = (
        important_misses
        .sort_values(
            "rank_error",
            ascending=False
        )
        .head(limit)
    )

    print()
    print("=" * 80)
    print(f"BIGGEST TOP {top_n} RANKING MISSES")
    print("=" * 80)

    print(
        important_misses[
            [
                "player_name",
                TARGET,
                "prediction",
                "actual_rank",
                "predicted_rank",
                "rank_error"
            ]
        ].to_string(index=False)
    )

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("TRAINING VETERAN MODELS")
    print("=" * 80)

    veteran_configs = [
        ("QB", qb_query, qb_features),
        ("RB", rb_query, rb_features),
        ("WR", wr_query, wr_features),
        ("TE", te_query, te_features),
        ("K", k_query, k_features),
        ("DST", dst_query, dst_features),
    ]

    veteran_predictions = {}

    for name, query_fn, feature_fn in veteran_configs:
        df = query_fn()
        features = feature_fn()

        best_model, tournament_results = train_position_model(
            df,
            features,
            name
        )

        predict_df = df[
            df.feature_season == PREDICT_SEASON - 1
            ].copy()

        predict_df["predicted_ppg"] = (
            best_model.predict(
                predict_df[features]
            )
        )

        predict_df["projection"] = (
                predict_df["predicted_ppg"] * 17
        )

        veteran_predictions[name] = (
            predict_df[
                [
                    "player_id",
                    "player_name",
                    "projection"
                ]
            ]
            .rename(
                columns={
                    "projection": "predicted_stats"
                }
            )
            .sort_values(
                "predicted_stats",
                ascending=False
            )
            .reset_index(drop=True)
        )

        print()
        print(veteran_predictions[name].head(50))

    print()
    print("=" * 80)
    print("TRAINING ROOKIE MODELS")
    print("=" * 80)

    rookie_configs = [
        ("QB", rookie_qb_query, rookie_qb_features),
        ("RB", rookie_rb_query, rookie_rb_features),
        ("WR", rookie_wr_query, rookie_wr_features),
        ("TE", rookie_te_query, rookie_te_features),
    ]

    rookie_predictions = {}

    for name, query_fn, feature_fn in rookie_configs:
        df = query_fn()
        features = feature_fn()

        model = train_rookie_model(df, features, f"Rookie {name}")

        # Incoming rookies
        rookies = df[df.rookie_season == PREDICT_SEASON].copy()

        rookies["predicted_ppg"] = model.predict(
            rookies[features]
        )

        rookies["projection"] = (
                rookies["predicted_ppg"] * 17
        )

        rookies["player_name"] = (
                rookies["first_name"] + " " + rookies["last_name"]
        )

        rookie_predictions[name] = (
            rookies[
                ["player_id", "player_name", "projection"]
            ]
            .rename(columns={
                "projection": "predicted_stats"
            })
            .assign(is_rookie=True)
            .sort_values("predicted_stats", ascending=False)
            .reset_index(drop=True)
        )

        print()
        print(rookie_predictions[name].head(25))

    print()
    print("=" * 80)
    print("COMBINED RANKINGS")
    print("=" * 80)

    combined_predictions = {}

    for position, veteran_df in veteran_predictions.items():
        rookie_df = rookie_predictions.get(
            position,
            pd.DataFrame(columns=veteran_df.columns)
        )

        combined = pd.concat(
            [veteran_df, rookie_df],
            ignore_index=True
        )

        combined = (
            combined
            .sort_values("predicted_stats", ascending=False)
            .reset_index(drop=True)
        )

        combined_predictions[position] = combined

        print(f"\n{position} Top 50")
        print(combined.head(50))

    # # Save veteran-only predictions
    # save_predictions(
    #     veteran_predictions["QB"],
    #     "quarterback_predictions_2026"
    # )
    #
    # save_predictions(
    #     veteran_predictions["RB"],
    #     "running_back_predictions_2026"
    # )
    #
    # save_predictions(
    #     veteran_predictions["WR"],
    #     "wide_receiver_predictions_2026"
    # )
    #
    # save_predictions(
    #     veteran_predictions["TE"],
    #     "tight_end_predictions_2026"
    # )
    #
    # save_predictions(
    #     veteran_predictions["K"],
    #     "kicker_predictions_2026"
    # )
    #
    # save_predictions(
    #     veteran_predictions["DST"],
    #     "defense_predictions_2026"
    # )
    #
    # # Save rookie-only predictions
    # save_predictions(
    #     rookie_predictions["QB"],
    #     "rookie_quarterback_predictions_2026"
    # )
    #
    # save_predictions(
    #     rookie_predictions["RB"],
    #     "rookie_running_back_predictions_2026"
    # )
    #
    # save_predictions(
    #     rookie_predictions["WR"],
    #     "rookie_wide_receiver_predictions_2026"
    # )
    #
    # save_predictions(
    #     rookie_predictions["TE"],
    #     "rookie_tight_end_predictions_2026"
    # )
    #
    # # Save combined position predictions
    # save_predictions(
    #     combined_predictions["QB"],
    #     "all_qb_predictions_2026"
    # )
    #
    # save_predictions(
    #     combined_predictions["RB"],
    #     "all_rb_predictions_2026"
    # )
    #
    # save_predictions(
    #     combined_predictions["WR"],
    #     "all_wr_predictions_2026"
    # )
    #
    # save_predictions(
    #     combined_predictions["TE"],
    #     "all_te_predictions_2026"
    # )
    #
    # # Save every position into one table
    # all_prediction_frames = []
    #
    # for position, position_df in combined_predictions.items():
    #     position_df = position_df.copy()
    #     position_df["position"] = position
    #     all_prediction_frames.append(position_df)
    #
    # all_predictions = pd.concat(
    #     all_prediction_frames,
    #     ignore_index=True
    # )
    #
    # save_predictions(
    #     all_predictions,
    #     "all_predictions_2026"
    # )