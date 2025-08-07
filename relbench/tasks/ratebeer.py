import duckdb
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

from relbench.base import Database, EntityTask, RecommendationTask, Table, TaskType
from relbench.metrics import (
    accuracy,
    average_precision,
    f1,
    link_prediction_map,
    link_prediction_precision,
    link_prediction_recall,
    mae,
    r2,
    rmse,
    roc_auc,
)

def link_prediction_mrr(
    pred_isin: NDArray[np.int_],  # shape (n_src, k)
    dst_count: NDArray[np.int_],   # shape (n_src,)
) -> float:
    # 1) filter out sources without positives
    pos_mask = dst_count > 0
    pred_isin = pred_isin[pos_mask]
    dst_count = dst_count[pos_mask]

    reciprocal_ranks = []
    for row in pred_isin:
        # find the first correct prediction
        hits = np.where(row == 1)[0]
        if len(hits):
            first_hit_rank = hits[0] + 1  # ranks are 1‑based
            reciprocal_ranks.append(1.0 / first_hit_rank)
        else:
            reciprocal_ranks.append(0.0)

    return float(np.mean(reciprocal_ranks))


class UserFavoriteBeerTask(RecommendationTask):
    r"""Predict the list of distinct beers each active user will add to their favorites in the next 90 days."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "user_id"
    src_entity_table = "users"
    dst_entity_col = "beer_id"
    dst_entity_table = "beers"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=90)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map, link_prediction_mrr]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        users = db.table_dict["users"].df
        beers = db.table_dict["beers"].df
        favorites = db.table_dict["favorites"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                f.user_id,
                LIST(DISTINCT f.beer_id) AS beer_id
            FROM
                timestamp_df t
            LEFT JOIN
                favorites as f
            ON
                f.created_at > t.timestamp AND
                f.created_at <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                f.user_id is not null and f.beer_id is not null
                AND EXISTS (
                    SELECT 1
                    FROM favorites as f2
                    WHERE f2.user_id = f.user_id
                    AND f2.created_at > t.timestamp - INTERVAL '{self.timedelta} days'
                    AND f2.created_at <= t.timestamp
                )
            GROUP BY
                t.timestamp,
                f.user_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )

class UserLikedPlaceTask(RecommendationTask):
    r"""Predict the list of distinct places each active user rates at least 80.0 / 100.0 in the next 90 days."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "user_id"
    src_entity_table = "users"
    dst_entity_col = "place_id"
    dst_entity_table = "places"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=90)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map, link_prediction_mrr]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        users = db.table_dict["users"].df
        places = db.table_dict["places"].df
        place_ratings = db.table_dict["place_ratings"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                pr.user_id,
                LIST(DISTINCT pr.place_id) AS place_id
            FROM
                timestamp_df t
            LEFT JOIN
                place_ratings as pr
                ON pr.created_at > t.timestamp
                AND pr.created_at <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                pr.user_id IS NOT NULL and pr.place_id IS NOT NULL
                AND pr.total_score >= 80
                AND EXISTS (
                    SELECT 1
                    FROM place_ratings as pr2
                    WHERE pr2.user_id = pr.user_id
                    AND pr2.created_at > t.timestamp - INTERVAL '{self.timedelta} days'
                    AND pr2.created_at <= t.timestamp
                )
            GROUP BY
                t.timestamp,
                pr.user_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )

class UserPlaceLikedBeerTask(RecommendationTask):
    r"""Predict the list of distinct beers each active user rates at least 4.0 / 5.0 in the next 90 days, given the place they rate the beer at."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "user_id"
    src_entity_table = "users"
    dst_entity_col = "beer_id"
    dst_entity_table = "beers"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=90)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map, link_prediction_mrr]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        users = db.table_dict["users"].df
        beers = db.table_dict["beers"].df
        beer_ratings = db.table_dict["beer_ratings"].df
        availability = db.table_dict["availability"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                br.user_id,
                a.place_id,
                LIST(DISTINCT br.beer_id) AS beer_id
            FROM
                timestamp_df t
            LEFT JOIN
                beer_ratings as br
                ON br.created_at > t.timestamp
                AND br.created_at <= t.timestamp + INTERVAL '{self.timedelta} days'
            LEFT JOIN
                availability as a
                ON a.beer_id = br.beer_id
                AND a.user_id = br.user_id
            WHERE
                br.user_id IS NOT NULL and br.beer_id IS NOT NULL
                AND br.total_score >= 4.0
                AND (a.is_out = false OR a.is_out IS NULL)
                AND EXISTS (
                    SELECT 1
                    FROM beer_ratings as br2
                    WHERE br2.user_id = br.user_id
                    AND br2.created_at > t.timestamp - INTERVAL '{self.timedelta} days'
                    AND br2.created_at <= t.timestamp
                )
            GROUP BY
                t.timestamp,
                br.user_id,
                a.place_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )

class UserLikedBeerTask(RecommendationTask):
    r"""Predict the list of distinct beers each active user rates at least 4.0 / 5.0 in the next 90 days."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "user_id"
    src_entity_table = "users"
    dst_entity_col = "beer_id"
    dst_entity_table = "beers"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=90)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map, link_prediction_mrr]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        users = db.table_dict["users"].df
        beers = db.table_dict["beers"].df
        beer_ratings = db.table_dict["beer_ratings"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                br.user_id,
                LIST(DISTINCT br.beer_id) AS beer_id
            FROM
                timestamp_df t
            LEFT JOIN
                beer_ratings as br
                ON br.created_at > t.timestamp
                AND br.created_at <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                br.user_id IS NOT NULL and br.beer_id IS NOT NULL
                AND br.total_score >= 4.0
                AND EXISTS (
                    SELECT 1
                    FROM beer_ratings as br2
                    WHERE br2.user_id = br.user_id
                    AND br2.created_at > t.timestamp - INTERVAL '{self.timedelta} days'
                    AND br2.created_at <= t.timestamp
                )
            GROUP BY
                t.timestamp,
                br.user_id
            """
        ).df()

        return Table(
            df=df,
            fkey_col_to_pkey_table={
                self.src_entity_col: self.src_entity_table,
                self.dst_entity_col: self.dst_entity_table,
            },
            pkey_col=None,
            time_col=self.time_col,
        )


# Mapping of task names to their corresponding task classes.
tasks_dict = {
    "user-favorite-beer": UserFavoriteBeerTask,
    "user-liked-place": UserLikedPlaceTask,
    "user-place-liked-beer": UserPlaceLikedBeerTask,
    "user-liked-beer": UserLikedBeerTask,
}
