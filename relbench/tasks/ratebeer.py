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

class BeerRecommendationTask(RecommendationTask):
    r"""Predict the list of distinct beers each customer will rate in the next 90 days."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "user_id"
    src_entity_table = "users"
    dst_entity_col = "beer_id"
    dst_entity_table = "beers"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365 // 4)
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
                beer_ratings.user_id,
                LIST(DISTINCT beer_ratings.beer_id) AS beer_id
            FROM
                timestamp_df t
            LEFT JOIN
                beer_ratings
            ON
                beer_ratings.created_at > t.timestamp AND
                beer_ratings.created_at <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                beer_ratings.user_id is not null and beer_ratings.beer_id is not null
            GROUP BY
                t.timestamp,
                beer_ratings.user_id
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

class UserFavoriteBeerTask(RecommendationTask):
    r"""Predict the list of distinct beers each customer will add to their favorites in the next 90 days."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "user_id"
    src_entity_table = "users"
    dst_entity_col = "beer_id"
    dst_entity_table = "beers"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365 // 4)
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
                favorites.user_id,
                LIST(DISTINCT favorites.beer_id) AS beer_id
            FROM
                timestamp_df t
            LEFT JOIN
                favorites
            ON
                favorites.created_at > t.timestamp AND
                favorites.created_at <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                favorites.user_id is not null and favorites.beer_id is not null
            GROUP BY
                t.timestamp,
                favorites.user_id
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

class ActiveUserFavoriteBeerTask(RecommendationTask):
    r"""Predict the list of distinct beers each customer will add to their favorites in the next 90 days, if they have favorited at least one beer in the last 90 days."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "user_id"
    src_entity_table = "users"
    dst_entity_col = "beer_id"
    dst_entity_table = "beers"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=365 // 4)
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

class ActiveUserFavoriteBeerTask180d(RecommendationTask):
    r"""Predict the list of distinct beers each customer adds to their favoritesin the next 180 days, if they have favorited at least one beer in the last 180 days."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "user_id"
    src_entity_table = "users"
    dst_entity_col = "beer_id"
    dst_entity_table = "beers"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=180)
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

class UserPlaceLikedBeerTask(RecommendationTask):
    r"""Predict the list of distinct beers each customer rate at least 4.0 in the next 90 days, if they have rated at least one beer at least 4.0 in the last 180 days."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "user_id"
    src_entity_table = "users"
    dst_entity_col = "beer_id"
    dst_entity_table = "beers"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=180)
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
                AND EXISTS (
                    SELECT 1
                    FROM beer_ratings as br2
                    WHERE br2.user_id = br.user_id
                    AND br2.total_score >= 4.0
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
    r"""Predict the list of distinct beers each customer rate at least 4.0 in the next 180 days, if they have rated at least one beer at least 4.0in the last 180 days."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "user_id"
    src_entity_table = "users"
    dst_entity_col = "beer_id"
    dst_entity_table = "beers"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=180)
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
                    AND br2.total_score >= 4.0
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

class UserPlaceTask(RecommendationTask):
    r"""Predict the list of distinct places each customer rate at least 80 in the next 90 days, if they have rated at least one place at least 80 in the last 90 days."""

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
                    AND pr2.total_score >= 80
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

class VeryActiveUserFavoriteBeerTask180d(RecommendationTask):
    r"""Predict the list of distinct beers each customer adds to their favoritesin the next 180 days, if they have favorited at least one beer in the last 180 days."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "user_id"
    src_entity_table = "users"
    dst_entity_col = "beer_id"
    dst_entity_table = "beers"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=180)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map, link_prediction_mrr]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        users = db.table_dict["users"].df
        beers = db.table_dict["beers"].df
        favorites = db.table_dict["favorites"].df
        beer_ratings = db.table_dict["beer_ratings"].df
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
                favorites AS f
            ON
                f.created_at > t.timestamp
                AND f.created_at <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                f.user_id IS NOT NULL
                AND f.beer_id IS NOT NULL
                AND (
                    SELECT COUNT(*)
                    FROM beer_ratings AS br
                    WHERE br.user_id = f.user_id
                    AND br.created_at > t.timestamp - INTERVAL '{self.timedelta} days'
                    AND br.created_at <= t.timestamp
                ) >= 10
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

class UserStyleLikedBeerTask(RecommendationTask):
    r"""Predict the list of distinct beers each customer rates at least 4.0 in the next 90 days given the beer style, if they have rated at least one beer at least 4.0 in the last 90 days."""

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
        beer_styles = db.table_dict["beer_styles"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                br.user_id,
                bs.style_id,
                LIST(DISTINCT br.beer_id) AS beer_id
            FROM
                timestamp_df t
            LEFT JOIN
                beer_ratings as br
                ON br.created_at > t.timestamp
                AND br.created_at <= t.timestamp + INTERVAL '{self.timedelta} days'
            LEFT JOIN
                beers as b
                ON b.beer_id = br.beer_id
            LEFT JOIN
                beer_styles as bs
                ON bs.style_id = b.style_id
            WHERE
                br.user_id IS NOT NULL and br.beer_id IS NOT NULL
                AND br.total_score >= 4.0
                AND EXISTS (
                    SELECT 1
                    FROM beer_ratings as br2
                    WHERE br2.user_id = br.user_id
                    AND br2.total_score >= 4.0
                    AND br2.created_at > t.timestamp - INTERVAL '{self.timedelta} days'
                    AND br2.created_at <= t.timestamp
                )
            GROUP BY
                t.timestamp,
                br.user_id,
                bs.style_id
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

class UserStyleLikedBrewerTask(RecommendationTask):
    r"""Predict the list of distinct brewers for the beers each customer rates at least 4.0 in the next 90 days given the beer style, if they have rated at least one beer at least 4.0 in the last 90 days."""

    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "user_id"
    src_entity_table = "users"
    dst_entity_col = "brewer_id"
    dst_entity_table = "brewers"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=90)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map, link_prediction_mrr]
    eval_k = 10

    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        users = db.table_dict["users"].df
        brewers = db.table_dict["brewers"].df
        beer_ratings = db.table_dict["beer_ratings"].df
        beer_styles = db.table_dict["beer_styles"].df
        beers = db.table_dict["beers"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                br.user_id,
                bs.style_id,
                LIST(DISTINCT b.brewer_id) AS brewer_id
            FROM
                timestamp_df t
            LEFT JOIN
                beer_ratings AS br
                ON br.created_at > t.timestamp
                AND br.created_at <= t.timestamp + INTERVAL '{self.timedelta} days'
                AND br.total_score >= 4.0
            LEFT JOIN
                beers AS b
                ON b.beer_id = br.beer_id
            LEFT JOIN
                beer_styles AS bs
                ON bs.style_id = b.style_id
            WHERE
                br.user_id IS NOT NULL
                AND b.brewer_id IS NOT NULL
                AND EXISTS (
                    SELECT 1
                    FROM beer_ratings AS br2
                    WHERE br2.user_id = br.user_id
                    AND br2.total_score >= 4.0
                    AND br2.created_at > t.timestamp - INTERVAL '{self.timedelta} days'
                    AND br2.created_at <= t.timestamp
                )
            GROUP BY
                t.timestamp,
                br.user_id,
                bs.style_id
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

class StyleBrewerTask(RecommendationTask):
    r"""Predict the list of distinct brewers that will release beers in the next 720 days for each beer style."""
    
    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "style_id"
    src_entity_table = "beer_styles"
    dst_entity_col = "brewer_id"
    dst_entity_table = "brewers"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=720)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map, link_prediction_mrr]
    eval_k = 10
    
    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table:
        brewers = db.table_dict["brewers"].df
        beer_styles = db.table_dict["beer_styles"].df
        beers = db.table_dict["beers"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                b.style_id,
                LIST(DISTINCT b.brewer_id) AS brewer_id
            FROM
                timestamp_df t
            LEFT JOIN
                beers AS b
            ON b.created_at > t.timestamp AND
                b.created_at <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                b.brewer_id IS NOT NULL AND b.style_id IS NOT NULL
            GROUP BY
                t.timestamp,
                b.style_id
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

class BrewerStyleTask(RecommendationTask):
    r"""Predict the list of distinct styles that each active brewer will release beers for in the next 720 days."""
    
    task_type = TaskType.LINK_PREDICTION
    src_entity_col = "brewer_id"
    src_entity_table = "brewers"
    dst_entity_col = "style_id"
    dst_entity_table = "beer_styles"
    time_col = "timestamp"
    timedelta = pd.Timedelta(days=720)
    metrics = [link_prediction_precision, link_prediction_recall, link_prediction_map, link_prediction_mrr]
    eval_k = 10
    
    def make_table(self, db: Database, timestamps: "pd.Series[pd.Timestamp]") -> Table: 
        brewers = db.table_dict["brewers"].df
        beer_styles = db.table_dict["beer_styles"].df
        beers = db.table_dict["beers"].df
        timestamp_df = pd.DataFrame({"timestamp": timestamps})

        df = duckdb.sql(
            f"""
            SELECT
                t.timestamp,
                b.brewer_id,
                LIST(DISTINCT b.style_id) AS style_id
            FROM
                timestamp_df t
            LEFT JOIN
                beers AS b
            ON b.created_at > t.timestamp AND
                b.created_at <= t.timestamp + INTERVAL '{self.timedelta} days'
            WHERE
                b.brewer_id IS NOT NULL AND b.style_id IS NOT NULL
                AND EXISTS (
                    SELECT 1
                    FROM beers AS b2
                    WHERE b2.brewer_id = b.brewer_id
                    AND b2.created_at > t.timestamp - INTERVAL '{self.timedelta} days'
                    AND b2.created_at <= t.timestamp
                )
            GROUP BY
                t.timestamp,
                b.brewer_id
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


class UserLovedBeerTask(RecommendationTask):
    r"""Predict the list of distinct beers each customer rate 5.0 in the next 180 days, if they have rated at least one beer at least 5.0 in the last 180 days."""

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
                AND br.total_score >= 5.0
                AND EXISTS (
                    SELECT 1
                    FROM beer_ratings as br2
                    WHERE br2.user_id = br.user_id
                    AND br2.total_score >= 5.0
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
    "beer-recommendation": BeerRecommendationTask,
    "user-favorite-beer": UserFavoriteBeerTask,
    "active-user-favorite-beer": ActiveUserFavoriteBeerTask,
    "active-user-favorite-beer-180d": ActiveUserFavoriteBeerTask180d,
    "user-place-liked-beer": UserPlaceLikedBeerTask,
    "user-liked-beer": UserLikedBeerTask,
    "user-place": UserPlaceTask,
    "very-active-user-favorite-beer-180d": VeryActiveUserFavoriteBeerTask180d,
    "user-style-liked-beer": UserStyleLikedBeerTask,
    "user-style-liked-brewer": UserStyleLikedBrewerTask,
    "style-brewer": StyleBrewerTask,
    "brewer-style": BrewerStyleTask,
    "user-loved-beer": UserLovedBeerTask,
}
