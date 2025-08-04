import duckdb
import pandas as pd
from relbench.base import Dataset, Database, Table


class RateBeerDataset(Dataset):
    # val_timestamp = pd.Timestamp("2024-07-01")
    # test_timestamp = pd.Timestamp("2024-10-04")

    val_timestamp = pd.Timestamp("2021-01-01")
    test_timestamp = pd.Timestamp("2023-01-01")

    def make_db(self) -> Database:
        con = duckdb.connect()
        con.execute("INSTALL httpfs; LOAD httpfs;")

        def load_df(path, query="*"):
            return con.execute(f"SELECT {query} FROM read_parquet('{path}')").df()

        tables = {}

        # ---------------------- Beers ----------------------
        beers = load_df("s3://relbench-ratebeer/beers.parquet")

        # Drop high NA columns
        beers.drop(columns=[
            "contract_brewer_id",       # 96.27% NA
            "contract_note",            # 99.96% NA
            "featured_beer_id",         # 100% NA
            "producer_style",           # 100% NA
            "LogoImage",                # 92.45% NA
            "beer_jobber_id",           # 99.65% NA
            # "year4_style",              # 94.35% NA
            # "year4_overall",            # 94.16% NA
            # "rating_std_dev",           # 15.61% NA
            # "last_9m_avg",              # 86.82% NA
            # "style_percentile",         # 79.37% NA
            # "overall_percentile",       # 79.37% NA
        ], inplace=True)

        tables["beers"] = Table(
            df=beers,
            fkey_col_to_pkey_table={
                "brewer_id": "brewers",
                "style_id": "beer_styles",
            },
            pkey_col="beer_id",
            time_col="created_at",
        )

        # ---------------------- Brewers ----------------------
        brewers = load_df("s3://relbench-ratebeer/brewers.parquet")
        brewers.drop(columns=[
            "newsletter_email",         # 99.85% NA
            "head_brewer",              # 100% NA
            "latitude",                 # 100% NA
            "longitude",                # 100% NA
            "msa",                      # 83.42% NA
            "instagram",                # 95.55% NA
        ], inplace=True)

        tables["brewers"] = Table(
            df=brewers,
            fkey_col_to_pkey_table={
                "country_id": "countries",
                "state_id": "states",
                "type_id": "place_types",
            },
            pkey_col="brewer_id",
        )

        # ---------------------- Beer Styles ----------------------
        tables["beer_styles"] = Table(
            df=load_df("s3://relbench-ratebeer/beer_styles.parquet"),
            fkey_col_to_pkey_table={},
            pkey_col="style_id",
        )

        # ---------------------- Countries ----------------------
        tables["countries"] = Table(
            df=load_df("s3://relbench-ratebeer/countries.parquet"),
            fkey_col_to_pkey_table={},
            pkey_col="country_id",
        )

        # ---------------------- Users ----------------------
        users = load_df("s3://relbench-ratebeer/users.parquet")
        users.drop(columns=[
            "favorite_first_added",     # 98.44% NA
            "favorite_last_added",      # 98.44% NA
        ], inplace=True)

        tables["users"] = Table(
            df=users,
            fkey_col_to_pkey_table={},
            pkey_col="user_id",
            time_col="created_at",
        )

        # ---------------------- Beer Ratings ----------------------
        beer_ratings = load_df("s3://relbench-ratebeer/beer_ratings.parquet")
        # Fix duplicate rating_id (rating_id = 1759935)
        duplicate_mask = beer_ratings.duplicated(subset=['rating_id'], keep='first')
        if duplicate_mask.any():
            print(f"Found {duplicate_mask.sum()} duplicate rating_id(s), fixing...")
            max_rating_id = beer_ratings['rating_id'].max()
            beer_ratings.loc[duplicate_mask, 'rating_id'] += max_rating_id + 1

        beer_ratings.drop(columns=[
            "served_in",                # 99.94% NA
            "latitude",                 # 100% NA
            "longitude",                # 100% NA
        ], inplace=True)

        tables["beer_ratings"] = Table(
            df=beer_ratings,
            fkey_col_to_pkey_table={
                "user_id": "users",
                "beer_id": "beers",
                "availability_id": "availability",
            },
            pkey_col="rating_id",
            time_col="created_at",
        )

        # ---------------------- Availability ----------------------
        availability = load_df("s3://relbench-ratebeer/availability.parquet")
        availability.drop(columns=[
            "area_code",               # 100% NA
            "rating_id",               # 100% NA
            "tap_lister",              # 100% NA
        ], inplace=True)

        tables["availability"] = Table(
            df=availability,
            fkey_col_to_pkey_table={
                "beer_id": "beers",
                "place_id": "places",
                "country_id": "countries",
                "user_id": "users",
            },
            pkey_col="avail_id",
            # time_col="created_at",
        )

        # ---------------------- Beer UPCs ----------------------
        beer_upcs = load_df("s3://relbench-ratebeer/beer_upcs.parquet")
        tables["beer_upcs"] = Table(
            df=beer_upcs,
            fkey_col_to_pkey_table={"beer_id": "beers"},
            pkey_col=None,  # Same UPC may map to multiple beers
        )

        # ---------------------- Favorites ----------------------
        tables["favorites"] = Table(
            df=load_df("s3://relbench-ratebeer/favorites.parquet"),
            fkey_col_to_pkey_table={
                "user_id": "users",
                "beer_id": "beers",
            },
            pkey_col="favorite_id",
            time_col="created_at",
        )

        # ---------------------- Places ----------------------
        places = load_df("s3://relbench-ratebeer/places.parquet")
        places.drop(columns=[
            "email",                   # 100% NA
            "opened_at",              # 100% NA
            "phone_country_code",     # 100% NA
            "last_edited_at",         # 99.98% NA
            # "score",                  # 86.23% NA
        ], inplace=True)

        tables["places"] = Table(
            df=places,
            fkey_col_to_pkey_table={
                "state_id": "states",
                "type_id": "place_types",
                "country_id": "countries",
            },
            pkey_col="place_id",
        )

        # ---------------------- Place Ratings ----------------------
        place_ratings = load_df("s3://relbench-ratebeer/place_ratings.parquet")
        place_ratings.drop(columns=[
            "latitude",               # 100% NA
            "longitude",              # 100% NA
        ], inplace=True)

        tables["place_ratings"] = Table(
            df=place_ratings,
            fkey_col_to_pkey_table={
                "place_id": "places",
                "user_id": "users",
            },
            pkey_col="rating_id",
            time_col="created_at",
        )

        # ---------------------- Place Types ----------------------
        tables["place_types"] = Table(
            df=load_df("s3://relbench-ratebeer/place_types.parquet"),
            fkey_col_to_pkey_table={},
            pkey_col="type_id",
        )

        # ---------------------- States ----------------------
        states = load_df("s3://relbench-ratebeer/states.parquet")
        states.drop(columns=[
            "Abbrev",                 # 86.66% NA
            # "hasbrewer",              # 79.58% NA
        ], inplace=True)

        tables["states"] = Table(
            df=states,
            fkey_col_to_pkey_table={"country_id": "countries"},
            pkey_col="state_id",
        )

        # # ---------------------- Reindex pkeys/fkeys ----------------------
        # def _reindex_primary_and_foreign_keys(table_dict):
        #     pk_maps = {}
        #     for name, tbl in table_dict.items():
        #         pk_col = tbl.pkey_col
        #         if pk_col is None:
        #             continue
        #         if tbl.df[pk_col].duplicated().any():
        #             dup_cnt = int(tbl.df[pk_col].duplicated().sum())
        #             raise ValueError(
        #                 f"Table '{name}' contains {dup_cnt} duplicate values "
        #                 f"in primary key column '{pk_col}'. Deduplicate first."
        #             )
        #         unique_ids = (
        #             pd.Index(tbl.df[pk_col].unique())
        #             .sort_values()
        #             .astype(tbl.df[pk_col].dtype)
        #         )
        #         mapping = dict(zip(unique_ids, range(len(unique_ids))))
        #         pk_maps[name] = mapping

        #     for name, tbl in table_dict.items():
        #         pk_col = tbl.pkey_col
        #         if pk_col is None:
        #             continue
        #         tbl.df[pk_col] = tbl.df[pk_col].map(pk_maps[name]).astype("int64")

        #     for name, tbl in table_dict.items():
        #         for fk_col, ref_table in tbl.fkey_col_to_pkey_table.items():
        #             if ref_table not in pk_maps:
        #                 raise KeyError(
        #                     f"Reference table '{ref_table}' for FK '{name}.{fk_col}' "
        #                     "has no primary-key mapping."
        #                 )
        #             mapped_series = tbl.df[fk_col].map(pk_maps[ref_table])
        #             if mapped_series.isna().any():
        #                 tbl.df[fk_col] = mapped_series.astype("Int64")
        #             else:
        #                 tbl.df[fk_col] = mapped_series.astype("int64")

        # _reindex_primary_and_foreign_keys(tables)

        return Database(table_dict=tables)
