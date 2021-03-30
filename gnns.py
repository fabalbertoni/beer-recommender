# %%

# Imports and paths
import pickle
import typing as T
from functools import wraps
from time import time
from enum import Enum
from IPython.display import display

import altair as alt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

# Same directory structure as the shared Google Drive folder
DATA_ROOT = "data/"
BEER_ADVOCATE_PATH = f"{DATA_ROOT}BeerAdvocate/"
BEER_ADVOCATE_CSV = f"{BEER_ADVOCATE_PATH}beer_reviews.csv"


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func:{f.__name__} took: {(te-ts):2.4f} sec")
        return result

    return wrap


@timing
def load_and_preprocess_beeradvocate_df():
    df = pd.read_csv(BEER_ADVOCATE_CSV)

    # Proper datetimes are more readable than unix timestamps
    df["review_time"] = pd.to_datetime(df["review_time"], unit="s")

    # Discard reviews where there's no user
    df.dropna(subset=["review_profilename"], inplace=True)

    # If a user reviewed the same beer multiple times, we keep the first review
    df.drop_duplicates(subset=["review_profilename", "beer_beerid"], inplace=True)

    def _keep_beer(dfg):
        return len(dfg) >= 20

    # This discards ratings for items with under 20 ratings
    # NOTE: If we don't do this then, due to a bug in Pandas, won't be able
    # to pivot the table.
    # https://github.com/pandas-dev/pandas/issues/26314
    df = df.groupby("beer_beerid").filter(_keep_beer)

    return df


@timing
def split_beers_df(
    df: pd.DataFrame,
) -> T.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the original "full DataFrame" into separate dateaframes
    for reviews, beers and breweries.
    """
    df_beer = (
        df[["beer_beerid", "brewery_id", "beer_name", "beer_style", "beer_abv"]]
        .drop_duplicates(subset="beer_beerid")
        .set_index("beer_beerid")
    )
    df_brewery = (
        df[["brewery_id", "brewery_name"]]
        .drop_duplicates(subset="brewery_id")
        .set_index("brewery_id")
    )
    df_review = df[
        list(
            set(
                ["beer_beerid", "brewery_id"]
                + list(set(df.columns) - set(df_beer.columns) - set(df_brewery.columns))
            )
        )
    ]

    return df_beer, df_brewery, df_review


@timing
def make_user_rating_matrix(
    df_rating: pd.DataFrame, rating_column: str
) -> pd.DataFrame:
    return df_rating.pivot(
        index="review_profilename", columns="beer_beerid", values=rating_column
    )


@timing
def train_val_test(
    df_user_rating: pd.DataFrame, recompute: bool = False
) -> pd.DataFrame:
    if not recompute:
        return (
            pd.read_csv(f"{BEER_ADVOCATE_PATH}df_user_rating_train.csv"),
            pd.read_csv(f"{BEER_ADVOCATE_PATH}df_user_rating_val.csv"),
            pd.read_csv(f"{BEER_ADVOCATE_PATH}df_user_rating_test.csv"),
        )

    split_train_val_proportions = (0.8, 0.1)

    indexes_arr = np.arange(len(df_user_rating))
    shuffled_indexes_arr = np.random.shuffle(indexes_arr)

    train_max = int(split_train_val_proportions[0] * len(df_user_rating))
    val_max = train_max + int(split_train_val_proportions[1] * len(df_user_rating))
    return (
        df_user_rating.iloc[:train_max].copy(),
        df_user_rating.iloc[train_max:val_max].copy(),
        df_user_rating.iloc[val_max:].copy(),  # The rest is for the test set
    )


@timing
def make_df_corr(
    df_user_rating: pd.DataFrame,
    rating_column: str,
    most_correlated: int = 80,
    min_ratings: int = 20,
    recompute: bool = False,
) -> T.Tuple[pd.DataFrame, nx.Graph, np.ndarray]:

    if not recompute:
        print("Loading persisted df_corr..")
        df_corr = pd.read_csv(f"{BEER_ADVOCATE_PATH}df_corr.csv")
        return df_corr

    df_corr = df_user_rating.corr(min_periods=min_ratings)

    # Set NaNs to zeros, and keep only most ranked
    df_corr.fillna(0, inplace=True)

    # For each beer, keep only the top 500 most correlated ones.
    df_corr.mask(
        df_corr.rank(axis="columns", method="min", ascending=False) < most_correlated,
        0,
        inplace=True,
    )

    return df_corr


@timing
def make_graph(df_corr: pd.DataFrame) -> T.Tuple[nx.Graph, np.ndarray]:
    G = nx.from_pandas_adjacency(df_corr, create_using=nx.Graph)

    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    Gcc0 = G.subgraph(Gcc[0])

    W = nx.to_numpy_array(Gcc0)

    np.fill_diagonal(W, 0)
    (w, v) = scipy.sparse.linalg.eigs(W, k=1, which="LM")
    Wnorm = W / np.abs(w[0])

    return Gcc0, Wnorm


# %%

# Load and split in entities
df = load_and_preprocess_beeradvocate_df()
df_beer, df_brewery, df_review = split_beers_df(df)
rating_column = "review_overall"
# %%

# Get df_user_rating and split it in train, val and test
df_user_rating = make_user_rating_matrix(df_review, rating_column)

df_user_rating_train, df_user_rating_val, df_user_rating_test = train_val_test(
    df_user_rating
)

# %%

df_corr = make_df_corr(df_user_rating_train, rating_column)

G, Wnorm = make_graph(df_corr)
