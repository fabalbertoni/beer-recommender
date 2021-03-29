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
        print(f"func:{f.__name__} args:[{args}, {kw}] took: {(te-ts):2.4f} sec")
        return result

    return wrap


def load_and_preprocess_beeradvocate_df():
    df = pd.read_csv(BEER_ADVOCATE_CSV)

    # Proper datetimes are more readable than unix timestamps
    df["review_time"] = pd.to_datetime(df["review_time"], unit="s")

    # Discard reviews where there's no user
    df.dropna(subset=["review_profilename"], inplace=True)

    # If a user reviewed the same beer multiple times, we keep the first review
    df.drop_duplicates(subset=["review_profilename", "beer_beerid"], inplace=True)

    return df


from enum import Enum


class SplitStrategy(Enum):
    USER = 0
    TIME = 1


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


def train_test_val(
    strategy: SplitStrategy, df_review: pd.DataFrame
) -> T.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a reviews DataFrame either by users or by time:
    - SplitStrategy.USER
      - 80% users are picked randomly -> Their reviews go to the
        training set.
      - From the rest, a random half goes to validation, and the
        rest to test.
    - SplitStrategy.TIME
      - First 80% of reviews are the training set
      - Following 10% are the validation set
      - The final 10% are the test set.
    """
    train_frac = 0.8

    if strategy is SplitStrategy.USER:
        users = df_review["review_profilename"].unique()
        # 80% of random users' reviews will be used for training
        # 10% for validation
        # 10% for test
        train_mask = np.random.rand(len(users)) < train_frac
        train_users = users[train_mask]

        # We split the remaining 20% in two.
        valtest_users = users[~train_mask]
        val_mask = np.random.rand(len(valtest_users)) < 0.5
        test_mask = ~val_mask

        val_users = valtest_users[val_mask]
        test_users = valtest_users[test_mask]

        split_train, split_val, split_test = (
            df_review[df_review["review_profilename"].isin(train_users)],
            df_review[df_review["review_profilename"].isin(val_users)],
            df_review[df_review["review_profilename"].isin(test_users)],
        )

    elif strategy is SplitStrategy.TIME:
        #
        #               split_train              split_val     split_test
        #
        #    |            80% time           |   10% time  |    10 % time   |
        # min_date                    train_maxdate    val_maxdate        max_date
        #
        min_date, max_date = (
            df_review["review_time"].min(),
            df_review["review_time"].max(),
        )
        range_in_days = max_date - min_date

        val_frac = (1 - train_frac) / 2

        train_maxdate = min_date + train_frac * range_in_days
        val_maxdate = train_maxdate + val_frac * range_in_days

        split_train, split_val, split_test = (
            df_review[df_review["review_time"] <= train_maxdate],
            df_review[
                (df_review["review_time"] > train_maxdate)
                & (df_review["review_time"] <= val_maxdate)
            ],
            df_review[df_review["review_time"] > val_maxdate],
        )
    else:
        raise NotImplementedError

    # We return copies just in case we want to modify the splits
    return split_train.copy(), split_val.copy(), split_test.copy()


@timing
def make_corr_graph(df_review: pd.DataFrame, rating_column: str) -> nx.Graph:
    def _keep_beer(dfb):
        return len(dfb) >= 20

    df_corr = (
        df_review.groupby("beer_beerid")
        .filter(_keep_beer)
        .pivot(index="review_profilename", columns="beer_beerid", values=rating_column)
        .corr()
    )

    return df_corr


# %%
df = load_and_preprocess_beeradvocate_df()
df_beer, df_brewery, df_review = split_beers_df(df)

# %%
suffix = "_split_by_user"
output_names = {
    "train": f"{BEER_ADVOCATE_PATH}train{suffix}",
    "val": f"{BEER_ADVOCATE_PATH}val{suffix}",
    "test": f"{BEER_ADVOCATE_PATH}test{suffix}",
}

# Here we can recompute the splits, or load them from files.
recompute_splits = False

if recompute_splits:

    train_df, val_df, test_df = train_test_val(SplitStrategy.USER, df_review)
    train_df.to_csv(f"{output_names['train']}.csv")
    val_df.to_csv(f"{output_names['val']}.csv")
    test_df.to_csv(f"{output_names['test']}.csv")
else:
    train_df, val_df, test_df = (
        pd.read_csv(f"{output_names['train']}.csv"),
        pd.read_csv(f"{output_names['val']}.csv"),
        pd.read_csv(f"{output_names['test']}.csv"),
    )

# %%
recompute_corrs = False
if recompute_corrs:
    df_corr = make_corr_graph(train_df, "review_overall")
else:
    df_corr = pd.read_csv(
        f"{BEER_ADVOCATE_PATH}df_corr.csv",
        dtype={"beer_beerid": str},
    ).set_index("beer_beerid")

# %%

# Set NaNs to zeros, and keep only most ranked
df_corr.fillna(0, inplace=True)

# For each beer, keep only the top 500 most correlated ones.
df_corr.mask(
    df_corr.rank(axis="columns", method="min", ascending=False) < 500, 0, inplace=True
)

# %%
recompute_G = False
if recompute_G:
    G = nx.from_pandas_adjacency(df_corr, create_using=nx.Graph)
    with open(f"{BEER_ADVOCATE_PATH}G_df_corr.pkl", "wb") as f:
        pickle.dump(G, f)
else:
    with open(f"{BEER_ADVOCATE_PATH}G_df_corr.pkl", "rb") as f:
        G = pickle.load(f)

# %%

#

Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
Gcc0 = G.subgraph(Gcc[0])

W = nx.to_numpy_array(Gcc0)

np.fill_diagonal(W, 0)
(w, v) = scipy.sparse.linalg.eigs(W, k=1, which="LM")
Wnorm = W / np.abs(w[0])

# %%

# This fills up the RAM :/
plt.figure(figsize=(10, 10))
sns.heatmap(Wnorm, cmap="Greys")


# %%
