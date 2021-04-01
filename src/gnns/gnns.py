# %%

# Imports, paths and functions
import pickle
import typing as T
from functools import wraps
from time import time
from enum import Enum

import alegnn.modules.architectures as architectures
import alegnn.utils.graphML as graphML
import altair as alt
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import torch
from IPython.display import display

from utils import train_model, plot_losses


# Same directory structure as the shared Google Drive folder
DATA_ROOT = "../../data/"
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
        print("Loading persisted train, val, test sets..")
        return (
            pd.read_csv(
                f"{BEER_ADVOCATE_PATH}df_user_rating_train.csv",
            ).set_index("review_profilename"),
            pd.read_csv(
                f"{BEER_ADVOCATE_PATH}df_user_rating_val.csv",
            ).set_index("review_profilename"),
            pd.read_csv(
                f"{BEER_ADVOCATE_PATH}df_user_rating_test.csv",
            ).set_index("review_profilename"),
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

        # So the columns and the index match.
        df_corr["beer_beerid"] = df_corr["beer_beerid"].astype(str)
        df_corr = df_corr.set_index("beer_beerid")
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


# %% [markdown]

## Procesamos los datos en un DataFrame de correlación y un grafo asociado
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

# df_corr = make_df_corr(df_user_rating_train, rating_column)

# G, Wnorm = make_graph(df_corr)

# By default it'll load from disk, no need to pass anything
df_corr = make_df_corr(None, None)

G, Wnorm = make_graph(df_corr)

########################################################################
########################################################################
########################################################################
########################################################################


# %% [markdown]

## Armamos un training, validation y test set
### Enfoque 1: Predecir rating para una cerveza sola.

# Random beer we want to train the GNN for.
the_beer = "6322"


def adjust_dfs_to_graph(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    G: nx.Graph,
    items_to_predict: T.List[str],
) -> T.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if len(items_to_predict) != 1:
        raise NotImplementedError("For now we're supporting prediction of one item")

    item_to_predict = items_to_predict[0]

    nodes_in_graph = list(G.nodes())

    # We remove the users' rows who didn't rate `the_beer`
    df_train_adjusted = df_user_rating_train.dropna(subset=[the_beer], axis="rows")
    df_val_adjusted = df_user_rating_val.dropna(subset=[the_beer], axis="rows")
    df_test_adjusted = df_user_rating_test.dropna(subset=[the_beer], axis="rows")

    # We remove those beers which were left out of the graph G
    beers_in_G = list(G.nodes())
    df_train_adjusted = df_train_adjusted[beers_in_G]
    df_val_adjusted = df_val_adjusted[beers_in_G]
    df_test_adjusted = df_test_adjusted[beers_in_G]

    return (df_train_adjusted, df_val_adjusted, df_test_adjusted)


def get_X_y(
    df: pd.DataFrame, W: np.ndarray, items_to_predict: T.List[str]
) -> T.Tuple[np.ndarray, np.ndarray]:
    if len(items_to_predict) != 1:
        raise NotImplementedError(
            "We do not support more than one element to predict yet."
        )

    the_item = items_to_predict[0]

    df_user_rating = df.copy()

    # `X` will have zeros for the beer, since we want to predict those values
    # We'll put them in `y`.
    the_item_rating = df_user_rating[the_item]
    df_user_rating[the_item] = 0

    X = (
        df_user_rating.fillna(0).to_numpy()  # We get rid of NaNs here
        # Here we convert each user's ratings into a 1xnumberOfNodex matrix
        # We might have more than 1 dimension later.
        .reshape((len(df_user_rating), 1, W.shape[0]))
    )

    # We put back the target item's rating, to assemble `y`
    df_user_rating[:] = 0
    df_user_rating[the_item] = the_item_rating

    y = (
        df_user_rating.fillna(0)
        .to_numpy()
        .reshape((len(df_user_rating), 1, W.shape[0]))
    )

    return X, y


# %%

# We remove the users' rows who didn't rate `the_beer`
df_user_rating_train, df_user_rating_val, df_user_rating_test = adjust_dfs_to_graph(
    df_user_rating_train, df_user_rating_val, df_user_rating_test, G, [the_beer]
)

# %%

recompute_X_y = False
if recompute_X_y:
    X_train, y_train = get_X_y(df_user_rating_train, Wnorm, [the_beer])
    X_val, y_val = get_X_y(df_user_rating_val, Wnorm, [the_beer])
    X_test, y_test = get_X_y(df_user_rating_test, Wnorm, [the_beer])

    with open(f"{BEER_ADVOCATE_PATH}X_train.npy", "wb") as f:
        np.save(f, X_train)

    with open(f"{BEER_ADVOCATE_PATH}y_train.npy", "wb") as f:
        np.save(f, y_train)

    with open(f"{BEER_ADVOCATE_PATH}X_val.npy", "wb") as f:
        np.save(f, X_val)

    with open(f"{BEER_ADVOCATE_PATH}y_val.npy", "wb") as f:
        np.save(f, y_val)

    with open(f"{BEER_ADVOCATE_PATH}X_test.npy", "wb") as f:
        np.save(f, X_test)

    with open(f"{BEER_ADVOCATE_PATH}y_test.npy", "wb") as f:
        np.save(f, y_test)

else:
    with open(f"{BEER_ADVOCATE_PATH}X_train.npy", "rb") as f:
        X_train = np.load(f)

    with open(f"{BEER_ADVOCATE_PATH}y_train.npy", "rb") as f:
        y_train = np.load(f)

    with open(f"{BEER_ADVOCATE_PATH}X_val.npy", "rb") as f:
        X_val = np.load(f)

    with open(f"{BEER_ADVOCATE_PATH}y_val.npy", "rb") as f:
        y_val = np.load(f)

    with open(f"{BEER_ADVOCATE_PATH}X_test.npy", "rb") as f:
        X_test = np.load(f)

    with open(f"{BEER_ADVOCATE_PATH}y_test.npy", "rb") as f:
        y_test = np.load(f)

print(f"Train shapes: {X_train.shape}; {y_train.shape}")
print(f"Val shapes: {X_val.shape}; {y_val.shape}")
print(f"Test shapes: {X_test.shape}; {y_test.shape}")


# %% [markdown]

## Entrenando la primera GNN

# %%
H = 64
number_of_beers = G.number_of_nodes()

train_data = torch.utils.data.TensorDataset(
    torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
)
val_data = torch.utils.data.TensorDataset(
    torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
)

# %%

gnn_model = architectures.LocalGNN(
    dimNodeSignals=[1, 64],
    nFilterTaps=[5],
    bias=True,
    nonlinearity=torch.nn.ReLU,
    nSelectedNodes=[number_of_beers],
    poolingFunction=graphML.NoPool,
    poolingSize=[1],
    dimReadout=[1],
    # Here we need the adjacency matrix from way above!
    GSO=torch.from_numpy(Wnorm).float(),
)
(trained_gnn_model, train_loss_gnn, val_loss_gnn) = train_model(
    gnn_model, train_data, val_data, n_epochs=1
)

# %%
plt.rcParams.update({"text.usetex": False})
plot_losses("GNN model", train_loss_gnn, val_loss_gnn)

# %%
