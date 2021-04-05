# %%
# Imports, paths and functions
import typing as T
from functools import wraps
from time import time

import alegnn.modules.architectures as architectures
import alegnn.utils.graphML as graphML
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy

import torch

from utils import (
    train_model,
    plot_losses,
    load_datasets_X_y,
    # save_datasets_X_y,
    BEER_ADVOCATE_CSV,
    load_train_val_test_df,
    load_df_corr,
)

# RECOMPUTES
recompute_train_val_test = False
recompute_df_corr = False
recompute_X_y = True


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
def train_val_test(df_user_rating: pd.DataFrame) -> pd.DataFrame:

    split_train_val_proportions = (0.8, 0.1)

    indexes_arr = np.arange(len(df_user_rating))
    np.random.shuffle(indexes_arr)

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
) -> pd.DataFrame:

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


if recompute_train_val_test:
    df_user_rating_train, df_user_rating_val, df_user_rating_test = train_val_test(
        df_user_rating
    )
else:
    (
        df_user_rating_train,
        df_user_rating_val,
        df_user_rating_test,
    ) = load_train_val_test_df()

# %%


if recompute_df_corr:
    df_corr = make_df_corr(df_user_rating_train, rating_column)
else:
    df_corr = load_df_corr()

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
    column_to_predict: str,
) -> T.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    # We remove the users' rows who didn't rate `item_to_predict`
    df_train_adjusted = df_train.dropna(subset=[column_to_predict], axis="rows")
    df_val_adjusted = df_val.dropna(subset=[column_to_predict], axis="rows")
    df_test_adjusted = df_test.dropna(subset=[column_to_predict], axis="rows")

    # We remove those beers which were left out of the graph G
    beers_in_G = list(G.nodes())
    df_train_adjusted = df_train_adjusted[beers_in_G]
    df_val_adjusted = df_val_adjusted[beers_in_G]
    df_test_adjusted = df_test_adjusted[beers_in_G]

    return (df_train_adjusted, df_val_adjusted, df_test_adjusted)


def get_X_y(
    df: pd.DataFrame, column_to_predict: str
) -> T.Tuple[np.ndarray, np.ndarray]:

    df_user_rating = df.copy()

    # `X` will have zeros for the beer, since we want to predict those values
    # We'll put them in `y`.
    the_item_rating = df_user_rating[column_to_predict]
    df_user_rating[column_to_predict] = 0

    X = (
        df_user_rating.fillna(0).to_numpy()  # We get rid of NaNs here
        # Here we convert each user's ratings into a 1xnumberOfNodex matrix
        # We might have more than 1 dimension later.
        .reshape((len(df_user_rating), 1, len(df_user_rating.columns)))
    )

    # We put back the target item's rating, to assemble `y`
    df_user_rating = pd.DataFrame(
        0, index=df_user_rating.index, columns=df_user_rating.columns
    )
    df_user_rating[column_to_predict] = the_item_rating

    y = (
        df_user_rating.fillna(0)
        .to_numpy()
        .reshape((len(df_user_rating), 1, len(df_user_rating.columns)))
    )

    return X, y


@timing
def get_X_y_columns(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    G: nx.Graph,
    columns: T.Optional[T.List[str]] = None,
) -> T.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    assert (df_train.columns == df_val.columns).all() and (
        df_train.columns == df_test.columns
    ).all()

    if columns is not None and not columns:
        raise ValueError("If provided, the columns list must not be empty.")
    elif not columns:
        print("Will process ALL columns")
        columns = list(df_train.columns)
    else:
        print(f"Will process columns: {columns}")

    X_train_lst, y_train_lst, X_val_lst, y_val_lst, X_test_lst, y_test_lst = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    five_percent_step = int(len(columns) * 0.05)
    for i, column in enumerate(columns):
        if five_percent_step and i % five_percent_step == 0:
            print(f"Column {i} of {len(columns)}")

        try:
            df_train_filtered, df_val_filtered, df_test_filtered = adjust_dfs_to_graph(
                df_train, df_val, df_test, G, column
            )

            X_train, y_train = get_X_y(df_train_filtered, column)
            X_val, y_val = get_X_y(df_val_filtered, column)
            X_test, y_test = get_X_y(df_test_filtered, column)

            X_train_lst.append(X_train)
            y_train_lst.append(y_train)
            X_val_lst.append(X_val)
            y_val_lst.append(y_val)
            X_test_lst.append(X_test)
            y_test_lst.append(y_test)
        except Exception as e:
            print(f"FAILED ON COLUMN: {column}: {e}")
            continue

    return (
        np.concatenate(X_train_lst, axis=0),
        np.concatenate(y_train_lst, axis=0),
        np.concatenate(X_val_lst, axis=0),
        np.concatenate(y_val_lst, axis=0),
        np.concatenate(X_test_lst, axis=0),
        np.concatenate(y_test_lst, axis=0),
    )


# %%

if recompute_X_y:
    print("Recomputing X, y...")
    (X_train, y_train, X_val, y_val, X_test, y_test) = get_X_y_columns(
        df_user_rating_train, df_user_rating_val, df_user_rating_test, G, [the_beer]
    )

    # save_datasets_X_y(X_train, y_train, X_val, y_val, X_train, y_test)

else:
    # This loads arrays just for `the_beer`
    [X_train, y_train, X_val, y_val, X_test, y_test] = load_datasets_X_y()

print(f"Train shapes: {X_train.shape}; {y_train.shape}")
print(f"Val shapes: {X_val.shape}; {y_val.shape}")
print(f"Test shapes: {X_test.shape}; {y_test.shape}")


# %% [markdown]

## Entrenando la primera GNN

# %%
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(f"Device: {device}")

train_data = torch.utils.data.TensorDataset(
    torch.from_numpy(X_train).to(device).float(),
    torch.from_numpy(y_train).to(device).float(),
)
val_data = torch.utils.data.TensorDataset(
    torch.from_numpy(X_val).to(device).float(),
    torch.from_numpy(y_val).to(device).float(),
)

# %%

gnn_model = architectures.LocalGNN(
    dimNodeSignals=[1, 64],
    nFilterTaps=[5],
    bias=True,
    nonlinearity=torch.nn.ReLU,
    nSelectedNodes=[Wnorm.shape[0]],  # Number of nodes
    poolingFunction=graphML.NoPool,
    poolingSize=[1],
    dimReadout=[1],
    # Here we need the adjacency matrix from way above!
    GSO=torch.from_numpy(Wnorm).to(device).float(),
)
gnn_model.to(device)

(trained_gnn_model, train_loss_gnn, val_loss_gnn) = train_model(
    gnn_model, train_data, val_data, n_epochs=3
)

# %%
print(f"train_loss_gnn: {train_loss_gnn}")
print(f"val_loss_gnn: {val_loss_gnn}")

# This doesn't work
# gnn_model.save("gnn40epochs.pth")

plt.rcParams.update({"text.usetex": False})
plot_losses("GNN model", train_loss_gnn, val_loss_gnn)

# %% [markdown]

## Probemos con más cervezas a la vez
### Elegimos cervezas con muchas reviews

# %%
n_target_beers = 1
n_ratings_by_beer = (~df_user_rating_train.isnull()).sum()
random_beers = (
    n_ratings_by_beer.sort_values(ascending=False)
    .iloc[:200]
    .sample(n_target_beers)
    .index
)
random_beer_names = df_beer.loc[random_beers.astype(int), "beer_name"]
random_beers = [str(beer_id) for beer_id in random_beers]
print(random_beer_names)

# %%
(X_train, y_train, X_val, y_val, X_test, y_test) = get_X_y_columns(
    df_user_rating_train, df_user_rating_val, df_user_rating_test, G, random_beers
)
# %%

train_data = torch.utils.data.TensorDataset(
    torch.from_numpy(X_train).to(device).float(),
    torch.from_numpy(y_train).to(device).float(),
)
val_data = torch.utils.data.TensorDataset(
    torch.from_numpy(X_val).to(device).float(),
    torch.from_numpy(y_val).to(device).float(),
)

# %%

gnn_model = architectures.LocalGNN(
    dimNodeSignals=[1, 64],
    nFilterTaps=[5],
    bias=True,
    nonlinearity=torch.nn.ReLU,
    nSelectedNodes=[Wnorm.shape[0]],  # Number of nodes
    poolingFunction=graphML.NoPool,
    poolingSize=[1],
    dimReadout=[1],
    # Here we need the adjacency matrix from way above!
    GSO=torch.from_numpy(Wnorm).to(device).float(),
)
gnn_model.to(device)

(trained_gnn_model, train_loss_gnn, val_loss_gnn) = train_model(
    gnn_model, train_data, val_data, n_epochs=3
)

# %%
print(f"train_loss_gnn: {train_loss_gnn}")
print(f"val_loss_gnn: {val_loss_gnn}")

# This doesn't work
# gnn_model.save("gnn40epochs.pth")

plt.rcParams.update({"text.usetex": False})
plot_losses("GNN model", train_loss_gnn, val_loss_gnn)
# %%
