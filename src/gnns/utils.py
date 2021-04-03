import typing as T
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Same directory structure as the shared Google Drive folder
DATA_ROOT = "../../data/"
BEER_ADVOCATE_PATH = f"{DATA_ROOT}BeerAdvocate/"
BEER_ADVOCATE_CSV = f"{BEER_ADVOCATE_PATH}beer_reviews.csv"


def train_model(model, train_data, test_data, batch_size=5, n_epochs=40, epsilon=0.005):

    optimizer = torch.optim.Adam(model.parameters(), lr=epsilon)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_data, batch_size=batch_size, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_data, batch_size=len(test_data), shuffle=True
    )

    train_loss = []
    test_loss = []
    for epoch in range(n_epochs):
        for x_batch, y_batch in train_loader:

            model.zero_grad()
            y_hat = model.forward(x_batch)
            loss = rating_loss(y_hat, y_batch)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        for x_batch, y_batch in test_loader:
            y_hattest = model(x_batch)
            test_loss.append(rating_loss(y_hattest, y_batch).item())
            # print("y_batch: "+str(y_batch))
            # print("y_hattest: "+str(y_hattest))

    return (model, train_loss, test_loss)


def rating_loss(prediction, ground_truth):
    # batchSize x dimFeatures x numberNodes
    # We only care about the item we want to predict
    ix_item = ground_truth[0, 0, :].nonzero(as_tuple=False)

    return torch.mean((prediction[:, :, ix_item] - ground_truth[:, :, ix_item]) ** 2)


def plot_losses(model_name, train_loss, test_loss):
    plt.figure()
    plt.semilogy(train_loss)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(model_name + " - Train Loss")
    plt.grid()
    plt.show()
    plt.figure()
    plt.plot(test_loss)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(model_name + " - Test Loss")
    plt.grid()
    plt.show()


def load_datasets_X_y() -> T.List[np.ndarray]:
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

    return [X_train, y_train, X_val, y_val, X_test, y_test]


def save_datasets_X_y(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
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


def load_train_val_test_df():
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


def load_df_corr() -> pd.DataFrame:
    print("Loading persisted df_corr..")
    df_corr = pd.read_csv(f"{BEER_ADVOCATE_PATH}df_corr.csv")

    # So the columns and the index match.
    df_corr["beer_beerid"] = df_corr["beer_beerid"].astype(str)
    df_corr = df_corr.set_index("beer_beerid")
    return df_corr
