import torch
import matplotlib.pyplot as plt


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

        # Después de cada epoch miramos el test loss
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
