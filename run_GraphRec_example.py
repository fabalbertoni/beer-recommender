import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import pickle
import numpy as np
import time
import random
from collections import defaultdict
from UV_Encoders import UV_Encoder
from UV_Aggregators import UV_Aggregator
from Social_Encoders import Social_Encoder
from Social_Aggregators import Social_Aggregator
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import datetime
import argparse
import os
import matplotlib.pyplot as plt

"""
GraphRec: Graph Neural Networks for Social Recommendation.
Wenqi Fan, Yao Ma, Qing Li, Yuan He, Eric Zhao, Jiliang Tang, and Dawei Yin.
In Proceedings of the 28th International Conference on World Wide Web (WWW), 2019. Preprint[https://arxiv.org/abs/1902.07243]

If you use this code, please cite our paper:
```
@inproceedings{fan2019graph,
  title={Graph Neural Networks for Social Recommendation},
  author={Fan, Wenqi and Ma, Yao and Li, Qing and He, Yuan and Zhao, Eric and Tang, Jiliang and Yin, Dawei},
  booktitle={WWW},
  year={2019}
}
```

"""


class GraphRec(nn.Module):

    def __init__(self, enc_u, enc_v_history, r2e):
        super(GraphRec, self).__init__()
        self.enc_u = enc_u
        self.enc_v_history = enc_v_history
        self.embed_dim = enc_u.embed_dim

        self.w_ur1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_ur2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr1 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_vr2 = nn.Linear(self.embed_dim, self.embed_dim)
        self.w_uv1 = nn.Linear(self.embed_dim * 2, self.embed_dim)
        self.w_uv2 = nn.Linear(self.embed_dim, 16)
        self.w_uv3 = nn.Linear(16, 5)
        self.r2e = r2e
        self.bn1 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn2 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn3 = nn.BatchNorm1d(self.embed_dim, momentum=0.5)
        self.bn4 = nn.BatchNorm1d(16, momentum=0.5)
        self.criterion = nn.MSELoss()

    def forward(self, nodes_u, nodes_v):
        embeds_u = self.enc_u(nodes_u)
        embeds_v = self.enc_v_history(nodes_v)

        x_u = F.relu(self.bn1(self.w_ur1(embeds_u)))
        x_u = F.dropout(x_u, training=self.training)
        x_u = self.w_ur2(x_u)
        x_v = F.relu(self.bn2(self.w_vr1(embeds_v)))
        x_v = F.dropout(x_v, training=self.training)
        x_v = self.w_vr2(x_v)

        x_uv = torch.cat((x_u, x_v), 1)
        x = F.relu(self.bn3(self.w_uv1(x_uv)))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.bn4(self.w_uv2(x)))
        x = F.dropout(x, training=self.training)
        scores = self.w_uv3(x)
        return scores.squeeze()

    def loss(self, nodes_u, nodes_v, labels_list):
        scores = self.forward(nodes_u, nodes_v)
        return self.criterion(scores, labels_list)


def train(model, device, train_loader, optimizer, epoch, best_rmse, best_mae):
    model.train()
    running_loss = 0.0
    loss_values = []
    for i, data in enumerate(train_loader, 0):
        batch_nodes_u, batch_nodes_v, labels_list = data
        # TODO: Not sure why we end up with shape (batch_size, 1, 5).
        # Investigate if we can get rid of that 1 from the start
        labels_list = torch.squeeze(labels_list)
        optimizer.zero_grad()
        loss = model.loss(batch_nodes_u.to(device), batch_nodes_v.to(device), labels_list.to(device))
        loss.backward(retain_graph=True)
        optimizer.step()
        running_loss += loss.item()
        if i % 1000 == 0 and i > 0:
            print('[%d, %5d] loss: %.3f, The best rmse/mae: %.6f / %.6f' % (
                epoch, i, running_loss / (1000 * i), best_rmse, best_mae))
            # running_loss = 0.0

    loss_values.append(running_loss / len(train_loader))

    return loss_values


def test(model, device, test_loader):
    model.eval()
    tmp_pred = []
    target = []
    with torch.no_grad():
        for test_u, test_v, tmp_target in test_loader:
            test_u, test_v, tmp_target = test_u.to(device), test_v.to(device), tmp_target.to(device)
            val_output = model.forward(test_u, test_v)
            tmp_pred.append(list(val_output.data.cpu().numpy()))
            target.append(list(tmp_target.data.cpu().numpy()))

    tmp_pred = np.array(sum(tmp_pred, []))
    target = np.array(sum(target, []))
    # TODO: Same as above, not sure we we have an extra dim.
    target = np.squeeze(target)
    expected_rmses = [sqrt(mean_squared_error(tmp_pred[:, i], target[:, i])) for i in range(target.shape[1])]
    maes = [mean_absolute_error(tmp_pred[:, i], target[:, i]) for i in range(target.shape[1])]
    return expected_rmses, maes


def run(data, batch_size=128, embed_dim=64, lr=0.001, test_batch_size=1000, epochs=100, use_similarity=False, gpu='0'):

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    history_u_lists, history_ur_lists, history_v_lists, history_vr_lists, train_u, train_v, train_r, test_u, test_v, test_r, item_adj_lists = data
    """
    ## toy dataset
    history_u_lists, history_ur_lists:  user's purchased history (item set in training set), and his/her rating score (dict)
    history_v_lists, history_vr_lists:  user set (in training set) who have interacted with the item, and rating score (dict)

    train_u, train_v, train_r: training_set (user, item, rating)
    test_u, test_v, test_r: testing set (user, item, rating)

    # please add the validation set

    social_adj_lists: items connected neighborhoods
    ratings_list: rating value from 1.0 to 5.0 (9 possible values), for 5 fields.
                  Let's use 9 embeddings of size n for each field, concatenate them, and end up with
                  a 5n size rating embedding
    """

    trainset = torch.utils.data.TensorDataset(torch.LongTensor(train_u), torch.LongTensor(train_v),
                                              torch.FloatTensor(train_r))
    testset = torch.utils.data.TensorDataset(torch.LongTensor(test_u), torch.LongTensor(test_v),
                                             torch.FloatTensor(test_r))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=True)
    num_users = history_u_lists.__len__()
    num_items = history_v_lists.__len__()
    # Not used for now
    # num_ratings = ratings_list.__len__()

    u2e = nn.Embedding(num_users, embed_dim).to(device)
    v2e = nn.Embedding(num_items, embed_dim).to(device)

    # Instead of using 9 embeddings, use a FC to compute the embedding given 5 review values.
    r2e = nn.Sequential(
        nn.Linear(5, 256),
        nn.ReLU(),
        nn.Linear(256, embed_dim),
    ).to(device)

    # user feature
    # features: item * rating
    agg_u_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=True)
    enc_u_history = UV_Encoder(u2e, embed_dim, history_u_lists, history_ur_lists, agg_u_history, cuda=device, uv=True)
    # neighobrs
    # Removing this since we don't have user interaction data
    # agg_u_social = Social_Aggregator(lambda nodes: enc_u_history(nodes).t(), u2e, embed_dim, cuda=device)
    # enc_u = Social_Encoder(lambda nodes: enc_u_history(nodes).t(), embed_dim, social_adj_lists, agg_u_social,
    #                        base_model=enc_u_history, cuda=device)

    # item feature: user * rating
    agg_v_history = UV_Aggregator(v2e, r2e, u2e, embed_dim, cuda=device, uv=False)
    enc_v_history = UV_Encoder(v2e, embed_dim, history_v_lists, history_vr_lists, agg_v_history, cuda=device, uv=False)

    # we do have item similarity data so we will add this piece and see if it works
    # item adjancency
    # TODO Use item similarity later (enc_v).
    agg_v_similarity = Social_Aggregator(lambda nodes: enc_v_history(nodes).t(), v2e, embed_dim, cuda=device)
    enc_v = Social_Encoder(lambda nodes: enc_v_history(nodes).t(), embed_dim, item_adj_lists, agg_v_similarity,
                           base_model=enc_v_history, cuda=device)

    # model
    if use_similarity:
        graphrec = GraphRec(enc_u_history, enc_v, r2e).to(device)
    else:
        graphrec = GraphRec(enc_u_history, enc_v_history, r2e).to(device)
    optimizer = torch.optim.RMSprop(graphrec.parameters(), lr=lr, alpha=0.9)

    best_rmse = 9999.0
    rmse_history = []
    best_mae = 9999.0
    mae_history = []
    endure_count = 0
    fields = ['overall', 'review_aroma', 'review_appearance', 'review_palate', 'review_taste']
    loss_history = []

    for epoch in range(1, epochs + 1):

        loss_values = train(graphrec, device, train_loader, optimizer, epoch, best_rmse, best_mae)
        loss_history.extend(loss_values)
        expected_rmses, maes = test(graphrec, device, test_loader)
        # please add the validation set to tune the hyper-parameters based on your datasets.

        print(f'Debug: {expected_rmses}, {maes}')

        for idx, (expected_rmse, mae) in enumerate(zip(expected_rmses, maes)):
            print(f'Metrics for field {fields[idx]}:')
            print(expected_rmse)
            print(mae)

        expected_rmse, mae = expected_rmses[0], maes[0]

        rmse_history.append(expected_rmse)
        mae_history.append(mae)

        # early stopping (no validation set in toy dataset)
        if best_rmse > expected_rmse:
            best_rmse = expected_rmse
            best_mae = mae
            endure_count = 0
        else:
            endure_count += 1
        print("rmse: %.4f, mae:%.4f " % (expected_rmse, mae))

        if endure_count > 5:
            break

    plt.title("Test metrics")
    plt.plot(rmse_history, label='RMSE')
    plt.plot(mae_history, label='MAE')
    plt.legend()
    plt.show()

    plt.title("Loss history")
    plt.plot(loss_history, label='Train Loss')
    plt.legend()
    plt.show()


# if __name__ == "__main__":
#     main()
