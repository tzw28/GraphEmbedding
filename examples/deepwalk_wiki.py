
import numpy as np

from ge.classify import read_node_label, Classifier
from ge import DeepWalk
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import sys
sys.path.append("./")
from layout.emb_fr import embedding_fr, COLOR_MAP, kmeans
import datetime


def evaluate_embeddings(embeddings):
    X, Y = read_node_label('./data/miserables/miserables_labels.txt')
    tr_frac = 0.8
    print("Training classifier using {:.2f}% nodes...".format(
        tr_frac * 100))
    clf = Classifier(embeddings=embeddings, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, tr_frac)


def plot_embeddings(embeddings,):
    X, Y = read_node_label('./data/miserables/miserables_labels.txt')

    emb_list = []
    for k in X:
        emb_list.append(embeddings[k])
    emb_list = np.array(emb_list)

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_list)

    color_idx = {}
    color_dict = {}
    for i in range(len(X)):
        color_idx.setdefault(Y[i][0], [])
        color_idx[Y[i][0]].append(i)
        color_dict[X[i]] = COLOR_MAP[int(Y[i][0])]


    plt.figure(figsize=(10, 10))
    ax = plt.axes([0.05, 0.05, 1 - 0.05 * 2, 1 - 0.05 * 2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    pos = {}
    for i, x in enumerate(X):
        pos[x] = node_pos[i]
    color_list = []
    cluster_color_list = []
    for node in G.nodes:
        color_list.append(color_dict[node])
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color=color_list, edgecolors='white', linewidths=0.7)
    #for c, idx in color_idx.items():

        #plt.scatter(node_pos[idx, 0], node_pos[idx, 1], label=c)  # c=node_colors)

    # plt.legend()
    t = datetime.datetime.now()
    plt.savefig("./pics/miserables/{}-{}-{}/dw-{}-{}-{}-emb-real.png".format(
        t.year, t.month, t.day,
        t.hour, t.minute, int(t.second)
    ))
    plt.cla()
    cluster = kmeans(embeddings, K=8)
    cluster_color = []
    for node in G.nodes:
        cluster_color.append(COLOR_MAP[cluster[node]])
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color=cluster_color, edgecolors='white', linewidths=0.7)
    plt.savefig("./pics/miserables/{}-{}-{}/dw-{}-{}-{}-emb-cluster.png".format(
        t.year, t.month, t.day,
        t.hour, t.minute, int(t.second)
    ))
    plt.cla()
    return color_dict, cluster


def plot_layout(G, embeddings, color_dict, cluster):
    color_list = []
    for node in G.nodes:
        color_list.append(color_dict[node])
    pos = embedding_fr(G, embeddings, cluster=cluster)
    plt.figure(figsize=(10, 10))
    ax = plt.axes([0.05, 0.05, 1 - 0.05 * 2, 1 - 0.05 * 2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    t = datetime.datetime.now()
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color=color_list, edgecolors='white', linewidths=0.7)
    nx.draw_networkx_edges(G, pos, width=0.7, alpha=0.2)
    plt.savefig("./pics/miserables/{}-{}-{}/dw-{}-{}-{}-real.png".format(
        t.year, t.month, t.day,
        t.hour, t.minute, int(t.second)
    ))
    plt.cla()
    cluster_color = []
    for node in G.nodes:
        cluster_color.append(COLOR_MAP[cluster[node]])
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color=cluster_color, edgecolors='white', linewidths=0.7)
    nx.draw_networkx_edges(G, pos, width=0.7, alpha=0.2)
    plt.savefig("./pics/miserables/{}-{}-{}/dw-{}-{}-{}-cluster.png".format(
        t.year, t.month, t.day,
        t.hour, t.minute, int(t.second)
    ))


if __name__ == "__main__":
    G = nx.read_edgelist('./data/miserables/miserables_edgelist.txt',
                         create_using=nx.Graph(), nodetype=None, data=[('weight', int)])

    model = DeepWalk(G, walk_length=10, num_walks=80, workers=1)
    model.train(window_size=5, iter=3)
    embeddings = model.get_embeddings()

    evaluate_embeddings(embeddings)
    color_dict, cluster = plot_embeddings(embeddings)
    plot_layout(G, embeddings, color_dict, cluster)
