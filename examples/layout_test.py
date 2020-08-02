import os
from ge import Node2Vec, DeepWalk, LINE, Struc2Vec
from layout.emb_fr import COLOR_MAP, kmeans, embedding_fr
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from layout.graph_reading import (
    read_citeseer_graph,
    read_cora_graph,
    read_miserables_graph,
    read_science_graph,
    read_facebook_graph
)
import datetime, json
from layout.mean_shift import mean_shift, optics, dbscan
from layout.emb_layout import generate_emb_adj_graph

plt.switch_backend('agg') 

emb_method_dict = {
    "node2vec": Node2Vec,
    "deepwalk": DeepWalk,
    "line": LINE,
    "struc2vec": Struc2Vec
}

layout_method_dict = {
    "circular_layout": nx.circular_layout,
    "kamada_kawai_layout": nx.kamada_kawai_layout,
    "random_layout": nx.random_layout,
    "shell_layout": nx.shell_layout,
    "spring_layout": nx.spring_layout,
    #"spectral_layout": nx.spectral_layout,
    # "planar_layout": nx.planar_layout,
    "fruchterman_reingold_layout": nx.fruchterman_reingold_layout,
}

reader_dict = {
    "citeseer": read_citeseer_graph,
    "cora": read_cora_graph,
    "miserables": read_miserables_graph,
    "science": read_science_graph,
    "facebook": read_facebook_graph
}


def run_tests():
    with open("small_tests.json", "r") as f:
        string = f.read()
        tests = json.loads(string)
    save_path = make_save_path()
    for test in tests:
        graph_name = test['graph']
        graph_tests = test['tests']
        for graph_test in graph_tests:
            emb_method = graph_test['method']
            cur_save_path = save_path + "/{}-{}".format(graph_name, emb_method)
            os.mkdir(cur_save_path)
            run_single_test(
                emb_method, graph_name, cur_save_path,
                graph_test['emb_params'],
                graph_test['train_params'],
                graph_test['layout_params'],
                graph_test['emb_pic_params'],
                graph_test['layout_pic_params'],
            )


def make_save_path():
    t = datetime.datetime.now()
    path = "./pics/{}-{}-{}".format(
        t.year, t.month, t.day
    )
    if not os.path.exists(path):
        os.mkdir(path)
    path += "/{}-{}".format(
        t.hour, t.minute
    )
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def run_single_test(emb_method, graph_name, save_path, emb_params, train_params, layout_params,
                    emb_pic_params, layout_pic_params):
    G, labels = read_graph(graph_name)
    emb = emb_method_dict[emb_method]
    model= emb(G, **emb_params)
    model.train(**train_params)
    vectors = model.get_embeddings()

    clusters, cluster_color = compute_cluster_color(G.nodes, vectors, layout_params['k'])
    label_color = compute_label_color(G.nodes, labels)

    emb_save_path = save_path + "/"
    for param, value in emb_params.items():
        emb_save_path += str(param) + str(value) + "-"
    emb_save_path = emb_save_path[:-1] + "-{}.png"
    tsne_pos = tsne(G, vectors)
    draw_emb_pictures(G, tsne_pos, cluster_color, label_color, emb_save_path, emb_pic_params)

    for layout_method in layout_method_dict.keys():
        layout = layout_method_dict[layout_method]

        layout1_save_path = save_path + "/{}(original)-".format(layout_method)
        for param, value in layout_params.items():
            layout1_save_path += str(param) + str(value) + "-"
        layout1_save_path = layout1_save_path[:-1] + "-{}.png"
        layout_pos_1 = layout(G)
        draw_layout_pictures(G, layout_pos_1, cluster_color, label_color, layout1_save_path, layout_pic_params)

        '''
        new_G = generate_emb_adj_graph(G, vectors)
        layout1_save_path = save_path + "/{}-".format(layout_method)
        for param, value in layout_params.items():
            layout1_save_path += str(param) + str(value) + "-"
        layout1_save_path = layout1_save_path[:-1] + "-{}.png"
        layout_pos_1 = layout(new_G)
        draw_layout_pictures(G, layout_pos_1, cluster_color, label_color, layout1_save_path, layout_pic_params)

        layout2_save_path = save_path + "/{}-".format(layout_method)
        layout_params['wa'] = 0
        new_G = generate_emb_adj_graph(G, vectors, wa=0)
        for param, value in layout_params.items():
            layout2_save_path += str(param) + str(value) + "-"
        layout2_save_path = layout2_save_path[:-1] + "-{}.png"
        layout_pos_2 = layout(new_G)
        draw_layout_pictures(G, layout_pos_2, cluster_color, label_color, layout2_save_path, layout_pic_params)

        layout1_save_path = save_path + "/{}(cluster)-".format(layout_method)
        layout_params['wa'] = 1
        new_G = generate_emb_adj_graph(G, vectors, clusters=clusters)
        for param, value in layout_params.items():
            layout1_save_path += str(param) + str(value) + "-"
        layout1_save_path = layout1_save_path[:-1] + "-{}.png"
        layout_pos_1 = layout(new_G)
        draw_layout_pictures(G, layout_pos_1, cluster_color, label_color, layout1_save_path, layout_pic_params)
        '''
        layout2_save_path = save_path + "/{}(em+)-".format(layout_method)
        new_G = generate_emb_adj_graph(G, vectors, clusters=clusters)
        for param, value in layout_params.items():
            layout2_save_path += str(param) + str(value) + "-"
        layout2_save_path = layout2_save_path[:-1] + "-{}.png"
        layout_pos_2 = layout(new_G)
        draw_layout_pictures(G, layout_pos_2, cluster_color, label_color, layout2_save_path, layout_pic_params)


def read_graph(graph_name):
    reader = reader_dict[graph_name]
    return reader()

def tsne(G, vectors):
    vector_list = []
    for key in vectors.keys():
        vector_list.append(vectors[key])
    nodes = list(G.nodes)
    tsne = TSNE(n_components=2)
    tsne.fit(vector_list)
    newX = tsne.fit_transform(vector_list)
    pos = {}
    for i in range(0, len(newX)):
        pos[nodes[i]] = newX[i]
    return pos


def compute_cluster_color(nodes, vectors, k):
    vector_list = []
    for key in vectors.keys():
        vector_list.append(vectors[key])
    tsne = TSNE(n_components=3)
    tsne.fit(vector_list)
    newX = tsne.fit_transform(vector_list)
    temp_vectors = {}
    temp_nodes = list(nodes)
    for i in range(0, len(newX)):
        temp_vectors[temp_nodes[i]] = newX[i]
    clusters = kmeans(vectors, K=k)
    # clusters = mean_shift(temp_vectors)
    # clusters = dbscan(vectors)
    # clusters = dbscan(temp_vectors)
    # clusters = optics(vectors)
    color_list = []
    for node in nodes:
        c = COLOR_MAP[clusters[node]]
        color_list.append(c)
    return clusters, color_list


def compute_label_color(nodes, labels):
    if not labels:
        return []
    color_list = []
    for node in nodes:
        c = COLOR_MAP[labels[node]]
        color_list.append(c)
    return color_list


def draw_emb_pictures(G, pos, cluster_color, label_color, save_path, emb_pic_params):
    if label_color:
        label_save_path = save_path.format("label")
        draw_graph_picture(
            G, pos, label_color, label_save_path, emb_pic_params, show_edge=False
        )
    cluster_save_path = save_path.format("cluster")
    draw_graph_picture(
        G, pos, cluster_color, cluster_save_path, emb_pic_params, show_edge=False
    )


def draw_layout_pictures(G, pos, cluster_color, label_color, save_path, layout_pic_params):
    if label_color:
        label_save_path = save_path.format("label")
        draw_graph_picture(
            G, pos, label_color, label_save_path, layout_pic_params, show_edge=True
        )
    cluster_save_path = save_path.format("cluster")
    draw_graph_picture(
        G, pos, cluster_color, cluster_save_path, layout_pic_params, show_edge=True
    )
    

def draw_graph_picture(G, pos, color, save_path, pic_params, show_edge=True):
    plt.figure(figsize=(pic_params['fig_size'], pic_params['fig_size']))
    gap = pic_params['fig_gap']
    ax = plt.axes([gap, gap, 1 - gap * 2, 1 - gap * 2])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    nx.draw_networkx_nodes(
        G, pos, node_size=pic_params['node_size'], node_color=color,
        edgecolors=pic_params['node_edge_color'], linewidths=pic_params['node_line_width'])
    if show_edge:
        nx.draw_networkx_edges(
            G, pos, width=pic_params['edge_width'], alpha=pic_params['edge_alpha'])
    plt.savefig(save_path)
    plt.close()