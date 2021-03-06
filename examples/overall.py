import os
from ge import Node2Vec, DeepWalk, LINE, Struc2Vec
from nodevectors import Node2Vec as n2v
from layout.emb_fr import COLOR_MAP, kmeans, embedding_fr, fast_kmeans
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from layout.graph_reading import (
    read_citeseer_graph,
    read_cora_graph,
    read_miserables_graph,
    read_science_graph,
    read_facebook_graph,
    read_json_graph
)
import datetime, json
from layout.mean_shift import mean_shift, optics, dbscan
from layout.emb_layout import generate_emb_adj_graph
import time
from scipy import spatial
from layout.sgd import sgd, largest_connected_subgraph
from layout.ph import read_ph_position
from layout.attr2vec import attr2vec
from layout.emb_layout import generate_emb_adj_graph
from layout.evaluator import LayoutEvaluator


emb_method_dict = {
    "node2vec": Node2Vec,
    "deepwalk": DeepWalk,
    "line": LINE,
    "struc2vec": Struc2Vec,
    "attr2vec": attr2vec
}

layout_method_dict = {

}

reader_dict = {
    "citeseer": read_citeseer_graph,
    "cora": read_cora_graph,
    "miserables": read_miserables_graph,
    "science": read_science_graph,
    "facebook": read_facebook_graph
}


def run_tests():
    with open("tests.json", "r") as f:
        string = f.read()
        tests = json.loads(string)
    save_path = make_save_path()
    time_records = {}
    for test in tests:
        graph_name = test['graph']
        graph_tests = test['tests']
        for graph_test in graph_tests:
            method = graph_test['method']
            cur_save_path = save_path + "/{}-{}".format(graph_name, method)
            os.mkdir(cur_save_path)
            time_rec = run_single_test(
                method, graph_name, cur_save_path,
                graph_test['emb_params'],
                graph_test['train_params'],
                graph_test['layout_params'],
                graph_test['emb_pic_params'],
                graph_test['layout_pic_params'],
            )
            time_records[graph_name + "-" + method] = time_rec
    write_time_records(save_path, time_records)

def run_time_tests():
    with open("time_tests.json", "r") as f:
        string = f.read()
        tests = json.loads(string)
    save_path = make_save_path()
    time_records = {}
    for test in tests:
        graph_name = test['graph']
        graph_tests = test['tests']
        for graph_test in graph_tests:
            method = graph_test['method']
            cur_save_path = save_path + "/{}-{}".format(graph_name, method)
            os.mkdir(cur_save_path)
            time_rec = run_single_test(
                method, graph_name, cur_save_path,
                graph_test['emb_params'],
                graph_test['train_params'],
                graph_test['layout_params'],
                graph_test['emb_pic_params'],
                graph_test['layout_pic_params'],
            )
            time_records[graph_name + "-" + method] = time_rec
    write_time_records(save_path, time_records)

def run_layout_section_tests():
    with open("layout_section_tests.json", "r") as f:
        string = f.read()
        tests = json.loads(string)
    save_path = make_save_path()
    time_records = {}
    for test in tests:
        graph_name = test['graph']
        graph_tests = test['tests']
        for graph_test in graph_tests:
            method = graph_test['method']
            cur_save_path = save_path + "/{}-{}".format(graph_name, method)
            os.mkdir(cur_save_path)
            time_rec = run_layout_section_test(
                method, graph_name, cur_save_path,
                graph_test['emb_params'],
                graph_test['train_params'],
                graph_test['layout_params'],
                graph_test['emb_pic_params'],
                graph_test['layout_pic_params'],
            )
            time_records[graph_name + "-" + method] = time_rec
    write_time_records(save_path, time_records)


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

def fast_node2vec(G, return_weight, neighbor_weight):
    g2v = n2v(
        n_components=16,
        walklen=10,
        epochs=15,
        return_weight=return_weight,
        neighbor_weight=neighbor_weight,
        threads=4,
        w2vparams={"window":5, "negative":5, "iter":5,
                   "batch_words":128}
    )
    g2v.fit(G)
    vectors = {}
    for i, key in enumerate(g2v.model.wv.index2word):
        vectors[key] = g2v.model.wv.vectors[i]
    return vectors

def save_json_graph(G, old_G, pos, graph_name):
    json_graph = {"nodes":[], "links": []}
    nodes = old_G.nodes(data=True)
    for node in nodes:
        if node[0] not in G:
            continue
        node_dict = {}
        node_dict['id'] = node[0]
        for attr in node[1].keys():
            node_dict[attr] = node[1][attr]
        x = pos[node[0]][0]
        y = pos[node[0]][1]
        node_dict['x'] = x
        node_dict["y"] = y
        json_graph['nodes'].append(node_dict)
    for s, t in G.edges:
        weight = G.get_edge_data(s, t)["weight"]
        edge_dict = {
            "source": s,
            "target": t,
            "value": weight
        }
        json_graph['links'].append(edge_dict)
    json_str = json.dumps(json_graph)
    with open("{}.json".format(graph_name), "w") as f:
        f.write(json_str)

def run_single_test(emb_method, graph_name, save_path, emb_params, train_params, layout_params,
                    emb_pic_params, layout_pic_params):
    G, labels = read_graph(graph_name)
    time_record = {}
    # if emb_method == "node2vec":
    if False:
        return_weight = emb_params.get("p", 4)
        return_weight = 1 / return_weight
        neighbor_weight = emb_params.get("p", 0.25)
        neighbor_weight = 1 / neighbor_weight
        fast_node2vec(G, return_weight=return_weight, neighbor_weight=neighbor_weight)
        emb_start = time.time()
        fast_node2vec(G, return_weight=return_weight, neighbor_weight=neighbor_weight)
        emb_end = time.time()
        emb_t = emb_end - emb_start
        time_record['fast_node2vec'] = round(emb_t, 3)
    emb = emb_method_dict[emb_method]
    model_0 = emb(G, **emb_params)
    model_0.train(**train_params)
    emb_start = time.time()
    emb = emb_method_dict[emb_method]
    model = emb(G, **emb_params)
    model.train(**train_params)
    emb_end = time.time()
    emb_t = emb_end - emb_start
    time_record['embedding'] = round(emb_t, 3)
    vectors = model.get_embeddings()

    cluster_start = time.time()
    clusters, cluster_color = compute_cluster_color(G.nodes, vectors, layout_params['k'])
    cluster_end = time.time()
    cluster_time = cluster_end - cluster_start
    time_record['cluster_slow'] = round(cluster_time, 3)
    cluster_start = time.time()
    compute_cluster_color_fast(G.nodes, vectors, layout_params['k'])
    cluster_end = time.time()
    cluster_time = cluster_end - cluster_start
    time_record['cluster_fast'] = round(cluster_time, 3)
    label_color = compute_label_color(G.nodes, labels)

    emb_save_path = save_path + "/emb-tsne-whole-"
    for param, value in emb_params.items():
        emb_save_path += str(param) + str(value) + "-"
    emb_save_path = emb_save_path[:-1] + "-{}.png"
    tsne_pos = tsne(G, vectors)
    draw_emb_pictures(G, tsne_pos, cluster_color, label_color, emb_save_path, layout_pic_params)

    k = layout_params['k']
    layout_params.pop("k")
    layout_params['wa'] = 1
    layout_params['te'] = 0.6
    layout1_save_path = save_path + "/layout-"
    for param, value in layout_params.items():
        layout1_save_path += str(param) + str(value) + "-"
    layout1_save_path = layout1_save_path[:-1] + "-{}.png"
    embedding_fr(G, vectors, **layout_params)
    layout_pos_1, sim_time, ea_time, pos_time = embedding_fr(G, vectors, cluster=clusters, **layout_params)
    time_record['similarity_matrix'] = sim_time
    time_record['emb_adj_matrix'] = ea_time
    time_record['position'] = pos_time
    draw_layout_pictures(G, layout_pos_1, cluster_color, label_color, layout1_save_path, layout_pic_params)

    new_G = generate_emb_adj_graph(G, vectors=vectors, clusters=clusters)
    new_G = largest_connected_subgraph(new_G)
    save_json_graph(new_G, G, layout_pos_1, "emb" + graph_name)

    layout1_save_path = save_path + "/fr-layout-"
    layout1_save_path = layout1_save_path[:-1] + "-{}.png"
    fr_layout = nx.fruchterman_reingold_layout(G, seed=17)
    draw_layout_pictures(G, fr_layout, cluster_color, label_color, layout1_save_path, layout_pic_params)

    LCG = largest_connected_subgraph(G)
    emb = emb_method_dict[emb_method]
    model_1 = emb(LCG, **emb_params)
    model_1.train(**train_params)
    model = emb(LCG, **emb_params)
    model.train(**train_params)
    vectors = model.get_embeddings()
    lcg_cluster_color_list = []
    for node in LCG.nodes:
        c = clusters[node]
        lcg_cluster_color_list.append(COLOR_MAP[c])
    lcg_lable_color_list = compute_label_color(LCG.nodes, labels)

    emb_save_path = save_path + "/emb-tsne-lcg-"
    for param, value in emb_params.items():
        emb_save_path += str(param) + str(value) + "-"
    emb_save_path = emb_save_path[:-1] + "-{}.png"
    tsne_pos = tsne(LCG, vectors)
    draw_emb_pictures(LCG, tsne_pos, lcg_cluster_color_list, lcg_lable_color_list, emb_save_path, layout_pic_params)

    layout1_save_path = save_path + "/ph-layout-"
    layout1_save_path = layout1_save_path[:-1] + "-{}.png"
    ph_layout = read_ph_position(graph_name)
    draw_layout_pictures(LCG, ph_layout, lcg_cluster_color_list, lcg_lable_color_list, layout1_save_path, layout_pic_params)
    pos_dict = {"ph": ph_layout}

    layout1_save_path = save_path + "/sgd-layout-"
    layout1_save_path = layout1_save_path[:-1] + "-{}.png"
    layout_pos_1 = sgd(LCG)
    draw_layout_pictures(LCG, layout_pos_1, lcg_cluster_color_list, lcg_lable_color_list, layout1_save_path, layout_pic_params)
    pos_dict["sgd"] = layout_pos_1

    layout1_save_path = save_path + "/fr-lcg-layout-"
    layout1_save_path = layout1_save_path[:-1] + "-{}.png"
    layout_pos_1 = nx.fruchterman_reingold_layout(LCG)
    draw_layout_pictures(LCG, layout_pos_1, lcg_cluster_color_list, lcg_lable_color_list, layout1_save_path, layout_pic_params)
    pos_dict["fr"] = layout_pos_1

    layout1_save_path = save_path + "/em-fr-lcg-layout-"
    layout1_save_path = layout1_save_path[:-1] + "-{}.png"
    layout_pos_1 = embedding_fr(LCG, vectors, **layout_params)[0]
    draw_layout_pictures(LCG, layout_pos_1, lcg_cluster_color_list, lcg_lable_color_list, layout1_save_path, layout_pic_params)    
    pos_dict["{}-fr".format(emb_method)] = layout_pos_1

    ev = LayoutEvaluator(LCG, pos_dict, clusters)
    ev.run()
    ev.save_json_result("LCG", save_path)

    new_G = read_json_graph("emb" + graph_name + ".json")
    emb = emb_method_dict[emb_method]
    model_1 = emb(new_G, **emb_params)
    model_1.train(**train_params)
    model = emb(new_G, **emb_params)
    model.train(**train_params)
    vectors = model.get_embeddings()
    # new_LCG = largest_connected_subgraph(new_G)
    # _, lcg_cluster_color_list = compute_cluster_color(new_G.nodes, vectors, k)
    lcg_lable_color_list = compute_label_color(new_G.nodes, labels)

    layout1_save_path = save_path + "/new-sgd-layout-"
    layout1_save_path = layout1_save_path[:-1] + "-{}.png"
    G, labels = read_graph(graph_name)
    layout_pos_1 = sgd(new_G)
    nodes = list(G.nodes)
    for node in nodes:
        if new_G.has_node(node):
            continue
        G.remove_node(node)
    lcg_cluster_color_list = []
    for node in G.nodes:
        c = clusters[node]
        lcg_cluster_color_list.append(COLOR_MAP[c])
    draw_layout_pictures(G, layout_pos_1, lcg_cluster_color_list, lcg_lable_color_list, layout1_save_path, layout_pic_params)
    pos_dict = {"new-sgd": layout_pos_1}
    ev = LayoutEvaluator(G, pos_dict, clusters)
    ev.run()
    ev.save_json_result("{}-sgd".format(emb_method), save_path)
    
    return time_record


def run_layout_section_test(emb_method, graph_name, save_path, emb_params, train_params, layout_params,
                    emb_pic_params, layout_pic_params):
    G, labels = read_graph(graph_name)
    time_record = {}
    emb = emb_method_dict[emb_method]
    model= emb(G, **emb_params)
    emb_start = time.time()
    model.train(**train_params)
    emb_end = time.time()
    emb_t = emb_end - emb_start
    time_record['embedding'] = round(emb_t, 3)
    vectors = model.get_embeddings()

    cluster_start = time.time()
    clusters, cluster_color = compute_cluster_color(G.nodes, vectors, layout_params['k'])
    cluster_end = time.time()
    cluster_time = cluster_end - cluster_start
    time_record['cluster'] = round(cluster_time, 3)
    label_color = compute_label_color(G.nodes, labels)

    emb_save_path = save_path + "/"
    for param, value in emb_params.items():
        emb_save_path += str(param) + str(value) + "-"
    emb_save_path = emb_save_path[:-1] + "-{}.png"
    tsne_start = time.time()
    tsne_pos = tsne(G, vectors)
    tsne_end = time.time()
    time_record['tsne'] = round(tsne_end - tsne_start, 3)
    draw_emb_pictures(G, tsne_pos, cluster_color, label_color, emb_save_path, layout_pic_params)

    layout_params.pop("k")
    layout_params['wa'] = 1
    layout_params['we'] = 1
    layout_params['te'] = 0.6
    layout1_save_path = save_path + "/layout-"
    for param, value in layout_params.items():
        layout1_save_path += str(param) + str(value) + "-"
    layout1_save_path = layout1_save_path[:-1] + "-{}.png"
    layout_pos_1, sim_time, ea_time, _ = embedding_fr(G, vectors, cluster=clusters, **layout_params)
    time_record['similarity_matrix'] = sim_time
    time_record['emb_adj_matrix'] = ea_time
    draw_layout_pictures(G, layout_pos_1, cluster_color, label_color, layout1_save_path, layout_pic_params)

    layout_params['we'] = 0
    layout1_save_path = save_path + "/layout-original-"
    for param, value in layout_params.items():
        layout1_save_path += str(param) + str(value) + "-"
    layout1_save_path = layout1_save_path[:-1] + "-{}.png"
    layout_pos_1, sim_time, ea_time, _ = embedding_fr(G, vectors, **layout_params)
    time_record['similarity_matrix'] = sim_time
    time_record['emb_adj_matrix'] = ea_time
    draw_layout_pictures(G, layout_pos_1, cluster_color, label_color, layout1_save_path, layout_pic_params)

    layout_params['wa'] = 1
    layout_params['we'] = 1
    layout_params['te'] = 0.6
    layout1_save_path = save_path + "/layout-nocluster-"
    for param, value in layout_params.items():
        layout1_save_path += str(param) + str(value) + "-"
    layout1_save_path = layout1_save_path[:-1] + "-{}.png"
    layout_pos_1, sim_time, ea_time, _ = embedding_fr(G, vectors, **layout_params)
    time_record['similarity_matrix'] = sim_time
    time_record['emb_adj_matrix'] = ea_time
    draw_layout_pictures(G, layout_pos_1, cluster_color, label_color, layout1_save_path, layout_pic_params)

    layout2_save_path = save_path + "/layout-"
    layout_params['wa'] = 0
    layout_params['we'] = 1
    layout_params['te'] = 0
    for param, value in layout_params.items():
        layout2_save_path += str(param) + str(value) + "-"
    layout2_save_path = layout2_save_path[:-1] + "-{}.png"
    layout_pos_2, sim_time, ea_time, _ = embedding_fr(G, vectors, **layout_params)
    draw_layout_pictures(G, layout_pos_2, cluster_color, label_color, layout2_save_path, layout_pic_params)

    layout2_save_path = save_path + "/layout-"
    layout_params['wa'] = 0
    layout_params['we'] = 1
    layout_params['te'] = 0.6
    for param, value in layout_params.items():
        layout2_save_path += str(param) + str(value) + "-"
    layout2_save_path = layout2_save_path[:-1] + "-{}.png"
    layout_pos_2, sim_time, ea_time, _ = embedding_fr(G, vectors, **layout_params)
    draw_layout_pictures(G, layout_pos_2, cluster_color, label_color, layout2_save_path, layout_pic_params)
    return time_record


def write_time_records(path, time_records):
    headers = []
    for graph in time_records.keys():
        if not headers:
            headers = ["graph-emb"]
            headers.extend(list(time_records[graph].keys()))
            break

    file_name = path + "/timerecord.csv"
    with open(file_name, "w", encoding='utf-8') as f:
        header_str = ','.join(list(headers)) + "\n"
        f.write(header_str)
        for graph in time_records.keys():
            line = [graph]
            line.extend(list(time_records[graph].values()))
            line = [str(value) for value in line]
            line_str = ','.join(line) + "\n"
            f.write(line_str)

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
    clusters = kmeans(vectors, k)
    # clusters = mean_shift(temp_vectors)
    # clusters = dbscan(vectors)
    # clusters = dbscan(temp_vectors)
    # clusters = optics(vectors)
    color_list = []
    for node in nodes:
        c = COLOR_MAP[clusters[node]]
        color_list.append(c)
    return clusters, color_list


def compute_cluster_color_fast(nodes, vectors, k):
    clusters = fast_kmeans(vectors, k)
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
    plt.cla()
