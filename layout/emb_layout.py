import networkx as nx
import numpy as np


def generate_emb_adj_matrix(A, E, cluster_list=None, wa=1, we=1, te=0.6, tel=0.65, teh=0.85):
    if E is not None:
        D1 = _normalize_mat(E)
        D1 = 1 - D1
        D2 = wa * A + we * D1
        if cluster_list is None:
            D2[D2 <= te] = 0
        else:
            for i in range(len(cluster_list)):
                for j in range(len(cluster_list)):
                    c_i = cluster_list[i]
                    c_j = cluster_list[j]
                    if c_i == c_j:
                        te = tel
                    else:
                        te = teh
                    D2[i][j] = 0 if D2[i][j] < te else D2[i][j]
        AE = _normalize_mat(D2)
    return AE


def generate_similarity_matrix(vectors, dis_method="euclidean"):
    vec_list = []
    for u in vectors.keys():
        vec_list.append(vectors[u])
    E = []
    for u in vectors.keys():
        similarity = []
        for v in vectors.keys():
            if u == v:
                similarity.append(1)
                continue
            d = _distance(u, v, vectors, dis_method)
            similarity.append(d)
        E.append(similarity)
    return np.array(E)


def generate_graph_from_matrix(A, old_G):
    node_list = list(old_G.nodes)
    G = nx.Graph()
    for i, line in enumerate(A):
        for j, weight in enumerate(line):
            if weight == 0:
                continue
            node_s = node_list[i]
            node_t = node_list[j]
            G.add_edge(node_s, node_t, weight=weight)
    return G


def generate_emb_adj_graph(G, vectors, clusters=None, wa=1, we=1, te=0.7, tel=0.65, teh=0.85):
    A = nx.to_numpy_array(G, weight=True)
    E = generate_similarity_matrix(vectors)
    if clusters:
        cluster_list = []
        for node in G.nodes:
            cluster_list.append(clusters[node])
    else:
        cluster_list = None
    AE = generate_emb_adj_matrix(A, E, cluster_list, wa, we, te, tel, teh)
    new_G = generate_graph_from_matrix(AE, G)
    return new_G


def _distance(u, v, vectors, method):
    from scipy.spatial import distance
    method_map = {
        "braycurtis": distance.braycurtis,
        "canberra": distance.canberra,
        "chebyshev": distance.chebyshev,
        "cityblock": distance.cityblock,
        "correlation": distance.correlation,
        "cosine": distance.cosine,
        "euclidean": distance.euclidean,
        "jensenshannon": distance.jensenshannon,
        "mahalanobis": distance.mahalanobis,
        "minkowski": distance.minkowski,
        "seuclidean": distance.seuclidean,
        "sqeuclidean": distance.sqeuclidean,
    }

    u_vec = vectors[u]
    v_vec = vectors[v]
    dis = method_map[method](u_vec, v_vec)
    return dis

    
def _normalize_mat(D):
    d_max = D.max()
    d_min = D.min()
    D1 = (D - d_min) / (d_max - d_min)
    return D1