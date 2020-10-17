import s_gd2
import networkx as nx


def largest_connected_subgraph(G):
    largest_component = max(nx.connected_components(G), key=len)
    LCG = nx.Graph()
    for s, t in G.edges:
        if s not in largest_component or t not in largest_component:
            continue
        try:
            weight = G.get_edge_data(s, t)["weight"]
            LCG.add_edge(s, t, weight=weight)
        except:
            LCG.add_edge(s, t)
    for n in LCG.nodes:
        ori_n = G.nodes[n]
        for key in ori_n.keys():
            LCG.nodes[n][key] = ori_n[key]
    return LCG

def generate_edge_indices(G):
    node_id_map = {}
    count = 0
    for node in G.nodes:
        if node in node_id_map.keys():
            continue
        node_id_map[node] = count
        count += 1
    I = []
    J = []
    for s, t in G.edges:
        I.append(node_id_map[s])
        J.append(node_id_map[t])
    return node_id_map, I, J
           

def sgd(G):
    LCG = largest_connected_subgraph(G)
    node_id_map, I, J = generate_edge_indices(LCG)
    result = s_gd2.layout(I, J)
    layout = {}
    for node in node_id_map.keys():
        node_id = node_id_map[node]
        layout[node] = result[node_id]
    return layout