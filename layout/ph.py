import json

def read_ph_position(graph_name):
    res_path = "data/ph/ph_{}.json".format(graph_name)
    pos = {}
    with open(res_path, "r") as f:
        text = f.read()
        json_graph = json.loads(text)
        nodes = json_graph["nodes"]
    for node in nodes:
        node_name = node['id'].replace(" ", "")
        x = node['x']
        y = node['y']
        pos[node_name] = (x, y)
    return pos