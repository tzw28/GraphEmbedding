import numpy as np
import json
from numba import jit, jitclass


ZERO = 1e-9


# 点
class Point(object):

    def __init__(self, x, y):
        self.x, self.y = x, y


# 向量
class Vector(object):

    def __init__(self, start_point, end_point):
        self.start_point, self.end_point = start_point, end_point
        self.x = end_point.x - start_point.x
        self.y = end_point.y - start_point.y

    def negative(self):
        return Vector(self.end_point, self.start_point)


class LayoutEvaluator(object):

    def __init__(self, G, pos_dict, class_map):
        self.G = G
        self.pos_dict = pos_dict
        self.class_map = class_map

    def _normalize_position(self, pos, x_range=(0, 1), y_range=(0, 1)):
        nm_pos = {}
        x_min = 100
        x_max = -100
        y_min = 100
        y_max = -100
        for node in pos.keys():
            p = pos[node]
            if p[0] < x_min:
                x_min = p[0]
            if p[0] > x_max:
                x_max = p[0]
            if p[1] < y_min:
                y_min = p[1]
            if p[1] > y_max:
                y_max = p[1]
        for node in pos.keys():
            x1 = pos[node][0]
            y1 = pos[node][1]
            x2 = (x1 - x_min) / (x_max - x_min) * \
                 (x_range[1] - x_range[0]) + x_range[0]
            y2 = (y1 - y_min) / (y_max - y_min) * \
                 (y_range[1] - y_range[0]) + y_range[0]
            nm_pos[node] = (x2, y2)
        return nm_pos

    def _distance(self, n1, n2, pos):
        pos1 = np.array(pos[n1])
        pos2 = np.array(pos[n2])
        dis = np.linalg.norm(pos1 - pos2)
        return dis

    def _vector_product(self, vector1, vector2):
        return vector1.x * vector2.y - vector2.x * vector1.y

    def _is_intersected(self, A, B, C, D):
        if min(A[0], B[0]) > max(C[0], D[0]) or \
           min(A[1], B[1]) > max(C[1], D[1]) or \
           min(C[0], D[0]) > max(A[0], B[0]) or \
           min(C[1], D[1]) > max(A[1], B[1]):
            return False
        PA = Point(A[0], A[1])
        PB = Point(B[0], B[1])
        PC = Point(C[0], C[1])
        PD = Point(D[0], D[1])
        VAC = Vector(PA, PC)
        VAD = Vector(PA, PD)
        VBC = Vector(PB, PC)
        VBD = Vector(PB, PD)
        VCA = VAC.negative()
        VCB = VBC.negative()
        VDA = VAD.negative()
        VDB = VBD.negative()
        temp1 = self._vector_product(VAC, VAD) * self._vector_product(VBC, VBD)
        temp2 = self._vector_product(VCA, VCB) * self._vector_product(VDA, VDB)
        return (temp1 <= ZERO) and (temp2 <= ZERO)

    def _compute_edge_lengths(self, pos):
        lens = []
        for s, t in self.G.edges:
            dis = self._distance(s, t, pos)
            lens.append(dis)
        return lens

    def _compute_edge_length_uniformity(self, pos):
        edge_lengths = self._compute_edge_lengths(pos)
        len_arr = np.std(edge_lengths, ddof=1)
        len_mean = np.mean(edge_lengths)
        uni = len_arr / len_mean
        return uni

    def _compute_node_distribution(self, pos):
        lens = self._compute_edge_lengths(pos)
        res = 0
        for l in lens:
            res += 1 / l**2
        return res

    def _compute_edge_crossings(self, pos):
        crossing_count = 0
        edge_num = len(self.G.edges)
        total_degree = 0
        for node in self.G.nodes:
            degree = self.G.degree(node)
            total_degree += degree
        print(edge_num)
        for s1, t1 in self.G.edges:
            for s2, t2 in self.G.edges:
                if s1 == t2 or s1 == s2 or s2 == t1 or \
                   t1 == s2 or t1 == t2 or t2 == s1:
                    # print("skip {}-{} {}-{}".format(s1, t1, s2, t2))
                    continue
                ps1, pt1 = pos[s1], pos[t1]
                ps2, pt2 = pos[s2], pos[t2]
                if self._is_intersected(ps1, pt1, ps2, pt2):
                    # print("cross {}-{} {}-{}".format(s1, t1, s2, t2))
                    crossing_count += 1
        print("crossings {}".format(crossing_count))
        return crossing_count / 2 / edge_num **2

    def _compute_community_significance(self, pos):
        inner_dis = []
        outer_dis = []
        for node1 in self.G.nodes:
            for node2 in self.G.nodes:
                dis = self._distance(node1, node2, pos)
                if self.class_map[node1] == self.class_map[node2]:
                    inner_dis.append(dis)
                else:
                    outer_dis.append(dis)
        return np.mean(inner_dis), np.mean(outer_dis)

    def run(self):
        print("开始布局效果评估计算")
        self.res_dict = {}
        for key in self.pos_dict.keys():
            pos = self._normalize_position(self.pos_dict[key])
            inner_avg_distance, outer_avg_distance = self._compute_community_significance(pos)
            self.res_dict[key] = {
                "edge_length_uniformity": self._compute_edge_length_uniformity(pos),
                "node_distribution": self._compute_node_distribution(pos),
                "edge_crossings": self._compute_edge_crossings(pos),
                "inner_avg_distance": inner_avg_distance,
                "outer_avg_distance": outer_avg_distance
            }

    def save_json_result(self, graph_name, save_path):
        file_path = save_path + "/layout_evaluation_{}.json".format(graph_name)
        with open(file_path, "w") as f:
            json.dump(self.res_dict, f)
        print("布局效果评估计算完成")
