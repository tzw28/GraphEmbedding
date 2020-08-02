from sklearn.cluster import MeanShift, estimate_bandwidth, DBSCAN, OPTICS
import numpy as np

def mean_shift(vectors):
    vector_list = []
    for key in vectors.keys():
        vector_list.append(np.array(vectors[key][:2]))
    vector_list = np.array(vector_list)
    bandwidth = estimate_bandwidth(vector_list, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(vector_list)
    labels = ms.labels_
    clusters = {}
    key_list = list(vectors.keys())
    for i, key in enumerate(key_list):
        clusters[key] = labels[i]
    return clusters


def dbscan(vectors):
    vector_list = []
    for key in vectors.keys():
        vector_list.append(np.array(vectors[key][:2]))
    vector_list = np.array(vector_list)
    bandwidth = estimate_bandwidth(vector_list, quantile=0.2, n_samples=500)
    model = DBSCAN(
        eps=bandwidth,  # 邻域半径
        min_samples=5,    # 最小样本点数，MinPts
        metric='euclidean',
        metric_params=None,
        algorithm='auto', # 'auto','ball_tree','kd_tree','brute',4个可选的参数 寻找最近邻点的算法，例如直接密度可达的点
        leaf_size=30, # balltree,cdtree的参数
        p=None, # 
        n_jobs=4
    )
    model.fit(vector_list)
    labels = model.labels_
    clusters = {}
    key_list = list(vectors.keys())
    for i, key in enumerate(key_list):
        clusters[key] = labels[i] + 1
    return clusters


def optics(vectors):
    vector_list = []
    for key in vectors.keys():
        vector_list.append(np.array(vectors[key][:2]))
    vector_list = np.array(vector_list)
    model = OPTICS()
    model.fit(vector_list)
    labels = model.labels_
    clusters = {}
    key_list = list(vectors.keys())
    for i, key in enumerate(key_list):
        clusters[key] = labels[i] + 1
    return clusters

'''
if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    

    data, label = make_blobs(n_samples=500, centers=5, cluster_std=1.2, random_state=7)
    MS = MeanShift(band_width=3, min_fre=3, bin_size=4, bin_seeding=True)
    MS.fit(data)
    labels = MS.labels
    print(MS.centers, np.unique(labels))
    import matplotlib.pyplot as plt
    from itertools import cycle


    def visualize(data, labels):
        color = 'bgrymk'
        unique_label = np.unique(labels)
        for col, label in zip(cycle(color), unique_label):
            partial_data = data[np.where(labels == label)]
            plt.scatter(partial_data[:, 0], partial_data[:, 1], color=col)
        plt.show()
        return


    visualize(data, labels)
'''