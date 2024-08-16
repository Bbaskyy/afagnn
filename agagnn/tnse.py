import numpy as np
import matplotlib.pyplot as plt

plt.rc('font', family='Times New Roman')
from sklearn import (manifold, decomposition, ensemble)
from scipy.interpolate import make_interp_spline
from collections import OrderedDict
import warnings

warnings.filterwarnings('ignore')


def visualize(data_list, label, title):
    def plot_embedding(X, y, title):
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        # ax = plt.subplot(1,3,flag)
        colorlist = ['#0d5b26', '#c94733', '#fddf8b', '#52b9d8', '#ff9200', '#e5086a', '#501d8a']
        label_list = ['Fault 1', 'Fault 2', 'Fault 3', 'Fault 4', 'Fault 5', 'Fault 6', 'Normal']

        for i in range(X.shape[0]):
            plt.scatter(X[i, 0], X[i, 1], s=10, c=colorlist[y[i]], label=label_list[y[i]])

        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            # if np.min(dist) < 4e-3:
            #     # don't show points that are too close
            #     continue
            if np.max(dist) > 4e-3:
                # don't show points that are too far
                continue
            shown_images = np.r_[shown_images, [X[i]]]
        plt.xticks(np.arange(0, 1, 0.1), fontsize=15), plt.yticks(np.arange(0, 1.5, 0.1), fontsize=15)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.xlabel('Dimension 1', fontsize=15)
        plt.ylabel('Dimension 2', fontsize=15)
        plt.legend(by_label.values(), by_label.keys(), loc='upper center', frameon=False, fontsize=15, ncol=3)
        # plt.title(title)

    def tSNE(X, y):
        # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
        tsne = manifold.TSNE()
        X_tsne = tsne.fit_transform(X)
        plot_embedding(X_tsne, y, title + " tSNE")

    plt.figure(figsize=(7.5, 6.3))
    tSNE(data_list, label)
    plt.savefig('vis.png', dpi=600)
    plt.show()

#
# after_x = np.load('E:/conda_project/comparison/after.npy')
# after_y = np.load('E:/conda_project/comparison/after_yy.npy')
# # print(after_x.shape)
# # print(after_y.shape)
# # print(after_y)
#
#
# with open('E:/conda_project/comparison/CAL/data/3pf/raw/3pf_node_attributes.txt', "r", encoding='utf-8') as f1:
#     att = []
#     for it in f1.readlines():
#         it = it.strip('\n')
#         # print(it)
#         att.append(it)
#     for k in range(len(att)):
#         at = att[k]
#         tmp = []
#         strlist = at.split(',')  # 用逗号分割str字符串，并保存到列表
#         for value in strlist:  # 循环输出列表值
#             tmp.append(float(value))
#         att[k] = tmp
#     # att = np.array(att)
#     # print(att.shape)
#     x = []
#     tt = []
#     for i in range(len(att)):
#         tt += att[i]
#         # print(len(tt))
#         if (i + 1) % 24 == 0 and i != 0:
#             x.append(tt)
#             tt = []
#     x = np.array(x)
#
# with open('E:/conda_project/comparison/CAL/data/3pf/raw/3pf_graph_labels.txt', "r", encoding='utf-8') as f2:
#     label = []
#     for la in f2.readlines():
#         la = la.strip('\n')
#         label.append(int(la))
#
# print(x.shape)
# print(after_x.shape)
# visualize(x, label, 'Raw data space')
# visualize(after_x, after_y, 'CTA-GNN')
