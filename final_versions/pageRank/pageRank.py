
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import time
import random
from math import *
import csv
import networkx as nx
import matplotlib.animation as animation


# columns in pagerank_dataset.csv: id,website_url,number_of_links,importance,links_it_contains

def read_csv_numpy(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    return np.array(data)


csv_filename = 'pagerank_dataset.csv'
data = read_csv_numpy(csv_filename)
# print(data)

alpha = 0.85
max_iter = 10
tol = 1e-6
num_websites = len(data) - 1

print(f"Number of websites: {num_websites}")

A = np.zeros((len(data) - 1, len(data) - 1))

for i in range(1, len(data)):
    website_url = data[i, 1]
    links_it_contains = data[i, 4].split(', ')
    # print(links_it_contains)

    for j in range(1, len(data)):
        if data[j, 1] in links_it_contains:
            A[i - 1, j - 1] = 1/len(links_it_contains)

    if len(links_it_contains) == 0:
        A[i - 1, :] = 1/num_websites

print(A)


plt.figure(figsize=(8, 8))

G = nx.from_numpy_array(A, create_using=nx.DiGraph)

# print(G.edges)
nx.spring_layout(G)

pagerank = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-8)

colormap = plt.cm.nipy_spectral
max_rank = max(pagerank.values())
min_rank = min(pagerank.values())
rank_range = max_rank - min_rank
edge_colors = [(pagerank[edge[1]] - min_rank) / rank_range for edge in G.edges()]
edge_colors = [colormap(color * 0.6) for color in edge_colors]

edge_widths = [pagerank[edge[1]] * 8 for edge in G.edges()]

print(pagerank)

node_sizes = [pagerank[node] * 1000 * len(G.edges(node)) for node in G.nodes()]
label = {i: data[i + 1, 0] for i in range(num_websites)}
pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue')
nx.draw_networkx_labels(G, pos, labels=label, font_size=2, font_weight='bold')

edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, width=edge_widths, edge_color=edge_colors, connectionstyle="arc3,rad=0.05", arrowsize=3, arrowstyle='-|>', alpha=0.6)
plt.show()

# export in very high quality png
# plt.savefig('pagerank.png', dpi=1200)