# !pip install spacy pattern
# !python -m spacy download en_core_web_sm

# !pip install nltk
# import nltk
# nltk.download('wordnet')
# nltk.download('brown')
# nltk.download('universal_tagset')

# !pip freeze | grep pattern

# sudo apt-get install mysql-client
# sudo apt-get install mysql-server
# sudo apt-get install libmysqlclient-dev
# !pip install mysqlclient==2.1.1 pattern

# import torch
import spacy
from pattern.en import singularize

import nltk
from nltk.corpus import brown
from nltk.corpus import wordnet as wn

import numpy as np
import networkx as nx
import yaml
import json

from tqdm.auto import tqdm

# import matplotlib.pyplot as plt
# plt.style.use("ggplot")
# plt.style.use("seaborn-v0_8-colorblind")


def word_definitions(word, return_strings=True):
    res = [[s.name(), s.definition()] for i, s in enumerate(wn.synsets(word))]
    if return_strings:
        return "\n".join([f"[{name}] {definition}" for name, definition in res])
    else:
        return res


def get_path(synset_name):
    return wn.synset(synset_name).hypernym_paths()


# ## word to best matching synset
def prefer_exact_match(synsets, word):
    """prefer exact match"""
    word = singularize(word)
    synset_singletons = [singularize(s.name().split(".")[0]) for s in synsets]
    synsets_exact_match = [s for s, ss in zip(synsets, synset_singletons) if ss == word]
    synsets_not_exact_match = [
        s for s, ss in zip(synsets, synset_singletons) if ss != word
    ]
    if len(synsets_exact_match) > 0:
        return synsets_exact_match
    else:
        return synsets_exact_match + synsets_not_exact_match


def prefer_nouns(synsets, word):
    """prefer exact match"""
    synsets_n = [s for s in synsets if s.name().split(".")[1] == "n"]
    synsets_adj = [s for s in synsets if s.name().split(".")[1] == "s"]
    synsets_other = [
        s
        for s in synsets
        if s.name().split(".")[1] != "n" and s.name().split(".")[1] != "s"
    ]
    return synsets_n + synsets_adj + synsets_other


def choose_synset(synsets, word):
    synsets = prefer_exact_match(synsets, word)
    #     synsets = prefer_nouns(synsets, word)
    return synsets[0], 0


#     best_synset, best_score = None, -1
#     #     print(big_line_break)
#     #     print('word', word)
#     for j, synset in enumerate(synsets):
#         lemma_names = [l.name() for l in synset.lemmas()]
#         lemma_scores = [word_frequency(ln.lower(), "en") for ln in lemma_names]
#         synset_score = np.sum(lemma_scores)
#         #         print(lemma_names)
#         #         print(lemma_scores)
#         #         print(line_break)
#         if synset_score > best_score:
#             best_score = synset_score
#             best_synset = synset
#     return best_synset, best_score


# ## [Only need run once] Save imagenet synsets as file
# from glob import glob
# from natsort import natsorted
# imagenet_word2synset = {}
# fns = natsorted(glob('/home/jack/data/dataset/imagenet/val/*/'))
# for i, fn in enumerate(fns):
#     synset_offset = int(fn.split('/')[-2][1:])
#     synset = wn.synset_from_pos_and_offset('n', synset_offset)
#     word = synset.name().split('.')[0]
# #     word += f'_{i}'
#     imagenet_word2synset[word] = synset.name()
# with open("my_data/imagenet_word2synset.yaml", "w") as f:
#     yaml.dump(imagenet_word2synset, f)

## Load word2synset correction and imagenet_word2synset
line_break = "-" * 40
big_line_break = "=" * 40
# word_frequency = nltk.FreqDist(w.lower() for w in brown.words())
with open("my_data/manual_word2synset.yaml", "r") as f:
    manual_word2synset = yaml.safe_load(f)
with open("my_data/imagenet_word2synset.yaml", "r") as f:
    imagenet_word2synset = yaml.safe_load(f)

# ## Word list used to grab synset objects
with open("data/20k.txt") as f:
    vocabulary = [l.strip() for l in f]
print(len(vocabulary))
words = [w for w in vocabulary]
# remove single and double-letter words
words = [w for w in words if len(w) > 2]
# singularize words while preserving word order, using the order-preserving property of python dictionaries
words = list(dict.fromkeys([singularize(w) for w in words]).keys())

# filter by POS tag=NOUN using spacy
# https://universaldependencies.org/u/pos/
nlp = spacy.load("en_core_web_sm")
words2 = []
print("Getting all nouns...")
for w in tqdm(words):
    tokens = nlp(w)
    pos = [token.pos_ for token in tokens]
    if "NOUN" in pos or "PROPN" in pos:
        words2.append(w)
words = words2

word_synsets = list([[w, wn.synset(s)] for w, s in imagenet_word2synset.items()])
for i, word in enumerate(words):
    # print(f"word{i}", word)
    synsets = wn.synsets(word, pos=wn.NOUN)
    #     synsets += wn.synsets(word, pos=wn.ADJ)
    if len(synsets) == 0:
        # print(f"no synset for {word}")
        # print(line_break)
        continue
    else:
        if word in manual_word2synset:
            synset = wn.synset(manual_word2synset[word])
        else:  # try the best to choose
            synset, score = choose_synset(synsets, word)
        word_synsets.append([word, synset])
        # print(synset)
        # print(synset.definition())
        # print(line_break)

print("word synsets", len(word_synsets))


# word_synsets_write = []
# word_synsets_write.append(
#     [word, synset.name(), synset.definition()]
# )

# with open('my_data/wordnet.csv', 'w') as f:
#     f.write('word,synset,definition\n')
#     for line in word_synsets_write:
#         f.write(','.join(line)  + '\n')


# ## Construct graph
# - Nodes and edges are identified by synset names (e.g., "tiger.n.02")
word_set = set(words)


def is_concept(word):
    return word in word_set


graph = nx.DiGraph()
for word, synset in word_synsets:
    paths = synset.hypernym_paths()
    for path in paths:
        path_nodes = [
            [
                s.name(),
                dict(
                    word=s.name().split(".")[0],
                    synset=s.name(),
                    definition=s.definition(),
                    is_concept=False,
                    hypernym_paths=None,
                ),
            ]
            for s in path[:-1]
        ]
        graph.add_nodes_from(path_nodes)

        for source, target in list(zip(path[:-1], path[1:])):
            # convert synset object to name
            source, target = source.name(), target.name()
            # edge (source->target) points toward more specific terms
            edge_id = (source, target)
            if edge_id in graph.edges:
                graph.edges[edge_id]["weight"] += 1 / len(path)
            else:
                edge_weight = 1 / len(path)
                if source == "entity.n.01" or target == "entity.n.01":
                    edge_weight = 10
                graph.add_edges_from([[*edge_id, dict(weight=edge_weight)]])


concept_nodes = []
for word, synset in word_synsets:
    paths = synset.hypernym_paths()
    concept_nodes.append(
        [
            synset.name(),
            dict(
                word=word,  # Want to keep the original word to it
                synset=synset.name(),
                definition=synset.definition(),
                is_concept=True,
                hypernym_paths=[[s.name() for s in path] for path in paths],
            ),
        ]
    )
graph.add_nodes_from(concept_nodes)

# Construct maximum_spanning_arborescence from the DAG and call it a tree
tree = nx.algorithms.maximum_spanning_arborescence(
    graph, attr="weight", preserve_attrs=True
)

# sync graph node data to tree nodes
for node_id in tree.nodes:
    tree.nodes[node_id].update(graph.nodes[node_id])


# Compress tree nodes if the node only have 1 incoming and 1 outgoing edge
for node in list(tree.nodes()):
    if tree.in_degree(node) == 1 and tree.out_degree(node) == 1:
        edges = list(tree.in_edges(node))[0], list(tree.out_edges(node))[0]
        tree.add_edge(edges[0][0], edges[1][1])
        tree.remove_node(node)

# out_graph = graph
out_graph = tree


# bfs layer of a tree
for layer, nodes1 in enumerate(nx.bfs_layers(out_graph, "entity.n.01")):
    print(layer, ", ".join(nodes1[:3]), "...")
    for node in nodes1:
        out_graph.nodes[node]["level"] = layer


# Write graph data to file
nodes = [node_data for node, node_data in out_graph.nodes.items()]
for i, n in enumerate(nodes):
    n["index"] = i  # add indice to nodes

edges = [edge for edge, edge_data in out_graph.edges.items()]

# Nodes
with open("my_data/wordnet_nodes.json", "w") as f:
    json.dump(nodes, f, indent=2)

# Edges
with open("my_data/wordnet_edges.json", "w") as f:
    json.dump(edges, f, indent=2)

# write word list from all nodes of the hierarchy
with open("data/wordnet_hierarchy.txt", "w") as f:
    f.writelines([n["word"] + "\n" for n in nodes])
