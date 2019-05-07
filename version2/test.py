from pynauty.graph import Graph,canonical_labeling
import networkx as nx

g = nx.Graph()
adj={
    'n0':['n1','n2','n4','n5'],
    'n1':['n0','n2','n5','n6'],
    'n2':['n0','n1','n5','n7'],
    'n3':['n5','n6'],
    'n4':['n0','n7'],
    'n5':['n0','n1','n2','n3'],
    'n6':['n1','n3'],
    'n7':['n2','n4'],
    'n8':[]
}
for node in adj.keys():
    for target in adj[node]:
        g.add_edge(node,target)
g.add_node('n8')

nodes_list = list(g.nodes)
print(nodes_list)

nauty = Graph(len(nodes_list),False)
nauty.set_vertex_coloring([{0,1,2,3,4,5,6,7},{8}])
G = nx.convert_node_labels_to_integers(g)
nauty.set_adjacency_dict({k:list(ndict) for k,ndict in G.adjacency()})



label = canonical_labeling(nauty)
print('nauty_label:',label)

nauty_map = {index:nodes_list[label[index]] for index in range(len(label))}
print('nauty_map',nauty_map)

G = nx.relabel_nodes(G,nauty_map)

print(G.nodes)