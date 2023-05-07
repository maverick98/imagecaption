from src.model.ds.probability_node import ProbabilityNode
from src.model.ds.beam_search import BeamSearch
rootNode = ProbabilityNode(0,'<s>')
node_1_1 = ProbabilityNode(1,'a',0.8)
node_1_2 = ProbabilityNode(1,'b',0.7)
node_1_3 = ProbabilityNode(1,'c',0.76)

rootNode.addChild(node_1_1)
rootNode.addChild(node_1_2)
rootNode.addChild(node_1_3)

node_1_1_1 = ProbabilityNode(2,'d',0.5)
node_1_1_2 = ProbabilityNode(2,'e',0.4)
node_1_1_3 = ProbabilityNode(2,'f',0.6)

node_1_1_found = rootNode.searchChild(1,'a')
node_1_1_found.addChild(node_1_1_1)
node_1_1_found.addChild(node_1_1_2)
node_1_1_found.addChild(node_1_1_3)

searchNode=rootNode.searchChild(2,'f')

print(searchNode.key,searchNode.probability)

print(searchNode.calculateProbability())
beamSearch=BeamSearch(rootNode,2)
nodes = beamSearch.get_top_nodes(2)
print(len(nodes))
for node in nodes:
    print(node.key,node.probability,node.level)

nodes = beamSearch.get_probability_nodes(2)
print("All nodes ",len(nodes))
for node in nodes:
    print(node.key,node.probability,node.level)    

beamSearch.prune_nodes(1)
print("---after pruning-----")
nodes = beamSearch.get_top_nodes(2)
print(len(nodes))
for node in nodes:
    print(node.key,node.probability,node.level)

nodes = beamSearch.get_probability_nodes(2)
print("All nodes ",len(nodes))
for node in nodes:
    print(node.key,node.probability,node.level)   

