from src.model.ds.probability_node import ProbabilityNode
from src.model.data.image_caption_output import ImageCaptionOutput
class BeamSearch:
    def __init__(self,rootNode,k, **kwargs):
       super().__init__(**kwargs)
       self.rootNode=rootNode
       self.k=k

    def _calculateHighestProbabilityPath(self,node,imageCaptionOutput,tokenizer):
        if node.isStopNode() == True:
            joint_probability=node.calculateJointProbability()
            if joint_probability > imageCaptionOutput.joint_probability:
                imageCaptionOutput.attention_weights=node.attention_weights
                imageCaptionOutput.output=node.output
                keysFromRoot=node.getKeysFromRoot()
                caption_words=[]
                for key in keysFromRoot:
                    caption_words.append(tokenizer.index_word[int(key)])
                imageCaptionOutput.caption=caption_words
                imageCaptionOutput.initialized=True
                imageCaptionOutput.joint_probability=joint_probability
            return
        for childNode in node.children:
            self._calculateHighestProbabilityPath(childNode,imageCaptionOutput,tokenizer) 
    def calculateHighestProbabilityPath(self,tokenizer):
        imageCaptionOutput=ImageCaptionOutput()
        self._calculateHighestProbabilityPath(self.rootNode,imageCaptionOutput,tokenizer)
        return imageCaptionOutput.caption,imageCaptionOutput.output,imageCaptionOutput.attention_weights

    
    def visit(self,onVisitFunction):
        leaf_nodes=self.get_leaf_nodes()
        for leaf_node  in leaf_nodes:
            onVisitFunction(leaf_node)
    
    def _get_leaf_nodes(self,node,leaf_nodes):
        if node.isLeafNode() == True:
            leaf_nodes.append(node)
            return
        for childNode  in node.children:
            self._get_leaf_nodes(childNode,leaf_nodes)

    def get_leaf_nodes(self):
        leaf_nodes=[]
        self._get_leaf_nodes(self.rootNode,leaf_nodes)
        return leaf_nodes
    def _contains(self,allNodes,curNode):
        for node in allNodes:
                if node.equals(curNode):
                    return True
        return False
    def _equals(self,nodes1,nodes2):
        if len(nodes1) !=  len(nodes2):
            return False
        for node1 in nodes1:
            if self._contains(nodes2,node1) == False:
                return False
        return True    
      
    def _prune_nodes(self,node,highProbabilityNodes):
        if node.isLeafNode() == False:
            for childNode in node.children:
                if childNode.isLeafNode() == True:
                    if self._contains(highProbabilityNodes,childNode) == False:
                        node.prune_leaf_node(childNode)
                else:
                    self._prune_nodes(childNode,highProbabilityNodes)

                         
    def calculate_probabilities(self):
        leaf_nodes=self.get_leaf_nodes()
        def sort_function(obj):
            return obj.calculateJointProbability()
        leaf_nodes.sort(key=sort_function,reverse=True)
        highProbabilityNodes=leaf_nodes[0:self.k]
        return highProbabilityNodes
    def prune_nodes(self):
        highProbabilityNodes=self.calculate_probabilities()
        self._prune_nodes(self.rootNode,highProbabilityNodes)
        leaf_nodes_pruned=self.get_leaf_nodes() 
        for node in leaf_nodes_pruned:
            if node.isStopNode() == False:
                return False
        return True  
     
        
    
   

        
    def get_top_nodes(self,level):
        level_nodes=self.get_probability_nodes(level)
        if len(level_nodes) > 0:
            def sort_function(obj):
                return obj.calculateJointProbability()
            level_nodes.sort(key=sort_function,reverse=True)
            return level_nodes[0:self.k]
        return  level_nodes  
    def get_probability_nodes(self,level):
        level_nodes=[]
        self._get_probability_nodes(self.rootNode,level,level_nodes)
        return level_nodes
    
    def _get_probability_nodes(self,node,level,level_nodes):
        if node == None:
            return
        if level == node.level:
            level_nodes.append(node)
        if level > node.level and node.isLeafNode() == False:
            for childNode in node.children:
                self._get_probability_nodes(childNode,level,level_nodes)


