from src.model.data.image_caption_output import ImageCaptionOutput
class ProbabilityNode:
    def __init__(self,level,key,probability=1.0, **kwargs):
       super().__init__(**kwargs)
       self.level=level
       self.key=key
       self.probability=probability
       self.children=[]
       self.parent=None
       self.stopNode=False
       self.output=None
       self.id=-1

    def isStopNode(self):
        return self.stopNode == True 

    
    
        


   
    def visit(self,onVisitFunction):
        onVisitFunction(self)

        for childNode in self.children:
            onVisitFunction(childNode)

    
    def belongsTo(self,nodes):
        result=False
        for node in nodes:
            if self.equals(node):
                result=True

        return result    
    def prune_leaf_node(self,leafNode):
        pruned_children=[]
        for childNode in self.children:
            if childNode.equals(leafNode) == False:
                pruned_children.append(childNode)
        self.children=pruned_children        

    def prune_children(self,highProbabilityNodes):
        if self.isLeafNode() == True:
            return False
        pruned_children=[]
        for childNode in self.children:
            if childNode.belongsTo(highProbabilityNodes):
                pruned_children.append(childNode)
        self.children= pruned_children
        if self.isLeafNode() == True:
            self.parent.prune_leaf_node(self)        

    def equals(self,otherNode):
        if otherNode == None:
            return False
        return self == otherNode
    def isRootNode(self):
        result=False
        if self.parent == None:
            return True
        return False
    def isLeafNode(self):
        result=False
        if len(self.children) == 0:
            return True
        return False   

    def addChild(self,childProbNode):
        if childProbNode == None:
            return False
        if childProbNode.level != self.level+1:
            return False
        self.children.append(childProbNode)
        childProbNode.parent=self

        return True
    
    def searchChild(self,level,key):
        if level < self.level:
            return None
        if level == self.level:
            if key == self.key:
                return self
        if level > self.level:    
            for child in self.children:
                node_found = child.searchChild(level,key)
                if node_found is not None:
                    return node_found

                
        return None
    
    def calculateJointProbability(self):
        score=1.0
        node = self
        while node != None:
            score *=node.probability
            node = node.parent
        return score

    def getKeysFromRoot(self):
        result=[]
        node = self
        while node !=None:
            result.append(node.key)
            node = node.parent
        result.reverse()          
        return result