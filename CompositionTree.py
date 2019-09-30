import numpy as np
import pandas as pd
import argparse
import random
import uuid 


labels=["N", "A", "B", "PP", "PN"]
nlabels = len(labels)
nclasses = 5
nfeatures = 4

def gini_impurity(data):
    prob = [0.] * nclasses
    N = len(data)
    for i in data:
        prob[i] += 1/N
    prob  = np.array(prob)
    return np.sum(prob*(1-prob))

class node_tree():
    def __init__(self, features, classes, gini, parent, split_rule):
        self.features = features
        self.classes = classes
        self.gini = gini
        self.parent = parent
        self.split_rule = split_rule
        self.id = uuid.uuid1()
    def set_features(self, features):    
        self.features = features
    def set_classes(self, classes): 
        self.classes = classes
    def set_gini(self, gini):
        self.gini = gini
    def set_parent(self, parent):    
        self.parent = parent
    def set_split_rule(self, split_rule):    
        self.split_rule = split_rule
    def dict(self):
        d = {"features": self.features, "classes": self.classes, "gini": self.gini, "id": self.id, "parent": self.parent, "split_rule": self.split_rule}
        return d

class composition_tree():
    def __init__(self, nclasses=2, nfeat=2, labels=None, iteration_max=10000):
        self.nclasses = nclasses
        self.nfeat = nfeat
        if labels:
            self.labels = labels
        else:
            self.labels = list(range(nfeat))
        self.tree = []
        self.queue = []
        self.rules = []
        self.root = None
        self.iteration_max = iteration_max

    def split(self, node):
        features = node.features
        classes = node.classes
        parent = node.parent
        gini_orig = node.gini
        split_rule_true = []
        split_rule_false = []
        classes_true = []
        classes_false = []
        features_true = []
        features_false = []
        gini_true = 0
        gini_false = 0
        N = len(classes)
        gain_gini = 0
        for index in range(self.nfeat-1):
            for f1 in self.labels: #range(self.nclasses):
                for f2 in self.labels: #range(self.nclasses):
                    split_true = [c for f, c in zip(features, classes) if f[index] == f1 and f[index+1] == f2]
                    split_false = [c for f, c in zip(features, classes) if not f[index] == f1 or not f[index+1] == f2]
                    g_true = gini_impurity(split_true)
                    g_false = gini_impurity(split_false)
                    g = ( g_true * (len(split_true)/N) + g_false * (len(split_false)/N) )
                    gain = gini_orig - g
                    if gain > gain_gini :
                        gain_gini = gain
                        gini_true = g_true
                        gini_false = g_false
                        split_rule_true = {"index":index, "features":[f1, f2], "condition": True, "rule" :  "(" +str(f1) + "_" + str(index) + "." + str(f2) + "_" + str(index+1) + ")"}     
                        split_rule_false = {"index":index, "features":[f1, f2], "condition": False, "rule": "not(" + str(f1) + "_" + str(index) + "." + str(f2) + "_" + str(index+1) + ")"  }     
                        classes_true = split_true
                        classes_false = split_false 
                        features_true = [f for f in features if f[index] == f1 and f[index+1] == f2]
                        features_false = [f for f in features if not f[index] == f1 or not f[index+1] == f2]
        node_true = node_tree(features_true, classes_true, gini_true, node.id, split_rule_true)
        node_false = node_tree(features_false, classes_false, gini_false, node.id, split_rule_false)
        return node_true, node_false, gain_gini

    def fit(self, features, classes):
        self.features = features
        self.classes = classes
        gini = gini_impurity(classes)
        root = node_tree(features, classes, gini, 0, None)
        root.set_parent(root.id)
        self.root = root
        self.tree = [ root ]
        self.queue = [ root ]
        n = 0
        index = 0
        while not len(self.queue) == 0 and n < self.iteration_max:
            node = self.queue.pop(0)
            node_true, node_false, gain_gini = self.split(node)
            self.tree.append( node_true ) 
            self.tree.append( node_false ) 
            if gain_gini > 0:
                if node_true.gini > 0:
                    self.queue.append(node_true)
                if node_false.gini > 0:
                    self.queue.append(node_false)
            n += 1
            index += 1
        self.rules = self.rules_per_class()

    def rules_per_class(self):
        leaves = self.get_leaves()
        branches = [(l.classes, self.get_branch(l)) for l in leaves]
        rules_per_class = [[] for _ in range(nclasses)]
        b = []
        for i, (classes, branch) in enumerate(branches):
            c = max(set(classes), key = classes.count)
            listofrule = [n.split_rule for n in branch if n.split_rule]
            rules_per_class[c].append(listofrule)
        return rules_per_class
 
    def composition(self):
        leaves = self.get_leaves()
        branches = [(l.classes, self.get_branch(l)) for l in leaves]
        rules_per_class = [[] for _ in range(nclasses)]
        for i, (classes, branch) in enumerate(branches):
            c = max(set(classes), key = classes.count)
            listofrule = [n.split_rule["rule"] for n in branch if n.split_rule]
            rules_per_class[c].append(listofrule)
        return rules_per_class

    def is_class(self, feat, c):
        predicted_class = []
        rules_for_class = self.rules[c]    
        boolean_class = []
        pc = [True]*len(rules_for_class)
        for i, srules in enumerate(rules_for_class):
            for rule in srules:
                f1 = feat[rule["index"]]
                f2 = feat[rule["index"]+1]
                condition = f1 == rule["features"][0] and f2 == rule["features"][1]
                pc[i] = pc[i] and condition == rule["condition"]
        return pc

  
    def predict(self, feat):
        predicted_class = []
        for c, rules_per_class in enumerate(self.rules):
            pc = [True]*len(rules_per_class)
            for i, srules in enumerate(rules_per_class):
                #print(rules)
                for rule in srules:
                    #print(rule)
                    f1 = feat[rule["index"]]
                    f2 = feat[rule["index"]+1]
                    condition = f1 == rule["features"][0] and f2 == rule["features"][1]
                    pc[i] = pc[i] and condition == rule["condition"]
                    if condition == rule["condition"]:
                        print( feat, condition, rule["rule"])
            predicted_class.append(pc)
        return predicted_class

    def get_root(self):
        for node in self.tree:
            if node.id == node.parent:     
                return node 
    def get_parent(self, id):
        for node in self.tree:
            if node.id == id:
                return node
    def get_childrens(self, id):
        childrens = []
        for node in self.tree:
            if not node == self.root and node.parent == id:
                childrens.append(node)
        return childrens
    def get_branch(self, leaf):
        branch = [leaf]
        node = leaf 
        while not node == self.root :
            node = self.get_parent(node.parent)
            branch.append(node)
        return branch
    def get_leaves(self):
        leaves = []
        for node in self.tree:
            if node.gini == 0:
                leaves.append(node)
            else:
                childrens = self.get_childrens(node.id)
                if len(childrens) == 0:
                    leaves.append(node)
        return leaves 
    
    
    


### MAIN ###


def main(args):
    cl = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 3, 3, 3, 4]
    ft = [["A","A","A","A"],["N","N","N","N"],["N","A","A","N"],["N","N","N","N"],["N","N","N","N"],["N","N","N","N"],["N","N","N","N"],["A","A","A","A"],["N","N","N","N"],["N","N","N","N"],["N","A","A","N"],["A","A","A","A"],["A","B","B","A"],["A","B","B","A"],["A","B","B","A"],["N","PN","N","N"],["PN","PP","PN","N"],["N","PN","PP","PN"],["PP","PN","N","N"],["N","PP","PN","N"],["N","N","PP","PN"],["N","PP","N","N"]]

    c = list(zip(ft, cl))
    random.shuffle(c)
    ft, cl = zip(*c)
    compotree = composition_tree(nclasses, nfeatures, labels)
    compotree.fit(ft, cl)
    #root = compotree.get_root()
    #print("root id", root.id)
    #print("children", [c.dict() for c in compotree.get_childrens(root.id)])
    
    compositions = compotree.composition()
    #print(compositions)
    #leaves = compotree.get_leaves()
    #for leaf in leaves:
    #    print(leaf.id) 
    #    
    #test = [[] for _ in range(nclasses)]
    #
    #branches = [ (l.classes, compotree.get_branch(l)) for l in leaves]
    #for classes, branch in branches:
    #    c = max(set(classes), key = classes.count)
    #    rules = [n.split_rule["rule"] for n in branch if n.split_rule]
    #    print(c, rules ) 
    #    test[c].append(rules) 
    #    print(test)
    #    print()
    for i, rpc in enumerate(compositions):
        print("class", i)
        for j, rules in enumerate(rpc):
            print(rules)
            print("or" if j < len(rpc)-1 else "")
        print()
    #class_0 = compotree.is_class(["N","PP","PN","N"], 0)
    #print(class_0)
    #class_1 = compotree.is_class(["N","PP","PN","N"], 1)
    #print(class_1)
    #class_2 = compotree.is_class(["N","PP","PN","N"], 2)
    #print(class_2)
    #class_3 = compotree.is_class(["N","PP","PN","N"], 3)
    #print(class_3)
    #class_4 = compotree.is_class(["N","PP","PN","N"], 4)
    #print(class_4)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Composition-based desicion tree utilities.")
    parser.add_argument("-d", "--dataset",
                        dest="dataset",
                        type=str,
                        help="dataset csv file",)  
    args = parser.parse_args()
    main(args)
