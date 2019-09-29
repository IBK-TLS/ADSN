import numpy as np
import pandas as pd
import argparse
import random
import uuid 


labels=["N", "A", "B", "PP", "PN"]
nlabels = len(labels)
nclasses = 5
nfeatures = 4

def genfakedb():
    #dataset = pd.read_csv(args.dataset)
    #print(dataset.tail())
    fakedb = []
    fakelb = []

    fakedb.append([1,1,1,1])
    fakelb.append(0)
    fakedb.append([1,1,1,1])
    fakelb.append(0)
    fakedb.append([1,1,1,1])
    fakelb.append(0)
    fakedb.append([0,0,0,0])
    fakelb.append(0)
    fakedb.append([0,1,1,0])
    fakelb.append(0)
    fakedb.append([0,0,0,0])
    fakelb.append(0)
    fakedb.append([0,0,0,0])
    fakelb.append(0)
    fakedb.append([0,0,0,0])
    fakelb.append(0)
    fakedb.append([0,0,0,0])
    fakelb.append(0)
    fakedb.append([1,1,1,1])
    fakelb.append(0)
    fakedb.append([0,0,0,0])
    fakelb.append(0)
    fakedb.append([0,0,0,0])
    fakelb.append(0)
    fakedb.append([0,1,1,0])
    fakelb.append(0)
    fakedb.append([1,1,1,1])
    fakelb.append(0)
    fakedb.append([1,2,2,1])
    fakelb.append(0)
    fakedb.append([1,2,2,1])
    fakelb.append(0)
    fakedb.append([1,2,2,1])
    fakelb.append(0)
    fakedb.append([0,4,0,0])
    fakelb.append(1)
    fakedb.append([4,3,4,0])
    fakelb.append(2)
    fakedb.append([0,4,3,4])
    fakelb.append(2)
    fakedb.append([3,4,0,0])
    fakelb.append(3)
    fakedb.append([0,3,4,0])
    fakelb.append(3)
    fakedb.append([0,0,3,4])
    fakelb.append(3)
    fakedb.append([0,3,0,0])
    fakelb.append(4)
    
    return fakedb, fakelb

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
    def __init__(self, nclasses=2, nfeat=2, iteration_max=10000):
        self.nclasses = nclasses
        self.nfeat = nfeat
        self.tree = {}
        self.queue = []
        self.root = None
        self.iteration_max = iteration_max

    def split(self, node):
        features = node.features
        classes = node.classes
        parent = node.parent
        gini_orig = node.gini
        split_rule = []
        classes_true = []
        classes_false = []
        features_true = []
        features_false = []
        gini_true = 0
        gini_false = 0
        N = len(classes)
        gain_gini = 0
        for index in range(self.nfeat-1):
            for f1 in range(self.nclasses):
                for f2 in range(self.nclasses):
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
                        split_rule_true = {"index":index, "features":[f1, f2], "condition": True }     
                        split_rule_false = {"index":index, "features":[f1, f2], "condition": False }     
                        classes_true = split_true
                        classes_false = split_false 
                        features_true = [f for f in features if f[index] == f1 and f[index+1] == f2]
                        features_false = [f for f in features if not f[index] == f1 or not f[index+1] == f2]
        node_true = node_tree(features_true, classes_true, gini_true, node.id, split_rule_true)
        node_false = node_tree(features_false, classes_false, gini_false, node.id, split_rule_false)
        return node_true, node_false, split_rule, gain_gini

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
            node_true, node_false, split_rules, gain_gini = self.split(node)
            #print(gain_gini, node_true.id == node_true.parent)
            self.tree.append( node_true ) 
            self.tree.append( node_false ) 
            if gain_gini > 0:
                if node_true.gini > 0:
                    self.queue.append(node_true)
                if node_false.gini > 0:
                    self.queue.append(node_false)
            n += 1
            index += 1

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
        return leaves 


    


### MAIN ###


def main(args):
    ft, cl = genfakedb()
    c = list(zip(ft, cl))
    random.shuffle(c)
    ft, cl = zip(*c)
    compotree = composition_tree(nclasses, nfeatures)
    compotree.fit(ft, cl)
    root = compotree.get_root()
    print("root", root.dict())
    print("children", [c.dict() for c in compotree.get_childrens(root.id)])
    leaves = compotree.get_leaves()
    print("leaves", len(leaves))
    branches = [(l.classes, compotree.get_branch(l)) for l in leaves]

    rules_per_class = [[]] * nclasses
    b = []
    for i, (classes, branch) in enumerate(branches):
        c = max(set(classes), key = classes.count)
        listofrule = [n.split_rule for n in branch]
        rules_per_class[c].append(listofrule)
        #print("branch:", str(i), "class:",   )
        #b.append([n.split_rule for n in branch])
        #print("number of rules:", len(b))
        #composition = ""
        #for node in branch:
        #    composition += str(node.split_rule) + " & "
        #print("composition:", composition)
    for i, rpc in enumerate(rules_per_class):
        print(i)
        for r in rpc:
            print(r)
        print()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Composition-based desicion tree utilities.")
    parser.add_argument("-d", "--dataset",
                        dest="dataset",
                        type=str,
                        help="dataset csv file",)  
    args = parser.parse_args()
    main(args)
