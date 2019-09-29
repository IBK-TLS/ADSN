import numpy as np
import pandas as pd
import argparse
import random
import uuid 


# write a cost function (entropy/Gini index ?)

# write a class tree
#class tree():

nlabels = 5
nclasses = 5

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
    fakedb.append([0,4,3,4])
    fakelb.append(2)
    fakedb.append([0,3,4,0])
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
    def __init__(self, nclasses=2, nfeat=4, iteration_max=10000):
        self.nclasses = nclasses
        self.nfeat = nfeat
        self.tree = {}
        self.queue = []
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
                        split_rule = {"index":index, "features":[f1, f2]}     
                        classes_true = split_true
                        classes_false = split_false 
                        features_true = [f for f in features if f[index] == f1 and f[index+1] == f2]
                        features_false = [f for f in features if not f[index] == f1 or not f[index+1] == f2]
        node_true = node_tree(features_true, classes_true, gini_true, node.id, split_rule)
        node_false = node_tree(features_false, classes_false, gini_false, node.id, split_rule)
        return node_true, node_false, split_rule, gain_gini

    def fit(self, features, classes):
        self.features = features
        self.classes = classes
        gini = gini_impurity(classes)
        root = node_tree(features, classes, gini, 0, None)
        root.set_parent(root.id)
        self.tree = [ root ]
        self.queue = [ root ]
        n = 0
        index = 0
        while not len(self.queue) == 0 and n < self.iteration_max:
            node = self.queue.pop(0)
            node_true, node_false, split_rules, gain_gini = self.split(node)
            print(gain_gini)
            n += 1
            self.tree.append( node_true ) 
            self.tree.append( node_false ) 
            if gain_gini > 0:
                if node_true.gini > 0:
                    self.queue.append(node_true)
                if node_false.gini > 0:
                    self.queue.append(node_false)

    def get_root(self):
        for node in self.tree:
            if node.id == node.parent:     
                return node 
    def get_parent(self, id):
        for node in self.tree:
            if node.parent == id:
                return node
    def get_childrens(self, id):
        childrens = []
        for node in self.tree:
            if node.parent == id:
                children.append(node)
        return childrens
    def get_branch(self, leaf):
        branch = [leaf]
        node = leaf 
        while node.parent == node.id:
            node = self.get_parent(node.id)
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
    compotree = composition_tree(5)
    compotree.fit(ft, cl)
    root = compotree.get_root().dict()
    print("root", root)
    leaves = compotree.get_leaves()
    print("leaves", [l.dict() for l in leaves])
    branches = [compotree.get_branch(l) for l in leaves]
    print("branches", branches)
    #branch_true, branch_false, split, gini , gain = best_split(db, lb, 0)
    #print(branch_false)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Composition-based desicion tree utilities.")
    parser.add_argument("-d", "--dataset",
                        dest="dataset",
                        type=str,
                        help="dataset csv file",)  
    args = parser.parse_args()
    main(args)
