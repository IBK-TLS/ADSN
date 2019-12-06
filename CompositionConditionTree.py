import numpy as np
import pandas as pd
import argparse
import random
import uuid 
import copy
import itertools

def simplification(c1, c2):
    dict_implication = {"1101": 1, "1110": 0, "1100": None, "0110": None, "0100": 0, "1000": 1}
    string = str(int(c1[0]))+str(int(c1[1]))+str(int(c2[0]))+str(int(c2[1]))
    return dict_implication[string]

def list_of_combination(labels, len_window):
    listofcombinations = []
    for l in np.arange(2, len_window):
        listofcombinations += [ list(c) for c in list(itertools.product(labels, repeat=l+1)) ]
    return listofcombinations

def list_of_possible_combination(features, max_size=10):
    listofcombinations = []
    for f in features:
        i = len(f)//2
        for w in range( min(len(f[i:]), max_size//2) ):
            listofcombinations.append(f[i-w-1:i+w+2])

    return listofcombinations 

def list_of_possible_conditions(values):
    listofconditions = []
    for v in values:
        cond = list(np.argsort(v))
        conditions = list( itertools.combinations(cond, 2) )
        for c in conditions:
            listofconditions.append(c)            

    return listofconditions
         
def list_of_possible_condition_combination(features, values):
    listofcombinations = []
    for f, v  in zip(features, values):
        i = len(f)//2
        for w in range(len(f[i:])):
            wv = v[i-w-1: i+w+2]
            wf = f[i-w-1:i+w+2] 
            conditions = list( itertools.combinations(list(np.argsort(wv)), 2) )
            for cond in conditions:
                listofcombinations.append((wf, cond))
    return listofcombinations 
   
def islistinlist(s, l):
    return True in [ s==sl for sl in  [l[index:index+len(s)] for index in range(len(l)-len(s)+1) ] ]

def whereislistinlist(s, l):
    indices = [ i for i, sl in  enumerate( [l[index:index+len(s)] for index in range(len(l)-len(s)+1) ] ) if s==sl ]
    return indices[0]

def listwhereislistinlist(s, l, v):
    indices = [ i for i, sl in  enumerate( [l[index:index+len(s)] for index in range(len(l)-len(s)+1) ] ) if s==sl ]
    if len(indices) > 0:
        return v[indices[0]:indices[0]+len(s)] # just return the first occurence
    else:
        return []

def checkcondition(v, c):
    return c in list( itertools.combinations(list(np.argsort(v)), 2) )
       
        
        

def gini_impurity(data, nclasses):
    prob = [0. for _ in  range(nclasses)]
    N = len(data)
    for i in data:
        if type(i) is int:
            prob[i] += 1/N
    # multiclass support 
    best_class = np.argmax(prob)
    for i in data:
        if type(i) is list:
            if best_class in i:
                prob[best_class] += 1/N
            else:
                for j in i:
                    prob[j] += (1/(len(i))) * (1/N)
    prob  = np.array(prob)
    return np.sum(prob*(1-prob))

class node_tree():
    def __init__(self, features, values, classes, gini, parent, split_rule, active=True):
        self.features = features
        self.classes = classes
        self.values = values
        self.gini = gini
        self.parent = parent
        self.split_rule = split_rule
        self.id = uuid.uuid1()
        self.active = active
    def set_features(self, features):    
        self.features = features
    def set_values(self, values):    
        self.values = values
    def set_classes(self, classes): 
        self.classes = classes
    def set_gini(self, gini):
        self.gini = gini
    def set_parent(self, parent):    
        self.parent = parent
    def set_split_rule(self, split_rule):    
        self.split_rule = split_rule
    def activate(self):    
        self.active = True
    def deactivate(self):    
        self.active = False
    def dict(self):
        d = {"features": self.features, "values": self.values,  "classes": self.classes, "gini": self.gini, "id": self.id, "parent": self.parent, "split_rule": self.split_rule, "active":self.active}
        return d

 


class branch_tree():
    def __init__(self, nodes, nfeat, nclasses):
        self.nodes = copy.deepcopy(nodes)
        self.nfeat = nfeat
        self.nclasses = nclasses
    def set_nodes(self, nodes):
        self.nodes = copy.deepcopy(nodes)
    def delete_node(self, node):
        for i, n in enumerate(self.nodes):
            if n.id == node.id:
                del self.nodes[i]
    def prune_branch(self):
        rules = {}
        for i in range(self.nfeat):
            rules[i] = None
        for node in self.nodes:
            rule = node.split_rule
            for i, condition in enumerate(rule["conditions"]):
                if condition == True:
                    rules[rule["indices"][i]] = { "labels" : rule["features"][i],
                                                  "condition": rule["conditions"][i] }

        for k, v in rules.items():
            if not v == None:
                for node in self.nodes:
                    rule = node.split_rule
                    
                    for i, conditions in enumerate(rule["conditions"]): 
                        if rule["indices"][i] == k:
                            rule["features"][i] = v["labels"]
                            rule["conditions"][i] = True
                    rule["rule"] = construct_rulestr(rule["features"], rule["conditions"], rule["indices"])
    
    def get_parent(self, id):
        for node in [n for n in self.nodes if n.active]:
            if node.id == id:
                return node
        return None
    def get_children(self, id):
        root = self.get_root()
        for node in [n for n in self.nodes if n.active]:
            if not node == root and node.parent == id:
                return node
        return None 
    def get_root(self):
        for node in [n for n in self.nodes if n.active]:
            if node.id == node.parent:
                return node
        return None 
    def get_leaf(self):
        for node in [n for n in self.nodes if n.active]:
            if self.get_children(node.id) is None:
                return node
    def branch_descent(self):
        descent = [self.get_root()]
        leaf = self.get_leaf()
        while not descent[-1] is leaf:
            descent.append(self.get_children(descent[-1].id))    
        return lnodes

    def merge_two_branches(self, branch):
        bd1 = self.branch_descent()
        bd2 = branch.branch_descent()
        merge = [] 
        if not len(bd1) == len(bd2):
            return None
        else:
            for n1, n2 in zip(bd1, bd2):
                if n1.id == n2.id:
                    merge.append(n1)
                elif n1.split_rule["features"] == n2.split_rule["features"] and simplification(n1.split_rule["conditions"], n2.split_rule["conditions"]):
                    index_to_keep = simplification(n1.split_rule["conditions"], n2.split_rule["conditions"])
                    feat = n1.split["features"][index_to_keep]
                    cond = n1.split["conditions"][index_to_keep]
                    classes = n1.classes + n2.classes
                    features = n1.features + n2.features
                    gini = gini_impurity(classes, self.nclasses)
                    strrule = construct_rulestr(feat, cond, index_to_keep)
                    split_rule = {"features": feat, "conditions": cond  }
                    #node_tree( features, classes, gini, n1.parent, split_rule_types[i]))
                 
        return branch 




class composition_condition_tree():
    def __init__(self, nclasses=2, nfeat=2, labels=None, iteration_max=10000, epsilon = 1e-6):
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
        self.epsilon = epsilon
        self.iteration_max = iteration_max

    def split(self, node):
        features = node.features
        values= node.values
        classes = node.classes
        parent = node.parent
        gini_orig = node.gini
        nodes = []
        split_rule_true = []
        split_rule_false = []
        classes_true = []
        classes_false = []
        classes_true_true = []
        classes_true_false = []
        features_true = []
        features_false = []
        features_true_true = []
        features_true_false = []
        values_true_true = []
        values_true_false = []
        gini_true = 0
        gini_false = 0
        N = len(classes)
        gain_gini = 0
        best_comb = []
        features_with_anomaly = [f for f,c in zip(features, classes) if c != 0  ]
        values_with_anomaly = [v for f, v,c in zip(features, values, classes) if c != 0  ]
        for comb, cond in list_of_possible_condition_combination(features_with_anomaly, values_with_anomaly):
            split_true_true = [c for f, v, c in zip(features, values, classes) if islistinlist(comb,f) and checkcondition( listwhereislistinlist(comb, f, v), cond )]
            split_true_false = [c for f, v, c in zip(features, values, classes) if islistinlist(comb,f) and not checkcondition( listwhereislistinlist(comb, f, v), cond )]
            split_false = [c for f, c in zip(features, classes) if not islistinlist(comb,f)]
            g_true_true = gini_impurity(split_true_true, self.nclasses)
            g_true_false = gini_impurity(split_true_false, self.nclasses)
            g_false = gini_impurity(split_false, self.nclasses)
            g = ( g_true_true * (len(split_true_true)/N)+ g_true_false * (len(split_true_false)/N)  + g_false * (len(split_false)/N) )
            gain = gini_orig - g
            print(comb, gain)
            if gain > gain_gini :#or (gain == gain_gini and len(comb) > len(best_comb)) :
                gain_gini = gain
                gini_true_true = g_true_true
                gini_true_false = g_true_false
                gini_false = g_false
                best_comb = comb
                split_rule_true_true = {"features": [comb, cond], "conditions": [True, True], "rule" :  str(comb) + " and " + str(cond)}     
                split_rule_true_false = {"features":[comb, cond], "conditions": [True, False], "rule" :  str(comb) + " and not(" + str(cond) + ")" }     
                split_rule_false = {"features":comb, "conditions": False, "rule" :   "not("+ str(comb) + ")" }     

                classes_true_true = split_true_true
                classes_true_false = split_true_false
                classes_false = split_false 
                        
                features_true_true = [f for f, v, c in zip(features, values, classes) if islistinlist(comb,f) and checkcondition( listwhereislistinlist(comb, f, v), cond )]
                features_true_false = [f for f, v, c in zip(features, values, classes) if islistinlist(comb,f) and not checkcondition( listwhereislistinlist(comb, f, v), cond )]
                features_false = [f for f, c in zip(features, classes) if not islistinlist(comb,f)]

                values_true_true = [v for f, v, c in zip(features, values, classes) if islistinlist(comb,f) and checkcondition( listwhereislistinlist(comb, f, v), cond )]
                values_true_false = [v for f, v, c in zip(features, values, classes) if islistinlist(comb,f) and not checkcondition( listwhereislistinlist(comb, f, v), cond )]
                values_false = [v for f,v, c in zip(features, values, classes) if not islistinlist(comb,f)]

        node_true_true =  node_tree(features_true_true, values_true_true, classes_true_true, gini_true_true, node.id, split_rule_true_true)
        node_true_false =  node_tree(features_true_false, values_true_false, classes_true_false, gini_true_false, node.id, split_rule_true_false)
        node_false = node_tree(features_false, values_false, classes_false, gini_false, node.id, split_rule_false)

        print("OHLIHO", len(classes_false), len(classes_true_true), len(classes_true_false), len(classes))

        return node_true_true, node_true_false, node_false, gain_gini


    def fit(self, features, values, classes):
        self.features = features
        self.classes = classes
        self.values = values
        gini = gini_impurity(classes, self.nclasses)
        root = node_tree(features, values,  classes, gini, 0, None)
        root.set_parent(root.id)
        self.root = root
        self.tree = [ root ]
        self.queue = [ root ]
        n = 0
        index = 0
        while not len(self.queue) == 0 and n < self.iteration_max:
            node = self.queue.pop(0)
            ntt, ntf, nf, gain_gini = self.split(node)
            splitted_nodes = [nf, ntt, ntf]

            for n_ in splitted_nodes:
               print(gain_gini, [x for i, x in enumerate(n_.classes) if i == n_.classes.index(x)], [x for i, x in enumerate(n_.classes) if i == n_.classes.index(x)])

            for n_ in splitted_nodes:
                if len(n_.classes)>0:
                    self.tree.append( n_ ) 
            
            if gain_gini > self.epsilon:
                for n_ in splitted_nodes:
                    if n_.gini > self.epsilon and len(n_.classes) >0:
                        self.queue.append(n_)
            n += 1
            index += 1
        self.rules = self.rules_per_class()

    def rules_per_class(self):
        leaves = self.get_leaves()
        branches = [(l.classes, self.get_branch(l)) for l in leaves]
        rules_per_class = [[] for _ in range(self.nclasses)]
        b = []
        for i, (classes, branch) in enumerate(branches):
            if not len(classes) == 0:
                setclasses =[x for i, x in enumerate(classes) if i == classes.index(x)]
                c = max( setclasses, key = classes.count)
                listofrule = [ (len(n.classes), n.split_rule, n.id) for n in branch if n.split_rule]
                #listofrule = [ (n.split_rule) for n in branch if n.split_rule]
                rules_per_class[c].append(listofrule)

        return rules_per_class
 
    def composition(self):
        leaves = self.get_leaves()
        branches = [(l.classes, self.get_branch(l)) for l in leaves]
        rules_per_class = [[] for _ in range(self.nclasses)]
        for i, (classes, branch) in enumerate(branches):
            if not len(classes) == 0:
                setclasses =[x for i, x in enumerate(classes) if i == classes.index(x)]
                c = max(setclasses, key = classes.count)
                listofrule = [n for n in branch if n.split_rule]
                rules_per_class[c].append(listofrule)
        return rules_per_class

    def is_class(self, feat, c):
        predicted_class = []
        rules_for_class = self.rules[c]    
        #pc = [True for _ in range(len(rules_for_class))]
        isclass = False
        for i, srules in enumerate(rules_for_class):
            fitrule = True
            for rule in srules:
                f1 = feat[rule["index"]]
                f2 = feat[rule["index"]+1]
                condition = f1 == rule["features"][0] and f2 == rule["features"][1]
                #pc[i] = pc[i] and condition == rule["condition"]
                fitrule = fitrule and condition == rule["conditions"]
            isclass = isclass or fitrule
        return isclass

    def inclusive_branch(self):
        leaves = self.get_leaves()
        branches = [(l.classes, self.get_branch(l)) for l in leaves]
        inclusive_branches = []
        for c, branch in branches:
            #print(c)
            condition = True
            for node in branch:
                if node.split_rule:
                    condition = condition and node.split_rule["conditions"]
                if condition == False:
                    break
            if condition == True:
                #inclusive_branch = branch
                inclusive_branches.append(branch)
        return inclusive_branches

    def predict(self, feat):
        predicted_class = []
        for c in range(len(self.rules)):
            ic = self.is_class(feat, c)    
            predicted_class.append(ic)
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
    labels=["N", "A", "B", "PP", "PN"]
    nlabels = len(labels)
    nclasses = 5
    nfeatures = 4


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
