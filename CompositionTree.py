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

def rule_horizontal_pruning(rules):
    for j, rule_branch in enumerate(rules):
        print("support:", rule_branch[0][0])
        for k, (n, r) in enumerate(rule_branch):
            print(r, end=" ")
            print("AND" if k < len(rule_branch)-1 else "\n", end=" " )
        print("OR" if j < len(rules)-1 else "" )


def construct_rulestr(feat, cond, ind):
    string = ""
    for f, c, i in zip(feat, cond, ind):
        sc = "" if c is True else "not("
        ec = "" if c is True else ")"
        string += sc+f+"_"+str(i)+ec+"."

    return string[:-1] # remove the last "." char

def rule2str(rule):
    
    cond1 = "" if rule["conditions"][0] is None else "not(" if not(rule["conditions"][0]) else ""
    endcond1 = "" if rule["conditions"][0] is None else ")" if not(rule["conditions"][0]) else ""

    cond2 = "" if rule["conditions"][1] is None else ".not(" if not(rule["conditions"][1]) else "."
    endcond2 = "" if rule["conditions"][1] is None else ")" if not(rule["conditions"][1]) else ""
    feat1 = "" if rule["features"][0] is None else rule["features"][0]
    i1 = "" if rule["features"][0] is None else "_"+str(rule["index"])
    feat2 = "" if rule["features"][1] is None else rule["features"][1]
    i2 = "" if rule["features"][1] is None else "_"+str(rule["index"]+1)
    
    return cond1+feat1+i1+endcond1+cond2+feat2+i2+endcond2
    

def gini_impurity(data, nclasses):
    prob = [0. for _ in  range(nclasses)]
    N = len(data)
    for i in data:
        prob[i] += 1/N
    prob  = np.array(prob)
    return np.sum(prob*(1-prob))

class node_tree():
    def __init__(self, features, classes, gini, parent, split_rule, active=True):
        self.features = features
        self.classes = classes
        self.gini = gini
        self.parent = parent
        self.split_rule = split_rule
        self.id = uuid.uuid1()
        self.active = active
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
    def activate(self):    
        self.active = True
    def deactivate(self):    
        self.active = False
    def dict(self):
        d = {"features": self.features, "classes": self.classes, "gini": self.gini, "id": self.id, "parent": self.parent, "split_rule": self.split_rule, "active":self.active}
        return d

class branch_tree():
    def __init__(self, nodes, nfeat):
        self.nodes = copy.deepcopy(nodes)
        self.nfeat = nfeat
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
        

class composition_tree_3():
    def __init__(self, nclasses=2, nfeat=2, labels=None, iteration_max=10000):
        self.nclasses = nclasses
        self.nfeat = nfeat
        if labels:
            self.labels = labels
        else:
            self.labels = list(range(nfeat))
        self.tree = []
        self.branches = []
        self.queue = []
        self.rules = []
        self.root = None
        self.iteration_max = iteration_max
        
    def split(self, node):
        features = node.features
        classes = node.classes
        parent = node.parent
        gini_orig = node.gini
        gini_types = []
        features_types = []
        classes_types = []
        N = len(classes)
        gain_gini = 0
        for index in range(self.nfeat-1):
            for f1 in self.labels:
                for f2 in self.labels: 
                    split_types = [] 
                    label_types = [] 
                    conds_types = []
                    feats_types = []
                    index_types = []
                    for cond1, cond2 in [(True, True), (False, True), (True, False), (False, False) ]:
                        split_types.append([c for f, c in zip(features, classes) if ((f[index] == f1) is cond1) and ((f[index+1] == f2) is cond2)])
                        label_types.append([f for f in features if ((f[index] == f1) is cond1) and ((f[index+1] == f2) is cond2)])
                        conds_types.append([cond1, cond2]) 
                        feats_types.append([f1, f2]) 
                        index_types.append([index, index+1]) 
                    
                    #### following commented code relies on merging branch during split (seems to not be very interesting... post merging seems to be better)
                    #c_types = [np.unique(st) for st in split_types]
                    #m_split_types = split_types 
                    #l_split_types = label_types 
                    #a_split_types = [True for _ in split_types]
                    #c_split_types = conds_types
                    #f_split_types = feats_types
                    #i_split_types = index_types
                    ## merging identical split
                    #for (i, j) in itertools.combinations(range(len(c_types)), 2):
                    #    ct1 = c_types[i]
                    #    ct2 = c_types[j]
                    #    st1 = split_types[i]
                    #    st2 = split_types[j]
                    #    la1 = label_types[i]
                    #    la2 = label_types[j]
                    #    co1 = conds_types[i] 
                    #    co2 = conds_types[j] 
                    #    fe1 = feats_types[i] 
                    #    in1 = index_types[i] 
                    #    #print("testing", ct1, ct2)
                    #    if len(ct1) == 1 and len(ct2)==1 and ct1 == ct2 and simplification(co1, co2):
                    #        a_split_types[i] = False
                    #        a_split_types[j] = False
                    #        new = simplification(co1, co2)
                    #        c_split_types.append([co1[new]])
                    #        f_split_types.append([fe1[new]])
                    #        i_split_types.append([in1[new]])
                    #        m_split_types.append(st1+st2)
                    #        l_split_types.append(la1+la2)
                    #        a_split_types.append(True)
                    #        
                    #split_types = [st for at, st in zip(a_split_types, m_split_types) if at ] 
                    #label_types = [la for at, la in zip(a_split_types, l_split_types) if at ] 
                    #feats_types = [ft for at, ft in zip(a_split_types, f_split_types) if at ] 
                    #conds_types = [ct for at, ct in zip(a_split_types, c_split_types) if at ] 
                    #index_types = [it for at, it in zip(a_split_types, i_split_types) if at ] 
                    g_types = [gini_impurity(st, self.nclasses) for st in split_types]
                    
                    g = 0
                    for gt, st in zip(g_types, split_types):
                        g += gt * (len(st)/N) 
                    gain = gini_orig - g
                    if gain > gain_gini :
                        gain_gini = gain
                        gini_types = [gt for gt in g_types]
                        split_rule_types = []
                        for f, c, i in zip(feats_types, conds_types, index_types):
                            strrule = construct_rulestr(f, c, i)
                            split_rule_types.append({"indices":i, "features":f, "conditions": c, "rule":strrule})
                        classes_types = [ st for st in split_types ]
                        features_types = [la for la in label_types]
                        
        node_types = []
        for i, ft in enumerate(features_types):
            node_types.append( node_tree(features_types[i], classes_types[i], gini_types[i], node.id, split_rule_types[i]))

        return node_types, gain_gini

    def fit(self, features, classes):
        self.features = features
        self.classes = classes
        gini = gini_impurity(classes, self.nclasses)
        root = node_tree(features, classes, gini, 0, None)
        root.set_parent(root.id)
        self.root = root
        self.tree = [ root ]
        self.queue = [ root ]
        n = 0
        index = 0
        while not len(self.queue) == 0 and n < self.iteration_max:
            node = self.queue.pop(0)
            nodes, gain_gini = self.split(node)
            for n_ in nodes:
                self.tree.append( n_ ) 
            if gain_gini > 0:
                #print(gain_gini)
                for n_ in nodes:
                    if n_.gini > 0:
                        #print(" ", n_.gini)
                        self.queue.append(n_)
            n += 1
            index += 1
        print("fit stopped:", n, " iterations - ", len(self.tree), " nodes" )
         
    def prune_branches_per_class(self):
        branches_per_class = self.branches_per_class() 
        pruned_branches_per_class = [ ]
        for c, branches in enumerate(branches_per_class):
            pruned_branches = []
            for i, branch in enumerate(branches):
                branch_ = branch_tree(branch, self.nfeat)
                branch_.prune_branch()
                pruned_branches.append(branch_.nodes)
            pruned_branches_per_class.append(pruned_branches)
        self.branches = pruned_branches_per_class

    def merge_branches_per_class(self):
        branches_per_class = self.branches_per_class()
        for c, branches in enumerate(branches_per_class):
            for i, branch1 in enumerate(branches):
                for j, branch2 in [(i+1+n, branches[i+1+n]) for n in range(len(branches)-(i+1))]:
                    for k, node1 in enumerate([branch1[0]]):
                        for l, node2 in enumerate([branch2[0]]):
                            if node1.parent == node2.parent:
                                if node1.split_rule["features"] == node2.split_rule["features"]:
                                    #print(c, i, j, k, l, node1.split_rule["rule"], node2.split_rule["rule"])
                                    c11 = node1.split_rule["conditions"][0]
                                    c12 = node1.split_rule["conditions"][1]
                                    c21 = node2.split_rule["conditions"][0]
                                    c22 = node2.split_rule["conditions"][1]
                        
                                    if (c11 is not(c21)) and (c12 is c22):
                                        print("merging..." )
                                        features = node1.features + node2.features
                                        classes = node1.classes + node2.classes
                                        gini = gini_impurity(classes, self.nclasses)
                                        parent = node1.parent
                                        split_rule = copy.deepcopy(node1.split_rule)
                                        split_rule["features"][0] = None
                                        split_rule["conditions"][0] = None
                                        cond = "" if split_rule["conditions"][1] else "not("
                                        endcond = "" if split_rule["conditions"][1] else ")"
                                        split_rule["rule"] = cond+split_rule["features"][1]+"_"+str(split_rule["indices"][1])+endcond
                                        
                                        merged_node = node_tree(features, classes, gini, parent, split_rule)
                                        
                                        self.tree.append(merged_node)
                                        self.delete_node(node1)
                                        self.delete_node(node2)

                                    elif (c11 is c21) and (c12 is not (c22)) :
                                        print("merging..." )
                                        features = node1.features + node2.features
                                        classes = node1.classes + node2.classes
                                        gini = gini_impurity(classes, self.nclasses)
                                        parent = node1.parent
                                        split_rule = copy.deepcopy(node1.split_rule)
                                        split_rule["features"][1] = None
                                        split_rule["conditions"][1] = None
                                        cond = "" if split_rule["conditions"][0] else "not("
                                        endcond = "" if split_rule["conditions"][0] else ")"
                                        split_rule["rule"] = cond+split_rule["features"][0]+"_"+str(split_rule["indices"][0])+endcond
                                                                                                                    
                                        merged_node = node_tree(features, classes, gini, parent, split_rule)
                                        self.tree.append(merged_node)
                                        self.delete_node(node1)
                                        self.delete_node(node2)
        
                                

    def branches_per_class(self):
        leaves = self.get_leaves()
        branches = [(l.classes, self.get_branch(l)) for l in leaves]
        branches_per_class = [[] for _ in range(self.nclasses)]
        for i, (classes, branch) in enumerate(branches):
            if not len(classes) == 0:
                c = max(set(classes), key = classes.count)
                listofnodes = [ n for n in branch if n.split_rule]
                branches_per_class[c].append(listofnodes)
        return branches_per_class

    def rules_per_class(self):
        leaves = self.get_leaves()
        branches = [(l.classes, self.get_branch(l)) for l in leaves]
        rules_per_class = [[] for _ in range(self.nclasses)]
        b = []
        for i, (classes, branch) in enumerate(branches):
            if not len(classes) == 0:
                c = max(set(classes), key = classes.count)
                listofrule = [ (len(n.classes), n.split_rule, n.id) for n in branch if n.split_rule]
                rules_per_class[c].append(listofrule)
        return rules_per_class
 
    def composition(self):
        leaves = self.get_leaves()
        branches = [(l.classes, self.get_branch(l)) for l in leaves]
        rules_per_class = [[] for _ in range(self.nclasses)]
        for i, (classes, branch) in enumerate(branches):
            if not len(classes) == 0:
                c = max(set(classes), key = classes.count)
                listofrule = [(len(n.classes), n.split_rule["rule"]) for n in branch if n.split_rule]
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
                f1 = feat[rule["indices"][0]]
                f2 = feat[rule["indices"][1]]
                condition = f1 == rule["features"][0] and f2 == rule["features"][1]
                #pc[i] = pc[i] and condition == rule["condition"]
                fitrule = fitrule and condition == rule["conditions"]
            isclass = isclass or fitrule
        return isclass

  
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

    def delete_node(self, node):
        for i, _node in enumerate(self.tree):
            if _node.id == node.id:
                del self.tree[i]
    

class composition_tree_2():
    def __init__(self, nclasses=2, nfeat=2, labels=None, iteration_max=10000):
        self.nclasses = nclasses
        self.nfeat = nfeat
        if labels:
            self.labels = labels
        else:
            self.labels = list(range(nfeat))
        self.tree = []
        self.branches = []
        self.queue = []
        self.rules = []
        self.root = None
        self.iteration_max = iteration_max
        
    def split(self, node):
        features = node.features
        classes = node.classes
        parent = node.parent
        gini_orig = node.gini
        split_rule_type1 = []
        split_rule_type2 = []
        split_rule_type3 = []
        split_rule_type4 = []
        classes_type1 = []
        classes_type2 = []
        classes_type3 = []
        classes_type4 = []
        features_type1 = []
        features_type2 = []
        features_type3 = []
        features_type4 = []
        gini_type1 = 0
        gini_type2 = 0
        gini_type3 = 0
        gini_type4 = 0
        N = len(classes)
        gain_gini = 0
        for index in range(self.nfeat-1):
            for f1 in self.labels: #range(self.nclasses):
                for f2 in self.labels: #range(self.nclasses):
                    split_type1 = [c for f, c in zip(features, classes) if f[index] == f1 and f[index+1] == f2]
                    split_type2 = [c for f, c in zip(features, classes) if not f[index] == f1 and f[index+1] == f2]
                    split_type3 = [c for f, c in zip(features, classes) if f[index] == f1 and not f[index+1] == f2]
                    split_type4 = [c for f, c in zip(features, classes) if not f[index] == f1 and not f[index+1] == f2]
                    g_type1 = gini_impurity(split_type1, self.nclasses)
                    g_type2 = gini_impurity(split_type2, self.nclasses)
                    g_type3 = gini_impurity(split_type3, self.nclasses)
                    g_type4 = gini_impurity(split_type4, self.nclasses)
                    g = 0
                    g += g_type1 * (len(split_type1)/N) 
                    g += g_type2 * (len(split_type2)/N) 
                    g += g_type3 * (len(split_type3)/N) 
                    g += g_type4 * (len(split_type4)/N) 
                    gain = gini_orig - g
                    if gain > gain_gini :
                        gain_gini = gain
                        
                        gini_type1 = g_type1
                        gini_type2 = g_type2
                        gini_type3 = g_type3
                        gini_type4 = g_type4
                        
                        split_rule_type1 = {"index":index, "features":[f1, f2], "conditions": [True, True], "rule":"" +str(f1)+"_"+str(index)+"."+str(f2)+"_"+str(index+1)+""}     
                        split_rule_type2 = {"index":index, "features":[f1, f2], "conditions": [False, True], "rule":"not("+str(f1)+"_"+str(index)+")."+str(f2)+"_"+str(index+1)+""}     
                        split_rule_type3 = {"index":index, "features":[f1, f2], "conditions": [True, False], "rule":"" +str(f1)+"_"+str(index)+".not("+str(f2)+"_"+str(index+1)+")"}     
                        split_rule_type4 = {"index":index, "features":[f1, f2], "conditions": [False, False], "rule":"not("+str(f1)+"_"+str(index)+").not("+str(f2)+"_"+str(index+1)+")"}     
                        classes_type1 = split_type1
                        classes_type2 = split_type2
                        classes_type3 = split_type3
                        classes_type4 = split_type4
                        
                                                
                        features_type1 = [f for f in features if f[index] == f1 and f[index+1] == f2]
                        features_type2 = [f for f in features if not f[index] == f1 and f[index+1] == f2]
                        features_type3 = [f for f in features if f[index] == f1 and not f[index+1] == f2]
                        features_type4 = [f for f in features if not f[index] == f1 and not f[index+1] == f2]
        node_type1 = node_tree(features_type1, classes_type1, gini_type1, node.id, split_rule_type1)
        node_type2 = node_tree(features_type2, classes_type2, gini_type2, node.id, split_rule_type2)
        node_type3 = node_tree(features_type3, classes_type3, gini_type3, node.id, split_rule_type3)
        node_type4 = node_tree(features_type4, classes_type4, gini_type4, node.id, split_rule_type4)
        return [node_type1, node_type2, node_type3, node_type4], gain_gini

    def fit(self, features, classes):
        self.features = features
        self.classes = classes
        gini = gini_impurity(classes, self.nclasses)
        root = node_tree(features, classes, gini, 0, None)
        root.set_parent(root.id)
        self.root = root
        self.tree = [ root ]
        self.queue = [ root ]
        n = 0
        index = 0
        while not len(self.queue) == 0 and n < self.iteration_max:
            node = self.queue.pop(0)
            nodes, gain_gini = self.split(node)
            for n_ in nodes:
                self.tree.append( n_ ) 
            if gain_gini > 0:
                #print(gain_gini)
                for n_ in nodes:
                    if n_.gini > 0:
                        #print(" ", n_.gini)
                        self.queue.append(n_)
            n += 1
            index += 1
        print("fit stopped:", n, " iterations - ", len(self.tree), " nodes" )
        self.rules = self.rules_per_class()
        self.branches = self.branches_per_class()
        
    def prune_branches_per_class(self):
        branches_per_class = self.branches_per_class() 
        for c, branches in enumerate(branches_per_class):
            for i, branch in enumerate(branches):
                rules = {}
                for i in range(self.nfeat):
                    rules[i] = None
                for node in branch:
                    rule = node.split_rule
                    if rule["conditions"][0] == True:
                        rules[rule["index"]] = { "labels" : rule["features"][0],
                                              "condition": rule["conditions"][0] }
                    if rule["conditions"][1] == True:
                        rules[rule["index"]+1] = { "labels" : rule["features"][1],
                                                "condition" :rule["conditions"][1] }
                for k, v in rules.items():
                    if not v == None:
                        for node in branch:
                            rule = node.split_rule
                            if rule["index"] == k:
                                rule["features"][0] = v["labels"]
                                rule["conditions"][0] = True
                            if rule["index"]+1 == k:
                                rule["features"][1] = v["labels"]
                                rule["conditions"][1] = True   
                            rule["rule"] = rule2str(rule)
                

    def merge_branches_per_class(self):
        branches_per_class = self.branches_per_class()
        for c, branches in enumerate(branches_per_class):
            for i, branch1 in enumerate(branches):
                for j, branch2 in [(i+1+n, branches[i+1+n]) for n in range(len(branches)-(i+1))]:
                    for k, node1 in enumerate([branch1[0]]):
                        for l, node2 in enumerate([branch2[0]]):
                            if node1.parent == node2.parent:
                                if node1.split_rule["features"] == node2.split_rule["features"]:
                                    #print(c, i, j, k, l, node1.split_rule["rule"], node2.split_rule["rule"])
                                    c11 = node1.split_rule["conditions"][0]
                                    c12 = node1.split_rule["conditions"][1]
                                    c21 = node2.split_rule["conditions"][0]
                                    c22 = node2.split_rule["conditions"][1]
                        
                                    if (c11 is not(c21)) and (c12 is c22):
                                        print("merging..." )
                                        features = node1.features + node2.features
                                        classes = node1.classes + node2.classes
                                        gini = gini_impurity(classes, self.nclasses)
                                        parent = node1.parent
                                        split_rule = copy.deepcopy(node1.split_rule)
                                        split_rule["features"][0] = None
                                        split_rule["conditions"][0] = None
                                        cond = "" if split_rule["conditions"][1] else "not("
                                        endcond = "" if split_rule["conditions"][1] else ")"
                                        split_rule["rule"] = cond+split_rule["features"][1]+"_"+str(split_rule["index"])+endcond
                                        
                                        merged_node = node_tree(features, classes, gini, parent, split_rule)
                                        
                                        self.tree.append(merged_node)
                                        self.delete_node(node1)
                                        self.delete_node(node2)

                                    elif (c11 is c21) and (c12 is not (c22)) :
                                        print("merging..." )
                                        features = node1.features + node2.features
                                        classes = node1.classes + node2.classes
                                        gini = gini_impurity(classes, self.nclasses)
                                        parent = node1.parent
                                        split_rule = copy.deepcopy(node1.split_rule)
                                        split_rule["features"][1] = None
                                        split_rule["conditions"][1] = None
                                        cond = "" if split_rule["conditions"][0] else "not("
                                        endcond = "" if split_rule["conditions"][0] else ")"
                                        split_rule["rule"] = cond+split_rule["features"][0]+"_"+str(split_rule["index"])+endcond
                                                                                                                    
                                        merged_node = node_tree(features, classes, gini, parent, split_rule)
                                        self.tree.append(merged_node)
                                        self.delete_node(node1)
                                        self.delete_node(node2)
        
                                

    def branches_per_class(self):
        leaves = self.get_leaves()
        branches = [(l.classes, self.get_branch(l)) for l in leaves]
        branches_per_class = [[] for _ in range(self.nclasses)]
        for i, (classes, branch) in enumerate(branches):
            if not len(classes) == 0:
                c = max(set(classes), key = classes.count)
                listofnodes = [ n for n in branch if n.split_rule]
                branches_per_class[c].append(listofnodes)
        return branches_per_class

    def rules_per_class(self):
        leaves = self.get_leaves()
        branches = [(l.classes, self.get_branch(l)) for l in leaves]
        rules_per_class = [[] for _ in range(self.nclasses)]
        b = []
        for i, (classes, branch) in enumerate(branches):
            if not len(classes) == 0:
                c = max(set(classes), key = classes.count)
                listofrule = [ (len(n.classes), n.split_rule, n.id) for n in branch if n.split_rule]
                rules_per_class[c].append(listofrule)
        return rules_per_class
 
    def composition(self):
        leaves = self.get_leaves()
        branches = [(l.classes, self.get_branch(l)) for l in leaves]
        rules_per_class = [[] for _ in range(self.nclasses)]
        for i, (classes, branch) in enumerate(branches):
            if not len(classes) == 0:
                c = max(set(classes), key = classes.count)
                listofrule = [(len(n.classes), n.split_rule["rule"]) for n in branch if n.split_rule]
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

    def delete_node(self, node):
        for i, _node in enumerate(self.tree):
            if _node.id == node.id:
                del self.tree[i]
    

class composition_tree():
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
                    g_true = gini_impurity(split_true, self.nclasses)
                    g_false = gini_impurity(split_false, self.nclasses)
                    g = ( g_true * (len(split_true)/N) + g_false * (len(split_false)/N) )
                    gain = gini_orig - g
                    #print(gain)
                    if gain > gain_gini :
                        gain_gini = gain
                        gini_true = g_true
                        gini_false = g_false
                        split_rule_true = {"index":index, "features":[f1, f2], "conditions": True, "rule" :  "(" +str(f1) + "_" + str(index) + "." + str(f2) + "_" + str(index+1) + ")"}     
                        split_rule_false = {"index":index, "features":[f1, f2], "conditions": False, "rule": "not(" + str(f1) + "_" + str(index) + "." + str(f2) + "_" + str(index+1) + ")"  }     
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
        gini = gini_impurity(classes, self.nclasses)
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
            if gain_gini > self.epsilon:
                if node_true.gini > self.epsilon:
                    self.queue.append(node_true)
                if node_false.gini > self.epsilon:
                    self.queue.append(node_false)
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
                c = max(set(classes), key = classes.count)
                listofrule = [n.split_rule for n in branch if n.split_rule]
                rules_per_class[c].append(listofrule)
        return rules_per_class
 
    def composition(self):
        leaves = self.get_leaves()
        branches = [(l.classes, self.get_branch(l)) for l in leaves]
        rules_per_class = [[] for _ in range(self.nclasses)]
        for i, (classes, branch) in enumerate(branches):
            if not len(classes) == 0:
                c = max(set(classes), key = classes.count)
                listofrule = [n.split_rule["rule"] for n in branch if n.split_rule]
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
    
    
class inclusive_composition_tree():
    def __init__(self, nclasses=2, nfeat=2, labels=None, iteration_max=10, epsilon = 1e-6):
        self.trees = []
        self.nclasses = nclasses
        self.nfeat = nfeat
        self.labels = labels
        self.iteration_max = iteration_max
        self.epsilon = epsilon
        self.features = []
        self.classes = []
        self.inclusive_branches = []

    def fit(self, features, classes):
        self.features = features
        self.classes = classes
        
        feat_ = self.features 
        class_ = self.classes
        n=0
        while len(feat_) > 0 and n < self.iteration_max:
            print(n , len(feat_), list(set(class_)) )
            compotree = composition_tree(self.nclasses, self.nfeat, self.labels)
            compotree.fit(feat_, class_)
            inclusive_branch = compotree.inclusive_branch()[0]
            feat_ib = inclusive_branch[0].features
            class_ib = inclusive_branch[0].classes
            self.inclusive_branches.append(inclusive_branch)
            self.trees.append(compotree)
            c = list(set(class_ib))
            print("  ", c, [n.split_rule["rule"] for n in inclusive_branch if n.split_rule ])    
            print("    ", feat_ib, class_ib)
            feat_ = [f for f in feat_ if not f in feat_ib]
            class_ = [c for (c,f) in zip(class_, feat_) if not f in feat_ib]
            n += 1    


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
