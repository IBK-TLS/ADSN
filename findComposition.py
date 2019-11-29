import numpy as np
import pandas as pd
import argparse
import random
import uuid 

from CompositionTree import composition_tree, composition_tree_2, composition_tree_3



def main(args):

    ### DATA PREPARATION ###
 
    composition_known = [
        ["X"],
        ["N", "PN", "N"], 
        ["N", "PP", "N"], 
        ["N", "PP", "PN", "N"], 
        ["N", "PN", "PP", "N"], 
        ["N", "PN", "PP", "PN", "N"], 
    ] 
    labels=["N", "A", "B", "PP", "PN"]
    nlabels = len(labels)
    nclasses = len(composition_known)
    window = 8
 
    dataset = pd.read_csv(args.dataset)
    features = list(dataset["label"])
    #features = [features[x:x+window] for x in range(0, len(features), window)]
    features = [features[x:x+window] for x in range(len(features)- window)]
    classes = [0 for _ in range(len(features))]
    for i, f in enumerate(features):
        for j, c in enumerate(composition_known):
            #print(i, f, c )
            c_indices = [(i, i+len(c)) for i in range(len(f)) if f[i:i+len(c)] == c] 
            win_feat = [f[i:i+len(c)] for i in range(len(f)) if f[i:i+len(c)] == c] 
            if not len(c_indices) == 0:
                #print(i, j, c, c_indices, f )
                classes[i] = j

    print("histo", np.histogram(classes, bins=list(range(nclasses))), nclasses)

    if not len(features[-1]) == window:
        _ = features.pop(-1)
        _ = classes.pop(-1) 

    ### 

    compotree = composition_tree_3(nclasses, window, labels, iteration_max=100000000)
    compotree.fit(features, classes)
    
    compositions = compotree.rules_per_class()
    for i, rules in enumerate(compositions):
        print("class", i, composition_known[i] )
        for j, rule_branch in enumerate(rules):
            #pruning_branch_rule(rule_branch, window)
            print("support:", rule_branch[0][0])
            for k, (n, r, uuid) in enumerate(rule_branch):
                print(r["rule"], end=" ")
                print("AND" if k < len(rule_branch)-1 else "\n", end=" " )
            print("OR" if j < len(rules)-1 else "" )
        print("#####")
   
    #compotree.merge_branches_per_class()
 
    #compositions = compotree.rules_per_class()
    #for i, rules in enumerate(compositions):
    #    print("class", i, composition_known[i] )
    #    for j, rule_branch in enumerate(rules):
    #        #pruning_branch_rule(rule_branch, window)
    #        #print("support:", rule_branch[0][0])
    #        for k, (n, r, uuid) in enumerate(rule_branch):
    #            print(r["rule"], end=" ")
    #            print("AND" if k < len(rule_branch)-1 else "\n", end=" " )
    #        print("OR" if j < len(rules)-1 else "" )
    #    print("#####")
     
    compotree.prune_branches_per_class()
    
    compositions = compotree.branches
    for i, rules in enumerate(compositions):
        print("class", i, composition_known[i] )
        for j, rule_branch in enumerate(rules):
            #print("support:", rule_branch[0][0])
            for k, node in enumerate(rule_branch):
                rule = node.split_rule
                print(rule["rule"], end=" ")
                print("AND" if k < len(rule_branch)-1 else "\n", end=" " )
            print("OR" if j < len(rules)-1 else "" )
        print("#####")
   
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Composition-based desicion tree utilities.")
    parser.add_argument("-f", "--file-dataset",
                        dest="dataset",
                        type=str,
                        help="dataset csv file",)  
    args = parser.parse_args()
    main(args)
