import numpy as np
import pandas as pd
import argparse
import random
import uuid 

from CompositionConditionTree import composition_condition_tree



def main(args):

    ### DATA PREPARATION ###
 
    window = 8
    step = 1 
 
    dataset = pd.read_csv(args.dataset)
    features = list(dataset["label"])
    classes = list(dataset["class"])
    values = list(dataset["value"])
    labels = [x for i, x in enumerate(features) if i == features.index(x)]
    uclasses = [x for i, x in enumerate(classes) if i == classes.index(x)]
    nlabels = len(labels)
    nclasses = len([x for i, x in enumerate(classes) if i == classes.index(x)])

    values = [values[x:x+window] for x in np.arange(0,len(values)- window, step)]
     
    fclasses = [f for f in classes ]
    #fclasses = [classes[x:x+window] for x in np.arange(0,len(features)- window, step) if classes[x:x+window][0] == 0 and classes[x:x+window][-1] == 0]
    fclasses = [classes[x:x+window] for x in np.arange(0,len(features)- window, step)]
    features = [f for f in features ]
    #features = [features[x:x+window] for x in np.arange(0,len(features)- window, step) if classes[x:x+window][0] == 0 and classes[x:x+window][-1] == 0]
    features = [features[x:x+window] for x in np.arange(0,len(features)- window, step)]
    classes = [0 for _ in range(len(fclasses))]
    
    for i, f in enumerate(fclasses):
        #uniqueclasses = [x for i, x in enumerate(f) if i == f.index(x) and x != 0 and ( 1 < i < len(f)-2 )]
        uniqueclasses = [x for i, x in enumerate(f) if i == f.index(x) and x!=0]
        if not len(uniqueclasses) == 0:
            print(i, f, uniqueclasses)
            if len(uniqueclasses) == 1:
                classes[i] = uniqueclasses[0]
            else:
                classes[i] = uniqueclasses

    histo = [0 for _ in range(nclasses)]
    for c in classes:
        if type(c) is int:
            histo[uclasses.index(c)] += 1
        elif type(c) is list:
            for i in c:
                histo[uclasses.index(i)] += 1
        
    print("histo", histo, nclasses)

    if not len(features[-1]) == window:
        _ = features.pop(-1)
        _ = classes.pop(-1) 

    ### 

    compotree = composition_condition_tree(nclasses, window, labels,  iteration_max=100000000)
    compotree.fit(features, values, classes)
    
    compositions = compotree.composition()
    for i, rules in enumerate(compositions):
        print("class", i )
        for j, rule_branch in enumerate(rules):
            #pruning_branch_rule(rule_branch, window)
            setclasses =[x for i, x in enumerate(rule_branch[0].classes) if i == rule_branch[0].classes.index(x)]
            print("support:", len(rule_branch[0].classes), setclasses)
            for k, r in enumerate(rule_branch):
                print(r.split_rule["rule"], end=" ")
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
