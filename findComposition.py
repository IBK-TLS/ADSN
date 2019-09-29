import numpy as np
import pandas as pd
import argparse
import random
import uuid 

from CompositionTree import composition_tree 




def main(args):
    composition_known = [
        ["N", "PN", "N"], 
        ["N", "PP", "N"], 
        ["N", "PP", "PN", "N"], 
        ["N", "PP", "PN", "N"], 
        ["N", "PN", "PP", "PN", "N"], 
    ] 
    labels=["N", "A", "B", "PP", "PN"]
    nlabels = len(labels)
    nclasses = len(composition_known)+1
    window = 4
 
    dataset = pd.read_csv(args.dataset)
    features = list(dataset["label"])
    features = [features[x:x+window] for x in range(0, len(features), window)]
    classes = [0] * len(features)
    #features = list(np.array(features).reshape((-1, 4)))
    #print(features[:5])
    for i, f in enumerate(features):
        for j, c in enumerate(composition_known):
            #print(i, f, c )
            c_indices = [(i, i+len(c)) for i in range(len(f)) if f[i:i+len(c)] == c] 
            if not len(c_indices) == 0:
                classes[i] = j+1
    
    compotree = composition_tree(nclasses, window, labels)
    compotree.fit(features, classes)
    compositions = compotree.composition()
    for i, rpc in enumerate(compositions):
        print("class", i)
        for r in rpc:
            print(r)
            print("or")
        print()

    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Composition-based desicion tree utilities.")
    parser.add_argument("-f", "--file-dataset",
                        dest="dataset",
                        type=str,
                        help="dataset csv file",)  
    args = parser.parse_args()
    main(args)
