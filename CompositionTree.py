import numpy as np
import pandas as pd
import argparse
import random

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

class composition_tree():
    def __init__(self):
        self.data = []
        self.classes = []
        self.levels = []
        

def best_split(db, lb, index):
    gini = gini_impurity(lb)
    split = []
    branch_true = []
    branch_false = []
    N = len(lb)
    gain_gini = 0
    for f1 in range(nlabels):
        for f2 in range(nlabels):
            fit = [l for d, l in zip(db, lb) if d[index] == f1 and d[index+1] == f2]
            notfit = [l for d, l in zip(db, lb) if not d[index] == f1 or not d[index+1] == f2]
            g_fit = gini_impurity(fit)
            g_notfit = gini_impurity(notfit)
            g = ( g_fit * (len(fit)/N) + g_notfit * (len(notfit)/N) )
            gain = gini - g
            #print(gini, g, gain, [f1, f2])
            if gain > gain_gini :
                #gini = g
                gain_gini = gain
                split = [f1, f2]     
                branch_true = [d for d in db if d[index] == f1 and d[index+1] == f2]
                branch_false = [d for d in db if not d[index] == f1 or not d[index+1] == f2]
                #print(gini, gain_gini, split)
                
    return branch_true, branch_false, split, gini, gain_gini


    
### MAIN ###


def main(args):
    db, lb = genfakedb()
    c = list(zip(db, lb))
    random.shuffle(c)
    db, lb = zip(*c)
    branch_true, branch_false, split, gini , gain = best_split(db, lb, 0)
    print(split, gini, gain)
    print(branch_true)
    print(branch_false)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Composition-based desicion tree utilities.")
    parser.add_argument("-d", "--dataset",
                        dest="dataset",
                        type=str,
                        help="dataset csv file",)  
    args = parser.parse_args()
    main(args)
