
# code for IDS with deterministic local search
# requires installation of python package apyori: https://pypi.org/project/apyori/

import numpy as np
import pandas as pd
import math
from apyori import apriori


# rule is of the form if A == a and B == b, then class_1
# one of the member variables is itemset - a set of patterns {(A,a), (B,b)}
# the other member variable is class_label (e.g., class_1)
class rule:
    
    def __init__(self,feature_list,value_list,class_label):
        self.itemset = set()
        self.class_label = None
        self.add_item(feature_list,value_list)
        self.set_class_label(class_label)
    
    def add_item(self,feature_list,value_list):
        
        if len(feature_list) != len(value_list):
            print("Some error in inputting feature value pairs")
            return
        for i in range(0,len(feature_list)):
            self.itemset.add((feature_list[i],value_list[i]))
    
    def print_rule(self):
        s = "If "
        for item in self.itemset:
            s += str(item[0]) + " == " +str(item[1]) + " and "
        s = s[:-5]
        s += ", then "
        s += str(self.class_label)
        print(s)
        
    def all_predicates_same(self, r):
        return self.itemset == r.itemset
    
    def class_label_same(self,r):
        return self.class_label == r.class_label
            
    def set_class_label(self,label):
        self.class_label = label
        
    def get_length(self):
        return len(self.itemset)
    
    def get_cover(self, df):
        dfnew = df.copy()
        for pattern in self.itemset: 
            dfnew = dfnew[dfnew[pattern[0]] == pattern[1]]
        return list(dfnew.index.values)

    def get_correct_cover(self, df, Y):
        indexes_points_covered = self.get_cover(df) # indices of all points satisfying the rule
        Y_arr = pd.Series(Y)                    # make a series of all Y labels
        labels_covered_points = list(Y_arr[indexes_points_covered])   # get a list only of Y labels of the points covered
        correct_cover = []
        for ind in range(0,len(labels_covered_points)):
            if labels_covered_points[ind] == self.class_label:
                correct_cover.append(indexes_points_covered[ind])
        return correct_cover, indexes_points_covered
    
    def get_incorrect_cover(self, df, Y):
        correct_cover, full_cover = self.get_correct_cover(df, Y)
        return (sorted(list(set(full_cover) - set(correct_cover))))


# below function basically takes a data frame and a support threshold and returns itemsets which satisfy the threshold
def run_apriori(df, support_thres):
    # the idea is to basically make a list of strings out of df and run apriori api on it 
    # return the frequent itemsets
    dataset = []
    for i in range(0,df.shape[0]):
        temp = []
        for col_name in df.columns:
            temp.append(col_name+"="+str(df[col_name][i]))
        dataset.append(temp)

    results = list(apriori(dataset, min_support=support_thres))
    
    list_itemsets = []
    for ele in results:
        temp = []
        for pred in ele.items:
            temp.append(pred)
        list_itemsets.append(temp)

    return list_itemsets


# This function converts a list of itemsets (stored as list of lists of strings) into rule objects
def createrules(freq_itemsets, labels_set):
    # create a list of rule objects from frequent itemsets 
    list_of_rules = []
    for one_itemset in freq_itemsets:
        feature_list = []
        value_list = []
        for pattern in one_itemset:
            fea_val = pattern.split("=")
            feature_list.append(fea_val[0])
            value_list.append(fea_val[1])
        for each_label in labels_set:
            temp_rule = rule(feature_list,value_list,each_label)
            list_of_rules.append(temp_rule)

    return list_of_rules


# compute the maximum length of any rule in the candidate rule set
def max_rule_length(list_rules):
    len_arr = []
    for r in list_rules:
        len_arr.append(r.get_length())
    return max(len_arr)


# compute the number of points which are covered both by r1 and r2 w.r.t. data frame df
def overlap(r1, r2, df):
    return sorted(list(set(r1.get_cover(df)).intersection(set(r2.get_cover(df)))))


# computes the objective value of a given solution set
def func_evaluation(soln_set, list_rules, df, Y, lambda_array):
    # evaluate the objective function based on rules in solution set 
    # soln set is a set of indexes which when used to index elements in list_rules point to the exact rules in the solution set
    # compute f1 through f7 and we assume there are 7 lambdas in lambda_array
    f = [] #stores values of f1 through f7; 
    
    # f0 term
    f0 = len(list_rules) - len(soln_set) # |S| - size(R)
    f.append(f0)
    
    # f1 term
    Lmax = max_rule_length(list_rules)
    sum_rule_length = 0.0
    for rule_index in soln_set:
        sum_rule_length += list_rules[rule_index].get_length()
    
    f1 = Lmax * len(list_rules) - sum_rule_length
    f.append(f1)
    
    # f2 term - intraclass overlap
    sum_overlap_intraclass = 0.0
    for r1_index in soln_set:
        for r2_index in soln_set:
            if r1_index >= r2_index:
                continue
            if list_rules[r1_index].class_label == list_rules[r2_index].class_label:
                sum_overlap_intraclass += len(overlap(list_rules[r1_index], list_rules[r2_index],df))
    f2 = df.shape[0] * len(list_rules) * len(list_rules) - sum_overlap_intraclass
    f.append(f2)
    
    # f3 term - interclass overlap
    sum_overlap_interclass = 0.0
    for r1_index in soln_set:
        for r2_index in soln_set:
            if r1_index >= r2_index:
                continue
            if list_rules[r1_index].class_label != list_rules[r2_index].class_label:
                sum_overlap_interclass += len(overlap(list_rules[r1_index], list_rules[r2_index],df))
    f3 = df.shape[0] * len(list_rules) * len(list_rules) - sum_overlap_interclass
    f.append(f3)
    
    # f4 term - coverage of all classes
    classes_covered = set() # set
    for index in soln_set:
        classes_covered.add(list_rules[index].class_label)
    f4 = len(classes_covered)
    f.append(f4)
    
    # f5 term - accuracy
    sum_incorrect_cover = 0.0
    for index in soln_set:
        sum_incorrect_cover += len(list_rules[index].get_incorrect_cover(df,Y))
    f5 = df.shape[0] * len(list_rules) - sum_incorrect_cover
    f.append(f5)
    
    #f6 term - cover correctly with at least one rule
    atleast_once_correctly_covered = set()
    for index in soln_set:
        correct_cover, full_cover = list_rules[index].get_correct_cover(df,Y)
        atleast_once_correctly_covered = atleast_once_correctly_covered.union(set(correct_cover))
    f6 = len(atleast_once_correctly_covered)
    f.append(f6)
    
    obj_val = 0.0
    for i in range(7):
        obj_val += f[i] * lambda_array[i]
    
    #print(f)
    return obj_val


# deterministic local search algorithm which returns a solution set as well as the corresponding objective value
def deterministic_local_search(list_rules, df, Y, lambda_array, epsilon):
    # step by step implementation of deterministic local search algorithm in the 
    # FOCS paper: https://people.csail.mit.edu/mirrokni/focs07.pdf (page 4-5)
    
    #initialize soln_set
    soln_set = set()
    n = len(list_rules)
    
    # step 1: find out the element with maximum objective function value and initialize soln set with it
    each_obj_val = []
    for ind in range(len(list_rules)):
        each_obj_val.append(func_evaluation(set([ind]), list_rules, df, Y, lambda_array))
        
    best_element = np.argmax(each_obj_val)
    soln_set.add(best_element)
    S_func_val = each_obj_val[best_element]
    
    restart_step2 = False
    
    # step 2: if there exists an element which is good, add it to soln set and repeat
    while True:
        
        each_obj_val = []
        
        for ind in set(range(len(list_rules))) - soln_set:
            func_val = func_evaluation(soln_set.union(set([ind])), list_rules, df, Y, lambda_array)
            
            if func_val > (1.0 + epsilon/(n*n)) * S_func_val:
                soln_set.add(ind)
                print("Adding rule "+str(ind))
                S_func_val = func_val
                restart_step2 = True
                break
                
        if restart_step2:
            restart_step2 = False
            continue
            
        for ind in soln_set:
            func_val = func_evaluation(soln_set - set([ind]), list_rules, df, Y, lambda_array)
            
            if func_val > (1.0 + epsilon/(n*n)) * S_func_val:
                soln_set.remove(ind)
                print("Removing rule "+str(ind))
                S_func_val = func_val
                restart_step2 = True
                break
        
        if restart_step2:
            restart_step2 = False
            continue
        
        s1 = func_evaluation(soln_set, list_rules, df, Y, lambda_array)
        s2 = func_evaluation(set(range(len(list_rules))) - soln_set, list_rules, df, Y, lambda_array)
        
        print(s1)
        print(s2)
        
        if s1 >= s2:
            return soln_set, s1
        else: 
            return set(range(len(list_rules))) - soln_set, s2
            


# input data and function calls 
df = pd.read_csv('titanic_train.tab',' ', header=None, names=['Passenger_Cat', 'Age_Cat', 'Gender'])
df1 = pd.read_csv('titanic_train.Y', ' ', header=None, names=['Died', 'Survived'])
Y = list(df1['Died'].values)

itemsets = run_apriori(df, 0.1)
list_of_rules = createrules(itemsets, list(set(Y)))
print("----------------------")
for r in list_of_rules:
    r.print_rule()

lambda_array = [0.5]*7     # use separate hyperparamter search routine
epsilon = 0.05
soln_set, obj_val = deterministic_local_search(list_of_rules, df, Y, lambda_array, epsilon)
print(soln_set)
print(obj_val)

