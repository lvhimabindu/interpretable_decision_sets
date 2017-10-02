
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import math
from apyori import apriori


# In[ ]:

# itemset (each rule predicate --> class label)
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


# In[ ]:

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


# In[ ]:

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


# In[ ]:

def max_rule_length(list_rules):
    len_arr = []
    for r in list_rules:
        len_arr.append(r.get_length())
    return max(len_arr)


# In[ ]:

def overlap(r1, r2, df):
    return sorted(list(set(r1.get_cover(df)).intersection(set(r2.get_cover(df)))))


# In[ ]:

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


# In[ ]:

def sample_random_set(soln_set, delta, len_list_rules):
    all_rule_indexes = set(range(len_list_rules))
    return_set = set()
    
    # sample in-set elements with prob. (delta + 1)/2
    p = (delta + 1.0)/2
    for item in soln_set:
        random_val = np.random.uniform()
        if random_val <= p:
            return_set.add(item)
    
    # sample out-set elements with prob. (1 - delta)/2
    p_prime = (1.0 - delta)/2
    for item in (all_rule_indexes - soln_set):
        random_val = np.random.uniform()
        if random_val <= p_prime:
            return_set.add(item)
    
    #print(soln_set)
    #print(all_rule_indexes - soln_set)
    return return_set


# In[ ]:

def estimate_omega_for_element(soln_set, delta, rule_x_index, list_rules, df, Y, lambda_array, error_threshold):
    #assumes rule_x_index is not in soln_set 
    
    Exp1_func_vals = []
    
    Exp2_func_vals = []
    
    while(True):
        
        # first expectation term (include x)
        for i in range(10):
            temp_soln_set = sample_random_set(soln_set, delta, len(list_rules))
            temp_soln_set.add(rule_x_index)
            Exp1_func_vals.append(func_evaluation(temp_soln_set, list_rules, df, Y, lambda_array))
        
        # second expectation term (exclude x)
        for j in range(10):
            temp_soln_set = sample_random_set(soln_set, delta, len(list_rules))
            if rule_x_index in temp_soln_set:
                temp_soln_set.remove(rule_x_index)
            Exp2_func_vals.append(func_evaluation(temp_soln_set, list_rules, df, Y, lambda_array))
    
        # compute standard error of mean difference
        variance_Exp1 = np.var(Exp1_func_vals, dtype=np.float64)
        variance_Exp2 = np.var(Exp2_func_vals, dtype=np.float64)
        std_err = math.sqrt(variance_Exp1/len(Exp1_func_vals) + variance_Exp2/len(Exp2_func_vals))
        print("Standard Error "+str(std_err))
        
        if std_err <= error_threshold:
            break
            
    return np.mean(Exp1_func_vals) - np.mean(Exp2_func_vals)


# In[ ]:

def compute_OPT(list_rules, df, Y, lambda_array):
    opt_set = set()
    for i in range(len(list_rules)):
        r_val = np.random.uniform()
        if r_val <= 0.5:
            opt_set.add(i)
    return func_evaluation(opt_set, list_rules, df, Y, lambda_array)


# In[ ]:

def smooth_local_search(list_rules, df, Y, lambda_array, delta, delta_prime):
    # step by step implementation of smooth local search algorithm in the 
    # FOCS paper: https://people.csail.mit.edu/mirrokni/focs07.pdf (page 6)
    
    # step 1: set the value n and OPT; initialize soln_set to empty
    n = len(list_rules)
    OPT = compute_OPT(list_rules, df, Y, lambda_array)
    print("2/n*n OPT value is "+str(2.0/(n*n)*OPT))
    
    soln_set = set()
    
    restart_omega_computations = False
    
    while(True):
    
        # step 2 & 3: for each element estimate omega within certain error_threshold; if estimated omega > 2/n^2 * OPT, then add 
        # the corresponding rule to soln set and recompute omega estimates again
        omega_estimates = []
        for rule_x_index in range(n):
                
            print("Estimating omega for rule "+str(rule_x_index))
            omega_est = estimate_omega_for_element(soln_set, delta, rule_x_index, list_rules, df, Y, lambda_array, 1.0/(n*n) * OPT)
            omega_estimates.append(omega_est)
            print("Omega estimate is "+str(omega_est))
            
             if rule_x_index in soln_set:
                continue
            
            if omega_est > 2.0/(n*n) * OPT:
                # add this element to solution set and recompute omegas
                soln_set.add(rule_x_index)
                restart_omega_computations = True
                print("-----------------------")
                print("Adding to the solution set rule "+str(rule_x_index))
                print("-----------------------")
                break    
        
        if restart_omega_computations: 
            restart_omega_computations = False
            continue
            
        # reaching this point of code means there is nothing more to add to the solution set, but we can remove elements
        for rule_ind in soln_set:
            if omega_estimates[rule_ind] < -2.0/(n*n) * OPT:
                soln_set.remove(rule_ind)
                restart_omega_computations = True
                
                print("Removing from the solution set rule "+str(rule_ind))
                break
                
        if restart_omega_computations: 
            restart_omega_computations = False
            continue
            
        # reaching here means there is no element to add or remove from the solution set
        return sample_random_set(soln_set, delta_prime, n)


# In[ ]:

df = pd.read_csv('titanic_train.tab',' ', header=None, names=['Passenger_Cat', 'Age_Cat', 'Gender'])
df1 = pd.read_csv('titanic_train.Y', ' ', header=None, names=['Died', 'Survived'])
Y = list(df1['Died'].values)
df1.head()


# In[ ]:

itemsets = run_apriori(df, 0.2)
list_of_rules = createrules(itemsets, list(set(Y)))
print("----------------------")
for r in list_of_rules:
    r.print_rule()

lambda_array = [1.0]*7     # use separate hyperparamter search routine
s1 = smooth_local_search(list_of_rules, df, Y, lambda_array, 0.33, 0.33)
s2 = smooth_local_search(list_of_rules, df, Y, lambda_array, 0.33, -1.0)
f1 = func_evaluation(s1, list_of_rules, df, Y, lambda_array)
f2 = func_evaluation(s2, list_of_rules, df, Y, lambda_array)
if f1 > f2:
    print("The Solution Set is: "+str(s1))
else:
    print("The Solution Set is: "+str(s2))

