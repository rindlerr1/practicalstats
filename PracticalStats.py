#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Basic Statistical Calculations
All functions can work with a list, an array, or the column of a dataframe

"""

"""
Hey Lisa
"""

def MeanAD(x):    
    import numpy as np
    x = np.array(x)
    
    mad = np.mean(abs(x - np.mean(x)))
    return mad

    
    
def MedianAD(x):
    import numpy as np
    x = np.array(x)
    
    median_x = np.median(abs((x) - (np.median(x))))
    return median_x


    
def Variance(x):
    import numpy as np
    x = np.array(x)
    
    var = (sum(x**2) - (float(sum(x)**2))/len(x))/(len(x) - 1)
    return var

    
    
def SD(x):
    import numpy as np
    x = np.array(x)
    
    var = (sum(x**2)-(float(sum(x)**2))/  len(x))/(len(x) - 1)
    sd = np.sqrt(var)
    return sd
   

    
def Ttest(x,y,Paired):
    import numpy as np
    x = np.array(x)
    y = np.array(y)    
    
    alpha = [12.706,4.303,3.182,2.776,2.571,2.447,2.365,2.306,2.262,2.228,2.201,2.179,2.16,2.145,2.131,2.12,2.11,2.101,2.093,2.086,2.08,2.074,2.069,2.064,2.06,2.056,2.052,2.048,2.045,2.042,2,1.98,1.96]   
    df = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,60,120,121]
    
    if Paired == True:
        diff = (x - y)
        t = (float(sum(diff))/len(diff))/np.sqrt(((sum(diff**2))-(float((sum(diff)**2))/len(diff)))/((len(diff) - 1) * len(diff)))
        type_paired = "Paired"
        for i in range(0, 33):
            if (len(diff) - 1) == df[i]:
                ttable = alpha[i]
        if (len(diff)- 1) >= 31 and (len(diff) - 1) <= 60:
            ttable = alpha[30]
        elif(len(diff) - 1) >= 61 and (len(diff) - 1) <= 120:
            ttable = alpha[31]
        elif(len(diff) - 1) >= 121:
            ttable = alpha[32] 
    else:
        t = (float(np.mean(x) - np.mean(y)))/np.sqrt(((float((sum(x**2) - (float(sum(x)**2)/len(x))) + (sum(y**2) - (float(sum(y)**2)/len(y)))))/((len(x)+len(y)) - 2)) * ((float(1)/len(x)) + (float(1)/len(y))))
        type_paired = "Independent"
        for i in range(0, 33):
            if ((len(x) - 1) + (len(y) - 1)) == df[i]:
                ttable = alpha[i]
        if ((len(x) - 1) + (len(y) - 1)) >= 31 and ((len(x) - 1) + (len(y) - 1)) <= 60:
            ttable = alpha[30]
        elif((len(x) - 1) + (len(y) - 1)) >= 61 and ((len(x) - 1) + (len(y) - 1)) <= 120:
            ttable = alpha[31]
        elif((len(x) - 1) + (len(y) - 1)) >= 121:
            ttable = alpha[32] 

    if abs(t) > ttable:
        result = 'Reject Null'
    else:
        result = 'Accept Null'
       
    values = {'T- Value' : t, "Table Value" : ttable, "Result" : result, "Test (Paired/Independent)" : type_paired}
    
    return values

"""
This is my random chance model.
It is an omnibus test for variability among groups.

Requires 
-'groups' a dictionary of list of arrays or serieses that are to be compared

"""

def rc_model(groups):
    import numpy as np
    import random   
    import PracticalStats as ps
    
    groups_l =[]
    for k,v in groups.items():
        groups_l.append(v)
    
    #creates a single source for all the values to be compared    
    bucket = []
    for i in range(0, len(groups_l)):
        for q in range(0, len(groups_l[i])):
            bucket.append(groups_l[i][q])
    
    #create a list of means and variances that correspond to the different groups to be compared
    means = []
    for i in range(0, len(groups_l)):
        means.append(np.mean(groups_l[i]))
    grand_variance = ps.Variance(means)    

    """
    actual permutation part
    for whatever number of permutations assigned....
    -shuffle the total number of observations
    -create new groups of data that are the same length as the original groups
    -get the means for those groups
    -assess the variance of those group means
    -the p value is the proportion of times the new variance exceeded the old variance
    -the permutation test is run 20 times and the mean p-value is used to assess whether or not to reject the null
         -this is done because due to the randomness of the model there is usually a range of p-values returned
    """
    def permutation():
        test_means = []
        p = 0
        total = 0
        for i in range(0, 500):
            random.shuffle(bucket)  
            a = 0
            for q in range(0, len(groups_l)):
                test_means.append(np.mean(bucket[a:(a+len(groups_l[q]))]))
                a += len(groups_l[q])
            variance = ps.Variance(test_means)
            if variance >= grand_variance:
                p += 1
            test_means = []
            total += 1
        p_value = p/total
        return p_value
    
    p_values = []
    for i in range(0, 20):
        p_values.append(permutation())
    
    p_value = np.mean(p_values)
    
    if p_value <= 0.05:
        result = 'Reject Null...Results are Different enough to not be random.'
    else:
        result = 'Accept Null...Results could be random.'
    
    end = {'Alpha' : 'With alpha set to .05', 'Result' : result,
           'Explaination':'Probability that the difference in the data are just random.',
           'pseudo p-value' : p_value}
    
    from prettytable import PrettyTable
    
    pt = PrettyTable(field_names = ["Sections", "Random Chance Model"])
    for key, val in end.items():
        pt.add_row([key, val])
    print(pt)
    return p_value, p_values 






"""

Effect Size: How different are the samples?

"""

def effect(groups):
    import pandas as pd
    import numpy as np
    from PracticalStats import SD
    
    #create several lists to include relevant names and values
    a = []
    b = []
    c = []
    d = []       
    for k,v in groups.items():   
        for i,q in groups.items():
            a.append(i)
            b.append(q)
            c.append(k)
            d.append(v)
        
    #remove the combinations that are the same.  Can't measure effect size against yourself    
    popit = 0
    for i in range(0, len(a)):
        if i == popit:
            popit = i + len(groups)
            a.pop(i)
            b.pop(i)
            c.pop(i)
            d.pop(i)
        
    #calculate the effect size for every combination
    effect_sizes = []
    for i in range(0, len(a)):
       effect_sizes.append((np.mean(np.asarray(b[i])) - np.mean(np.asarray(d[i]))) / SD(pd.concat([pd.Series(b[i]),pd.Series(d[i])])))
    df = pd.DataFrame({'Experimental' : a, 'Control':c, 'Effect Size' : effect_sizes})
    
    
    #visualize the data in tables
    performance = df.groupby(['Experimental', 'Control'])
    print('Scores go from left to right')
    print(performance.mean().unstack())
    print('   ')
   
    avg_performance = df.groupby('Experimental')
    print('Average Performance')
    print(avg_performance.mean().sort_values('Effect Size', ascending = False))
   
    return df
    

"""

Multi-Arm Bandit Algorithm

"""

def basic_bandit(groups, epsilon):
    import random
    import numpy as np
    
    #names creates a list to choose from at random if necessitated by the algorithm
    #creates a mean value for the scores of the different groups
    #        -----This is very important, the mean value of the groups are used to rank performance-------
    names = []   
    for k, v in groups.items():
        groups[k] = np.mean(v)
        names.append(k)
    
    #creates the number to compare to epsilon              
    x = (random.randint(0,100))/100
    
    #creates a 'winner' of the resources
    if x >= epsilon:
        winner = max(groups, key = groups.get)
    else:
        winner = random.choice(names)
        
    print(winner)
    return winner 

"""

ols model

must be organized as follows using x and y as keys
dictionary{'x':x, 'y':y}


"""

def ols(train):
    import numpy as np
    
    x = train['x']
    y = train['y']
    
    coeff = np.sum((y - np.mean(y))*(x- np.mean(x)))/np.sum((x - np.mean(x))**2)

    intercept = (np.mean(y)) - (coeff*np.mean(x))
    
    return coeff, intercept
    

"""

prediction function for OLS
necessaries: data = list of values to be predicted against, coeff, intercept


"""
def ols_pred(data, coeff, intercept):
        
    pred_vals = []
    for i in range(0, len(data)):
        pred_vals.append(intercept + (coeff*data[i]))
    
    return pred_vals

