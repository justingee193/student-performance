import pandas as pd
import numpy as np
import random as rand
from scipy.stats import chi2_contingency, normaltest, levene, wilcoxon

class Statistics(object):
    def __init__(self, var1, var2):
            self.var1 = var1
            self.var2 = var2
            
    def corr(self, decimal=4):
        
        return round(np.corrcoef(x=np.array([self.var1, self.var2]))[0,1], decimal)
    
    def chi2_stat(self):
        
        obs = pd.crosstab(self.var1, columns=self.var2)
        chi2, p, dof, expected = chi2_contingency(obs)
        
        return chi2, p

class HypothesisTest(object):
    def __init__(self, var1, var2):
        self.var1 = var1
        self.var2 = var2
        
    def normality_test(self):
        
        test_stat1, p1 = normaltest(self.var1)
        test_stat2, p2 = normaltest(self.var2)
        
        return test_stat1, test_stat2, p1, p2
    
    def equal_variance(self):
        
        test_stat, p = levene(self.var1, self.var2)
        
        return test_stat, p
    
    def para_t_test(self):
        
        seed = rand.seed(444)

        X = self.var1
        Y = rand.sample(list(self.var2), 181)
        
        t_stat, p = wilcoxon(x=X, y=Y)
        
        return t_stat, p
                
class Tables(object):
    def __init__(self, var1, var2, indices):
        self.var1 = var1
        self.var2 = var2
        self.indices = indices
    
    def freq(self, var):
        
        freq = pd.crosstab(index=var, columns="count")
        freq.index = self.indices
        
        print(freq)
    
    def joint_dist(self):
        
        joint_dist = pd.crosstab(index=self.var1, columns=self.var2)
        joint_dist.index = self.indices
        
        print(joint_dist)
    
    def conditional_dist(self):
        
        conditional_dist = pd.crosstab(self.var1, self.var2).apply(lambda r : r/r.sum(), axis=0)
        conditional_dist.index = self.indices
        
        print(conditional_dist)
