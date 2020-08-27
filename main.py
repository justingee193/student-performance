import analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    df = pd.read_csv("student-mat.csv", sep=";")
    
    print(df.head())
    
    print(df.info())
    
    print(df.describe())
    
    # Does alcohol consumption affect student performance?
    
    df['Dalc'] = df['Dalc'] + df['Walc']
    
    plt.figure()
    alcohol_count_plot = sns.catplot('Dalc', data=df, kind='count')
    
    avg = round(sum(df.G3) / len(df), 2)
    print("Student grade average is {}. Roughly a {}.".format(avg, avg*5))
    
    df['Average'] = ['above average' if i > avg else 'below average' for i in df.G3]
    plt.figure()
    average_grade_plot = sns.swarmplot('Dalc', 'G3', hue = 'Average', data=df, palette={'above average':'lime', 'below average': 'red'})
    
    q1 = analysis.Statistics(var1=df.G3, var2=df.Dalc)
    cor = q1.corr(decimal=4)
    print("The correlation coefficient between final grade score and drinking intensity is {}.".format(cor))
    
    # What factors impact student study time?
    
    ## Activities
    
    indices = ["< 2 hrs", "2 - 5 hrs", "5 - 10 hrs", "> 10 hrs"]
    
    activites_tables = analysis.Tables(var1=df["studytime"], var2=df["activities"], indices=indices)
    
    activites_tables.freq(var=df["studytime"])
    
    activites_tables.joint_dist()
    
    plt.figure()
    activities_cat_plot = sns.catplot('studytime', col="activities", data=df, kind='count')
    
    activites_tables.conditional_dist()
    
    activities_stat = analysis.Statistics(var1=df["studytime"], var2=df["activities"])
    chi2, p = activities_stat.chi2_stat()
    print('The chi2 statistic is {}, with a p-value of {} and significance level of 0.05'.format(round(chi2, 3), round(p, 3)))
    
    ## Relationship Status
    
    relationship_tables = analysis.Tables(var1=df['studytime'], var2=df['romantic'], indices=indices)
    
    relationship_tables.joint_dist()
    
    relationship_tables.conditional_dist()
    
    plt.figure()
    relationship_cat_plot = sns.catplot('studytime', col="romantic", data=df, kind='count')
    
    relationship_stat = analysis.Statistics(var1=df['studytime'], var2=df['romantic'])
    chi2, p = relationship_stat.chi2_stat()
    print('The chi2 statistic is {}, with a p-value of {} and significance level of 0.05'.format(round(chi2, 3), round(p, 3)))
    
    # Do extra paid classes help to improve student performance?
    
    print(pd.crosstab(df['paid'], columns='count'))
    
    yes = df[df['paid']=='yes']
    no = df[df['paid']=='no']
    
    yes_avg = yes['G3'].mean()
    no_avg = no['G3'].mean()
    print("Students that took extra classes got an average of {}, while those that did not got an average of {}".format(round(yes_avg, 3), round(no_avg, 3)))

    yes_std = yes['G3'].std()
    no_std = no['G3'].std()
    print("Students that took extra classes have a standard deviation of {}, while those that did not have a standard deviation of {}".format(round(yes_std, 3), round(no_std, 3)))
        
    plt.figure()
    boxplot = sns.boxplot(x=df['paid'], y=df['G3'])
    swarmplot = sns.swarmplot(x=df['paid'], y=df['G3'], color='.25')
    
    plt.figure()
    bins = np.arange(0, 20)
    grades = sns.FacetGrid(df, col="paid")
    grades.map(plt.hist, "G3", bins=bins, color='purple')
    
    ## Hypothesis Testing
    
    hypothesis_test = analysis.HypothesisTest(var1=df[df['paid']=='yes']['G3'], var2=df[df['paid']=='no']['G3'])
    
    ### Normality Test
    
    test_stat1, test_stat2, p1, p2 = hypothesis_test.normality_test()
    print("The test statistic is {}, with a p-value of {}".format(round(test_stat1, 3), round(p1, 5)))
    print("The test statistic is {}, with a p-value of {}".format(round(test_stat2, 3), round(p2, 5)))
    
    ### Equal Variance Test
    
    test_stat, p = hypothesis_test.equal_variance()
    print("The test statistic is {}, with a p-value of {}".format(round(test_stat, 3), round(p, 5)))
    
    ### Difference in Median Test
    
    t_stat, p = hypothesis_test.para_t_test()
    print("The test statistic is {}, with a p-value of {}.".format(round(t_stat, 3), round(p, 3)))
    
if __name__ == "__main__":
    main()
    
    
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    