#-*-coding:utf8-*-
from pylab import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ranksums
from numpy import argsort
from xlwt import Workbook



def importation_valeurs(nom,a):
    X = []
    for fichier in [nom]:  
        fic = open(fichier).read().split('\n')[1:-1] # parce que sinon il ajoute une ligne vide.
        for ligne in fic:
            X.append([eval(u.replace(',', ".")) for u in ligne.split('\t')[a:]])
    return X

def importation_info(nom,a):
    X = []
    for fichier in [nom]:  
        fic = open(fichier).read().split('\n')[1:-1]
        for ligne in fic:
            X.append(ligne.split('\t')[:a])
    return X

def importation_entete(nom,a):
    return open(nom).read().split('\n')[0].split('\t')[:a]



def ROC_ligne(x,m,n,cat0,cat1,y):
    '''
    ROC_ligne

    Perform a ROC analysis for a given quantitative variable and a given binary variable.

    @param x: Vector that contains the value of the subjects for the given quantitative variable.
    @param m: The number of subjects from the first category.
    @param n: The number of subjects from the second category.
    @param cat0 : Name of the first category.
    @param cat1: Name of the second category.
    @param y: Vector that contains the value of the subjects for the given binary variable. 
    y is defined as follows: y[i] = 0 if the subject i is in the first category
    and y[i] = 1 if the subject i is in the second category.
    @return "threshold_inf" and "threshold_sup": The two consecutive values of the quantitative variable such as any cut-off taken in the interval
    [threshold_inf" , "threshold_sup] maximize the sum between sensitivity and specificity.
    @return threshold: Mean between threshold_inf" and "threshold_sup.  
    @return Preference: The category considered as positive in the analysis. In other terms, the category that is 
    mostly above the threshold. 
    @return True positive rate: The proportion of elements of the postitive category that are above the threshold.
    @return True negative rate: The proportion of elements of the negative category that are below the threshold.
    @return Sum true positive and true negative: Sum of true positive rate and true negative rate.
    @return Well classified subjects: well classified subjects
    @return AUC: Area under the curve.
    @return delta_norm: Difference between threshold_inf and threshold_sup after normalization of x.  
    '''
    Mean = np.mean(x)
    Std = np.std(x,ddof=1)
    Norm = np.zeros(len(x))
    for i in range(len(x)):
        if Std <> 0:
            Norm[i] = float(x[i]-Mean)/Std
        if Std == 0:
            Norm[i] = 0
    
    vrai_pos = [n] # The proportion of elements of the positive category that are above the threshold.
    vrai_neg = [0] # The proportion of elements of the negative category that are below the threshold.
    faux_pos = [m] # The proportion of elements of the negative category that are above the threshold.
    faux_neg = [0] # The proportion of elements of the positive category that are below the threshold.
    indice = argsort(x)
    
    for i in range(m+n):
        if y[indice[i]] == 0: 
            vrai_pos.append(vrai_pos[i])
            vrai_neg.append(vrai_neg[i]+1)
            faux_pos.append(faux_pos[i]-1)
            faux_neg.append(faux_neg[i])
        
        if y[indice[i]] == 1: 
            vrai_pos.append(vrai_pos[i]-1)
            vrai_neg.append(vrai_neg[i])
            faux_pos.append(faux_pos[i])
            faux_neg.append(faux_neg[i]+1)

        if i > 0:  
            k = 1
            while x[indice[i-k]] == x[indice[i]] and (i-k) >= 0:
                vrai_pos[i+1-k] = vrai_pos[i+1]
                vrai_neg[i+1-k] = vrai_neg[i+1]
                faux_pos[i+1-k] = faux_pos[i+1]
                faux_neg[i+1-k] = faux_neg[i+1]
                k+=1

    AUC = 0.
    for i in range (m+n):
        AUC = AUC + (faux_pos[i]-faux_pos[i+1])*(vrai_pos[i+1]+vrai_pos[i])
    AUC = float(AUC) /(m*n*2)
    tot = []
    for i in range(len(vrai_pos)): 
        vrai_pos[i] = float(vrai_pos[i])/n 
        vrai_neg[i] = float(vrai_neg[i])/m
        faux_pos[i] = float(faux_pos[i])/m
        faux_neg[i] = float(faux_neg[i])/n
        tot.append(vrai_pos[i] + vrai_neg[i]) 

    tot_seuil = max(tot)
    Min = min(tot)
    
    if AUC >= 0.5 and tot_seuil <> 1:
        pref = cat1
        k = argmax(tot)
        
        l = k+1
        while (tot[k] == tot[l]) and (l+1 < len(tot)):
            l+=1
        
        seuil_inf = x[indice[k-1]]
        seuil_inf_norm = Norm[indice[k-1]]
        seuil_sup = x[indice[l-1]]
        seuil_sup_norm = Norm[indice[l-1]]
        delta_norm = seuil_sup_norm - seuil_inf_norm
        seuil = float(seuil_sup + seuil_inf)/2  ######  Erreur ici !!!!!!!!!!!!!! 

        vrai_pos_seuil = vrai_pos[k]
        vrai_neg_seuil = vrai_neg[k]
        bien_classe = vrai_neg[k] * m + vrai_pos[k] * n
        
    if AUC < 0.5:
        pref = cat0
        AUC = 1 - AUC
        tot_seuil = 1 - min(tot)
        k = argmin(tot)
        
        
        l = k+1
        while (tot[k] == tot[l]) and (l+1 < len(tot)):
            l+=1
    
        seuil_inf = x[indice[k-1]]
        seuil_inf_norm = Norm[indice[k-1]]
        seuil_sup = x[indice[l-1]]
        seuil_sup_norm = Norm[indice[l-1]]
        delta_norm = seuil_sup_norm - seuil_inf_norm
        seuil = float(seuil_sup + seuil_inf)/2

        for i in range(m+n+1):
            b = vrai_pos[i]
            vrai_pos[i] = faux_pos[i]
            faux_pos[i] = b
            c = vrai_neg[i]
            vrai_neg[i] = faux_neg[i]
            faux_neg[i] = c

        vrai_pos_seuil = vrai_pos[k]
        vrai_neg_seuil = vrai_neg[k]
        bien_classe = vrai_neg[k] * n + vrai_pos[k] * m
        tot_seuil = vrai_neg_seuil + vrai_pos_seuil         

    if AUC == 0.5 and tot_seuil == 1 and Min == 1:
        pref = "neutre"
        seuil_inf = seuil_sup = seuil = delta_norm = vrai_pos_seuil = 0
        vrai_neg_seuil = 1
        bien_classe = n
    return [vrai_pos,vrai_neg,faux_pos,tot,AUC,delta_norm,seuil,pref,vrai_pos_seuil,vrai_neg_seuil,tot_seuil,bien_classe,seuil_inf,seuil_sup]




def non_nul(X,y):
    res = []
    for i in range (len(X)):
        res_i_0 = 0
        res_i_1 = 0
        x = X[i]
        for j in range(len(x)):
            if y[j] == 0 and x[j] <> 0:
                res_i_0 += 1
            if y[j] == 1 and x[j] <> 0:
                res_i_1 += 1
        res.append([res_i_0,res_i_1])
    return(res)


def ROC_base(X,I,m,n,cat0,cat1,y):
    if y == []:
        y =  [0] * m + [1] * n
    y2 = []
    for j in range(len(y)):
        if y[j] == 0:
            y2.append(0)
        if y[j] == 1:
            y2.append(1)
    res = []
    for i in range(len(X)):
        x = X[i]
        x2 = []
        x2_cat0 = []
        x2_cat1 = []
        for j in range(len(x)):
            if y[j] == 0:
                x2.append(x[j])
                x2_cat0.append(x[j])
            if y[j] == 1:
                x2.append(x[j])
                x2_cat1.append(x[j])
        Wilcoxon = ranksums(x2_cat0,x2_cat1) #mannwhitneyu(x, y, use_continuity=True) 
        res.append(ROC_ligne(x2,m,n,cat0,cat1,y=y2)[4:] + I[i] + [Wilcoxon[1]] + non_nul(X,y)[i])
    res2 = sorted (res, reverse=True)
    return res2



def draw_curve(nom_base, a, m, n, nom_id, titre, colonne_id = 0, y = [] ):
    '''
    Draw a ROC curve.
    '''
    X = importation_valeurs(nom_base,a)
    I = importation_info(nom_base,a)
    N = len(I)
    i = -1
    item = []
    while (i < N) and item <> nom_id:
        i += 1
        item = I[i][colonne_id]
    if i == N:
        print "Probleme with fonction 'draw_curve': '", nom_id, "' not found in collumn ", colonne_id, "of database '", nom_base, "'."
    else:
        x = X[i]
        if y == []:
            y =  [0] * m + [1] * n
        print n
        print m
        print x
        res = ROC_ligne(x,m,n,'cat0','cat1',y)
        vrai_pos, faux_pos = res[0], res[2]

        fig = plt.figure()
        fig.patch.set_facecolor('w')
        plt.plot(faux_pos, vrai_pos,markeredgecolor='w',markerfacecolor='w')
        plt.plot([0,1],[0,1],markeredgecolor='w',markerfacecolor='w')
        plt.xlim(-0.1,1.1)
        plt.ylim(-0.1,1.1)
        plt.xlabel('False positive rate',fontsize = 14)
        plt.ylabel('True positive rate',fontsize = 14)
        plt.title(titre,fontsize = 14)
        show()


def rast(name_data,a,m,n,cat0,cat1,y=[], name_outup = 'ROC_table.xls'):
    '''
    rast:ROC analysis sorted table

    Import your data. Compute a ROC analysis for each quantitative variable of the database. Meet all these results in the same table. 
    Sort this table to show the most discriminating variables and export this table in excel.  

    @param nane_data : The name of your database (input).
    @param a : The number of descriptives collumns. (see exemple)
    @param m : The number of subjects from the first categorie.
    @param n : The number of subjects from the second categorie.
    @param cat0 : Name of the first categorie.
    @param cat1 : Name of the second categorie.
    @param y: Optional vector indicating which column corresponds to category 0 and which column corresponds to category 1. 
    By default, if no value is provided for y, it is assumed that the columns between (a + 1) and (a + m) correspond to category 0 
    and the n following columns correspond to category 1.
    @param name_output : Optional parameter. The name of the excel table returned. By default, name_output = "ROC_table.xls"
    @param nb_of_nonzero : Optional parameter. Write nb_of_nonzero = TRUE if you want to get the number of nonzero subjets 
    in each categrorie for every quantitative variables.
    @return "threshold_inf" and "threshold_sup": The two consecutive values of the quantitative variable such as any cut-off taken in the interval
    [threshold_inf" , "threshold_sup] maximize the sum between sensitivity and specificity.
    @return threshold: Mean between threshold_inf" and "threshold_sup.  
    @return Preference: The category considered as positive in the analysis. In other terms, the category that is 
    mostly above the threshold. 
    @return True positive rate: The proportion of elements of the postitive category that are above the threshold.
 @return True negative rate: The proportion of elements of the negative category that are below the threshold.
 @return Sum true positive and true negative: Sum of true positive rate and true negative rate.
 @return Well classified subjects: well classified subjects
 @return AUC: Area under the curve.
 @return delta_norm: Difference between threshold_inf and threshold_sup after normalization of x.   
 @return Wilcoxon p-value: Wilcoxon p-value
    '''
    X = importation_valeurs(name_data,a)
    I = importation_info(name_data,a)
    res0 = ROC_base(X,I,m,n,cat0,cat1,y=y)
    res0 = np.matrix(res0)
    res1 = res0.take((range(10,10+a)),axis = 1)
    res2 = res0.take((range(2,8)),axis = 1)
    res3 = res0.take((range(0,2)),axis = 1)
    res4 = res0.take((range(8,10)),axis = 1)
    res5 = res0.take((range(10+a,13+a)),axis = 1)
    res = np.concatenate((res1,res2,res3,res4,res5), axis=1)
    E1 = importation_entete(name_data,a)
    E2 = ['threshold','Preference','True positive rate','True negative rate','Sum true positive and true negative','well classified subjects','AUC',
          'Delta_norm','threshold inf','threshold sup','Wilcoxon p-value', 'Number of nonzero subjects in ' + cat0, 'Number of nonzero subjects in ' + cat1]
    
    book = Workbook()
    feuil1 = book.add_sheet('feuille 1')
    ligne = feuil1.row(0)
    for j in range(a):
        ligne.write(j,E1[j])
    for j in range(13):
        ligne.write(a+j,E2[j])
    for i in range(len(res)):
        ligne = feuil1.row(i+1)
        for j in range(res.shape[1]):
            if j >= a and set(list(res[i,j])).issubset(set(list('0123456789.'))):
                ligne.write(j,eval(res[i,j]))
            else:
                ligne.write(j,res[i,j])
    book.save(name_outup)
    return(res)
    
# Example 1:
# In this first example, the database used as input is the followig (without #).
#
# Id	caractere	setosa1	setosa2	setosa3	setosa4	setosa5	setosa6	setosa7	setosa8	versicolor1	versicolor2	versicolor3	versicolor4	versicolor5	versicolor6	versicolor7	versicolor8
# 1	Sepal_Length	5.1	4.9	4.7	4.6	5	5.4	4.6	5	7	6.4	6.9	5.5	6.5	5.7	6.3	4.9
# 2	Sepal_Width	3.5	3	3.2	3.1	3.6	3.9	3.4	3.4	3.2	3.2	3.1	2.3	2.8	2.8	3.3	2.4
# 3	Petal_Length	1.4	1.4	1.3	1.5	1.4	1.7	1.4	1.5	4.7	4.5	4.9	4	4.6	4.5	4.7	3.3
# 4	Petal_Width	0.2	0.2	0.2	0.2	0.2	0.4	0.3	0.2	1.4	1.5	1.5	1.3	1.5	1.3	1.6	1
#
# This database, named "Exeample_1" is a part of the classical "Iris" database.
# The columns are not correcty sorted here (every setosa then evey versicolor). Thus it is not necessary to defined an input "y".
#
rast('example1.txt',2,8,8,'setosa','versicolor')
#
# The output of the fonction is the table:
#
# Id	caractere	threshold	Preference	True positive rate	True negative rate	Sum true positive and true negative	well classified subjects	#AUC	Delta_norm	threshold inf	threshold sup	Wilcoxon p-value
# 3	Petal_Length	2.5	  versicolor	1	    1	2	    16	1	          1.02192446713797	1.7	3.3	0.000837046362390583
# 4	Petal_Width	  0.7	  versicolor	1	    1	2	    16	1	          0.983959199337784	0.4	1	  0.000649688280521811
# 1	Sepal_Length	5.45	versicolor	0.875	1	1.875	15	0.9296875	  0.120476827067682	5.4	5.5	0.00448535921842047
# 2	Sepal_Width	  3.35	setosa	    0.625	1	1.625	13	0.8515625	  0.240307791085116	3.3	3.4	0.0202088881490031
#
#  Warning: Never leave empty lines at the end of your dataset (python really do not like that).
#
# Example 2:
#
# In this exemple, the database used as input is the same that in example 1, exepted that the columns are not correcty sorted (every setosa then evey versicolor) as in example 1.
# This implies that the input "y" as to be defined here.
#
# Id	caractere	setosa1	setosa2	setosa3	setosa4	versicolor1	versicolor2	versicolor3	versicolor4	setosa5	setosa6	setosa7	setosa8	versicolor5	versicolor6	versicolor7	versicolor8
# 1	Sepal.Length	5,1	4,9	4,7	4,6	7	6,4	6,9	5,5	5	5,4	4,6	5	6,5	5,7	6,3	4,9
# 2	Sepal.Width	3,5	3	3,2	3,1	3,2	3,2	3,1	2,3	3,6	3,9	3,4	3,4	2,8	2,8	3,3	2,4
# 3	Petal.Length	1,4	1,4	1,3	1,5	4,7	4,5	4,9	4	1,4	1,7	1,4	1,5	4,6	4,5	4,7	3,3
# 4	Petal.Width	0,2	0,2	0,2	0,2	1,4	1,5	1,5	1,3	0,2	0,4	0,3	0,2	1,5	1,3	1,6	1
#

y=[1]*4+[0]*4+[1]*4+[0]*4
res = rast('example2.txt',2,8,8,'setosa','versicolor',y, name_outup = "ROC_table_2.xls")



