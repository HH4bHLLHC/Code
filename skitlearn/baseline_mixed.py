#!/usr/bin/env python
# Baseline example
### to use matplotlib in cmssw:
# python baseline.py -dbackend

import matplotlib
#matplotlib.use('PS')   # generate postscript output by default
import matplotlib.pyplot as plt
from matplotlib import cm as cm
import numpy as np
from numpy import logical_and
import pandas
import math
from sklearn.externals import joblib
#from sklearn.cross_validation import train_test_split , cross_val_score 
from sklearn.metrics import roc_curve, roc_auc_score 
from rep.estimators import TMVAClassifier , XGBoostClassifier 
from sklearn.ensemble import AdaBoostClassifier
from rep.metaml import  ClassifiersFactory
from copy import deepcopy
from rep.report.metrics import RocAuc
from rep.utils import train_test_split
#from sklearn.metrics import roc_curve, auc
#from sigopt_sklearn.search import SigOptSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import root_numpy
from root_numpy import root2array, rec2array, array2root
import time
import ROOT
from pandas import DataFrame, read_csv


typedata="Data"
#typedata="QCD"
outputCentral =typedata+'_BDT_01_12_16_30h'
text_file = open(outputCentral+'.txt', "w")
subset="baseline-"
BKG="plainQCD"
BKG="plainQCDmixed"
ext=".png"


"""
#print "Samples contains", len(signalData), "signal events "
TTData = pandas.DataFrame(root_numpy.root2array("../datasets/full-TT_noTrg-toBDT.root", treename = "treeout"))
TTData["target"] = 0
#####################################################################################################################
##################################################################################################################
signalData = pandas.DataFrame(root_numpy.root2array("../datasets/"+subset+"HHTo4B_SM-toBDT.root", treename = "treeout"))
signalData["target"] = 1
"""

#####################################################################################################################
signalData8 = pandas.DataFrame(root_numpy.root2array("../datasets/"+subset+"HHTo4B_BM8-toBDT.root", treename = "treeout"))
signalData8["target"] = 1
##################################################################################################################
signalData = pandas.DataFrame(root_numpy.root2array("../datasets/"+subset+"HHTo4B_SM-toBDT.root", treename = "treeout"))
signalData["target"] = 1
if typedata=="Data" :
  ##################################################################################################################
  Data1 = pandas.DataFrame(root_numpy.root2array("../datasets/baseline_frac20-Plain-BTagCSVRun2016B-v2-toBDT.root", treename = "treeout"))
  Data1["target"] = 0
  Data2 = pandas.DataFrame(root_numpy.root2array("../datasets/baseline_frac20-Plain-BTagCSVRun2016C-v2-toBDT.root", treename = "treeout"))
  Data2["target"] = 0
  Data3 = pandas.DataFrame(root_numpy.root2array("../datasets/baseline_frac20-Plain-BTagCSVRun2016D-v2-toBDT.root", treename = "treeout"))
  Data3["target"] = 0
  dataset20 = signalData.append(Data1, ignore_index = True) 
  dataset20 = dataset20.append(Data2, ignore_index = True)   
  dataset20 = dataset20.append(Data3, ignore_index = True) 
  ##################################################################################################################
  Data1 = pandas.DataFrame(root_numpy.root2array("../datasets/baseline-Plain-BTagCSVRun2016B-v2-toBDT.root", treename = "treeout"))
  Data1["target"] = 0
  Data2 = pandas.DataFrame(root_numpy.root2array("../datasets/baseline-Plain-BTagCSVRun2016C-v2-toBDT.root", treename = "treeout"))
  Data2["target"] = 0
  Data3 = pandas.DataFrame(root_numpy.root2array("../datasets/baseline-Plain-BTagCSVRun2016D-v2-toBDT.root", treename = "treeout"))
  Data3["target"] = 0
  dataset = signalData.append(Data1, ignore_index = True) 
  dataset = dataset.append(Data2, ignore_index = True)   
  dataset = dataset.append(Data3, ignore_index = True)   
  ###############################################################################
  Data1mix = pandas.DataFrame(root_numpy.root2array("../datasets/baseline-Mixed-BTagCSVRun2016B-v2-toBDT.root", treename = "treeout"))
  Data1mix["target"] = 0
  Data2mix = pandas.DataFrame(root_numpy.root2array("../datasets/baseline-Mixed-BTagCSVRun2016C-v2-toBDT.root", treename = "treeout"))
  Data2mix["target"] = 0
  Data3mix = pandas.DataFrame(root_numpy.root2array("../datasets/baseline-Mixed-BTagCSVRun2016D-v2-toBDT.root", treename = "treeout"))
  Data3mix["target"] = 0
  datasetmix = signalData.append(Data1mix, ignore_index = True) 
  datasetmix = datasetmix.append(Data2mix, ignore_index = True)   
  datasetmix = datasetmix.append(Data3mix, ignore_index = True)  
if typedata =="QCD" : 
  QCDbackgroundDataPlain=[]
  QCDbackgroundDataPlainExt=[]
  HT=["200to300","300to500","500to700","700to1000","1000to1500","1500to2000","2000toInf"]
  for ifile in range(2,7) :
    print HT[ifile]
    QCDbackgroundDataPlain.append(pandas.DataFrame(root_numpy.root2array("../datasets/"+subset+"QCD_HT"+HT[ifile]+"-toBDT.root", treename = "treeout")))
    QCDbackgroundDataPlainExt.append(pandas.DataFrame(root_numpy.root2array("../datasets/"+subset+"QCD_HT"+HT[ifile]+"_ext-toBDT.root", treename = "treeout")))
    QCDbackgroundDataPlain[ifile-2]["target"] = 0
    QCDbackgroundDataPlainExt[ifile-2]["target"] = 0
  dataset = signalData.append(QCDbackgroundDataPlain[0], ignore_index = True)  
  dataset = dataset.append(QCDbackgroundDataPlainExt[0], ignore_index = True)  
  for ifile in range(1,5) : 
    dataset = dataset.append(QCDbackgroundDataPlain[ifile], ignore_index = True)   
    dataset = dataset.append(QCDbackgroundDataPlainExt[ifile], ignore_index = True)   
  ###############################################################################
  mixDataHigh = pandas.DataFrame(root_numpy.root2array("../datasets/baseline-Mixed-QCDHT500toInf_noTrg-toBDT.root", treename = "treeout"))
  mixDataHigh["target"] = 0
  datasetmix = signalData.append(mixDataHigh, ignore_index = True) 
################################################################################
weights = "weight"
weights1P = "weight"
weights1M = "weight"

hlFeatures = [hl for hl in dataset.columns if (str.startswith(hl, "jet")) ] #
FeaturesJets = hlFeatures

hlFeatures = [hl for hl in dataset.columns if (str.startswith(hl, "C")) or\
             (str.startswith(hl, "m")) or (str.startswith(hl, "pth")) or (str.startswith(hl, "HHC"))  or (str.startswith(hl, "HHpt"))  or\
             (str.startswith(hl, "H1C")) or (str.startswith(hl, "H2C")) or (str.startswith(hl, "H1Dphi")) or (str.startswith(hl, "H2Dphi")) or (str.startswith(hl, "D")) or (str.startswith(hl, "DR")) or (str.startswith(hl, "jetpt")) or (str.startswith(hl, "jetHT"))  or (str.startswith(hl, "jeteta")) ] #  or (str.startswith(hl, "HHD")) 
trainFeaturesplot = hlFeatures

hlFeatures = [hl for hl in dataset.columns if (str.startswith(hl, "C")) or (str.startswith(hl, "mX")) or\
             (str.startswith(hl, "mh1")) or (str.startswith(hl, "mh2"))  or\
             (str.startswith(hl, "HHC"))  or (str.startswith(hl, "HHDhi"))   or (str.startswith(hl, "H1C")) or (str.startswith(hl, "H2C")) or (str.startswith(hl, "DR")) or (str.startswith(hl, "jeteta"))  or (str.startswith(hl, "H1Dphi")) or (str.startswith(hl, "H2Dphi")) ]
trainFeaturesAll = hlFeatures
print "Training full on:", [var for var in trainFeaturesAll]

text_file.write("Training full on: "+ str([var for var in trainFeaturesAll])+"\n") 

hlFeatures = [hl for hl in dataset.columns if (str.startswith(hl, "CSV")) or (str.startswith(hl, "mX")) or\
             (str.startswith(hl, "mh1")) or (str.startswith(hl, "mh2"))  or \
             (str.startswith(hl, "HHC"))  or (str.startswith(hl, "H1C")) or (str.startswith(hl, "H2C"))  or (str.startswith(hl, "jetHTrest")) or (str.startswith(hl, "jeteta")) ]
trainFeaturesObvious = hlFeatures
print "Training obvious on:", [var for var in trainFeaturesObvious]
text_file.write("Training mass on: "+ str([var for var in trainFeaturesObvious])+"\n")

hlFeatures = [hl for hl in dataset.columns if (str.startswith(hl, "CSV"))  or (str.startswith(hl, "jetHTrest") )or \
             (str.startswith(hl, "HHC"))  or   (str.startswith(hl, "H1C")) or (str.startswith(hl, "H2C"))   or (str.startswith(hl, "jeteta"))   ] # (str.startswith(hl, "DR")) or or (str.startswith(hl, "mhh")) or (str.startswith(hl, "pt")) (str.startswith(hl, "HHDphi"))  or (str.startswith(hl, "H1Dphi")) or or (str.startswith(hl, "jetHTfull")) 
trainFeaturesHH = hlFeatures
print "Training diHiggs on:", [var for var in trainFeaturesHH]
text_file.write( "Training angular on: "+ str([var for var in trainFeaturesHH])+"\n")

#################################################################################
### Plot some histograms
################################################################################# 
print "plain QCD"
### against QCD
hist_params = {'normed': False, 'bins': 18, 'alpha': 0.4}
plt.figure(figsize=(30, 20))
if typedata=="Data": fraction="1/20 of" 
elif typedata=="Data": fraction=" "
for n, feature in enumerate(trainFeaturesplot):
    # add sub plot on our figure
    plt.subplot(5, 6, n+1)
    # define range for histograms by cutting 1% of data from both ends
    if n == 0 or n == 1 or n == 2 or n == 4 or n == 5 : min_value, max_value = np.percentile(dataset[feature], [1, 99])
    else : min_value, max_value = np.percentile(dataset[feature], [1, 99])
    #"""
    if typedata=="Data": 
        values, bins, _ = plt.hist(dataset20.ix[dataset20.target.values == 0, feature].values,  weights= dataset20.ix[dataset20.target.values == 0, weights].values , 
                               range=(min_value, max_value), label=fraction+typedata, **hist_params )
    elif typedata == "QCD" : 
            values, bins, _ = plt.hist(dataset.ix[dataset.target.values == 0, feature].values, weights= dataset.ix[dataset.target.values == 0, weights].values , 
                               range=(min_value, max_value), label=typedata, **hist_params)
    values, bins, _ = plt.hist(datasetmix.ix[datasetmix.target.values == 0, feature].values,  weights= (datasetmix.ix[datasetmix.target.values == 0, weights].values) ,  range=(min_value, max_value), label='Mixed '+typedata, **hist_params)
    areaBKG2 = sum(np.diff(bins)*values)   
    #"""
    values, bins, _ = plt.hist(dataset.ix[dataset.target.values == 1, feature].values, weights= dataset.ix[dataset.target.values == 1, weights].values , 
                               range=(min_value, max_value), label='Signal', **hist_params)
    areaSig = sum(np.diff(bins)*values)
    #"""
    #print areaBKG, " ",areaBKG2 ," ",areaSig
    if n == 0 : plt.legend(loc='best')
    plt.title(feature)
plt.savefig("Variables_"+subset+BKG+ext)
plt.clf()
#################################################################################
hist_params = {'normed': True, 'bins': 18, 'alpha': 0.4}
plt.figure(figsize=(30, 20))
for n, feature in enumerate(trainFeaturesplot):
    # add sub plot on our figure
    plt.subplot(5, 6, n+1)
    # define range for histograms by cutting 1% of data from both ends
    if n == 0 or n == 1 or n == 2 or n == 4 or n == 5 : min_value, max_value = np.percentile(dataset[feature], [1, 99])
    else : min_value, max_value = np.percentile(dataset[feature], [1, 99])
    """
    y,binEdges = np.histogram(dataset.ix[dataset.target.values == 0, feature].values,  bins = 18)
    bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    menStd     = np.sqrt(y)
    values, bins, _ = plt.hist(dataset.ix[dataset.target.values == 0, feature].values,  weights= dataset.ix[dataset.target.values == 0, weights].values , 
                               range=(min_value, max_value), label='Background', **hist_params )
    """
    #plt.errorbar(bincenters, values, yerr=menStd )
    #areaBKG = sum(np.diff(bins)*values)
    #y,binEdges = np.histogram(datasetmix.ix[datasetmix.target.values == 0, feature].values,  bins = 18)
    #bincenters = 0.5*(binEdges[1:]+binEdges[:-1])
    #menStd     = np.sqrt(y)
    values, bins, _ = plt.hist(signalData8.ix[signalData8.target.values == 1, feature].values,  weights= (signalData8.ix[signalData8.target.values == 1, weights].values) ,  range=(min_value, max_value), label='Benchmark 8 (v1)', **hist_params)
    areaBKG2 = sum(np.diff(bins)*values)   
    #"""
    values, bins, _ = plt.hist(dataset.ix[dataset.target.values == 1, feature].values, weights= dataset.ix[dataset.target.values == 1, weights].values , 
                               range=(min_value, max_value), label='Signal SM', **hist_params)
    areaSig = sum(np.diff(bins)*values)
    #"""
    #print areaBKG, " ",areaBKG2 ," ",areaSig
    if n == 0 : plt.legend(loc='best')
    plt.title(feature)
plt.savefig("Variables_"+subset+BKG+"_benchmarks_"+ext)
plt.clf()
#################################################################################
### Define classifiers to test
traindataset, valdataset = train_test_split(dataset, random_state=11, train_size=0.50)
traindatasetmix, valdatasetmix = train_test_split(datasetmix, random_state=11, train_size=0.50)
#################################################################################
arr = valdatasetmix.to_records()
array2root(arr, outputCentral+"_AppliedToMixed"+typedata+".root" , 'tree', 'recreate') 
arr = dataset.to_records()
array2root(arr, outputCentral+"_AppliedToPlain"+typedata+".root" , 'tree', 'recreate')
if typedata=="Data": 
  arr = dataset20.to_records()
  array2root(arr, outputCentral+"_AppliedTo20pOfPlain"+typedata+".root" , 'tree', 'recreate')
#

for ii in range(0,3):
   if ii==0 :
     train= trainFeaturesplot
     Var='All'
   if ii==1 :
     train= trainFeaturesObvious
     Var='Mass'
   if ii==2 :
     train= trainFeaturesHH
     Var='HH'
   xgb = XGBoostClassifier(train) #,
   """
            n_estimators =  200,
            eta = 0.1,
            max_depth = 7,
            subsample = 0.9,
            colsample = 0.6)
   """
   xgb.fit(traindatasetmix[train].astype(np.float64), traindatasetmix.target.astype(np.bool),\
        sample_weight= (traindatasetmix[weights].astype(np.float64))) 
   prob = xgb.predict_proba(valdatasetmix[train].astype(np.float64)  )
   if ii==0 : reportAll = xgb.test_on(traindatasetmix[trainFeaturesplot].astype(np.float64), traindatasetmix.target.astype(np.bool))
   if ii==1 : reportObvious = xgb.test_on(traindatasetmix[trainFeaturesObvious].astype(np.float64), traindatasetmix.target.astype(np.bool))
   if ii==2 : reportHH = xgb.test_on(traindatasetmix[trainFeaturesHH].astype(np.float64), traindatasetmix.target.astype(np.bool))
   joblib.dump(xgb, outputCentral+"_AppliedToMixed"+typedata+'.pkl')
   print "train in mixed"
   print 'ROC AUC '+Var+' variables:', roc_auc_score(valdatasetmix.target.astype(np.bool) , prob[:, 1] )
   text_file.write("train in mixed ROC AUC "+Var+' variables:'+str( roc_auc_score(valdatasetmix.target.astype(np.bool) , prob[:, 1] ))+"\n")
   prob.dtype = [('bdt'+Var+'Variables', np.float32)]
   array2root(prob[:, 0], outputCentral+"_AppliedToMixed"+typedata+".root", "tree")
   prob2 = xgb.predict_proba(dataset[train].astype(np.float64))
   prob2.dtype = [('bdt'+Var+'Variables', np.float32)]
   array2root(prob2[:, 0], outputCentral+"_AppliedToPlain"+typedata+".root", "tree")
   if typedata=="Data": 
      prob3 = xgb.predict_proba(dataset20[train].astype(np.float64))
      prob3.dtype = [('bdt'+Var+'Variables', np.float32)]
      array2root(prob3[:, 0], outputCentral+"_AppliedTo20pOfPlain"+typedata+".root", "tree")
text_file.close()
################################################################################
correlation_pairs = []
correlation_pairs.append((trainFeaturesAll[3], trainFeaturesAll[4]))
correlation_pairs.append((trainFeaturesAll[3], trainFeaturesAll[4]))
reportAll.scatter(correlation_pairs, alpha=0.01).plot()
plt.savefig('Scatter_'+subset+BKG+ext)
plt.clf()
################################################################################
plt.figure('ROC QCD', figsize=(7,7))
reportAll.roc(physics_notion=True).plot(ylim=(0.7, 1))
reportObvious.roc(physics_notion=True).plot(new_plot=False)
reportHH.roc(physics_notion=True).plot(new_plot=False)
plt.legend(('All variables', 'Mass variables', 'HH system'), loc='lower left')
plt.title('QCD b enriched, train in 40%')
plt.show()
plt.savefig('ROC_'+subset+BKG+ext)
plt.clf()
############################################################################
plt.figure('Importances', figsize=(16,10))
plt.subplot(1, 3, 1)
features_importances = reportAll.feature_importance()
features_importances.plot(figsize=(16,8))
plt.savefig('ImportancesAll_'+subset+BKG+ext)
plt.clf()
features_importancesObvious = reportObvious.feature_importance()
features_importancesObvious.plot(figsize=(16,8))
plt.savefig('ImportancesObvious_'+subset+BKG+ext)
plt.clf()
features_importancesHH = reportHH.feature_importance()
features_importancesHH.plot(figsize=(16,8))
plt.savefig('ImportancesHH_'+subset+BKG+ext)
plt.clf()
############################################################################
cmap = cm.get_cmap('jet', 30)
reportAll.features_correlation_matrix(\
    features=dataset.ix[(dataset.target.values == 0), trainFeaturesplot].astype(np.float64), cmap=cmap).plot(new_plot=True, show_legend=False, figsize=(10, 10))
plt.savefig('CorrelationsMatrix_AllVar_signal_'+subset+typedata+ext)  #_by_class
plt.clf()
reportAll.features_correlation_matrix(\
    features=dataset.ix[(dataset.target.values == 1), trainFeaturesplot].astype(np.float64), cmap=cmap).plot(new_plot=True, show_legend=False, figsize=(10, 10))
plt.savefig('CorrelationsMatrix_AllVar_bkg_'+subset+typedata+ext)  #_by_class
plt.clf()
############################################################################
trainFeaturesMasses  = [hl for hl in dataset.columns if (str.startswith(hl, "mX")) or (str.startswith(hl, "mh"))  or (str.startswith(hl, "pth")) or (str.startswith(hl, "HHpt")) ] #
reportAll.features_correlation_matrix(\
    features=dataset.ix[(dataset.target.values == 1), trainFeaturesMasses].astype(np.float64), cmap=cmap).plot(new_plot=True, show_legend=False, figsize=(4, 4))
plt.savefig('CorrelationsMatrix_Masses_signal_'+subset+typedata+ext)  #_by_class
plt.clf()
reportAll.features_correlation_matrix(\
    features=dataset.ix[(dataset.target.values == 0), trainFeaturesMasses].astype(np.float64), cmap=cmap).plot(new_plot=True, show_legend=False, figsize=(4, 4))
plt.savefig('CorrelationsMatrix_Masses_bkg_'+subset+typedata+ext)  #_by_class
plt.clf()
############################################################################
trainFeaturesMasses  = [hl for hl in dataset.columns if (str.startswith(hl, "mX")) or (str.startswith(hl, "mh")) ] #
reportAll.features_correlation_matrix(\
    features=dataset.ix[(dataset.target.values == 1), trainFeaturesMasses].astype(np.float64), cmap=cmap).plot(new_plot=True, show_legend=False, figsize=(4, 4))
plt.savefig('CorrelationsMatrix_MassesOnly_signal_'+subset+typedata+ext)  #_by_class
plt.clf()
reportAll.features_correlation_matrix(\
    features=dataset.ix[(dataset.target.values == 0), trainFeaturesMasses].astype(np.float64), cmap=cmap).plot(new_plot=True, show_legend=False, figsize=(4, 4))
plt.savefig('CorrelationsMatrix_MassesOnly_bkg_'+subset+typedata+ext)  #_by_class
plt.clf()
############################################################################
trainFeaturesMasses  = [hl for hl in dataset.columns if (str.startswith(hl, "mX")) or (str.startswith(hl, "mh1")) or (str.startswith(hl, "mh2")) or (str.startswith(hl, "HHC")) or (str.startswith(hl, "H1C")) or (str.startswith(hl, "H2C")) or (str.startswith(hl, "H1Dphi")) or (str.startswith(hl, "H2Dphi")) or (str.startswith(hl, "DR"))] #
reportAll.features_correlation_matrix(\
    features=dataset.ix[(dataset.target.values == 1), trainFeaturesMasses].astype(np.float64), cmap=cmap).plot(new_plot=True, show_legend=False, figsize=(4.5, 4.5))
plt.savefig('CorrelationsMatrix_Angles_signal_'+subset+typedata+ext)  #_by_class
plt.clf()
reportAll.features_correlation_matrix(\
    features=dataset.ix[(dataset.target.values == 0), trainFeaturesMasses].astype(np.float64), cmap=cmap).plot(new_plot=True, show_legend=False, figsize=(4.5, 4.5))
plt.savefig('CorrelationsMatrix_Angles_bkg_'+subset+typedata+ext)  #_by_class
plt.clf()
############################################################################
trainFeaturesMasses  = [hl for hl in dataset.columns if (str.startswith(hl, "mX")) or (str.startswith(hl, "mh1")) or (str.startswith(hl, "mh2")) or (str.startswith(hl, "jetpt")) or (str.startswith(hl, "jeteta")) or (str.startswith(hl, "CSV")) or (str.startswith(hl, "jetHT")) ] #
reportAll.features_correlation_matrix(\
    features=dataset.ix[(dataset.target.values == 1), trainFeaturesMasses].astype(np.float64), cmap=cmap).plot(new_plot=True, show_legend=False, figsize=(5.5, 5.5))
plt.savefig('CorrelationsMatrix_Jets_signal_'+subset+typedata+ext)  #_by_class
plt.clf()
reportAll.features_correlation_matrix(\
    features=dataset.ix[(dataset.target.values == 0), trainFeaturesMasses].astype(np.float64), cmap=cmap).plot(new_plot=True, show_legend=False, figsize=(5, 5))
plt.savefig('CorrelationsMatrix_Jets_bkg_'+subset+typedata+ext)  #_by_class
plt.clf()
#plt.figure('Correlations Matrix', figsize=(16, 16))
#reportAll.features_correlation_matrix_by_class(\
#    features=traindataset[FeaturesJets].astype(np.float64)).plot(new_plot=True, show_legend=False, figsize=(20, 10))
#plt.savefig('CorrelationsMatrix_JetVar_'+subset+BKG+ext)
#plt.clf()
############################################################################
#plt.figure('Correlations Matrix', figsize=(16, 16))
#reportAll.features_correlation_matrix_by_class(\
#    features=traindataset[trainFeaturesObvious].astype(np.float64)).plot(new_plot=True, show_legend=False, figsize=(20, 10))
#plt.savefig('CorrelationsMatrix__Mass-full.png')
#plt.clf()
############################################################################
#plt.figure('Correlations Matrix', figsize=(16, 16))
#reportAll.features_correlation_matrix_by_class(\
#    features=traindataset[trainFeaturesHH].astype(np.float64)).plot(new_plot=True, show_legend=False, figsize=(20, 10))
#plt.savefig("CorrelationsMatrix-"+subset+"-HH.png")
plt.clf()
###########################################################################
plt.clf()
plt.figure('Features',figsize=(6, 6))
#xmin, xmax = plt.xlim()
reportAll.prediction_pdf().plot(new_plot=False )
reportObvious.prediction_pdf().plot(new_plot=False) 
reportHH.prediction_pdf().plot(new_plot=False)
#plt.xscale('log')
#plt.yscale('log')
plt.legend(('BKG All variables', 'Signal All variables', 'BKG Mass variables', 'Signal Mass variables', 'BKG HH system', 'Signal HH system'), loc='best')
plt.savefig("Features"+subset+BKG+outputCentral+ext)
###########################################################################
plt.clf()
plt.figure('Features',figsize=(6, 6))
#xmin, xmax = plt.xlim()
reportAll.prediction_pdf().plot(new_plot=False  )
#plt.xscale('log')
#plt.yscale('log')
plt.legend(loc='best')
plt.savefig("FeaturesAll"+subset+BKG+outputCentral+ext)
###########################################################################
plt.clf()
plt.figure('Features',figsize=(6, 6))
#xmin, xmax = plt.xlim()
reportHH.prediction_pdf().plot(new_plot=False)
#plt.xscale('log')
#plt.yscale('log')
plt.legend( loc='best')
plt.savefig("FeaturesHH"+subset+BKG+outputCentral+ext)
###########################################################################
###########################################################################
plt.clf()
plt.figure('Features',figsize=(6, 6))
#xmin, xmax = plt.xlim()
reportObvious.prediction_pdf().plot(new_plot=False) 
#plt.xscale('log')
#plt.yscale('log')
plt.legend( loc='best')
plt.savefig("FeaturesMass"+subset+BKG+outputCentral+ext)
plt.clf()
###########################################################################
plt.clf()
plt.figure('Features',figsize=(6, 6))
#xmin, xmax = plt.xlim()
reportAll.prediction_pdf(plot_type='bar').plot(new_plot=False )
#plt.xscale('log')
#plt.yscale('log')
plt.legend(('BKG All variables', 'Signal All variables', 'BKG Mass variables', 'Signal Mass variables', 'BKG HH system', 'Signal HH system'), loc='best')
plt.savefig("Features-All-"+subset+BKG+outputCentral+ext)
plt.clf()
############################################################################
### Plot some histograms
#################################################################################
print "plain QCD valdataset"
ValData = pandas.DataFrame(root_numpy.root2array(outputCentral+"_AppliedToMixed"+typedata+".root", treename = "tree"))
ValDataPlain = pandas.DataFrame(root_numpy.root2array(outputCentral+"_AppliedToPlain"+typedata+".root", treename = "tree"))
if typedata=="Data":  ValDataPlain20 = pandas.DataFrame(root_numpy.root2array(outputCentral+"_AppliedTo20pOfPlainData.root", treename = "tree"))
################################################################################
hlFeatures = [hl for hl in ValData.columns if (str.startswith(hl, "mX")) or (str.startswith(hl, "mh2")) or (str.startswith(hl, "mh1"))  ] #
trainFeaturesplotVal = hlFeatures
hist_params = {'normed': True, 'bins': 20, 'alpha': 0.4}
################################################################################
plt.figure(figsize=(8,20))
if typedata=="Data" :
  for n, feature in enumerate(trainFeaturesplotVal):
    plt.subplot(3, 1, n+1)
    min_value, max_value = np.percentile(valdataset[feature], [1, 80])
    values, bins, _ = plt.hist(ValDataPlain20.ix[(ValDataPlain20.target.values == 0) & (ValDataPlain20.bdtMassVariables.values > 0.5) ,  feature].values,\
                               weights= ValDataPlain20.ix[(ValDataPlain20.target.values == 0) & (ValDataPlain20.bdtMassVariables.values > 0.5) ,  weights].values , 
                               range=(min_value, max_value), label='1/20 Plain '+typedata+' (Mass BDT > 0.5)', **hist_params)

    values, bins, _ = plt.hist(ValData.ix[(ValData.target.values == 0) & (ValData.bdtMassVariables.values > 0.5) ,  feature].values,\
                               weights= (ValData.ix[(ValData.target.values == 0) & (ValData.bdtMassVariables.values > 0.5) ,  weights].values) , 
                               range=(min_value, max_value), label='Mixed '+typedata+' (Mass BDT > 0.5)', **hist_params)
    areaBKG2 = sum(np.diff(bins)*values)
    """
    values, bins, _ = plt.hist(valdataset.ix[(ValData.target.values == 0) & (ValData.bdtMassVariables.values > 0.8),  feature].values,\
                               weights= valdataset.ix[(ValData.target.values == 0) & (ValData.bdtMassVariables.values > 0.8),  weights].values , 
                               range=(min_value, max_value), label='BKG  bdtMass > 0.8', **hist_params)
    areaSig = sum(np.diff(bins)*values)
    """
    #print areaBKG," ",areaBKG2 #," ",areaSig
    if n ==1 : plt.legend(loc='best')
    plt.title(feature)
  plt.savefig("Variables_"+BKG+"-"+subset+outputCentral+"MassVar_sigRegion.png")
  plt.clf()
elif typedata=="QCD" :
  for n, feature in enumerate(trainFeaturesplotVal):
    plt.subplot(3, 1, n+1)
    min_value, max_value = np.percentile(valdataset[feature], [1, 80])
    values, bins, _ = plt.hist(ValDataPlain.ix[(ValDataPlain.target.values == 0) & (ValDataPlain.bdtMassVariables.values > 0.5) ,  feature].values,\
                               weights= ValDataPlain.ix[(ValDataPlain.target.values == 0) & (ValDataPlain.bdtMassVariables.values > 0.5) ,  weights].values , 
                               range=(min_value, max_value), label='Plain '+typedata+' (Mass BDT > 0.5)', **hist_params)

    values, bins, _ = plt.hist(ValData.ix[(ValData.target.values == 0) & (ValData.bdtMassVariables.values > 0.5) ,  feature].values,\
                               weights= (ValData.ix[(ValData.target.values == 0) & (ValData.bdtMassVariables.values > 0.5) ,  weights].values) , 
                               range=(min_value, max_value), label='Mixed '+typedata+' (Mass BDT > 0.5)', **hist_params)
    areaBKG2 = sum(np.diff(bins)*values)
    """
    values, bins, _ = plt.hist(valdataset.ix[(ValData.target.values == 0) & (ValData.bdtMassVariables.values > 0.8),  feature].values,\
                               weights= valdataset.ix[(ValData.target.values == 0) & (ValData.bdtMassVariables.values > 0.8),  weights].values , 
                               range=(min_value, max_value), label='BKG  bdtMass > 0.8', **hist_params)
    areaSig = sum(np.diff(bins)*values)
    """
    #print areaBKG," ",areaBKG2 #," ",areaSig
    if n ==1 : plt.legend(loc='best')
    plt.title(feature)
  plt.savefig("Variables_"+BKG+"-"+subset+outputCentral+"MassVar_sigRegion.png")
  plt.clf()
################################################################################
plt.figure(figsize=(8,20))
for n, feature in enumerate(trainFeaturesplotVal):
    # add sub plot on our figure
    plt.subplot(3, 1, n+1)
    min_value, max_value = np.percentile(valdataset[feature], [1, 80])
    values, bins, _ = plt.hist(ValDataPlain.ix[(ValDataPlain.target.values == 0) & (ValDataPlain.bdtMassVariables.values < 0.5) ,  feature].values,\
                               weights= ValDataPlain.ix[(ValDataPlain.target.values == 0) & (ValDataPlain.bdtMassVariables.values < 0.5) ,  weights].values , 
                               range=(min_value, max_value), label='Plain '+typedata+' (Mass BDT < 0.5)', **hist_params)
    values, bins, _ = plt.hist(ValData.ix[(ValData.target.values == 0) & (ValData.bdtMassVariables.values < 0.5) ,  feature].values,\
                               weights= (ValData.ix[(ValData.target.values == 0) & (ValData.bdtMassVariables.values < 0.5) ,  weights].values) , 
                               range=(min_value, max_value), label='Mixed '+typedata+' (Mass BDT < 0.5)', **hist_params)
    areaBKG2 = sum(np.diff(bins)*values)
    """
    values, bins, _ = plt.hist(valdataset.ix[(ValData.target.values == 0) & (ValData.bdtMassVariables.values > 0.8),  feature].values,\
                               weights= valdataset.ix[(ValData.target.values == 0) & (ValData.bdtMassVariables.values > 0.8),  weights].values , 
                               range=(min_value, max_value), label='BKG  bdtMass > 0.8', **hist_params)
    areaSig = sum(np.diff(bins)*values)
    """
    #print areaBKG," ",areaBKG2 #," ",areaSig
    if n ==1 : plt.legend(loc='best')
    plt.title(feature)
plt.savefig("Variables_"+BKG+"-"+subset+outputCentral+"MassVar_controlRegion.png")
plt.clf()
#################################################################################
plt.figure(figsize=(8, 20))
if typedata=="Data" :
  for n, feature in enumerate(trainFeaturesplotVal):
    # add sub plot on our figure
    plt.subplot(3, 1, n+1)
    min_value, max_value = np.percentile(valdataset[feature], [1, 99])
    values, bins, _ = plt.hist(ValDataPlain20.ix[(ValDataPlain20.target.values == 0) & (ValDataPlain20.bdtHHVariables.values > 0.5) ,  feature].values,\
                               weights= ValDataPlain20.ix[(ValDataPlain20.target.values == 0) & (ValDataPlain20.bdtHHVariables.values > 0.5) ,  weights].values , 
                               range=(min_value, max_value), label='1/20 Plain '+typedata+' (HH BDT > 0.5)', **hist_params)
    values, bins, _ = plt.hist(ValData.ix[(ValData.target.values == 0) & (ValData.bdtHHVariables.values > 0.5)  ,  feature].values,\
                               weights= (ValData.ix[(ValData.target.values == 0) & (ValData.bdtHHVariables.values > 0.5)  ,  weights].values) , 
                               range=(min_value, max_value), label='Mixed '+typedata+' (HH BDT > 0.5)', **hist_params)
    areaBKG = sum(np.diff(bins)*values)
    #values, bins, _ = plt.hist(ValData.ix[(ValData.target.values == 0)  ,  feature].values,\
    #                           weights= (ValData.ix[(ValData.target.values == 0)  ,  weights].values) , 
    #                           range=(min_value, max_value), label='Background ', **hist_params)
    #areaBKG = sum(np.diff(bins)*values)
    if n ==1 : plt.legend(loc='best')
    plt.title(feature)
  plt.savefig("Variables_"+BKG+"-"+subset+outputCentral+"HHVar_sigRegion.png")
  plt.clf()
elif typedata=="QCD" :
  for n, feature in enumerate(trainFeaturesplotVal):
    # add sub plot on our figure
    plt.subplot(3, 1, n+1)
    min_value, max_value = np.percentile(valdataset[feature], [1, 99])
    values, bins, _ = plt.hist(ValDataPlain.ix[(ValDataPlain.target.values == 0) & (ValDataPlain.bdtHHVariables.values > 0.5) ,  feature].values,\
                               weights= ValDataPlain.ix[(ValDataPlain.target.values == 0) & (ValDataPlain.bdtHHVariables.values > 0.5) ,  weights].values , 
                               range=(min_value, max_value), label='Plain '+typedata+' (HH BDT > 0.5)', **hist_params)
    values, bins, _ = plt.hist(ValData.ix[(ValData.target.values == 0) & (ValData.bdtHHVariables.values > 0.5)  ,  feature].values,\
                               weights= (ValData.ix[(ValData.target.values == 0) & (ValData.bdtHHVariables.values > 0.5)  ,  weights].values) , 
                               range=(min_value, max_value), label='Mixed '+typedata+' (HH BDT > 0.5)', **hist_params)
    areaBKG = sum(np.diff(bins)*values)
    #values, bins, _ = plt.hist(ValData.ix[(ValData.target.values == 0)  ,  feature].values,\
    #                           weights= (ValData.ix[(ValData.target.values == 0)  ,  weights].values) , 
    #                           range=(min_value, max_value), label='Background ', **hist_params)
    #areaBKG = sum(np.diff(bins)*values)
    if n ==1 : plt.legend(loc='best')
    plt.title(feature)
  plt.savefig("Variables_"+BKG+"-"+subset+outputCentral+"HHVar_sigRegion.png")
  plt.clf()
#################################################################################
plt.figure(figsize=(8, 20))
for n, feature in enumerate(trainFeaturesplotVal):
    # add sub plot on our figure
    plt.subplot(3, 1, n+1)
    min_value, max_value = np.percentile(valdataset[feature], [1, 99])
    values, bins, _ = plt.hist(ValDataPlain.ix[(ValDataPlain.target.values == 0) & (ValDataPlain.bdtHHVariables.values < 0.5) ,  feature].values,\
                               weights= ValDataPlain.ix[(ValDataPlain.target.values == 0) & (ValDataPlain.bdtHHVariables.values < 0.5) ,  weights].values , 
                               range=(min_value, max_value), label='Plain '+typedata+' (HH BDT < 0.5)', **hist_params)
    values, bins, _ = plt.hist(ValData.ix[(ValData.target.values == 0) & (ValData.bdtHHVariables.values < 0.5)  ,  feature].values,\
                               weights= (ValData.ix[(ValData.target.values == 0) & (ValData.bdtHHVariables.values < 0.5)  ,  weights].values) , 
                               range=(min_value, max_value), label='Mixed '+typedata+' (HH BDT < 0.5)', **hist_params)
    areaBKG = sum(np.diff(bins)*values)
    #values, bins, _ = plt.hist(ValData.ix[(ValData.target.values == 0)  ,  feature].values,\
    #                           weights= (ValData.ix[(ValData.target.values == 0)  ,  weights].values) , 
    #                           range=(min_value, max_value), label='Background ', **hist_params)
    #areaBKG = sum(np.diff(bins)*values)
    if n ==1 : plt.legend(loc='best')
    plt.title(feature)
plt.savefig("Variables_"+BKG+"-"+subset+outputCentral+"HHVar_controlRegion.png")
plt.clf()
#################################################################################
plt.figure(figsize=(8, 20))
for n, feature in enumerate(trainFeaturesplotVal):
    # add sub plot on our figure
    plt.subplot(3, 1, n+1)
    # define range for histograms by cutting 1% of data from both ends
    min_value, max_value = np.percentile(valdataset[feature], [1, 99])
    #selected  = ValData(logical_and(ValData.target.values == 0, ValData.bdtMassVariables.values >0.6))
    """
    values, bins, _ = plt.hist(ValDataPlain.ix[(ValDataPlain.target.values == 0) & (ValDataPlain.bdtHHVariables.values > 0.6) ,  feature].values,\
                               weights= ValDataPlain.ix[(ValDataPlain.target.values == 0) & (ValDataPlain.bdtHHVariables.values > 0.6) ,  weights].values , 
                               range=(min_value, max_value), label='Plain data (HH BDT > 0.8)', **hist_params)
    areaBKG = sum(np.diff(bins)*values)
    # """
    values, bins, _ = plt.hist(ValData.ix[(ValData.target.values == 0) & (ValData.bdtHHVariables.values > 0.6)  ,  feature].values,\
                               weights= (ValData.ix[(ValData.target.values == 0) & (ValData.bdtHHVariables.values > 0.6)  ,  weights].values) , 
                               range=(min_value, max_value), label='Mixed '+typedata+' (HH BDT > 0.6)', **hist_params)
    areaBKG = sum(np.diff(bins)*values)
    values, bins, _ = plt.hist(ValData.ix[(ValData.target.values == 0)  ,  feature].values,\
                               weights= (ValData.ix[(ValData.target.values == 0)  ,  weights].values) , 
                               range=(min_value, max_value), label='Mixed '+typedata+' ', **hist_params)
    areaBKG = sum(np.diff(bins)*values)
    if n ==1 : plt.legend(loc='best')
    plt.title(feature)
plt.savefig("Variables_"+BKG+"-"+subset+outputCentral+"HHVar_BDTcomparison.png")
plt.clf()

#################################################################################
plt.figure(figsize=(8, 20))
for n, feature in enumerate(trainFeaturesplotVal):
    # add sub plot on our figure
    plt.subplot(3, 1, n+1)
    # define range for histograms by cutting 1% of data from both ends
    min_value, max_value = np.percentile(valdataset[feature], [1, 99])
    #selected  = ValData(logical_and(ValData.target.values == 0, ValData.bdtMassVariables.values >0.6))
    """
    values, bins, _ = plt.hist(ValDataPlain.ix[(ValDataPlain.target.values == 0) & (ValDataPlain.bdtHHVariables.values > 0.8) ,  feature].values,\
                               weights= ValDataPlain.ix[(ValDataPlain.target.values == 0) & (ValDataPlain.bdtHHVariables.values > 0.8) ,  weights].values , 
                               range=(min_value, max_value), label='Background bdtHH > 0.8', **hist_params)
    areaBKG = sum(np.diff(bins)*values)
    """
    values, bins, _ = plt.hist(ValData.ix[(ValData.target.values == 1) & (ValData.bdtHHVariables.values > 0.6)  ,  feature].values,\
                               weights= (ValData.ix[(ValData.target.values == 1) & (ValData.bdtHHVariables.values > 0.6)  ,  weights].values) , 
                               range=(min_value, max_value), label='Signal bdt > 0.6', **hist_params)
    values, bins, _ = plt.hist(ValData.ix[(ValData.target.values == 1)  ,  feature].values,\
                               weights= (ValData.ix[(ValData.target.values == 1)  ,  weights].values) , 
                               range=(min_value, max_value), label='signal ', **hist_params)
    if n ==0 : plt.legend(loc='best')
    plt.title(feature)
plt.savefig("Variables_"+BKG+"-"+subset+outputCentral+"HHVar_signal.png")
plt.clf()
#################################################################################
# nsig nbkg after selection



