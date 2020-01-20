from feature_selector import FeatureSelector
import glob, os, sys
from sklearn.ensemble import ExtraTreesClassifier
from feature_selector import FeatureSelector
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import pandas as pd
import numpy as np
import sklearn

#fs = glob.glob('otutable_outtask_out_n_Zero_Cor_cleaned.tsv')
#full_df = pd.read_csv('otutable_outtask_out_n_Zero_Cor_cleaned.tsv',sep='\t')

def Remove_corr_features(in_file, outFile, corr_cutoff):

    df = pd.read_csv(in_file, sep='\t')
    clm_list = df.columns.tolist()

    X_train = df[clm_list[0:len(clm_list)-1]]
    y_train = df[clm_list[len(clm_list)-1]]

    print ("Full Data: ",X_train.shape[1])
    df = X_train.loc[:, (X_train != 0).any(axis=0)]
    print ("After Removed Zero Value Columns: ",df.shape[1])

    filter_list = []
    col_list = df.columns.values

    for a in col_list:

        if df[a].tolist().count(0)/float(len(df[a].tolist()))*100 >=95.00:
            pass
        else:
            filter_list.append(a)

    processed_df = df[filter_list]

    print ("After Rmove high Zero Value Columns: ",processed_df.shape[1])

    corr_matrix = processed_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > float(corr_cutoff))]
    final_df = processed_df.drop(df[to_drop], axis=1)
    print  ("After Rmove correlated Columns: ",final_df.shape[1])
    reuduced_corr_features  = pd.concat([final_df.round(3), y_train], axis=1)
    final_df.to_csv(outFile,sep='\t', index=None)

def Selected_best_features(in_file, outFile, nFeatures):

    df = pd.read_csv(in_file, sep='\t')
    #print df
    clm_list = df.columns.tolist()
    X_train = df[clm_list[0:len(clm_list)-1]]
    y_train = df[clm_list[len(clm_list)-1]]


    print (X_train)

    bestfeatures = SelectKBest(score_func=f_classif, k=10)

    fit = bestfeatures.fit(X_train,y_train)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(X_train.columns)
    featureScores = pd.concat([dfcolumns,dfscores],axis=1)
    featureScores.columns = ['Specs','Score']  

    f_clm_list = featureScores.nlargest(int(nFeatures),'Score')['Specs'].tolist()
    fdf = pd.concat([df[f_clm_list].round(3), y_train ], axis=1)
    fdf.to_csv(outFile, index=None, sep='\t')

def Select_important_features(in_file, outFile, nFeatures):

    df = pd.read_csv(in_file, sep='\t')

    clm_list = df.columns.tolist()
    X_train = df[clm_list[0:len(clm_list)-1]]
    y_train = df[clm_list[len(clm_list)-1]]
    model = ExtraTreesClassifier()
    model.fit(X_train, y_train)
    feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
    fdf =  feat_importances.nlargest(int(nFeatures)).to_frame()
    fdf = pd.concat([df[fdf.reset_index()['index'].tolist()], y_train ], axis=1)
    fdf.to_csv(outFile, index=None, sep='\t')

def Bestfeature_from_cummulative_importance(inFile, outFile ):

    df = pd.read_csv(inFile,sep='\t')
    print (df.shape)
    train_labels = df['class_label']
    train = df.drop(columns = ['class_label'])
    fs = FeatureSelector(data = train, labels = train_labels)
    fs.identify_zero_importance(task = 'classification', eval_metric = 'auc', n_iterations = 10, early_stopping = True)
    zero_importance_features = fs.ops['zero_importance']
    #fs.plot_feature_importances(threshold = 0.99, plot_n = 12)
    importance_index = np.min(np.where(fs.feature_importances['cumulative_importance'] > 0.99))
    fs.identify_low_importance(cumulative_importance = 0.99)
    print (importance_index)
    train_removed_all = fs.remove(methods = ['zero_importance'],keep_one_hot=False)
    train_removed_all = pd.concat([train_removed_all,train_labels],axis=1)
    train_removed_all.to_csv(outFile, sep='\t', index=None) 

if __name__=="__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Deployment tool')
    subparsers = parser.add_subparsers()

    corf = subparsers.add_parser('Corr')
    corf.add_argument("--C", required=True, help="")
    corf.add_argument("--InFile", required=True, help="")
    corf.add_argument("--OutFile", required=True, help="")

    slbf = subparsers.add_parser('SelBest')
    slbf.add_argument("--N", required=True, default=1.0, help="")
    slbf.add_argument("--InFile", required=True, default='rbf', help="")
    slbf.add_argument("--OutFile", required=True, help="")

    slif = subparsers.add_parser('SelImpo')
    slif.add_argument("--N", required=True, help="")
    slif.add_argument("--InFile", required=True, help="")
    slif.add_argument("--OutFile", required=True, help="")

    bfcf = subparsers.add_parser('CumImpo')
    bfcf.add_argument("--InFile", required=True, help="")
    bfcf.add_argument("--OutFile", required=True, help="")

    args = parser.parse_args()

if   sys.argv[1] == 'Corr':
    Remove_corr_features(args.InFile, args.OutFile, args.C)
elif sys.argv[1] == 'SelBest':
    Selected_best_features(args.InFile, args.OutFile, args.N)   
elif sys.argv[1] == 'SelImpo':
    Select_important_features(args.InFile, args.OutFile, args.N)
elif sys.argv[1] == 'CumImpo':
    Bestfeature_from_cummulative_importance(args.InFile, args.OutFile)
else:
    print ("its not accurate")
    exit()


