
# from pycaret.classification import setup
# from pycaret.classification import *
# from pycaret.datasets import get_data
import sys
import argparse
import pycaret
import json
import re
from pycaret.classification import *
from pycaret.datasets import get_data
from pycaret.classification import setup, create_model
import pandas   as pd
from pycaret.datasets import get_data

data = get_data('iris')  

def run_pycaret( custom_params, model_type, algo=None, file_path=None, **kwargs):
    df = pd.read_csv(file_path, sep='\t') 
    clf = setup(data=data, **kwargs)
   
    if algo=='lr':
        if model_type == 'custom':
            custom_params = custom_params.replace('"false"', 'false')
            custom_params = custom_params.replace('"true"', 'true')
            custom_params = custom_params.replace('"null"', 'null')
            custom_params = custom_params.replace('"None"', 'null')
            custom_params=json.loads(custom_params)
            model = clf.create_model(algo, **custom_params,)

        elif model_type == 'tune':
            param= {}

            custom_params = json.loads(custom_params)
            
            for k in custom_params.keys():
                if k == 'C':
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                elif k == 'l1_ratio': 
                    if  custom_params[k] == "None":
                        param[k] =  [None]
                    else:
                        param[k] = [float(i) for i in custom_params[k].split(',')]
                elif k == "max_iter":
                    param[k] =  [ int(i) for i in custom_params[k].split(',')]
                elif k == "dual":
                    if custom_params[k]:
                        param[k] = [True]
                    else:
                        param[k] = [False]
                elif k == 'random_state': 
                    if  custom_params[k]:
                        param[k]= [int(i) for i in custom_params[k].split(',')] 
                    else:
                        param[k] = [None]
                elif k == 'fit_intercept':
                    if custom_params[k]:
                        param[k] = [True]
                    else:
                        param[k] = [False]
                elif k == 'intercept_scaling':
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                elif k == 'multi_class':
                    param[k] =  [i for i in custom_params[k].split(',')]
                elif k == 'penalty':
                    param[k] =  [i for i in custom_params[k].split(',')]
                elif k == 'solver':
                    param[k] = [i for i in custom_params[k].split(',')]
                elif k == 'tol':
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                elif k == 'verbose':
                    if custom_params[k]:
                        param[k] = [True]
                    else:
                        param[k] = [False]
                elif k == 'warm_start':
                    if custom_params[k]:
                        param[k] = [True]
                    else:
                        param[k] = [False] 

            model = clf.create_model(algo)
            tuned_model = tune_model(svm_model, custom_grid=param)

    elif algo=='knn':
        if model_type == 'custom':
            custom_params = custom_params.replace('"false"', 'false')
            custom_params = custom_params.replace('"true"', 'true')
            custom_params = custom_params.replace('"null"', 'null')
            custom_params = custom_params.replace('"None"', 'null')
            custom_params=json.loads(custom_params)
            model = clf.create_model(algo, **custom_params,)

        elif model_type == 'tune':
            param= {}
            custom_params = json.loads(custom_params)

            for k in custom_params.keys():
                if k == 'algorithm':
                    param[k] = [i for i in custom_params[k].split(',')]            
                elif k == "leaf_size":
                    param[k] =  [ int(i) for i in custom_params[k].split(',')]
                elif k == "metric":
                    param[k] = [i for i in custom_params[k].split(',')]  
                elif k == 'n_neighbors':
                    param[k] = [int(i) for i in custom_params[k].split(',')]
                elif k == 'p':
                    param[k] =  [int(i) for i in custom_params[k].split(',')]
                elif k == 'weights':
                    param[k] =  [i for i in custom_params[k].split(',')] 

            model = clf.create_model(algo)
            tuned_model = tune_model(model, custom_grid=param)

    elif algo=='nb':
        if model_type == 'custom':
            param= {}
            custom_params = custom_params.replace('"false"', 'false')
            custom_params = custom_params.replace('"true"', 'true')
            custom_params = custom_params.replace('"null"', 'null')
            custom_params = custom_params.replace('"None"', 'null')
            custom_params=json.loads(custom_params)

            for i in custom_params.keys():
                if i == "priors":
                    if custom_params[i] == "None":
                        param[i] = None
                    else:
                        param[i] = [float(i) for i in custom_params[i].split(',')]  
                elif i == "var_smoothing":
                    param[i] = custom_params[i]

            model = clf.create_model(algo, **custom_params,)

        elif model_type == 'tune':
            param= {}
            custom_params = json.loads(custom_params)

            for i in custom_params.keys():
                if i == "priors":
                    if custom_params[i] == "None":
                        param[i] = None
                    else:
                        param[i] = [float(i) for i in custom_params[i].split(',')]
                        
                elif i == "var_smoothing":
                    param[i] = [float(i) for i in custom_params[i].split(',')]

            model = clf.create_model(algo)
            tuned_model = tune_model(model, custom_grid=param)

    elif algo=='xboost':
        if model_type == 'custom':
            param= {}
            custom_params = custom_params.replace('"false"', 'false')
            custom_params = custom_params.replace('"true"', 'true')
            custom_params = custom_params.replace('"null"', 'null')
            custom_params = custom_params.replace('"None"', 'null')
            custom_params = custom_params.replace('"None"', 'null')
            custom_params =  custom_params.replace("__ob__", "[")
            custom_params =  custom_params.replace("__cb__", "[")
            custom_params=json.loads(custom_params)
            
            for i in custom_params.keys():
                if i == "priors":
                    if custom_params[i] == "None":
                        param[i] = None
                    else:
                        param[i] = [float(i) for i in custom_params[i].split(',')]  
                elif i == "var_smoothing":
                    param[i] = custom_params[i]

            model = clf.create_model(algo, **custom_params,)

        elif model_type == 'tune':
            param= {}
            custom_params = json.loads(custom_params)

            for i in custom_params.keys():
                if i ==  "objective" :
                    param[i] = [i for i in custom_params[i].split(',')] 
                elif i ==  "base_score" :
                    param[i] = [float(i) for i in custom_params[i].split(',')] 
                elif i ==  "booster" :
                    param[i] = [i for i in custom_params[i].split(',')] 
                elif i ==  "colsample_bylevel" :
                    param[i] = [float(i) for i in custom_params[i].split(',')] 
                elif i ==  "colsample_bynode" :
                    param[i] = [float(i) for i in custom_params[i].split(',')] 
                elif i ==  "colsample_bytree" :
                    param[i] = [float(i) for i in custom_params[i].split(',')] 
                elif i ==  "early_stopping_rounds" :
                    param[i] = [int(i) for i in custom_params[i].split(',')] 
                elif i ==  "enable_categorical" :
                    param[i] = [custom_params[i]] 
                elif i ==  "eval_metric" :
                    param[i] = [i for i in custom_params[i].split(',')] 
                elif i ==  "gamma" :
                    param[i] = [float(i) for i in custom_params[i].split(',')] 
                elif i ==  "grow_policy" :
                    param[i] = [i for i in custom_params[i].split(',')] 
                elif i ==  "importance_type" :
                    param[i] = [i for i in custom_params[i].split(',')] 
                elif i ==  "interaction_constraints" :
                    param[i] = [i for i in custom_params[i].split(',')] 
                elif i ==  "learning_rate" :
                    param[i] = [float(i) for i in custom_params[i].split(',')] 
                elif i ==  "max_bin" :
                    param[i] = [int(i) for i in custom_params[i].split(',')] 
                elif i ==  "max_delta_step" :
                    param[i] = [float(i) for i in custom_params[i].split(',')] 
                elif i ==  "max_depth" :
                    param[i] = [int(i) for i in custom_params[i].split(',')] 
                elif i ==  "max_leaves" :
                    param[i] = [int(i) for i in custom_params[i].split(',')] 
                elif i ==  "min_child_weight" :
                    param[i] = [int(i) for i in custom_params[i].split(',')] 
                elif i ==  "multi_strategy" :
                    param[i] = [i for i in custom_params[i].split(',')] 
                elif i ==  "n_estimators" :
                    param[i] = [int(i) for i in custom_params[i].split(',')] 
                elif i ==  "num_parallel_tree" :
                    param[i] = [int(i) for i in custom_params[i].split(',')] 
                elif i ==  "random_state" :
                    param[i] = [int(i) for i in custom_params[i].split(',')]  
                elif i ==  "reg_alpha" :
                    param[i] = [float(i) for i in custom_params[i].split(',')] 
                elif i ==  "reg_lambda" :
                    param[i] = [float(i) for i in custom_params[i].split(',')] 
                elif i ==  "sampling_method" :
                    param[i] = [i for i in custom_params[i].split(',')] 
                elif i ==  "scale_pos_weight" :
                    param[i] = [float(i) for i in custom_params[i].split(',')] 
                elif i ==  "subsample" :
                    param[i] = [float(i) for i in custom_params[i].split(',')] 
                elif i ==  "tree_method" :
                    param[i] = [i for i in custom_params[i].split(',')] 

            print(param)

            model = clf.create_model(algo)
            tuned_model = tune_model(model, custom_grid=param)

    elif algo=='lightgbm':
        if model_type == 'custom':
            param= {}
            custom_params = custom_params.replace('"false"', 'false')
            custom_params = custom_params.replace('"true"', 'true')
            custom_params = custom_params.replace('"null"', 'null')
            custom_params = custom_params.replace('"None"', 'null')
            custom_params = custom_params.replace('"None"', 'null')
            custom_params =  custom_params.replace("__ob__", "[")
            custom_params =  custom_params.replace("__cb__", "[")
            custom_params=json.loads(custom_params)

            model = clf.create_model(algo, **custom_params,)

        elif model_type == 'tune':
            param = {}

            for k in custom_params.keys():
                if k == 'boosting_type':
                    param[k] = [i for i in custom_params[k].split(',')]
                elif k == 'class_weight':
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                elif k == 'colsample_bytree':
                    param[k] = [float(i) for i in custom_params[k].split(',')]     
                elif k == 'importance_type':
                    param[k] = [i for i in custom_params[k].split(',')]
                elif k == 'learning_rate':
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                elif k == 'max_depth':
                    param[k] = [int(i) for i in custom_params[k].split(',')]
                elif k == 'min_child_samples':
                    param[k] = [int(i) for i in custom_params[k].split(',')]
                elif k == 'min_child_weight':
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                elif k == 'min_split_gain':
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                elif k == 'n_estimators':
                    param[k] = [int(i) for i in custom_params[k].split(',')]
                elif k ==  'num_leaves':
                    param[k] = [int(i) for i in custom_params[k].split(',')]
                elif k =='objective':
                    param[k] = [i for i in custom_params[k].split(',')]               
                elif k =='random_state':
                    param[k] = [int(i) for i in custom_params[k].split(',')]      
                elif k =='reg_alpha':
                    param[k] = [float(i) for i in custom_params[k].split(',')]      
                elif k =='reg_lambda':
                    param[k] = [float(i) for i in custom_params[k].split(',')]      
                elif k =='subsample':
                    param[k] = [float(i) for i in custom_params[k].split(',')]   
                elif k =='subsample_for_bin':
                    param[k] = [int(i) for i in custom_params[k].split(',')] 
                elif k =='subsample_freq':
                    param[k] = [int(i) for i in custom_params[k].split(',')] 
            
            print( custom_params)

            model = clf.create_model(algo)
            model = tune_model(model, custom_grid=param)
    



            # lr        sklearn.linear_model._logistic.LogisticRegression   True  
            # knn       sklearn.neighbors._classification.KNeighborsCl...   True  
            # nb                           sklearn.naive_bayes.GaussianNB   True    
            # dt             sklearn.tree._classes.DecisionTreeClassifier   True  
            # svm       sklearn.linear_model._stochastic_gradient.SGDC...   True  
            # rbfsvm                             sklearn.svm._classes.SVC  False  
            # gpc       sklearn.gaussian_process._gpc.GaussianProcessC...  False  
            # mlp       sklearn.neural_network._multilayer_perceptron....  False  
            # ridge           sklearn.linear_model._ridge.RidgeClassifier   True  
            # rf          sklearn.ensemble._forest.RandomForestClassifier   True  
            # qda       sklearn.discriminant_analysis.QuadraticDiscrim...   True  
            # ada       sklearn.ensemble._weight_boosting.AdaBoostClas...   True  
            # gbc         sklearn.ensemble._gb.GradientBoostingClassifier   True  
            # lda       sklearn.discriminant_analysis.LinearDiscrimina...   True  
            # et            sklearn.ensemble._forest.ExtraTreesClassifier   True  
            # xgboost                       xgboost.sklearn.XGBClassifier   True  
            # lightgbm                    lightgbm.sklearn.LGBMClassifier   True  
            # catboost                   catboost.core.CatBoostClassifier   True  
            # dummy                         sklearn.dummy.DummyClassifier   True  

    elif algo=='dt':
        if model_type == 'custom':
            param= {}
            custom_params = custom_params.replace('"false"', 'false')
            custom_params = custom_params.replace('"true"', 'true')
            custom_params = custom_params.replace('"null"', 'null')
            custom_params = custom_params.replace('"None"', 'null')
            custom_params = custom_params.replace('"None"', 'null')
            custom_params =  custom_params.replace("__ob__", "[")
            custom_params =  custom_params.replace("__cb__", "[")
            custom_params=json.loads(custom_params)

            model = clf.create_model(algo, **custom_params,)

        elif model_type == 'tune':
            param = {}

            custom_params=json.loads(custom_params)

            for k in custom_params.keys():
                if k == 'ccp_alpha':
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                elif k == 'class_weight':
                    if custom_params[k] == "None" :
                        param[k] = [None]
                    else:
                        param[k] = [float(i) for i in custom_params[k].split(',')]
                elif k == 'criterion':
                    param[k] = [i for i in custom_params[k].split(',')]    
                elif k == 'max_depth':
                    if custom_params[k] == "None" :
                        param[k] = [None]
                    else:
                        param[k] = [float(i) for i in custom_params[k].split(',')]
                elif k == 'max_features':
                    if custom_params[k] == "None" :
                        param[k] = [None]
                    else:
                        param[k] = [float(i) for i in custom_params[k].split(',')]
                elif k == 'max_leaf_nodes':
                    if custom_params[k] == "None" :
                        param[k] = [None]
                    else:
                        param[k] = [float(i) for i in custom_params[k].split(',')]
                elif k == 'min_impurity_decrease':
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                elif k == 'min_samples_leaf':
                    param[k] = [int(i) for i in custom_params[k].split(',')]
                elif k ==  'min_samples_split':
                    param[k] = [int(i) for i in custom_params[k].split(',')]
                elif k =='min_weight_fraction_leaf':
                    param[k] = [float(i) for i in custom_params[k].split(',')]               
                elif k =='monotonic_cst':
                    if custom_params[k] == "None" :
                        param[k] = [None]
                    else:
                        param[k] = [float(i) for i in custom_params[k].split(',')]      
                elif k =='random_state':
                    param[k] = [int(i) for i in custom_params[k].split(',')]      
                elif k =='splitter':
                    param[k] = [i for i in custom_params[k].split(',')]      
    
            model = clf.create_model(algo)
            model = tune_model(model, custom_grid=param)

    elif algo=='rbfsvm':
        if model_type == 'custom':
            param= {}
            custom_params = custom_params.replace('"false"', 'false')
            custom_params = custom_params.replace('"true"', 'true')
            custom_params = custom_params.replace('"null"', 'null')
            custom_params = custom_params.replace('"None"', 'null')
            custom_params =  custom_params.replace("__ob__", "[")
            custom_params =  custom_params.replace("__cb__", "[")
            custom_params=json.loads(custom_params)

            model = clf.create_model(algo, **custom_params,)

        elif model_type == 'tune':
            param = {}
            custom_params=json.loads(custom_params)

            for k in custom_params.keys():
                if k == 'C':
                    print(custom_params[k])
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                if k ==  'break_ties':
                    param[k] = [custom_params[k]]
                if k == 'cache_size':
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                if k == 'class_weight':
                    if custom_params[k] == "None" :
                        param[k] = [None]
                    else:
                        param[k] = [float(i) for i in custom_params[k].split(',')]   
                if k ==  'coef0':
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                if k ==  'decision_function_shape':
                    param[k] = [i for i in custom_params[k].split(',')]
                if k == 'degree':
                    param[k] = [i for i in custom_params[k].split(',')]
                if k == 'gamma':
                    param[k] = [i for i in custom_params[k].split(',')]
                if k == 'kernel':
                    param[k] = [i for i in custom_params[k].split(',')]
                if k == 'max_iter':
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                if k == 'probability':
                    param[k] = [custom_params[k]]
                if k ==  'random_state':
                    param[k] = [int(i) for i in custom_params[k].split(',')]
                if k ==  'shrinking':
                    param[k] = [custom_params[k]]
                if k == 'tol':
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                if k == 'verbose':
                    param[k] =  [custom_params[k]]

    elif algo=='rf':

        custom_params = custom_params.replace('"false"', 'false')
        custom_params = custom_params.replace('"true"', 'true')
        custom_params = custom_params.replace('"null"', 'null')
        custom_params = custom_params.replace('"None"', 'null')
        custom_params = custom_params.replace('None', 'null')
        custom_params =  custom_params.replace("__ob__", "[")
        custom_params =  custom_params.replace("__cb__", "[")
        custom_params=json.loads(custom_params)

        if model_type == 'custom':
        
            model = clf.create_model(algo, **custom_params,)

        elif model_type == 'tune':
            param = {}

            for k in custom_params.keys():
                if k == 'bootstrap':
                    param[k] = [custom_params[k]]
                if k ==   'ccp_alpha':
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                if k == 'class_weight':
                    if custom_params[k] :
                        param[k] = [float(i) for i in custom_params[k].split(',')]  
                    else:
                        param[k] = [None]
                if k ==  'criterion':
                    param[k] = [i for i in custom_params[k].split(',')]
                if k ==  'max_depth':
                    if custom_params[k] :
                        param[k] = [float(i) for i in custom_params[k].split(',')]  
                    else:
                        param[k] = [None] 
                if k == 'max_features':
                    param[k] = [i for i in custom_params[k].split(',')]
                if k == 'max_leaf_nodes':
                    if custom_params[k] :
                        param[k] = [float(i) for i in custom_params[k].split(',')]  
                    else:
                        param[k] = [None] 
                if k ==  'max_samples':
                    if custom_params[k] :
                        param[k] = [float(i) for i in custom_params[k].split(',')]  
                    else:
                        param[k] = [None]  
                if k ==  'min_impurity_decrease':
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                if k ==  'min_samples_leaf':
                    param[k] =  [int(i) for i in custom_params[k].split(',')]
                if k ==   'min_samples_split':
                    param[k] = [int(i) for i in custom_params[k].split(',')]
                if k ==   'min_weight_fraction_leaf':
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                if k == 'monotonic_cst':
                    if custom_params[k] :
                        param[k] = [float(i) for i in custom_params[k].split(',')]  
                    else:
                        param[k] = [None] 
                if k ==  'n_estimators':
                    param[k] =   [int(i) for i in custom_params[k].split(',')]
                if k ==  'oob_score':
                    param[k] =  [custom_params[k]]
                if k ==   'random_state':
                    param[k] =   [int(i) for i in custom_params[k].split(',')]
                if k ==   'verbose':
                    param[k] =  [custom_params[k]]
                if k ==   'warm_start':
                    param[k] =  [custom_params[k]]

    elif algo=='gpc':
        custom_params = custom_params.replace('"false"', 'false')
        custom_params = custom_params.replace('"true"', 'true')
        custom_params = custom_params.replace('"null"', 'null')
        custom_params = custom_params.replace('"None"', 'null')
        custom_params = custom_params.replace('None', 'null')
        custom_params =  custom_params.replace("__ob__", "[")
        custom_params =  custom_params.replace("__cb__", "[")
        custom_params=json.loads(custom_params)

        if model_type == 'custom':
        
            model = clf.create_model(algo, **custom_params,)

        elif model_type == 'tune':
            param = {}

            for k in custom_params.keys():
                if k == 'bootstrap':
                    param[k] = [custom_params[k]]
                if k ==   'ccp_alpha':
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                if k == 'class_weight':
                    if custom_params[k] :
                        param[k] = [float(i) for i in custom_params[k].split(',')]  
                    else:
                        param[k] = [None]
                if k ==  'criterion':
                    param[k] = [i for i in custom_params[k].split(',')]
                if k ==  'max_depth':
                    if custom_params[k] :
                        param[k] = [float(i) for i in custom_params[k].split(',')]  
                    else:
                        param[k] = [None] 
                if k == 'max_features':
                    param[k] = [i for i in custom_params[k].split(',')]
                if k == 'max_leaf_nodes':
                    if custom_params[k] :
                        param[k] = [float(i) for i in custom_params[k].split(',')]  
                    else:
                        param[k] = [None] 
                if k ==  'max_samples':
                    if custom_params[k] :
                        param[k] = [float(i) for i in custom_params[k].split(',')]  
                    else:
                        param[k] = [None]  
                if k ==  'min_impurity_decrease':
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                if k ==  'min_samples_leaf':
                    param[k] =  [int(i) for i in custom_params[k].split(',')]
                if k ==   'min_samples_split':
                    param[k] = [int(i) for i in custom_params[k].split(',')]
                if k ==   'min_weight_fraction_leaf':
                    param[k] = [float(i) for i in custom_params[k].split(',')]
                if k == 'monotonic_cst':
                    if custom_params[k] :
                        param[k] = [float(i) for i in custom_params[k].split(',')]  
                    else:
                        param[k] = [None] 
                if k ==  'n_estimators':
                    param[k] =   [int(i) for i in custom_params[k].split(',')]
                if k ==  'oob_score':
                    param[k] =  [custom_params[k]]
                if k ==   'random_state':
                    param[k] =   [int(i) for i in custom_params[k].split(',')]
                if k ==   'verbose':
                    param[k] =  [custom_params[k]]
                if k ==   'warm_start':
                    param[k] =  [custom_params[k]]




if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run PyCaret ML setup.')
    parser.add_argument('--algo', type=str, required=False, help='path to data file')
    parser.add_argument("--user_define_hyper_para", required=False, default='log', help="The loss function to be used. Defaults to 'hinge', which gives a linear SVM. The possible options are 'hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', or a regression loss: 'squared_loss', 'huber', 'epsilon_insensitive', or squared_epsilon_insensitive'.")
    parser.add_argument('--data_file', type=str, required=True, help='path to data file')
    parser.add_argument('--target', type=str, required=False, help='Target column for prediction')
    parser.add_argument('--session_id', type=int, default=123, help='Session ID for reproducibility')
    parser.add_argument('--feature_selection', type=bool, default=False, help='Whether to perform feature selection')
    parser.add_argument('--fold', type=int, default=10, help='Number of folds for cross-validation')
    parser.add_argument('--log_experiment', type=bool, default=False, help='Whether to log experiment')
    parser.add_argument('--train_size', type=float, required=False, default=0.7, help='Proportion of dataset to include in the train split')
    parser.add_argument('--data_split_shuffle', type=bool, required=False, default=True, help='Whether to shuffle data before splitting')
    parser.add_argument('--data_split_stratify', type=bool, required=False, default=False, help='Whether to stratify data during splitting')
    parser.add_argument('--normalize', type=bool, default=False, required=False,  help='Whether to normalize the data')
    parser.add_argument('--transformation', type=bool, required=False, default=False, help='Whether to apply power transformation')
    parser.add_argument('--remove_outliers', type=bool, required=False,  default=False, help='Whether to remove outliers from the data')
    parser.add_argument('--outliers_threshold', type=float, required=False, default=0.05, help='Threshold for removing outliers')
    parser.add_argument('--pca', type=bool, required=False, default=False, help='Whether to apply PCA for dimensionality reduction')
    parser.add_argument('--bin_numeric_features', nargs='+', required=False, help='List of numeric features to bin into discrete intervals')
    parser.add_argument('--remove_multicollinearity', type=bool, required=False, default=False, help='Whether to remove multicollinear features')
    parser.add_argument('--multicollinearity_threshold', type=float,required=False,  default=0.9, help='Threshold for removing multicollinear features')
    parser.add_argument('--log_data', type=bool, required=False, default=False, help='Whether to apply log transformation to the data')
    parser.add_argument('--polynomial_features', type=bool, required=False, default=False, help='Whether to create polynomial features')
    # parser.add_argument('--test', type=str, required=True, default=False, help='Whether to create polynomial features')
    parser.add_argument('--model_type', required=True, default='custom', help='Whether to create polynomial features')
    args = parser.parse_args()

    # run_pycaret(
    #     algo=args.algo,
    #     file_path=args.data_file,
    #     target=args.target,
    #     session_id=args.session_id,
    #     feature_selection=args.feature_selection,
    #     fold=args.fold,
    #     log_experiment=args.log_experiment,
    #     train_size=args.train_size,
    #     data_split_shuffle=args.data_split_shuffle,
    #     data_split_stratify=args.data_split_stratify,
    #     normalize=args.normalize,
    #     transformation=args.transformation,
    #     remove_outliers=args.remove_outliers,
    #     outliers_threshold=args.outliers_threshold,
    #     pca=args.pca,
    #     bin_numeric_features=args.bin_numeric_features,
    #     remove_multicollinearity=args.remove_multicollinearity,
    #     multicollinearity_threshold=args.multicollinearity_threshold,
    #     log_data=args.log_data,
    #     polynomial_features=args.polynomial_features,
    #     custom_hyperparams=args.user_define_hyper_para,
    #     test=args.test
    #     )
    # print(args.user_define_hyper_para)
    # run_pycaret(algo=args.algo, custom_hyperparams=args.user_define_hyper_para)
    # if len(sys.argv) > 1:
    #     custom_hyper_para = {}
    # # for i in args.user_define_hyper_para.split(';'): 
    #     run_pycaret(args.algo, custom_hyperparams=args.user_define_hyper_para, target=args.target)
    # else:
    #     print("Select... one of the algorithms from this list as a first argument.. ")

    run_pycaret(custom_params = args.user_define_hyper_para, algo=args.algo, model_type=args.model_type, file_path=args.data_file)
