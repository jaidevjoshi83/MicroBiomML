import pandas as pd
import numpy as np
from pycaret.classification import ClassificationExperiment
import glob
from pycaret.classification import *
import sys, os
import time


fs = glob.glob('./datasets-ra/*.tsv')
ds = ['DTC', 'LRC', 'RFC', 'SVC']
file_names  = [f.split('/')[len(f.split('/'))-1] for f in fs ]

for d in ds:
    heading = ["Accuracy", "F1", "Prec.", "Recall", "MCC"] 
    result_file = open(os.path.join('./', d+"_results", 'result.tsv'), 'w')
    result_file.write("name"+","+",".join(heading)+'\n')
    error_log = open(os.path.join('./', d+"_results", 'error.log'), 'w')
    time_log = open(os.path.join('./', d+"_results", 'time.log'), 'w')
    # heading = ["Accuracy", "F1", "Prec.", "Recall", "MCC"] * 8
    for r in file_names:
        try:
            df = pd.read_csv(os.path.join('./', 'datasets-ra', r), sep="\t")
            # print(df)
            features_file = open(os.path.join('./', d, r))
            selected_features = features_file.readline()
            features = selected_features.rstrip('\n\\n').split('\t')
            df_new = df[list(features) + ['label']]
            start_time = time.time()
            s = ClassificationExperiment()
            # algos =   [  :'rf', 'gbc', 'dt', 'xgboost',  'lr',  'nb', 'svm']
            algos = {'DTC':'dt', 'LRC':'lr', 'RFC':'rf', 'SVC':'svm'}
            # algos =   [ 'rf']
            data_values = []
            data_values_name = []
            clf = setup(df_new, target = 'label', session_id = 123, feature_selection=False, n_jobs=20, fold=5,  log_experiment=False, use_gpu=True)
            result = clf.create_model(algos[d])
            print("Now running tuning")
            tuned_model = clf.tune_model(result)   
            results = clf.pull()
            lists = np.array(results)
            end_time = time.time()
            data_values.extend([str(lists[5][0]),str(lists[5][4]),str(lists[5][3]),str(lists[5][2]),str(lists[5][6])]) 
            result_file.write(r+","+",".join(data_values)+'\n')
            print("#######################################################")
            print(r)
            print("#######################################################")
            # result_file.close()
            duration = end_time - start_time
            time_log.write(r+'\t'+ str(duration)+'\n')
        except:
            error_log.write(os.path.join('./', d, r)+'\n')

    result_file.close()
    error_log.close()
    time_log.close()
    