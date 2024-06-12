import glob, os

out_file = open('./merged_results.tsv', 'w')

header = ['Acc_LRC','F1_LRC','Prec_LRC','Recall_LRC','MCC_LRC', 'Acc_DTC','F1_DTC','Prec_DTC','Recall_DTC','MCC_DTC', 'Acc_SVC','F1_SVC','Prec_SVC','Recall_SVC','MCC_SVC', 'Acc_RFC','F1_RFC','Prec_RFC','Recall_RFC','MCC_RFC','Acc_HDC','F1_HDC','Prec_HDC','Recall_HDC','MCC_HDC']
header = '\t'.join(header)

out_file.write('name'+'\t'+header+'\n')

def return_hdc_data(file_name):
    f = open('./MicroBiomML/hdlib-ra-levels1000-retrain20-cv5-feature-selection/HDC_result_files/'+file_name+'.txt')
    lines = f.readlines()
    for i, line in enumerate(lines):
        if 'Total elapsed time:' in line:
            result = [f"{float(lines[i-5].split(' ')[1]):.2f}",  f"{float(lines[i-4].split(' ')[1]):.2f}", f"{float(lines[i-3].split(' ')[1]):.2f}", f"{float(lines[i-2].split(' ')[1]):.2f}", f"{float(lines[i-1].split(' ')[3]):.2f}"]
    return result

fs = glob.glob('./datasets-ra/*.tsv')
fex = open('FilesToExclude.txt')

files_to_exclude = [ f.split('/')[2].rstrip('\n') for f in fex.readlines()]
data_files = [f.split('/')[2] for f in fs]

algs = ['LRC', 'DTC', 'SVC', 'RFC']
#Acc,F1,Prec.,Recall,MCC

for d in data_files:
    if d not in files_to_exclude:
        # print(d)
        result_list = []
        print("#######################")
        for alg in algs:
            data =  open(os.path.join('./', alg+"_results", 'result.tsv'))
            lines = data.readlines()
            print(alg)
            for i in lines:
                if d in i:
                    result_list = result_list + i.rstrip('\n').split(',')[1:]
        result_list =  [f"{float(v):.2f}" for v in result_list]
        
        result = return_hdc_data(d)
        out_file.write(d+'\t'+'\t'.join(result_list+result)+'\n')
