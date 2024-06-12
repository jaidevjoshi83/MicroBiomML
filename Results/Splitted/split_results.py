f = open('merged_results.tsv')

lines = f.readlines()
category =  list(set([ line.split('\t')[0].split('.')[3] for line in lines[1:] ]))

'LRC', 'DTC', 'LRC', 'RFC'

header = ['Acc_LRC','F1_LRC','Prec_LRC','Recall_LRC','MCC_LRC', 'Acc_DTC','F1_DTC','Prec_DTC','Recall_DTC','MCC_DTC', 'Acc_SVC','F1_SVC','Prec_SVC','Recall_SVC','MCC_SVC', 'Acc_RFC','F1_RFC','Prec_RFC','Recall_RFC','MCC_RFC','Acc_HDC','F1_HDC','Prec_HDC','Recall_HDC','MCC_HDC']
header = '\t'.join(header)


for c in category:
    out = open(c+'.tsv', 'w')
    out.write('name'+'\t'+header+'\n')

    for l in lines:
        if c in l:
            rename_list = [l.split('\t')[0].split('.')[1]] + l.split('\t')[1:]

            # print("\t".join(rename_list))
            out.write("\t".join(rename_list))
            # print(l)
out.close()
f.close()
   