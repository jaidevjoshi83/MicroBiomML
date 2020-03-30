import glob
import pandas as pd 
import sys

files = sys.argv[1]
out_file = sys.argv[2]


data_frame = pd.read_csv(files.split(',')[0],sep='\t')


for file in files.split(',')[1:]: 

    df1 = pd.read_csv(file,sep='\t')
    data_frame =  pd.concat([data_frame,df1])

final_DF = data_frame.fillna(0)

print (final_DF)

final_DF.to_csv(out_file,sep="\t", index=False)









