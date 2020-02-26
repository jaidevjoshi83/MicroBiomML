import pandas as pd 
import sys  
import os
import glob

def RetriveData(DataSet, GetProp, OutFile):

	#p = {'Age'	:'Age.tsv',
	#	'Cancer':'Cancer.tsv',
	#	'Geo'	:'Geo.tsv'
	#	}

	#DataFilePath = os.path.join('DATASETS', p[DataSet])

	df = pd.read_csv(DataSet, sep='\t')

	if GetProp == 'FullData':
		df.to_csv(OutFile,index=None, sep='\t')

	elif GetProp == 'OTUs': 

		df = df.drop(['#SampleID','class_label','Var'], axis=1)
		df.to_csv(OutFile,index=None, sep='\t')

	elif GetProp == "MetaData":

		df = df[['class_label','Var']]
		df.to_csv(OutFile,index=None, sep='\t')

	else:
		print ("Enter Correct Option")

if __name__=='__main__':

    import argparse
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-D", "--DataSet",
                        required=True,
                        default=None,
                        help="Path to target tsv file")

    parser.add_argument("-G", "--GetProperty",
                        required=True,
                        default=None,
                        help="Path to target tsv file")   

    parser.add_argument('-O', "--OutFile",
    					required=True,
    					default=None,
    					help="OutFile")     
                       
    args = parser.parse_args()
    RetriveData(args.DataSet,args.GetProperty, args.OutFile)




