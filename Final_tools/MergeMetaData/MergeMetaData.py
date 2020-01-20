import pandas as pd


def MergeDataFrame(DataFile, MetaDataFile, MataDataColumns, OutFile):

    dtafile = pd.read_csv(DataFile, sep="\t")
    MetaDataFile = pd.read_csv(MetaDataFile, sep="\t")

    MataDataColumns = MataDataColumns.split(',')

    MetaDataFile = MetaDataFile[MataDataColumns]

    df  = pd.concat([dtafile, MetaDataFile], axis=1)
    print (df)
    df.to_csv(OutFile, sep='\t', index=False)


if __name__=="__main__":

    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-I", "--in_file",
                        required=True,
                        default=None,
                        help="Input file")

    parser.add_argument("-M", "--MetaDataFile",
                        required=True,
                        default=None,
                        help="In put MetaDataFile")

    parser.add_argument("-C", "--MetaDataColumn",
                        required=True,
                        default=None,
                        help="Input columns")

    parser.add_argument("-O", "--OutFile",
                        required=True,
                        default='Out.tsv',
                        help="Output Files")          
                       
    args = parser.parse_args()

    MergeDataFrame(args.in_file, args.MetaDataFile, args.MetaDataColumn, args.OutFile)





