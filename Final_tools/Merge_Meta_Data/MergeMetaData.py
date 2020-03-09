import pandas as pd 



def AddClassLabel(DataFile, MetaDataFile, IgnorIndex, Out_file):

    out_list = []

    file = open(MetaDataFile)
    lines = file.readlines()[1:]

    df1  = pd.read_csv(DataFile, sep="\t")
    ls = df1['#OTU ID'].tolist()

    for line in lines:
        a = line.split('\t')[0]
        for l in ls:
            if a == l:
                out_list.append(int(line.split('\t')[2].strip('\n')))

    p = pd.DataFrame(out_list, columns=['class'])
    
    if IgnorIndex == "yes":
        df1 = df1[df1.columns.tolist()[1:]]
    else:
        pass

    fdf = pd.concat([df1,p], axis=1)


    fdf.to_csv(Out_file, sep="\t", index=None)

    

if __name__=='__main__':


    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-D", "--DataFile",
                        required=True,
                        default=None,
                        help="Data file name")

    parser.add_argument("-M", "--MetaDataFile",
                        required=True,
                        default=None,
                        help="Metadata file")

    parser.add_argument("-R", "--IgnorIndex",
                        required=False,
                        default="yes",
                        help="Ignore index ")

    parser.add_argument("-O", "--Outfile",
                        required=True,
                        default=None,
                        help="Out file ")
                        
                       
    args = parser.parse_args()
    AddClassLabel(args.DataFile, args.MetaDataFile, args.IgnorIndex, args.Outfile )






