

def MergeDataFrame(DataFile, MetaDataFile,  OutFile):

 
    metadatafile = open(MetaDataFile)
    lines1 = metadatafile.readlines()

    datafile = open(DataFile)

    lines = datafile.readlines()
    first_line =  lines[0]
    all_lines = lines[1:]

    outfile = open(OutFile,'w')

    #print "\t".join(first_line.split('\t')[1:])

    outfile.write("\t".join(first_line.split('\t')[1:]).strip('\n')+'\t'+'class_label'+'\n')

    for line1 in lines1:
        lines  = datafile.readlines()[1:]
 
        for line in all_lines:
            if line1.split('\t')[0] == line.split('\t')[0]:
                outfile.write("\t".join(line.split('\t')[1:]).strip('\n')+'\t'+line1.split('\t')[1])
                #print "\t".join(line.split('\t')[1:]).strip('\n')+'\t'+line1.split('\t')[1]

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

    parser.add_argument("-O", "--OutFile",
                        required=True,
                        default='Out.tsv',
                        help="Output Files")          
                       
    args = parser.parse_args()

    MergeDataFrame(args.in_file, args.MetaDataFile, args.OutFile)