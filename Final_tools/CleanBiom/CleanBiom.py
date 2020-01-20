import pandas as pd

def biom_main(in_file, Out_file_data, Out_file_Taxonomy,skiprows):

    df = pd.read_csv(in_file, sep='\t', skiprows=int(skiprows))
    data =  df.drop(['taxonomy'], axis=1)
    data = data.T

    taxonomy_OTS = df[['taxonomy']]
    data.to_csv(Out_file_data, sep='\t', index=None)
    taxonomy_OTS.to_csv(Out_file_Taxonomy,sep='\t', index=None)



if __name__=='__main__':


    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-I", "--in_file",
                        required=True,
                        default=None,
                        help="Path to target tsv file")

    parser.add_argument("-O_T", "--Out_file_data",
                        required=False,
                        default='Out_file_data.tsv',
                        help="Path to target tsv file")


    parser.add_argument("-O_D", "--Out_file_Taxonomy",
                        required=False,
                        default='Out_file_Taxonomy.tsv',
                        help="Path to target tsv file")


    parser.add_argument("-S", "--skiprows",
                        required=False,
                        default=1,
                        help="Path to target tsv file")

    args = parser.parse_args()
    biom_main(args.in_file, args.Out_file_data, args.Out_file_Taxonomy, args.skiprows)
