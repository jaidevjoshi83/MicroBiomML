
import pandas as pd


def DF_main(in_files, out_file):


    df = pd.read_csv(in_files, sep="\t")
    df = df.T
    df.to_csv(out_file, sep='\t', index=None)


if __name__=='__main__':


    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-I", "--in_file",
                        required=True,
                        default=None,
                        help="Path to target tsv file")

    parser.add_argument("-O", "--Out_file",
                        required=True,
                        default=None,
                        help="Path to target tsv file")
                        
                       
    args = parser.parse_args()
    DF_main(args.in_file,args.Out_file)


