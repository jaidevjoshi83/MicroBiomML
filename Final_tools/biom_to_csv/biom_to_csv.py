import sys  
import os
import glob


def biom_main(in_files,out_file_dir):


    os.environ['f'] = in_files
    os.environ['o'] = out_file_dir
    os.system('biom convert -i $f -o $o --to-tsv ') 


if __name__=='__main__':


    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-I", "--in_file",
                        required=True,
                        default=None,
                        help="Path to target tsv file")

    parser.add_argument("-O", "--Out_dir",
                        required=True,
                        default=None,
                        help="Path to target tsv file")
                        
                       
    args = parser.parse_args()
    biom_main(args.in_file,args.Out_dir)


