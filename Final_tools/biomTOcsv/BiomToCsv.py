import sys  
import os
import glob


def biom_main(in_files,out_file_dir):


    if not os.path.exists(out_file_dir):
            os.makedirs(out_file_dir)

    for i,in_file in enumerate(in_files.split(',')):

        #print ("in_files")

        os.environ['f'] = in_file
        #print ('%s_out.tsv'%in_file.replace('.biom',''))
        os.environ['o'] = os.path.join(out_file_dir,'%i_out.tsv'%i)
        #os.environ['o'] = os.path.join(out_file_dir,'%s_out.tsv'%in_file.replace('.biom',''))
        #os.environ['o'] = os.path.join(out_file_dir,in_file.replace('.biom','_out.tsv'))
        os.system('biom convert -i $f -o $o --to-tsv --header-key taxonomy') 


if __name__=='__main__':


    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-I", "--in_file",
                        required=True,
                        default=None,
                        help="Path to target tsv file")

    parser.add_argument("-O", "--Out_dir",
                        required=True,
                        default='Out_dir',
                        help="Path to target tsv file")
                        
                       
    args = parser.parse_args()
    biom_main(args.in_file,args.Out_dir)


