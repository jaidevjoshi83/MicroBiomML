

import matplotlib.pyplot as plt
import seaborn as sns, os
import pandas as pd


def HTML_Gen(html):

    out_html = open(html,'w')             
    part_1 =  """

    <!DOCTYPE html>
    <html lang="en">
    <head>
      <title>Bootstrap Example</title>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1">
      <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/css/bootstrap.min.css">
      <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.0/jquery.min.js"></script>
      <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.4.0/js/bootstrap.min.js"></script>
    <body>
    <style>
    div.container_1 {
      width:600px;
      margin: auto;
     padding-right: 10; 
    }
    div.table {
      width:600px;
      margin: auto;
     padding-right: 10; 
    }
    </style>
    </head>
    <div class="jumbotron text-center">
      <h1> Machine Learning Algorithm Assessment Report </h1>
    </div>
    <div class="container">
      <div class="row">
        <div class="col-sm-4">
          <img src="1.png" alt="Smiley face" height="500" width="400">
        </div>

      </div>
    </div>
    </body>
    </html>
    """ 
    out_html.write(part_1)
    out_html.close()

def HeatMapPlot(Infile,  FigHight, FigWidth, Rotation, Workdirpath, htmlOutDir, htmlFname):

    if not os.path.exists(htmlOutDir):
        os.makedirs(htmlOutDir)

    df  = pd.read_csv(Infile, sep="\t")
    #print (df.columns[1:])

    plt.figure(figsize=(float(FigHight),float(FigWidth)))

    x_axis_labels = df.columns[1:]
    y_axis_labels = df[df.columns[0]].tolist()

    sns.set(font_scale=2)
    sns.heatmap(df[df.columns[1:]], center=0, vmin=0, vmax=1, yticklabels=y_axis_labels, xticklabels=x_axis_labels)

    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    plt.savefig(os.path.join(Workdirpath, htmlOutDir, "1.png"))

    HTML_Gen(os.path.join(Workdirpath, htmlOutDir, htmlFname))

    #plt.show()


if __name__=="__main__":

    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-I", "--in_file",
                        required=True,
                        default=None,
                        help="Path to target tsv file")

    #parser.add_argument("-O", "--Out_file",
    #                    required=False,
    #                    default='Out_file_data.tsv',
    #                    help="Path to target tsv file")

    parser.add_argument("-H", "--FigHight",
                        required=False,
                        default=4,
                        help="Path to target tsv file")


    parser.add_argument("-W", "--FigWidth",
                        required=False,
                        default=6,
                        help="Path to target tsv file")

    parser.add_argument("-R", "--Rotation",
                        required=False,
                        default=1,  
                        help="Path to target tsv file")

    parser.add_argument("--htmlOutDir", 
                        required=False, 
                        default=os.path.join(os.getcwd(),'report_dir'), 
                        help="HTML Out Dir")

    parser.add_argument("--htmlFname", 
                        required=False, 
                        help="HTML out file", 
                        default="jai.html")

    parser.add_argument("--Workdirpath", 
                        required=False, 
                        default=os.getcwd(), 
                        help="Working Directory Path")

    args = parser.parse_args()

    #biom_main(args.in_file, args.Out_file_data, args.Out_file_Taxonomy, args.skiprows)

    HeatMapPlot(args.in_file,  args.FigHight, args.FigWidth, args.Rotation, args.Workdirpath, args.htmlOutDir, args.htmlFname)


