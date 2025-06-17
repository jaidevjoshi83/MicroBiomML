import pandas as pd #pandas==2.2.3
import plotly.graph_objects as go #plotly==5.24.0
import os
import argparse


def Analysis(values, thr=0.05):
    # print(values)
    better = []
    comparable = []
    thr = 0.05 

    last_value = values[4]
    
    for v in values[0:4]:
        # print(v)
        better.append(round(last_value -v, 2) > thr)
        comparable.append(abs(round(last_value -v, 2)) <= thr)

    if all(better):
        return (True, 'better_all' )
    elif True in better:
        return (True, 'better_one' )
    elif all( comparable):
        return (True, 'Comp_with_all' )
    elif True in comparable:
        return (True, 'Comp_with_one' )
    

color_scale=[
    [0, 'green'],    # Value -1 will be red
    [0.5, 'red'], # Value 0 will be yellow
    [1, 'yellow']    # Value 1 will be blue
]

# Define the color scale constant
COLOR_SCALE = {
    'Comp_with_all': 'blue',
    'better_all': 'violet',
    'Comp_with_one': 'black',
    'better_one': 'red'
}
clm_list = [1, 6, 11, 16, 21]

def ResultSummary(file, threshold, column_list):
    print(file)
    new_DF = pd.read_csv(file, sep='\t')
    new_DF.set_index('name', inplace=True)

    DF = new_DF.T
    DF.columns = new_DF.index
    DF.index = new_DF.columns
    df = DF.iloc[column_list]

    column_anno_per = {}
    comparable = {}

    for n in df.columns.to_list():
        comparable[n] = Analysis(df[n].values, threshold)
    return comparable

def Plot(input_file, column_list, figure_size=(300, 400), saveSVG=False, file_type=None, color_labels='Greens', font_size=12, tick_font=14, tick_angle=-45, threshold = 0.05, outfile=None):
    
    column_list = [int(i) for i in column_list.split(',')]
    
    result_1 = ResultSummary(input_file, threshold, column_list)

    true_columns = []
    true_column_comp = []

    for i, k in enumerate(result_1.keys()):
        if result_1[k]:
            true_column_comp.append((i, result_1[k], k))

    plotting_columns = {
        'Comp_with_all': [],
        'better_all': [],
        'Comp_with_one': [],
        'better_one': [],
        'None': [],
    }

    colors = COLOR_SCALE
    arranged_columns = []
    counter = 0

    for c in colors.keys():
        for i, a in enumerate(true_column_comp):
            if c == a[1][1]:
                counter += 1
                plotting_columns[c].append((a[2], counter - 1))
                arranged_columns.append(a[2])

    # DF = pd.read_csv(input_file, sep='\t')
    # DF.set_index('name', inplace=True)

    new_DF = pd.read_csv(input_file, sep='\t')
    new_DF.set_index('name', inplace=True)
    # Assuming you have your heatmap data
    DF = new_DF.T
    DF.columns = new_DF.index
    DF.index = new_DF.columns
    df = DF.iloc[clm_list]  # Acc
    df = df[arranged_columns]
    df.index.name = 'name'

    # print(height, width)

    heatmap = go.Heatmap(
        z=df.values,
        x=df.columns,
        zmin=0,
        zmax=1,
        y=['LRC', 'DTC', 'SVC', 'RFC', 'HDC'],
        # colorbar=dict(title='Value'),
        text=df.values,  # Display values in each cell
        texttemplate="%{text}",  # Format for text
        colorscale=color_labels, 
        textfont=dict(size=font_size, color='white')
    )

    shapes = []

    for i in range(5, len(df), 5):
        shapes.append(
            go.layout.Shape(
                type='line',
                x0=-0.5,
                x1=len(df.columns) - 0.5,
                y0=i - 0.5,
                y1=i - 0.5,
                line=dict(color='white', width=1),
            )
        )

    ind = 0
    for t in plotting_columns.keys():
        if t != 'None' and len(plotting_columns[t]) > 0:
            col_idx = plotting_columns[t][0][1]
            row_idx = 4
            shape1 = go.layout.Shape(
                type='rect',
                x0=col_idx - 0.48,
                x1=plotting_columns[t][-1][1] + 0.48,
                y0=row_idx - 4.5,
                y1=row_idx + 0.5,
                line=dict(color=colors[t], width=2.5),  # Use color from the color scale constant
                fillcolor='rgba(255, 255, 255, 0)',  # Transparent fill
            )
            shapes.append(shape1)

    fig = go.Figure(data=[heatmap])

    print(input_file.split('/')[len(input_file.split('/'))-1].split('.')[0])

    fig.update_layout(
        width=figure_size[0],
        height=figure_size[1],
        shapes=shapes,
        title=input_file.split('/')[len(input_file.split('/'))-1].split('.')[0],
        xaxis=dict(title='Study', tickfont=dict(size=24),  tickangle=tick_angle),
        yaxis=dict(title='Classifier', tickfont=dict(size=24) ),
        yaxis_autorange='reversed',
        # colorscale=[[1, 'blue'], [-1, 'red']],
        autosize=False,
    )

    if saveSVG:
        # fig.write_image( "out.png",  scale=2, format='png')
        fig.write_html(outfile)
    
    # fig.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Plot function with specified parameters.")
    parser.add_argument("--input_file", type=str,  help="Category index (integer)")
    parser.add_argument("--column_list", type=str , default='4', help="Category index (integer)")
    parser.add_argument("--figure_size", type=int, nargs=2, default=[2460, 800],
                        metavar=('WIDTH', 'HEIGHT'), help="Figure size as two integers (width height)")
    parser.add_argument("--saveSVG", default=True, help="Flag to save as SVG (default: False)")
    parser.add_argument("--file_type", type=str, default="svg", choices=['svg', 'png', 'pdf'], help="File type for saving")
    parser.add_argument("--color_labels", type=str, default="emrld", help="Color scheme for labels")
    parser.add_argument("--font_size", type=int, default=22, help="Font size for labels")
    parser.add_argument("--tick_font", type=int, default=26, help="Font size for tick labels")
    parser.add_argument("--tick_angle", type=int, default=-80, help="Angle of tick labels")
    parser.add_argument("--threshold", type=float, default=0.05, help="Angle of tick labels")
    parser.add_argument("--output", type=str, default="out.html", help="Category index (integer)")

    args = parser.parse_args()

    Plot(
        input_file=args.input_file,
        column_list=args.column_list,
        figure_size=tuple(args.figure_size),
        saveSVG=args.saveSVG,
        file_type=args.file_type,
        color_labels=args.color_labels,
        font_size=int(args.font_size),
        tick_font=int(args.tick_font),
        tick_angle=int(args.tick_angle),
        threshold=float(args.threshold),
        outfile=args.output
    )