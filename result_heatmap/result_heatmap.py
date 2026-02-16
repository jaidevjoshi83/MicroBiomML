import pandas as pd #pandas==2.1.4
import plotly.graph_objects as go #plotly==5.20.0
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
def ResultSummary(file, threshold, column_list=None):
    print(file)
    new_DF = pd.read_csv(file, sep='\t')
    new_DF.set_index('name', inplace=True)

    DF = new_DF.T
    DF.columns = new_DF.index
    DF.index = new_DF.columns
    
    # If no column_list provided, use all columns
    if column_list is None:
        df = DF
    else:
        df = DF.iloc[column_list]

    column_anno_per = {}
    comparable = {}

    for n in df.columns.to_list():
        comparable[n] = Analysis(df[n].values, threshold)
    return comparable

def Plot(input_file, width=2460, height=800, color_labels='Greens', font_size=22, tick_font=26, tick_angle=-80, threshold=0.05, column_list=None, outfile='out.html'):
    
    # Parse column_list if it's a string (from command line)
    # Convert from 1-indexed (XML) to 0-indexed (Python)
    if isinstance(column_list, str) and column_list:
        column_list = [int(i) - 2 for i in column_list.split(',')]
    
    figure_size = (width, height)

    print(column_list)
    
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

    # Read and prepare data for plotting - use the same processing as ResultSummary
    new_DF = pd.read_csv(input_file, sep='\t')
    new_DF.set_index('name', inplace=True)
    
    # Transpose to get classifiers as rows and metrics as columns
    DF = new_DF.T
    DF.columns = new_DF.index
    DF.index = new_DF.columns
    
    column_list

    # Apply column_list filter if provided
    if column_list is None:
        df = DF
    else:
        df = DF.iloc[column_list]

    print(df)
    
    # Filter to only keep the arranged_columns (columns that pass the analysis)
    if arranged_columns:
        df = df[arranged_columns]
    
    df.index.name = 'name'

    # print(height, width)

    heatmap = go.Heatmap(
        z=df.values,
        x=df.columns,
        zmin=0,
        zmax=1,
        y=df.index,
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

    # Create legend annotations for border colors at the top
    legend_annotations = []
    legend_labels = {
        'better_all': 'Better than all (≥threshold)',
        'better_one': 'Better than some',
        'Comp_with_all': 'Comparable with all',
        'Comp_with_one': 'Comparable with some'
    }
    
    x_position = 0.0
    for color_key, label_text in legend_labels.items():
        legend_annotations.append(
            dict(
                x=x_position,
                y=1.12,
                xref='paper',
                yref='paper',
                text=f'<b style="color:{colors[color_key]};font-size:14px;">■</b> {label_text}',
                showarrow=False,
                xanchor='left',
                yanchor='bottom',
                font=dict(size=11)
            )
        )
        x_position += 0.25

    fig.update_layout(
        width=figure_size[0],
        height=figure_size[1],
        shapes=shapes,
        title='',
        xaxis=dict(title='Study', tickfont=dict(size=24),  tickangle=tick_angle),
        yaxis=dict(title='Classifier', tickfont=dict(size=24) ),
        yaxis_autorange='reversed',
        # colorscale=[[1, 'blue'], [-1, 'red']],
        autosize=False,
        annotations=legend_annotations,
        margin=dict(t=200)  # Add top margin for legend
    )

    # Save the figure as HTML
    fig.write_html(outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot heatmap from TSV data with classification results.")
    parser.add_argument("--input_file", type=str, default="test_data_age_category.tsv", help="Path to input TSV file (default: test_data_age_category.tsv)")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot heatmap from TSV data with classification results.")
    parser.add_argument("--input_file", type=str, default="test_data_age_category.tsv", help="Path to input TSV file (default: test_data_age_category.tsv)")
    parser.add_argument("--column_list", type=str, default=None, help="Comma-separated column indices to plot (default: None - plots all data)")
    parser.add_argument("--width", type=int, default=2460, help="Figure width in pixels (default: 2460)")
    parser.add_argument("--height", type=int, default=800, help="Figure height in pixels (default: 800)")
    parser.add_argument("--color_labels", type=str, default="Greens", help="Color scheme for heatmap (default: Greens)")
    parser.add_argument("--font_size", type=int, default=22, help="Font size for cell text (default: 22)")
    parser.add_argument("--tick_font", type=int, default=26, help="Font size for tick labels (default: 26)")
    parser.add_argument("--tick_angle", type=int, default=-80, help="Angle of x-axis tick labels in degrees (default: -80)")
    parser.add_argument("--threshold", type=float, default=0.05, help="Threshold for comparison analysis (default: 0.05)")
    parser.add_argument("--output", type=str, default="out.html", help="Output file path (default: out.html)")

    args = parser.parse_args()

    Plot(
        input_file=args.input_file,
        width=args.width,
        height=args.height,
        color_labels=args.color_labels,
        font_size=int(args.font_size),
        tick_font=int(args.tick_font),
        tick_angle=int(args.tick_angle),
        threshold=float(args.threshold),
        column_list=args.column_list,
        outfile=args.output
    )