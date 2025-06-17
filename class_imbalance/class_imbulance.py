import argparse
import pandas as pd
import plotly.graph_objects as go
import glob

# Argparse setup
parser = argparse.ArgumentParser(description="Balance check and boxplot visualization from label counts.")

parser.add_argument('--input_files', type=str, nargs='+', required=True, help="Input tab-delimited files")
parser.add_argument('--label_column', type=str, default='label', help="Name of the label column")
parser.add_argument('--min_rows', type=int, default=100, help="Minimum number of rows required to process a file")
parser.add_argument('--plot_title', type=str, default='Distribution of Class Imbalance Across Datasets', help="Title for the box plot")
parser.add_argument('--output_html', type=str, default='boxplot.html', help="Output HTML file for the plot")
parser.add_argument("--figure_size", type=int, nargs=2, default=[2460, 800],metavar=('WIDTH', 'HEIGHT'), help="Figure size as two integers (width height)")
parser.add_argument('--x_axis_title', type=str, default='Datasets', help="Name of the label column")
parser.add_argument('--y_axis_title', type=str, default='Class imbalance score', help="Name of the label column")
parser.add_argument('--outputdata', type=str, default='Class imbalance score', help="Name of the label column")


args = parser.parse_args()


# Initialize
counter = 0
class_labels = None
data_list = []



# Process files
for f in args.input_files:
    NewDf = pd.read_csv(f, sep='\t')
    if NewDf.shape[0] >= args.min_rows:
        counter += NewDf.shape[0]
        class_labels = NewDf[args.label_column].value_counts().to_dict()
        data_list.append(class_labels)

# Compute data balance
dataBalance = []

for i in data_list:
    values = []
    values.append(i[list(i.keys())[0]])
    values.append(i[list(i.keys())[1]])
    min_value = min(values)
    normalized_value = round((min_value / sum(values)), 3)
    dataBalance.append(normalized_value)

# Normalize for boxplot
normalized_data = [(x - min(dataBalance)) / (max(dataBalance) - min(dataBalance)) for x in dataBalance]

# Plot boxplot
fig = go.Figure()
fig.add_trace(go.Box(
    y=normalized_data,
    name=args.x_axis_title,
    boxpoints='all',
    notched=True,
    marker=dict(size=5, color='blue')
))

fig.update_layout(
    width=int(args.figure_size[0]),
    height=int(args.figure_size[1]),
    title=args.plot_title,
    yaxis_title=args.y_axis_title,
)

# Save as HTML
fig.write_html(args.output_html)
# fig.show()

print(f"Processed {len(data_list)} files with total samples: {counter}")
print(f"Plot saved as {args.output_html}")
