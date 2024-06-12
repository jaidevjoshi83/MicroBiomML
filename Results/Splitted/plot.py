import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys


###Fix Duplicate Column Names ############################
def ReturnUniqueColumnName(columns):
    name_count = {}
    unique_columns = []
    for name in columns:
        if name in name_count:
            name_count[name] += 1
            unique_columns.append(f"{name}_{name_count[name]}")
        else:
            name_count[name] = 1
            unique_columns.append(name)
    return unique_columns

# data  = open(sys.argv[1])

DF = pd.read_csv(sys.argv[1], sep='\t')
DF.set_index('name', inplace=True)
df = DF.T
df.index.name = 'name'
df.columns = ReturnUniqueColumnName(df.columns.to_list())
###Fix Duplicate Column Names ############################

heatmap = go.Heatmap(
    z=df.values,
    x=df.columns,
    y=df.index,
    # colorscale='Viridis',
    colorbar=dict(title='Value'),
    text=df.values,  # Display values in each cell
    texttemplate="%{text}",  # Format for text
    textfont=dict(color='green'),  # Ensure text is visible on darker colors
)

shapes = []


for i in range(5, len(df), 5):  # Start from the 3rd row and add lines every 2 rows
    shapes.append(
        go.layout.Shape(
            type='line',
            x0=-0.5,  # Start from the left edge
            x1=len(df.columns) - 0.5,  # End at the right edge
            y0=i - 0.5,  # Position for the horizontal line
            y1=i - 0.5,
            line=dict(color='white', width=4),  # White line with a thickness of 2
        )
    )

# Find all cells with values greater than 8

for row in df.index:
    for col in df.columns:
        if df.loc[row, col] > 0.96:
            # Determine the coordinates for the rectangle that covers each cell
            col_idx = df.columns.get_loc(col)  # Get the index of the column
            row_idx = df.index.get_loc(row)    # Get the index of the row
            shape = go.layout.Shape(
                type='rect',
                x0=col_idx - 0.5,
                x1=col_idx + 0.5,
                y0=row_idx - 0.5,
                y1=row_idx + 0.5,
                line=dict(color='red', width=1),  # Red border for the rectangle
                fillcolor='rgba(255, 255, 255, 0)',  # Transparent fill
            )
            shapes.append(shape)  # Add this shape to the list of shapes

# Create the figure and add the heatmap and shapes


fig = go.Figure(data=[heatmap])
fig.update_layout(
    width=550,  # Adjust based on your needs
    height=800,
    shapes=shapes,  # Add all the shapes (rectangles)
    title='Heatmap with Multiple Rectangles around Values Greater Than 8',
    xaxis=dict(title='Metrics'),
    yaxis=dict(title='Rows'),
    yaxis_autorange='reversed',  # Ensures correct orientation
)

# Display the heatmap with the rectangles
fig.write_image("scatter_plot.svg", format="svg")
fig.show()