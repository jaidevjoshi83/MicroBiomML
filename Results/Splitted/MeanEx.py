# print(df[df.columns[0]])

import numpy as np
# Initialize an empty list to store the slicing indices
import pandas as pd

DF = pd.read_csv('location.tsv', sep='\t')
DF.set_index('name', inplace=True)
df = DF.T
# df.index.name = 'name'
slices = []

# Set the step size for the intervals
step = 5

# Create a loop to generate the slice indices
for i in range(0, 25, step):  # 25 represents the total number of elements
    # Define the start and end of the slice
    start = i
    end = i + 4  # 4 represents the span or width of each slice
    
    # Append the slice to the list
    slices.append(f"{start}:{end}")
    print(np.mean(df[df.columns[0]][start:end]))
    print(df[df.columns[0]][start:end])

