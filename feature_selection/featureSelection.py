import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import os
import time
import argparse
import json

#scikit-learn==1.4.2
#pandas==2.2.3

# Argument parsing
parser = argparse.ArgumentParser(description="Feature selection using SequentialFeatureSelector on a single TSV file")
parser.add_argument('--input', required=True, help="Path to input TSV file")
parser.add_argument('--threads', type=int, required=True, help="Number of threads")
parser.add_argument('--classifier', required=True, choices=['lr', 'dt', 'sv', 'rf'], help="Classifier choice")
parser.add_argument('--label', required=True, help="Name of the class label column in the dataset")
parser.add_argument('--tol', type=float, default=1e-5, help="Tolerance for SequentialFeatureSelector convergence (default: 1e-5)")
parser.add_argument('--index_clm', type=None, default='sample_id', help="Index Column")
parser.add_argument('--feature_out', type=str, default='out.tsv', help="Index Column")
parser.add_argument('--log', type=str, default='out.log', help="Index Column")


args = parser.parse_args()

# Classifiers dictionary
classifiers = {
    'lr': LogisticRegression(),
    'dt': DecisionTreeClassifier(),
    'sv': SVC(),
    'rf': RandomForestClassifier()
}

classifier = classifiers[args.classifier]

# Directory to save results
dir_name = os.path.splitext(os.path.basename(args.input))[0] + "_" + args.classifier
os.makedirs(f"./{dir_name}", exist_ok=True)

# Log file

log_file = open(args.log, 'w')

print(f"Processing: {args.input}")


start_time = time.time()

df = pd.read_csv(args.input, sep="\t")


# print(df.drop(columns=[args.label, args.index_clm]))

if args.label not in df.columns:
    raise ValueError(f"Label column '{args.label}' not found in {args.input}")

if  args.index_clm == 'None' :
    X = df.drop(columns=[args.label])
else: 
    X = df.drop(columns=[args.label, args.index_clm])

labels = list(set(df[args.label].to_list()))

if len(labels) != 2:
    raise ValueError(f"Expected exactly 2 class labels in {args.input}, found {len(labels)}: {labels}")

label_mapping = {labels[0]: 0, labels[1]: 1}
y = [label_mapping[label] for label in df[args.label].to_list()]

# Sequential feature selector
sfs = SequentialFeatureSelector(
    classifier,
    n_features_to_select='auto',
    direction='backward',
    tol=args.tol,
    n_jobs=args.threads
)

sfs.fit(X, y)

selected_feature_indices = sfs.get_support()
selected_feature_names = [name for name, selected in zip(X.columns, selected_feature_indices) if selected]

print("Selected Features:", selected_feature_names)

out_df = pd.DataFrame(selected_feature_names, columns=['feature_name'])
out_df.to_csv(args.feature_out, sep="\t")

# Save selected features
elapsed_time = round(time.time() - start_time, 3)
print("Time taken:", elapsed_time)

log_file.write(f"{'time in seconds'}\t{'algorithm'}\n")
log_file.write(f"{elapsed_time}\t{args.classifier}\n")
log_file.close()
