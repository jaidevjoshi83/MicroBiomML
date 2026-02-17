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
import subprocess

def retrieve_results_from_hdc_folds(n_folds, text):
    
    split_text = text.splitlines()
    n_folds = n_folds
    df_list = []
    for i in range(n_folds):
        for n, line in enumerate(split_text):
            if f"Fold {i}" in split_text[n]:
                df_list.append([float(split_text[n+2].split(":")[1]), 'NaN', float(split_text[n+5].split(":")[1]), float(split_text[n+4].split(":")[1]), float(split_text[n+3].split(":")[1]), "NaN", float(split_text[n+6].split(":")[1])])

    df = pd.DataFrame(df_list, columns=["Accuracy", "AUC", "Recall", "Prec.", "F1", "Kappa", "MCC"])

    mean_row = df.mean(numeric_only=True)
    std_row = df.std(numeric_only=True)

    mean_df = mean_row.to_frame().T
    mean_df['Fold'] = 'Mean'

    std_df = std_row.to_frame().T
    std_df['Fold'] = 'Std'

    df = df.reset_index().rename(columns={'index': 'Fold'})

    df_with_stats = pd.concat([df, mean_df, std_df], ignore_index=True)

    return df_with_stats

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Feature selection using SequentialFeatureSelector on a single TSV file")
    parser.add_argument('--input', required=True, help="Path to count matrix file")
    parser.add_argument('--metadata', required=True, help="Path to metadata file")
    parser.add_argument('--threads', type=int, required=True, help="Number of threads")
    parser.add_argument('--classifier', required=True, choices=['lr', 'dt', 'sv', 'rf', 'hdc'], help="Classifier choice")
    parser.add_argument('--label', required=True, help="Name of the class label column in the dataset")
    parser.add_argument('--tol', type=float, default=1e-5, help="Tolerance for SequentialFeatureSelector convergence (default: 1e-5)")
    parser.add_argument('--index_clm', type=str, default='sample_id', help="Index Column")
    parser.add_argument('--feature_out', type=str, default='out.tsv', help="Output file for selected features")
    parser.add_argument('--log', type=str, default='out.log', help="Log file")
    parser.add_argument('--feature_selection', type=str, default=None, help="Path to feature selection file")
    parser.add_argument('--dp_columns', type=str, help='Columns to drop from training data')
    return parser.parse_args()

def load_and_preprocess_data(args):
    """Load and preprocess the input data."""
    df_counts = pd.read_csv(args.input, sep="\t")

    if args.dp_columns:
        dp_column_list = [df_counts.columns.tolist()[int(i) - 1] for i in args.dp_columns.split(',')]
        df_counts.drop(columns=dp_column_list, inplace=True)
    
    df_metadata = pd.read_csv(args.metadata, sep="\t")
    target_column = df_metadata.columns.to_list()[int(args.label)-1]
   
    if args.index_clm and args.index_clm in df_counts.columns and args.index_clm in df_metadata.columns:
        df_counts.set_index(args.index_clm, inplace=True)
        df_metadata.set_index(args.index_clm, inplace=True)

    df = pd.concat([df_counts, df_metadata[target_column]], axis=1)

    df.to_csv(f"temp_input.tsv", sep='\t', index=False)

    if args.feature_selection:
        with open(args.feature_selection) as f:
            features = [line.strip() for line in f]
        df = df[features]
    return df, target_column

def run_hdc_classification(args, df, dir_name, start_time):
    """Run HDC classification and extract features."""
    temp_input_file = f"./{dir_name}/temp_hdc_input.tsv"
    df.to_csv(temp_input_file, sep='\t', index=False)

    command = ["chopin2.py", "--input", temp_input_file, "--kfolds", "5", "--levels", "100", "--feature-selection", "backward"]
    result = subprocess.run(command, capture_output=True, text=True)
    
    if result.returncode == 0:
        text = result.stdout
        
        # Extract and save features
        try:
            features_start = text.index("Features\n----------") + len("Features\n----------\n")
            features_end = text.index("\n\nTotal ML models:")
            selected_features_str = text[features_start:features_end]
            selected_features = selected_features_str.strip().split('\n')
            
            print("Selected Features:", selected_features)

            out_df = pd.DataFrame(selected_features, columns=['feature_name'])

            out_df.to_csv("selected_features.tsv", sep="\t", index=False)

        except ValueError:
            print("Could not find the feature list in the HDC output.")

        # Parse and write scores to the log file
        try:
            df_scores = retrieve_results_from_hdc_folds(5, text)
            with open("hdc_performance.log", 'a') as log_file:
                log_file.write("\n--- HDC Performance Scores ---\n")
                df_scores.to_string(log_file)
                log_file.write("\n")
        except Exception as e:
            print(f"\nCould not parse performance scores: {e}")

    else:
        print("Command failed:", result.stderr)
    
    elapsed_time = round(time.time() - start_time, 3)
    with open(args.log, 'w') as log_file:
        log_file.write(f"{'time in seconds'}\t{'algorithm'}\n")
        log_file.write(f"{elapsed_time}\t{args.classifier}\n")

def run_sequential_feature_selection(args, df, start_time, target_label):

    """Run sequential feature selection for non-HDC classifiers."""
    classifiers = {
        'lr': LogisticRegression(),
        'dt': DecisionTreeClassifier(),
        'sv': SVC(),
        'rf': RandomForestClassifier()
    }
    classifier = classifiers[args.classifier]


    if target_label not in df.columns:
        raise ValueError(f"Label column '{target_label}' not found in the data.")

    X = df.drop(columns=[target_label])


    labels = list(set(df[target_label].to_list() ))

    if len(labels) != 2:
        raise ValueError(f"Expected exactly 2 class labels, found {len(labels)}: {labels}")

    label_mapping = {labels[0]: 0, labels[1]: 1}

    y = df[target_label].map(label_mapping).tolist()

    sfs = SequentialFeatureSelector(
        classifier,
        n_features_to_select='auto',
        direction='backward',
        tol=args.tol,
        n_jobs=args.threads
    )
    sfs.fit(X, y)

    selected_feature_names = X.columns[sfs.get_support()]


    out_df = pd.DataFrame(selected_feature_names, columns=['feature_name'])
    out_df.to_csv(args.feature_out, sep="\t", index=False)

    elapsed_time = round(time.time() - start_time, 3)
    with open(args.log, 'w') as log_file:
        log_file.write(f"{'time in seconds'}\t{'algorithm'}\n")
        log_file.write(f"{elapsed_time}\t{args.classifier}\n")

def main():
    """Main function to orchestrate the feature selection process."""
    args = parse_arguments()
    
    dir_name = os.path.splitext(os.path.basename(args.input))[0] + "_" + args.classifier
    os.makedirs(f"./{dir_name}", exist_ok=True)
    
    start_time = time.time()
    
    df, target_label = load_and_preprocess_data(args)

    if args.classifier == 'hdc':
        run_hdc_classification(args, df, dir_name, start_time)
    else:
        run_sequential_feature_selection(args, df, start_time, target_label)

if __name__ == "__main__":
    main()

