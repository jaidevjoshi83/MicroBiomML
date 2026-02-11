from pycaret.classification import setup, create_model, tune_model, pull
import subprocess
import itertools
import sys
import argparse
import pandas as pd
import json
import io



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

def convert_value(val):
    """Convert string to appropriate Python type."""
    val = val.strip()
    if val.lower() == 'true':
        return True
    elif val.lower() == 'false':
        return False
    try:
        if '.' in val:
            return float(val)
        else:
            return int(val)
    except ValueError:
        return val

def read_params(filename):
    """Read hyperparameter values from file."""
    params = {}
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            key = parts[0].strip()
            values = [convert_value(val) for val in parts[1:]]
            params[key] = values
    return params


def tune_hdc(tune_param, data):
    combinations = list(itertools.product(
        tune_param['dimensionality'], tune_param['levels'], tune_param['retrain']
    ))

    full_score, f1_score = {}, {}

    for n, combination in enumerate(combinations):
        command = [
            "chopin2.py", "--input", data,
            "--dimensionality", str(combination[0]),
            "--kfolds", "5",
            "--levels", str(combination[1]),
            "--retrain", str(combination[2])
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        text = result.stdout


        print("##########################")

        scores, f1 = {}, []
        for i, line in enumerate(text.split("\n")):
            if "Total elapsed time" in line:
                scores["MCC"] = text.split("\n")[i-1].split(' ')[3]
                scores["Recall"] = text.split("\n")[i-2].split(' ')[1]
                scores["Prec."] = text.split("\n")[i-3].split(' ')[1]
                scores["F1"] = text.split("\n")[i-4].split(' ')[1]
                scores["Accuracy"] = text.split("\n")[i-5].split(' ')[1]
                f1.append(scores["F1"])

        full_score[n] = scores
        f1_score[n] = f1

    max_key = max(f1_score, key=lambda k: f1_score[k])
    return full_score[max_key]


def run_pycaret(algo=None, custom_para=None, tune_para=None, file_path=None, setup_param=None, target_label=None, metadata_file=None, output_tabular=None, output_html=None, dp_columns=None):

    # print(target_label)
    df = pd.read_csv(file_path, sep='\t')
    df_metadata = pd.read_csv(metadata_file, sep='\t')  

    dp_column_list = [df.columns.tolist()[int(i)-1] for i in dp_columns.split(',')] if dp_columns else []

    if dp_column_list:
        df = df.drop(columns=dp_column_list)

    # Index column drop removed
    setup_dict = json.loads(setup_param)
    
    # Handle target_label (index or name)
    try:
        col_idx = int(target_label) - 1
        setup_dict['target'] = df_metadata.columns.tolist()[col_idx]
    except ValueError:
        setup_dict['target'] = target_label

    combine_df = pd.concat([df, df_metadata[setup_dict['target']]], axis=1)

    combine_df.to_csv("./training_data_with_target_columns.tsv", sep='\t', index=False)

    # Check for empty or too small dataframe before setup
    if combine_df.empty or len(combine_df) < 2:
        print("Error: Not enough samples after filtering for PyCaret setup. Please check your input data and parameters.")
        sys.exit(1)

    if algo == 'hdc':

        file_path = "./training_data_with_target_columns.tsv"
        if custom_para and not tune_para:
          
            custom_params = json.loads(custom_para)
            command = ['chopin2.py', "--input", file_path, "--kfolds", "5"]

            for c, v in custom_params.items():
                command.append("--" + c)
                command.append(str(v))

            result = subprocess.run(command, capture_output=True, text=True)
            print("--- HDC (chopin2.py) STDOUT ---")
            print(result.stdout)
            print("--- HDC (chopin2.py) STDERR ---")
            print(result.stderr)
            print("--- End HDC Output ---")
            if result.returncode == 0:
                text = result.stdout
                df_scores = retrieve_results_from_hdc_folds(4, text)
                if output_tabular:
                    df_scores.to_csv(output_tabular, sep='\t', index=False)
                if output_html:
                    df_scores.to_html(output_html, index=False)
            else:
                print("Command failed:", result.stderr)

        elif tune_para:
            
            params = read_params('params.txt')
            result = tune_hdc(params, file_path)
            print("Best Tune Result:\n", result)

        else:
            command = ["chopin2.py", "--input", file_path, "--levels", "100", "--kfolds", "5"]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                text = result.stdout
                df_scores =retrieve_results_from_hdc_folds(4, text)
                if output_tabular:
                    df_scores.to_csv(output_tabular, sep='\t', index=False)
                if output_html:
                    df_scores.to_html(output_html, index=False)
            else:
                print("Command failed:", result.stderr)

    else:
        clf = setup(data=combine_df, **setup_dict)
        if custom_para:
            custom_params = json.loads(custom_para)
            model = create_model(algo, **custom_params)
            df_result = pull()
            res = df_result.T['Mean']
            print(res)
            with open('logs.log', 'a') as f:
                f.write(str(res) + '\n')
            # Add three-letter classifier suffix (algorithm + 'C') to columns except 'Fold'
            algo_abbr = (str(algo).upper()[:2] + 'C') if algo else "ALC"
            df_result.columns = [col if col == 'Fold' else f"{col}_{algo_abbr}" for col in df_result.columns]
            if output_tabular:
                df_result.to_csv(output_tabular, sep='\t')
            if output_html:
                df_result.to_html(output_html)

        elif tune_para:
            params = read_params(tune_para)
            model = create_model(algo)
            tuned_model = tune_model(model, custom_grid=params)
            df_result = pull()
            res = df_result.T['Mean']
            print(res)
            with open('logs.log', 'a') as f:
                f.write(str(res) + '\n')
            # Add three-letter classifier suffix (algorithm + 'C') to columns except 'Fold'
            algo_abbr = (str(algo).upper()[:2] + 'C') if algo else "ALC"
            df_result.columns = [col if col == 'Fold' else f"{col}_{algo_abbr}" for col in df_result.columns]
            if output_tabular:
                df_result.to_csv(output_tabular, sep='\t')
            if output_html:
                df_result.to_html(output_html)

        else:
            model = create_model(algo)
            df_result = pull()
            res = df_result.T['Mean']
      
            with open('logs.log', 'a') as f:
                f.write(str(res) + '\n')
            # Add three-letter classifier suffix (algorithm + 'C') to columns except 'Fold'
            algo_abbr = (str(algo).upper()[:2] + 'C') if algo else "ALC"
            df_result.columns = [col if col == 'Fold' else f"{col}_{algo_abbr}" for col in df_result.columns]
            if output_tabular:
                df_result.to_csv(output_tabular, sep='\t')
            if output_html:
                df_result.to_html(output_html)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run PyCaret ML setup.')
    parser.add_argument('--algo', type=str, required=False, help='Algorithm to run')
    parser.add_argument('--data_file', type=str, required=True, help='Path to data file')
    parser.add_argument('--metadata_file', type=str, required=True, help='Path to metadata file')
    parser.add_argument('--custom_para', required=False, default=None, help='Custom hyperparameters (JSON string)')
    parser.add_argument('--tune_para', required=False, default=None, help='Flag for tuning hyperparameters')
    parser.add_argument('--setup', required=True, type=str, help='Setup parameters as JSON string')
    parser.add_argument('--target_label', required=False, type=str, help='Name of the target label Column')
    parser.add_argument('--output_tabular', required=False, type=str, help='Path to output tabular file')
    parser.add_argument('--output_html', required=False, type=str, help='Path to output HTML file')
    parser.add_argument('--dp_columns', required=False, type=str, help='Columns to drop from training data')

    args = parser.parse_args()

    run_pycaret(
        algo=args.algo,
        file_path=args.data_file,
        custom_para=args.custom_para,
        tune_para=args.tune_para,
        setup_param=args.setup, 
        target_label=args.target_label,
        metadata_file=args.metadata_file,
        output_tabular=args.output_tabular,
        output_html=args.output_html,
        dp_columns=args.dp_columns
    )



