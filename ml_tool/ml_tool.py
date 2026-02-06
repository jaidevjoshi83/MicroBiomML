from pycaret.classification import setup, create_model, tune_model, pull
import subprocess
import itertools
import sys
import argparse
import pandas as pd
import json
import io


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
    dp_column_list = [df.columns.tolist()[int(i)] for i in dp_columns.split(',')] if dp_columns else []

    if dp_column_list:
        df = df.drop(columns=dp_column_list)

    # print(df)
    file = open(metadata_file)
    lines = file.readlines()

    if len(lines[0].strip().split('\t')) < len(lines[1].strip().split('\t')):
        
        # Fix header
        column_names = lines[0].strip().split('\t')
        column_names.insert(0, 'sample_name')
        
        # Rebuild the corrected file as a string
        new_content = "\t".join(column_names) + "\n" + "".join(lines[1:])
        df_metadata = pd.read_csv(io.StringIO(new_content), sep="\t")
        # print("OK1")
        # print(df_metadata)
    else:
        df_metadata = pd.read_csv(metadata_file, sep='\t')

    # Index column drop removed
    setup_dict = json.loads(setup_param)
    
    # Handle target_label (index or name)
    try:
        col_idx = int(target_label) - 1
        setup_dict['target'] = df_metadata.columns.tolist()[col_idx]
    except ValueError:
        setup_dict['target'] = target_label

    combine_df = pd.concat([df, df_metadata[setup_dict['target']]], axis=1)
    combine_df.to_csv("combined_data.tsv", sep='\t', index=False)
    
    # Filter out classes with only 1 sample to avoid split errors
    if 'target' in setup_dict:
        target_col = setup_dict['target']
        # Ensure target is treated as string/object if it's categorical 
        # (though value_counts works on numbers too)
        vc = combine_df[target_col].value_counts()
        valid_classes = vc[vc >= 2].index
        rows_before = len(combine_df)
        combine_df = combine_df[combine_df[target_col].isin(valid_classes)]
        rows_after = len(combine_df)
        if rows_after < rows_before:
            print(f"Removed {rows_before - rows_after} samples with rare target classes (count < 2).")

    # Check for empty or too small dataframe before setup
    if combine_df.empty or len(combine_df) < 2:
        print("Error: Not enough samples after filtering for PyCaret setup. Please check your input data and parameters.")
        sys.exit(1)
    clf = setup(data=combine_df, **setup_dict)

    if algo == 'hdc':
        if custom_para and not tune_para:
            custom_params = json.loads(custom_para)
            command = ['chopin2.py', "--input", file_path, "--kfolds", "5"]

            for c, v in custom_params.items():
                command.append("--" + c)
                command.append(str(v))

            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode == 0:
                text = result.stdout
                scores = {}
                for i, line in enumerate(text.split("\n")):
                    if "Total elapsed time" in line:
                        scores["MCC"] = [text.split("\n")[i-1].split(' ')[3]]
                        scores["Recall"] = [text.split("\n")[i-2].split(' ')[1]]
                        scores["Prec."] = [text.split("\n")[i-3].split(' ')[1]]
                        scores["F1"] = [text.split("\n")[i-4].split(' ')[1]]
                        scores["Accuracy"] = [text.split("\n")[i-5].split(' ')[1]]
                df_scores = pd.DataFrame(scores)
                print(df_scores)
                if output_tabular:
                    df_scores.to_csv(output_tabular, sep='\t', index=False)
                if output_html:
                    df_scores.to_html(output_html)
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
                scores = {}
                for i, line in enumerate(text.split("\n")):
                    if "Total elapsed time" in line:
                        scores["MCC"] = [text.split("\n")[i-1].split(' ')[3]]
                        scores["Recall"] = [text.split("\n")[i-2].split(' ')[1]]
                        scores["Prec."] = [text.split("\n")[i-3].split(' ')[1]]
                        scores["F1"] = [text.split("\n")[i-4].split(' ')[1]]
                        scores["Accuracy"] = [text.split("\n")[i-5].split(' ')[1]]
                df_scores = pd.DataFrame(scores)
                print(df_scores)
                if output_tabular:
                    df_scores.to_csv(output_tabular, sep='\t', index=False)
                if output_html:
                    df_scores.to_html(output_html)
            else:
                print("Command failed:", result.stderr)

    else:
        if custom_para:
            custom_params = json.loads(custom_para)
            model = create_model(algo, **custom_params)
            df_result = pull()
            res = df_result.T['Mean']
            print(res)
            with open('logs.log', 'a') as f:
                f.write(str(res) + '\n')
            
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
            
            if output_tabular:
                df_result.to_csv(output_tabular, sep='\t')
            if output_html:
                df_result.to_html(output_html)

        else:
            model = create_model(algo)
            df_result = pull()
            res = df_result.T['Mean']
            print(res)
            with open('logs.log', 'a') as f:
                f.write(str(res) + '\n')
            
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



