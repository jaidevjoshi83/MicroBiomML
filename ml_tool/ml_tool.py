from pycaret.classification import setup, create_model, tune_model, pull
import subprocess
import itertools
import sys
import argparse
import pandas as pd
import json


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


def run_pycaret(algo=None, custom_para=None, tune_para=None, file_path=None, setup_param=None, index_clm_name=None):

    df = pd.read_csv(file_path, sep='\t')

    if index_clm_name:
        df = df.drop(columns=[index_clm_name])

    setup_dict = json.loads(setup_param)


    clf = setup(data=df, **setup_dict)

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
                print(pd.DataFrame(scores))
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
                print(pd.DataFrame(scores))
            else:
                print("Command failed:", result.stderr)

    else:
        if custom_para:
            custom_params = json.loads(custom_para)
            model = create_model(algo, **custom_params)
            df_result = pull()
            print(df_result.T['Mean'])

        elif tune_para:
            params = read_params(param_file)
            model = create_model(algo)
            tuned_model = tune_model(model, custom_grid=params)
            df_result = pull()
            print(df_result.T['Mean'])

        else:
            model = create_model(algo)
            df_result = pull()
            print(df_result.T['Mean'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run PyCaret ML setup.')
    parser.add_argument('--algo', type=str, required=False, help='Algorithm to run')
    parser.add_argument('--data_file', type=str, required=True, help='Path to data file')
    parser.add_argument('--custom_para', required=False, default=None, help='Custom hyperparameters (JSON string)')
    parser.add_argument('--tune_para', required=False, default=None, help='Flag for tuning hyperparameters')
    parser.add_argument('--setup', required=True, type=str, help='Setup parameters as JSON string')
    parser.add_argument('--index_clm_name', required=False, type=str, help='Name of the index Column')

    args = parser.parse_args()

    run_pycaret(
        algo=args.algo,
        file_path=args.data_file,
        custom_para=args.custom_para,
        tune_para=args.tune_para,
        setup_param=args.setup, 
        index_clm_name=args.index_clm_name,
    )