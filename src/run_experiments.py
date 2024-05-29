import argparse
import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.logistic_regression import train_logistic_regression
from src.mlp import train_mlp
from src.transformer_based import train_transformer


def load_data(dataset_id: str):
    """
    Downloads and loads the dataset using a dataset_id path of the HuggingFace Hub
    :param dataset_id: The name of the dataset in the HuggingFace Huv
    :return: The loaded dataset
    """

    dataset = load_dataset(path=dataset_id)

    df = pd.DataFrame(dataset['train'])

    x_train = df['abstractText']
    y_train = df[['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'L', 'M', 'N', 'Z']]

    return x_train, y_train


def run_experiments(args):
    """
    Runs the experiments one by one
    """

    model_names = args.model_names
    supported_models = {'logistic_regression', 'mlp', 'transformer_based'}
    if not set(model_names).issubset(supported_models):
        raise ValueError(f'Not supported model(s) are given. '
                         f'The supported models are: {supported_models}')

    x_train, y_train = load_data(
        dataset_id="owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH")

    # Split the data into train/test sets
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    dataset = {"x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}

    if 'logistic_regression' in model_names:
        # Benchmarks
        train_logistic_regression(**dataset)

    if 'mlp' in model_names:
        train_mlp(**dataset)

    if 'transformer_based' in model_names:
        if not args.transformer_path:
            raise ValueError("Please give a transform_path")
        input_params = {**dataset, **{"transformer_path": args.transformer_path}}
        train_transformer(**input_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experimentation on multi-label classification task.")
    parser.add_argument('-mn', '--model_names', nargs='+', required=True,
                        help='The models you want to train')
    parser.add_argument('-tp', '--transformer_path', type=str, default='bert-base-uncased',
                        help='The transformer name in HuggingFace.com')
    arguments = parser.parse_args()

    run_experiments(args=arguments)
