import pandas as pd

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.logistic_regression import train_logistic_regression


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


def run_experiments(model_names: list):
    """
    Runs the experiments one by one
    """

    supported_models = {'logistic_regression', 'mlp', 'bert', 'bio-bert'}
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


if __name__ == '__main__':
    run_experiments(model_names=['logistic_regression'])
