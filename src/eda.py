import ast
import matplotlib.pyplot as plt
import os
import pandas as pd
import seaborn as sns

from datasets import load_dataset

from output import OUTPUT_DIR


def load_data(dataset_id: str):
    """
    Downloads and loads the dataset using a dataset_id path of the HuggingFace Hub
    :param dataset_id: The name of the dataset in the HuggingFace Huv
    :return: The loaded dataset
    """

    dataset = load_dataset(path=dataset_id)

    return dataset['train']


def plot_text_length_distribution(text_column: str, dataset: pd.DataFrame):
    """
    Displays the distribution of the texts length
    :param text_column: The column name of the text data
    :param dataset: The pandas DataFrame of the dataset
    """

    # Calculate the length of each text
    text_lengths_df = dataset[text_column].apply(len)

    # Calculate the mean and standard deviation of text lengths
    mean_length = text_lengths_df.mean()
    std_length = text_lengths_df.std()

    # Plot the distribution of text lengths
    plt.figure(figsize=(10, 6))
    sns.histplot(text_lengths_df, bins=50, kde=True)

    # Add vertical lines for mean and std
    plt.axvline(mean_length, color='r', linestyle='dashed', linewidth=2)
    plt.axvline(mean_length + std_length, color='g', linestyle='dashed', linewidth=2)
    plt.axvline(mean_length - std_length, color='g', linestyle='dashed', linewidth=2)

    # Annotate the mean and std values
    plt.text(mean_length, plt.ylim()[1] * 0.9, f'Mean: {mean_length:.2f}', color='r')
    plt.text(mean_length + std_length, plt.ylim()[1] * 0.8, f'+1 Std: {mean_length + std_length:.2f}', color='g')
    plt.text(mean_length - std_length, plt.ylim()[1] * 0.8, f'-1 Std: {mean_length - std_length:.2f}', color='g')

    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.title('Distribution of Text Lengths')
    plt.savefig(os.path.join(OUTPUT_DIR, 'text_length_distribution.png'))


def plot_number_of_labels_distribution(label_column: str, dataset: pd.DataFrame):
    """
    Displays the number of labels distribution of the dataset
    :param label_column: The column name of the labels
    :param dataset: The pandas DataFrame of the dataset
    """

    # Convert string values to lists
    labels_column_df = dataset[label_column].apply(ast.literal_eval)

    # Plot the distribution of the number of labels per sample
    num_labels_per_sample = labels_column_df.apply(len)

    # Calculate the mean and standard deviation
    mean_labels = num_labels_per_sample.mean()
    std_labels = num_labels_per_sample.std()

    # Plot the distribution of the number of labels per sample with mean and std
    plt.figure(figsize=(10, 6))
    sns.histplot(num_labels_per_sample, bins=range(1, max(num_labels_per_sample) + 1), kde=False)

    # Add vertical lines for mean and std
    plt.axvline(mean_labels, color='r', linestyle='dashed', linewidth=2)
    plt.axvline(mean_labels + std_labels, color='g', linestyle='dashed', linewidth=2)
    plt.axvline(mean_labels - std_labels, color='g', linestyle='dashed', linewidth=2)

    # Annotate the mean and std values
    plt.text(mean_labels, plt.ylim()[1] * 0.9, f'Mean: {mean_labels:.2f}', color='r')
    plt.text(mean_labels + std_labels, plt.ylim()[1] * 0.8, f'+1 Std: {mean_labels + std_labels:.2f}', color='g')
    plt.text(mean_labels - std_labels, plt.ylim()[1] * 0.8, f'-1 Std: {mean_labels - std_labels:.2f}', color='g')

    plt.xlabel('Number of Labels per Sample')
    plt.ylabel('Frequency')
    plt.title('Distribution of Number of Labels per Sample with Mean and Std Dev')
    plt.savefig(os.path.join(OUTPUT_DIR, 'number_of_label_distribution.png'))


def plot_label_distribution(label_column: str, dataset: pd.DataFrame):
    """
    Displays the label distribution
    :param label_column: The column name of the labels
    :param dataset: The pandas DataFrame of the dataset
    """

    # Convert string values to lists
    labels_column_df = dataset[label_column].apply(ast.literal_eval)

    # Flatten the list of labels
    all_labels = [label for sublist in labels_column_df for label in sublist]

    # Plot the distribution of each label
    label_counts = pd.Series(all_labels).value_counts()
    plt.figure(figsize=(10, 6))
    label_counts.plot(kind='bar')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    plt.title('Distribution of Labels')
    plt.savefig(os.path.join(OUTPUT_DIR, 'label_distribution.png'))


def plot_label_correlation(labels_column: str, dataset: pd.DataFrame):
    """
    Displays the correlation between different labels to see if there are common co-occurrences.
    :param labels_column: The column name of the labels
    :param dataset: The pandas DataFrame of the dataset
    """

    # Convert string values to lists
    labels_column_df = dataset[labels_column].apply(ast.literal_eval)

    # Create a DataFrame with one-hot encoded labels
    label_df = labels_column_df.apply(lambda x: pd.Series(1, index=x)).fillna(0)

    # Calculate the correlation matrix
    correlation_matrix = label_df.corr()

    # Plot the heatmap of the correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
    plt.title('Correlation Matrix of Labels')
    plt.savefig(os.path.join(OUTPUT_DIR, 'label_correlation.png'))


def perform_eda_analysis(dataset):
    """
    A function to perform the Exploratory Data Analysis (EDA) on the give dataset
    :param dataset: The HuggingFace dataset
    """

    # Convert to a pandas DataFrame for easier manipulation
    df = pd.DataFrame(dataset)

    print(f"Total Articles: {len(dataset)}")

    plot_text_length_distribution(text_column='abstractText', dataset=df)
    plot_number_of_labels_distribution(label_column='meshroot', dataset=df)
    plot_label_distribution(label_column='meshroot', dataset=df)
    plot_label_correlation(labels_column='meshroot', dataset=df)


if __name__ == '__main__':
    data = load_data(
        dataset_id="owaiskha9654/PubMed_MultiLabel_Text_Classification_Dataset_MeSH")

    perform_eda_analysis(dataset=data)
