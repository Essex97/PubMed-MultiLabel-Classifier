import os
import torch
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.metrics import classification_report
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    EarlyStoppingCallback,
    Trainer, TrainingArguments
)

from output import OUTPUT_DIR


class PubMedDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)


def train_transformer(transformer_path, x_train, y_train, x_test, y_test):
    """
    Applies hyper parameter tuning on a MultiLayer Perceptron (MLP)
    :param transformer_path: The model id of a predefined model in huggingface.com
    :param x_train: The train texts
    :param y_train: The train ground true labels
    :param x_test: The test texts
    :param y_test: The test ground true labels
    """

    # Initialize the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained(transformer_path)

    # Tokenize the input texts
    train_encodings = tokenizer(list(x_train), truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(list(x_test), truncation=True, padding=True, max_length=512)

    # Construct the train/test datasets
    train_dataset = PubMedDataset(train_encodings, y_train.values)
    test_dataset = PubMedDataset(test_encodings, y_test.values)

    # Load the BERT model for sequence classification
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=y_train.shape[1], problem_type="multi_label_classification")

    # Construct the output path
    now = datetime.now()
    date_time = now.strftime("%d_%m_%Y_%H:%M:%S")
    output_path = os.path.join(OUTPUT_DIR, transformer_path, date_time)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, 'results'),
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(OUTPUT_DIR, 'logs'),
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch"
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Save the model and tokenizer
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Predict on the test set
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions

    # Apply sigmoid to get probabilities
    pred_probs = torch.sigmoid(torch.tensor(preds)).numpy()

    # Binarize the predictions by setting a threshold to 0.5
    y_pred = (pred_probs >= 0.5).astype(int)

    # Print classification report
    print(classification_report(y_test, y_pred, target_names=y_train.columns))

    # Extract the training logs
    training_logs = trainer.state.log_history

    # Get the evaluation loss values
    eval_loss = [log['eval_loss'] for log in training_logs if 'eval_loss' in log.keys()]
    epochs = list(range(1, len(eval_loss) + 1))

    # Plot the training and evaluation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, eval_loss, label='Evaluation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Evaluation Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{transformer_path}_losses.png"))
