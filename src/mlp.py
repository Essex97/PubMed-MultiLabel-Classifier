import keras_tuner as kt
import time

from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

from output import OUTPUT_DIR


class MyHyperModel(kt.HyperModel):
    def __init__(self, input_dim, num_labels):
        super().__init__()
        self.input_dim = input_dim
        self.num_labels = num_labels

    def build(self, hp):
        model = Sequential()
        model.add(Dense(hp.Int('units', min_value=32, max_value=512, step=32),
                        input_dim=self.input_dim, activation='relu'))
        model.add(Dropout(hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))

        for i in range(hp.Int('num_layers', 1, 3)):
            model.add(Dense(hp.Int(f'layer_{i}_units', min_value=32, max_value=512, step=32), activation='relu'))
            model.add(Dropout(hp.Float(f'layer_{i}_dropout', min_value=0.2, max_value=0.5, step=0.1)))

        # projection layer
        model.add(Dense(self.num_labels, activation='sigmoid'))

        model.compile(optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model


def train_mlp(x_train, y_train, x_test, y_test):
    """
    Applies hyper parameter tuning on a MultiLayer Perceptron (MLP)
    :param x_train: The train texts
    :param y_train: The train ground true labels
    :param x_test: The test texts
    :param y_test: The test ground true labels
    """

    # Record the start time
    start_time = time.time()

    # Apply TF-IDF vectorization for the input texts
    vectorizer = TfidfVectorizer(max_features=10000)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    # Initialize the custom HyperModel with the input dimension and the num_labels
    hypermodel = MyHyperModel(input_dim=x_train_tfidf.shape[1],
                              num_labels=y_train.shape[1])

    # Initialize the KerasTuner
    tuner = kt.Hyperband(hypermodel,
                         objective='val_accuracy',
                         max_epochs=10,
                         factor=3,
                         directory=OUTPUT_DIR,
                         project_name='mlp_hyper_parameter_tuning')

    # Define a callback to stop training early if no improvement is seen
    stop_early = EarlyStopping(monitor='val_loss', patience=5)

    # Perform hyper parameter tuning
    tuner.search(x_train_tfidf, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

    # Get the optimal hyper parameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the model with the optimal hyper parameters and train it
    model = tuner.hypermodel.build(best_hps)
    history = model.fit(x_train_tfidf, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

    # Evaluate the model on the test set
    y_pred = (model.predict(x_test_tfidf) > 0.5).astype("int32")

    # Print the classification report
    print(classification_report(y_test, y_pred, target_names=y_train.columns))

    # Record the end time
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
