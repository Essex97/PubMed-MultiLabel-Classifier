from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier


def train_logistic_regression(x_train, y_train, x_test, y_test):
    """
    Trains and evaluates a LogisticRegression model o the provided data
    :param x_train: The train texts
    :param y_train: The train ground true labels
    :param x_test: The test texts
    :param y_test: The test ground true labels
    """

    # Apply TF-IDF vectorization for the input texts
    vectorizer = TfidfVectorizer(max_features=10000)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    # Initialize and train a logistic regression model using OneVsRestClassifier
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    model.fit(x_train_tfidf, y_train)

    # Make predictions on the testing set
    y_pred = model.predict(x_test_tfidf)

    # Evaluate the model
    print(classification_report(y_test, y_pred, target_names=y_train.columns))
