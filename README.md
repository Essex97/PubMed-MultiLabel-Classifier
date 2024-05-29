# AtyponAssignment

## Exploratory Data Analysis (EDA)

## Experiments
Our experimentation procedure includes multiple models. 
We started with the Logistic Regression to have a benchmarks, 
we continued with a MultiLayer Perceptron (MLP) and a hyper parameter tuning of it 
and we ended up with the use of Transformers and especially BERT and pre-trained models
in the biomedical domain which we observed that had the best performance.

### 1. Logistic Regression
By design, logistic regression models automatically handle binary classification problems in which the target vector (label column) 
has only two classes. However, three extensions to logistic regression are available to use logistic regression for multiclass 
classification in which the target vector has more than two classes.
* One-vs-Rest (OvR) multiclass strategy
* One-vs-One (OvO) multiclass strategy
* Multinomial method

The One-vs-Rest methodology achieved:
```text
              precision    recall  f1-score   support

           A       0.81      0.76      0.78      4606
           B       0.95      1.00      0.97      9276
           C       0.87      0.84      0.85      5284
           D       0.90      0.90      0.90      6157
           E       0.81      0.96      0.88      7842
           F       0.87      0.63      0.73      1762
           G       0.83      0.89      0.86      6660
           H       0.64      0.10      0.18      1272
           I       0.73      0.43      0.54      1159
           J       0.78      0.29      0.42      1141
           L       0.77      0.36      0.49      1486
           M       0.87      0.88      0.88      4280
           N       0.81      0.78      0.79      4519
           Z       0.77      0.54      0.64      1651

   micro avg       0.86      0.82      0.84     57095
   macro avg       0.82      0.67      0.71     57095
weighted avg       0.85      0.82      0.82     57095
 samples avg       0.86      0.83      0.83     57095
```

### 2. MLP
The hyper parameter tuning results:
```text
              precision    recall  f1-score   support

           A       0.46      1.00      0.63      4606
           B       0.93      1.00      0.96      9276
           C       0.53      1.00      0.69      5284
           D       0.62      1.00      0.76      6157
           E       0.78      1.00      0.88      7842
           F       0.18      1.00      0.30      1762
           G       0.67      1.00      0.80      6660
           H       0.13      1.00      0.23      1272
           I       0.12      1.00      0.21      1159
           J       0.11      1.00      0.20      1141
           L       0.15      1.00      0.26      1486
           M       0.43      1.00      0.60      4280
           N       0.45      1.00      0.62      4519
           Z       0.17      1.00      0.28      1651

   micro avg       0.41      1.00      0.58     57095
   macro avg       0.41      1.00      0.53     57095
weighted avg       0.58      1.00      0.70     57095
 samples avg       0.41      0.99      0.57     57095
```

### 3. MLP
