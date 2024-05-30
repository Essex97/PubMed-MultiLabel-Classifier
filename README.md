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

           A       0.83      0.46      0.59      4606
           B       0.93      1.00      0.96      9276
           C       0.74      0.85      0.79      5284
           D       0.75      0.91      0.82      6157
           E       0.78      1.00      0.88      7842
           F       0.65      0.42      0.51      1762
           G       0.72      0.94      0.82      6660
           H       0.00      0.00      0.00      1272
           I       0.72      0.16      0.26      1159
           J       1.00      0.00      0.00      1141
           L       0.00      0.00      0.00      1486
           M       0.79      0.89      0.84      4280
           N       0.69      0.75      0.72      4519
           Z       0.60      0.16      0.26      1651

   micro avg       0.78      0.77      0.78     57095
   macro avg       0.66      0.54      0.53     57095
weighted avg       0.75      0.77      0.73     57095
 samples avg       0.78      0.78      0.77     57095
```

### 3. Bert-based models
bert-base-uncased
```text
              precision    recall  f1-score   support

           A       0.23      0.00      0.01      4606
           B       0.93      1.00      0.96      9276
           C       0.53      1.00      0.69      5284
           D       0.53      0.06      0.10      6157
           E       0.78      1.00      0.88      7842
           F       0.22      0.05      0.09      1762
           G       0.68      0.02      0.03      6660
           H       0.00      0.00      0.00      1272
           I       0.00      0.00      0.00      1159
           J       0.00      0.00      0.00      1141
           L       0.00      0.00      0.00      1486
           M       0.43      0.99      0.60      4280
           N       0.36      0.19      0.25      4519
           Z       0.00      0.00      0.00      1651

   micro avg       0.64      0.49      0.56     57095
   macro avg       0.33      0.31      0.26     57095
weighted avg       0.53      0.49      0.42     57095
 samples avg       0.65      0.49      0.55     57095
```

bio-bert
```text
              precision    recall  f1-score   support

           A       0.77      0.81      0.79      4606
           B       0.96      0.99      0.97      9276
           C       0.88      0.83      0.85      5284
           D       0.86      0.94      0.90      6157
           E       0.82      0.95      0.88      7842
           F       0.78      0.75      0.76      1762
           G       0.80      0.90      0.85      6660
           H       0.60      0.07      0.12      1272
           I       0.66      0.55      0.60      1159
           J       0.53      0.50      0.52      1141
           L       0.69      0.37      0.48      1486
           M       0.85      0.90      0.87      4280
           N       0.83      0.73      0.78      4519
           Z       0.73      0.60      0.66      1651

   micro avg       0.83      0.84      0.84     57095
   macro avg       0.77      0.71      0.72     57095
weighted avg       0.83      0.84      0.82     57095
 samples avg       0.84      0.84      0.83     57095
```

## API
An API using FastAPI and uvicorn libraries created and below you can find some instruction on how to run it locally 
or use the deployed version directly.

Local usage:
1. Start the app
    ```bash
    cd AtyponAssignment/api
    uvicorn api:app --reload
    ```

2. Copy paste the below command
    ```bash
    curl -H "Content-Type: application/json" -X POST -d '{"text":"A case of a patient with type 1 neuroromatosis associated with popliteal and coronary artery aneurysms is described in which cross-sectional\nimaging provided diagnostic information. The aim of this study was to compare the exercise intensity and competition load during Time Trial (TT),\nFlat (FL), Medium Mountain (MM) and High Mountain (HM) stages based heart rate (HR) and session rating of perceived exertion (RPE).METHODS:\nWe monitored both HR and RPE of 12 professional cyclists during two consecutive 21-day cycling races in order to analyze the exercise intensity and competition load (TRIMPHR and TRIMPRPE).\nRESULTS:The highest (P<0.05) mean HR was found in TT (169±2 bpm) versus those observed in FL (135±1 bpm), MM (139±3 bpm), HM (143±1 bpm)"}' http://127.0.0.1:8000/v1/predict
    ```
    Response:
    ```json
    {      
      "STATUS":"OK",
      "RESPONSE": {
        "article": "A case of a patient with type 1 neuroromatosis associated with popliteal and coronary artery aneurysms is described in which cross-sectional\nimaging provided diagnostic information. The aim of this study was to compare the exercise intensity and competition load during Time Trial (TT),\nFlat (FL), Medium Mountain (MM) and High Mountain (HM) stages based heart rate (HR) and session rating of perceived exertion (RPE).METHODS:\nWe monitored both HR and RPE of 12 professional cyclists during two consecutive 21-day cycling races in order to analyze the exercise intensity and competition load (TRIMPHR and TRIMPRPE).\nRESULTS:The highest (P<0.05) mean HR was found in TT (169±2 bpm) versus those observed in FL (135±1 bpm), MM (139±3 bpm), HM (143±1 bpm)",
        "scores": {
          "Named Groups [M]": 0.9950736165046692,
          "Organisms [B]": 0.9930197596549988,
          "Analytical, Diagnostic and Therapeutic Techniques, and Equipment [E]":0.8981314301490784,
          "Phenomena and Processes [G]": 0.8844825625419617,
          "Diseases [C]": 0.8118777275085449,
          "Health Care [N]": 0.7473238110542297,
          "Anatomy [A]": 0.6173810362815857,
          "Psychiatry and Psychology [F]": 0.5503154993057251,
          "Anthropology, Education, Sociology, and Social Phenomena [I]": 0.5282009243965149,
          "Chemicals and Drugs [D]": 0.44657811522483826,
          "Technology, Industry, and Agriculture [J]": 0.17374999821186066,
          "Geographicals [Z]": 0.15315715968608856,
          "Disciplines and Occupations [H]": 0.06349503993988037,
          "Information Science [L]": 0.06157442554831505,
        }
      }
    }
    ```