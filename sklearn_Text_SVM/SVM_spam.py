# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
from sklearn.svm import SVC 

from sklearn.model_selection import train_test_split

#Was causing the program to perform much slower. Didnt need it, results were not much different.
#from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#Got rid of this.
#from sklearn.linear_model import LogisticRegression 

if __name__ == "__main__":

    # Importing the dataset
    dataset = pd.read_csv('spambase.csv')
    y_hold = len(dataset.columns)
    print(dataset.columns)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, y_hold-1].values
    
    #Data Set Check
    #dataset.describe()
    
    #Create Training Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5, random_state = 0)
    
    #Formatting to match SVM.py
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    C_vals = [0.01, 0.1, 1, 10, 100]
    gammas = ['scale']
    
    tests = []
    results = []
    
    for kernel in kernels:
        for C in C_vals:
            for gamma in gammas:
                tests.append({
                    'name': f'{kernel}_{C}_{gamma}',
                    'kernel': kernel,
                    'C': C,
                    'gamma': gamma,
                    })
    
    for test in tests:
        test_name = test['name']
        print(f'Running test {test_name}')
        svm = SVC(kernel = test['kernel'], C = test['C'], gamma = test['gamma'])
        svm.fit(X_train, y_train)
        pred = svm.predict(X_test)
        acc = accuracy_score(y_test, pred)
        conf_mat = confusion_matrix(y_test, pred)
        report = classification_report(y_test, pred)
        results.append({
            'name': test['name'],
            'C': test['C'],
            'gamma': test['gamma'],
            'kernel': test['kernel'],
            'accuracy': acc,
            'confusion_matrix': conf_mat.tolist(),
            'classification_report': report,            
            })
    
    #Same as above, matcing SVM.py
    for result in results:
        result_path = f'/{result["kernel"]}/{result["C"]}/{result["gamma"]}'
        # if path doesn't exist, create it
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # save confusion matrix
        plt.figure(figsize=(10, 10))
        plt.imshow(result['confusion_matrix'], interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix for {result["name"]}\nAccuracy: {result["accuracy"] * 100:.2f}%\n')
        plt.colorbar()
        tick_marks = np.arange(2)     
        plt.xticks(tick_marks, ['OK', 'DEFECTIVE'], rotation=45)
        plt.yticks(tick_marks, ['OK', 'DEFECTIVE'])
        plt.tight_layout()
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{result_path}/confusion_matrix.png')
     

        plt.show()
        print(f'Confusion Matrix saved to {result_path}/confusion_matrix.png')
          
        print(f'Accuracy for {kernel} kernel: {result["accuracy"]}')
        print(f'Classification report for {kernel} kernel:')
        print(result['classification_report'])
              
          # save results to JSON
        with open(f'{result_path}/{result["name"]}.json', 'w') as f:
            result = {
                'name': result['name'],
                'C': result['C'],
                'gamma': result['gamma'],
                'kernel': result['kernel'],
                'accuracy': result['accuracy'],
                'confusion_matrix': result['confusion_matrix'],
                'confusion_matrix_image': f'{result_path}/confusion_matrix.png',
                'classification_report': result['classification_report']
            }   
            json.dump(result, f)

    results.sort(key=lambda x: x['accuracy'], reverse=True)
    print("Best results:")
    for index, result in enumerate(results):
        print(f'{index + 1}. {result["name"]} - {result["accuracy"] * 100:.2f}%')