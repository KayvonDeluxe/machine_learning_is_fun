# machine_learning_practice

Just a small program I made for fun while reading about machine learning.  Took me a few hours to make it.
The program attempts to use clinical findings and histopathological features to classify a dermatologic condition as one of the following:
psoriasis, seboreic dermatitis, lichen planus, pityriasis rosea, chronic dermatitis, pityriasis rubra pilaris


The dataset is one of the many publicly available UC Irvine datasets and additional information can be found  here:
https://archive.ics.uci.edu/ml/datasets/dermatology


Obvs you need to install dependencies such as sklearn, pandas, numpy, etc.


Here is a sample of its output.

                *****************************************
                *********** Linear Regression ***********
                *****************************************
                Accuracy: 71 / 74 - 96.0%

                Confusion Matrix:

                [[12  0  0  0  0  0]
                 [ 1  7  0  0  0  0]
                 [ 0  0  9  0  0  1]
                 [ 0  0  0  1  0  0]
                 [ 0  0  0  0 28  0]
                 [ 0  0  1  0  0 14]]


                Classification Report:

                                          precision    recall  f1-score   support

                      chronic dermatitis       0.92      1.00      0.96        12
                           lichen planus       1.00      0.88      0.93         8
                        pityriasis rosea       0.90      0.90      0.90        10
                pityriasis rubra pilaris       1.00      1.00      1.00         1
                               psoriasis       1.00      1.00      1.00        28
                     seboreic dermatitis       0.93      0.93      0.93        15

                             avg / total       0.96      0.96      0.96        74



                **********************************************************
                *********** Logistic Regression -- One vs Rest ***********
                **********************************************************
                Accuracy: 71 / 74 - 96.0%

                Confusion Matrix:

                [[12  0  0  0  0  0]
                 [ 0  7  0  0  1  0]
                 [ 0  0  9  0  1  0]
                 [ 0  0  0  1  0  0]
                 [ 0  0  0  0 28  0]
                 [ 0  0  0  0  1 14]]


                Classification Report:

                                          precision    recall  f1-score   support

                      chronic dermatitis       1.00      1.00      1.00        12
                           lichen planus       1.00      0.88      0.93         8
                        pityriasis rosea       1.00      0.90      0.95        10
                pityriasis rubra pilaris       1.00      1.00      1.00         1
                               psoriasis       0.90      1.00      0.95        28
                     seboreic dermatitis       1.00      0.93      0.97        15

                             avg / total       0.96      0.96      0.96        74



                **************************************
                *********** Random Forests ***********
                **************************************
                Accuracy: 70 / 74 - 95.0%

                Confusion Matrix:

                [[12  0  0  0  0  0]
                 [ 0  7  0  0  1  0]
                 [ 0  0  9  0  0  1]
                 [ 0  0  0  1  0  0]
                 [ 0  0  0  0 28  0]
                 [ 0  0  1  0  1 13]]


                Classification Report:

                                          precision    recall  f1-score   support

                      chronic dermatitis       1.00      1.00      1.00        12
                           lichen planus       1.00      0.88      0.93         8
                        pityriasis rosea       0.90      0.90      0.90        10
                pityriasis rubra pilaris       1.00      1.00      1.00         1
                               psoriasis       0.93      1.00      0.97        28
                     seboreic dermatitis       0.93      0.87      0.90        15

                             avg / total       0.95      0.95      0.95        74



                ***************************************************************************************
                *********** Artificial Neural Network "Multi-Layer Perceptron Classifier" ***********
                ***************************************************************************************
                Accuracy: 73 / 74 - 99.0%

                Confusion Matrix:

                [[12  0  0  0  0  0]
                 [ 0  8  0  0  0  0]
                 [ 0  0  9  0  0  1]
                 [ 0  0  0  1  0  0]
                 [ 0  0  0  0 28  0]
                 [ 0  0  0  0  0 15]]


                Classification Report:

                                          precision    recall  f1-score   support

                      chronic dermatitis       1.00      1.00      1.00        12
                           lichen planus       1.00      1.00      1.00         8
                        pityriasis rosea       1.00      0.90      0.95        10
                pityriasis rubra pilaris       1.00      1.00      1.00         1
                               psoriasis       1.00      1.00      1.00        28
                     seboreic dermatitis       0.94      1.00      0.97        15

                             avg / total       0.99      0.99      0.99        74



                *******************************************
                *********** K Nearest Neighbors ***********
                *******************************************
                Accuracy: 72 / 74 - 97.0%

                Confusion Matrix:

                [[12  0  0  0  0  0]
                 [ 0  8  0  0  0  0]
                 [ 0  0 10  0  0  0]
                 [ 0  0  0  1  0  0]
                 [ 0  0  0  0 28  0]
                 [ 0  0  2  0  0 13]]


                Classification Report:

                                          precision    recall  f1-score   support

                      chronic dermatitis       1.00      1.00      1.00        12
                           lichen planus       1.00      1.00      1.00         8
                        pityriasis rosea       0.83      1.00      0.91        10
                pityriasis rubra pilaris       1.00      1.00      1.00         1
                               psoriasis       1.00      1.00      1.00        28
                     seboreic dermatitis       1.00      0.87      0.93        15

                             avg / total       0.98      0.97      0.97        74



                **********************************************
                *********** Support Vector Machine ***********
                **********************************************
                Accuracy: 72 / 74 - 97.0%

                Confusion Matrix: 
                [[12  0  0  0  0  0]
                 [ 0  8  0  0  0  0]
                 [ 0  0 10  0  0  0]
                 [ 0  0  0  1  0  0]
                 [ 0  0  0  0 28  0]
                 [ 0  0  2  0  0 13]]


                Classification Report: 
                                          precision    recall  f1-score   support

                      chronic dermatitis       1.00      1.00      1.00        12
                           lichen planus       1.00      1.00      1.00         8
                        pityriasis rosea       0.83      1.00      0.91        10
                pityriasis rubra pilaris       1.00      1.00      1.00         1
                               psoriasis       1.00      1.00      1.00        28
                     seboreic dermatitis       1.00      0.87      0.93        15

                             avg / total       0.98      0.97      0.97        74




Yay!
