# machine_learning_is_fun

Just a small program I made for fun while reading about machine learning.  Took me a few hours to make it.
The program attempts to use clinical findings and histopathological features to classify a dermatologic condition as one of the following:
psoriasis, seboreic dermatitis, lichen planus, pityriasis rosea, chronic dermatitis, pityriasis rubra pilaris


The dataset is one of the many publicly available UC Irvine datasets and additional information can be found  here:
https://archive.ics.uci.edu/ml/datasets/dermatology


Obvs you need to install dependencies such as sklearn, pandas, numpy, etc.


Here is a sample of its output.

         *********** Linear Regression ***********
         70 / 74 - 95.0%
         RMSE: 17%


         *********** Logistic Regression -- One vs Rest ***********
         70 / 74 - 95.0%
         RMSE: 12%


         *********** Random Forests ***********
         72 / 74 - 97.0%
         RMSE: 9%


         *********** Artificial Neural Network "Multi-Layer Perceptron Classifier" ***********
         73 / 74 - 99.0%
         RMSE: 7%



Yay!
