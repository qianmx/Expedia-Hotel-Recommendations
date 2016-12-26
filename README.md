# Expedia-Hotel-Recommendations
**Key Word: Machine Learning-Classification, Algorithm, Feature Engineering, Python**

##Introduction
This project is a Kaggle competition. In this project, Expedia is challenging Kagglers to contextualize customer data and predict the likelihood a user will stay at 100 different hotel groups. It aims to predict the top 5 hotel clusters that a customer is most likely to book.

_Detailed project and data description can be found at (https://www.kaggle.com/c/expedia-hotel-recommendations)._

##Machine Learning Approach
We tested different classificaiton algorithm: LDA, QDA, Logistic Regression, AdaBoost, Gradient Boosting, Decision Trees, Frandom Forest, Bagging. 

And in order to get top 5 clusters for each observation, for each algorithm, in the code we: 

* Loop across each unique hotel_cluster.
* Train a classifier using 2-fold cross validation.
* Extract the probabilities from the classifier that the row is in the unique hotel_cluster
* Combine all the probabilities.
* For each row, find the 5 largest probabilities, and assign those hotel_cluster values as predictions.
* Compute accuracy using mapk.

##Aggregation Approach-Customized Algorithm
The ML Approach didn't give us desired prediction accuracy in this case. we decided to use alternative means of achieving better classification accuracy. 

This approach relies on the count of historical booking and clicking history. Since the training data is huge, this approach tries to exploit it by predicting the clusters based on frequency the particular cluster was booked for same search criteria. That is, the greater the number of times a hotel cluster say ‘H’ is booked for say Destination id ’D’, the greater probability it has of being selected in the future too. The relevance of clusters is decided by their booking and clicking frequency on the historical data.

##Output
![alt tag]()

_Detailed project report can be found in this repository(Report.pdf)_

