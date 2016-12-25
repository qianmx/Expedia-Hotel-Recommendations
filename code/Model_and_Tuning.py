import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, cross_val_score
from data_process import *
import numpy as np
warnings.filterwarnings("ignore")

# model selection


def run_model_methods(train_x, train_y):
    """ given training data, print best performed model according to
    cross-validation mis-classification rate , print out all the mis-rate"""

    models = {"Logistic Regression": LogisticRegression(),
              "QDA": QuadraticDiscriminantAnalysis(),
              "LDA": LinearDiscriminantAnalysis(),
              "Decission Tree Classification": DecisionTreeClassifier(criterion="gini", max_depth=depth),
              "Bagging": BaggingClassifier(n_estimators=29),
              "Ada Boost": AdaBoostClassifier(learning_rate=0.1 ** power),
              "Random Forest": RandomForestClassifier(n_estimators=estimator),
              "Gradient Boosting": GradientBoostingClassifier(n_estimators=10)}

    score_list = []
    model_list = []
    for algo in models.keys():
        model = models[algo]
        model.fit(train_x, train_y)
        model_list.append(algo)

        kfold = KFold(n_splits=n, shuffle=True)
        mis = 1 - abs(np.mean(cross_val_score(model, train_x, train_y, cv=kfold, scoring='accuracy')))
        score_list.append(mis)

    print "Misclassification Rate by %s: %s" % (model_list[score_list.index(max(score_list))], max(score_list))
    print model_list
    print score_list


# parameter tuning

kfold = KFold(n_splits=10, shuffle=True)

""" Tree"""


def tree():
    crit = "gini"
    acc = []
    Depth = range(1, 30)
    for depth in Depth:
        classifier = DecisionTreeClassifier(criterion=crit, max_depth=depth)
        classifier.fit(train_x, train_y)
        acc.append(abs(np.mean(cross_val_score(classifier, train_x, train_y, cv=kfold, scoring='accuracy'))))

    print "Accuracy rate by depth %s is %s" % (Depth[acc.index(max(acc))], max(acc))
    plt.plot(Depth, acc)
    plt.ylabel('Mean cv-accuracy')
    plt.xlabel('Depth')
    plt.title("Decision Tree")
    plt.show()

""" Bagging """


def bagging():
    acc = []
    Estimators = range(2, 30)
    kfold = KFold(n_splits=10, shuffle=True)
    for estimator in Estimators:
        model = BaggingClassifier(n_estimators=estimator)
        model.fit(train_x, train_y)
        acc.append(abs(np.mean(cross_val_score(model, train_x, train_y, cv=kfold, scoring='accuracy'))))
    print "Accuracy rate by depth %s is %s" % (Estimators[acc.index(max(acc))], max(acc))

    # prediction
    plt.plot(Estimators, acc)
    plt.ylabel('Mean cv-accuracy')
    plt.xlabel('estimator')
    plt.title("Bagging")
    plt.show()

""" Ada Boost """


def ada_boost():
    rng = np.random.RandomState(1)
    acc = []
    rates = range(1, 10)
    for rate in rates:
        classifier2 = AdaBoostClassifier(learning_rate=0.1 ** rate)
        classifier2.fit(train_x, train_y)
        acc.append(abs(np.mean(cross_val_score(classifier2, train_x, train_y, cv=kfold, scoring='accuracy'))))
    print max(acc)
    print rates[acc.index(max(acc))]

    # prediction
    plt.plot(rates, acc)
    plt.ylabel('Mean cv-accuracy')
    plt.xlabel('rates')
    plt.title("Ada Boost")
    plt.show()

""" Random Forest """


def random_forest():
    acc = []
    Estimators = range(2, 30)
    kfold = KFold(n_splits=10, shuffle=True)
    for estimator in Estimators:
        model = RandomForestClassifier(n_estimators=estimator)
        model.fit(train_x, train_y)
        acc.append(abs(np.mean(cross_val_score(model, train_x, train_y, cv=kfold, scoring='accuracy'))))
    print "Accuracy rate by depth %s is %s" % (Estimators[acc.index(max(acc))], max(acc))

    # prediction
    plt.plot(Estimators, acc)
    plt.ylabel('Mean cv-accuracy')
    plt.xlabel('estimator')
    plt.show()

""" Gradient Boosting """


def gradient_boosting():
    acc = []
    P = range(1, 5)
    kfold = KFold(n_splits=10, shuffle=True)
    for p in P:
        model = GradientBoostingClassifier(learning_rate=0.1 ** p)
        model.fit(train_x, train_y)
        acc.append(abs(np.mean(cross_val_score(model, train_x, train_y, cv=kfold, scoring='accuracy'))))
    print "Accuracy rate by depth %s is %s" % (P[acc.index(max(acc))], max(acc))

    # prediction
    plt.plot(P, acc)
    plt.ylabel('Mean cv-accuracy')
    plt.xlabel('learning rate power vs mean cv-accuracy(base 0.1)')
    plt.show()

if __name__ == "__main__":
    train_x, train_y = training_processor(train)
    test_x, test_y = testing_processor(test)

    run_model_methods(train_x, train_y)
