import warnings
warnings.filterwarnings("ignore")
from data_process import *
import numpy as np
import matplotlib.pylab as plt

# feature importance


def feature_importance(df):
    x, y = training_processor(df)

    for algo in MODELS.keys():
        model = MODELS[algo]
        # print x.shape
        model.fit(x, y)
        importances = model.feature_importances_
        std = np.std([tree.feature_importances_ for tree in model.estimators_],axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")
        for f in range(x.shape[1]):
            print("%d. %s (%f)" % (f + 1, selected_features[indices[f]], importances[indices[f]]))

        for i in range(len(feature_col)):
            print 'feature ', i+1, ':', feature_col[i]

        #  Plot the feature importances of the forest

        #  use horizontal bar
        Indices = indices
        val = importances[indices].tolist()
        pos = range(x.shape[1])  # the bar centers on the y axis
        plt.figure(1)
        plt.barh(pos, val, align='center')  # notice the 'height' argument
        plt. yticks(pos, [selected_features[Indices[f]] for f in range(x.shape[1])])
        plt.gca().axvline(0, color='k', lw=3)  # poor man's zero level
        plt.xlabel('Performance')
        plt.title('horizontal bar chart using matplotlib')
        plt.grid(True)
        plt. show()

        # use vertical bar
        plt.figure()
        plt.title("Feature importances")
        plt.bar(range(x.shape[1]), importances[indices],color="r", yerr=std[indices], align="center")
        plt.xticks(range(x.shape[1]), [selected_features[Indices[f]] for f in range(x.shape[1])], size='small',rotation='vertical')
        plt.xlim([-1, x.shape[1]])
        plt.show()

if __name__ == "__main__":
    selected_features = feature_col
    feature_importance(train)
