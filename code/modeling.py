import ml_metrics as metrics
import warnings
warnings.filterwarnings("ignore")
from data_process import *


def find_top_n(row, n):
    return list(row.nlargest(n).index)


def testing_mapk_calculation(train_x, train_y_dummy, test_x, test_y, num):

    for algo in MODELS.keys():

        model = MODELS[algo]

        all_probs = []
        for col in train_y_dummy.columns:
            y = train_y_dummy[col]
            model.fit(train_x, y)

            probs = []
            preds = model.predict_proba(test_x)
            probs.append([p[1] for p in preds])
            all_probs.append(probs[0])

        prediction_frame = pd.DataFrame(all_probs).T
        preds = []
        for index, row in prediction_frame.iterrows():
            preds.append(find_top_n(row,num))
        target = [[l] for l in test_y]

        print "(%s) mapk:%s" % (algo, metrics.mapk(target, preds, k=num))


def main(num):
    train_x, train_y = training_processor(train)
    test_x, test_y = testing_processor(test)
    train_Y_dummy = pd.get_dummies(train_y)
    print "got dummy"
    return testing_mapk_calculation(train_x, train_Y_dummy, test_x, test_y, num)

if __name__ == "__main__":
    main(5)  # this return the MAPk accuracy score
