import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier,GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# feature information
category_lst = ['site_name','posa_continent','user_location_country',
                'user_location_region','user_location_city',
                'channel','srch_destination_id','srch_destination_type_id',
                'hotel_continent','hotel_country','hotel_market',
                'hotel_cluster']

feature_col = ['site_name','posa_continent','user_location_country','user_location_region',
               'user_location_city','is_mobile','is_package',
               'channel','srch_adults_cnt','srch_children_cnt','srch_rm_cnt','srch_destination_id',
               'srch_destination_type_id','is_booking','cnt','hotel_continent','hotel_country',
               'hotel_market','month','day','dow','stay_days','before_days','pca1','pca2','pca3']

# selected_features = ['srch_destination_id','is_booking','before_days','pca1','pca2','pca3','cnt']
selected_features = ['srch_destination_id','is_booking']

# model selection parameters
depth = 17
neighbour = 5
n = 10
estimator = 26
power = 1

MODELS = {"Logistic Regression": LogisticRegression(),
              "QDA": QuadraticDiscriminantAnalysis(),
              "LDA": LinearDiscriminantAnalysis(),
              "Decission Tree Classification": DecisionTreeClassifier(criterion="gini", max_depth=17),
              "Bagging": BaggingClassifier(n_estimators=29),
              "Ada Boost": AdaBoostClassifier(learning_rate=0.1 ** power),
              "Random Forest": RandomForestClassifier(n_estimators=26),
              "Gradient Boosting": GradientBoostingClassifier(n_estimators=10)}

# read files as data frame
DEST_FILE = "destinations.csv"
train = pd.read_csv("training_sub.0.1.csv")
test = pd.read_csv("test_sub.0.1.csv")

