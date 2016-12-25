import pandas as pd
import numpy as np
from sklearn.cross_validation import KFold
import sys
import metrics


def most_relevant(group, n_max=5):
    relevance = group['relevance'].values
    hotel_cluster = group['hotel_cluster'].values
    relevant = hotel_cluster[np.argsort(relevance)[::-1]][:n_max]
    return relevant

train_file = sys.argv[1]
train_data = pd.read_csv(train_file)

for click in range(5,70,5):
    click1 = click/100.0
    kf = KFold(len(train_data), n_folds=3, shuffle = True)
    for train_index, test_index in kf:
        train= train_data.iloc[train_index,:]
        test= train_data.iloc[test_index,:]
        grp_agg = train.groupby(['srch_destination_id','hotel_cluster'])['is_booking'].grp_agg(['sum','count'])
        grp_agg.reset_index(inplace=True)
        grp_agg = grp_agg.groupby(['srch_destination_id','hotel_cluster']).sum().reset_index()
        grp_agg['count'] -= grp_agg['sum']
        grp_agg = grp_agg.rename(columns={'sum':'bookings','count':'clicks'})
        grp_agg['relevance'] = grp_agg['bookings'] + click1 * grp_agg['clicks']
        most_rel = grp_agg.groupby(['srch_destination_id']).apply(most_relevant)
        most_rel = pd.DataFrame(most_rel).rename(columns={0:'hotel_cluster'})
        test = test.merge(most_rel, how='left',left_on=['srch_destination_id'],right_index=True)
        test=test.dropna()
        preds=[]
        for index, row in test.iterrows():
            preds.append(row['hotel_cluster_y'])
        target = [[l] for l in test["hotel_cluster_x"]]

        print metrics.mapk(target, preds, k=5)
