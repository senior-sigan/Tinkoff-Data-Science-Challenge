import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import random
import datetime

def prepare_labels(train, test):
    # нужно категориальные признаки пометить числами, а в тестовой выборке есть новые данные к сожалению
    all_data = pd.concat([train, test])
    for cat in ['education', 'gender', 'job_position', 'marital_status', 'education', 'tariff_id', 'living_region']:
        all_data[cat] = LabelEncoder().fit_transform(all_data[cat])
    return all_data.ix[train.index], all_data.ix[test.index]

def predict_proba(X, y, test, count):
    predictions = pd.DataFrame()
    predictions["_ID_"] = test.index
    for i in range(count):
        seed = random.randint(1, 100000000)
        depth = random.randint(15, 25)
        clf = RandomForestClassifier(n_estimators=300, max_depth=depth, random_state=seed, n_jobs=4)
        start_time = datetime.datetime.now()

        clf.fit(X, y)
        predictions["rfc_{}".format(i)] = clf.predict_proba(test)[:, 1]

        end_time = datetime.datetime.now()
        print("{}. RandomForestClassifier(seed={}, depth={}). Time to fit_predict: {}".format(i, seed, depth, end_time-start_time))
    return predictions

def main():
    train = pd.read_csv('data/credit_train_final_optimized.csv.gz', index_col='client_id')
    test = pd.read_csv('data/credit_test_final_optimized.csv.gz', index_col='client_id')
    print(train.columns)

    tr, te = prepare_labels(train, test)
    te.fillna(0, inplace=True)
    te.drop(['living_region', 'open_account_flg'], axis=1, inplace=True)

    X = tr.drop(['open_account_flg', 'living_region'], axis = 1)
    y = tr['open_account_flg']

    preds = predict_proba(X, y, te, 100)
    preds.to_csv('submissions/rfc_ensemle.csv', index=False)

if __name__ == '__main__':
    main()
