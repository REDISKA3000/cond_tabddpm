import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import explained_variance_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
# from sklearn.utils._testing import ignore_warnings
# from sklearn.exceptions import ConvergenceWarning
import logging
from prdc import compute_prdc
from tqdm import tqdm
import json
import os

CATEGORICAL = "categorical"
CONTINUOUS = "continuous"

_MODELS = {
    'binclass': [
        {
            'class': XGBClassifier,  # 36
            'kwargs': {
                'n_estimators': [10, 50, 100],
                'min_child_weight': [1, 10],
                'max_depth': [5, 10, 20],
                'gamma': [0.0, 1.0],
                'objective': ['binary:logistic'],
                'nthread': [-1],
                'tree_method': ['gpu_hist']
            },
        }

    ]
}


def feat_transform(data, info, label_encoder=None, encoders=None, cmax=None, cmin=None, conditional_columns=[]):
    num_col_idx = info['num_col_idx']
    cat_col_idx = info['cat_col_idx']
    target_col_idx = info['target_col_idx']
    print(conditional_columns)
    print('cat idx init', cat_col_idx)
    print('num idx init', num_col_idx)
    print('target idx init', target_col_idx)
    for i in conditional_columns:
        for j in range(len(cat_col_idx)):
            if cat_col_idx[i] < cat_col_idx[j]:
                cat_col_idx[j] = cat_col_idx[j] - 1

        for j in range(len(num_col_idx)):
            if cat_col_idx[i] < num_col_idx[j]:
                num_col_idx[j] = num_col_idx[j] - 1

        for j in range(len(target_col_idx)):
            if cat_col_idx[i] < target_col_idx[j]:
                target_col_idx[j] = target_col_idx[j] - 1

        cat_col_idx.pop(i)

    print('cat idx', cat_col_idx)
    print('num idx', num_col_idx)
    print('target idx', target_col_idx)

    print(cat_col_idx)
    # display(pd.DataFrame(data))

    num_cols = len(num_col_idx + cat_col_idx + target_col_idx)
    features = []

    if not encoders:
        encoders = dict()

    print(data.shape)
    labels = []
    for idx in range(num_cols):
        print(idx)
        col = data[:, idx]
        if idx in target_col_idx:

            if info['task_type'] != 'regression':

                if not label_encoder:
                    label_encoder = LabelEncoder()
                    label_encoder.fit(col)

                encoded_labels = label_encoder.transform(col)
                print('labels', encoded_labels)
                labels = encoded_labels
            else:
                col = col.astype(np.float32)
                labels = col.astype(np.float32)

            continue

        if idx in num_col_idx:
            col = col.astype(np.float32)

            if not cmin:
                cmin = col.min()

            if not cmax:
                cmax = col.max()

            if cmin >= 0 and cmax >= 1e3:
                feature = np.log(np.maximum(col, 1e-2))

            else:
                feature = (col - cmin) / (cmax - cmin) * 5

        elif idx in cat_col_idx:
            encoder = encoders.get(idx)
            col = col.reshape(-1, 1)
            if encoder:
                feature = encoder.transform(col)
            else:
                encoder = OneHotEncoder(
                    sparse_output=False, handle_unknown='ignore')
                encoders[idx] = encoder
                feature = encoder.fit_transform(col)

        features.append(feature)
    features = np.column_stack(features)
    return features, labels, label_encoder, encoders, cmax, cmin


def prepare_ml_problem(train, test, info, conditional_columns=[]):

    train_X, train_y, label_encoder, encoders, cmax, cmin = feat_transform(
        train, info, conditional_columns=conditional_columns)
    test_X, test_y, _, _, _, _ = feat_transform(
        test, info, label_encoder, encoders, cmax, cmin)

    print('train x', train_X.shape)

    total_train_num = train_X.shape[0]
    val_num = int(total_train_num / 9)

    total_train_idx = np.arange(total_train_num)
    np.random.shuffle(total_train_idx)
    train_idx = total_train_idx[val_num:]
    val_idx = total_train_idx[:val_num]

    print(train_idx, val_idx, val_idx)
    print(train_X[val_idx])
    print(train_y)
    val_X, val_y = train_X[val_idx], train_y[val_idx]
    train_X, train_y = train_X[train_idx], train_y[train_idx]

    model = _MODELS[info['task_type']]

    return train_X, train_y, val_X, val_y, test_X, test_y, model


class FeatureMaker:

    def __init__(self, metadata, label_column='label', label_type='int', sample=50000):
        self.columns = metadata['columns']
        self.label_column = label_column
        self.label_type = label_type
        self.sample = sample
        self.encoders = dict()

    def make_features(self, data):
        data = data.copy()
        np.random.shuffle(data)
        data = data[:self.sample]

        features = []
        labels = []

        for index, cinfo in enumerate(self.columns):
            col = data[:, index]
            if cinfo['name'] == self.label_column:
                if self.label_type == 'int':
                    labels = col.astype(int)
                elif self.label_type == 'float':
                    labels = col.astype(float)
                else:
                    assert 0, 'unkown label type'
                continue

            if cinfo['type'] == CONTINUOUS:
                cmin = cinfo['min']
                cmax = cinfo['max']
                if cmin >= 0 and cmax >= 1e3:
                    feature = np.log(np.maximum(col, 1e-2))

                else:
                    feature = (col - cmin) / (cmax - cmin) * 5

            else:
                if cinfo['size'] <= 2:
                    feature = col

                else:
                    encoder = self.encoders.get(index)
                    col = col.reshape(-1, 1)
                    if encoder:
                        feature = encoder.transform(col)
                    else:
                        encoder = OneHotEncoder(
                            sparse_output=False, handle_unknown='ignore')
                        self.encoders[index] = encoder
                        feature = encoder.fit_transform(col)

            features.append(feature)

        features = np.column_stack(features)

        return features, labels


def _prepare_ml_problem(train, val, test, metadata, eval):
    fm = FeatureMaker(metadata)
    x_trains, y_trains = [], []

    for i in train:
        x_train, y_train = fm.make_features(i)
        x_trains.append(x_train)
        y_trains.append(y_train)

    x_val, y_val = fm.make_features(val)
    if eval is None:
        x_test = None
        y_test = None
    else:
        x_test, y_test = fm.make_features(test)
    model = _MODELS[metadata['problem_type']]

    return x_trains, y_trains, x_val, y_val, x_test, y_test, model


def _weighted_f1(y_test, pred):
    report = classification_report(y_test, pred, output_dict=True)
    classes = list(report.keys())[:-3]
    proportion = [report[i]['support'] / len(y_test) for i in classes]
    weighted_f1 = np.sum(list(map(
        lambda i, prop: report[i]['f1-score'] * (1-prop)/(len(classes)-1), classes, proportion)))
    return weighted_f1


# @ignore_warnings(category=ConvergenceWarning)
def _evaluate_binary_classification(train, test, info, conditional_columns=[]):
    x_trains, y_trains, x_valid, y_valid, x_test, y_test, classifiers = prepare_ml_problem(
        train, test, info, conditional_columns=conditional_columns)

    unique_labels = np.unique(y_trains)

    best_f1_scores = []
    best_weighted_scores = []
    best_auroc_scores = []
    best_acc_scores = []
    best_avg_scores = []
    print('rrfrfrfr')
    for model_spec in classifiers:

        model_class = model_spec['class']
        model_kwargs = model_spec.get('kwargs', dict())
        model_repr = model_class.__name__

        unique_labels = np.unique(y_trains)
        print('kwargs', model_kwargs)

        param_set = list(ParameterGrid(model_kwargs))

        results = []
        for param in tqdm(param_set):
            print('param', param)
            param.update({'tree_method': 'hist',
                          'device': 'cpu'
                          })
            model = model_class(**param)
            print('fefeefef')
            print('fefeefef')
            print(model)
            print(x_trains.shape, y_trains.shape)

            try:
                model.fit(x_trains, y_trains)

            except ValueError:
                print('error')
                pass

            if len(unique_labels) == 1:
                pred = [unique_labels[0]] * len(x_valid)
                pred_prob = np.array([1.] * len(x_valid))
            else:
                pred = model.predict(x_valid)
                pred_prob = model.predict_proba(x_valid)
            print(pred)
            binary_f1 = f1_score(y_valid, pred, average='binary')
            weighted_f1 = _weighted_f1(y_valid, pred)
            acc = accuracy_score(y_valid, pred)
            precision = precision_score(y_valid, pred, average='binary')
            recall = recall_score(y_valid, pred, average='binary')
            macro_f1 = f1_score(y_valid, pred, average='macro')

            # auroc
            size = 2
            rest_label = set(range(size)) - set(unique_labels)
            tmp = []
            j = 0
            for i in range(size):
                if i in rest_label:
                    tmp.append(np.array([0] * y_valid.shape[0])[:, np.newaxis])
                else:
                    try:
                        tmp.append(pred_prob[:, [j]])
                    except:
                        tmp.append(pred_prob[:, np.newaxis])
                    j += 1
            roc_auc = roc_auc_score(np.eye(size)[y_valid], np.hstack(tmp))

            results.append(
                {
                    "name": model_repr,
                    "param": param,
                    "binary_f1": binary_f1,
                    "weighted_f1": weighted_f1,
                    "roc_auc": roc_auc,
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "macro_f1": macro_f1
                }
            )

        # test the best model
        results = pd.DataFrame(results)
        results['avg'] = results.loc[:, ['binary_f1',
                                         'weighted_f1', 'roc_auc']].mean(axis=1)
        best_f1_param = results.param[results.binary_f1.idxmax()]
        best_weighted_param = results.param[results.weighted_f1.idxmax()]
        best_auroc_param = results.param[results.roc_auc.idxmax()]
        best_acc_param = results.param[results.accuracy.idxmax()]
        best_avg_param = results.param[results.avg.idxmax()]

        def _calc(best_model):
            best_scores = []

            best_model.fit(x_trains, y_trains)

            if len(unique_labels) == 1:
                pred = [unique_labels[0]] * len(x_test)
                pred_prob = np.array([1.] * len(x_test))
            else:
                pred = best_model.predict(x_test)
                pred_prob = best_model.predict_proba(x_test)

            binary_f1 = f1_score(y_test, pred, average='binary')
            weighted_f1 = _weighted_f1(y_test, pred)
            acc = accuracy_score(y_test, pred)
            precision = precision_score(y_test, pred, average='binary')
            recall = recall_score(y_test, pred, average='binary')
            macro_f1 = f1_score(y_test, pred, average='macro')

            # auroc
            size = 2
            rest_label = set(range(size)) - set(unique_labels)
            tmp = []
            j = 0
            for i in range(size):
                if i in rest_label:
                    tmp.append(np.array([0] * y_test.shape[0])[:, np.newaxis])
                else:
                    try:
                        tmp.append(pred_prob[:, [j]])
                    except:
                        tmp.append(pred_prob[:, np.newaxis])
                    j += 1
            try:
                roc_auc = roc_auc_score(np.eye(size)[y_test], np.hstack(tmp))
            except ValueError:
                tmp[1] = tmp[1].reshape(20000, 1)
                roc_auc = roc_auc_score(np.eye(size)[y_test], np.hstack(tmp))

            best_scores.append(
                {
                    "name": model_repr,
                    # "param": param,
                    "binary_f1": binary_f1,
                    "weighted_f1": weighted_f1,
                    "roc_auc": roc_auc,
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "macro_f1": macro_f1
                }
            )

            return pd.DataFrame(best_scores)

        def _df(dataframe):
            return {
                "name": model_repr,
                "binary_f1": dataframe.binary_f1.values[0],
                "roc_auc": dataframe.roc_auc.values[0],
                "weighted_f1": dataframe.weighted_f1.values[0],
                "accuracy": dataframe.accuracy.values[0],
            }

        best_f1_scores.append(_df(_calc(model_class(**best_f1_param))))
        best_weighted_scores.append(
            _df(_calc(model_class(**best_weighted_param))))
        best_auroc_scores.append(_df(_calc(model_class(**best_auroc_param))))
        best_acc_scores.append(_df(_calc(model_class(**best_acc_param))))
        best_avg_scores.append(_df(_calc(model_class(**best_avg_param))))

    return best_f1_scores, best_weighted_scores, best_auroc_scores, best_acc_scores, best_avg_scores

# @ignore_warnings(category=ConvergenceWarning)


def compute_diversity(train, fake):
    nearest_k = 5
    if train.shape[0] >= 50000:
        num = np.random.randint(0, train.shape[0], 50000)
        real_features = train[num]
        fake_features_lst = [i[num] for i in fake]
    else:
        num = train.shape[0]
        real_features = train[:num]
        fake_features_lst = [i[:num] for i in fake]
    scores = []
    for i, data in enumerate(fake_features_lst):
        fake_features = data
        metrics = compute_prdc(real_features=real_features,
                               fake_features=fake_features,
                               nearest_k=nearest_k)
        metrics['i'] = i
        scores.append(metrics)
    return pd.DataFrame(scores).mean(axis=0), pd.DataFrame(scores).std(axis=0)


_EVALUATORS = {
    'binclass': _evaluate_binary_classification,
    # 'multiclass': _evaluate_multi_classification,
    # 'regression': _evaluate_regression
}


def get_evaluator(problem_type):
    return _EVALUATORS[problem_type]


def compute_scores(train, test, synthesized_data, metadata, eval):
    a, b, c = _EVALUATORS[metadata['problem_type']](
        train=train, test=test, fake=synthesized_data, metadata=metadata, eval=eval)
    if eval is None:
        return a.mean(axis=0), a.std(axis=0), a[['name', 'param']]
    else:
        return a.mean(axis=0), a.std(axis=0)


def MLE(train_path, test_path, dataname, model, conditional_columns=[]):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    test = test[train.columns.tolist()].to_numpy()
    train = train.to_numpy()

    with open(f'/content/drive/MyDrive/adult/info.json', 'r') as f:
        info = json.load(f)

    task_type = info['task_type']

    evaluator = get_evaluator(task_type)
    print(evaluator)

    best_f1_scores, best_weighted_scores, best_auroc_scores, best_acc_scores, best_avg_scores = evaluator(
        train, test, info, conditional_columns=conditional_columns)
    print(best_f1_scores, best_weighted_scores,
          best_auroc_scores, best_acc_scores, best_avg_scores)

    overall_scores = {}
    for score_name in ['best_f1_scores', 'best_weighted_scores', 'best_auroc_scores', 'best_acc_scores', 'best_avg_scores']:
        overall_scores[score_name] = {}

        scores = eval(score_name)
        for method in scores:
            name = method['name']
            method.pop('name')
            overall_scores[score_name][name] = method

    # if not os.path.exists(f'/content/drive/MyDrive/results/evall/mle/{dataname}'):
    #     os.makedirs(f'/content/drive/MyDrive/results/eval/mle/{dataname}')

    cond_cols = list(map(lambda x: str(x), conditional_columns))
    save_path = f'/content/drive/MyDrive/results/eval/{model}_cond_cols_'+'_'.join(
        cond_cols)+'.json'
    print('Saving scores to ', save_path)
    with open(save_path, "w") as json_file:
        json.dump(overall_scores, json_file, indent=4, separators=(", ", ": "))
