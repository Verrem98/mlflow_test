import argparse

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb
import matplotlib as mpl

import mlflow
import mlflow.xgboost

mpl.use("Agg")


def column_comma_to_point(column):
    return column.str.replace(',', '.')


def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost")
    parser.add_argument(
        "--data",
        type=str,
        default='',
        help="path to the csv file you want to use",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.3,
        help="learning rate to update step size at each boosting step (default: 0.3)",
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=1.0,
        help="subsample ratio of columns when constructing each tree (default: 1.0)",
    )
    parser.add_argument(
        "--subsample",
        type=float,
        default=1.0,
        help="subsample ratio of the training instances (default: 1.0)",
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="number of training iterations",
    )

    parser.add_argument(
        "--alpha",
        type=int,
        default=0,
        help="L1 regularization term on weights ",
    )

    parser.add_argument(
        "--max-depth",
        type=int,
        default=1000,
        help="max tree depth",
    )

    return parser.parse_args()


def main():
    # parse command-line arguments
    args = parse_args()

    df = pd.read_csv(args.data,
                     encoding='unicode escape',
                     sep=';',
                     na_values='.'
                     )

    df['meer_man'] = df['a_man'] > df['a_vrouw']
    df['g_afs_sc'] = column_comma_to_point(df['g_afs_sc'])
    # df['p_wmo_t'] = column_comma_to_point(df['p_wmo_t'])
    y = df['meer_man']
    X = df[['a_gesch', 'a_geb', 'g_afs_sc', 'p_wmo_t', 'a_bed_a']].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42)

    # enable auto logging
    mlflow.xgboost.autolog()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    with mlflow.start_run():
        # train model
        params = {
            "objective": "multi:softprob",
            "num_class": 2,
            "learning_rate": args.learning_rate,
            "eval_metric": "mlogloss",
            "colsample_bytree": args.colsample_bytree,
            "subsample": args.subsample,
            "seed": 42,
            "alpha": args.alpha,
            "max_depth": args.max_depth
        }
        model = xgb.train(params, dtrain,
                          evals=[(dtrain, "train"), (dtest, 'test')],
                          num_boost_round=args.iterations)

        # evaluate model
        y_proba = model.predict(dtest)
        y_pred = y_proba.argmax(axis=1)
        loss = log_loss(y_test, y_proba)
        acc = accuracy_score(y_test, y_pred)

        # log metrics

        mlflow.log_metrics({"log_loss": loss, "accuracy_score": acc})


if __name__ == "__main__":
    main()
