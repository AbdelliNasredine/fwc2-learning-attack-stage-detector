import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier


def main(dataset_name: str = 'dapt20'):
    print(f'** staring with {dataset_name}')

    train_set = pd.read_csv(f'./data/{dataset_name}/supervised/train.csv')
    test_set = pd.read_csv(f'./data/{dataset_name}/supervised/test.csv')

    scaler = StandardScaler()
    label_encoder = LabelEncoder()

    X_train, y_train = train_set.drop(columns=['label']), train_set['label']
    X_test, y_test = test_set.drop(columns=['label']), test_set['label']

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.fit_transform(y_test)

    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
        "CatBoost": CatBoostClassifier(iterations=100, random_seed=42, verbose=False)
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        print(f"\n{'*' * 20}")
        print(f"Classification Report for {model_name}:")
        print(classification_report(y_test, y_pred, digits=3))
        print(f"{'*' * 20}\n")


if __name__ == "__main__":
    for ds in ['dapt20', 'scvic21']:
        main(ds)
