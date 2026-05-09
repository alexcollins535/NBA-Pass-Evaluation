import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, log_loss, classification_report, average_precision_score, roc_auc_score
from pathlib import Path
import matplotlib.pyplot as plt

# File Paths
BASE_DIR = Path.cwd()
PARQUET_DIR = BASE_DIR / 'NBA_SportVU_Parquet'
CACHE_DIR = BASE_DIR / 'Dataset_Cache'
GRAPH_CACHE_DIR = BASE_DIR / 'Graph_Cache'
MODEL_DATA_DIR = BASE_DIR / 'Model_Data'

JSON_PATH = MODEL_DATA_DIR / 'data_splits.json'
OUTPUT_FILE = BASE_DIR / 'all_games_passes_3.0.csv'
MODEL_PATH = MODEL_DATA_DIR / 'gbc_pass_model_real_only.joblib'


if __name__ == '__main__':
    # Importing File Splits
    with open(JSON_PATH, 'r') as f:
        json_data = json.load(f)

    train_files = json_data['train']
    val_files = json_data['val']
    test_files = json_data['test']

    df = pd.read_csv(OUTPUT_FILE)

    # Real passes only, game-level splits
    real_df = df[df['is_synthetic'] == 0]
    train_df = real_df[real_df['game_id'].isin(train_files)]
    test_df  = real_df[real_df['game_id'].isin(test_files)]

    feature_cols = [
        'pass_distance',
        'pass_angle',
        'nearest_defender_dist',
        'max_defender_lane_depth',
        'pass_trajectory_crowding',
        'passer_velocity',
        'passer_nearest_defender_dist',
        'receiver_velocity',
        'receiver_nearest_defender_dist',
        'receiver_defender_closing_speed',
        'receiver_separation_ratio',
        'offensive_spacing',
        'defenders_in_lane'
    ]

    X_train, y_train = train_df[feature_cols].values, train_df['success'].values
    X_test,  y_test  = test_df[feature_cols].values,  test_df['success'].values

    # Load model 
    model = joblib.load(MODEL_PATH)

    # Evaluate 
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print('\n' + '=' * 45)
    print('     CLEAN TEST SET EVALUATION RESULTS')
    print('=' * 45)
    print(f'  Accuracy      : {accuracy_score(y_test, y_pred):.4f}')
    print(f'  Log Loss      : {log_loss(y_test, y_prob):.4f}')
    print(f'  ROC-AUC       : {roc_auc_score(y_test, y_prob):.4f}')
    print(f'  Avg Precision : {average_precision_score(y_test, y_prob):.4f}')
    print(f'  Mean pred prob: {y_prob.mean():.3f}  (actual: {y_test.mean():.3f})')
    print('=' * 45)
    print(classification_report(y_test, y_pred, target_names=['Incomplete', 'Complete']))

    #  Extract importances 
    importances = model.feature_importances_

    # Create DataFrame for nicer plotting
    fi_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=True)

    #  Plot horizontal bar chart 
    plt.figure(figsize=(8, 6))
    plt.barh(fi_df['feature'], fi_df['importance'])
    plt.xlabel('Feature Importance')
    plt.title('Gradient Boosting Feature Importances (Pass Model)')
    plt.tight_layout()
    plt.show()