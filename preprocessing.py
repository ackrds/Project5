import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from math import prod


def split_df(year, month):

    df = pd.read_csv('tennis_preprocessed.csv')
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    df = df[df['tourney_date'] > '1990-01-01']
    df = df.sort_values('tourney_date')

    num_features = [
                'Player Momentum Difference',
                'Player Deviation Difference',
                'Player Glicko Difference',

                'Player Rank Points Difference',
                'Player Rank Difference',

                'Player Age Difference',
                'Player Height Difference',

                # 'Player bpSaved% Avg Difference',
                # 'Player ServicePointsWon% Avg Difference',
                # 'Player ReturnPointsWon% Avg Difference',

                'Player Completeness Avg Difference',
                'Player Serve Advantage Avg Difference',

                'Total Matches Difference',
                'Wins Difference',
                'Win Rate Difference',
                'Titles Difference',

                'Grand Slam Win Rate Difference',
                'Masters Win Rate Difference',
                'Qualifiers Win Rate Difference',

                'Player Head2Head Wins', 
                'Player Head2Head Games',

                ]
    cat_features = [
                'isMasters',
                'isQualifiers',
                'isGrandSlam',

                'Round', 

                'Surface Clay',
                'Surface Grass',
                'Surface Hard',
                    ]
    test_columns = [
                    'player_name',
                    'opponent_name',
                    'Player 1 Odd', 
                    'Player 2 Odd', 
                    'Result', 
                    'tourney_id', 
                    'tourney_date'
                    ]
    cat_feature_info = {key:{'dimension':1, 'categories':len(list(set(df[key])))} for key in cat_features}
    num_feature_info = {key:{'dimension':1, 'categories':None} for key in num_features}

    features = num_features + cat_features

    # df['player_name'] = hash_features(df['player_name'].values)
    # df['opponent_name'] = hash_features(df['opponent_name'].values)

    test_start = f'{year}-{month:02d}-01'
    if year == 2021 :
        val_start = pd.to_datetime(test_start) - pd.DateOffset(months=24) - pd.DateOffset(days=1)
        val_end = pd.to_datetime(test_start) - pd.DateOffset(months=12) - pd.DateOffset(days=1)
        test_end = pd.to_datetime(test_start) + pd.DateOffset(months=3) - pd.DateOffset(days=1)

        # Split train, validation, and test
        train = df[df['tourney_date'] < val_start]  # Train and validate on data before test period
        val = df[(df['tourney_date'] >= val_start) & (df['tourney_date'] < val_end)  ]
        test = df[(df['tourney_date'] >= test_start) & (df['tourney_date'] <= test_end)]  # Test on 3-month window
    else:
        val_start = pd.to_datetime(test_start) - pd.DateOffset(months=12) - pd.DateOffset(days=1)
        test_end = pd.to_datetime(test_start) + pd.DateOffset(months=3) - pd.DateOffset(days=1)

        # Split train, validation, and test
        train = df[df['tourney_date'] < val_start]  # Train and validate on data before test period
        val = df[(df['tourney_date'] >= val_start) & (df['tourney_date'] < test_start) ]
        test = df[(df['tourney_date'] >= test_start) & (df['tourney_date'] <= test_end)]  # Test on 3-month window

    test_columns = test[test_columns]
    print(f"Validation start date: {val_start}")
    print(f"Test start date: {test_start}")
    print(f"Test end date: {test_end}")

    # x_train= train[features]
    x_train_num = train[num_features]
    x_train_cat = train[cat_features]
    y_train = torch.tensor(train['Result'].values).long().unsqueeze(-1)

    x_val_num = val[num_features]
    x_val_cat = val[cat_features]
    y_val = torch.tensor(val['Result'].values).long().unsqueeze(-1)

    x_test_num = test[num_features]
    x_test_cat = test[cat_features]
    y_test = torch.tensor(test['Result'].values).long().unsqueeze(-1)

    # Convert numerical features to tensors
    x_train_num = [torch.tensor(x_train_num[f].values, dtype=torch.float32).unsqueeze(-1) for f in num_features]
    x_train_cat = [torch.tensor(x_train_cat[f].values).long() for f in cat_features]

    x_val_num = [torch.tensor(x_val_num[f].values, dtype=torch.float32).unsqueeze(-1) for f in num_features]
    x_val_cat = [torch.tensor(x_val_cat[f].values).long() for f in cat_features]

    x_test_num = [torch.tensor(x_test_num[f].values, dtype=torch.float32).unsqueeze(-1) for f in num_features]
    x_test_cat = [torch.tensor(x_test_cat[f].values).long() for f in cat_features]

    return x_train_num, x_train_cat, x_val_num, x_val_cat, x_test_num, x_test_cat, y_train, y_val, y_test, num_feature_info, cat_feature_info, test_columns


def hash_features(feature, num_bins=1, num_hashes=1):
    """
    Feature hash with multiple hash functions to reduce collisions.
    Uses the double hashing technique.

    Args:
        feature: Input tensor of categorical indices
        num_bins: Size of embedding vocabulary (preferably power of 2)
        num_hashes: Number of hash functions to use

    Returns:
        Tensor of integer indices for embedding lookup
    """
    # Use two different hash functions
    h1 = feature * 2654435761 % num_bins  # Knuth's multiplicative method
    h2 = feature * 2246822519 % num_bins  # Another large prime multiplier

    # Combine hashes using double hashing technique
    combined = h1.clone()
    for i in range(1, num_hashes):
        combined = (combined + h2) % num_bins

    return combined.long()




if __name__ == '__main__':
    split_df()