import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


def split_df(val_start='2024-01-01',train_start='2000-01-01'):

    df = pd.read_csv('tennis_preprocessed.csv')
    df['tourney_date'] = pd.to_datetime(df['tourney_date'])
    # df = df[df['tourney_date'] > '2005-01-01']
    df = df.sort_values('tourney_date')

    num_features = ['Player Momentum Difference',
                    'Player Deviation Difference',
                    'Player Glicko Difference',
                    'Player Rank Points Difference',
                    'Player Rank Difference',
                    'Player Age Difference',
                    'Player Height Difference',
                    'Average Glicko Last 6 Months Difference',
                    'Player bpSaved% Avg Difference',
                    'Player ServicePointsWon% Avg Difference',
                    'Player ReturnPointsWon% Avg Difference',
                    'Player Completeness Avg Difference',
                    'Player Serve Advantage Avg Difference',
                    'Same Surface Matches Won Difference',
                    'Surface Win Percentage Difference',
                    'Wins Difference',
                    'Win Rate Difference',
                    'Titles Difference',
                    'Grand Slam Win Rate Difference',
                    'Masters Win Rate Difference',
                    'Qualifiers Win Rate Difference',
                    'Total Matches Difference',
                    'H2H Games Won',
                    'H2H Games Played',
                    'Common Opponent Advantage',
                    'Num Common Opponents']

    cat_features = ['Surface Clay',
                    'Surface Grass',
                    'Surface Hard',
                    'isMasters',
                    'isQualifiers',
                    'isGrandSlam']

    cat_feature_info = {key:{'dimension':1, 'categories':len(list(set(df[key])))} for key in cat_features}
    num_feature_info = {key:{'dimension':1, 'categories':None} for key in cat_features}
    print(cat_feature_info)
    features = num_features + cat_features

    # df['player_name'] = hash_features(df['player_name'].values)
    # df['opponent_name'] = hash_features(df['opponent_name'].values)

    train = df[(df['tourney_date'] >= train_start) & (df['tourney_date'] < val_start)]  # Train and validate on data before test period
    val = df[(df['tourney_date'] >= val_start)]
    # test = df[(df['tourney_date'] >= test_start) & (df['tourney_date'] <= test_end)]  # Test on 3-month window
    # test = test.dropna(subset=['Player 1 Odd'])
    # test = test[(test['isMasters']==1) | (test['isGrandSlam']==1)]

    scaler = StandardScaler()
    # x_train= train[features]
    x_train_num = train[num_features]
    x_train_cat = train[cat_features]
    y_train = torch.tensor(train['Result'].values).long().unsqueeze(-1)

    x_val_num = val[num_features]
    x_val_cat = val[cat_features]
    y_val = torch.tensor(val['Result'].values).long().unsqueeze(-1)

    # Convert numerical features to tensors
    x_train_num = [torch.tensor(x_train_num[f].values, dtype=torch.float32).unsqueeze(-1) for f in num_features]
    x_train_cat = [torch.tensor(x_train_cat[f].values).long() for f in cat_features]

    x_val_num = [torch.tensor(x_val_num[f].values, dtype=torch.float32).unsqueeze(-1) for f in num_features]
    x_val_cat = [torch.tensor(x_val_cat[f].values).long() for f in cat_features]

    return x_train_num, x_train_cat, x_val_num, x_val_cat, y_train, y_val, num_feature_info, cat_feature_info


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