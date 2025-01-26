import pandas as pd
from math import prod


def calculate_multipliers(best_test_pred, maxw_maxl_result):
    probability = pd.DataFrame({'Probability': best_test_pred}).reset_index(drop=True)
    test_results = maxw_maxl_result.reset_index(drop=True)
    test_results = pd.concat([test_results, probability], axis=1).dropna()  # Concatenate Probability as a new column
    print(len(test_results))
    test_results['Prediction'] = test_results['Probability'].apply(lambda x: 1 if x > 0.5 else 0)
    test_results['Month'] =  test_results['tourney_date'].dt.month
    test_results['Return'] = 1
    for index, row in test_results.iterrows():

        implied_probability_player1 = 1 / row['Player 1 Odd']
        implied_probability_player2 = 1 / row['Player 2 Odd']

        p = row['Probability']  # Model's predicted probability
        q = 1 - p  # Probability of the other outcome

        if p - implied_probability_player1 > 0.01 and p > 0.3:
            if row['Result'] == 1:  # Model correctly predicted Player 1 would win
                test_results.at[index, 'Return'] = row['Player 1 Odd']  # You win and get back odds
            else:
                test_results.at[index, 'Return'] = 0  # You lose and get back 0

        elif q - implied_probability_player2 > 0.01 and q > 0.3:
            if row['Result'] == 0:  # Model correctly predicted Player 2 would win
                test_results.at[index, 'Return'] = row['Player 2 Odd']  # You win and get back odds
            else:
                test_results.at[index, 'Return'] = 0  # You lose and get back 0
        else:
            test_results.at[index, 'Return'] = 1

    filtered_results = test_results.dropna().groupby('tourney_date').filter(lambda x: len(x) > 10)
    multipliers = filtered_results[filtered_results['Return'] != 1].groupby('tourney_date')['Return'].mean()  # Exclude rows where 'Return' is 1
    print('Profit: ', prod(multipliers))
