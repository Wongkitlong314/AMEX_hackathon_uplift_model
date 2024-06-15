import pandas as pd

### Scoring function for participating teams :
def incr_act_top10(input_df: pd.DataFrame,
                   pred_col: str,
                   cm_key='customer',
                   treated_col='ind_recommended',
                   actual_col='activation'):
    '''
    Function that returns the incremental activation score for the AMEX Singapore Hackathon 2024

    input_df : pandas Dataframe which has customer, ind_recommended, activation and pred_col
    pred_col : name of your prediction score variable
    cm_key : customer unique ID (do not change)
    treated_col : indicator variable whether a merchant was recommended
    actual_col : whether a CM had transacted at a given merchant (target variable)

    Returns - incremental activation
    '''
    print("original input_df:\n", input_df)
	#for correcting variable types
    input_df[[treated_col, actual_col, pred_col]] = input_df[[treated_col, actual_col, pred_col]].apply(pd.to_numeric, errors='coerce')
    print("After numerical:\n", input_df)
	
    input_df['rank_per_cm1'] = input_df.groupby(cm_key)[pred_col].rank(method='first', ascending=False)
    print("After Rank:\n", input_df)
    
    input_df = input_df.loc[input_df.rank_per_cm1 <= 10,:]
    print("After Filter:\n", input_df)
    
    agg_df = input_df.groupby(treated_col,as_index=False).agg({actual_col:'mean'})
    print("After Agg:\n", agg_df)
    agg_df.columns = [treated_col,'avg_30d_act']
    print("After Agg:\n", agg_df)
    
    # print(agg_df)
    recommended_avg_30d_act = float(agg_df.loc[agg_df[treated_col]==1,'avg_30d_act'])
    not_recommended_avg_30d_act = float(agg_df.loc[agg_df[treated_col]==0,'avg_30d_act'])
    
	
    return (recommended_avg_30d_act-not_recommended_avg_30d_act)
	
# example usage
# incr_act_top_10(input_df = my_test_pd_df, pred_col = 'prediction_score')
