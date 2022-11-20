# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 21:07:32 2022

@author: WilliamKiger
"""
import pandas as pd
import Sentiment as s 

def rankBySentiment(df): 
    
    ###########################################################################
    #We might want to think about loading the model and processing this sentiment
    #analysis at the beginning of the program.  We now have data.csv that is a
    #static database (.csv) saved off from our scraper for the demonstration 
    #purposes.  This will speed things up on the query->results.  
    #
    # ******I am adding a new column to the dataframe here named score******
    # 
    ###########################################################################

    path_to_model = "model\sentiment_model.pt"
    model, device, tokenizer, init_token_id, eos_token_id = s.loadSentimentModel(path_to_model)
    

    df['score'] = ''
               
    for row in df.itertuples():
        
        if  pd.isna(df.at[row.Index, 'contents']):
            inference = s.predictSentiment(model, device, tokenizer, init_token_id, eos_token_id, str(df.at[row.Index, 'title']))
            df.at[row.Index, "score"] = inference
        else: 
            inference = s.predictSentiment(model, device, tokenizer, init_token_id, eos_token_id, str(df.at[row.Index, 'contents']))
            df.at[row.Index, "score"] = inference
            
            
    #sort, highest value first        
    sorted_sentiment_df = df.sort_values(by=['score'], ascending=False)
        
    return sorted_sentiment_df


def main():  
    df = pd.read_csv ('database\data.csv')
    df = rankBySentiment(df)
    print(df)
    
if __name__ == '__main__':
    main()
