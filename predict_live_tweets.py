
import numpy as np
import torch
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-cased', return_dict=False)
MAX_LEN = 80  # All tweets in the data set contain fewer than 80 tokens
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict_tweets(model, df, risk_level=0.2):
    '''
    The method that handles prediction of tweet sentiment accepts as arguments the sentiment 
    classification model, the Pandas data frame of recent tweets, and the user’s desired risk 
    level for stock trading, which is a value between zero (most conservative) and one (most risky). 
    
    Next, it determines whether to trade a stock based on the certainty of the model’s prediction 
    about the sentiment of tweets it has just analyzed. If the user sets their desired risk level 
    closer to zero, then the BERT model is required to be more certain about its prediction before 
    making a trade, thus decreasing the volume of trades that occur. Since the BERT model computes 
    a likelihood score for both the bullish and bearish classifications for a tweet, certainty is 
    defined as the difference between the two scores. 

    The method also keeps a tally for a total polarity score for the incoming data frame, which 
    increments by one with each certain bullish prediction and decrements by one with each certain 
    bearish prediction. Bullish and bearish predictions that do not meet the minimum level of 
    certainty specified by the risk level do not affect the polarity score.

    The absolute value of the polarity score functions as a multiplier that contributes towards 
    the decision of how many shares to buy of the specified stock. Thus, a polarity score of zero 
    multiplies by zero, which results in not making any trades with that stock. Essentially, a 
    score of zero means that the accumulated sentimental predictions of certainty from all of 
    the tweets in the data frame was neutral, and this begins to happen more often as the user 
    decreases the input of the risk level of their trading strategy.
    '''
    trade_bool = False
    polarity = 0
    min_certainty = 7 - (risk_level*4)
    predictions = []
    for tweet in df.tweet:
        encoded_tweet = tokenizer.encode_plus(tweet, max_length=MAX_LEN, 
        add_special_tokens=True, return_token_type_ids=False, padding=True, 
        return_attention_mask=True, return_tensors='pt')
        input_ids = encoded_tweet['input_ids'].to(device)
        attention_mask = encoded_tweet['attention_mask'].to(device)
        output = model(input_ids, attention_mask)
        max_score, prediction = torch.max(output, dim=1)
        predictions.append(prediction.item())
        min_score, _ = torch.min(output, dim=1)
        certainty = (max_score - min_score).item()
        if certainty >= min_certainty:
            if prediction.item() == 0:
                polarity -= 1
            else:
                polarity += 1
    if polarity > 0 or polarity < 0:
        trade_bool = True

    df['prediction'] = np.array(predictions)

    return polarity, trade_bool, df
