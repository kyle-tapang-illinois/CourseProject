import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import pandas as pd

class BertSentimentModel(nn.Module):
    def __init__(self,bert,hidden_dim,output_dim,n_layers,bidirectional,dropout):
        super(BertSentimentModel, self).__init__()
    
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.rnn = nn.GRU(
          embedding_dim,
          hidden_dim,
          num_layers=n_layers,
          bidirectional=bidirectional,
          batch_first=True,
          dropout=0 if n_layers < 2 else dropout
        )
        self.out = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
      
    def forward(self, text):
        with torch.no_grad():
            embedded = self.bert(text)[0]
            
        _, hidden = self.rnn(embedded)
    
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
    
        output = self.out(hidden)
        return output


def loadSentimentModel(path_to_model):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    init_token_id = tokenizer.cls_token_id
    eos_token_id  = tokenizer.sep_token_id

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    
    model = BertSentimentModel(bert_model, 256, 1, 2, True, 0.25)
    #bert,hidden_dim,output_dim,n_layers,bidirectional,dropout
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(path_to_model))
    else:
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()
    model = model.to(device)
    
    return model, device, tokenizer, init_token_id, eos_token_id
    
def predictSentiment(model, device, tokenizer, init_token_id, eos_token_id, sentence):
    model.eval()
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:500]
    indexed = [init_token_id] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_id]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    
    return prediction.item()

def getSentiment(df):
    ###########################################################################
    # We might want to think about loading the model and processing this sentiment
    # analysis at the beginning of the program.  We now have data.csv that is a
    # static database (.csv) saved off from our scraper for the demonstration
    # purposes.  This will speed things up on the query->results.
    #
    # ******I am adding a new column to the dataframe here named score******
    #
    ###########################################################################

    path_to_model = "model/sentiment_model.pt"
    model, device, tokenizer, init_token_id, eos_token_id = loadSentimentModel(path_to_model)

    df['score'] = ''
    df['is_content'] = ''

    for row in df.itertuples():

        if pd.isna(df.at[row.Index, 'contents']):
            inference = predictSentiment(model, device, tokenizer, init_token_id, eos_token_id,
                                           str(df.at[row.Index, 'title']))
            df.at[row.Index, "score"] = inference
            df.at[row.Index, "is_content"] = False
        else:
            inference = predictSentiment(model, device, tokenizer, init_token_id, eos_token_id,
                                           str(df.at[row.Index, 'contents']))
            df.at[row.Index, "score"] = inference
            df.at[row.Index, "is_content"] = True

    sentiment_df = df.sort_values(by=['id'], ascending=True)

    return sentiment_df

def rankBySentiment(df):
    df = getSentiment(df)
    return df.sort_values(by=['score'], ascending=False)

def main():
    df = pd.read_csv('database/data.csv')
    df = getSentiment(df)

    header = ["id", "score", "is_content"]
    df.to_csv('database/data_sentiment_scores.csv', columns=header, index=False)
    print(df)

if __name__ == '__main__':
    main()
