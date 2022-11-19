import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn

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
    model.load_state_dict(torch.load(path_to_model))
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