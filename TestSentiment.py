import Sentiment as s 

#this only needs to be done once
path_to_model = "model\sentiment_model.pt"
model, device, tokenizer, init_token_id, eos_token_id = s.loadSentimentModel(path_to_model)


#Each call will look like this: 
text = "I love NLP"
inference = s.predictSentiment(model, device, tokenizer, init_token_id, eos_token_id, text)
print(inference)

text = "I hate NLP"
inference = s.predictSentiment(model, device, tokenizer, init_token_id, eos_token_id, text)
print(inference)