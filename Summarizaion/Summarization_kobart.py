import json # import json module
import kobart
import transformers
from transformers import BartModel, AdamW
import pandas as pd
from kobart_transformers import get_kobart_tokenizer, get_kobart_model, get_kobart_for_conditional_generation
from torch.utils.data import Dataset, DataLoader
from pyrouge import Rouge

import torch
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import torch.nn.functional as F

text=[]
ans = []
with open('NIKL_SC.json', "r") as json_file:
    summary = json.load(json_file)
    summary = summary["data"]
    for d in summary:
        doc_id = str(d["document_id"]).strip().split(".")
        ans_text1 = str(d["summary_sentences"][0])
        ans_text2 = str(d["summary_sentences"][1])
        ans_text3 = str(d["summary_sentences"][2])
        # print(d["document_id"])
        for root, dirs, files in os.walk('./news/data'):
            for f in files:
                if(str(f) == doc_id[0]+".json"):
                    with open('./news/data/'+f) as json_file:
                        origin_data = json.load(json_file)
                        origin_data = origin_data["document"]
                        # print(origin_data[0])
                        for w in origin_data:
                            if (w["id"] == d["document_id"]):
                                # print(w["id"], d["document_id"])
                                q = w["paragraph"]
                                preprocess_text = ""
                                for e in q:
                                    # print(e["form"])
                                    preprocess_text = preprocess_text +" " +str(e["form"])
                                text.append((preprocess_text[:500], ans_text1))
                                text.append((preprocess_text[:500], ans_text2))
                                text.append((preprocess_text[:500], ans_text3))
                                break
            break

text = pd.DataFrame(text)

print(len(text))
# print(text.head())
# print(text.loc[:3, [0]])
# print(text.loc[:3, [1]])

train_data=text.sample(frac=0.99,random_state=200) #random state is a seed value
test_data=text.drop(train_data.index)



class newsDataset(Dataset):
    ''' Naver Sentiment Movie Corpus Dataset '''
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        topic = self.df.iloc[idx, 0]
        ans = self.df.iloc[idx, 1]
        return topic, ans

k = newsDataset(train_data)

# print(k[0])
train_loader = DataLoader(k, batch_size=1, shuffle=True, num_workers=2)
print(len(train_loader))
#
#
#
kobart_tokenizer = get_kobart_tokenizer()
kobart_model = get_kobart_model()
model = get_kobart_for_conditional_generation()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

itr = 1
p_itr = 500
epochs = 10
total_loss = 0
total_len = 0
total_correct = 0
model.to(device)
model.train()

for epoch in range(epochs):

    for text, ans in train_loader:
        optimizer.zero_grad()

        # encoding and zero padding
        encoded_list = [kobart_tokenizer.encode(t, add_special_tokens=True) for t in text]
        padded_list = [e + [0] * (512 - len(e)) for e in encoded_list]

        decoded_list = [kobart_tokenizer.encode(t, add_special_tokens=True) for t in ans]
        summ = [e + [0] * (512 - len(e)) for e in decoded_list]

        sample = torch.tensor(padded_list)
        qw = torch.tensor(summ)
        sample, qw = sample.to(device), qw.to(device)
        # print(sample, qw)
        outputs = model(sample, labels=qw)
        # print(outputs)
        total_loss += outputs.loss
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        if itr % p_itr == 0:
            print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}'.format(epoch + 1, epochs, itr,
                                                                                              total_loss / p_itr
                                                                                              ))
            total_loss = 0
            total_len = 0
            total_correct = 0

        itr += 1

torch.save(model.state_dict(), 'Summarization_model.pt')
model.load_state_dict(torch.load('Summarization_model.pt'))
model.eval() #test mode로 변경, dropout같은 함수가 적용 될지 안될지 결정
# iterate over test data
nsmc_eval_dataset = newsDataset(test_data)
test_loader = DataLoader(nsmc_eval_dataset, batch_size=1, shuffle=False, num_workers=2)
total_loss = 0
total_len = 0
total_r = 0.0
r = Rouge()
for text, ans in test_loader:
    enc_list = [kobart_tokenizer.encode(t, add_special_tokens=True) for t in text]
    pad_list =  [e + [0] * (512-len(e)) for e in enc_list]
    sample = torch.tensor(pad_list)

    sample = sample.to(device)

    summary_ids = model.generate(sample,
                                 num_beams=4,
                                 no_repeat_ngram_size=4,
                                 min_length=20,
                                 max_length=40).to(device)

    output = kobart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print("원래기사 : " , str(text))
    print("\n요약기사 : ", str(ans))
    print("\nSummarized text: \n", output)
    [precision, recall, f_score] = r.rouge_l([str(output)], [str(ans)])
    total_r+=f_score
    print("\nPrecision is :" + str(precision) + "\nRecall is :" + str(recall) + "\nF Score is :" + str(f_score))
    print("\n\n\n")

print(total_r/len(test_data))

#




























