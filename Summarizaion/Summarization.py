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
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch.nn.functional as F

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
early_stopping = EarlyStopping()

text=[]
ans = []
# with open('NIKL_SC.json', "r") as json_file:
#     summary = json.load(json_file)
#     summary = summary["data"]
#     for d in summary:
#         doc_id = str(d["document_id"]).strip().split(".")
#         ans_text1 = str(d["summary_sentences"][0])
#         ans_text2 = str(d["summary_sentences"][1])
#         ans_text3 = str(d["summary_sentences"][2])
#         to_ans = ans_text1 + ans_text2 + ans_text3
#         # print(d["document_id"])
#         for root, dirs, files in os.walk('./news/data'):
#             for f in files:
#                 if str(f) == doc_id[0]+".json":
#                     with open('./news/data/'+f) as json_file:
#                         origin_data = json.load(json_file)
#                         origin_data = origin_data["document"]
#                         # print(origin_data[0])
#                         for w in origin_data:
#                             if w["id"] == d["document_id"]:
#                                 # print(w["id"], d["document_id"])
#                                 q = w["paragraph"]
#                                 preprocess_text = ""
#                                 for e in q:
#                                     w = len(e["form"])
#                                     if len(preprocess_text)+ w <=1000: preprocess_text = preprocess_text +" " +str(e["form"])
#                                     else:break
#                                 text.append((preprocess_text, to_ans))
#
#                                 break
#             break
# # #
# import pickle
# # #
# # #
# ## Save pickle
# with open("data.pickle", "wb") as fw:
#     pickle.dump(text, fw)

## Load pickle
with open("data.pickle", "rb") as fr:
    data = pickle.load(fr)
# print(data)

text = pd.DataFrame(data)
import numpy as np
# train_data=text.sample(frac=0.9,random_state=200) #random state is a seed value
# test_data=text.drop(train_data.index)
# valid_data=train_data.sample(frac=0.11, random_state=200)
train_data, valid_data, test_data = np.split(text.sample(frac=1, random_state=200), [int(.8*len(text)), int(.9*len(text))])
print(len(train_data), len(valid_data),len(test_data))
# print(len(train_data), len(test_data))
class Dataset(Dataset):
    ''' Naver Sentiment Movie Corpus Dataset '''
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        topic = self.df.iloc[idx, 0]
        ans = self.df.iloc[idx, 1]
        return topic, ans

train_dataset = Dataset(train_data)
valid_dataset = Dataset(valid_data)
# print(k[0])
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=20, shuffle=True, num_workers=2,  drop_last=True)
# print(len(train_loader))
#
#
#
kobart_tokenizer = get_kobart_tokenizer()
model = get_kobart_for_conditional_generation()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)

itr = 1
p_itr = 500
epochs = 20
best_valid_loss=987654321
total_loss = 0

model.to(device)
model.train()


# print("1")
for epoch in range(epochs):
    for text, ans in train_loader:
        optimizer.zero_grad()
        # encoding and zero padding

        ori_doc = [kobart_tokenizer.encode_plus(t, add_special_tokens=True, max_length = 1000, pad_to_max_length = True)["input_ids"] for t in text]
        # ori_doc = [e + [0] * (512 - len(e)) for e in encoded_list]
        ref_sum = [kobart_tokenizer.encode_plus(t, add_special_tokens=True, max_length = 512, pad_to_max_length = True)["input_ids"] for t in ans]
        # ref_sum = [e + [0] * (512 - len(e)) for e in decoded_list]
        # print(type(ori_doc))
        ori_doc = torch.tensor(ori_doc)
        ref_sum = torch.tensor(ref_sum)
        ori_doc, ref_sum = ori_doc.to(device), ref_sum.to(device)
        outputs = model(ori_doc, labels=ref_sum)
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
    model.eval()
    with torch.no_grad():
            for val_text1, val_ans1 in valid_loader:
                valid_doc = [kobart_tokenizer.encode_plus(t, add_special_tokens=True, max_length = 1000, pad_to_max_length = True)["input_ids"] for t in val_text1]
                valid_summary = [kobart_tokenizer.encode_plus(t, add_special_tokens=True, max_length = 512, pad_to_max_length = True)["input_ids"] for t in val_ans1]

                valid_doc= torch.tensor(valid_doc)
                valid_summary = torch.tensor(valid_summary)
                valid_doc, valid_summary = valid_doc.to(device), valid_summary.to(device)
                outputs1 = model(valid_doc, labels=valid_summary)

                if outputs1.loss < best_valid_loss:
                    best_valid_loss = outputs1.loss
                    print(best_valid_loss)
                    torch.save(model.state_dict(), "Summarization_model3.pt")



# model.load_state_dict(torch.load("Summarization_model3.pt"))
# model.to(device)
# # iterate over test data
# test_dataset = Dataset(test_data)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
# model.eval()
# total_r=0
# r = Rouge()
# with torch.no_grad():
#     for text, ans in test_loader:
#         test_doc = [kobart_tokenizer.encode_plus(t, add_special_tokens=True, max_length = 1000, pad_to_max_length = True)["input_ids"] for t in text]
#
#         test_doc = torch.tensor(test_doc)
#         test_doc = test_doc.to(device)
#         summary_ids = model.generate(test_doc,
#                                      num_beams=4,
#                                      no_repeat_ngram_size=3,
#                                      min_length=30,
#                                      max_length=80).to(device)
#
#         output = kobart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
#         print("원래기사 : ", str(text))
#         print("\n요약기사 : ", str(ans))
#         print("\nSummarized text: \n", output)
#         [precision, recall, f_score] = r.rouge_l([str(output)], [str(ans)])
#         total_r += f_score
#         print("\nPrecision is :" + str(precision) + "\nRecall is :" + str(recall) + "\nF Score is :" + str(f_score))
#         print("\n\n\n")
#
#
#
# print(total_r/len(test_data))
#
#

