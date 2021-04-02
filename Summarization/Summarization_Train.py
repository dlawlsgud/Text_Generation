from kobart_transformers import get_kobart_tokenizer, get_kobart_model, get_kobart_for_conditional_generation
from torch.utils.data import Dataset, DataLoader
import torch
import pickle
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
text=[]
ans = []


#DATA 전처리


torch.manual_seed(200)
with open("data_국립200_1.pickle", "rb") as fr:
    data = pickle.load(fr)
train_data = data[0]
valid_data = data[1]
test_data = data[2]

train_data = train_data[:3509]
valid_data = valid_data[:439]
test_data = test_data[:439]
print(len(train_data), len(valid_data),len(test_data))

print(len(train_data), len(valid_data),len(test_data))
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
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
valid_loader = DataLoader(valid_dataset, batch_size=5, shuffle=True, num_workers=2,  drop_last=True)


#Load Model

kobart_tokenizer = get_kobart_tokenizer()
model = get_kobart_for_conditional_generation()
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
model.load_state_dict(torch.load("Summarization_model_textrank3_summa3.pt"))
# # model.to(device)
itr = 1
p_itr = 500
epochs = 7
best_valid_loss=987654321
total_loss = 0

model.to(device)
model.train()

#TRAIN

for epoch in range(epochs):
    for text, ans in train_loader:
        optimizer.zero_grad()
        # encoding and zero padding

        ori_doc = [kobart_tokenizer.encode_plus(t, add_special_tokens=True, max_length = 1024, pad_to_max_length = True)["input_ids"] for t in text]
        # ori_doc = [e + [0] * (512 - len(e)) for e in encoded_list]
        ref_sum = [kobart_tokenizer.encode_plus(t, add_special_tokens=True, max_length = 512, pad_to_max_length = True)["input_ids"] for t in ans]
        # ref_sum = [e + [0] * (512 - len(e)) for e in decoded_list]
        # print(ref_sum[0])
        dec_in=[3]
        for i in ref_sum[0]:
            dec_in.append(i)
        dec_in = dec_in[:-1]
        # print(len(ref_sum[0])
        for i in range(len(ref_sum[0])):
            if ref_sum[0][i] == 3 :
                ref_sum[0][i] = 1
                break
            if i == len(ref_sum[0])-1 : ref_sum[0][len(ref_sum[0])-1] = 1
        # print(ref_sum)
        ori_doc = torch.tensor(ori_doc)
        dec_in = torch.tensor([dec_in])
        ref_sum = torch.tensor(ref_sum)
        # print(dec_in.shape,ref_sum.shape)
        ori_doc, dec_in, ref_sum = ori_doc.to(device), dec_in.to(device),ref_sum.to(device)
        outputs = model(ori_doc, decoder_input_ids=dec_in, labels=ref_sum)

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
                valid_doc = [kobart_tokenizer.encode_plus(t, add_special_tokens=True, max_length = 1024, pad_to_max_length = True)["input_ids"] for t in val_text1]
                valid_summary = [kobart_tokenizer.encode_plus(t, add_special_tokens=True, max_length = 512, pad_to_max_length = True)["input_ids"] for t in val_ans1]
                for i in range(len(valid_summary[0])):
                    if valid_summary[0][i] == 3:
                        valid_summary[0][i] = 1
                        break
                    if i == len(valid_summary[0]) - 1: valid_summary[0][len(valid_summary[0]) - 1] = 1
                valid_doc= torch.tensor(valid_doc)
                valid_summary = torch.tensor(valid_summary)
                valid_doc, valid_summary = valid_doc.to(device), valid_summary.to(device)
                outputs1 = model(valid_doc,  labels=valid_summary)

                if outputs1.loss < best_valid_loss:
                    best_valid_loss = outputs1.loss
                    print(best_valid_loss)
                    torch.save(model.state_dict(), "Summarization_model_textrank3_summa3.pt")




