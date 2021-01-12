import spacy
spacy_en = spacy.load('en_core_web_sm') # 영어 토큰화(tokenization)
spacy_de = spacy.load('de_core_news_sm') # 독일어 토큰화(tokenization)


# 간단히 토큰화(tokenization) 기능 써보기
tokenized = spacy_en.tokenizer("I am a graduate student.")

for i, token in enumerate(tokenized):
    print(f"인덱스 {i}: {token.text}")

# 독일어(Deutsch) 문장을 토큰화한 뒤에 순서를 뒤집는 함수
def tokenize_de(text):
    return [token.text for token in spacy_de.tokenizer(text)][::-1]

# 영어(English) 문장을 토큰화 하는 함수
def tokenize_en(text):
    return [token.text for token in spacy_en.tokenizer(text)]

from torchtext.data import Field, BucketIterator

SRC = Field(tokenize=tokenize_de, init_token="<sos>", eos_token="<eos>", lower=True)
TRG = Field(tokenize=tokenize_en, init_token="<sos>", eos_token="<eos>", lower=True)
from torchtext.datasets import Multi30k

train_dataset, valid_dataset, test_dataset = Multi30k.splits(exts=(".de", ".en"), fields=(SRC, TRG))
print(f"학습 데이터셋(training dataset) 크기: {len(train_dataset.examples)}개")
print(f"평가 데이터셋(validation dataset) 크기: {len(valid_dataset.examples)}개")
print(f"테스트 데이터셋(testing dataset) 크기: {len(test_dataset.examples)}개")

SRC.build_vocab(train_dataset, min_freq=2)
TRG.build_vocab(train_dataset, min_freq=2)

print(f"len(SRC): {len(SRC.vocab)}")
print(f"len(TRG): {len(TRG.vocab)}")

print(TRG.vocab.stoi["abcabc"]) # 없는 단어: 0
print(TRG.vocab.stoi[TRG.pad_token]) # 패딩(padding): 1
print(TRG.vocab.stoi["<sos>"]) # <sos>: 2
print(TRG.vocab.stoi["<eos>"]) # <eos>: 3
print(TRG.vocab.stoi["hello"])
print(TRG.vocab.stoi["world"])

import torch


BATCH_SIZE = 128

# 일반적인 데이터 로더(data loader)의 iterator와 유사하게 사용 가능
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_dataset, valid_dataset, test_dataset),
    batch_size=BATCH_SIZE)

import torch.nn as nn


# 인코더(Encoder) 아키텍처 정의
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, enc_hidden_dim, dec_hidden_dim, dropout_ratio):
        super().__init__()

        # 임베딩(embedding)은 원-핫 인코딩(one-hot encoding)을 특정 차원의 임베딩으로 매핑하는 레이어
        self.embedding = nn.Embedding(input_dim, embed_dim)

        # 양방향(bidirectional) GRU 레이어
        self.rnn = nn.GRU(embed_dim, enc_hidden_dim, bidirectional=True)

        # FC 레이어
        self.fc = nn.Linear(enc_hidden_dim * 2, dec_hidden_dim)

        # 드롭아웃(dropout)
        self.dropout = nn.Dropout(dropout_ratio)

    # 인코더는 소스 문장을 입력으로 받아 문맥 벡터(context vector)를 반환
    def forward(self, src):
        # src: [단어 개수, 배치 크기]: 각 단어의 인덱스(index) 정보
        embedded = self.dropout(self.embedding(src))
        # embedded: [단어 개수, 배치 크기, 임베딩 차원]

        outputs, hidden = self.rnn(embedded)
        # outputs: [단어 개수, 배치 크기, 인코더 히든 차원 * 방향의 수]: 전체 단어의 출력 정보
        # hidden: [레이어 개수 * 방향의 수, 배치 크기, 인코더 히든 차원]: 현재까지의 모든 단어의 정보

        # hidden은 [forward_1, backward_1, forward_2, backward_2, ...] 형태로 구성
        # 따라서 hidden[-2, :, :]은 forwards의 마지막 값
        # 따라서 hidden[-1, :, :]은 backwards의 마지막 값
        # 디코더(decoder)의 첫 번째 hidden (context) vector는 인코더의 마지막 hidden을 이용
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))

        # outputs은 Attention 목적으로, hidden은 context vector 목적으로 사용
        return outputs, hidden

import torch.nn.functional as F

# 어텐션(Attention) 아키텍처 정의
class Attention(nn.Module):
    def __init__(self, enc_hidden_dim, dec_hidden_dim):
        super().__init__()

        self.attn = nn.Linear(1280, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden, enc_outputs):
        # hidden: [배치 크기, 히든 차원]: 현재까지의 모든 단어의 정보
        # enc_outputs: [단어 개수, 배치 크기, 인코더 히든 차원 * 방향의 수]: 전체 단어의 출력 정보
        batch_size = enc_outputs.shape[1]
        src_len = enc_outputs.shape[0]

        # 현재 디코더의 히든 상태(hidden state)를 src_len만큼 반복
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        enc_outputs = enc_outputs.permute(1, 0, 2)
        # hidden: [배치 크기, 단어 개수, 디코더 히든 차원]: 현재까지의 모든 단어의 정보
        # enc_outputs: [배치 크기, 단어 개수, 인코더 히든 차원 * 방향의 수]: 전체 단어의 출력 정보
        x = torch.cat((hidden, enc_outputs), dim=2)
        #print(x.size())
        energy = torch.tanh(self.attn(x))
        # energy: [배치 크기, 단어 개수, 디코더 히든 차원]

        attention = self.v(energy).squeeze(2)
        # attention: [배치 크기, 단어 개수]: 실제 각 단어에 대한 어텐선(attention) 값들

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, enc_hidden_dim, dec_hidden_dim, dropout_ratio, attention):
        super().__init__()

        self.output_dim = output_dim
        self.attention = attention

        # 임베딩(embedding)은 원-핫 인코딩(one-hot encoding) 말고 특정 차원의 임베딩으로 매핑하는 레이어
        self.embedding = nn.Embedding(output_dim, embed_dim)

        # GRU 레이어
        self.rnn = nn.GRU((enc_hidden_dim * 2) + embed_dim, dec_hidden_dim)

        # FC 레이어
        self.fc_out = nn.Linear((enc_hidden_dim * 2) + dec_hidden_dim + embed_dim, output_dim)

        # 드롭아웃(dropout)
        self.dropout = nn.Dropout(dropout_ratio)

    # 디코더는 현재까지 출력된 문장에 대한 정보를 입력으로 받아 타겟 문장을 반환
    def forward(self, input, hidden, enc_outputs):
        # input: [배치 크기]: 단어의 개수는 항상 1개이도록 구현
        # hidden: [배치 크기, 히든 차원]
        # enc_outputs: [단어 개수, 배치 크기, 인코더 히든 차원 * 방향의 수]: 전체 단어의 출력 정보
        input = input.unsqueeze(0)
        # input: [단어 개수 = 1, 배치 크기]

        embedded = self.dropout(self.embedding(input))
        # embedded: [단어 개수 = 1, 배치 크기, 임베딩 차원]

        attention = self.attention(hidden, enc_outputs)
        # attention: [배치 크기, 단어 개수]: 실제 각 단어에 대한 어텐선(attention) 값들
        attention = attention.unsqueeze(1)
        # attention: [배치 크기, 1, 단어 개수]: 실제 각 단어에 대한 어텐선(attention) 값들

        enc_outputs = enc_outputs.permute(1, 0, 2)
        # enc_outputs: [배치 크기, 단어 개수, 인코더 히든 차원 * 방향의 수]: 전체 단어의 출력 정보

        weighted = torch.bmm(attention, enc_outputs)
        # weighted: [배치 크기, 1, 인코더 히든 차원 * 방향의 수]

        weighted = weighted.permute(1, 0, 2)
        # weighted: [1, 배치 크기, 인코더 히든 차원 * 방향의 수]

        rnn_input = torch.cat((embedded, weighted), dim=2)
        # rnn_input: [1, 배치 크기, 인코더 히든 차원 * 방향의 수 + embed_dim]: 어텐션이 적용된 현재 단어 입력 정보

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        # output: [단어 개수, 배치 크기, 디코더 히든 차원 * 방향의 수]
        # hidden: [레이어 개수 * 방향의 수, 배치 크기, 디코더 히든 차원]: 현재까지의 모든 단어의 정보

        # 현재 예제에서는 단어 개수, 레이어 개수, 방향의 수 모두 1의 값을 가짐
        # 따라서 output: [1, 배치 크기, 디코더 히든 차원], hidden: [1, 배치 크기, 디코더 히든 차원]
        # 다시 말해 output과 hidden의 값 또한 동일
        assert (output == hidden).all()

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.fc_out(torch.cat((output, weighted, embedded), dim=1))
        # prediction = [배치 크기, 출력 차원]

        # (현재 출력 단어, 현재까지의 모든 단어의 정보)
        return prediction, hidden.squeeze(0)
import random


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder


    # 학습할 때는 완전한 형태의 소스 문장, 타겟 문장, teacher_forcing_ratio를 넣기
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [단어 개수, 배치 크기]
        # trg: [단어 개수, 배치 크기]
        # 먼저 인코더를 거쳐 전체 출력과 문맥 벡터(context vector)를 추출
        enc_outputs, hidden = self.encoder(src)

        # 디코더(decoder)의 최종 결과를 담을 텐서 객체 만들기
        trg_len = trg.shape[0] # 단어 개수
        batch_size = trg.shape[1] # 배치 크기
        trg_vocab_size = self.decoder.output_dim # 출력 차원
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size)

        # 첫 번째 입력은 항상 <sos> 토큰
        input = trg[0, :]

        # 타겟 단어의 개수만큼 반복하여 디코더에 포워딩(forwarding)
        for t in range(1, trg_len):
            output, hidden = self.decoder(input, hidden, enc_outputs)
            outputs[t] = output # FC를 거쳐서 나온 현재의 출력 단어 정보
            top1 = output.argmax(1) # 가장 확률이 높은 단어의 인덱스 추출

            # teacher_forcing_ratio: 학습할 때 실제 목표 출력(ground-truth)을 사용하는 비율
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[t] if teacher_force else top1 # 현재의 출력 결과를 다음 입력에서 넣기

        return outputs
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
ENCODER_EMBED_DIM = 256
DECODER_EMBED_DIM = 256
HIDDEN_DIM = 512

ENC_DROPOUT_RATIO = 0.5
DEC_DROPOUT_RATIO = 0.5

# 어텐션(attention) 객체 선언
attn = Attention(ENCODER_EMBED_DIM, DECODER_EMBED_DIM)

enc = Encoder(INPUT_DIM, ENCODER_EMBED_DIM, HIDDEN_DIM, DECODER_EMBED_DIM,ENC_DROPOUT_RATIO)
dec = Decoder(OUTPUT_DIM, ENCODER_EMBED_DIM,HIDDEN_DIM,DECODER_EMBED_DIM,  DEC_DROPOUT_RATIO,attn)

# Seq2Seq 객체 선언
model = Seq2Seq(enc, dec)

import torch.optim as optim

# Adam optimizer로 학습 최적화
optimizer = optim.Adam(model.parameters())

# 뒷 부분의 패딩(padding)에 대해서는 값 무시
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):
    model.train()  # 학습 모드
    epoch_loss = 0

    # 전체 학습 데이터를 확인하며
    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output = model(src, trg)
        # output: [출력 단어 개수, 배치 크기, 출력 차원]
        output_dim = output.shape[-1]

        # 출력 단어의 인덱스 0은 사용하지 않음
        output = output[1:].view(-1, output_dim)
        # output = [(출력 단어의 개수 - 1) * batch size, output dim]
        trg = trg[1:].view(-1)
        # trg = [(타겟 단어의 개수 - 1) * batch size]

        # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
        loss = criterion(output, trg)
        loss.backward()  # 기울기(gradient) 계산

        # 기울기(gradient) clipping 진행
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        # 파라미터 업데이트
        optimizer.step()

        # 전체 손실 값 계산
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


# 모델 평가(evaluate) 함수
def evaluate(model, iterator, criterion):
    model.eval()  # 평가 모드
    epoch_loss = 0

    with torch.no_grad():
        # 전체 평가 데이터를 확인하며
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            # 평가할 때 teacher forcing는 사용하지 않음
            output = model(src, trg, 0)
            # output: [출력 단어 개수, 배치 크기, 출력 차원]
            output_dim = output.shape[-1]

            # 출력 단어의 인덱스 0은 사용하지 않음
            output = output[1:].view(-1, output_dim)
            # output = [(출력 단어의 개수 - 1) * batch size, output dim]
            trg = trg[1:].view(-1)
            # trg = [(타겟 단어의 개수 - 1) * batch size]

            # 모델의 출력 결과와 타겟 문장을 비교하여 손실 계산
            loss = criterion(output, trg)

            # 전체 손실 값 계산
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

#
# import time
# import math
# import random
#
# N_EPOCHS = 20
# CLIP = 1
# best_valid_loss = float('inf')
#
# for epoch in range(N_EPOCHS):
#     start_time = time.time() # 시작 시간 기록
#
#     train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
#     valid_loss = evaluate(model, valid_iterator, criterion)
#
#     end_time = time.time() # 종료 시간 기록
#     epoch_mins, epoch_secs = epoch_time(start_time, end_time)
#
#     if valid_loss < best_valid_loss:
#         best_valid_loss = valid_loss
#         torch.save(model.state_dict(), 'seq2seq2.pt')
#
#     print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
#     print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):.3f}')
#     print(f'\tValidation Loss: {valid_loss:.3f} | Validation PPL: {math.exp(valid_loss):.3f}')
#
#
# test_loss = evaluate(model, test_iterator, criterion)
#
# print(f'Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}')

def translate_sentence(sentence, src_field, trg_field, model, max_len=50):
    model.eval() # 평가 모드

    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

    # 처음에 <sos> 토큰, 마지막에 <eos> 토큰 붙이기
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]
    print(f"전체 소스 토큰: {tokens}")

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    print(f"소스 문장 인덱스: {src_indexes}")

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1)

    # 인코더(endocer)에 소스 문장을 넣어 문맥 벡터(context vector) 계산
    with torch.no_grad():
        enc_outputs, hidden = model.encoder(src_tensor)

    # 처음에는 <sos> 토큰 하나만 가지고 있도록 하기
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):
        # 이전에 출력한 단어가 현재 단어로 입력될 수 있도록
        trg_tensor = torch.LongTensor([trg_indexes[-1]])

        with torch.no_grad():
            output, hidden = model.decoder(trg_tensor, hidden, enc_outputs)

        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token) # 출력 문장에 더하기

        # <eos>를 만나는 순간 끝
        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    # 각 출력 단어 인덱스를 실제 단어로 변환
    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    # 첫 번째 <sos>는 제외하고 출력 문장 반환
    return trg_tokens[1:]

example_idx = 7
model.load_state_dict(torch.load('seq2seq2.pt'))
src = vars(test_dataset.examples[example_idx])['src']
trg = vars(test_dataset.examples[example_idx])['trg']

print(f'소스 문장: {src}')
print(f'타겟 문장: {trg}')
print("모델 출력 결과:", " ".join(translate_sentence(src, SRC, TRG, model)))
from nltk.translate.bleu_score import corpus_bleu

src = tokenize_de("Guten Abend.")

print(f'소스 문장: {src}')
print("모델 출력 결과:", " ".join(translate_sentence(src, SRC, TRG, model)))