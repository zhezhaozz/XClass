import argparse
import os
import re
import torch
import pandas as pd
import numpy as np

from collections import defaultdict
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.feature_extraction import text 


if torch.cuda.is_available():
    device = torch.device(0)
else:
	device = "cpu"

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='Pain-study')
parser.add_argument('--model', default='bbu')
args = parser.parse_args()

if args.model == 'bbu':
	bert_model = 'bert-base-uncased'
elif args.model == 'bcb':
	bert_model = 'emilyalsentzer/Bio_ClinicalBERT'
elif args.model == 'pbbf':
	bert_model = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
elif args.model == 'pbba':
	bert_model = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'
else:
	bert_model = 'biobert-v1.1/'

data_path = f'../data/datasets/{args.dataset}'
corpus_file = f'{data_path}/dataset.txt'
seeds_file = f'{data_path}/seeds.txt'
bert_file = f'../data/embeddings/embedding_{args.model}.txt'

# initiate BERT
tokenizer = BertTokenizer.from_pretrained(bert_model)
model = BertModel.from_pretrained(bert_model, output_hidden_states=True).to(device)
model.eval()

# load data
cnt = defaultdict(int)
with open(corpus_file) as fin:
	for line in fin:
		abs = re.split(r'[^a-zA-Z_]+', line)
		for word in abs:
		    cnt[word] += 1

min_count = 3
vocabulary = set()
for word in cnt:
	if cnt[word] >= min_count and word.replace('_', ' ').strip() != '':
		vocabulary.add(word)

with open(seeds_file) as fin, open(f'{data_path}/oov_{args.model}.txt', 'w') as fout:
	for line in fin:
		seeds = line.strip().split(',')
		for seed in seeds:
			if seed not in vocabulary:
				fout.write(seed+'\n')
				vocabulary.add(seed)

# create embeddings for words
print(f"create embeddings in {args.model} space...")
with torch.no_grad():
	with open(bert_file, 'w') as f:
		f.write(f'{len(vocabulary)} 768\n')
		for word in tqdm(vocabulary):
			text = word.replace('_', ' ')
			input_ids = torch.tensor(tokenizer.encode(text, max_length=768, truncation=True)).unsqueeze(0).to(device)
			outputs = model(input_ids)
			hidden_states = outputs[2][-1][0]
			emb = torch.mean(hidden_states, dim=0).cpu()

			f.write(f'{word} '+' '.join([str(x.item()) for x in emb])+'\n')