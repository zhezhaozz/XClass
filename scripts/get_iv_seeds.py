import numpy as np
from collections import defaultdict
import argparse

parser = argparse.ArgumentParser(description='main', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', default='Pain-study')
parser.add_argument('--model', default='bbu')
parser.add_argument('--topm', default=5, type=int)
args = parser.parse_args()

dataset = args.dataset
model = args.model
topm = args.topm

data_path = f'../data/datasets/{args.dataset}'
corpus_file = f'{data_path}/dataset.txt'
seeds_file = f'{data_path}/seeds.txt'
bert_file = f'../data/embeddings/embedding_{args.model}.txt'
out_file = f'{data_path}/classes.txt'

topics = []
with open(seeds_file) as fin:
	for line in fin:
		data = line.strip().split(',')
		topics.append(data[:])

word2emb = {}
with open(bert_file) as fin:
	for line in fin:
		data = line.strip().split()
		if len(data) != 769:
			continue
		word = data[0]
		emb = np.array([float(x) for x in data[1:]])
		emb = emb / np.linalg.norm(emb)
		word2emb[word] = emb

oov = set()
with open(f'{data_path}/oov_{args.model}.txt') as fin:
	for line in fin:
		data = line.strip()
		oov.add(data)
		

with open(out_file, 'w') as fout:
	for idx, topic in enumerate(topics):
		word2score = defaultdict(float)
		new_topic = []
		cnt = 0
		for word in word2emb:
			if word in oov or word in topic:
				continue
			for term in topic:
				if term not in oov and term not in new_topic:
					new_topic.append(term) 
					cnt += 1
				word2score[word] += np.dot(word2emb[word], word2emb[term])
		extra_num = topm-cnt
		if extra_num > 0:
			score_sorted = sorted(word2score.items(), key=lambda x: x[1], reverse=True)[:extra_num]
			new_topic = new_topic + [x[0] for x in score_sorted]
		topics[idx] = new_topic
		fout.write(','.join(new_topic)+'\n')