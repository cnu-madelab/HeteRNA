import dgl
import torch

from helper import *
from data_loader import *

# sys.path.append('./')
from model.models import *

import logging
import os
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, precision_recall_curve, auc
from scipy.stats import uniform
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, random_split, DataLoader
import math

# ##Gridsearch
# from sklearn.model_selection import GridSearchCV
# from sklearn.base import BaseEstimator
#
# ##Random search
# from sklearn.model_selection import RandomizedSearchCV

import matplotlib.pyplot as plt
from collections import defaultdict
#os.chdir('./ICLR_CompGCN/')



class Runner(object):

	def __init__(self, params):
		"""
		Constructor of the runner class

		Parameters
		----------
		params:         List of hyper-parameters of the model

		Returns
		-------
		Creates computational graph and optimizer

		"""

		self.p			= params ## 여기서 파라미터 값들이 들어오게됨
		self.logger		= get_logger(self.p.name, self.p.log_dir, self.p.config_dir)
		self.best_threshold = 0.022
		self.mrr_results = []
		self.train_ratio = self.p.train_ratio
		self.test_ratio = self.p.test_ratio
		self.valid_ratio = self.p.valid_ratio

		#self.threshold_optimizer = torch.nn.parameter()

		##var는 입력으로들어오는 parameter를 dic로 변환해주는 역할임
		self.logger.info(vars(self.p))
		print(vars(self.p))

		if self.p.gpu != '-1' and torch.cuda.is_available():
			self.device = torch.device('cuda')
			# torch.cuda.set_rng_state(torch.cuda.get_rng_state())
			torch.backends.cudnn.deterministic = True
		# torch.backends.cudnn.benchmark = False  # 성능 최적화를 막아 결정적 실행 보장
		else:
			self.device = torch.device('cpu')

		self.load_data()
		## 여기서 입력한 score_func이 들어감
		# add model은 바로 아래에 있음
		# self.p.model : default compgcn / p.score_func conve
		self.model        = self.add_model(self.p.model, self.p.score_func)
		## 여기서 모델이 추가됨 즉 self.model에 add_model이추가됨 그리고 , score_fucn이 추가됨

		self.optimizer    = self.add_optimizer(self.model.parameters())
		#self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.991)

	def count_values(self, input_dict, circ_set, dise_set):
		count_dict = {}
		for key, value in input_dict.items():
			count_dict[key] = len(value)

		total = sum(count_dict.values())
		real_num = len(circ_set)*len(dise_set)
		print(total, real_num)
		return count_dict

	def split_dataset_f(self, total_dataset, total_size):

		total_size = total_size
		# 정수로 변환할 때 비율이 정확하게 유지되도록 함
		train_size = math.floor(total_size * self.train_ratio)  # 내림
		valid_size = math.floor(total_size * self.valid_ratio)  # 내림
		test_size = total_size - (train_size + valid_size)  # 남은 데이터는 테스트셋으로 배정

		# 데이터 랜덤 분할
		train_dataset, valid_dataset, test_dataset = random_split(
			total_dataset, [train_size, valid_size, test_size], generator=torch.Generator().manual_seed(self.p.seed)
		)
		return train_dataset, valid_dataset, test_dataset

	def load_data(self):
		"""
		Reading in raw triples and converts it into a standard format.

		Parameters
		----------
		self.p.dataset:         Takes in the name of the dataset (FB15k-237)

		Returns
		-------
		self.ent2id:            Entity to unique identifier mapping
		self.id2rel:            Inverse mapping of self.ent2id
		self.rel2id:            Relation to unique identifier mapping
		self.num_ent:           Number of entities in the Knowledge graph
		self.num_rel:           Number of relations in the Knowledge graph
		self.embed_dim:         Embedding dimension used
		self.data['train']:     Stores the triples corresponding to training dataset
		self.data['valid']:     Stores the triples corresponding to validation dataset
		self.data['test']:      Stores the triples corresponding to test dataset
		self.data_iter:		The dataloader for different data splits

		"""
		self.data = ddict(list)
		sr2o = ddict(set)
		ent_set, rel_set, obj_set, circ, dise, self.index_set = OrderedSet(), OrderedSet(), OrderedSet(), OrderedSet(), OrderedSet(), OrderedSet()

		for split in ['train', 'test', 'valid']:
			with open('./data/{}/{}/{}.txt'.format(self.p.dataset, self.p.data_name, split)) as file:
				for line in file:
					sub, rel, obj = map(str.lower, line.strip().split('\t'))
					ent_set.add(sub)
					rel_set.add(rel)
					ent_set.add(obj)

					self.index_set.add((sub,rel,obj))

					if rel == 'circ-disease':
						circ.add(sub)
						dise.add(obj)


		if self.p.extra_data == True: # 새로운 데이터 추가여부!
			self.p.num_rel = 6
			extra_set = {
				'ceRNA': 'lncrna_pairgene(long).txt',
				'lncRNA': 'lncrna_pairgene(short).txt',
				'miRNA': 'mirna_lncrna.txt'
			}
			for key, value in extra_set.items():
				with open('./data/ENCORI/{}/{}'.format(key, value), 'r') as file:
					for line in file:
						sub, rel, obj = map(str.lower, line.strip().split('\t'))
						ent_set.add(sub)
						rel_set.add(rel)
						ent_set.add(obj)

			## 새로추가한 데이터
			extra_set2 = { ##dise는 여기서 cancer data 추가한것임
				'circRNA': 'lnc2Cancer_CircRNA_dise.txt',
				'lncRNA': 'lnc2Cancer_lncRNA_dise.txt',
			}

			for key, value in extra_set2.items():
				with open('./data/lnc2Cancer_v3.0/{}/{}'.format(key, value), 'r') as file:
					for line in file:
						sub, rel, obj = map(str.lower, line.strip().split('\t'))
						ent_set.add(sub)
						rel_set.add(rel)
						ent_set.add(obj)

			with open('./data/LncRNADisease_v3.0/lncRNA_disease_3.0_lncRNA_disease.txt', 'r') as file:
				for line in file:
					sub, rel, obj = map(str.lower, line.strip().split('\t'))
					ent_set.add(sub)
					rel_set.add(rel)
					ent_set.add(obj)

		else:

			### xxx2id 로 끝나는 애들은 ent값(cir_has_12315)이 들어가면 id(숫자값) 을 출력

			## ent : key | value : idx
			self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)} ## 여기서 ent_set, rel_set에 존재하는 value들이 index를 가진 값 즉 숫자로 바뀌게 됨

			## rel : key : idx : value
			self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)} ## 여기서 ent_set, rel_set에 존재하는 value들이 index를 가진 값 즉 숫자로 바뀌게 됨

			## rel_reverse : key idx : value
			self.rel2id.update({rel+'_reverse': idx+len(self.rel2id) for idx, rel in enumerate(rel_set)})  ## 여기서 ent_set, rel_set에 존재하는 value들이 index를 가진 값 즉 숫자로 바뀌게 됨


			### xxx2ent 로 끝나는 애들은 id(숫자값)이 들어가면 해당 ent(circ_has_123) 를 출력
			## index : key | rel : value
			self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}

			## index : key | rel : value
			self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}


			self.p.num_ent		= len(self.ent2id)
			self.p.num_rel		= len(self.rel2id) // 2 # reverse가 없는 관계의 수

			## embedding dim      CONVE k_w 10 k_h 10 / 왜 10일까?  kernel weight * kernel hidden
			self.p.embed_dim	= self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim


			'''
			self.data = ddict(list):
	
			self.data는 defaultdict로, 빈 리스트([])를 기본값으로 가지는 딕셔너리입니다.
			예를 들어, self.data['key']에 처음 접근하면 자동으로 self.data['key'] = []가 됩니다.
			이후 self.data['key'].append('value')와 같이 리스트에 값을 추가할 수 있습니다.
			
			'''
			'''
			앞에 붙는 defaultdict는 ddict로 변환 할수 있음 
			
			defalutdict(dict), defaultdict(list)
			'''


			## CircRNA-disease 확률 계산을 위해서
			self.rel_matrix = defaultdict(dict)

			self.rel_indexMatrix = defaultdict(dict)
			self.true_rel = defaultdict(set)

			for circ_rna in circ:
				for disease in dise:
					# index_set에 circRNA-질병 관계가 존재하는지 확인
					sub, obj = self.ent2id[circ_rna], self.ent2id[disease]  ##여기서 해당하는 circ 01418이 숫자로 바뀌게 된다

					if (circ_rna, 'circ-disease', disease) in self.index_set:
						self.rel_matrix[circ_rna][disease] = 1  # 관계가 존재하면 1
						self.rel_indexMatrix[sub]=obj
						self.true_rel['true'].add((circ_rna, disease))
					else:
						self.rel_matrix[circ_rna][disease] = 0  # 존재하지 않으면 0

			self.count = self.count_values(self.rel_matrix, circ, dise) ## circ, dise가 몇개 존재하는지
			self.logger.info('현재 사용하는 데이터셋의 {} 독립된 갯수 입니다.'.format(self.count))

			self.extra_dataset = ddict(list)
			self.extra_dataset_all = ddict(list)

			if self.p.extra_data==True:
				extra_set={
					'ceRNA':'lncrna_pairgene(long).txt',
					'lncRNA':'lncrna_pairgene(short).txt',
					'miRNA': 'mirna_lncrna.txt'
				}

				for key,value in extra_set.items():
					with open('./data/ENCORI/{}/{}'.format(key,value), 'r') as file:
						for line in file:
							sub, rel, obj = map(str.lower, line.strip().split('\t')) # ent_set, 이나 거게서 추가한 데이터를 기반으로
							sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj] ##여기서 해당하는 circ 01418이 숫자로 바뀌게 된다

							self.extra_dataset['encori'].append((sub, rel, obj))
							self.extra_dataset_all['all'].append((sub, rel, obj))

							sr2o[(sub, rel)].add(obj)
							sr2o[(obj, rel + self.p.num_rel)].add(sub)

				extra_set2 = {  ##dise는 여기서 cancer data 추가한것임
					'circRNA': 'lnc2Cancer_CircRNA_dise.txt',
					'lncRNA': 'lnc2Cancer_lncRNA_dise.txt',
				}

				for key, value in extra_set2.items():
					with open('./data/lnc2Cancer_v3.0/{}/{}'.format(key, value), 'r') as file:
						for line in file:
							sub, rel, obj = map(str.lower, line.strip().split('\t'))
							sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj] ##여기서 해당하는 circ 01418이 숫자로 바뀌게 된다 ㅅㅂ

							self.extra_dataset['lnc2cancer'].append((sub, rel, obj))
							self.extra_dataset_all['all'].append((sub, rel, obj))

							sr2o[(sub, rel)].add(obj)
							sr2o[(obj, rel + self.p.num_rel)].add(sub)

				with open('./data/LncRNADisease_v3.0/lncRNA_disease_3.0_lncRNA_disease.txt', 'r') as file:
					for line in file:
						sub, rel, obj = map(str.lower, line.strip().split('\t'))
						sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj] ##여기서 해당하는 circ 01418이 숫자로 바뀌게 된다

						self.extra_dataset['lncRNAdise'].append((sub, rel, obj))
						self.extra_dataset_all['all'].append((sub, rel, obj))

						sr2o[(sub, rel)].add(obj)
						sr2o[(obj, rel + self.p.num_rel)].add(sub)

				size = len(self.extra_dataset_all['all'])
				tr, te, val = self.split_dataset_f(self.extra_dataset_all['all'], size)
				print('done')

				for i in tr:
					self.data['train'].append((i[0], i[1], i[-1])) ##train, test, valid 별로 나뉘어져서 들어감

				for i in val:
					self.data['valid'].append((i[0], i[1], i[-1])) ##train, test, valid 별로 나뉘어져서 들어감

				for i in te:
					self.data['test'].append((i[0], i[1], i[-1])) ##train, test, valid 별로 나뉘어져서 들어감

			## 아 맘에 안든다 왜 ??? 이렇게 작성했지??
			### 아니 파일은 한번에 불러오고 불러온 변수가지고 처리해야지

			## ent 값을 넣어서 index 생성하기 위해서
			for split in ['train', 'test', 'valid']:
				with open('./data/{}/{}/{}.txt'.format(self.p.dataset, self.p.data_name, split)) as file:
					for line in file:
						sub, rel, obj = map(str.lower, line.strip().split('\t'))
						conv_sub, conv_rel, conv_obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj] ## key 값인 ent를 불러오고, indx값을 가져와서 넣는다

						self.data[split].append((conv_sub, conv_rel, conv_obj)) ## train, test, valid 별로 index값을 값으로 넣게됨

						if split == 'train':
							sr2o[(conv_sub, conv_rel)].add(conv_obj)
							sr2o[(conv_obj, conv_rel+self.p.num_rel)].add(conv_sub)


			self.data = dict(self.data) ## dictionary로 변화해서 key
			self.extra_dataset = dict(self.extra_dataset)
			self.extra_datasplit = ddict(list)




			##sr2o 은 train set을 가지고온거임 근데 index로 되어있음
			## sr2o은 search to object임
			self.sr2o = {k: list(v) for k, v in sr2o.items()}
			for split in ['test', 'valid']:
				for sub, rel, obj in self.data[split]: ## index 값 추출
					sr2o[(sub, rel)].add(obj)
					sr2o[(obj, rel+self.p.num_rel)].add(sub)

			## sr2o_all 여기서 키는 sub, rel이 됨 내 새악ㄱ에는 이게 실제 train에 쓸데이터가 아닌가?
			self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
			self.triples  = ddict(list)


			# sr2o는 Train dataset
			## sr2o 에 내꺼를 넣어야 한다
			for (sub, rel), obj in self.sr2o.items():
				self.triples['train'].append({'triple':(sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})
				## 여기서 -1로 지정해둔 이유는 정답을 확정짓지 않고 두기 위해서? 그래서 실제로는 sub rel이 label로 들어간다? 음


			for split in ['test', 'valid']:
				for sub, rel, obj in self.data[split]:
					rel_inv = rel + self.p.num_rel
					self.triples['{}_{}'.format(split, 'tail')].append({'triple': (sub, rel, obj), 	   'label': self.sr2o_all[(sub, rel)]})
					self.triples['{}_{}'.format(split, 'head')].append({'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]}) # 2개의 값으로 이루어짐

			self.triples = dict(self.triples)
			print('done')



			# ##tiples 에 엑스트라 데이터를 추가해야한다 추가하느것도 extra_data = True 일 때만 활성화되게 한다
			# if self.p.extra_data == True: # 새로운 데이터 추가여부!
			# 	for split in ['train', 'test', 'valid']:
			# 		triples[split].append()
			#



		def get_data_loader(dataset_class, split, batch_size, shuffle=True):
			### DataLoader는 helper.py에 정의되어 있음 import torch.utils.data.Dataloader

			## get_data_loader를 실행하게되면 batch_size수에 맞게 데이터가 return됨
			## 선택하고자하는 데이터는 split 변수를 통해서 선택된다
			## self.triples[split] 여기서 split에서 train, test_head, test_tail
			return  DataLoader(
					dataset_class(self.triples[split], self.p),
					batch_size      = batch_size,
					shuffle         = shuffle,
					num_workers     = max(0, self.p.num_workers),
					collate_fn      = dataset_class.collate_fn
				)



		'''
		# dic keys는 train, valid_head, valid_tail, test_head, test_tail
		## get_data_loader를 실행하게되면 batch_size수에 맞게 데이터가 return됨
		## 위에 선언된 get_data_loader를 통해서 self.data_iter를 통해 dict type으로 key를 입력받아서
		## key에 해당하는 데이터를 추출 할 수 있게 해놨음
		
		
		self.triples['train']에 학습데이터가 들어있는데
		len(self.triples['train']) 8885개가 있고,
		triple(1413,5,-1) lebel [1538, ... , ...]
		위에 처럼 들어있음 -1 자리에 해당하는 것을 맞추는것!!
		해당 구조는 self.triples['train']에 key가 3개임 'triples','label', 'sub_samp',
		self.triples 의구조는 dict(list(dict())) 이렇게 되어있음
		그래서 self.triples['train'] 안에 여러 리스트가 있고, 그 리스트 안에 dict가 있고, 여기에 triples , label, sub_samp가 존재함
		
		
		'''



		self.data_iter = {
			'train':    	get_data_loader(TrainDataset, 'train', 	    self.p.batch_size),
			'valid_head':   get_data_loader(TestDataset,  'valid_head', self.p.batch_size),
			'valid_tail':   get_data_loader(TestDataset,  'valid_tail', self.p.batch_size),
			'test_head':   	get_data_loader(TestDataset,  'test_head',  self.p.batch_size),
			'test_tail':   	get_data_loader(TestDataset,  'test_tail',  self.p.batch_size),
		}


		## data load가 끝난뒤에 adj를 call!
		## 여기서 데이터를 가지고 이제 grpah를 생성함
		self.edge_index, self.edge_type = self.construct_adj()



	def construct_adj(self):
		"""
		그냥 이거 edge 생성하는 함수임 실질적으로 adjacency matrixfmfm

		Constructor of the runner class

		Parameters
		----------

		Returns
		-------
		Constructs the adjacency matrix for GCN

		"""
		edge_index, edge_type = [], []

		for sub, rel, obj in self.data['train']:
			edge_index.append((sub, obj))
			edge_type.append(rel)

		# Adding inverse edges
		for sub, rel, obj in self.data['train']:
			edge_index.append((obj, sub))
			edge_type.append(rel + self.p.num_rel)

		edge_index	= torch.LongTensor(edge_index).to(self.device).t()
		edge_type	= torch.LongTensor(edge_type). to(self.device)

		return edge_index, edge_type

	def get_logger(name, log_dir, config_dir):
		"""
		Initializes a logger for logging the results

		Parameters
		----------
		name:       Name of the logger (Typically the model name)
		log_dir:    Directory where logs will be saved
		config_dir: Directory where config files are stored

		Returns
		-------
		logger:     Logger object for recording logs
		"""

		# 로그 디렉토리가 없는 경우 생성
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)

		# 로그 파일 경로 설정
		log_file = os.path.join(log_dir, name.replace(':', '_') + ".log")

		# 로거 생성
		logger = logging.getLogger(name)
		logger.setLevel(logging.INFO)

		# 콘솔 핸들러 설정 (기존의 로그가 콘솔에 출력되는 부분)
		console_handler = logging.StreamHandler()
		console_handler.setLevel(logging.INFO)

		# 파일 핸들러 설정 (로그를 파일에 기록)
		file_handler = logging.FileHandler(log_file)
		file_handler.setLevel(logging.INFO)

		# 로그 포맷 지정
		formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
		console_handler.setFormatter(formatter)
		file_handler.setFormatter(formatter)

		# 핸들러를 로거에 추가
		logger.addHandler(console_handler)
		logger.addHandler(file_handler)

		return logger


	def add_model(self, model, score_func):
		"""
		Creates the computational graph

		Parameters
		----------
		model_name:     Contains the model name to be created

		Returns
		-------
		Creates the computational graph for model and initializes it

		"""

		model_name = '{}_{}'.format(model, score_func)

		# 각각의 edge type, index가 들어감 모델의 egde가 어떻게 들어가는 지 보려면  edge_index, edge_type을 확인
		if   model_name.lower()	== 'compgcn_transe': 	model = CompGCN_TransE(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower()	== 'compgcn_distmult': 	model = CompGCN_DistMult(self.edge_index, self.edge_type, params=self.p)
		elif model_name.lower()	== 'compgcn_conve': 	model = CompGCN_ConvE(self.edge_index, self.edge_type, params=self.p)
		else: raise NotImplementedError # 위에 설정되지 않는 SCORE Function은 오류를 발생!

		model.to(self.device) # gpu에 올라감
		return model

	def add_optimizer(self, parameters):
		"""
		Creates an optimizer for training the parameters

		Parameters
		----------
		parameters:         The parameters of the model

		Returns
		-------
		Returns an optimizer for learning the parameters of the model

		"""

		return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)

	def read_batch(self, batch, split):
		"""
		Function to read a batch of data and move the tensors in batch to CPU/GPU

		Parameters
		----------
		batch: 		the batch to process
		split: (string) If split == 'train', 'valid' or 'test' split


		Returns
		-------
		Head, Relation, Tails, labels
		"""
		if split == 'train':
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label
		else:
			triple, label = [ _.to(self.device) for _ in batch]
			return triple[:, 0], triple[:, 1], triple[:, 2], label

	def save_model(self, save_path):
		"""
		Function to save a model. It saves the model parameters, best validation scores,
		best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

		Parameters
		----------
		save_path: path where the model is saved

		Returns
		-------
		"""
		state = {
			'state_dict'	: self.model.state_dict(),
			'best_val'	: self.best_val,
			'best_epoch'	: self.best_epoch,
			'optimizer'	: self.optimizer.state_dict(),
			'args'		: vars(self.p) # vars를 사용하면 안에 들어오는 변수들이 dict로 변환됨
		}
		torch.save(state, save_path)

	def load_model(self, load_path):
		"""
		Function to load a saved model

		Parameters
		----------
		load_path: path to the saved model

		Returns
		-------
		"""
		state			= torch.load(load_path)
		state_dict		= state['state_dict']
		self.best_val		= state['best_val']
		self.best_val_mrr	= self.best_val['mrr']


		self.best_epoch = state['best_epoch']

		self.model.load_state_dict(state_dict)
		self.optimizer.load_state_dict(state['optimizer'])
		print("model loaded:"+load_path)

	def plot_roc_auc(self, y_true, y_score):
		"""
		Function to plot ROC curve and calculate AUC.

		Parameters:
		----------
		y_true : list or array
			True binary labels.
		y_score : list or array
			Predicted probabilities.
		Returns:
		-------
		auc_score : float
			Computed AUC score.
		"""
		fpr, tpr, _ = roc_curve(y_true, y_score)
		auc_score = auc(fpr, tpr)

		# Plotting ROC curve
		plt.figure(figsize=(10, 6))
		plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
		plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
		plt.title('Receiver Operating Characteristic (ROC) Curve')
		plt.xlabel('False Positive Rate (FPR)')
		plt.ylabel('True Positive Rate (TPR)')
		plt.legend(loc='lower right')
		plt.grid()
		plt.show()

		return auc_score


	def evaluate_rna_disease_metrics(self, rna_disease_probs, index_set, threshold):
		"""
        Function to calculate Precision, Recall, F1 Score, and Accuracy between RNA and disease.

        Parameters:
        ----------
        rna_disease_probs : dict
            Dictionary containing RNA-Disease pair as keys and their predicted probabilities as values.
        true_labels : dict
            Dictionary containing RNA-Disease pair as keys and their true labels (0 or 1) as values.

        Returns:
        -------
        metrics : dict
            Dictionary containing calculated precision, recall, f1 score, and accuracy.
        """

		y_true = []  # 실제 라벨
		y_pred = []  # 예측 확률을 바탕으로 이진 라벨 (threshold 사용)

		# Threshold to decide binary classification based on predicted probabilities


		for rna_disease_pair, pred_prob in rna_disease_probs.items():
			# 실제 라벨 (True: RNA가 질병과 연관되어 있으면 1, 아니면 0)

			#print(pred_prob)
			true_label = 1 if rna_disease_pair in index_set else 0

			if isinstance(threshold, torch.Tensor):
				threshold = threshold.cpu().item()  # GPU Tensor -> CPU -> float

			# 예측된 확률을 사용하여 이진 라벨 생성
			pred_label = 1 if pred_prob >= threshold else 0
			# if pred_label == 0:
			# 	self.threshold += 0.0000001  # threshold 증가
			# # 실제 라벨과 예측된 라벨을 각각 리스트에 추가
			y_true.append(true_label)
			y_pred.append(pred_label)

		# Convert lists to numpy arrays for compatibility with sklearn metrics
		y_true = np.array(y_true)
		y_pred = np.array(y_pred)

		# Calculate metrics
		precision = precision_score(y_true, y_pred, average='binary',zero_division=1)
		recall = recall_score(y_true, y_pred, average='binary',zero_division=1)
		f1 = f1_score(y_true, y_pred, average='binary',zero_division=1)
		acc = accuracy_score(y_true, y_pred)
		auc_score = roc_auc_score(y_true, y_pred)

		pre, rec, th = precision_recall_curve(y_true, y_pred)

		aupr = auc(rec, pre)
		roc = roc_curve(y_true, y_pred)

		print(f'Precision: {precision:.4f}')
		print(f'Recall: {recall:.4f}')
		print(f'F1 Score: {f1:.4f}')
		print(f'Accuracy: {acc:.4f}')
		print(f'AUC: {auc_score:.4f}')

		# Return metrics as a dictionary (optional)
		return {
			'precision': precision,
			'recall': recall,
			'f1': f1,
			'accuracy': acc,
			'aupr' : aupr,
			'roc' : roc,
			'auc' :auc_score,
			'hear' : 'hwang'
		}

	def get_rna_disease_probabilities_new(self):
		"""
		This function calculates the probabilities of diseases associated with RNAs.
		Now optimized with batch processing.
		"""

		self.model.eval()
		rna_disease_probs = {}
		self.list_target = []

		sub_list = []
		rel_list = []
		obj_list = []
		key_list = []

		# 1. Triple 전부 수집
		for c, dise_label in self.rel_matrix.items():
			for disease in dise_label:
				sub_list.append(self.ent2id[c])
				rel_list.append(self.rel2id['circ-disease'])
				obj_list.append(self.ent2id[disease])
				key_list.append((c, 'circ-disease', disease))

		# 2. 텐서로 변환
		sub_tensor = torch.tensor(sub_list).to(self.device)
		rel_tensor = torch.tensor(rel_list).to(self.device)
		obj_tensor = torch.tensor(obj_list).to(self.device)

		with torch.no_grad():
			# 3. 모델 예측
			pred_scores = self.model.forward(sub_tensor, rel_tensor)

			# 4. 예측값 중 타겟 disease에 해당하는 값 추출
			b_range = torch.arange(pred_scores.size()[0], device=self.device).long()
			target_pred = pred_scores[b_range, obj_tensor.long()]



			self.list_target = target_pred.tolist()

			# 5. 딕셔너리에 저장
			for i, key in enumerate(key_list):
				rna_disease_probs[key] = target_pred[i].item()

		# 6. 평균 threshold 저장
		self.threshold = target_pred.mean().item()
		print('get_rna_disease_probabilitues done')

		return rna_disease_probs

	def get_rna_disease_probabilities(self):
		"""
		This function calculates the probabilities of diseases associated with RNAs.
		"""

		self.model.eval()  # Set the model to evaluation mode
		rna_disease_probs = {}
		self.rna_diseLabel = []
		self.list_target = []

		with torch.no_grad():
			for c, dise_label in self.rel_matrix.items():  # Loop through all RNA entities
				for disease, label in dise_label.items():

					# Get all triples where the subject is RNA and the relation is associated with diseases
					rna_triples = [(c, 'circ-disease', disease)]
					# Convert triples into tensor form
					sub_tensor = torch.tensor([self.ent2id[triple[0]] for triple in rna_triples]).to(device=self.device)
					rel_tensor = torch.tensor([self.rel2id[triple[1]] for triple in rna_triples]).to(device=self.device)
					obj_tensor = torch.tensor([self.ent2id[triple[2]] for triple in rna_triples]).to(device=self.device)


					'''
					pred			= self.model.forward(sub, rel)
					b_range			= torch.arange(pred.size()[0], device=self.device)
					target_pred		= pred[b_range, obj]
					# pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
					pred 			= torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)
					pred[b_range, obj] 	= target_pred					
					'''

					# label = torch.tensor([dise_label[disease]]).to(self.device)

					pred_scores = self.model.forward(sub_tensor, rel_tensor)


					b_range = torch.arange(pred_scores.size()[0], device=self.device)
					target_pred = pred_scores[b_range, obj_tensor]
					#pred_scores = torch.where(label.byte(), -torch.ones_like(pred_scores) * 10000000, pred_scores)
					#pred_scores[b_range, obj_tensor] = target_pred



					## torch.where(condition, x , y) condition이 true이면, x, 아니면 y

					self.list_target.append(target_pred)
					#prob_scores = torch.softmax(target_pred, dim=1).max()

					# Store the probabilities for each RNA-disease pair
					# for idx, triple in enumerate(rna_triples):
					# 	## tuple == key
					rna_disease_probs[(c, 'circ-disease',disease)] = target_pred.cpu().numpy()




		average = sum(self.list_target) / len(self.list_target)
		self.threshold = average
		print('get_rna_disease_probabilitues done')

		return rna_disease_probs

	def evaluate(self, split, epoch):
		"""
		Function to evaluate the model on validation or test set

		Parameters
		----------
		split: (string) If split == 'valid' then evaluate on the validation set, else the test set
		epoch: (int) Current epoch count

		Returns
		-------
		resutls:			The evaluation results containing the following:
			results['mr']:         	Average of ranks_left and ranks_right
			results['mrr']:         Mean Reciprocal Rank
			results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

		"""

		self.acc, self.pre, self.f1, self.recall = 0., 0., 0., 0.

		left_results, target1 ,label1 = self.predict(split=split, mode='tail_batch')
		right_results, target2, label2 = self.predict(split=split, mode='head_batch')
		results       = get_combined_results(left_results, right_results)

		mrr = results['mrr']
		self.mrr_results.append(mrr)  # MRR 결과 저장

		self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5}, Head : {:.5}, Avg : {:.5}'.format(epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))



		self.rna_disease_probs = self.get_rna_disease_probabilities()
		best_threshold = self.random_search_threshold(self.rna_disease_probs, self.index_set, self.threshold)

		self.best_threshold = best_threshold
		metrics = self.evaluate_rna_disease_metrics(self.rna_disease_probs, self.index_set, best_threshold)


		self.logger.info(
			'[Epoch {} {}] Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}, Accuracy: {:.4f}, AUPR: {:.4f}, AUC: {:.4f}'.format(
				epoch, split,
				metrics["precision"],
				metrics["recall"],
				metrics["f1"],
				metrics["accuracy"],
				metrics["aupr"],
				metrics["auc"])
		)
		return results



	## 개선된 threshold값을 가져와서 그리고
	def random_search_threshold(self, rna_disease_probs, index_set, threshold):
		"""
			고도화된 threshold 탐색: precision-recall curve 기반
		"""
		y_scores = []
		y_true = []

		for rna_disease_pair, prob in rna_disease_probs.items():
			y_scores.append(prob.item() if isinstance(prob, torch.Tensor) else prob)
			y_true.append(1 if rna_disease_pair in index_set else 0)

		y_scores = np.array(y_scores)
		y_true = np.array(y_true)

		# precision-recall curve 기반으로 threshold 후보 생성
		precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)

		# F1 또는 Fβ score 계산
		beta = 1.0  # F1 기준, Precision 중심은 0.5, Recall 중심은 2.0
		f_beta = (1 + beta ** 2) * (precisions * recalls) / (beta ** 2 * precisions + recalls + 1e-8)

		best_idx = np.argmax(f_beta)
		best_threshold = thresholds[best_idx]
		best_f1 = f_beta[best_idx]

		print('changed_random_search_threshold')
		print(f"Best Threshold: {best_threshold}")
		print(f"Best F{beta} Score: {best_f1:.4f}")
		print(f"Precision: {precisions[best_idx]:.4f}")
		print(f"Recall: {recalls[best_idx]:.4f}")

		return best_threshold

	def random_search_threshold_old(self, rna_disease_probs1, index_set, threshold):
		"""
		    Function to apply Random Search to find the best threshold for binary classification.

		    Parameters:
		    - rna_disease_probs: Dictionary containing RNA-Disease pair probabilities.
		    - index_set: Set containing actual RNA-Disease relations.
		    - n_iter: Number of iterations for random search.

		    Returns:
		    - best_threshold: The best threshold found by random search.
		    """

		cuda_tensor = torch.tensor(threshold, device='cuda:0')

		# 해결 방법: .cpu()로 복사 후 .numpy() 호출
		threshold = cuda_tensor.cpu().numpy()

		# Define scorer for threshold
		def score_threshold(threshold):
			y_true = []
			y_pred = []

			for rna_disease_pair, prob in rna_disease_probs1.items(): ## 이 부분이 말이 안됨
				true_label = 1 if rna_disease_pair in index_set else 0
				pred_label = 1 if prob >= threshold else 0

				y_true.append(true_label)
				y_pred.append(pred_label)

			return f1_score(y_true, y_pred, zero_division=1)

		# Sample thresholds between 0 and 0.25
		np.random.seed(1818)
		thresholds = uniform(loc=threshold, scale=0.019).rvs(size=20)


		best_threshold = None
		best_score = -1

		# Test each threshold and find the one that maximizes F1 score
		for thre in thresholds:
			score = score_threshold(thre)
			if score > best_score:
				best_score = score
				best_threshold = thre

		print(f'Best Threshold: {best_threshold}')
		print(f'Best F1 Score: {best_score}')

		return best_threshold


	def predict(self, split='valid', mode='tail_batch'):
		"""
		Function to run model evaluation for a given mode

		Parameters
		----------
		split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set

		Returns
		-------
		resutls:			The evaluation results containing the following:
			results['mr']:         	Average of ranks_left and ranks_right
			results['mrr']:         Mean Reciprocal Rank
			results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

		"""
		self.model.eval()

		###
		self.pred_sub = {}
		self.pred_obj = {}
		self.pred_label  = {}
		###

		with torch.no_grad():
			results = {}
			train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])]) ### model : tail, head

			for step, batch in enumerate(train_iter):
				sub, rel, obj, label	= self.read_batch(batch, split)
				pred			= self.model.forward(sub, rel)
				b_range			= torch.arange(pred.size()[0], device=self.device)
				target_pred		= pred[b_range, obj]
				# pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
				pred 			= torch.where(label.byte(), -torch.ones_like(pred) * 10000000, pred)
				pred[b_range, obj] 	= target_pred


				# 정답의 순위 계산 (1부터 시작, 높은 값이 높은 순위)
				ranks			= 1 + torch.argsort(torch.argsort(pred, dim=1, descending=True), dim=1, descending=False)[b_range, obj]
				ranks 			= ranks.float()
				results['count']	= torch.numel(ranks) 		+ results.get('count', 0.0)
				results['mr']		= torch.sum(ranks).item() 	+ results.get('mr',    0.0) ## 커질수록 좋은것
				# Mean Reciprocal Rank(MRR): 순위의 역수(1/rank)의 합
				results['mrr']		= torch.sum(1.0/ranks).item()   + results.get('mrr',   0.0) ## mr의 평균을 낸것

				self.pred_obj[obj[0]] = torch.softmax(target_pred.cpu(), dim=0)
				self.pred_sub[sub] = pred.cpu().numpy()
				self.pred_label['label']= label.cpu().numpy()








				for k in range(10):
					results['hits@{}'.format(k+1)] = torch.numel(ranks[ranks <= (k+1)]) + results.get('hits@{}'.format(k+1), 0.0)

				if step % 100 == 0:
					self.logger.info('[{}, {} Step {}]\t{}'.format(split.title(), mode.title(), step, self.p.name))


		return results, self.pred_obj, self.pred_label


	def run_epoch(self, epoch, val_mrr = 0):
		"""
		Function to run one epoch of training

		Parameters

		----------
		epoch: current epoch count

		Returns
		-------
		loss: The loss value after the completion of one epoch
		"""
		self.model.train()
		losses = []
		train_iter = iter(self.data_iter['train'])

		for step, mini_batch in enumerate(train_iter):
			self.optimizer.zero_grad() ## optimizer를 통해서 gradient를 초기화함

			sub, rel, obj, label = self.read_batch(mini_batch, 'train') # 학습데이터를 불러오게됨

			pred	= self.model.forward(sub, rel)
			## pred.size 하면 forward를 지난 각 데이터들의 weight 를 확인할 수 있음

			loss	= self.model.loss(pred, label)

			loss.backward() # loss로 부터 backpropagation 진행

			self.optimizer.step() #가중치 업에이트 !!

			losses.append(loss.item()) ## 여기서 loss값을 넣게된다
			##여기서 threshold 업데이트를 해야한다


			if step % 100 == 0:
				self.logger.info('[Epoch:{}| Step{}]: Train Loss:{:.5},  Val MRR:{:.5}\t{}'.format(epoch, step, np.mean(losses), self.best_val_mrr, self.p.name))


		loss = np.mean(losses)
		#self.scheduler.step()
		self.logger.info('[Epoch:{}]:  Training Loss:{:.4}\n'.format(epoch, loss))
		return loss





	def fit(self):
		"""
		Function to run training and evaluation of model

		Parameters
		----------

		Returns
		-------
		"""
		self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.

		save_path = os.path.join('./checkpoints', self.p.name) ## p.name? testrun 파일을 가지고 그게 있으면 그걸로 돌려라

		if self.p.restore: ## parser.add_arguments에 restore는 True로 존재함 그래서 무조건 실행됨
			self.load_model(save_path)
			self.logger.info('Successfully Loaded previous model')

		kill_cnt = 0
		for epoch in range(self.p.max_epochs):
			train_loss  = self.run_epoch(epoch, val_mrr)
			val_results = self.evaluate('valid', epoch)

			## run_epoch으로 학습을 진행하고 train_loss 를 추출하게 된다
			## 그러고 valdiation set을 통해 학습이 업데이트 된다ㅋ




			## mrr 값을 기준으로 최고 성능이 나왔을때 모델 저장
			## 또는 학습 모델이 성능이 개선되지 않을때 gamma를 통해 학습 속도 조절
			if val_results['mrr'] > self.best_val_mrr:
				self.best_val	   = val_results
				self.best_val_mrr  = val_results['mrr']
				self.best_epoch	   = epoch
				self.save_model(save_path)
				kill_cnt = 0
			else:
				kill_cnt += 1;
				if kill_cnt % 10 == 0 and self.p.gamma > 5:
					self.p.gamma -= 5  # 학습 속도를 조절 할 수 있음
					## 위에 gamma가 변경되면 어디서 수정되냐?
					##
					self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
				if kill_cnt > 25:
					self.logger.info("Early Stopping!!")
					break

			self.logger.info('[Epoch {}]: Training Loss: {:.5}, Valid MRR: {:.5}\n\n'.format(epoch, train_loss, self.best_val_mrr))

		self.logger.info('Loading best model, Evaluating on Test data')
		self.load_model(save_path)


		#랜덤 서치를 사용해 최적의 threshold를 찾음
		best_threshold = self.random_search_threshold(self.rna_disease_probs, self.index_set, self.threshold)

		# # 평가를 진행할 때 최적의 threshold를 사용하도록 설정
		self.threshold = best_threshold

		print(f'Best Threshold: {best_threshold}')

		metrics = self.evaluate_rna_disease_metrics(self.rna_disease_probs, self.index_set, best_threshold)
		self.logger.info(
		 	'log_info'+f'Precision: {metrics["precision"]:.4f}, Recall: {metrics["recall"]:.4f}, F1 Score: {metrics["f1"]:.4f}, Accuracy: {metrics["accuracy"]:.4f}')

		test_results = self.evaluate('test', epoch)




if __name__ == '__main__':
	os.chdir('./ICLR_CompGCN')
	parser = argparse.ArgumentParser(description='Parser For Arguments', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	'''
	parser 사용하는 방법 
	parser.add_argument('-lr', dest='learning_rate', type=float, default=0.01, help="Learning rate")
	위 코드는 parser.add_argument를 통해 원하는 옵션을 저장하는 코드이다 
	불러올 때는 아래의 parser.parse_args()를 실행하여 불러올 수 있음  
	args = parser.parse_args() 함수를 실행해서 parser에 추가된 옵션들에 접근하여 가져올 수 있음  
	args.learning_rate = 0.01 이 실행되는것이다. 
	그래서 
	args.learning_rate에 값을 불러오려면 parser.parse_args()에 접근하여, learning_rate에 접근해야함

	단 이 파일을 실행할때 꼭 넣어줘야함 cmd창에서 python model.py --restore 라고 넣어야 True값이 restore에 저장됨 
	action 옵션은 store_true를 설정하면 값에 아무 값도 넣지 않게 되면 False가 저장되고, 값을 설정하면! True로 설정된다 
	
	
	
	'''

	parser.add_argument('-name',		default='hwangs_compgcn',					help='Set run name for saving/restoring models')
	parser.add_argument('-data',		dest='dataset',         default='KGRACDA',            help='Dataset to use, default: FB15k-237')
	parser.add_argument('-data_name',		dest='data_name',         default='dataset2',            help='Dataset to use, default: dataset2`')
	parser.add_argument('-extra_data', default=False, help='hwang made dataset add?')
	parser.add_argument('-data_experiments', default=False, help='(train,test,valid)data compare split rate experiments')
	parser.add_argument('-train_ratio', default=0.6, help='train ratio')
	parser.add_argument('-valid_ratio', default=0.2, help='valid ratio')
	parser.add_argument('-test_ratio', default=0.2, help='test data ratio')


	parser.add_argument('-model',		dest='model',		default='compgcn',		help='Model Name')

	parser.add_argument('-score_func',	dest='score_func',	default='conve',		help='Score Function for Link prediction')
	parser.add_argument('-opn',             dest='opn',             default='corr',                 help='Composition Operation to be used in CompGCN')

	parser.add_argument('-batch',           dest='batch_size',      default=64,    type=int,       help='Batch size')
	parser.add_argument('-gamma',		type=float,             default=40.0,			help='Margin')
	parser.add_argument('-gpu',		type=str,               default='3',			help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
	parser.add_argument('-epoch',		dest='max_epochs', 	type=int,       default=200,  	help='Number of epochs')
	parser.add_argument('-l2',		type=float,             default=0.0,			help='L2 Regularization for Optimizer')
	parser.add_argument('-lr',		type=float,             default=0.001,			help='Starting Learning Rate')
	parser.add_argument('-lbl_smooth',      dest='lbl_smooth',	type=float,     default=0.1,	help='Label Smoothing')
	parser.add_argument('-num_workers',	type=int,               default=3,                     help='Number of processes to construct batches')
	parser.add_argument('-seed',            dest='seed',            default=41504,  type=int,     	help='Seed for randomization')
									## restore 는 False로 저장됨
	parser.add_argument('-restore',         dest='restore',         action='store_true',            help='Restore from the previously saved model')
	parser.add_argument('-bias',            dest='bias',            action='store_true',            help='Whether to use bias in the model')

	parser.add_argument('-num_bases',	dest='num_bases', 	default=-1,   	type=int, 	help='Number of basis relation vectors to use')
	parser.add_argument('-init_dim',	dest='init_dim',	default=100,	type=int,	help='Initial dimension size for entities and relations')
	parser.add_argument('-gcn_dim',	  	dest='gcn_dim', 	default=200,   	type=int, 	help='Number of hidden units in GCN')
	parser.add_argument('-embed_dim',	dest='embed_dim', 	default=None,   type=int, 	help='Embedding dimension to give as input to score function')
	parser.add_argument('-gcn_layer',	dest='gcn_layer', 	default=3,   	type=int, 	help='Number of GCN Layers to use')
	parser.add_argument('-gcn_drop',	dest='dropout', 	default=0.2,  	type=float,	help='Dropout to use in GCN Layer')
	parser.add_argument('-hid_drop',  	dest='hid_drop', 	default=0.3,  	type=float,	help='Dropout after GCN')

	# ConvE specific hyperparameters
	parser.add_argument('-hid_drop2',  	dest='hid_drop2', 	default=0.3,  	type=float,	help='ConvE: Hidden dropout')
	parser.add_argument('-feat_drop', 	dest='feat_drop', 	default=0.3,  	type=float,	help='ConvE: Feature Dropout')
	parser.add_argument('-k_w',	  	dest='k_w', 		default=10,   	type=int, 	help='ConvE: k_w')
	parser.add_argument('-k_h',	  	dest='k_h', 		default=20,   	type=int, 	help='ConvE: k_h')
	parser.add_argument('-num_filt',  	dest='num_filt', 	default=200,   	type=int, 	help='ConvE: Number of filters in convolution')
	parser.add_argument('-ker_sz',    	dest='ker_sz', 		default=5,   	type=int, 	help='ConvE: Kernel size to use')

	parser.add_argument('-logdir',          dest='log_dir',         default='./log/',               help='Log directory')
	parser.add_argument('-config',          dest='config_dir',      default='./config/',            help='Config directory')
	args = parser.parse_args()


	# args.restore가 False일때 이 문장이 실행이된다. 처음에
	# 만약 False라면 args_name(모델이름) 뒤에 현재 날짜와 시간을 추가함

	## 이전에 모델을 불러오는게 아니라면, 새로 모델이름을 지정해서 저장하게끔 하려고 이 코드가 있는거임
	if not args.restore: args.name = args.name + '_' + time.strftime('%Y_%m_%d') + '_' + time.strftime('%H_%M_%S')
	'''
		args.name의 기본값은 "testrun" 이고 (-name argument)
		거기에 현재 날짜와 시간이 붙습니다
	'''

	set_gpu(args.gpu)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)

	model = Runner(args)
	model.fit()

	for i in range(10):
		print(f"\n[Test Evaluation {i + 1}]")
		model.evaluate('test', model.best_epoch)
