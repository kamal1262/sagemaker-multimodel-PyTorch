import os 
import argparse
import torch
from torch.autograd import Variable
import numpy as np
import torch.functional as F
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


import torch
import torch.nn as nn
import torch.nn.functional as F
# import pyarrow as pa
# import pyarrow.parquet as pq


## sequence 
import itertools
from collections import Counter
from typing import Dict, List, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

## train module 
import gzip
import pickle
import datetime
import itertools

from typing import Any
from collections import Counter
from typing import Dict, List, Tuple

from torch.utils.data import DataLoader

###
parser = argparse.ArgumentParser(description='Word2Vec recommendation system')

# Data and model checkpoints/otput directories from the container environment
parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])

# args = parser.parse_args()
# model_dir = os.environ['SM_MODEL_DIR']
# output_data_dir = os.environ['SM_OUTPUT_DATA_DIR']
# data_dir = os.environ['SM_CHANNEL_TRAINING']

args = parser.parse_args()
print('model_dir:', args.model_dir)

model_path = os.path.join(args.model_dir, 'model.pth')
checkpoint_path = os.path.join(args.model_dir, 'model_checkpoint.pth')

model_info_path = os.path.join(args.output_data_dir, 'model_info.pth')
checkpoint_state_path = os.path.join(args.output_data_dir, 'model_info.pth')

### 

#load sequnece data
# data_path = '/Users/md.kamal/work-code-sample/location-recommendation/notebooks'

with open(f"{args.data_dir}/list_seq.pickle", 'rb') as f:
    list_seq = pickle.load(f)

with open(f"{args.data_dir}/dict_loc.pickle", 'rb') as f:
    dict_loc = pickle.load(f)

loc2idx = {w: idx for (idx, w) in enumerate(dict_loc)}
idx2loc = {idx: w for (idx, w) in enumerate(dict_loc)}
vocab_size = len(dict_loc)
print(f"vocab_size:{vocab_size}")


vocab_size = 628
torch.manual_seed(1368)

shuffle = True
embedding_dims = 128
epochs = 3
initial_lr = 0.025
batch_size = 16
n_workers = 16

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class SkipGram(nn.Module):

    def __init__(self, emb_size, emb_dim):
        super().__init__()
        self.emb_size = emb_size
        self.emb_dim = emb_dim
        self.center_embeddings = nn.Embedding(emb_size, emb_dim, sparse=True)
        self.context_embeddings = nn.Embedding(emb_size, emb_dim, sparse=True)
        self.init_emb()

    def init_emb(self):
        """
        Init embeddings like word2vec
        Center embeddings have uniform distribution in [-0.5/emb_dim , 0.5/emb_dim].
        Context embeddings are initialized with 0s.
        Returns:
        """
        emb_range = 0.5 / self.emb_dim

        # Initializing embeddings:
        # https://stackoverflow.com/questions/55276504/different-methods-for-initializing-embedding-layer-weights-in-pytorch
        self.center_embeddings.weight.data.uniform_(-emb_range, emb_range)
        self.context_embeddings.weight.data.uniform_(0, 0)

    def forward(self, center, context, neg_context):
        """
        Args:
            center: List of center words
            context: List of context words
            neg_context: List of list of negative context words
        Returns:
        """
        # Calculate positive score
        emb_center = self.center_embeddings(center)  # Get embeddings for center word
        emb_context = self.context_embeddings(context)  # Get embeddings for context word
        emb_neg_context = self.context_embeddings(neg_context)  # Get embeddings for negative context words

        # Next two lines equivalent to torch.dot(emb_center, emb_context) but for batch
        score = torch.mul(emb_center, emb_context)  # Get dot product (part 1)
        score = torch.sum(score, dim=1)  # Get dot product (part2)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)  # Get score for the positive pairs

        # Calculate negative score (for negative samples)
        neg_score = torch.bmm(emb_neg_context, emb_center.unsqueeze(2)).squeeze()  # Get dot product
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        # Return combined score
        return torch.mean(score + neg_score)

    def get_center_emb(self, center):
        return self.center_embeddings(center)

    def get_embeddings(self):
        return self.center_embeddings.weight.cpu().data.numpy()

    def save_embeddings(self, file_name):
        embedding = self.center_embeddings.weight.cpu().data.numpy()
        np.save(file_name, embedding)


class Sequences:
    NEGATIVE_SAMPLE_TABLE_SIZE = 1e7
    WINDOW = 5
    
    def __init__(self, seq_list, vocab_dict, subsample: float = 0.001, power: float = 0.75):
        """
        Initializes a Sequences object for use in a Dataset.
        Args:
            seq_list: rows of sequences (2d array) - global location ID
            vocab_dict: all locations details in dict format
            subsample: Subsampling parameter; suggested range (0, 1e-5)
            power: Negative sampling parameter; suggested 0.75
        """
        self.negative_idx = 0
        self.n_unique_tokens = 0
        
        # Load sequences list
        self.sequences_raw = seq_list
        self.n_sequences = len(self.sequences_raw)
        print('Sequences loaded (length = {:,})'.format(self.n_sequences))
        
        self.loc_freq = self.get_loc_freq()
        print('Location frequency calculated')
        
        # Load vocab dict
        self.vocab_dict = vocab_dict
        self.loc2id, self.id2loc = self.get_mapping_dicts()
        self.n_unique_tokens = len(self.loc2id)
        print('No. of unique tokens: {}'.format(self.n_unique_tokens))
#         save_model(self.loc2id, '{}/loc2id'.format(MODEL_PATH))
#         save_model(self.id2loc, '{}/id2loc'.format(MODEL_PATH))
        print('Loc2Id and Id2Loc created and saved')
        
        self.sequences = self.convert_sequence_to_id()
        self.loc_freq = self.convert_loc_freq_to_id()
        print('Convert sequence and location freq to ID')
        
        self.discard_probs = self.get_discard_probs(sample=subsample)
        print('Discard probability calculated')

        self.neg_table = self.get_negative_sample_table(power=power)
        print('Negative sample table created')
    
    def get_mapping_dicts(self):
        loc2id = {w: idx for (idx, w) in enumerate(self.vocab_dict)}
        id2loc = {idx: w for (idx, w) in enumerate(self.vocab_dict)}
        return loc2id, id2loc
    
    def get_loc_freq(self) -> Counter:
        """
        Returns a dictionary of location frequencies.
        Returns:
        """
        # Flatten list
        seq_flat = list(itertools.chain.from_iterable(self.sequences_raw))

        # Get word frequency
        loc_freq = Counter(seq_flat)
        return loc_freq
    
    def convert_sequence_to_id(self):
        vfunc = np.vectorize(lambda x: self.get_list_location_id(x))
        return vfunc(self.sequences_raw)

    def get_list_location_id(self, x):
        return np.array([self.get_location_id(i) for i in x], dtype=object)

    def get_location_id(self, x):
        return self.loc2id.get(x, -1)

    def convert_loc_freq_to_id(self):
        return {self.loc2id[k]: v for k, v in self.loc_freq.items()}
    
    def get_discard_probs(self, sample=0.001) -> Dict[int, float]:
        """
        Returns a dictionary of locations and their associated discard probability, where the location should be discarded
        if np.random.rand() < probability.
        Args:
            sample: 
        Returns:
        """
        # Convert to array
        loc_freq = np.array(list(self.loc_freq.items()), dtype=np.float64)

        # Convert to probabilities
        loc_freq[:, 1] = loc_freq[:, 1] / loc_freq[:, 1].sum()

        # Perform subsampling
        # http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/
        loc_freq[:, 1] = (np.sqrt(loc_freq[:, 1] / sample) + 1) * (sample / loc_freq[:, 1])

        # Get dict
        discard_probs = {int(k): v for k, v in loc_freq.tolist()}
        return discard_probs

    def get_negative_sample_table(self, power=0.75) -> np.array:
        """
        Returns a table (size = NEGATIVE_SAMPLE_TABLE_SIZE) of negative samples which can be selected via indexing.
        Args:
            power:
        Returns:
        """
        # Convert to array
        loc_freq = np.array(list(self.loc_freq.items()), dtype=np.float64)

        # Adjust by power
        loc_freq[:, 1] = loc_freq[:, 1] ** power

        # Get probabilities
        loc_freq_sum = loc_freq[:, 1].sum()
        loc_freq[:, 1] = loc_freq[:, 1] / loc_freq_sum

        # Multiply probabilities by sample table size
        loc_freq[:, 1] = np.round(loc_freq[:, 1] * self.NEGATIVE_SAMPLE_TABLE_SIZE)

        # Convert to int
        loc_freq = loc_freq.astype(int).tolist()

        # Create sample table
        sample_table = [[tup[0]] * tup[1] for tup in loc_freq]
        sample_table = np.array(list(itertools.chain.from_iterable(sample_table)))
        np.random.shuffle(sample_table)
        return sample_table
    
    # Works on per sequence
    def get_pairs(self, idx, window=5):
        pairs = []
        sequence = self.sequences[idx]

        for center_idx, node in enumerate(sequence):
            for i in range(-window, window + 1):
                context_idx = center_idx + i
                if context_idx > 0 and context_idx < len(sequence) and node != sequence[
                    context_idx] and np.random.rand() < self.discard_probs[sequence[context_idx]]:
                    pairs.append((node, sequence[context_idx]))

        return pairs
    
    def get_negative_samples(self, context, sample_size=5) -> np.array:
        """
        Returns a list of negative samples, where len = sample_size.
        Args:
            sample_size:
        Returns:
        """
        while True:
            # Get a batch from the shuffled table
            neg_sample = self.neg_table[self.negative_idx:self.negative_idx + sample_size]

            # Update negative index
            self.negative_idx = (self.negative_idx + sample_size) % len(self.neg_table)

            # Check if batch insufficient
            if len(neg_sample) != sample_size:
                neg_sample = np.concatenate((neg_sample, self.neg_table[:self.negative_idx]))

            # Check if context in negative sample
            if not context in neg_sample:
                return neg_sample


class SequencesDataset(Dataset):
    def __init__(self, sequences: Sequences, neg_sample_size=5):
        self.sequences = sequences
        self.neg_sample_size = neg_sample_size

    def __len__(self):
        return self.sequences.n_sequences

    def __getitem__(self, idx):
        pairs = self.sequences.get_pairs(idx)
        neg_samples = []
        for center, context in pairs:
            neg_samples.append(self.sequences.get_negative_samples(context))

        return pairs, neg_samples

    @staticmethod
    def collate(batches):
        # logger.info('Batches: {}'.format(batches))
        pairs_batch = [batch[0] for batch in batches]
        neg_contexts_batch = [batch[1] for batch in batches]

        pairs_batch = list(itertools.chain.from_iterable(pairs_batch))
        neg_contexts = list(itertools.chain.from_iterable(neg_contexts_batch))

        centers = [center for center, _ in pairs_batch]
        contexts = [context for _, context in pairs_batch]
        return torch.LongTensor(centers), torch.LongTensor(contexts), torch.LongTensor(neg_contexts)

    @staticmethod
    def collate_for_mf(batches):
        batch_list = []

        for batch in batches:
            pairs = np.array(batch[0])
            negs = np.array(batch[1])
            negs = np.vstack((pairs[:, 0].repeat(negs.shape[1]), negs.ravel())).T

            pairs_arr = np.ones((pairs.shape[0], pairs.shape[1] + 1), dtype=int)
            pairs_arr[:, :-1] = pairs

            negs_arr = np.zeros((negs.shape[0], negs.shape[1] + 1), dtype=int)
            negs_arr[:, :-1] = negs

            all_arr = np.vstack((pairs_arr, negs_arr))
            batch_list.append(all_arr)

        batch_array = np.vstack(batch_list)
        # np.random.shuffle(batch_array)

        # Return item1, item2, label
        return (torch.LongTensor(batch_array[:, 0]), torch.LongTensor(batch_array[:, 1]),
                torch.FloatTensor(batch_array[:, 2]))

# Load dataloader
sequences = Sequences(seq_list=list_seq, vocab_dict=dict_loc)
dataset = SequencesDataset(sequences)
dataloader = DataLoader(dataset, 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        # num_workers=n_workers, 
                        collate_fn=dataset.collate)

# Initialize model
skipgram = SkipGram(vocab_size, embedding_dims).to(device)

# Train loop
optimizer = optim.SparseAdam(list(skipgram.parameters()), lr=initial_lr)

results = []
start_time = datetime.datetime.now()
for epoch in tqdm(range(epochs), total=epochs, position=0, leave=True):
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))
    running_loss = 0
    
    # Training loop
    for i, batches in enumerate(dataloader):
        centers = batches[0].to(device)
        contexts = batches[1].to(device)
        neg_contexts = batches[2].to(device)

        optimizer.zero_grad()
        loss = skipgram.forward(centers, contexts, neg_contexts)
        loss.backward()
        optimizer.step()

        scheduler.step()
        running_loss = running_loss * 0.9 + loss.item() * 0.1
        
    print("Epoch: {}, Loss: {:.4f}, Lr: {:.6f}".format(epoch, running_loss, optimizer.param_groups[0]['lr']))
    results.append([epoch, i, running_loss])
    running_loss = 0

    # save model
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
#     state_dict_path = '{}/skipgram_epoch_{}_{}.pt'.format(MODEL_PATH, epoch, current_datetime)
    # state_dict_path="/Users/md.kamal/work-code-sample/location-recommendation/notebooks/all-model-state.pt"
    # state_dict_path =f"{data_path}/all-model-state.pt"
    checkpoint = { "epoch": epoch,
                  "model_state": skipgram.state_dict(),
                  "optim_state": optimizer.state_dict(),
                  "loc2idx": loc2idx,
                  "idx2loc": idx2loc
                  
    }
    torch.save(checkpoint, checkpoint_path)
#     torch.save(skipgram.state_dict(), state_dict_path)
    print('Model state dict saved to {}'.format(checkpoint_path))

end_time = datetime.datetime.now()
time_diff = round((end_time - start_time).total_seconds() / 60, 2)
print('Total time taken: {:,} minutes'.format(time_diff))

with open(model_path, 'wb') as f:
    torch.save(skipgram.cpu().state_dict(), f)
# save_model(skipgram, model_dir)
# skipgram.save_embeddings(f"{model_dir}/embeddings.npy")
