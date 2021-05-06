# Python Built-Ins:
from collections import Counter
import itertools
import logging
import sys
from typing import Dict

# External Dependencies:
import numpy as np
import torch
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


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
