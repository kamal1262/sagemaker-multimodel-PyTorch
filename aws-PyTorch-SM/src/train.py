### Import Required Libraries #####
###################################

# Python Built-Ins:
import argparse
import datetime
import logging
import os
import pickle
import shutil
import sys
from typing import Any, List, Tuple

# External Dependencies:
import numpy as np
# import pyarrow as pa
# import pyarrow.parquet as pq
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local Dependencies:
from data import Sequences, SequencesDataset
from model import SkipGram
# Optional SM one-click deploy enablement:
#from inference import *


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


# def enable_sm_oneclick_deploy(model_dir):
#     """Copy current running source code folder to model_dir, to enable Estimator.deploy()
#     PyTorch framework containers will load custom inference code if:
#     - The code exists in a top-level code/ folder in the model.tar.gz
#     - The entry point argument matches an existing file
#     ...So to make one-click estimator.deploy() work (without creating a PyTorchModel first), we need
#     to:
#     - Copy the current working directory to model_dir/code
#     - `from inference import *` because "train.py" will still be the entry point (same as the training job)
#     """
#     code_path = os.path.join(model_dir, "code")
#     logger.info(f"Copying working folder to {code_path}")
#     for currpath, dirs, files in os.walk("."):
#         for file in files:
#             # Skip any filenames starting with dot:
#             if file.startswith("."):
#                 continue
#             filepath = os.path.join(currpath, file)
#             # Skip any pycache or dot folders:
#             if ((os.path.sep + ".") in filepath) or ("__pycache__" in filepath):
#                 continue
#             relpath = filepath[len("."):]
#             if relpath.startswith(os.path.sep):
#                 relpath = relpath[1:]
#             outpath = os.path.join(code_path, relpath)
#             logger.info(f"Copying {filepath} to {outpath}")
#             os.makedirs(outpath.rpartition(os.path.sep)[0], exist_ok=True)
#             shutil.copy2(filepath, outpath)
#     return code_path


###### Main application  ############
#####################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ###### Parse input arguments ############
    #########################################
    parser.add_argument("--hosts", type=list, default=os.environ["SM_HOSTS"])
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--embedding-dims', type=str, default=128)
    parser.add_argument('--epochs', type=int, default=128)
    parser.add_argument('--initial-lr', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--n-workers', type=int, default=4)


    args,_ = parser.parse_known_args()
    print('model_dir:', args.model_dir)
    print('Args here:', args)

    model_path = os.path.join(args.model_dir, 'model.pth')
    checkpoint_path = os.path.join(args.model_dir, 'model_checkpoint.pth')

    model_info_path = os.path.join(args.output_data_dir, 'model_info.pth')
    checkpoint_state_path = os.path.join(args.output_data_dir, 'model_info.pth')

    shuffle = args.shuffle
    embedding_dims = args.embedding_dims
    epochs = args.epochs
    initial_lr = args.initial_lr
    batch_size = args.batch_size
    n_workers = args.n_workers

    ###### Load data from input channels ############
    #################################################

    with open(f"{args.data_dir}/list_seq.pickle", 'rb') as f:
        list_seq = pickle.load(f)

    with open(f"{args.data_dir}/dict_loc.pickle", 'rb') as f:
        dict_loc = pickle.load(f)

    loc2idx = {w: idx for (idx, w) in enumerate(dict_loc)}
    idx2loc = {idx: w for (idx, w) in enumerate(dict_loc)}
    vocab_size = len(dict_loc)
    print(f"vocab_size:{vocab_size}")


    
    ###### Train the model ############
    #################################################
    
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - %d", args.num_gpus)
    device = torch.device("cuda" if use_cuda else "cpu")
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    
    torch.manual_seed(1368)
    
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

    # (TQDM is not actually a great idea in a SageMaker job because of how CloudWatch logs work)
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
    
    ###### Save model ############
    ##############################

    with open(model_path, "wb") as f:
        torch.save(skipgram.cpu().state_dict(), f)
    #enable_sm_oneclick_deploy(args.model_dir)
