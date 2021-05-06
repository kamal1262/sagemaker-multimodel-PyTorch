# Python Built-Ins:
import gzip
import json
import logging
import os
import pickle
from typing import Any

# External Dependencies:
import numpy as np
from scipy.spatial import distance
import torch

# Local Dependencies:
from model import SkipGram


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

VOCAB_SIZE = 628
EMB_SIZE = 128
torch.manual_seed(1368)


# def load_index_mapping(model_dir, model_fname):
#     """
#     Loads model from gzip format
#     Args:
#         model_dir: Path to load model from
#         model_fname: File to load
#     Returns:
#     """
#     model_path = os.path.join(model_dir, model_fname)
#     with gzip.open(model_path, 'rb') as f:
#         model = pickle.load(f)
    
#     logging.info('Model loaded from: {}'.format(model_path))
#     return model


def input_fn(request_body, content_type):
    """An input_fn that loads a json input"""
    logging.info(f"Inference request body: {request_body}")
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        logging.info(f"Inference request json body: {input_data}")
        
        params = {}
        # params['locationIDInput'] = [ loc2id_dict[input_d] for input_d in input_data['locationIDInput'] ]
        params['locationIDInput'] = [ input_d for input_d in input_data['locationIDInput'] ]
        params['count'] = input_data['count'] if 'count' in input_data else 5;
    else:
        raise Exception(
            f'Requested unsupported ContentType in content_type {content_type}'
        )
    return params

def model_fn(model_dir):
    print('declare the model now')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SkipGram(VOCAB_SIZE, EMB_SIZE).to(device)
    
    print('loading the model .....')
    # # loaded_checkpoint = torch.load(model_dir+'/all-model-state.pt', map_location=device)
    # with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
    #     model.load_state_dict(torch.load(f, map_location=device))
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()

    print('loading the model artefacts.......')
    with open(os.path.join(model_dir, 'model_checkpoint.pth'), 'rb') as f:
        model_checkpoint= torch.load(f, map_location=device)
    loc2idx = model_checkpoint['loc2idx']
    idx2loc = model_checkpoint['idx2loc']
    logger.info(f'loc2idx loaded from checkpoint file, length of dic {len(loc2idx)}')
    
    return {"model": model.eval(),"loc2idx": loc2idx, "idx2loc": idx2loc }

def predict_fn(input_data, model_artifact):
    model = model_artifact['model']
    loc2idx = model_artifact['loc2idx']
    idx2loc = model_artifact['idx2loc']

    with torch.no_grad():
        embeddings = model.get_embeddings()
    searched_idx = loc2idx[input_data['locationIDInput'][-1]]
    center_embeddings = embeddings[searched_idx]
    
    closest_index = distance.cdist([center_embeddings], embeddings, "cosine")[0]
    result = zip(range(len(closest_index)), closest_index)
    result = sorted(result, key=lambda x: x[1])
    
    res = []
    for idx, score in result[ :input_data['count']+1 ]:
        tmp_dict = { "id": idx, "score": score }
        res.append(tmp_dict)

    for idx, pred in enumerate(res):
        res[idx]["global_id"] = idx2loc[pred["id"]]
    return res

    
def output_fn(predictions, content_type):
    # for idx, pred in enumerate(predictions):
    #     predictions[idx]["global_id"] = id2loc_dict[pred["id"]]
    return predictions


# if __name__ == "__main__":
    
#     input_sample = """{"locationIDInput": ["mysta_25733"], "count": 10}"""
#     request_content_type = "application/json"
#     response_content_type = "application/json"
# #     # model_dir = "models/"
#     model_dir = '/Users/md.kamal/work-code-sample/temp/test-location-recommendation/artifact/'
 

#     input_obj = input_fn(input_sample, request_content_type)
#     model = model_fn(model_dir)
#     prediction = predict_fn(input_obj, model)
#     output = output_fn(prediction, response_content_type)
    
#     print(output)
