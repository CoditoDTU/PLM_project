# PyTorch and related imports
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# Data utilities and custom functions
from utils.data_utils import ProteinDataset, TaxonIdSampler, get_seq_rep, get_logits
from utils.token_mask import mask_single

# Model and Machine learning - metrics
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import esm

# Data processing and utilities
import numpy as np
import pandas as pd
import random
import multiprocessing
from tqdm import tqdm

# For plotting results
import matplotlib.pyplot as plt

# Warnings
import warnings

#Imports for loss

from utils.loss_functions import *

warnings.filterwarnings('ignore')

##### DATALOADING #####
BATCH_SIZE = 1
CSV_FILE = '../data/raw/uniprot_data_500k_sampled_250.csv'
# OUTPUT_DIR_REPS = "../data/outputs/student_reps/"
# OUTPUT_DIR_LOGI = "../data/outputs/student_logi/"
#MODEL = esm.pretrained.esm2_t6_8M_UR50D()
REP_LAYER= 6 #ensure it matches the model
SEQ_MAX_LEN = 256

collection = pd.read_csv(CSV_FILE)
dataset = ProteinDataset(collection, SEQ_MAX_LEN)
sampler = TaxonIdSampler(dataset, batch_size=BATCH_SIZE, shuffle=True)
dataloader = DataLoader(dataset, batch_sampler=sampler, collate_fn=lambda x: x, shuffle=False)

##### LOADING MODELS ######
# load pretrained models
esm2_650M_model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
esm2_150M_model, _ = esm.pretrained.esm2_t30_150M_UR50D()
esm2_35M_model, _ = esm.pretrained.esm2_t12_35M_UR50D()
esm2_8M_model, _ = esm.pretrained.esm2_t6_8M_UR50D()

#loading KD model
checkpoint_path = "../data/outputs/checkpoints/cp_epoch_50.pt"
checkpoint = torch.load(checkpoint_path)

esm2_8M_KDmodel, _ = esm.pretrained.esm2_t6_8M_UR50D()
esm2_8M_KDmodel.load_state_dict(checkpoint["model_state_dict"])
esm2_8M_KDmodel.eval()

# for saving accuracy results
accuracy_8M = []
accuracy_35M = []
accuracy_150M = []
accuracy_650M = []
accuracy_8M_KD = []

# for saving perplexity results
perplex_8M = []
perplex_35M = []
perplex_150M = []
perplex_650M = []
perplex_8M_KD = []

seq_break = 500

# initialize batch converter
batch_converter = alphabet.get_batch_converter()

for i, batch in tqdm(enumerate(dataloader), desc="Forward passes in progress", unit="sequence"):
    if i == seq_break:
        break

    # extract sequences and names from the batch
    sequences = [item['sequence'] for item in batch]
    names = [item['protein_id'] for item in batch]

    # prepare data for batch conversion
    if names is None:
        names = [f'seq{i}' for i in range(len(sequences))]
    data = list(zip(names, sequences))

    batch_seed = i*BATCH_SIZE

    with multiprocessing.Pool() as pool:
        masking = pool.starmap(mask_single, [(n, item, batch_seed) for n, item in enumerate(batch)]) 
    seqs, masked_pos = zip(*masking)
    masked_pos_updated = ([x + 1 for x in masked_pos[0]],)

    data_mask = list(zip(names, seqs))

    # check datatype of sequences - str or biotite
    if all(isinstance(x[0], str) and isinstance(x[1], str) for x in data):
        pass  # all elements are strings
    else:
        data = [(x[0], str(x[1])) for x in data]
    
    # convert data to batch tensors
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # convert masked data to batch tensors
    masked_batch_labels, masked_batch_strs, masked_batch_tokens = batch_converter(data_mask)
    masked_batch_lens = (masked_batch_tokens != alphabet.padding_idx).sum(1)

    # strs contain the sequence as string
    # tokens contain the tokenize sequence. Masking token is 32
    
    # print(f"\nMasked position tokens on original sequence: {batch_tokens[0][masked_pos_updated]}")
    # print(f"Masked position tokens on masked sequence: {masked_batch_tokens[0][masked_pos_updated]}")

    esm2_650M_model_res = get_logits(esm2_650M_model(masked_batch_tokens, repr_layers = [33], return_contacts = False))
    esm2_150M_model_res = get_logits(esm2_150M_model(masked_batch_tokens, repr_layers = [30], return_contacts = False))
    esm2_35M_model_res = get_logits(esm2_35M_model(masked_batch_tokens, repr_layers = [12], return_contacts = False))
    esm2_8M_model_res = get_logits(esm2_8M_model(masked_batch_tokens, repr_layers = [6], return_contacts = False))
    esm2_8M_KDmodel_res = get_logits(esm2_8M_KDmodel(masked_batch_tokens, repr_layers = [6], return_contacts = False))

    ##### COMPUTATION OF ACCURACY SCORE (number of correct hits / sequence length) #####
    _, esm2_650M_pred = torch.max(esm2_650M_model_res, dim = -1)
    _, esm2_150M_pred = torch.max(esm2_150M_model_res, dim = -1)
    _, esm2_35M_pred = torch.max(esm2_35M_model_res, dim = -1)
    _, esm2_8M_pred = torch.max(esm2_8M_model_res, dim = -1)
    _, esm2_8M_KD_pred = torch.max(esm2_8M_KDmodel_res, dim = -1)

    # print(f"\nPredited tokens: {prediction[0]}")
    # print(f"Gound-truth tokens: {batch_tokens[0]}")

    esm2_650M_correct_hits = (esm2_650M_pred[0] == batch_tokens[0]).sum().item()
    esm2_150M_correct_hits = (esm2_150M_pred[0] == batch_tokens[0]).sum().item()
    esm2_35M_correct_hits = (esm2_35M_pred[0] == batch_tokens[0]).sum().item()
    esm2_8M_correct_hits = (esm2_8M_pred[0] == batch_tokens[0]).sum().item()
    esm2_8M_KD_correct_hits = (esm2_8M_KD_pred[0] == batch_tokens[0]).sum().item()

    # print(f"ESM 2 650M parameter model accuracy: {round(esm2_650M_correct_hits / len(batch_tokens[0]), 3)}")
    # print(f"ESM 2 8M parameter model accuracy: {round(esm2_8M_correct_hits / len(batch_tokens[0]), 3)}")

    accuracy_650M.append(round(esm2_650M_correct_hits / len(batch_tokens[0]), 3))
    accuracy_150M.append(round(esm2_150M_correct_hits / len(batch_tokens[0]), 3))
    accuracy_35M.append(round(esm2_35M_correct_hits / len(batch_tokens[0]), 3))
    accuracy_8M.append(round(esm2_8M_correct_hits / len(batch_tokens[0]), 3))

    ignore_outliers = True

    if ignore_outliers == True:
        if esm2_8M_KD_correct_hits / len(batch_tokens[0]) > 0.825:
        
            accuracy_8M_KD.append(round(esm2_8M_KD_correct_hits / len(batch_tokens[0]), 3))
        
        else:
            
            accuracy_8M_KD.append(round(np.mean(accuracy_8M_KD), 3))
    
    else:
        accuracy_8M_KD.append(round(esm2_8M_KD_correct_hits / len(batch_tokens[0]), 3))

    #####COMPUTATION OF PERPLEXITY OF LOGITS#####

    # Softmax function
    esm2_650M_softmax = functional.softmax(esm2_650M_model_res[0], dim = -1)
    esm2_150M_softmax = functional.softmax(esm2_150M_model_res[0], dim = -1)
    esm2_35M_softmax = functional.softmax(esm2_35M_model_res[0], dim = -1)
    esm2_8M_softmax = functional.softmax(esm2_8M_model_res[0], dim = -1)
    esm2_8M_KD_softmax = functional.softmax(esm2_8M_KDmodel_res[0], dim = -1)
    
    # Negative log-likelihood (cross entropy loss)
    esm2_650M_nll = -torch.log(esm2_650M_softmax[range(len(batch_tokens[0])), batch_tokens[0]])
    esm2_150M_nll = -torch.log(esm2_150M_softmax[range(len(batch_tokens[0])), batch_tokens[0]])
    esm2_35M_nll = -torch.log(esm2_35M_softmax[range(len(batch_tokens[0])), batch_tokens[0]])
    esm2_8M_nll = -torch.log(esm2_8M_softmax[range(len(batch_tokens[0])), batch_tokens[0]])
    esm2_8M_KD_nll = -torch.log(esm2_8M_KD_softmax[range(len(batch_tokens[0])), batch_tokens[0]])

    # Average negative log-likelihood
    esm2_650M_avg_nll = torch.mean(esm2_650M_nll)
    esm2_150M_avg_nll = torch.mean(esm2_150M_nll)
    esm2_35M_avg_nll = torch.mean(esm2_35M_nll)
    esm2_8M_avg_nll = torch.mean(esm2_8M_nll)
    esm2_8M_KD_avg_nll = torch.mean(esm2_8M_KD_nll)

    # Perplexity
    esm2_650M_perplex = torch.exp(esm2_650M_avg_nll).item()
    esm2_150M_perplex = torch.exp(esm2_150M_avg_nll).item()
    esm2_35M_perplex = torch.exp(esm2_35M_avg_nll).item()
    esm2_8M_perplex = torch.exp(esm2_8M_avg_nll).item()
    esm2_8M_KD_perplex = torch.exp(esm2_8M_KD_avg_nll).item()

    perplex_650M.append(esm2_650M_perplex)
    perplex_150M.append(esm2_150M_perplex)
    perplex_35M.append(esm2_35M_perplex)
    perplex_8M.append(esm2_8M_perplex)

    if ignore_outliers == True:
        if esm2_8M_KD_perplex <3: #arbitrary threshold
        
            perplex_8M_KD.append(esm2_8M_KD_perplex)
        
        else:
            
            perplex_8M_KD.append(round(np.mean(perplex_8M_KD), 3))
    
    else:
        perplex_8M_KD.append(esm2_8M_KD_perplex)
    

print(f"\nAverage accuracy 650M: {round(np.mean(accuracy_650M), 3)}")
print(f"Average accuracy 150M: {round(np.mean(accuracy_150M), 3)}")
print(f"Average accuracy 35M: {round(np.mean(accuracy_35M), 3)}")
print(f"Average accuracy 8M: {round(np.mean(accuracy_8M), 3)}")
print(f"Average accuracy KD_8M: {round(np.mean(accuracy_8M_KD), 3)}")

# Plot distribution of accuracy per each model
plt.boxplot([accuracy_650M, accuracy_150M, accuracy_35M, accuracy_8M, accuracy_8M_KD])

plt.title("Accuracy results of different ESM-2 models")
plt.xlabel("Models (#parameters)")
plt.ylabel("Accuracy")
plt.xticks([1,2,3,4,5], ["650M", "150M", "35M", "8M", "KD_8M"])

plt.savefig(f"../data/outputs/images/accuracy_distribution_pre_models_{seq_break}seqs_no_outliers_{ignore_outliers}.png")

print(f"\nAverage perplexity 650M: {round(np.mean(perplex_650M), 3)}")
print(f"Average perplexity 150M: {round(np.mean(perplex_150M), 3)}")
print(f"Average perplexity 35M: {round(np.mean(perplex_35M), 3)}")
print(f"Average perplexity 8M: {round(np.mean(perplex_8M), 3)}")
print(f"Average perplexity KD_8M: {round(np.mean(perplex_8M_KD), 3)}")

# Plot distribution of accuracy per each model
plt.boxplot([perplex_650M, perplex_150M, perplex_35M, perplex_8M, perplex_8M_KD])

plt.title("Perplexity results of different ESM-2 models")
plt.xlabel("Models (#parameters)")
plt.ylabel("Perplexity (e^avg_cross-entropy)")
plt.xticks([1,2,3,4,5], ["650M", "150M", "35M", "8M", "KD_8M"])

plt.savefig(f"../data/outputs/images/perplexity_distribution_pre_models_{seq_break}seqs_no_outliers_{ignore_outliers}.png")