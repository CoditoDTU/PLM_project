import torch
import random
import torch
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
import esm
from typing import Union
from pathlib import Path
import os


def esm_compute(seqs: list, names: list=None, model: Union[str, torch.nn.Module]="esm1v", rep_layer: int=33, device=None):
    """
    Compute the of esm_tools models for a list of sequences.
 
    Args:
        seqs (list): protein sequences either as str or biotite.sequence.ProteinSequence.
        names (list, default None): list of names/labels for protein sequences.
            If None sequences will be named seq1, seq2, ...
        model (str, torch.nn.Module): choose either esm2, esm1v or a pretrained model object.
        rep_layer (int): choose representation layer. Default 33.
        device (str): Choose hardware for computation. Default 'None' for autoselection
                          other options are 'cpu' and 'cuda'.
 
    Returns: representations (list) of sequence representation, batch lens and batch labels
 
    Example:
        seqs = ["AGAVCTGAKLI", "AGHRFLIKLKI"]
        results, batch_lens, batch_labels = esm_compute(seqs)
    """
    # detect device
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
 
    # on M1 if mps available
    #if device == torch.device(type='cpu'):
    #    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
 
    # load model
    if isinstance(model, str):
        if model == "esm2":
            model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        elif model == "esm1v":
            model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S()
        else:
            raise ValueError(f"{model} is not a valid model")
    elif isinstance(model, torch.nn.Module):
        alphabet = torch.load(os.path.join(Path(__file__).parent, "alphabet.pt"))
    else:
        raise TypeError("Model should be either a string or a torch.nn.Module object")
 
 
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    model.to(device)
 
    if names == None:
        names = names = [f'seq{i}' for i in range(len(seqs))]
 
    data = list(zip(names, seqs))
 
    # check datatype of sequences - str or biotite
    if all(isinstance(x[0], str) and isinstance(x[1], str) for x in data):
        pass  # all elements are strings
    else:
        data = [(x[0], str(x[1])) for x in data]
 
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
 
    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens.to(device), repr_layers=[rep_layer], return_contacts=True)
 
    return results, batch_lens, batch_labels, alphabet

def get_seq_rep(results, batch_lens, layers):
    """
    Get sequence representations from esm_compute
    """
    token_representations = results["representations"][layers]
 
    # Generate per-sequence representations via averaging
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1: tokens_len - 1].mean(0))
 
    return sequence_representations


def get_taxon_sequence_data(collection):

    documents = collection['results']

    minimal_documents = [] # Initialize new empty dictionary

    for doc in documents:
    # Create a new dictionary with only the desired properties
        new_obj = {
            "primaryAccession": doc.get("primaryAccession",{}),
            "taxonId": doc.get("organism", {}).get("taxonId"),
            "value": doc.get("sequence", {}).get("value"),
            "length": doc.get("sequence", {}).get("length")
        }
        minimal_documents.append(new_obj)
    return minimal_documents


class ProteinDataset(Dataset):

    def __init__(self, data, max_size):

        self.max_size = max_size
        self.data = data[data['sequence_length'] <= max_size]

    def __len__(self):

        return len(self.data)
    
    def __getitem__(self, index):

        
        item = self.data.iloc[index]

        # Extract relevant fields
        sequence = item['sequence']  # The protein sequence
        length = item['sequence_length']   # Length of the sequence
        taxon_id = item['taxon_id']  # Taxon ID
        primary_accession = item['sequence_id']  # Primary accession

        return {
            'sequence': sequence,  # Return sequence as a string (you could modify this later)
            'length': length,
            'taxon_id': taxon_id,
            'protein_id': primary_accession
        }


class TaxonIdSampler(Sampler):
    def __init__(self, dataset: Dataset, batch_size, length_bin_size=5, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.length_bin_size = length_bin_size
        self.shuffle = shuffle

        # Group sample indices by taxonId
        self.taxon_length_bins = defaultdict(lambda: defaultdict(list))

        for idx, sample in enumerate(dataset):
            taxon_id = sample['taxon_id']
            sequence_length = sample['length']
            length_bin = (sequence_length // length_bin_size) * length_bin_size  # integer division to know in which bucket the sequence is

            # Ensure that length_bin is properly initialized
            self.taxon_length_bins[taxon_id][length_bin].append(idx)

        '''
        structure of self.taxon_length_bins:

        {
            taxon_id_1: {
                length_bin_1: [sample_idx_1, sample_idx_2, ...],
                length_bin_2: [sample_idx_3, sample_idx_4, ...],
                ...
            },
            taxon_id_2: {
                length_bin_3: [sample_idx_5, sample_idx_6, ...],
                ...
            },
        }
        '''
        
        # Prepare batches based on taxon groups
        self.batches = []

        for taxon, length_bins in self.taxon_length_bins.items():
            for length_bin, indices in length_bins.items():
                if self.shuffle:
                    random.seed(42)
                    random.shuffle(indices)  # Shuffle the indices if needed
                for i in range(0, len(indices), batch_size):
                    self.batches.append(indices[i:i + batch_size])

        # Shuffle the batches if needed
        if self.shuffle:
            random.shuffle(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
    

def get_logits(results):
    """
    Get logits from esm_compute
    """
    logits = results["logits"]
    return logits