import math
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from rdkit import DataStructs
from rdkit.Chem import AllChem, MACCSkeys
from rdkit.Chem import AllChem as Chem
import random
import pandas as pd
from sklearn.metrics import mean_squared_error, cohen_kappa_score, mean_absolute_error
from scipy.stats import kendalltau

def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)

def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a ** 2) * fan))
        tensor.data.uniform_(-bound, bound)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

def metrics(y_test, y_pred):
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    # acc = accuracy_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    # recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    # f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    mae = mean_absolute_error(y_test, y_pred)
    qwk = cohen_kappa_score(y_test, y_pred, weights="quadratic")
    tau, _ = kendalltau(y_test, y_pred)
    tolerance_accuracy = np.mean(np.abs(y_test-y_pred) <= 1)
    return mae, qwk, tau, tolerance_accuracy 

def get_Morgan(smiles, radius=2, nBits=512):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(nBits, dtype=np.int8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
    arr = np.zeros((nBits,), dtype=np.int8)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def get_MACCS(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.zeros((167,), dtype=np.int8)
    fp = MACCSkeys.GenMACCSKeys(m)  # 167 bits
    arr = np.zeros((167,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def get_Torsion(smiles, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,), dtype=np.int8)
    arr = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=n_bits)
    return np.array(arr, dtype=np.float32)

def set_seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_report(model, loader, device):
    """classification_report, accuracy, f1"""
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in loader:
            emb1 = batch["emb1"].to(device)
            emb2 = batch["emb2"].to(device)
            fp1 = batch['fp1'].to(device)
            fp2 = batch['fp2'].to(device)
            gene = batch["gene"].to(device)
            labels = batch["label"].cpu().numpy()

            outputs,_ = model(emb1, emb2, gene, fp1, fp2)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_trues.extend(labels.tolist())

    mae, qwk, tau, acc = metrics(all_trues, all_preds)  
    report = classification_report(all_trues, all_preds)
    return mae, qwk, tau, acc, report

def swap_drugs(row):
    return pd.Series({
        'LIBRARY_ID': row['ANCHOR_ID'],
        'ANCHOR_ID': row['LIBRARY_ID'],
        'COSMIC_ID': row['COSMIC_ID'],
        'Label': row['Label']
    })