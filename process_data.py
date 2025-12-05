import os
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from utils import get_Morgan, get_MACCS
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Disable RDKit warnings


@torch.no_grad()
def drug_feature(unique_smiles, model_dir=None):
    """
    Compute SMILES embeddings using a local ChemBERTa model.
    Returns: Dict[str, np.ndarray]: Mapping from SMILES to 768-dim embeddings.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    smiles_embedding_dict = {}

    # Model loading: use local files if provided
    if model_dir is not None and os.path.isdir(model_dir):
        tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        bert_model = AutoModel.from_pretrained(model_dir, local_files_only=True)
    else:
        SMILES_MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
        tokenizer = AutoTokenizer.from_pretrained(SMILES_MODEL_NAME)
        bert_model = AutoModel.from_pretrained(SMILES_MODEL_NAME)

    bert_model.eval().to(device)

    for sm in unique_smiles:
        encoded = tokenizer(sm, padding="max_length", truncation=True,
                            max_length=128, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        smiles_embedding_dict[sm] = cls_embedding

    return smiles_embedding_dict

class SynergyDataset(Dataset):
    def __init__(self, df, gene_df, smiles_embedding_dict, transform=None):
        self.df = df.reset_index(drop=True)
        self.gene_df = gene_df.set_index('COSMIC_ID')
        self.smiles_embedding_dict = smiles_embedding_dict
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        smiles1 = row['LIBRARY_ID']
        smiles2 = row['ANCHOR_ID']

        emb1 = torch.tensor(
            self.smiles_embedding_dict[smiles1], dtype=torch.float32)
        emb2 = torch.tensor(
            self.smiles_embedding_dict[smiles2], dtype=torch.float32)

        # New fingerprint modality
        fp1 = torch.tensor(get_Morgan(smiles1), dtype=torch.float32)
        fp2 = torch.tensor(get_Morgan(smiles2), dtype=torch.float32)

        gene_vec = torch.tensor(
            self.gene_df.loc[row['COSMIC_ID']].values.astype(np.float32),
            dtype=torch.float32
        )
        label = torch.tensor(row['Label'], dtype=torch.long)

        sample = {
            'emb1': emb1,
            'emb2': emb2,
            'fp1': fp1,
            'fp2': fp2,
            'gene': gene_vec,
            'label': label
        }
        if self.transform:
            sample = self.transform(sample)
        return sample
