import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from process_data import drug_feature, SynergyDataset
from utils import make_report, swap_drugs, metrics, set_seed_all
from model import SynergyModel, OrdinalContrastiveLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cv_mode = 3

set_seed_all(42)
drug_file = './GDSC_Combo_data/drug_info.csv'
cline_file = './GDSC_Combo_data/omics_expression.csv'
synergy_file = './GDSC_Combo_data/synergy_data.csv'

synergy_df = pd.read_csv(synergy_file)
drug_df = pd.read_csv(drug_file)
gene_df = pd.read_csv(cline_file)
synergy_df['LIBRARY_ID'] = synergy_df['LIBRARY_ID'].astype(str)
synergy_df['ANCHOR_ID'] = synergy_df['ANCHOR_ID'].astype(str)

# Iterate through each row of df_ids and replace the 'ID' column with the corresponding numerical value.
id_to_value = dict(zip(drug_df['Pubchem_ID'].astype(str), drug_df['SMILES']))
for idx, row in synergy_df.iterrows():
    lib_drug = row['LIBRARY_ID']
    anc_drug = row['ANCHOR_ID']
    synergy_df.at[idx, 'LIBRARY_ID'] = id_to_value.get(lib_drug, lib_drug)
    synergy_df.at[idx, 'ANCHOR_ID'] = id_to_value.get(anc_drug, anc_drug)
    
# Extract all unique SMILES strings and batch convert them into embeddings
unique_smiles = pd.concat([synergy_df["LIBRARY_ID"], synergy_df["ANCHOR_ID"]]).unique()
print(f"[Info] total {len(unique_smiles)} drug SMILES，initialize embedding ...")
smiles_embedding_dict = drug_feature(unique_smiles, model_dir = './ChemBERTa-zinc-base-v1')
# indepent testing
# synergy_df, independ_df = train_test_split(synergy_df, test_size=0.10, 
#                                          stratify=synergy_df['Label'].values, random_state=42)
# 5-fold cross-validation in three scenarios
labels = synergy_df['Label'].values
if cv_mode == 1:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    groups = None
elif cv_mode == 2:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    groups = synergy_df['COSMIC_ID'].values
elif cv_mode == 3:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    groups = synergy_df.apply(lambda r: f"{r['LIBRARY_ID']}__{r['ANCHOR_ID']}", axis=1).values
else:
    raise ValueError('cv_mode must be one of 1 (normal), 2 (cell line), 3 (drug pair)')

fold_results = []
for fold, (tr_idx, te_idx) in enumerate(splitter.split(synergy_df, labels, groups), start=1):
    print(f"\n===== Fold {fold} =====")
    train_df = synergy_df.iloc[tr_idx].reset_index(drop=True)
    test_df  = synergy_df.iloc[te_idx].reset_index(drop=True)
    # Dataset & DataLoader
    train_ds = SynergyDataset(train_df, gene_df, smiles_embedding_dict)
    test_ds  = SynergyDataset(test_df,  gene_df, smiles_embedding_dict)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)
    # Model building
    model = SynergyModel(
        emb_dim=768,
        gene_dim=gene_df.shape[1]-1,
        fp_dim=512,
        hidden_dim=256,
        num_classes=len(synergy_df['Label'].unique())
    ).to(device)

    criterion = OrdinalContrastiveLoss(num_classes=4, ord_temperature=0.8, w_ord=0.3,
                                       con_temperature=0.8, w_con=0.2, class_weight=None)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    fold = fold + 1
    # Epoch training
    best_acc = 0.0
    for epoch in range(0, 30):
        # training process
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            emb1 = batch["emb1"].to(device)
            emb2 = batch["emb2"].to(device)
            fp1 = batch['fp1'].to(device)
            fp2 = batch['fp2'].to(device)
            gene = batch["gene"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs, embed = model(emb1, emb2, gene, fp1, fp2)
            loss = criterion(outputs, embed, labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        # testing process
        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for batch in test_loader:
                emb1 = batch["emb1"].to(device)
                emb2 = batch["emb2"].to(device)
                fp1 = batch['fp1'].to(device)
                fp2 = batch['fp2'].to(device)
                gene = batch["gene"].to(device)
                labels = batch["label"].cpu().numpy()
                outputs, _ = model(emb1, emb2, gene, fp1, fp2)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds.tolist())
                all_trues.extend(labels.tolist())
        mae, qwk, tau, acc = metrics(all_trues, all_preds)  
        print('Epoch:{:3d},'.format(epoch), 'loss_train: {:.4f},'.format(avg_loss),
                           'mae: {:.4f},'.format(mae), 'qwk: {:.4f},'.format(qwk),
                           'tau: {:.4f},'.format(tau), 'acc: {:.4f}'.format(acc))
        # acc, precision, recall, f1 = metrics(all_trues, all_preds)  
        # print('Epoch:{:3d},'.format(epoch), 'loss_train: {:.4f},'.format(avg_loss),
        #                    'acc: {:.4f},'.format(acc), 'precision: {:.4f},'.format(precision),
        #                    'recall: {:.4f},'.format(recall), 'f1: {:.4f}'.format(f1))
        # saving the model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"Results/best_model_fold{fold}.pt")
    # load the best model
    model.load_state_dict(torch.load(f"Results/best_model_fold{fold}.pt"))
    mae, qwk, tau, acc, report = make_report(model, test_loader, device)
    print(f"\n--- Fold {fold-1} Test Report ---\n{report}")
    fold_results.append({"fold": fold, "mae": mae, "qw": qwk, 
                         "tau": tau, "acc": acc})

# 5-cv result saving
res_df = pd.DataFrame(fold_results)
res_df.to_csv('Results/model_performance.csv')
print("\n===== Cross-Validation Summary =====")
print(res_df)

#%%
# clean GPU memory
import torch
torch.cuda.empty_cache()
import gc
gc.collect()
