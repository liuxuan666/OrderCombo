import torch
import torch.nn as nn
import torch.nn.functional as F

class SynergyModel(nn.Module):
    def __init__(self, emb_dim=768, gene_dim=600, fp_dim=512, hidden_dim=256, num_classes=4):
        super(SynergyModel, self).__init__()
        # SMILES embedding branches
        self.drug1_fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.drug2_fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        # Fingerprint branches
        self.fp1_fc = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.fp2_fc = nn.Sequential(
            nn.Linear(fp_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        # Gene expression branch
        self.gene_fc = nn.Sequential(
            nn.Linear(gene_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.pos_emb = nn.Parameter(torch.zeros(1, 3, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, 
                                                   dropout=0.3, 
                                                   dim_feedforward=1024,
                                                   nhead=2, 
                                                   batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Classifier
        self.gate_fc = nn.Linear(hidden_dim * 3, hidden_dim * 3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, emb1, emb2, gene, fp1, fp2):
        emb1 = self.drug1_fc(emb1)
        emb2 = self.drug2_fc(emb2)
        fp1 = self.fp1_fc(fp1)
        fp2 = self.fp2_fc(fp2)
        gene = self.gene_fc(gene)

        # Fuse SMILES embedding and fingerprint for each drug
        drug1 = emb1 + fp1
        drug2 = emb2 + fp2
        
        # Concatenate with gene features
        # prot = drug1 * drug2 * gene  # [B, H]
        # x1 = torch.cat([drug1, drug2, gene, prot], dim=1) # (B,4H)
        x1 = torch.cat([drug1, drug2, gene], dim=1) # (B,3H)
        
        feat = torch.stack([drug1, drug2, gene], dim=1)   # (B,3,H)
        x2 = self.encoder(feat + self.pos_emb)                       # (B,3,H)
        x2 = x2.reshape(x2.size(0), -1)                              # (B, 3H)
        
        gate = torch.sigmoid(self.gate_fc(x2)) 
        x = gate * x1 + (1-gate) * x2
        out = self.classifier(x)
        
        return out, x


class OrdinalContrastiveLoss(nn.Module):
    def __init__(self, num_classes: int, ord_temperature: float = 1.0, w_ord: float = 1.0,
                 con_temperature: float = 0.1, w_con: float = 0.2,
                 class_weight: torch.Tensor = None):
        super().__init__()
        self.num_classes = num_classes
        self.ord_temperature = ord_temperature
        self.w_ord = w_ord

        # 预compute distance matrix [C, C]
        idx = torch.arange(num_classes).unsqueeze(0)
        self.register_buffer('distance_matrix', (idx.T - idx).float().pow(2))

        self.cross_entropy = nn.CrossEntropyLoss()

        self.con_temperature = con_temperature
        self.w_con = w_con
        self.class_weight = class_weight

    def forward(self, logits: torch.Tensor, features: torch.Tensor, 
                labels: torch.LongTensor) -> torch.Tensor:
        device = logits.device
        B, C = logits.shape

        # 1) Cross entropy
        L_ce = self.cross_entropy(logits, labels)

        # 2) Ordinal penalty
        #   P = softmax(logits / T)
        probs = F.softmax(logits / self.ord_temperature, dim=1)  # [B, C]
        #   gather distance row for each sample
        dm = self.distance_matrix.to(device)                    # [C, C]
        tm = dm[labels]                                         # [B, C]
        L_ord = (probs * tm).sum(dim=1).mean()              # scalar

        # 3) Class-aware contrastive
        # 3.1 normalize features
        z = F.normalize(features, p=2, dim=1)                   # [B, D]
        # 3.2 cosine sim / temperature
        sim = (z @ z.T) / self.con_temperature                  # [B, B]
        # mask self-sim
        mask_self = torch.eye(B, device=device).bool()
        sim = sim.masked_fill(mask_self, float('-inf'))

        # 3.3 positive mask: same label & not self
        lbl = labels.view(-1,1)
        mask_pos = (lbl == lbl.T).float().to(device)
        mask_pos = mask_pos.masked_fill(mask_self, 0.0)         # [B, B]

        exp_sim = sim.exp()                                     # [B, B]
        denom = exp_sim.sum(dim=1)                              # [B]

        # weighted numerator if class_weight given
        if self.class_weight is not None:
            w = self.class_weight.to(device)                   # [C]
            wj = w[labels].view(1, B)                          # [1, B]
            num = (exp_sim * mask_pos * wj).sum(dim=1)          # [B]
        else:
            num = (exp_sim * mask_pos).sum(dim=1)               # [B]

        eps = 1e-8
        frac = num / (denom + eps)                             # [B]
        L_con_each = -torch.log(frac + eps)                    # [B]
        L_con = L_con_each.mean()

        # 4) Combine
        loss = (1 - self.w_ord - self.w_con) * L_ce + self.w_ord * L_ord + self.w_con * L_con
        return loss

