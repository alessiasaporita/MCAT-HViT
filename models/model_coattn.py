import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import *

"""
Multimodal MCAT
"""

###########################
### MCAT Implementation ###
###########################
class MCAT_Surv(nn.Module):
    def __init__(self, fusion='concat', omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=2,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25):
        super(MCAT_Surv, self).__init__()
        self.fusion = fusion
        self.omic_sizes = omic_sizes #[82, 323, 510, 431, 1461, 444]
        self.n_classes = n_classes #2
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### FC Layer over WSI bag
        """
        Sequential(
            (0): Linear(in_features=1024, out_features=256, bias=True)
            (1): ReLU()
            (2): Dropout(p=0.25, inplace=False)
        """
        size = self.size_dict_WSI[model_size_wsi] 
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic] #[256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)

        ### Multihead Attention
        self.coattn = MultiheadAttention(embed_dim=256, num_heads=1)

        ### Path Transformer + Attention Head
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
        self.path_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.path_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        ### Omic Transformer + Attention Head
        omic_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)
        self.omic_attention_head = Attn_Net_Gated(L=size[2], D=size[2], dropout=dropout, n_classes=1)
        self.omic_rho = nn.Sequential(*[nn.Linear(size[2], size[2]), nn.ReLU(), nn.Dropout(dropout)])
        
        ### Fusion Layer
        if self.fusion == 'concat':
            self.mm = nn.Sequential(*[nn.Linear(256*2, size[2]), nn.ReLU(), nn.Linear(size[2], size[2]), nn.ReLU()])
        elif self.fusion == 'bilinear':
            self.mm = BilinearFusion(dim1=256, dim2=256, scale_dim1=8, scale_dim2=8, mmhid=256)
        else:
            self.mm = None
        
        ### Classifier
        self.classifier = nn.Linear(size[2], n_classes)

    def forward(self, **kwargs):
        x_path = kwargs['x_path'] #(n_patches, 1024)
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)] #[146, 555, 865, 736, 2424, 733] with met

        h_path_bag = self.wsi_net(x_path).unsqueeze(1) ### path embeddings are fed through a FC layer, (n_patches, 1, 256)
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic).unsqueeze(1) ### omic embeddings are stacked (to be used in co-attention), (6, 1, 256)

        # Coattn
        h_path_coattn, A_coattn = self.coattn(h_omic_bag, h_path_bag, h_path_bag) #(6, 1, 256), (1, 1, 6, 3002)

        ###Set-Based MIL Transformer

        ### Path: TRANSFORMER WSI
        h_path_trans = self.path_transformer(h_path_coattn) #(6, 1, 256)
        #global attention pooling
        A_path, h_path = self.path_attention_head(h_path_trans.squeeze(1)) #(6, 1), (6, 256)
        A_path = torch.transpose(A_path, 1, 0)
        h_path = torch.mm(F.softmax(A_path, dim=1) , h_path) #(1, 256)
        h_path = self.path_rho(h_path).squeeze() #(256)
        
        ### Omic: TRANSFORMER GENETIC DATA
        h_omic_trans = self.omic_transformer(h_omic_bag)
        #global attention pooling
        A_omic, h_omic = self.omic_attention_head(h_omic_trans.squeeze(1))
        A_omic = torch.transpose(A_omic, 1, 0)
        h_omic = torch.mm(F.softmax(A_omic, dim=1) , h_omic)
        h_omic = self.omic_rho(h_omic).squeeze() #(256)
        
        if self.fusion == 'bilinear':
            h = self.mm(h_path.unsqueeze(dim=0), h_omic.unsqueeze(dim=0)).squeeze()
        elif self.fusion == 'concat':
            h = self.mm(torch.cat([h_path, h_omic], axis=0)) #(512)->(256)
                
        ### Survival Layer --> classification dead or alive 
        logits = self.classifier(h).unsqueeze(0) #(1, 2)
        
        attention_scores = {'coattn': A_coattn, 'path': A_path, 'omic': A_omic}

        return logits, attention_scores 