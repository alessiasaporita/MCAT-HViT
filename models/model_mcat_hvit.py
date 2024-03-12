import torch
import torch.nn as nn
from models.model_utils import *
from models.ViT import MultiModalViT


###########################
### MCAT_HViT_Surv Implementation ###
###########################
class MCAT_HViT_Surv(nn.Module):
    def __init__(self, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=2,
                model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25, visual_dropout=0.3):
        super(MCAT_HViT_Surv, self).__init__()
        self.omic_sizes = omic_sizes #[82, 323, 510, 431, 1461, 444]
        self.n_classes = n_classes #2
        self.size_dict_WSI = {"small": [1024, 256, 256], "big": [1024, 512, 384]}
        self.size_dict_omic = {'small': [256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### FC Layer over WSI bag
        """
        Sequential(
            (0): Linear(in_features=1024, out_features=384, bias=True)
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

        ### Path Transformer 
        path_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.path_transformer = nn.TransformerEncoder(path_encoder_layer, num_layers=2)
       
        ### Omic Transformer 
        omic_encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=512, dropout=dropout, activation='relu')
        self.omic_transformer = nn.TransformerEncoder(omic_encoder_layer, num_layers=2)
       
        ### Multimodal Hierarchical Transformer 
        self.transformer = MultiModalViT(num_classes=2, dim=256, depth=4, heads=4, mlp_dim=768, dim_head=128, dropout=0., emb_dropout=0., p_visual_dropout=visual_dropout)


    def forward(self, **kwargs):
        x_path = kwargs['x_path'] #(n_patches, 1024)
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)] #[146, 555, 865, 736, 2424, 733] with met

        h_path_bag = self.wsi_net(x_path).unsqueeze(1) ### path embeddings are fed through a FC layer, (n_patches, 1, 256)
        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic).unsqueeze(1) ### omic embeddings are stacked (to be used in co-attention), (6, 1, 256)

        # Coattn, (Q, K, V)
        h_path_coattn, A_coattn = self.coattn(h_omic_bag, h_path_bag, h_path_bag) #(6, 1, 256), (1, 1, 6, n_patches)

        ### Path: TRANSFORMER WSI
        h_path_trans = self.path_transformer(h_path_coattn).transpose(0, 1) #(6, 1, 256)->(1, 6, 256)
        
        ### Omic: TRANSFORMER GENETIC DATA
        h_omic_trans = self.omic_transformer(h_omic_bag).transpose(0, 1) #(6, 1, 256)->(1, 6, 256)

        ###MultiModal Hierarchical Transformer
        logits = self.transformer(h_path_trans, h_omic_trans)
        attention_scores = {'coattn': A_coattn}

        return logits, attention_scores 


 

