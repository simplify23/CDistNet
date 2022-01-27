import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn import functional as F
from torch.nn.modules import Dropout
from torch.nn.modules import Linear
from torch.nn.modules import LayerNorm
from torch.nn.modules import Conv2d

from cdistnet.model.blocks import PositionalEncoding, Embeddings, TransformerEncoderLayer, \
    TransformerEncoder, CommonDecoderLayer, MDCDP, CommonAttentionLayer
# from cdistnet.utils.init import init_weights_xavier
from cdistnet.model.stage.backbone import ResNet45, ResNet31, MTB_nrtr
from cdistnet.model.stage.tps import TPS_SpatialTransformerNetwork


def generate_square_subsequent_mask(sz):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def generate_padding_mask(x):
    padding_mask = x.eq(0)
    return padding_mask

class VIS_Pre(nn.Module):
    def __init__(self,cfg,dim_feedforward):
        super(VIS_Pre, self).__init__()
        self.keep_aspect_ratio = cfg.keep_aspect_ratio
        if cfg.tps_block == 'TPS':
            self.transform = TPS_SpatialTransformerNetwork(
                F=cfg.num_fiducial, I_size=(cfg.height, cfg.width), I_r_size=(cfg.height, cfg.width),
                I_channel_num=1 if cfg.rgb2gray else 3)

        if cfg.feature_block == 'Resnet45':
            self.backbone = ResNet45()
        elif cfg.feature_block == 'Resnet31':
            self.backbone = ResNet31()
        elif cfg.feature_block == 'MTB':
            self.backbone = MTB_nrtr()
        self.positional_encoding = PositionalEncoding(
            dropout=cfg.residual_dropout_rate,
            dim=cfg.hidden_units,
        )

        encoder_layer = TransformerEncoderLayer(cfg.hidden_units, cfg.num_heads, dim_feedforward, cfg.attention_dropout_rate, cfg.residual_dropout_rate)
        self.trans_encoder = TransformerEncoder(encoder_layer, cfg.num_encoder_blocks, None)

    def forward(self,image):
        x = image

        src = torch.sum(torch.abs(image).view(image.shape[0], -1, image.shape[-1]), dim=1)
        src_padding_mask = generate_padding_mask(src)

        x = self.transform(x)
        x = self.backbone(x)

        # # x(b,c,h,w)
        if self.keep_aspect_ratio:
            r = round(src_padding_mask.shape[1] / x.shape[1])
            src_key_padding_mask = src_padding_mask[:, ::r]
            memory_key_padding_mask = src_key_padding_mask
        else:
            src_key_padding_mask, memory_key_padding_mask = None, None

        # 1d
        x = self.positional_encoding(x.permute(1, 0, 2))
        # memory: S N E
        memory = self.trans_encoder(x, mask=None, src_key_padding_mask=src_key_padding_mask)
        return memory,memory_key_padding_mask


class SEM_Pre(nn.Module):
    def __init__(self,cfg):
        super(SEM_Pre, self).__init__()
        self.embedding = Embeddings(
            d_model=cfg.hidden_units,
            vocab=cfg.dst_vocab_size,
            padding_idx=0,
            scale_embedding=cfg.scale_embedding
        )

        self.positional_encoding = PositionalEncoding(
            dropout=cfg.residual_dropout_rate,
            dim=cfg.hidden_units,
        )
    def forward(self,tgt):
        # tgt = tgt[:, :-1]
        # tgt(b,max_len)
        # # image(b,c,h,w)

        tgt_key_padding_mask = generate_padding_mask(tgt)
        # tgt_key_padding_mask:record 0 padding for true in tgt.same as image
        tgt = self.embedding(tgt).permute(1, 0, 2)
        # print("tgt.shape{}".format(tgt.shape))
        tgt = self.positional_encoding(tgt)
        # tgt(max_len,b,d_model(hidden_unit))
        tgt_mask = generate_square_subsequent_mask(tgt.shape[0]).to(device=tgt.device)
        return tgt,tgt_mask,tgt_key_padding_mask

class POS_Pre(nn.Module):
    def __init__(self, cfg):
        super(POS_Pre, self).__init__()
        d_model = cfg.hidden_units
        self.pos_encoding = PositionalEncoding(
            dropout=cfg.residual_dropout_rate,
            dim=d_model,
        )
        self.linear1 = Linear(d_model, d_model)
        self.linear2 = Linear(d_model, d_model)

        self.norm2 = LayerNorm(d_model)


    def forward(self,tgt):
        pos = tgt.new_zeros(*tgt.shape)
        pos = self.pos_encoding(pos)

        pos2 = self.linear2(F.relu(self.linear1(pos)))
        pos = self.norm2(pos + pos2)
        return pos

class CDistNet(nn.Module):
    def __init__(self, dim_feedforward=2048, cfg=None):
        super(CDistNet, self).__init__()

        self.d_model = cfg.hidden_units
        self.nhead = cfg.num_heads
        self.keep_aspect_ratio = cfg.keep_aspect_ratio

        self.visual_branch = VIS_Pre(cfg,dim_feedforward)
        self.semantic_branch = SEM_Pre(cfg)
        self.positional_branch = POS_Pre(cfg)

        decoder_layer = CommonAttentionLayer(cfg.hidden_units, cfg.num_heads, dim_feedforward // 2, cfg.attention_dropout_rate,
                                           cfg.residual_dropout_rate)
        self.mdcdp = MDCDP(decoder_layer, cfg.num_decoder_blocks)
        self._reset_parameters()

        self.tgt_word_prj = nn.Linear(cfg.hidden_units, cfg.dst_vocab_size, bias=False)
        self.tgt_word_prj.weight.data.normal_(mean=0.0, std=cfg.hidden_units ** -0.5)

    def forward(self, image, tgt):
        tgt = tgt[:, :-1]
        vis_feat,vis_key_padding_mask = self.visual_branch(image)
        sem_feat,sem_mask,sem_key_padding_mask = self.semantic_branch(tgt)
        pos_feat = self.positional_branch(sem_feat)
        # if x.size(1) != tgt.size(1):
        #     raise RuntimeError("the batch number of src and tgt must be equal")
        #
        # if x.size(2) != self.d_model or tgt.size(2) != self.d_model:
        #     raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        output = self.mdcdp(sem_feat, vis_feat, pos_feat,
                            tgt_mask=sem_mask,
                            memory_mask=None,
                            tgt_key_padding_mask=sem_key_padding_mask,
                            memory_key_padding_mask=vis_key_padding_mask)

        output = output.permute(1, 0, 2)

        logit = self.tgt_word_prj(output)
        return logit.view(-1, logit.shape[2])

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

def build_CDistNet(cfg):
    net = CDistNet(dim_feedforward=2048, cfg=cfg)
    return net

