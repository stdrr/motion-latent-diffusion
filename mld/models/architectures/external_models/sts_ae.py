import torch
import torch.nn as nn

from mld.models.architectures.external_models.motion_encoders.stsgcn import STS_Encoder
from mld.models.architectures.external_models.motion_decoders.stsgcn import STS_Decoder

def filter_state_dict(state_dict:dict, prefix=''):
    return {k.replace(prefix,''):v for k,v in state_dict.items() if k.startswith(prefix)}


class STSGCN(nn.Module):
    def __init__(self, cfg, conditioning=False):
        super(STSGCN, self).__init__()

        self.c_in = cfg.DATASET.num_coords
        self.h_dim = cfg.motion_sts.params.h_dim
        self.latent_dim = cfg.model.latent_dim[-1]
        if conditioning:
            self.n_frames = abs(cfg.DATASET.condition_len)
        elif cfg.DATASET.condition_len > 0:
            self.n_frames = cfg.DATASET.seg_len - cfg.DATASET.condition_len
        else:
            self.n_frames = cfg.DATASET.seg_len
        self.n_joints = cfg.DATASET.NJOINTS
        self.ckpt = cfg.TRAIN.PRETRAINED_ENC

        self.encoder = STS_Encoder(self.c_in, self.h_dim, self.latent_dim, self.n_frames, self.n_joints)
        self.decoder = STS_Decoder(self.c_in, self.h_dim, self.latent_dim, self.n_frames, self.n_joints)

        self.feat_shape = [self.c_in, self.n_frames, self.n_joints]

        if self.ckpt is not None:
            sd = filter_state_dict(torch.load(self.ckpt)['state_dict'], prefix='vae.')
            self.load_state_dict(sd)
            print("Loaded STS model from {}".format(self.ckpt))


    def forward(self, x, lengths=None):
        x = self.encoder.reshape(x)
        x = self.encoder(x)
        x = self.decoder(x)
        return x, None, None

    
    def encode(self,
               features,
               lengths=None,
               return_lengths=False):
        features = self.encoder.reshape(features)
        z = self.encoder(features).unsqueeze(0)
        if return_lengths:
            return z, 0, lengths # this 0 replaces the dist_m variable
        return z, 0 # this 0 replaces the dist_m variable

    
    def decode(self,
               z,
               lengths=None):
        z = z.squeeze(0)
        feats_rst = self.decoder(z, self.feat_shape)
        feats_rst = feats_rst.permute(0,2,3,1).contiguous().view(-1, self.n_frames, self.n_joints*self.c_in)
        return feats_rst


    def encode_condition(self, features):
        features,_ = self.encode(features)
        return features