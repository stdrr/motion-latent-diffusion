import torch.nn as nn

from mld.models.architectures.external_models.modules import stsgcn

class Encoder(nn.Module):
    def __init__(self, c_in, h_dim, n_frames, n_joints, dropout) -> None:
        super().__init__()
        
        self.model = nn.ModuleList()
        
        self.model.append(stsgcn.ST_GCNN_layer(c_in,128,[1,1],1,n_frames,
                                           n_joints,dropout))
        self.model.append(stsgcn.ST_GCNN_layer(128,64,[1,1],1,n_frames,
                                               n_joints,dropout))
            
        self.model.append(stsgcn.ST_GCNN_layer(64,128,[1,1],1,n_frames,
                                               n_joints,dropout))
                                               
        self.model.append(stsgcn.ST_GCNN_layer(128,h_dim,[1,1],1,n_frames,
                                               n_joints,dropout))  
        
        self.model = nn.Sequential(*self.model)
        
    def forward(self, x):
        '''
        input shape: [BatchSize, in_Channels, n_frames, n_joints]
        output shape: [BatchSize, h_dim, n_frames, n_joints]
        '''
        return self.model(x)


class STS_Encoder(nn.Module):
    
    name = 'sts_enc'
    
    def __init__(self, c_in, h_dim=32, latent_dim=512, n_frames=12, n_joints=18, reshape=False, **kwargs) -> None:
        super(STS_Encoder, self).__init__()
        
        dropout = kwargs.get('dropout', 0.3)

        self.encoder = Encoder(c_in, h_dim, n_frames, n_joints, dropout)
        
        self.btlnk = nn.Linear(in_features=h_dim * n_frames * n_joints, out_features=latent_dim)

        if reshape:
            self.reshape = lambda x: x.view(-1, n_frames, n_joints, c_in).permute(0,3,1,2).contiguous()
        else:
            self.reshape = lambda x: x
        

    def encode(self, x, return_shape=False):
        x = self.reshape(x)
        assert len(x.shape) == 4, f'Expected 4D input, got {x.shape}'
            
        x = self.encoder(x)
        N, C, T, V = x.shape
        x_shape = x.size()
        x = x.view([N, -1]).contiguous()
        # x = x.view(N, C, T, V)
        # x_shape = x.size()
        # x = x.view(N, -1)
        x = self.btlnk(x)
        
        if return_shape:
            return x, x_shape
        return x
        
    def forward(self, x):
        x = self.encode(x)
        
        return x