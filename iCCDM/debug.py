import torch

from models import UNet_CCDM, UNet_EDM, DiT_models


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

NC = 3
# IMG_SIZE = 64
# DIM_MULTS= [1,2,2,4,8]
# IMG_SIZE = 128
# DIM_MULTS= [1,2,2,4,4,8]
# IMG_SIZE = 192
# DIM_MULTS = [1,2,2,4,4,8,8]
IMG_SIZE = 256
DIM_MULTS = [1,2,2,4,4,8,8]
# DIM_MULTS = [1,2,2,4,4,8]

embed_input_dim = 128
timesteps = 100

#################################################
# UNet_CCDM


model = UNet_CCDM(
    dim = 80,
    init_dim = None,
    out_dim = None,
    dim_mults = DIM_MULTS,
    channels = 3,
    self_condition = False,
    learned_variance = False,
    learned_sinusoidal_cond = False,
    random_fourier_features = False,
    learned_sinusoidal_dim = 16,
    sinusoidal_pos_emb_theta = 10000,
    dropout = 0.1, # Droputou rate for Resblock, not for cond drop
    attn_dim_head = 32,
    attn_heads = 4,
    full_attn = None,    # defaults to full attention only for inner most layer
    flash_attn = False
)
model = model.cuda()

N=4
x = torch.randn(N, NC, IMG_SIZE, IMG_SIZE).cuda()
t = torch.randint(0,timesteps, size=(N,)).cuda()
c = torch.randn(N, embed_input_dim).cuda()

x_hat = model(x, t, c, return_bottleneck=True)

print(x_hat.size())

print('model size:', get_parameter_number(model))



#################################################
# UNet_EDM

model = UNet_EDM(
    img_resolution = IMG_SIZE,                     
    in_channels = NC,                       
    out_channels = NC,                       
    label_dim           = embed_input_dim,           
    model_channels      = 64,          
    channel_mult        = DIM_MULTS,  
    channel_mult_emb    = 4,            
    num_blocks          = 3,            
    attn_resolutions    = [32,16,8],   
    dropout             = 0.10,        
    cond_drop_prob       = 0.5
)
model = model.cuda()

N=4
x = torch.randn(N, NC, IMG_SIZE, IMG_SIZE).cuda()
t = torch.randint(0,timesteps, size=(N,)).cuda()
c = torch.randn(N, embed_input_dim).cuda()

x_hat = model(x, t, c, return_bottleneck=True)

print(x_hat.size())

print('model size:', get_parameter_number(model))



#################################################
# DiT

model = DiT_models['DiT-B/4'](
    input_size=IMG_SIZE,
    in_channels=NC,
)
model = model.cuda()

N=4
x = torch.randn(N, NC, IMG_SIZE, IMG_SIZE).cuda()
t = torch.randint(0,timesteps, size=(N,)).cuda()
c = torch.randn(N, embed_input_dim).cuda()

x_hat = model(x, t, c)

print(x_hat.size())

print('model size:', get_parameter_number(model))
