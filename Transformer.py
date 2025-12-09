import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import wandb
wandb.login()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



import torch
import torch.nn as nn
import torch.nn.functional as F


class IndexPositionalEncoding(nn.Module):
    def __init__(self, max_len):
        super().__init__()
        self.register_buffer("positions", torch.arange(max_len).float())

    def forward(self, x):
        pos = self.positions[:x.size(1)].unsqueeze(0).unsqueeze(-1)  
        pos = pos.repeat(x.size(0), 1, 1)                           
        return torch.cat([x, pos], dim=-1)                           



class TransformerVAE(nn.Module):
    def __init__(self, channels=6, time_steps=20,
                 latent_T=10, d_model=64, nhead=4, num_layers=3):
        super().__init__()

        self.channels = channels
        self.time_steps = time_steps
        self.latent_T = latent_T
        self.d_model = d_model

        self.pos_encoder = IndexPositionalEncoding(time_steps)
        self.encoder_input_proj = nn.Linear(channels + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=128,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.to_latent = nn.Linear(time_steps, latent_T)

        self.mu = nn.Linear(d_model, channels)
        self.logvar = nn.Linear(d_model, channels)

        self.from_latent = nn.Linear(latent_T, time_steps+1)

        self.decoder_input_proj = nn.Linear(channels, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True,
            dim_feedforward=128,
            dropout=0.1,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_layer = nn.Linear(d_model, channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        x = self.pos_encoder(x)                 
        x = self.encoder_input_proj(x)           
        h = self.encoder(x)                    

        h = h.transpose(1, 2)

        h_latent = self.to_latent(h)             

        h_latent = h_latent.transpose(1, 2)

        mu = self.mu(h_latent)                   
        logvar = self.logvar(h_latent)           

        return mu, logvar


    def decode(self, z):
        
        z = z.transpose(1, 2)                    
        z = self.from_latent(z)                 
        z = z.transpose(1, 2)                    

        h = self.decoder_input_proj(z)          

        memory = h.clone()

        out = self.decoder(h, memory)           
        out = self.output_layer(out)            

        return out


    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)      
        recon = self.decode(z)                   
        return recon, mu, logvar


    
def loss_function(y, x_recon, mu, logvar):
    recon_loss = F.mse_loss(x_recon, y, reduction='sum')
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def training_(inp_model, inp_x, inp_y, inp_datasets, inp_n_epochs=2000):
    wandb.init(project = f'nonCausalGranTVAE_{inp_datasets}')
    print('Training TransformerVAE... ')
    optimizer = optim.Adam(inp_model.parameters(), lr=1e-3)
    inp_model.to('cuda')
    inp_model.train()
    loss_values = []
    for epochs in tqdm(range(inp_n_epochs)):
        x_recon, mu, logvar = inp_model(inp_x)
        loss = loss_function(inp_y, x_recon, mu, logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
        wandb.log({'Train/Loss':loss.item()})
    return inp_model, loss
    
def inference_(inp_model, inp_x):
    inp_model.eval()
    pred, _, _ = inp_model(inp_x)
    return pred[:,-1,:]

wandb.finish()