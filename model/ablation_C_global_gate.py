import torch, torch.nn as nn
from mamba_ssm.modules.mamba_simple import Mamba

class RadarPatchEmbed(nn.Module):
    def __init__(self, in_bins=128, patch_size=16, d_model=256):
        super().__init__()
        self.num_patches = in_bins // patch_size
        self.proj = nn.Conv1d(1, d_model, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(d_model)
    def forward(self, x):
        B, T, Bins = x.shape
        x = x.reshape(B*T, 1, Bins)
        x = self.proj(x).transpose(1,2).contiguous()
        return self.norm(x).reshape(B, T*self.num_patches, -1)

class GlobalGateBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mamba_Rng=Mamba(d_model=d_model,d_state=16,d_conv=4,expand=2)
        self.mamba_mD =Mamba(d_model=d_model,d_state=16,d_conv=4,expand=2)
        self.gate_mD_to_Rng=nn.Sequential(nn.Linear(d_model,d_model),nn.Sigmoid())
        self.gate_Rng_to_mD=nn.Sequential(nn.Linear(d_model,d_model),nn.Sigmoid())
        self.norm_Rng=nn.LayerNorm(d_model); self.norm_mD=nn.LayerNorm(d_model)
        self.ffn_Rng=nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,d_model*2),nn.SiLU(),nn.Linear(d_model*2,d_model))
        self.ffn_mD =nn.Sequential(nn.LayerNorm(d_model),nn.Linear(d_model,d_model*2),nn.SiLU(),nn.Linear(d_model*2,d_model))
    def forward(self,x_Rng,x_mD):
        out_Rng=x_Rng+self.mamba_Rng(self.norm_Rng(x_Rng))
        out_mD =x_mD +self.mamba_mD(self.norm_mD(x_mD))
        g_Rng=self.gate_mD_to_Rng(out_mD.mean(dim=1,keepdim=True))
        g_mD =self.gate_Rng_to_mD(out_Rng.mean(dim=1,keepdim=True))
        x_Rng=x_Rng+out_Rng*g_Rng; x_mD=x_mD+out_mD*g_mD
        x_Rng=x_Rng+self.ffn_Rng(x_Rng); x_mD=x_mD+self.ffn_mD(x_mD)
        return x_Rng,x_mD

class main_Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.d_model=256; self.num_layers=4; self.time_frames=16; self.num_patches=8
        self.pool_mD=nn.AvgPool1d(32,32); self.pool_Rng=nn.AvgPool1d(8,8)
        self.patch_embed_Rng=RadarPatchEmbed(128,16,self.d_model)
        self.patch_embed_mD =RadarPatchEmbed(128,16,self.d_model)
        self.view_embeddings=nn.Embedding(2,self.d_model)
        self.layers=nn.ModuleList([GlobalGateBlock(self.d_model) for _ in range(self.num_layers)])
        self.final_norm_Rng=nn.LayerNorm(self.d_model); self.final_norm_mD=nn.LayerNorm(self.d_model)
        self.frame_head=nn.Sequential(nn.Linear(self.d_model*2,512),nn.SiLU(),nn.Dropout(0.1),nn.Linear(512,17*3))
    def _time_compress(self,x,is_mD):
        p=self.pool_mD if is_mD else self.pool_Rng
        return p(x.transpose(1,2)).transpose(1,2)
    def _process_modality(self,x,is_mD,view_idx):
        x=self._time_compress(x,is_mD)
        e=self.patch_embed_mD if is_mD else self.patch_embed_Rng
        return e(x)+self.view_embeddings(torch.tensor(view_idx,device=x.device))
    def _tokens_to_frames(self,t):
        B,_,d=t.shape
        return t.view(B,2,self.time_frames,self.num_patches,d).mean(dim=(1,3))
    def forward(self,x_mD,x_R):
        x_mD=x_mD.transpose(-1,-2); x_R=x_R.transpose(-1,-2)
        xR=torch.cat([self._process_modality(x_R[:,0],False,0),self._process_modality(x_R[:,1],False,1)],dim=1)
        xM=torch.cat([self._process_modality(x_mD[:,0],True,0),self._process_modality(x_mD[:,1],True,1)],dim=1)
        for l in self.layers: xR,xM=l(xR,xM)
        f=torch.cat([self._tokens_to_frames(self.final_norm_Rng(xR)),self._tokens_to_frames(self.final_norm_mD(xM))],dim=-1)
        o=self.frame_head(f)
        return o.view(o.shape[0],self.time_frames,17,3).contiguous()
