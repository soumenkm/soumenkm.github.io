import torch
import math

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, embedding_dim: int, max_context_length: int):
        super(PositionalEmbedding, self).__init__()
        self.d = embedding_dim
        self.Tmax = max_context_length
        self.register_buffer(name="pos_emb", 
                             tensor=self.get_positional_embedding())
        
    def get_positional_embedding(self) -> torch.tensor:
        mid_index = int(math.ceil((self.d-1)/2)) # excluding mid_index
        position_tensor = torch.arange(0, self.Tmax).unsqueeze(1) # (Tmax, 1)
        pos_emb_list = []
        for i in range(mid_index):
            omega_i = 1 / (10000 ** (2 * i / self.d))
            pos_emb_list.append(torch.sin(position_tensor * omega_i))
            pos_emb_list.append(torch.cos(position_tensor * omega_i))
        
        pos_emb = torch.cat(pos_emb_list, dim=1)  # (Tmax, d)
        assert pos_emb.shape == (self.Tmax, self.d)
        return pos_emb
        
    def forward(self, x):
        """x.shape = (b, T)"""
        b, T = x.shape
        pos_emb = self.pos_emb[:T, :]  # (T, d)
        pos_emb = pos_emb.unsqueeze(0).expand(b, T, self.d)  # (b, T, d)
        return pos_emb

pos_emb_layer = PositionalEmbedding(embedding_dim=4, max_context_length=6)
print(pos_emb_layer(x=torch.rand(size=(2,3))))