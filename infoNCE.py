import torch
import torch.nn.functional as F

class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positives, negatives):

        #anchor = F.normalize(anchor, dim=-1)
        #positives = F.normalize(positives, dim=-1)
        #negatives = F.normalize(negatives, dim=-1)
        
        #sim_pos = torch.einsum('bd,bpd->bp', anchor, positives)

        sim_pos = torch.einsum('bd,bd->b', anchor, positives).unsqueeze(-1)
        sim_neg = torch.einsum('bd,bnd->bn', anchor, negatives)

        sim_pos /= self.temperature
        sim_neg /= self.temperature

        logits = torch.cat([sim_pos, sim_neg], dim=-1)

        log_softmax = F.log_softmax(logits, dim=-1)

        loss = -log_softmax[:, 0]

        return loss.mean()
