import torch
import torch.optim as optim
from tqdm import tqdm

from mate_in_k_dataloader import get_dataloader
from infoNCE import InfoNCELoss
from solis import SOLIS

model = SOLIS().cuda()
# print parameter count
print(sum(p.numel() for p in model.parameters()))

loss_fn = InfoNCELoss(temperature=0.07).cuda()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
dataloader = get_dataloader(batch_size=128)

num_epochs = 10000

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}')
    for anchor_fen, positive_fen, negative_fens in progress_bar:
        anchor = model(anchor_fen.cuda())
        positive = model(positive_fen.cuda())
        negatives = torch.stack([model(fen.cuda()) for fen in negative_fens], dim=1)

        loss = loss_fn(anchor, positive, negatives)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())

    scheduler.step()
    print(f'Epoch {epoch+1}: avg loss: {total_loss / len(dataloader)}')

    if(epoch + 1) % 100 == 0:
        torch.save(model.state_dict(), f'/data/hamaraa/solis_epoch_{epoch+1}.pth')
