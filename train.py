import torch
import torch.optim as optim
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler

from mate_in_k_dataloader import get_dataloader
from infoNCE import SupConLoss
from solis import SOLIS

gpus = [0,1,2,3]

print('setting up model...')
model = SOLIS().cuda()
model = torch.nn.DataParallel(model, device_ids=gpus)
# print parameter count
print(sum(p.numel() for p in model.parameters()))

print('setting up training items...')
loss_fn = SupConLoss().cuda()
optimizer = optim.Adam(model.parameters(), lr=3e-5)

print('loading data...')
batch_size = 128
train_dataloader = get_dataloader(batch_size=batch_size, split='train')

EPOCHS = 200

best_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}')

    for batch in progress_bar:

        # unpack batch
        tokens, labels = batch
        anchor_tokens, positive_tokens = tokens

        # move to gpu
        anchor_tokens = anchor_tokens.cuda()
        positive_tokens = positive_tokens.cuda()
        labels.cuda()

        # extract batch size
        b = labels.shape[0]


        all_tokens = torch.cat([anchor_tokens, positive_tokens], dim=0)

        # anchor forward
        embeddings = model(all_tokens)
        e1, e2 = torch.split(embeddings, [b, b], dim=0)
        embeddings = torch.cat([e1.unsqueeze(1), e2.unsqueeze(1)], dim=1)

        # loss
        loss = loss_fn(embeddings, labels)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        # clip gradients
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())
    #scheduler.step()
    print(f'Epoch {epoch+1} | avg loss: {total_loss / len(train_dataloader):.4f}')

    if loss < best_loss:
        torch.save(model.module.state_dict(), f'/data/hamaraa/solis_best.pth')

print('training complete!')
torch.save(model.module.state_dict(), '/data/hamaraa/solis_final.pth')
progress_bar.close()
