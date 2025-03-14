import torch
import torch.optim as optim
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from mate_in_k_dataloader import get_dataloader
from infoNCE import InfoNCELoss
from solis import SOLIS

gpus = [0,1,2,3]

print('setting up model...')
model = SOLIS().cuda()
model = torch.nn.DataParallel(model, device_ids=gpus)
# print parameter count
print(sum(p.numel() for p in model.parameters()))


print('setting up training items...')
loss_fn = InfoNCELoss(temperature=0.07).cuda()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1e-5)

print('loading data...')
train_dataloader = get_dataloader(batch_size=32, split='train')

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

first_1000_samples = []
target_size = 10_000

## Iterate through the dataloader
#for batch in train_dataloader:
#    anchor_token, positive_token, negative_tokens = batch
#
#    batch_size = anchor_token.shape[0]
#    
#    for i in range(batch_size):
#        if len(first_1000_samples) >= target_size:
#            break
#        first_1000_samples.append([anchor_token, positive_token, negative_tokens])
#    
#    if len(first_1000_samples) >= target_size:
#        break


EPOCHS = 10
SAVE_INTERVAL = 1

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{EPOCHS}')

    for batch in progress_bar:

        # unpack batch and move to gpu
        anchor_token, positive_token, negative_tokens = batch
        anchor_token, positive_token, negative_tokens = anchor_token.cuda(), positive_token.cuda(), negative_tokens.cuda()

        # anchor forward
        anchor = model(anchor_token)

        # positive forward
        positive = model(positive_token)

        # negatives forward
        negatives = model(negative_tokens.view(-1, negative_tokens.size(-1)))
        negatives = negatives.view(negative_tokens.size(0), negative_tokens.size(1), -1)

        # loss
        loss = loss_fn(anchor, positive, negatives)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()

        # clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)

        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())
    scheduler.step()
    print(f'Epoch {epoch+1} | avg loss: {total_loss / len(train_dataloader):.4f}')

    if epoch % SAVE_INTERVAL == 0 and epoch > 0:
        torch.save(model.module.state_dict(), f'/data/hamaraa/solis_epoch_{epoch}.pth')

print('training complete!')
torch.save(model.module.state_dict(), '/data/hamaraa/solis_final.pth')
progress_bar.close()
