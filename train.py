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

MAX_STEPS = 2_000_000
#EPOCHS = 200

best_loss = float('inf')
steps = 0

while steps < MAX_STEPS:
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_dataloader, desc=f'Step {steps+1}/{MAX_STEPS}', dynamic_ncols=True)
    for batch in progress_bar:

        # unpack batch
        tokens, labels = batch
        anchor_tokens, positive_tokens = tokens

        # move to gpu
        anchor_tokens = anchor_tokens.cuda()
        positive_tokens = positive_tokens.cuda()
        labels = labels.cuda()

        # extract batch size
        b = labels.shape[0]

        # anchor embeddings
        ae = model(anchor_tokens)

        # positive embeddings
        pe = model(positive_tokens.view(-1, 77))
        pe = pe.view(b, 7, -1)

        # combine
        embeddings = torch.cat([ae.unsqueeze(1), pe], dim=1)

        # loss
        loss = loss_fn(embeddings, labels)
        total_loss += loss.item()

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        steps += 1

        progress_bar.set_postfix(loss=loss.item())

        if steps % 20000 == 0:
            avg_loss = total_loss / len(train_dataloader)
            print(f'avg loss: {avg_loss}')
            # save if best
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.module.state_dict(), f'/data/hamaraa/solis_best.pth')

            # reset total loss for next 20k steps
            total_loss = 0

print('training complete!')
torch.save(model.module.state_dict(), '/data/hamaraa/solis_final.pth')
