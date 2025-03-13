import torch
import torch.optim as optim
from tqdm import tqdm

from mate_in_k_dataloader import get_dataloader
from infoNCE import InfoNCELoss
from solis import SOLIS

gpus = [0,1,2,3]

print(torch.cuda.device_count())

print('setting up model...')
model = SOLIS().cuda()
model = torch.nn.DataParallel(model, device_ids=gpus)
# print parameter count
print(sum(p.numel() for p in model.parameters()))


print('setting up training items...')
loss_fn = InfoNCELoss(temperature=0.07).cuda()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

print('loading data...')
train_dataloader = get_dataloader(batch_size=16, split='train')
val_dataloader = get_dataloader(batch_size=16, split='val')

MAX_STEPS = 1_000_000
VAL_INTERVAL = 5_000
SAVE_INTERVAL = 10_000

step = 0

progress_bar = tqdm(total=MAX_STEPS, desc='training progress', dynamic_ncols=True)

while step < MAX_STEPS:
    model.train()
    train_loss = 0

    #progress_bar = tqdm(train_dataloader, desc=f'Step {step}/{MAX_STEPS}')
    for anchor_fen, positive_fens, negative_fens in train_dataloader:
        if step >= MAX_STEPS:
            break
        anchor = model(anchor_fen.cuda())

        positive_fens = positive_fens.cuda()
        positives = model(positive_fens.view(-1, positive_fens.size(-1)))
        positives = positives.view(positive_fens.size(0), positive_fens.size(1), -1)

        negative_fens = negative_fens.cuda()
        negatives = model(negative_fens.view(-1, negative_fens.size(-1)))
        negatives = negatives.view(negative_fens.size(0), negative_fens.size(1), -1)

        loss = loss_fn(anchor, positives, negatives)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss.item(), step=step)

        if step % VAL_INTERVAL == 0 and step > 0:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for val_anchor_fen, val_positive_fens, val_negative_fens in val_dataloader:
                    val_anchor = model(anchor_fen.cuda())

                    val_positive_fens = val_positive_fens.cuda()
                    val_positives = model(val_positive_fens.view(-1, val_positive_fens.size(-1)))
                    val_positives = val_positives.view(val_positive_fens.size(0), val_positive_fens.size(1), -1)

                    val_negative_fens = val_negative_fens.cuda()
                    val_negatives = model(val_negative_fens.view(-1, val_negative_fens.size(-1)))
                    val_negatives = negatives.view(val_negative_fens.size(0), val_negative_fens.size(1), -1)

                    v_loss = loss_fn(val_anchor, val_positives, val_negatives)
                    val_loss += v_loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f'validation loss at step {step}: {avg_val_loss:.4f}')
            model.train()

              
        if step % SAVE_INTERVAL == 0 and step > 0:
            torch.save(model.module.state_dict(), f'/data/hamaraa/solis_step_{step}.pth')
        step += 1
    scheduler.step()

print('training complete!')
torch.save(model.module.state_dict(), '/data/hamaraa/solis_final.pth')
progress_bar.close()
