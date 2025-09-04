import torch
import torch.optim as optim
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler

from dataloader import get_dataloader
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
#optimizer = optim.Adam(model.parameters(), lr=1e-5)
optimizer = optim.SGD(model.parameters(), lr=.05, momentum=.9)

print('loading data...')
batch_size = 256
NUM_POSITIVES = 5
train_dataloader = get_dataloader(batch_size=batch_size, split='train', k_pos=NUM_POSITIVES)

MAX_STEPS = 1_000_000
P_THRESHOLD = 0.05

best_loss = float('inf')
steps = 0

while steps < MAX_STEPS:
    model.train()
    total_loss = 0

    progress_bar = tqdm(train_dataloader, desc=f'Step {steps+1}/{MAX_STEPS}', dynamic_ncols=True)
    for batch in progress_bar:

        anchor_token = batch["anchor"]
        positive_tokens = batch["positives"]
        ps = batch["label"]

        anchor_token = anchor_token.cuda()
        positive_tokens = positive_tokens.cuda()
        ps = ps.cuda()

        # extract batch size
        b = ps.shape[0]

        # anchor embeddings
        ae = model(anchor_token)

        # positive embeddings
        pe = model(positive_tokens.view(-1, 77))
        pe = pe.view(b, NUM_POSITIVES, -1)

        # combine
        embeddings = torch.cat([ae.unsqueeze(1), pe], dim=1)

        with torch.no_grad():
            ps = ps.view(-1, 1)
            p_diffs = torch.abs(ps - ps.T)
            mask = (p_diffs < P_THRESHOLD).float()

        # loss
        loss = loss_fn(embeddings, labels=None, mask=mask)
        total_loss += loss.item()

        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        steps += 1

        progress_bar.set_postfix(loss=loss.item())

        if steps % 20000 == 0:
            torch.save(model.module.state_dict(), f'/data/hamaraa/solis_latest_small.pth')
            total_loss = 0

print('training complete!')
torch.save(model.module.state_dict(), '/data/hamaraa/solis_final_small.pth')
