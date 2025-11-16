import os
import subprocess

# sweep configs
depths = [2, 3, 4, 5]
widths = [1, 2, 3, 5]

# 8 GPUs indexed 0–7
gpus = list(range(8))

# launch all jobs
procs = []
for i, (d, w) in enumerate([(d, w) for d in depths for w in widths]):
    gpu = gpus[i % len(gpus)]
    log_file = f"logs/solis_d{d}_w{w}.log"
    os.makedirs("logs", exist_ok=True)

    # each job runs your experiment script with args
    cmd = f"python relative_base.py {d} {w} {gpu}"
    print("Launching:", cmd, "->", log_file)
    procs.append(subprocess.Popen(cmd, shell=True))

# wait for everything
for p in procs:
    p.wait()
