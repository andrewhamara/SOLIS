# launch_all.py
import subprocess, os

depths = [2, 3, 4, 5]
widths = [1, 2, 3, 5]
gpus   = list(range(8))

os.makedirs("logs", exist_ok=True)

procs = []
for i, (d, w) in enumerate([(d, w) for d in depths for w in widths]):
    gpu = gpus[i % len(gpus)]  # round-robin over 8 gpus
    log_file = f"logs/d{d}_w{w}.log"
    cmd = f"python run_experiment.py {d} {w} {gpu} > {log_file} 2>&1"
    print("Launching:", cmd)
    procs.append(subprocess.Popen(cmd, shell=True))

# wait for all jobs to finish
for p in procs:
    p.wait()
