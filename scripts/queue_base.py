# launch_all.py
import subprocess, os

# configs
depths = [2, 3, 4, 5]
widths = [1, 2, 3, 5]
elos_by_depth = {
    2: [2000, 2050, 2100, 2150, 2200, 2250, 2300],
    3: [2200, 2250, 2300, 2350, 2400, 2450, 2500],
    4: [2700, 2650, 2600, 2550, 2500, 2450],
    5: [2700, 2650, 2600, 2550, 2500, 2450],
}
elos_by_depth_small = {
    2: [1500, 1550, 1600, 1650, 1700, 1750, 1800],
    3: [1800, 1850, 1900, 1950, 2000, 2050, 2100],
    4: [1950, 2000, 2050, 2100, 2150, 2200],
    5: [1950, 2000, 2050, 2100, 2150, 2200],
}

gpus = list(range(8))
os.makedirs("logs", exist_ok=True)

procs = []
i = 0
for d in depths:
    for w in widths:
        # pick correct Elo list
        if w < 3:
            elos = elos_by_depth_small[d]
        else:
            elos = elos_by_depth[d]

        for elo in elos:
            gpu = gpus[i % len(gpus)]  # round-robin across GPUs
            log_file = f"logs/d{d}_w{w}_elo{elo}.log"
            cmd = f"python relative_base.py {d} {w} {gpu} {elo}"
            print("Launching:", cmd, "->", log_file)
            with open(log_file, "w") as lf:
                procs.append(subprocess.Popen(cmd, shell=True, stdout=lf, stderr=lf))
            i += 1

# wait for all jobs
for p in procs:
    p.wait()
