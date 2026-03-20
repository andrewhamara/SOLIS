import re

# path to your log file
file_path = "nodes.log"

nodes = []

with open(file_path, "r") as f:
    for line in f:
        match = re.search(r"nodes searched:\s*(\d+)", line)
        if match:
            nodes.append(int(match.group(1)))

if nodes:
    avg = sum(nodes) / len(nodes)
    print(f"Nodes: {nodes}")
    print(f"Max: {max(nodes)}")
    print(f"Min: {min(nodes)}")
    print(f"Average nodes searched: {avg:.2f}")
else:
    print("No nodes searched entries found.")
