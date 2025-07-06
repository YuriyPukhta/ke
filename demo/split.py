import os

input_file = r"d:\projec\kokoro\demo\gatsby5k.md"
output_dir = r"d:\projec\kokoro\demo\chunks"

os.makedirs(output_dir, exist_ok=True)

with open(input_file, "r", encoding="utf-8") as f:
    content = f.read()

chunks = [c.strip() for c in content.split("\n\n") if c.strip()]

for i, chunk in enumerate(chunks, 1):
    output_file = os.path.join(output_dir, f"chunk_{i:02d}.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(chunk)

print(f"Created {len(chunks)} chunks in {output_dir}")
