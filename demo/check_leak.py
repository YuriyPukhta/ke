import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kokoro.pipeline import KPipeline

pipeline = KPipeline(lang_code='a')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load original voices
start_voice = pipeline.load_voice('am_michael').to(device)
target_voice = pipeline.load_voice('af_heart').to(device)

# The token we optimized (index 509)
idx = 509

start_tensor = start_voice[idx]
target_tensor = target_voice[idx]
optimized_tensor = torch.load("demo/optimized_ref_s.pt", weights_only=True).to(device)

print(f"Start tensor shape: {start_tensor.shape}")
print(f"Target tensor shape: {target_tensor.shape}")
print(f"Optimized tensor shape: {optimized_tensor.shape}")

# Calculate distances
dist_start_to_target = torch.nn.functional.l1_loss(start_tensor, target_tensor)
dist_opt_to_target = torch.nn.functional.l1_loss(optimized_tensor, target_tensor)
dist_opt_to_start = torch.nn.functional.l1_loss(optimized_tensor, start_tensor)

print(f"\nDistance (Start -> Target): {dist_start_to_target.item():.4f}")
print(f"Distance (Optimized -> Target): {dist_opt_to_target.item():.4f}")
print(f"Distance (Optimized -> Start): {dist_opt_to_start.item():.4f}")

# Were we just copying the target?
# If Optimized -> Target is very close to 0, it means the gradient perfectly found the exact vector!
# But since our final loss was 0.4892 (not 0.0), it should be close but not exact.
