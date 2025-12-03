import wandb
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

api = wandb.Api()

run_ids = [
    "l8l55lix", # seed 8
    "myskhkze", # seed 42
    "8y599lfu", # seed 84
    "tkr7jkm6", # seed 727
    "bwne1xjk" # seed 980
]

project_path = "ghtjdaleka-seoul-national-university/defmarl"

full_steps = np.arange(0, 100001)

interp_rewards = []

for rid in run_ids:
    df = api.run(f"{project_path}/{rid}").history(samples=100000)
    
    orig_x = np.arange(len(df))
    orig_y = df["eval/reward"].to_numpy()
    
    masks = ~np.isnan(orig_y)
    orig_x = orig_x[masks]
    orig_y = orig_y[masks]
    
    # interpolate the rewards to original steps
    f = interp1d(orig_x, orig_y, bounds_error=False, fill_value="extrapolate")
    interp_y = f(full_steps)
    interp_rewards.append(interp_y)

interp_rewards = np.array(interp_rewards)

mean_curve = interp_rewards.mean(axis=0)
std_curve = interp_rewards.std(axis=0)

plt.figure(figsize=(10, 5))
plt.plot(full_steps, mean_curve, label="average reward", linewidth=2)
plt.fill_between(full_steps, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

plt.xlabel("Steps")
plt.ylabel("Eval reward")
plt.ylim(-2, 1)
plt.legend()

plt.show()