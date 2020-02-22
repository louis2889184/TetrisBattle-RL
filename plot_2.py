import os 
import matplotlib
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import seaborn as sns

names = [
        # 'logs/tetris_single_ppo_skip4_grid_large',
        #  'logs/tetris_single_ppo_skip4_grid_large_num8',
        #  'logs/tetris_single_ppo_skip4_grid_large_num128',]
        #  'logs/tetris_single_ppo_skip4_grid_large_mini32',]
        #  'logs/tetris_single_ppo_skip4_grid_large_ent0',]
        #  'logs/tetris_single_ppo_skip1_grid_large',]
         'logs/tetris_single_ppo_IOL_{c,h,d}_224_frames12_2mins',
         'logs/tetris_single_ppo_IOL_{c,h,d}_224_frames8_2mins']

rewards = []
line_sents = []
iterations = []

min_max_iteration = np.inf

for name in names:
    with open(name) as f:
        lines = f.readlines()
        rewards.append([])
        line_sents.append([])
        iterations.append([])
        for line in lines:
            line = re.split("[ ,/]+", line)
            # print(line)
            if (len(line) > 5):
                if (line[0] == "Updates"):
                    iterations[-1].append(int(line[4]))
                if (line[1] == "Last"):
                    rewards[-1].append(float(line[8]))
                if (line[1] == "lines"):
                    line_sents[-1].append(float(line[6]))

        min_max_iteration = min(min_max_iteration, iterations[-1][-1])

for i, name in enumerate(names):
    break_point = 0
    for j, iteration in enumerate(iterations[i]):
        if (iteration < min_max_iteration):
            break_point = j
        else:
            break
    
    iterations[i] = iterations[i][:j]
    rewards[i] = rewards[i][:j]
    line_sents[i] = line_sents[i][:j]

## post-process data, set the iteration i ~ i + k to i
freq = 25

for i, name in enumerate(names):
    for j, iteration in enumerate(iterations[i]):
        if j % freq == 0:
            pre = iteration
        else:
            iterations[i][j] = pre

styles = [':', '-']

colors_light = ['lightblue', 'lightgreen', 'mistyrose', 'lightcyan', 'violet', 
    'lightyellow', 'lightgray', 'lightpink'] 
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'gray', 'pink'] 

# fig, ax1 = plt.subplots()
sns.set()
for i, name in enumerate(names):
    # df = pd.DataFrame(rewards[i])
    # print(rewards[i])
    x = np.asarray(iterations[i])
    y = np.asarray(rewards[i])

    ax = sns.lineplot(x=x, y=y, label=name)
    # ax1.plot(iterations[i], df, colors_light[i])
    # ax1.plot(iterations[i], df.rolling(50).mean(), colors[i], label=name)
# ax1.tick_params(axis='y', labelcolor=color)

ax.set_xlabel('iterations')
ax.set_ylabel('rewards')

# plt.legend()
# fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.savefig("rewards.png")

plt.clf()

# fig, ax2 = plt.subplots()

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'

for i, name in enumerate(names):
    x = np.asarray(iterations[i])
    y = np.asarray(line_sents[i])

    ax = sns.lineplot(x=x, y=y, label=name)

plt.legend()

ax.set_xlabel('iterations')

ax.set_ylabel('line_sents')  # we already handled the x-label with ax1

plt.savefig("line.png")
