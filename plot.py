import os 
import matplotlib
import re
import numpy as np
import matplotlib.pyplot as plt

names = ['logs/tetris_single_ppo_IO_holes', 'logs/tetris_single_ppo_skip4']
fig_name = "fig_holes.png"
rewards = []
line_sents = []
iterations = []

min_rewards = []
max_rewards = []

min_sents = []
max_sents = []

min_max_iteration = np.inf

for name in names:
    with open(name) as f:
        lines = f.readlines()
        rewards.append([])
        line_sents.append([])
        iterations.append([])
        min_rewards.append([])
        max_rewards.append([])

        min_sents.append([])
        max_sents.append([])
        for line in lines:
            line = re.split("[ ,/\n]+", line)
            # print(line)
            if (len(line) > 5):
                if (line[0] == "Updates"):
                    iterations[-1].append(int(line[4]))
                if (line[1] == "Last"):
                    rewards[-1].append(float(line[8]))
                    min_rewards[-1].append(float(line[-3]))
                    max_rewards[-1].append(float(line[-2]))
                if (line[1] == "lines"):
                    line_sents[-1].append(float(line[6]))
                    min_sents[-1].append(float(line[-3]))
                    max_sents[-1].append(float(line[-2]))

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

    min_rewards[i] = min_rewards[i][:j]
    max_rewards[i] = max_rewards[i][:j]
    min_sents[i] = min_sents[i][:j]
    max_sents[i] = max_sents[i][:j]

styles = [':', '-']

# fig, ax1 = plt.subplots(2, 2, 1)

fig, axes = plt.subplots(2, 2)

axes = axes.flatten()

ax1 = axes[0]

color = 'tab:red'
ax1.set_xlabel('iterations')
ax1.set_ylabel('mean rewards', color=color)

for i, name in enumerate(names):
    ax1.plot(iterations[i], rewards[i], color=color, linestyle=styles[i], label=name)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('mean line_sents', color=color)  # we already handled the x-label with ax1
for i, name in enumerate(names):
    # print(line_sents[i])
    ax2.plot(iterations[i], line_sents[i], color=color, linestyle=styles[i], label=name)
ax2.tick_params(axis='y', labelcolor=color)
plt.legend()


# fig, ax1 = plt.subplots(2, 2, 2)
ax1 = axes[1]

color = 'tab:red'
ax1.set_xlabel('iterations')
ax1.set_ylabel('max rewards', color=color)

for i, name in enumerate(names):
    ax1.plot(iterations[i], max_rewards[i], color=color, linestyle=styles[i], label=name)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('max line_sents', color=color)  # we already handled the x-label with ax1
for i, name in enumerate(names):
    # print(line_sents[i])
    ax2.plot(iterations[i], max_sents[i], color=color, linestyle=styles[i], label=name)
ax2.tick_params(axis='y', labelcolor=color)
plt.legend()


# fig, ax1 = plt.subplots(2, 2, 2)
ax1 = axes[2]

color = 'tab:red'
ax1.set_xlabel('iterations')
ax1.set_ylabel('min rewards', color=color)

for i, name in enumerate(names):
    ax1.plot(iterations[i], min_rewards[i], color=color, linestyle=styles[i], label=name)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('min line_sents', color=color)  # we already handled the x-label with ax1
for i, name in enumerate(names):
    # print(line_sents[i])
    ax2.plot(iterations[i], min_sents[i], color=color, linestyle=styles[i], label=name)
ax2.tick_params(axis='y', labelcolor=color)
plt.legend()


fig.tight_layout()  # otherwise the right y-label is slightly clipped

plt.savefig(fig_name)