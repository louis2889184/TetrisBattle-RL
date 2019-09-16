import os 
import matplotlib
import re

a2c_rewards = []
a2c_line_sent = []

a2c_iterations = []

_type = 'grid'
a2c_name = 'tetris_%s_a2c.txt' % _type
ppo_name = 'tetris_%s_ppo.txt' % _type

figure_name = '_type.png'
with open(a2c_name) as f:
    lines = f.readlines()
    for line in lines:
        line = re.split("[ ,/]+", line)
        # print(line)
        if (len(line) > 5):
            if (line[0] == "Updates"):
                a2c_iterations.append(int(line[4]))
            if (line[1] == "Last"):
                a2c_rewards.append(float(line[8]))
            if (line[1] == "lines"):
                a2c_line_sent.append(float(line[6]))

print(len(a2c_iterations), len(a2c_rewards), len(a2c_line_sent))


ppo_rewards = []
ppo_line_sent = []

ppo_iterations = []

figure_name = '_type.png'
with open(ppo_name) as f:
    lines = f.readlines()
    for line in lines:
        line = re.split("[ ,/]+", line)
        # print(line)
        if (len(line) > 5):
            if (line[0] == "Updates"):
                ppo_iterations.append(int(line[4]))
            if (line[1] == "Last"):
                ppo_rewards.append(float(line[8]))
            if (line[1] == "lines"):
                ppo_line_sent.append(float(line[6]))

print(len(ppo_iterations), len(ppo_rewards), len(ppo_line_sent))


import numpy as np
import matplotlib.pyplot as plt

# Create some mock data
t = np.arange(0.01, 10.0, 0.01)
data1 = np.exp(t)
data2 = np.sin(2 * np.pi * t)

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('iterations')
ax1.set_ylabel('rewards', color=color)
ax1.plot(a2c_iterations, a2c_rewards, color=color)
ax1.plot(ppo_iterations, ppo_rewards, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
ax2.plot(t, data2, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()