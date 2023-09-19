
#Sample Code for Service Coverage vs ABS density 


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
fig = plt.figure(figsize=(3.5, 2.4))

x = []
y = []
with open('EVRELAY-ABS-NUMBER.txt','r') as f:
            lines = f.readlines();
            for row in lines:
                rw = row.split(',')
                x.append(float(rw[0]))
                y.append(float(rw[1])) 

confidence_level = 0.95 # 95% confidential interval

sample_mean = np.mean(y)
sample_std = np.std(y, ddof=1)
sample_size = len(y)

#Student T test
t_critical = stats.t.ppf((1 + confidence_level) / 2, df=sample_size - 1)
margin_of_error = t_critical * (sample_std / np.sqrt(sample_size))
# Make the plot
plt.plot(x, y, marker='x', linestyle='-', color='green', label='EVRELAY')
plt.errorbar(x, EVRELAY, yerr=margin_of_error, marker='o', linestyle='-', color='green', capsize=5)

plt.annotate('Upper bound', xy=(10, 21), xytext=(9, 23.5),arrowprops=dict(facecolor='black', arrowstyle='-|>'), fontsize=8.5)
plt.annotate('Lower bound', xy=(10, 19), xytext=(10.5, 11.5),arrowprops=dict(facecolor='black', arrowstyle='-|>'), fontsize=8.5)


plt.xlabel('Number of ABSs')
plt.ylabel('Avg. service coverage')
plt.grid(alpha=.6)
plt.legend(loc="upper left")
plt.tight_layout()
plt.savefig("TVT_Service_coverage_1.pdf")
plt.show()


#Sample Code for Average Service Coverage



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(3.5, 2.4))

MCSM = OpenFile('MCSM-ABV-COVERAGE.txt'); 
DQN = OpenFile('DQN-ABV-COVERAGE.txt'); 
MADQN = OpenFile('MADQN-ABV-COVERAGE.txt'); 
DDQN = OpenFile('DDQN-ABV-COVERAGE.txt'); 
EVRELAY = OpenFile('EVRELAY-ABV-COVERAGE.txt'); 

data = pd.DataFrame({
    'MCSM': MCSM,
    'DQN': DQN,
    'MADQN': MADQN,
    'DDQN': DDQN,
    'EVRELAY': EVRELAY
})

labels = list(data)
data = np.array(data)

ax = plt.axes()
bp = ax.boxplot(data, labels=labels)
ax.set_ylabel('Avg. service coverage')
#ax.set_xlabel('Number of ABSs')

# Colour the outlines
colours = ['red', 'orange', 'blue', 'magenta', 'green']
for i, box in enumerate(bp['boxes']):
    # Iterate over the colours
    j = i % len(colours)
    # Set the colour of the boxes' outlines
    plt.setp(bp['boxes'][i], color=colours[j])
    # Set the colour of the median lines
    plt.setp(bp['medians'][i], color=colours[j])
    # Set the colour of the lower whiskers
    plt.setp(bp['whiskers'][2 * i], color=colours[j])
    # Set the colour of the upper whiskers
    plt.setp(bp['whiskers'][2 * i + 1], color=colours[j])
    # Set the colour of the lower caps
    plt.setp(bp['caps'][2 * i], color=colours[j])
    # Set the colour of the upper caps
    plt.setp(bp['caps'][2 * i + 1], color=colours[j])
# Fill the boxes with colours (requires patch_artist=True)

plt.annotate('First Quartile', xy=(2.8, 16.9), xytext=(1, 18),arrowprops=dict(facecolor='black', arrowstyle='-|>'), fontsize=8.5)
plt.annotate('Third Quartile', xy=(3.2, 16), xytext=(3.6, 15),arrowprops=dict(facecolor='black', arrowstyle='-|>'), fontsize=8.5)
plt.annotate('Maximum', xy=(3, 18), xytext=(2.2, 20),arrowprops=dict(facecolor='black', arrowstyle='-|>'), fontsize=8.5)

plt.annotate('Minimum', xy=(3, 15), xytext=(3, 13),arrowprops=dict(facecolor='black', arrowstyle='-|>'), fontsize=8.5)

plt.annotate('Median', xy=(3, 16.5), xytext=(2, 12.5),arrowprops=dict(facecolor='black', arrowstyle='-|>'), fontsize=8.5)
    
plt.grid(alpha=.6)
plt.tight_layout()
plt.xlabel('(Number of ABSs = 10)')
plt.savefig("TVT_Service_coverage_2.pdf")
plt.show()