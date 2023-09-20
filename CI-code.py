
#Sample Code for line chart


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


#Sample Code for boxplot



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(3.5, 2.4))

data = pd.DataFrame({
    'MCSM': [0.09321181,0.09969275,0.10918288,0.1372484,0.1632106,0.20420827,0.23462534,0.3456679,0.37589992,0.41393599,0.43286338,0.54422208,0.55517286,0.57365243,0.652452,0.67511702,0.69274451,0.69741647,0.70675777,0.71962455,0.7657866,0.81990511,0.8224048,0.84876784,0.86445131,0.91099811,0.92751382,0.93908045,0.98629411,0.9890413,0.99124731],
    'DQN': [0.00420827,0.03462534,0.1456679,0.17589992,0.21393599,0.23286338,0.34422208,0.35517286,0.37365243,0.452452,0.47511702,0.49274451,0.49741647,0.50675777,0.51962455,0.5657866,0.61990511,0.6224048,0.64876784,0.66445131,0.71099811,0.72751382,0.73908045,0.78629411,0.7890413,0.79124731,0.80963672,0.81048874,0.83243348,0.97885783,0.99650891],
    'MADQN': [0.0856679,0.11589992,0.15393599,0.17286338,0.28422208,0.29517286,0.31365243,0.392452,0.41511702,0.43274451,0.43741647,0.44675777,0.45962455,0.5057866,0.55990511,0.5624048,0.58876784,0.60445131,0.65099811,0.66751382,0.67908045,0.72629411,0.7290413,0.73124731,0.74963672,0.75048874,0.77243348,0.91885783,0.93650891,0.96124511,0.99169209],
    'DDQN': [0.0056679,0.03589992,0.07393599,0.09286338,0.20422208,0.21517286,0.23365243,0.312452,0.33511702,0.35274451,0.35741647,0.36675777,0.37962455,0.4257866,0.47990511,0.4824048,0.50876784,0.52445131,0.57099811,0.58751382,0.59908045,0.64629411,0.6490413,0.65124731,0.66963672,0.67048874,0.69243348,0.83885783,0.85650891,0.88124511,0.91169209],
    'EVRELAY': [0.01393599,0.03286338,0.14422208,0.15517286,0.17365243,0.252452,0.27511702,0.29274451,0.29741647,0.30675777,0.31962455,0.3657866,0.41990511,0.4224048,0.44876784,0.46445131,0.51099811,0.52751382,0.53908045,0.58629411,0.5890413,0.59124731,0.60963672,0.61048874,0.63243348,0.77885783,0.79650891,0.82124511,0.85169209,0.95006715,0.98482563]
})
# You can use the column headings from the data frame as labels
labels = list(data)
# A data frame needs to be converted into an array before it can be plotted this way
data = np.array(data)

ax = plt.axes()
bp = ax.boxplot(data, labels=labels)
ax.set_ylabel('Avg. energy consumption')
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
    
plt.grid(alpha=.6)
plt.ylim([0, 1])
plt.tight_layout()
plt.xlabel('(Number of ABSs = 15)')
plt.savefig("TVT_Service_coverage_4.pdf")
plt.xticks(fontsize=9)
plt.show()
