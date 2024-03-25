import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
sns.set(style="whitegrid")
df = pd.read_csv('pickup.csv')

### Create histogram ###
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='price', bins=5, color='blue', edgecolor='black')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.savefig('histogram.png')
plt.show()

### Create box plot ###
order = ["Dodge", "Ford", "GMC"]
plt.figure(figsize=(12, 6))
sns.boxplot(x='make', y='price', data=df, color='lightblue', width=0.5, order=order)
plt.xlabel('Make')
plt.ylabel('Price')
plt.savefig('boxplot.png')
plt.show()

### Create a scatter plot ###
plt.figure(figsize=(12, 6))
sns.scatterplot(x='year', y='price', hue='make', data=df, marker='o', hue_order=['Dodge','Ford','GMC'])
plt.xlabel('Year')
plt.ylabel('Price')
plt.legend(title='Make')
plt.savefig('scatterplot.png')
plt.show()


df2 = pd.read_csv('web-browsers.csv')
spend = df2[["spend"]]
sns.histplot(spend['spend'], bins=13,
             kde=False, stat='proportion', color='skyblue', log_scale=True)
plt.xscale('log')
plt.xlabel('total online spend')
plt.ylabel('density')

# Set the custom x-axis ticks
custom_xticks = [1, 10, 100, 1000, 10000, 100000]
plt.xticks(custom_xticks, [f'1e+0{int(np.log10(x))}' for x in custom_xticks])

plt.savefig('density.png')
plt.show()
