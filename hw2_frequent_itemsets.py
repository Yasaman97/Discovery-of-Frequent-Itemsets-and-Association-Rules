
# Importing useful libraries
import numpy as np
import pandas as pd
from itertools import combinations
import time
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Defining the path to the dataset
path = "pathtofile/T10I4D100K.dat"



def read_dataset(file):
  """Function read_dataset reads the file and outputs the baskets as a list of sets. 
  Each set represents the shopping items for each basket or transaction."""
  with open(path, "r") as f:
   baskets = list(map(lambda basket: {int(item) for item in basket.split()},f.read().splitlines()))

  return baskets

baskets = read_dataset(path)

baskets

num_baskets = len(baskets)
print(num_baskets)

"""1. Finding frequent itemsets with support at least s"""



def frequent_singletons(baskets, s):
  """Function frequent_singletons receives the list of baskets and the support threshold s, 
  and keeps the count of each item in a dictionary, itemSupport. In the end, this dictionary 
  is filtered to make another dictionary, freq_singles, which keeps each frequent item as its key, 
  and the number of times it appeared in a basket as the value. """
  itemSupport = dict()
  t0 = time.time()
  for basket in tqdm(baskets):
    for item in basket:
      if item not in itemSupport.keys():
        itemSupport[item] = 1
      else:
        itemSupport[item] += 1

  freq_singles = dict(filter(lambda element: element[1] > s, itemSupport.items()))
  print("Finding " + str(len(freq_singles)) + " frequent items from " + str(len(baskets)) + " baskets with support threshold " + str(s) + ", took %.2f sec." % (time.time() - t0))
  return freq_singles

freq_singles = frequent_singletons(baskets, num_baskets//100)

freq_singles

def Frequents(baskets, freq_singles, s):
  t0 = time.time()
  pairs = list(combinations(sorted(freq_singles), 2))

  Support = dict()
  """Function Frequents receives the baskets, frequent items, and the support threshold, 
  and calculates all possible itemsets (pairs and triplets) of them. Then, keeps the number of times 
  the itemset appeared in baskets, and returns the ones above the threshold as the frequent itemsets in a dictionary."""
  for pair in tqdm(pairs):
    for basket in baskets:
      if pair[0] in basket and pair[1] in basket:
        if pair not in Support.keys():
          Support[pair] = 1
        else:
          Support[pair] += 1
  freq_pairs = dict(filter(lambda element: element[1] > s, Support.items()))

  items = set()
  for key in freq_pairs:
    for k in key:
      items.add(k)
    
  triplets = list(combinations(sorted(items), 3))
  for trip in tqdm(triplets):
    for basket in baskets:
      if trip[0] in basket and trip[1] in basket and trip[2] in basket:
        if trip not in Support.keys():
          Support[trip] = 1
        else:
          Support[trip] += 1

  frequents = dict(filter(lambda element: element[1] > s, Support.items()))
  print("Finding " + str(len(frequents)) + " frequent itemsets from " + str(len(freq_singles)) + " frequent items took %.2f sec." % (time.time() - t0))
  return frequents

freq_itemsets = Frequents(baskets, freq_singles, num_baskets//100)
freq_itemsets  

"""The first part for finding frequent itemsets through A-Priori algorithm is done.
Now, for the second part, association rules are generated

2. Generating association rules with interest at least c"""

def Generating_asociation_rules(frequents, c= 0.5):
  """Function Generating_association_rules receives the frequent itemsets and the confidence threshold c. 
  The output is a dictionary that maps the rule to the confidence of the rule. """
  association_rules = dict()
  t0 = time.time()
  for freq in frequents:
    if type(freq)== int:
        t=[]
        t.append(freq)
        temp=t
    else:
      temp = list(freq)
    for length in range(1, len(temp)):
      for comb in [list(combination) for combination in combinations(freq, length)]:
        if len(comb)==1:
            confidence = frequents[freq] / frequents[comb[0]]
        else :
            confidence = frequents[freq] / frequents[tuple(comb)]
        if confidence >= c:
            consequent = list(set(temp)-set(comb))
            association_rules[(tuple(comb), tuple(consequent))] = confidence
            


  print("Finding " + str(len(association_rules)) + " association rules with interest threshold " + str(c) +  ", took %.2f sec." % (time.time() - t0))
  return association_rules

frequents = freq_singles.copy()
frequents.update(freq_itemsets)
rules = Generating_asociation_rules(frequents, 0.5)

rules

"""3. Evaluation

Evaluating the process of finding frequent items with different support thresholds
"""

supports = [700, 800, 900, 1000, 1100, 1200, 1300]
durations = []
n_freq_items = []

for s in supports:
  t0 = time.time()
  freq_items = frequent_singletons(baskets, s)
  duration = time.time() - t0
  
  durations.append(duration)
  n_freq_items.append(len(freq_items))

"""Plotting the the execution time for different support thresholds"""

plt.plot(supports, durations)
plt.xlabel('Support threshold')
plt.ylabel('Execution time (s)')
plt.title('Execution time vs. Support threshold')
plt.show()

"""Plotting the number of frequent items found for different support thresholds"""

plt.plot(supports, n_freq_items)
plt.xlabel('Support threshold')
plt.ylabel('Number of frequent items')
plt.title('Number of frequent items vs. Support threshold')
plt.show()

"""For this part, the frequent items and frequent itemsets found with support threshold of 1000 or number of baskets/100, have been given to the function, and different confidence thresholds are tested."""

confidences = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
durations = []
n_rules = []

for c in confidences:
  t0 = time.time()
  rules = Generating_asociation_rules(frequents, c)
  duration = time.time() - t0
  
  durations.append(duration)
  n_rules.append(len(rules))

"""Plotting the the execution time for different confidence thresholds"""

plt.plot(confidences, durations)
plt.xlabel('Confidence threshold')
plt.ylabel('Execution time (s)')
plt.title('Execution time vs. Confidence threshold')
plt.show()

"""Plotting the number of association rules found for different confidence/ thresholds"""

plt.plot(confidences, n_rules)
plt.xlabel('Confidence threshold')
plt.ylabel('Number of rules')
plt.title('Number of rules vs. Confidence threshold')
plt.show()