from sklearn.linear_model import LogisticRegression
import sys
import numpy as np

def unique(items):
	'''Removes duplicates from the list'''
	found = set([])
	keep = []

	for item in items:
		if item not in found:
			found.add(item)
			keep.append(item)

	return list(keep)

##
##

firstTeam = [] # list of (list of champions) in first teams for a given game
secondTeam = [] # list of (list of champions) in second teams for a given game
results = [] # list of results (1 or 2) for a given game
champions = [] # list of all champions

games = 0 # counter of total number of games in trainingdata
with open('trainingdata.txt', 'r') as f:
	for line in f:
		line = line.rstrip().split(',')

		firstTeam.append(line[:5])
		secondTeam.append(line[5:-1])

		results.append(int(line[-1]))
		champions += line[:-1]

		games += 1

champions = unique(champions) # we do not want duplicates

NUM_HEROES = len(champions) # number of heroes in our data sets

# for every champion set them new id
championsID = {}
for idx, champion in enumerate(champions):
	championsID[str(champion)] = int(idx)

# construct input and expectedOutput for logisticRegression
X = np.zeros((games, NUM_HEROES), dtype=np.int) # input
Y = np.zeros((games,), dtype=np.int) # labels
for game in range(0, games):
	for champion in firstTeam[game]:
		X[game, championsID[champion]] = 1

	for champion in secondTeam[game]:
		X[game, championsID[champion]] = -1

	Y[game] = results[game]


# train model
model = LogisticRegression(C=0.1).fit(X, Y)

# now we want to predict
# results from the input
numberOfGames = int(input())
Z = np.zeros((numberOfGames, NUM_HEROES), dtype=np.int)
for game in range(0, numberOfGames):
	line = str(input()).rstrip().split(',')

	for champion in line[:5]:
		Z[game, championsID[champion]] = 1

	for champion in line[5:]:
		Z[game, championsID[champion]] = -1

# print predicted labels
predictions = model.predict(Z)
for prediction in predictions:
	print(prediction)
