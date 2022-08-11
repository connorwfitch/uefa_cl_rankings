import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt

# Some useful functions from Prof. Pietraho
def column_vector(A,k):
  '''prints out and returns the kth column of a matrix
  indexing starts at 0'''
  v = A[:,k]
  for i in v:
    print('{0:4.4f}  '.format(i),end='')
  print()
  return v

def normalize_columns(A):
  '''divides each column by its sum to obtain an Markov matrix
  provided that no entries are negative'''
  #first check that all columns are non-zero
  sums = np.sum(A,axis = 0)
  if np.count_nonzero(sums) < len(sums):
    print('Some columns sum to zero.  Cannot normalize this matrix')
    return
  else:
    return A/sums

# Setting up our data structures

# csv reading for our teams
# this url method is ugly
# one needs to reinstate it every time in my github, but that's life
url = 'https://raw.githubusercontent.com/connorfitch/uefa_cl_rankings/master/teams_index_country.csv'
teams_df = pd.read_csv(url)

# turning the df into a dictionary
teams_dict = {k: g['index'].tolist() for k,g in teams_df.groupby('team')}
nations_dict = {k: g['team'].tolist() for k,g in teams_df.groupby('nation')}

# Want matches, outcomes, and scores matrices
matches_mat = np.zeros((len(teams_dict), len(teams_dict)))
losses_mat = np.zeros((len(teams_dict), len(teams_dict)))
scores_mat = np.zeros((len(teams_dict), len(teams_dict)))

# Some useful functions for constructing our matrices

def add_match(team1, team2, score1, score2, matches=matches_mat, 
              losses=losses_mat, scores=scores_mat):
  '''adds the result of a match to our data'''
  i = teams_dict[team1]
  j = teams_dict[team2]
  matches[i,j] += 1
  matches[j,i] += 1
  losses[i,j] += (score1 > score2)*1
  losses[j,i] += (score2 > score1)*1
  scores[i,j] += score1
  scores[j,i] += score2

def fake_invert(mat):
  '''takes the termwise multiplicative inverse of a matrix
  puts in 0 for divide by zero errors'''
  shape = mat.shape
  out = []
  for x in np.nditer(mat):
    if x == 0:
      out.append(x)
    else:
      out.append(1/x)
  out = np.array(out).reshape(shape[0],shape[1])
  return out

# Waiting to run this code for when we get some matches inputted into a csv
url = 'https://raw.githubusercontent.com/connorfitch/uefa_cl_rankings/master/games_data.csv'
games_df = pd.read_csv(url)

for index, row in games_df.iterrows():
  add_match(row['team1'], row['team2'], row['score1'], row['score2'])


# Assembling our final matrix
# Assigning our weight value 'c'
c = 0.1
# number losses + avg goals allowed
M = losses_mat + c * fake_invert(matches_mat) * scores_mat

# Setting the dampening factor
e = 0.001

# Adding the dampening factor to the matrix, then normalizing
M = M + (e*np.ones(M.shape))
M = normalize_columns(M)

# Producing a ranking
eigvalues, eigvects = np.linalg.eig(M)
v = eigvects[0]

#print(np.where(np.sum(losses_mat, axis=0)==0))
#print(np.sum(M,axis=0))

#Getting the names of the top ranked teams, take 2
v_copy = v
print("Top teams ranking:")

for i in range(20):
  find_index = np.where(v==max(v_copy))[0][0]
  #print(find_index, max(v_copy))

  for club, index in teams_dict.items():
    if index==find_index:
      print((i+1), club)

  v_copy = v_copy[v_copy != max(v_copy)]


# getting a set of our nations represented
nations_set = set(nations_dict.keys())
print(nations_set, '\n',len(nations_set))


# METROPOLIS HASTINGS ZONE
# Metropolis-Hastings algorithm for finding optimal c and epsilon

# Resetting our matrices
matches_mat = np.zeros((len(teams_dict), len(teams_dict)))
losses_mat = np.zeros((len(teams_dict), len(teams_dict)))
scores_mat = np.zeros((len(teams_dict), len(teams_dict)))

# idk why but the does doesn't run properly unless I redefine this function
# yikes...lol
def add_match(team1, team2, score1, score2, matches=matches_mat, 
              losses=losses_mat, scores=scores_mat):
  '''adds the result of a match to our data'''
  i = teams_dict[team1]
  j = teams_dict[team2]
  matches[i,j] += 1
  matches[j,i] += 1
  losses[i,j] += (score1 > score2)*1
  losses[j,i] += (score2 > score1)*1
  scores[i,j] += score1
  scores[j,i] += score2

# Adding our data

removed_games = list()
for index, row in games_df.iterrows():
  if index % 20 == 0 and index > 125:
    # Want to remove every 10th game
    # Also want to keep the first 125 games since they are the UEFA games
    if row['score1'] == row['score2']:
      # Can't use ties for our predictive model so adding those back in
      add_match(row['team1'], row['team2'], row['score1'], row['score2'])
    elif row['score1'] > row['score2']:
      # Store the winner first
      removed_games.append([teams_dict[row['team1']],teams_dict[row['team2']]])
    else:
      # Store the winner first
      removed_games.append([teams_dict[row['team2']],teams_dict[row['team1']]])
  else:
    add_match(row['team1'], row['team2'], row['score1'], row['score2'])

print('Length of removed games:\n', len(removed_games))

from scipy import stats
# Initializing our Metropolis-Hastings algorithm
iterations = 10000

N = len(teams_dict)

cs = np.zeros((iterations,1))
es = np.zeros((iterations,1))
llks = np.ones((iterations,1))
lprs = np.ones((iterations,1))
props = np.ones((iterations,2))

# both variables have prior beta(1/2,1/2), initiating at their expected value
cs[0] = 0.5
es[0] = 0.5
props[0,:] = [cs[0],es[0]]

M = normalize_columns(losses_mat + cs[0]*fake_invert(matches_mat)*scores_mat + 
                        es[0]*np.ones(losses_mat.shape))
eigvalues, eigvects = np.linalg.eig(M)
v = eigvects[:,0].real

# calculating log likelihood of data given parameters
for game in removed_games:
  llks[0] *= v[game[0]]/(v[game[0]] + v[game[1]])
llks[0] = np.log(llks[0])

# calculating log prior of our parameters
prior = stats.beta(0.5,0.5)
lprs[0] = np.log(prior.pdf(cs[0]) * prior.pdf(es[0]))

# Actually running our metropolois hastings algorithm
for i in range(1,iterations):
  if i % 200 == 0:
    print(i)
  if i % 2 == 0:
    # Propose new c from a beta distribution centered at old c
    cs[i] = stats.beta(1,1).rvs(1)
    es[i] = es[i-1]
  else:
    # Propose new e from a beta distribution centered at old e
    cs[i] = cs[i-1]
    es[i] = stats.beta(1,1).rvs(1)
  props[i,:] = [cs[i],es[i]]
  # now calculate the log likelihood
  M = normalize_columns(losses_mat + cs[i]*fake_invert(matches_mat)*scores_mat + 
                        es[i]*np.ones(losses_mat.shape))
  eigvalues, eigvects = np.linalg.eig(M)
  v = eigvects[:,0].real
  for game in removed_games:
    llks[i] *= v[game[0]]/(v[game[0]] + v[game[1]])
  llks[i] = np.log(llks[i])
  lprs[i] = np.log(prior.pdf(cs[i]) * prior.pdf(es[i]))

  # getting our (log) metropolis-hastings ratio
  alpha = llks[i] - llks[i-1] + lprs[i] - lprs[i-1]
  if alpha < stats.uniform(0,1).rvs(1):
    cs[i] = cs[i-1]
    es[i] = es[i-1]
    llks[i] = llks[i-1]
    lprs[i] = lprs[i-1]

# Getting a heatmap of our results
# Don't forget to include initial burn off 
plt.plot(cs, es, 'o', color='gray', alpha=0.3)
axes = plt.gca()
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
plt.xlabel('c')
plt.ylabel('e')
plt.show()
print('\n')
# Looking at proposals to see if we're exploring the space properly
plt.plot(props[:,0], props[:,1], 'o', color='red', alpha=0.3)
axes = plt.gca()
axes.set_xlim([0, 1])
axes.set_ylim([0, 1])
plt.xlabel('c')
plt.ylabel('e')
plt.show()
print('\n')