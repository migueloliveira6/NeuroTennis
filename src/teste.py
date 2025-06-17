import csv
import math
import pickle
import pandas
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from pandas import DataFrame
from pandas import Series


#define k factor assumptions
def k_factor(matches_played):
	K = 250
	offset = 5
	shape = 0.4
	return K/(matches_played + offset)**shape

#winning a match regardless the number of sets played = 1
score = 1

#define a function for calculating the expected score of player_A
#expected score of player_B = 1 - expected score of player
def calc_exp_score(playerA_rating, playerB_rating):
	exp_score = 1/(1+(10**((playerB_rating - playerA_rating)/400)))
	return exp_score
	
#define a function for calculating new elo
def update_elo(old_elo, k, actual_score, expected_score):
	new_elo = old_elo + k *(actual_score - expected_score)	
	return new_elo

#read player CSV file and store important columns into lists
with open('D:/projetos/Tenis ML-AI/data/tennis_atp/atp_players.csv') as csvfile:
	readCSV = csv.reader(csvfile, delimiter= ',')
	col_index = [0,1,2,5]
	all_players = []
	for row in readCSV:
		player_info = []
		for i in col_index:
			player_info.append(row[i])
		all_players.append(player_info)

#Column headers for player dataframe
player_col_header = ['player_id', 'last_name', 'first_name', 'country']

#Create a dataframe for keeping track of player info
#every player starts with an elo rating of 1500
players = DataFrame(all_players[1:], columns=all_players[0])
players['current_elo'] = Series(1500, index=players.index)
players['last_tourney_date'] = Series('N/A', index=players.index)
players['matches_played'] = Series(0, index=players.index)
players['peak_elo'] = Series(1500, index=players.index)
players['peak_elo_date'] = Series('N/A', index=players.index)

#Convert objects within dataframe to numeric
for col in ['player_id', 'current_elo', 'matches_played', 'peak_elo']:
	players[col] = pandas.to_numeric(players[col], errors='ignore')

# Create a dictionary for fast player_id to DataFrame index lookup (after player_id is numeric)
player_id_to_index = {pid: idx for idx, pid in players['player_id'].items()}

#Create an empty dataframe to store time series elo for top 10 players based on peak elo rating
#Use player_id as the column header of the dataframe
#Top ten players consist of: Djokovic, Federer, McEnroe, Nadal, Borg, Lendl, Becker, Murray, Sampras, Connors 
elo_timeseries_col_header = [104925, 103819, 100581, 104745, 100437, 100656, 101414, 104918, 101948, 100284]
elo_timeseries = DataFrame(columns=elo_timeseries_col_header, index=pandas.DatetimeIndex([]))

for current_year in range(1999, 2025):
	current_year_file_name = 'D:/projetos/Tenis ML-AI/data/tennis_atp/atp_matches_'+ str(current_year) + '.csv'

	#read match CSV file and store important columns into lists
	with open(current_year_file_name) as csvfile:
		readCSV = csv.reader(csvfile, delimiter= ',')
		col_index = [0,5,7,15,24,25,26]
		all_matches = []
		for row in readCSV:
			match_info = []
			for i in col_index:
				match_info.append(row[i])
			all_matches.append(match_info)
		
	#separate column names and match info
	header_info = all_matches[0]
	all_matches = all_matches[1:]
	if not all_matches:
		continue  # Skip this year if there are no matches
	#Create a dataframe to store match info
	matches = DataFrame(all_matches, columns=header_info)
	matches = matches.dropna(subset=['winner_id', 'loser_id', 'tourney_date'])

	#Convert only numeric columns within matches dataframe to numeric
	numeric_cols = ['tourney_date', 'winner_id', 'loser_id']
	for col in numeric_cols:
		matches[col] = pandas.to_numeric(matches[col], errors='ignore')

	print(f"Processando ano: {current_year}")
	print(f"Total de partidas: {len(matches)}")
	matches = matches.apply(pandas.to_numeric, errors='ignore')
	
	#Sort matches dataframe by tourney_date and then by round
	sorter = ['RR', 'R128', 'R64', 'R32', 'R16', 'QF', 'SF', 'F']
	matches['round'] = pandas.Categorical(matches['round'], categories=sorter, ordered=True)
	matches = matches.sort_values(['tourney_date', 'round'])

	for row in matches.itertuples(index=False):
		winner_id = getattr(row, 'winner_id')
		loser_id = getattr(row, 'loser_id')
		tourney_date = getattr(row, 'tourney_date')
		index_winner = player_id_to_index.get(winner_id)
		index_loser = player_id_to_index.get(loser_id)
		if index_winner is None or index_loser is None:
			continue  # Skip this match if either player is not found
		old_elo_winner = players.loc[index_winner, 'current_elo'] 
		old_elo_loser = players.loc[index_loser, 'current_elo']
		exp_score_winner = calc_exp_score(old_elo_winner, old_elo_loser)
		exp_score_loser = 1 - exp_score_winner 
		matches_played_winner = players.loc[index_winner, 'matches_played']
		matches_played_loser = players.loc[index_loser, 'matches_played']
		winner_score = 1
		loser_score = 0
		new_elo_winner = update_elo(old_elo_winner, k_factor(matches_played_winner), winner_score, exp_score_winner)
		new_elo_loser = update_elo(old_elo_loser, k_factor(matches_played_loser), loser_score, exp_score_loser)
		players.loc[index_winner, 'current_elo'] = new_elo_winner
		players.loc[index_winner, 'last_tourney_date'] = tourney_date
		players.loc[index_winner, 'matches_played'] = players.loc[index_winner, 'matches_played'] + 1
		players.loc[index_loser, 'current_elo'] = new_elo_loser
		players.loc[index_loser, 'last_tourney_date'] = tourney_date
		players.loc[index_loser, 'matches_played'] = players.loc[index_loser, 'matches_played'] + 1
		if new_elo_winner > players.loc[index_winner, 'peak_elo']:
			players.loc[index_winner, 'peak_elo'] = new_elo_winner
			players.loc[index_winner, 'peak_elo_date'] = tourney_date
		#Convert tourney_date to a time stamp, then update elo_timeseries data frame
		try:
			tourney_date_timestamp = pandas.to_datetime(tourney_date, format='%Y%m%d')
		except Exception as e:
			print(f"Warning: Could not parse tourney_date '{tourney_date}' for match (winner_id={winner_id}, loser_id={loser_id}): {e}")
			continue  # Skip updating timeseries for this match if date is invalid

		if tourney_date_timestamp not in elo_timeseries.index:
			elo_timeseries.loc[tourney_date_timestamp, elo_timeseries_col_header] = np.nan
		
			elo_updates = [(winner_id, new_elo_winner), (loser_id, new_elo_loser)]
			for pid, new_elo in elo_updates:
				if pid in elo_timeseries_col_header:
					elo_timeseries.loc[tourney_date_timestamp, pid] = new_elo
			for pid, new_elo in elo_updates:
				if pid in elo_timeseries_col_header:
					elo_timeseries.loc[tourney_date_timestamp, pid] = new_elo

	##Uncomment to output year end elo_rankings for every year between 1968 and 2015
			
	##Uncomment to output year end elo_rankings for every year between 1968 and 2015
	#output_file_name = str(current_year) + '_yr_end_elo_ranking.csv'
	#players.to_csv(output_file_name)

	# current_year increment is no longer needed

	# current_year increment is no longer needed


players.to_csv('2024_yr_end_elo_ranking.csv')
players.dropna(subset=['last_tourney_date'], inplace=True)
players = pandas.read_csv('2024_yr_end_elo_ranking.csv')

# Carregar os dados (substitua pelo seu caminho real)
players = pandas.read_csv('2024_yr_end_elo_ranking.csv')

# Filtrar jogadores com Elo atualizado (que já jogaram pelo menos uma partida)
players_updated = players[
    (players['current_elo'] != 1500) | 
    (players['matches_played'] > 0) |
    (players['peak_elo'] != 1500)
]

# Opcional: ordenar por peak_elo para ver os melhores primeiro
players_updated = players_updated.sort_values('peak_elo', ascending=False)

# Salvar o resultado em um novo arquivo CSV
players_updated.to_csv('players_with_updated_elo.csv', index=False)
#Print all-time top 10 peak_elo
print(players_updated.sort_values(by='peak_elo', ascending=False).head(10))

#Save elo_timeseries dataframe for plotting purposes
elo_timeseries.to_pickle('elo_timeseries.pkl')

#Open saved pickle file and save into a dataframe
elo_timeseries = pandas.read_pickle('elo_timeseries.pkl')

#Convert objects within elo_timeseries dataframe to numeric
elo_timeseries = elo_timeseries.apply(pandas.to_numeric, errors='ignore')

#Use linear interpolation for elo_ratings
elo_timeseries = elo_timeseries.interpolate(method='linear')

#Store the indices in the elo_timeseries in a list
index_timestamp = list(elo_timeseries.index.values)

#Get rid of elo ratings since known last_tourney_date
for player in elo_timeseries_col_header:
	player_index = players[players['player_id'] == player].index.tolist()
	if not player_index:
		continue
	player_last_played = players.loc[player_index[0], 'last_tourney_date']
	try:
		player_last_played_timestamp = np.datetime64(pandas.to_datetime(player_last_played, format='%Y%m%d'))
		if player_last_played_timestamp in index_timestamp:
			idx = index_timestamp.index(player_last_played_timestamp)
			elo_ratings_remove = index_timestamp[idx+1:]
			for i in elo_ratings_remove:
				elo_timeseries.loc[i, player] = np.nan
	except Exception as e:
		print(f"Warning: Could not process last_tourney_date '{player_last_played}' for player {player}: {e}")

style.use('D:/projetos/Tenis ML-AI/src/stylesheet.mplstyle')
plt.plot(elo_timeseries.index, elo_timeseries[104925]) #Djokovic
plt.plot(elo_timeseries.index, elo_timeseries[103819]) #Federer
plt.plot(elo_timeseries.index, elo_timeseries[207989]) #Jannik Sinner
plt.plot(elo_timeseries.index, elo_timeseries[104745]) #Nadal
plt.plot(elo_timeseries.index, elo_timeseries[100437]) #Borg
plt.plot(elo_timeseries.index, elo_timeseries[100656]) #Lendl
plt.plot(elo_timeseries.index, elo_timeseries[101414]) #Becker
plt.plot(elo_timeseries.index, elo_timeseries[104918]) #Murray
plt.plot(elo_timeseries.index, elo_timeseries[101948]) #Sampras
plt.plot(elo_timeseries.index, elo_timeseries[100284]) #Connors

plt.title("Historical elo ratings for top 10 ATP players", fontsize=25, y=1.1, weight = 'bold')   
plt.xlabel("Years starting in the Open-Era", labelpad= 25)
plt.ylabel("Elo rating", labelpad= 32)
plt.axhline(1200, color='grey', linewidth=5)

plt.show()