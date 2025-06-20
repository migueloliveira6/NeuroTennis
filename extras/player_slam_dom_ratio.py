# -*- coding: utf-8 -*-
import altair as alt
import pandas as pd
from difflib import get_close_matches
pd.options.mode.chained_assignment = None  # default='warn'

## path to repo with relevant data

data_prefix = 'D:/projetos/Tenis ML-AI/data/tennis_atp/'

# Carrega uma lista de jogadores disponíveis a partir dos dados

# Busca todos os arquivos de partidas
all_players = set()
for y in range(2000, 2025):
    try:
        matches = pd.read_csv(data_prefix + f'atp_matches_{y}.csv', usecols=['winner_name', 'loser_name'])
        all_players.update(matches['winner_name'].dropna().unique())
        all_players.update(matches['loser_name'].dropna().unique())
    except FileNotFoundError:
        continue

def find_closest_player(name, players):
    matches = get_close_matches(name, players, n=1, cutoff=0.6)
    return matches[0] if matches else None

while True:
    player = input("Nome do Jogador: ").strip()
    if player in all_players:
        break
    suggestion = find_closest_player(player, all_players)
    if suggestion:
        resp = input(f"Jogador não encontrado. Você quis dizer '{suggestion}'? (s/n): ").strip().lower()
        if resp == 's':
            player = suggestion
            break
    print("Jogador não encontrado. Tente novamente.")

player_slams = []
for y in range(2000, 2025):  # agora inclui até 2025
    try:
        matches = pd.read_csv(data_prefix + 'atp_matches_' + str(y) + '.csv')
    except FileNotFoundError:
        print(f"Arquivo para o ano {y} não encontrado, pulando...")
        continue
    pmatches = matches.loc[(matches['winner_name'] == player) | (matches['loser_name'] == player)]
    first_rounds = ['R128', 'R64', 'R32', 'R16']
    tmatches = pmatches.loc[(pmatches['tourney_level'] == 'G') & (pmatches['round'].isin(first_rounds))]
    
    tmatches['pSvPt'] = tmatches.apply(lambda row: row.w_svpt 
                                       if row.winner_name == player else row.l_svpt, axis=1)
    tmatches['pSPW'] = tmatches.apply(lambda row: row['w_1stWon'] + row['w_2ndWon']
                                       if row.winner_name == player else row['l_1stWon'] + row['l_2ndWon'], axis=1)
    tmatches['pRetPt'] = tmatches.apply(lambda row: row.l_svpt 
                                       if row.winner_name == player else row.w_svpt, axis=1)
    tmatches['pRPW'] = tmatches.apply(lambda row: (row['l_svpt'] - row['l_1stWon'] - row['l_2ndWon'])
                                       if row.winner_name == player 
                                       else (row['w_svpt'] - row['w_1stWon'] - row['w_2ndWon']), axis=1)
    print('Processing year:', y, 'with', len(tmatches), 'matches for', player)
    
    slams = set(tmatches['tourney_id'].tolist())

    ## This code analyzes a tennis player's performance in Grand Slam tournaments from 2000 to 2023. For each year, it 
    # loads the match data from a CSV file and filters the matches to include only those where the target player (specified by the variable `player`) 
    # was either the winner or the loser. It further narrows down the matches to only those played in the first four rounds of Grand Slam events, 
    # identified by the round codes `'R128'`, `'R64'`, `'R32'`, and `'R16'`, and where the tournament level is `'G'` (Grand Slam).

    ##For each filtered match, the code calculates several statistics relevant to the player's dominance ratio (DR), 
    # which is typically defined as the ratio of return points won (RPW) to service points lost (SPL). It creates new columns in the DataFrame:
    ##- `pSvPt`: The number of service points played by the player in the match.
    ##- `pSPW`: The number of service points won by the player (sum of first and second serve points won).
    ##- `pRetPt`: The number of return points played by the player (i.e., the opponent's service points).
    ##- `pRPW`: The number of return points won by the player (opponent's service points minus opponent's first and second serve points won).

    ##These columns are computed using `apply` with a lambda function, which checks if the player was the winner or loser in each match and 
    # selects the appropriate columns accordingly.

    ##After processing, the code prints a summary for the year, showing how many relevant matches were found for the player. Finally, 
    # it creates a set of unique Grand Slam tournament IDs (`slams`) that the player participated in during that year, which can be useful 
    # for further analysis or aggregation.

    for slam in slams:
        ## check if player won this tournament (not [yet] using in this viz)
        titles = pmatches.loc[(pmatches['winner_name'] == player) & (pmatches['round'] == 'F') & (pmatches['tourney_id'] == slam)]
        won_tourney = 1 if len(titles) == 1 else 0
        ## get matches from this tournament and calculate aggregate DR
        smatches = tmatches.loc[tmatches['tourney_id'] == slam]
        numeric_cols = ['pSvPt', 'pSPW', 'pRetPt', 'pRPW']
        slam_total = smatches[numeric_cols].sum(axis=0)
        rpw = slam_total['pRPW'] / slam_total['pRetPt']
        spw = slam_total['pSPW'] / slam_total['pSvPt']
        denominator = 1 - spw
        if denominator == 0:
            dr = float('nan')
        else:
            dr = rpw / denominator
        row = [smatches.tail(1)['tourney_name'].item(), smatches.tail(1)['tourney_date'].item(), dr, won_tourney] 
        player_slams.append(row)                                                             
        if slam_total['pSvPt'].item() == 0:
            spw = float('nan')
        else:
            spw = slam_total['pSPW'].item() / slam_total['pSvPt'].item()
        if pd.isna(rpw) or pd.isna(spw) or (1 - spw) == 0:
            dr = float('nan')
        else:
            dr = rpw / (1 - spw)
        row = [smatches.tail(1)['tourney_name'].item(), smatches.tail(1)['tourney_date'].item(), dr, won_tourney] 
        player_slams.append(row)                                                             

## sort slams ascending by date
player_slams = sorted(player_slams, key=lambda x: x[1])
    
df = pd.DataFrame(player_slams, columns=['Tourney', 'Date', 'DR', 'Title'])

df.astype({'DR': 'float',
           'Title': 'int'
                }).dtypes

slam_abvs = {'Wimbledon': 'Wimb',
             'US Open': 'USO',
             'Us Open': 'USO',
             'Australian Open': 'AO',
             'Roland Garros': 'RG'
             }
df['FullName'] = df.apply(lambda row: str(row['Date'])[:4] + ' ' + slam_abvs[row['Tourney']], axis=1)

## store list in *date* order for the chart to use:
x_sort = df['FullName'].tolist()

## subset of the data with only tournaments where he won the title, for second layer
titles = df.loc[df['Title'] == 1]

## line chart with all tournaments
line = alt.Chart(df).mark_line(point=True).encode(
    alt.X('FullName',
          sort=x_sort,
          axis=alt.Axis(title='Tournament (first four rounds)')),
    alt.Y('DR',
          axis=alt.Axis(title='Dominance Ratio'),
          scale=alt.Scale(domain=(0.7,2.5))),
)
    
## mark larger, different-colored points for tournaments that he won
points = alt.Chart(titles).mark_point(filled=True, size=200, color='orange').encode(
    alt.X('FullName',
          sort=x_sort),
    alt.Y('DR'),
    alt.Tooltip(['Tourney', 'Date', 'DR'])
).properties(
    title="Pontos laranja indicam torneios Grand Slam vencidos pelo jogador"
)
print('Saving chart for', player, 'with', len(df), 'tournaments')
(line + points).save(f'output/{player}_slam_dr.html')