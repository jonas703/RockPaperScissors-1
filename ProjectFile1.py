
RPSM1 = [['RR', 'PR', 'SR'],['RP', 'PP', 'SP'],['RS', 'PS', 'SS']]
RPS = ['R', 'P', 'S']    
WorL_matrix = [
    [0, 1, -1],
    [-1, 0, 1],
    [1, -1, 0]
]


# player 2 selection is the second letter in the element

def player2Choice(numChoices):
    valid_inputs = {'0': 0, '1': 1, '2': 2}
    choices=[]
    print(f"\n\n")
    for i in range(numChoices):
     while True:
        choice = input(f"Select row position (0, 1, or 2) to get player 2 choice #{i+1}: ")
        if choice in valid_inputs:
            row = valid_inputs[choice]
            result = RPS[row]
            print(f"Selected element: {result}")
            choices.append(result)
            break
        else:
            print("Invalid input. Please enter 0, 1, or 2.")
   
    return choices

# player 1 selection is the first letter in the element

def player1Choice(numChoices):
     valid_inputs = {'0': 0, '1': 1, '2': 2}
     choices=[]
     print(f"\n\n")
     for i in range(numChoices):
            while True:
                choice = input(f"Select col position (0, 1, or 2) to get player 1 choice #{i+1}: ")
                if choice in valid_inputs:
                    col = valid_inputs[choice]
                    result = RPS[col]
                    print(f"Selected element: {result}")
                    choices.append(result)
                    break
                else:
                    print("Invalid input. Please enter 0, 1, or 2.")
   
     return choices
# Example usage

def map_choices_to_matrix(player1_choices, player2_choices):
   global probWin, problose, probDraw, all_combinations
   probWin = 0
   problose = 0
   probDraw = 0
   all_combinations = []

   print("\nAll possible combinations:")
   for p1 in player1_choices:
        for p2 in player2_choices:
            p1_index = RPS.index(p1)
            p2_index = RPS.index(p2)
            combination = {
                'p1_move': p1,
                'p2_move': p2,
                'outcome': RPSM1[p2_index][p1_index],
                'numeric_outcome': WorL_matrix[p2_index][p1_index]
            }
            all_combinations.append(combination)
            print(f"Possible combo - Player1: {p1}, Player2: {p2} maps to: {combination['outcome']} (Score: {combination['numeric_outcome']})")
            if combination['numeric_outcome'] == 1:
                probWin += 1
            elif combination['numeric_outcome'] == -1:
                problose += 1
            else:
                probDraw += 1


   return all_combinations, probWin, problose, probDraw



# Function to analyze losing combinations for Player 1 and chose which to drop 
# Need to develop logic for differnt scenarios
# rough draft below

def minusOne(all_combinations):
    global possMinusOne
    possMinusOne = []
    possDraw = []
    possWin = []
    for combo in all_combinations:
        if combo['numeric_outcome'] == -1:
            possMinusOne.append(combo)
        elif combo['numeric_outcome'] == 0:
            possDraw.append(combo)
        else:
            possWin.append(combo)

    if len(possMinusOne) == 0:
        print("ALL ROADS LEAD TO VICTORY!")
        return
    if len(possMinusOne) == 1:
        print(f"\nDROP {possMinusOne[0]['p1_move']} from possible combinations to minimize Player 1 loss.")
        return
    else: 
      # Check if all entries are identical if there are multiple losing combinations check the losing combinations against the possible wins
        first_entry = possMinusOne[0]
        all_same = all(entry == first_entry for entry in possMinusOne)
        
        if all_same:
            print(f"\nAll losing combinations are identical, so DROP: {first_entry}")
            return
        # List unique losing p1 moves
    losing_p1_moves = list({c['p1_move'] for c in possMinusOne})
    print(f"\nMultiple different options to DROP to minimize Player 1 loss: {losing_p1_moves}")

    # Prefer a losing combo that also appears as a possible win
    match = next((lose for lose in possMinusOne if lose in possWin), None)
    if match:
        print(f"\nDROP {match['p1_move']} (losing combo also appears as a possible win).")
        return
    
    # Next, prefer a losing combo that also appears as a possible draw
    match = next((lose for lose in possMinusOne if lose in possDraw), None)
    if match:
        print(f"\nDROP {match['p1_move']} (losing combo also appears as a possible draw).")
        return

    # Otherwise recommend the most frequent losing p1_move
    from collections import Counter
    cnt = Counter(c['p1_move'] for c in possMinusOne)
    most_common_move, freq = cnt.most_common(1)[0]
    print(f"\nNo losing combinations match possible draws. Consider dropping the most frequent losing move: {most_common_move} (appears {freq} times).")


     

       
          # different losing combos -> show options and try to match any with possible draws
    # print(f"\nMultiple different options to DROP to minimize Player 1 loss: {[c['p1_move'] for c in possMinusOne]}")

    # if possDraw:
    #     # try to find a losing combo that matches a draw (prefer dropping that p1_move)
    #     matches = [lose for lose in possMinusOne if lose in possDraw]
    #     if matches:
    #         print(f"\nDROP {matches[0]['p1_move']} from possible combinations to minimize Player 1 loss.")
    #         return

    # # no matching draws, just list losing combos
    # print("No losing combinations match possible draws. Consider dropping one of the above losing options.")

        # else:
        #     print(f"\nMultiple different options to DROP to minimize Player 1 loss: {possMinusOne['p1_move']}")
        #     if len(possDraw) > 0:
        #         firstDraw = possDraw[0]
        #         if len(possDraw)>1:
        #             sencondDraw = possDraw[1]
        #         x=False
        #         while x:
        #             for lose in possMinusOne:
        #                 if lose == firstDraw:
        #                     print(f"\nDROP {lose[0]['p1_move']} from possible combinations to minimize Player 1 loss.")
        #                     x=True
        #                     break
        #                 elif lose == sencondDraw:
        #                     print(f"\nDROP {lose[0]['p1_move']} from possible combinations to minimize Player 1 loss.")
        #                     x=True
        #                     break
        #                 else:
        #                     print("No losing combinations match possible draws.")
        #                     x=True
        #                     break
                    
 

player1 = player1Choice(2)
player2 = player2Choice(2)
# Example usage after getting player choices
mapped_results = map_choices_to_matrix(player1, player2)
# print("Final mapped elements:", mapped_results)
PROBWin = probWin / len(mapped_results[0])
PROBLose = problose / len(mapped_results[0])
PROBDraw = probDraw / len(mapped_results[0])
probMatrix = [[PROBWin, PROBLose, PROBDraw]]
print(f"\nProbability Matrix: {probMatrix}")
print(f"\nProbability of Player 1 Winning: {PROBWin}")
minusOne (all_combinations)







