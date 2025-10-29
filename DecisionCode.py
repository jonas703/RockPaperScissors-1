from ProjectFile1 import map_choices_to_matrix, probWin, problose, probDraw, player2Choice, all_combinations

RPSM1 = [['RR', 'PR', 'SR'],['RP', 'PP', 'SP'],['RS', 'PS', 'SS']]
RPS = ['R', 'P', 'S']    
WorL_matrix = [
    [0, 1, -1],
    [-1, 0, 1],
    [1, -1, 0]
]

# I know there is some duplocaiton from the other file, I had it imported but was having issues with all the print calls within functions

'''

def decision_function(possibilities):
    posLoss = []
    posWin = []
    posDraw = []
    for outcome in possibilities:
        if outcome[2] == 1:
            posWin.append(outcome)
        elif outcome[2] == -1:
            posLoss.append(outcome)
        else:
            posDraw.append(outcome)
    print(posDraw, posLoss, posWin)

    return


p1Choices = player1Choice(2)
p2Choices = player2Choice(2)
# mapping all possible combinations with given choices
possibleCombinations = map_choices_to_matrix(p1Choices, p2Choices)

probWin = probWin / len(possibleCombinations)
probLose = problose / len(possibleCombinations)
probDraw = probDraw / len(possibleCombinations)
decision_function(all_combinations)
'''