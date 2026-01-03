# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFood = successorGameState.getFood().asList()
        minFoodDistance= float("inf")
        for food in newFood:
            minFoodDistance = min(minFoodDistance, manhattanDistance(newPos, food))

        for ghostPosition in successorGameState.getGhostPositions():
            if (manhattanDistance(newPos, ghostPosition) < 2):
                return -float('inf')

        return successorGameState.getScore() + 1.0/minFoodDistance

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState, 0, 0)[0]

    def minimaxValue(self, gameState, index, depth):
        if depth is gameState.getNumAgents() * self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if index == 0:
            return self.maxValue(gameState, index, depth)[1]
        else:
            return self.minValue(gameState, index, depth)[1]
        
    def minValue(self, gameState, index, depth):
        bestMove = ("min",float("inf"))
        
        for action in gameState.getLegalActions(index):
            successorAction = (action,self.minimaxValue(gameState.generateSuccessor(index,action),(depth + 1)%gameState.getNumAgents(),depth+1))
            bestMove = min(bestMove, successorAction, key=lambda x:x[1])
        
        return bestMove

    def maxValue(self, gameState, index, depth):
        bestMove = ("max",-float("inf"))
        
        for action in gameState.getLegalActions(index):
            successorAction = (action,self.minimaxValue(gameState.generateSuccessor(index,action),(depth + 1)%gameState.getNumAgents(),depth+1))
            bestMove = max(bestMove, successorAction, key=lambda x:x[1])
        
        return bestMove

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.maxValue(gameState, 0, 0, -float("inf"), float("inf"))[0]
    
    def alphaBetaValue(self, gameState, index, depth, alpha, beta):
        if depth is gameState.getNumAgents() * self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        if index == 0:
            return self.maxValue(gameState, index, depth, alpha, beta)[1]
        else:
            return self.minValue(gameState, index, depth, alpha, beta)[1]
        
    def minValue(self, gameState, index, depth, alpha, beta):
        bestMove = ("min",float("inf"))
        
        for action in gameState.getLegalActions(index):
            successorAction = (action,self.alphaBetaValue(gameState.generateSuccessor(index,action),(depth + 1)%gameState.getNumAgents(),depth+1, alpha, beta))
            bestMove = min(bestMove, successorAction, key=lambda x:x[1])

            if bestMove[1] < alpha:
                return bestMove
            else:
                beta = min(beta,bestMove[1])
        
        return bestMove

    def maxValue(self, gameState, index, depth, alpha, beta):
        bestMove = ("max",-float("inf"))
        
        for action in gameState.getLegalActions(index):
            successorAction = (action,self.alphaBetaValue(gameState.generateSuccessor(index,action),(depth + 1)%gameState.getNumAgents(),depth+1,alpha,beta))
            bestMove = max(bestMove, successorAction, key=lambda x:x[1])

            if bestMove[1] > beta:
                return bestMove
            else:
                alpha = max(alpha,bestMove[1])

        return bestMove

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        action, score = self.getValue(gameState, 0, 0)

        return action

    def getValue(self, gameState, index, depth):
    
        if len(gameState.getLegalActions(index)) == 0 or depth == self.depth:
            return "", self.evaluationFunction(gameState)

        if index == 0:
            return self.maxValue(gameState, index, depth)

        else:
            return self.expectedValue(gameState, index, depth)

    def maxValue(self, gameState, index, depth):
        
        legalMoves = gameState.getLegalActions(index)
        maxValue = float("-inf")
        maxAction = ""

        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action)
            successorIndex = index + 1
            successorDepth = depth

            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            cuurentMove, current_value = self.getValue(successor, successorIndex, successorDepth)

            if current_value > maxValue:
                maxValue = current_value
                maxAction = action

        return maxAction, maxValue

    def expectedValue(self, gameState, index, depth):
        
        legalMoves = gameState.getLegalActions(index)
        expectedValue = 0
        expectedMove = ""

        successor_probability = 1.0 / len(legalMoves)

        for action in legalMoves:
            successor = gameState.generateSuccessor(index, action)
            successorIndex = index + 1
            successorDepth = depth

            if successorIndex == gameState.getNumAgents():
                successorIndex = 0
                successorDepth += 1

            currentMove, current_value = self.getValue(successor, successorIndex, successorDepth)

            expectedValue += successor_probability * current_value

        return expectedMove, expectedValue

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    pacmanLocation = currentGameState.getPacmanPosition()
    ghostLocations = currentGameState.getGhostPositions()
    foodList = currentGameState.getFood().asList()
    foodCount = len(foodList)
    capsuleCount = len(currentGameState.getCapsules())
    nearestFood = 1
    score = currentGameState.getScore()

    food_distances = [manhattanDistance(pacmanLocation, foodLocation) for foodLocation in foodList]

    if foodCount > 0:
        nearestFood = min(food_distances)

    for ghostLocation in ghostLocations:
        ghostDistance = manhattanDistance(pacmanLocation, ghostLocation)
        if ghostDistance < 2:
            nearestFood = 99999

    features = [1.0 / nearestFood,score,foodCount,capsuleCount]
    weights = [10,200,-100,-10]

    return sum([feature * weight for feature, weight in zip(features, weights)])

better = betterEvaluationFunction
