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
        some Directions.X for some X in the set {North, South, West, East, Stop}
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
        distance=float("-Inf")  
        newGhostPosition=newGhostStates[0].getPosition()

        distance_from_ghost= util.manhattanDistance(newPos,newGhostPosition)
        distance_from_food= map(lambda x: manhattanDistance(newPos,x),currentGameState.getFood().asList()) 

        if action == 'Stop':
            return float("-Inf")

        for state in newGhostStates:
            if state.getPosition() == newPos and (state.scaredTimer == 0):
                return float("-Inf")

        if distance_from_ghost>0:
          if distance<min(distance_from_food):
                distance=min(distance_from_food)      
        return  -1*distance #successorGameState.getScore() + max(distance_from_food+[0])


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
        """
        "*** YOUR CODE HERE ***"
        def minimax(agent, depth, gameState):
            minimum=float("inf") 
            maximum1=float("-inf") 
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  
                return self.evaluationFunction(gameState)
            if agent == 0: 
                for state in gameState.getLegalActions(agent):
                    val=minimax(1,depth,gameState.generateSuccessor(agent,state))
                    if maximum1<val or maximum1==float("-inf"):
                          maximum1=val
                return maximum1
            else: 
                nextAgent = agent + 1  
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                   depth += 1
                for state in gameState.getLegalActions(agent):
                    val=minimax(nextAgent,depth,gameState.generateSuccessor(agent,state))
                    if minimum>val or minimum==float("inf"):
                          minimum=val
                return minimum


        """Performing maximize action for the root node i.e. pacman"""
        maximum = float("-inf")
        for agentState in gameState.getLegalActions(0):
            fn_value = minimax(1, 0, gameState.generateSuccessor(0, agentState))
            if fn_value > maximum or maximum == float("-inf"):
                maximum = fn_value
                action = agentState

        return action   

'''References:
        https://tonypoer.io/2016/10/28/implementing-minimax-and-alpha-beta-pruning-using-python/
        https://github.com/iamjagdeesh/Artificial-Intelligence-Pac-Man/blob/master/Project%202%20Multi-Agent%20Pacman/multiAgents.py
'''

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alphaBeta(agent,depth,gameState,a,b):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  
                return self.evaluationFunction(gameState)
            maximum1=float("-inf")
            minimum=float("inf")
            if agent==0:
                for state in gameState.getLegalActions(agent):
                    val=alphaBeta(1,depth,gameState.generateSuccessor(agent,state),a,b)
                    if maximum1<val or maximum1==float("-inf"):
                          maximum1=val
                    a=max(a,maximum1)
                    if a>b:
                          break
                return maximum1
            else:
                nextAgent = agent + 1  
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                   depth += 1
                for state in gameState.getLegalActions(agent):
                    val=alphaBeta(nextAgent,depth,gameState.generateSuccessor(agent,state),a,b)
                    if minimum>val or minimum==float("inf"):
                          minimum=val
                    b=min(b,minimum)
                    if a>b:
                          break
                return minimum
        maximum = float("-inf")
        a = float("-inf")
        b = float("inf")
        for agentState in gameState.getLegalActions(0):
            fn_value = alphaBeta(1, 0, gameState.generateSuccessor(0, agentState),a,b)
            if fn_value > maximum or maximum == float("-inf"):
                maximum = fn_value
                action = agentState
            a = max(a, maximum)
        
        return action

        util.raiseNotDefined()

'''
References: https://en.wikipedia.org/wiki/Alpha%E2%80%93beta_pruning
'''

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
        "*** YOUR CODE HERE ***"
        def expectimax(agent, depth, gameState):
            x=0
            minimum=float("inf") 
            maximum1=float("-inf") 
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  
                return self.evaluationFunction(gameState)
            if agent == 0: 
                for state in gameState.getLegalActions(agent):
                    val=expectimax(1,depth,gameState.generateSuccessor(agent,state))
                    if maximum1<val or maximum1==float("-inf"):
                          maximum1=val
                return maximum1
            else: 
                nextAgent = agent + 1  
                if gameState.getNumAgents() == nextAgent:
                    nextAgent = 0
                if nextAgent == 0:
                   depth += 1
                for state in gameState.getLegalActions(agent):
                    val=expectimax(nextAgent,depth,gameState.generateSuccessor(agent,state))
                    x+=val
                x=float(x/len(gameState.getLegalActions(agent)))
                return x


        """Performing maximize action for the root node i.e. pacman"""
        maximum = float("-inf")
        for agentState in gameState.getLegalActions(0):
            fn_value = expectimax(1, 0, gameState.generateSuccessor(0, agentState))
            if fn_value > maximum or maximum == float("-inf"):
                maximum = fn_value
                action = agentState

        return action
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacPos=currentGameState.getPacmanPosition()
    #foodPos=currentGameState.getFood().asList()
    
    distance=float("-Inf")  

    #distance_from_ghost= util.manhattanDistance(newPos,newGhostPosition)
    distance_from_food= map(lambda x: manhattanDistance(pacPos,x),currentGameState.getFood().asList()) 
    distance_to_ghost=map(lambda x: manhattanDistance(pacPos,x),currentGameState.getGhostPositions())
    capsulePos=currentGameState.getCapsules()
    for x in distance_from_food:
        if distance>=x or distance==float("-Inf"):
             distance=x
    
    #distance_from_capsule=map(lambda x: manhattanDistance(pacPos,x),capsulePos)
    #print max(distance_from_capsule)," ", distance

    return currentGameState.getScore()+(1/float(distance))

    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

