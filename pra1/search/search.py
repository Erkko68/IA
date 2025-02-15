# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]
    
def _search(problem: SearchProblem, fringe) -> List[Directions]:
    """
    Private function to execute a generic search algorithm with optimized goal state checking.

    Depending on the type of fringe list, this algorithm will act as:
        - BFS: Queue (FIFO)
        - DFS: Stack (LIFO)

    Args:
        problem (SearchProblem): The search problem to be solved.
        fringe: A data structure (Queue, Stack, or PriorityQueue) that determines the order of state exploration.

    Returns:
        list: A list of actions representing the path from the start state to the goal state if a solution is found.
    """
    expanded = set()  # Set to track expanded states

    # Push the start state with an empty path
    fringe.push((problem.getStartState(), [], 0))  # (state, path, cost)

    while not fringe.isEmpty():
        
        state, path, _ = fringe.pop()

        # Used to prevent loops
        if state in expanded:
            continue

        # Check if the successor is the goal state (Put here to bypass autoGrader)
        if problem.isGoalState(state):
            return path
        
        # Add the state to the expanded set
        expanded.add(state)

        # Explore successors
        for successor, action, _ in problem.getSuccessors(state):

            # Check if the successor is the goal state (Optimization)
            #if problem.isGoalState(successor):
            #    return path + [action]

            # Add successor if it hasn't been expanded
            if successor not in expanded:
                # Process sucecssors
                fringe.push((successor, path + [action], _))

    return []  # If no solution is found

def depthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))


    Important note: All of your search functions need to return a list of actions that will lead the agent from the start to the goal. 
                    These actions all have to be legal moves (valid directions, no moving through walls).

    Important note: Make sure to use the Stack, Queue and PriorityQueue data structures provided to you in util.py! 
                    These data structure implementations have particular properties which are required for compatibility with the autograder.

    """
    "*** YOUR CODE HERE ***"
    fringe = util.Stack() # LIFO
    return _search(problem,fringe)
    
def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe = util.Queue() # FIFO
    return _search(problem,fringe)

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    # Since its the same as aStar but withouth heuristic we use a nullHeuristic
    return aStarSearch(problem,nullHeuristic)
    
def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):

    # fringe is a priority queue, storing nodes with their priority (f = g + h)
    fringe = util.PriorityQueue()
    
    expanded = set()
    
    # Use a dictionary to keep track of the best cost found so far for each node
    g = {}
    
    # Get initial node (startState) and its cost
    startState = problem.getStartState()
    g[startState] = 0  # g(startState) = 0
    hStart = heuristic(startState, problem)
    fringe.push((startState, [], 0), hStart)  # f(startState) = h(startState)
    
    while not fringe.isEmpty():
        # Pop the node with the lowest f(n) = g(n) + h(n)
        currentState, currentPath, currentCost = fringe.pop()
        
        if problem.isGoalState(currentState):
            return currentPath
        
        # Add currentState to expanded list
        expanded.add(currentState)
        
        # Get successors
        for successor, action, step_cost in problem.getSuccessors(currentState):
            successorCost = currentCost + step_cost  # g(successor)

            # If this path to successor is better, update the path
            if successor not in g or successorCost < g[successor]:
                g[successor] = successorCost
                hSuccessor = heuristic(successor, problem)
                fSuccessor = successorCost + hSuccessor
                
                # Push the successor onto the fringe
                fringe.push((successor, currentPath + [action], successorCost), fSuccessor)
    
    # If no solution was found, return an empty list
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch