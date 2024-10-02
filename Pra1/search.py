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
    
def _firstSearch(problem: SearchProblem, fringe):
    """
    Private function to execute a generic search algorithm.

    Depending on the type of fringe list, this algorithm will act as:
        - BFS: Queue (FIFO) - This approach explores all neighbors at the present depth prior to moving on to nodes at the next depth level.
        - DFS: Stack (LIFO) - This approach explores as far as possible along each branch before backtracking.

    The function starts from the initial state of the given problem and expands nodes using the specified fringe structure. 
    It keeps track of expanded states to avoid processing the same state multiple times, ensuring efficiency.

    Args:
        problem (SearchProblem): The search problem to be solved, providing methods to get the start state, check goal state, and get successors.
        fringe: A data structure (Queue or Stack) that determines the order of state exploration.

    Returns:
        list: A list of actions representing the path from the start state to the goal state if a solution is found; otherwise, returns an empty list.
    """
    expanded = set()  # Set to keep track of expanded

    # Push the start state with and initialize an empty path
    fringe.push((problem.getStartState(), []))

    while not fringe.isEmpty():
        # Get current state and its associated path
        state, path = fringe.pop()

        # Skip if current state is expanded
        if state in expanded:
            continue

        # Add state to expanded set
        expanded.add(state)

        # Check if state is goal
        if problem.isGoalState(state):
            return path

        # Explore the successors
        for successor, action, _ in problem.getSuccessors(state):
            if successor not in expanded:
                # Add successors to the fringe list to proces them
                fringe.push((successor, path + [action]))

    return []  # If we reach here, no solution has been found, return an empty list


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
    fringe = util.Stack() # FIFO
    return _firstSearch(problem,fringe)
    
def breadthFirstSearch(problem: SearchProblem) -> List[Directions]:
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    fringe = util.Queue() # LIFO
    return _firstSearch(problem,fringe)

def uniformCostSearch(problem: SearchProblem) -> List[Directions]:
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
