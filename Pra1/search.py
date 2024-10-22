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

        # Explore successors
        for successor, action, _ in problem.getSuccessors(state):

            # Check if the successor is the goal state (Optimization)
            #if problem.isGoalState(successor):
            #    return path + [action]

            # Add successor if it hasn't been expanded
            if successor not in expanded:
                # Process sucecssors
                fringe.push((successor, path + [action], _))
        
        # Add the state to the expanded set
        expanded.add(state)

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
    return aStarSearch(problem,nullHeuristic)
    
def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):
    """A* search following the given pseudocode with fringe and CLOSE lists."""
    
    # fringe is a priority queue, storing nodes with their priority (f = g + h)
    fringe = util.PriorityQueue()
    
    # expanded is a set of nodes that have already been expanded
    expanded = set()
    
    # Use a dictionary to keep track of the best cost found so far for each node
    g = {}
    
    # Initial node (node_start) and its cost
    node_start = problem.getStartState()
    g[node_start] = 0  # g(node_start) = 0
    h_start = heuristic(node_start, problem)
    fringe.push((node_start, [], 0), h_start)  # f(node_start) = h(node_start)
    
    # Main loop of the algorithm
    while not fringe.isEmpty():
        # Pop the node with the lowest f(n) = g(n) + h(n)
        current_state, current_path, current_cost = fringe.pop()
        
        # If node_current is goal state, return the solution
        if problem.isGoalState(current_state):
            return current_path
        
        # Get successors
        for successor, action, step_cost in problem.getSuccessors(current_state):
            successor_cost = current_cost + step_cost  # g(successor)

            # If this path to successor is better, update the path
            # Only keep track of the best path so far (in the dictionary)
            if successor not in g or successor_cost < g[successor]:
                g[successor] = successor_cost
                h_successor = heuristic(successor, problem)
                f_successor = successor_cost + h_successor
                
                # Push the successor onto the fringe
                fringe.push((successor, current_path + [action], successor_cost), f_successor)

        # Add current_state to expanded list
        expanded.add(current_state)
    
    # If no solution was found, return an empty list
    return []


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch