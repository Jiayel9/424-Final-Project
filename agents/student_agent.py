# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("student_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "StudentAgent"

  def alpha_beta_search(self, board, depth, alpha, beta, max_player, color, evaluate_board):
    """
    Let's use alpha-beta pruning to search the game tree

    parameters:
    - board: Current game board : (numpy.array) numpy.array
    - depth: Max depth for search, how many moves into the future do we want to see : int
    - alpha: value for max player : int
    - beta: value for min player : int
    - color: agent color (1 or 2) : int
    - eval_board: heuristic function that returns a score for a given board : fun 'a .... -> int


    returns: best move score
    """

    # Base case: terminal state or depth-limit
    is_endgame, player_score, opponent_score = check_endgame(board, color, 3 - color)
    if is_endgame or depth == 0:
      return evaluate_board(board, color, player_score, opponent_score)
    
    # Get all valid moves for the current player
    valid_moves = get_valid_moves(board, color if max_player else 3 - color)

    if not valid_moves:  # If no moves are valid, return the evaluation
        return evaluate_board(board, color, player_score, opponent_score)
    
    if max_player:
      best_score = float('-inf')
      for move in valid_moves:

        # let's simulate the move
        simulate_board = deepcopy(board)
        execute_move(simulate_board, move, color)

        # Recursive call for min player
        score = self.alpha_beta_search(simulate_board, depth - 1, alpha, beta, False, color, evaluate_board)

        best_score = max(best_score, score)
        alpha = max(alpha, best_score)

        # Prune if beta is less than or equal to alpha
        if beta <= alpha:
            break

      return best_score
    
    else: 
      best_score = float('inf')
      for move in valid_moves:
          # Simulate the move
          simulated_board = deepcopy(board)
          execute_move(simulated_board, move, 3 - color)

          # Recursive call for the maximizing player
          score = self.alpha_beta_search(simulated_board, depth - 1, alpha, beta, True, color, evaluate_board)

          best_score = min(best_score, score)
          beta = min(beta, best_score)

          # Prune if beta is less than or equal to alpha
          if beta <= alpha:
              break

      return best_score


  # Heuristic Search strategies here
  def step(self, board, player, opponent):
    """
    Implement the step function of your agent here.
    You can use the following variables to access the chess board:
    - board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.

    Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
    """
    # let's see all our legal moves
    legal_moves = get_valid_moves(board, player)

    if not legal_moves:
      return None
    
    best_move = None
    best_score = float('-inf')

    alpha = float('-inf')
    beta = float('inf')
    
    for move in legal_moves:
        simulated_board = deepcopy(board)
        execute_move(simulated_board, move, player)

        # Evaluate the move using alpha-beta
        # Start at depth 3
        score = self.alpha_beta_search(simulated_board, 2, alpha, beta, False, player, self.evaluate_board)

        if score > best_score:
            best_score = score
            best_move = move

        alpha = max(alpha, best_score)

    # Some simple code to help you with timing. Consider checking 
    # time_taken during your search and breaking with the best answer
    # so far when it nears 2 seconds.
    start_time = time.time()  
    time_taken = time.time() - start_time 
    print("My AI's turn took ", time_taken, "seconds.") 


    # Returning a random valid move as an example
    return best_move
  

  def dynamic_weighting(self, board, color):
    """
    Takes board and color as input
    Returns a dictionary of weights
    """
    total_pieces = np.sum(board != 0) # Total number of placed pieces
    total_positions = board.shape[0] * board.shape[1] # Shape -> (row, col)
    game_progress = total_pieces/total_positions # pieces/positions

    if game_progress < 0.4: # 0% - 30% of board is filled 
       weights_early = {
        "corner_weight": 10,
        "mobility_weight": 10,
        "stability_weight": 0,
        "score_weight": 1
      }
       return weights_early
    elif game_progress < 0.8:
       weights_mid = {
          "corner_weight": 15,
          "mobility_weight": 10,
          "stability_weight": 5,
          "score_weight": 1
          }
       return weights_mid
    else:  # Late game! Final phases
      weights_late = {
          "corner_weight": 30,
          "mobility_weight": 2,
          "stability_weight": 10,
          "score_weight": 5
          }
      return weights_late

     

  # Implement my heuristic here!
  def evaluate_board(self, board, color, player_score, opponent_score):
    """
    Evaluate the board state based on multiple factors. This is your heuristic

    Parameters:
    - board: 2D numpy array representing the game board.
    - color: Integer representing the agent's color (1 for Player 1/Blue, 2 for Player 2/Brown).
    - player_score: Score of the current player (the number of pieces the player has on the board)
    - opponent_score: Score of the opponent (the number of pieces the other player has on the board)

    Returns:
    - int: The evaluated score of the board. (Positive for player 1, negative for player 2)
    """
 
    corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]
    weights = self.dynamic_weighting(board, color)

    # 10 points for the current player for every corner they control (Have a piece there)
    corner_score = sum(1 for corner in corners if board[corner] == color) * weights["corner_weight"]
    corner_penalty = sum(1 for corner in corners if board[corner] == 3 - color) * -weights["corner_weight"]

    # Mobility: the number of moves the opponent can make
    opponent_moves = len(get_valid_moves(board, 3 - color))
    mobility_score = -opponent_moves

    
    total_board_score = (corner_score 
                         + corner_penalty 
                         + weights["mobility_weight"] * mobility_score
                         + weights["score_weight"] * (player_score - opponent_score )
    )
                         
    return total_board_score
  

# Ensure to test with:
# python simulator.py --player_1 student_agent --player_2 random_agent --display

# python simulator.py --player_1 student_agent --player_2 student_agent  --display