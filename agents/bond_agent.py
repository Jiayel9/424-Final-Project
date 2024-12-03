# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time
from helpers import random_move, count_capture, execute_move, check_endgame, get_valid_moves

@register_agent("bond_agent")
class StudentAgent(Agent):
  """
  A class for your implementation. Feel free to use this class to
  add any helper functionalities needed for your agent.
  """

  def __init__(self):
    super(StudentAgent, self).__init__()
    self.name = "bond_agent"

  def dynamic_weighting(self, board, color):
    """
    Takes board and color as input
    Returns a dictionary of weights
    """
    total_pieces = np.sum(board != 0) # Total number of placed pieces
    total_positions = board.shape[0] * board.shape[1] # Shape -> (row, col)
    game_progress = total_pieces/total_positions # pieces/positions

    if game_progress < 0.2: # 0% - 30% of board is filled 
       weights_early = {
        "corner_weight": 10,
        "mobility_weight": 8,
        "stability_weight": 6,
        "parity_weight": 1
      }
       return weights_early
    elif game_progress < 0.7:
       weights_mid = {
          "corner_weight": 20,
          "mobility_weight": 12,
          "stability_weight": 10,
          "parity_weight": 1
          }
       return weights_mid
    else:  # Late game! Final phases
      weights_late = {
          "corner_weight": 30,
          "mobility_weight": 1,
          "stability_weight": 15,
          "parity_weight": 5
          }
      return weights_late
    
  def calculate_piece_parity(self, board, color):
   """
   Calculates the concept of coin parity - very similar to check_endgame
   input: board, current colour

   source: https://courses.cs.washington.edu/courses/cse573/04au/Project/mini1/RUSSIA/miniproject1_vaishu_muthu/Paper/Final_Paper.pdf
   """

   max_pieces = np.sum(board == color)
   min_pieces = np.sum(board == 3-color)


   difference_pieces = max_pieces - min_pieces
   total_pieces = max_pieces + min_pieces

   # Error catch
   if total_pieces == 0:
    print("No pieces on the board?")
    return 0

   parity_score = 100 * (difference_pieces/total_pieces)
   return parity_score

  def calculate_mobility(self, board, color):
      """
      Calculate Mobility
      """

      max_actual_mobility = len(get_valid_moves(board, color))
      min_actual_mobility = len(get_valid_moves(board, 3 - color))

      if (max_actual_mobility + min_actual_mobility) != 0:
          actual_mobility_score = (
          100 * (max_actual_mobility - min_actual_mobility)/(max_actual_mobility + min_actual_mobility)
          )
      else:
        actual_mobility_score = 0

      return actual_mobility_score


    # Implement my heuristic here!
  def heuristic_eval_board(self, board, color, player_score, opponent_score):
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

    # Parity Evaluation
    parity_score = self.calculate_piece_parity(board, color)

    # Mobility Evaulation
    mobility_score = self.calculate_mobility(board, color)

    # Dynamic Weighting
    dynamic_weights = self.dynamic_weighting(board, color)
    
    total_board_score = (
      parity_score * dynamic_weights["parity_weight"]
      + mobility_score * dynamic_weights["mobility_weight"]
    )
    
    # printf("") print all the heuristic values
    return total_board_score

  def alpha_beta_search(self, board, depth, alpha, beta, max_player, color, heuristic_eval_board, time_start, time_limit):
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
    # Alpha-Beta Pruning implementation inspired by
    # "Simple Minimax with Alpha-Beta Pruning" by Sebastian Lague
    # Source: https://www.youtube.com/watch?v=l-hh51ncgDI

    # Time limit check
    if time.time() - time_start >= time_limit:
        raise TimeoutError  

    # Base case: Evaluate the board when the game ends so the search algorithm can "see branch endgame scores"
    is_endgame, player_score, opponent_score = check_endgame(board, color, 3 - color)
    if is_endgame or depth == 0:
        return heuristic_eval_board(board, color, player_score, opponent_score)
    
    # Get all valid moves for the current player
    if max_player:
        valid_moves = get_valid_moves(board, color)
    else:
        valid_moves = get_valid_moves(board, 3 - color)
        
    if not valid_moves:  # If no moves are valid, return the evaluation
        return heuristic_eval_board(board, color, player_score, opponent_score)
    
    if max_player:
      max_eval = float('-inf')
      for move in valid_moves:
        simulate_board = deepcopy(board)
        execute_move(simulate_board, move, color)

        # Recursive call for min player
        score = self.alpha_beta_search(simulate_board, depth - 1, alpha, beta, False, color, heuristic_eval_board, time_start, time_limit)

        max_eval = max(max_eval, score)
        alpha = max(alpha, max_eval)

        # Prune if beta is less than or equal to alpha
        if beta <= alpha:
            break

      return max_eval
    
    else: 
      min_eval = float('inf')
      for move in valid_moves:
        simulated_board = deepcopy(board)
        execute_move(simulated_board, move, 3 - color)

        # Recursive call for the max player
        score = self.alpha_beta_search(simulated_board, depth - 1, alpha, beta, True, color, heuristic_eval_board, time_start, time_limit)

        min_eval = min(min_eval, score)
        beta = min(beta, min_eval)

        # Prune if beta is less than or equal to alpha
        if beta <= alpha:
            break

      return min_eval

  def step(self, board, player, opponent):
    """
    You can use the following variables to access the chess board:
    - chess_board: a numpy array of shape (board_size, board_size)
      where 0 represents an empty spot, 1 represents Player 1's discs (Blue),
      and 2 represents Player 2's discs (Brown).
    - player: 1 if this agent is playing as Player 1 (Blue), or 2 if playing as Player 2 (Brown).
    - opponent: 1 if the opponent is Player 1 (Blue), or 2 if the opponent is Player 2 (Brown).

    You should return a tuple (r,c), where (r,c) is the position where your agent
    wants to place the next disc. Use functions in helpers to determine valid moves
    and more helpful tools.
    """
    time_start = time.time()
    time_limit = 1.99

      # 6, 8, 10, 12 accounts for board sizes
    board_sizes = [6, 8, 10, 12]
    depths = [5,4,3,3]
    depth = depths[board_sizes.index(board.shape[0])] #Selects initial search depth based on board size


    # Initialize overall best move variable
    best_score = float('-inf')
    best_move = None

    try:
        while time.time() - time_start < time_limit:
          alpha = float('-inf') 
          beta = float('inf') 

          current_best_move = None 
          current_best_score = float('-inf') 

          # Recompute legal moves in each iteration
          for move in get_valid_moves(board, player): 
              simulated_board = deepcopy(board) 
              execute_move(simulated_board, move, player) 
              score = self.alpha_beta_search(simulated_board, depth, alpha, beta, True, player, self.heuristic_eval_board, time_start, time_limit)

              if score > current_best_score:
                  current_best_score = score
                  current_best_move = move

                  # See if you can do better than this, helps pruning a lot!
                  alpha = max(alpha, current_best_score)

          # Update the best move and score
          if current_best_score > best_score:
              best_score = current_best_score
              best_move = current_best_move

          depth += 1

    except TimeoutError:
        print("Timeout! Returning the best move found so far.")
        print("Current search depth is: ", depth)

    time_taken = time.time() - time_start
    print("My AI's turn took ", time_taken, "seconds.")

    if not best_move:
        print("Searched too slow!")

        if not current_best_move:
           print("No moves?")
           return random_move
        
        print(current_best_score)
        return current_best_move
    
    print(best_score)
    return best_move

  # TEST CASES

  # Old bot
  # python3 simulator.py --player_1 bond_agent --player_2 student_agent  --display

  # New tester
  # python simulator.py --player_1 the007_agent --player_2 tester_agent  --display

  # Testing against minimax greedy
  # python simulator.py --player_1 the007_agent --player_2 isaac_agent --display --autoplay --autoplay_runs 10 --board_size_min 6 --board_size_max 8