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

  def order_moves(self, board, moves, color):
    move_scores = []

    for move in moves:
      simulated_board = deepcopy(board)
      execute_move(simulated_board, move, color)
      score = self.evaluate_board(simulated_board, color, 0, 0)
      move_scores.append((score, move))

    # Sort moves based on the heuristic score in descending order
    move_scores.sort(reverse=True)
    ordered_moves = [move for score, move in move_scores]
    return ordered_moves

  def alpha_beta_search(self, board, depth, alpha, beta, max_player, color, evaluate_board, time_start, time_limit):
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
       return evaluate_board(board, color, player_score, opponent_score)
    
    # Get all valid moves for the current player
    if max_player:
       valid_moves = get_valid_moves(board, color)
    else:
       valid_moves = get_valid_moves(board, 3 - color)
       
    ordered_moves = self.order_moves(board, valid_moves, color if max_player else 3 - color)
    if not valid_moves:  # If no moves are valid, return the evaluation
        return evaluate_board(board, color, player_score, opponent_score)
    
    if max_player:
      max_eval = float('-inf')
      for move in ordered_moves:
        simulate_board = deepcopy(board)
        execute_move(simulate_board, move, color)

        # Recursive call for min player
        score = self.alpha_beta_search(simulate_board, depth - 1, alpha, beta, False, color, evaluate_board, time_start, time_limit)

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
        score = self.alpha_beta_search(simulated_board, depth - 1, alpha, beta, True, color, evaluate_board, time_start, time_limit)

        min_eval = min(min_eval, score)
        beta = min(beta, min_eval)

        # Prune if beta is less than or equal to alpha
        if beta <= alpha:
            break

      return min_eval


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
    # Initialize a few variables
    time_start = time.time() 
    time_limit = 1.99 

    # 6, 8, 10, 12 accounts for board sizes
    board_sizes = [6, 8, 10, 12]
    depths = [6,4,3,2]
    depth = depths.index(board.shape[0]) #Selects initial search depth based on board size
    best_move = None 
    best_score = float('-inf')

    # let's see all our legal moves
    legal_moves = get_valid_moves(board, player)
    if not legal_moves:
      return None
    
    try:
      while time.time() - time_start < time_limit:
        alpha = float('-inf') 
        beta = float('inf') 

        current_best_move = None 
        current_best_score = float('-inf') 

        for move in legal_moves: 
          simulated_board = deepcopy(board) 
          execute_move(simulated_board, move, player) 
          score = self.alpha_beta_search(simulated_board, depth, alpha, beta, False, player, self.evaluate_board, time_start, time_limit)
        
          if score > current_best_score:
            current_best_score = score
            current_best_move = move
            alpha = max(alpha, current_best_score)
        
        
        if current_best_score > best_score:
          best_score = current_best_score
          best_move = current_best_move

        # If the n + 1, matta fact don't stop we check deeper
        # May reconsider this, as sometimes it takes multiple depths 
        if current_best_score <= best_score:
          depth += 3
        else:
          depth += 1
    except TimeoutError:
      pass
    
    time_taken = time.time() - time_start 
    print("My AI's turn took ", time_taken, "seconds.") 

    print(best_score)
    return best_move

  def count_stable(self, board, color):
    stable_count = 0
    board_size = board.shape[0]

    # Up Down Left Right + Diagonals
    directions = [
      (-1, 0), (1, 0), (0, -1), (0, 1),  # vertical and horizontal
      (-1, -1), (-1, 1), (1, -1), (1, 1)  # diagonals
    ]

    # Helper function to check if a piece is stable
    def is_stable(row, col):
      """
      Takes row, col of piece as input
      Returns whether or not this piece is stable 

      Note: If there is a piece adjacent to the piece, or empty space the piece is unstable
      """
      for dr, dc in directions:
        r, c = row, col

        # check if move is within bounds
        while 0 <= r < board_size and 0 <= c < board_size:
          # empty spot or opponent piece, break
          if board[r][c] != color: 
              break
          
          r += dr
          c += dc
        else:
          # If we exited without encountering an opponent's piece, direction is stable
          # continue to check other directions
          continue 

        return False  # If any direction is unstable, the piece is not stable
      return True
    
    # Check corners first, as they are always stable
    corners = [(0, 0), (0, board_size - 1), (board_size - 1, 0), (board_size - 1, board_size - 1)]
    for corner in corners:
        r, c = corner
        if board[r][c] == color:
            stable_count += 1


    # Check edges and interior pieces
    for r in range(board_size):
        for c in range(board_size):
            if board[r][c] == color and is_stable(r, c):
                stable_count += 1

    return stable_count
  

  def dynamic_weighting(self, board, color):
    """
    Takes board and color as input
    Returns a dictionary of weights
    """
    total_pieces = np.sum(board != 0) # Total number of placed pieces
    total_positions = board.shape[0] * board.shape[1] # Shape -> (row, col)
    game_progress = total_pieces/total_positions # pieces/positions

    if game_progress < 0.3: # 0% - 30% of board is filled 
       weights_early = {
        "corner_weight": 10,
        "mobility_weight": 8,
        "stability_weight": 1,
        "score_weight": 1
      }
       return weights_early
    elif game_progress < 0.7:
       weights_mid = {
          "corner_weight": 15,
          "mobility_weight": 10,
          "stability_weight": 5,
          "score_weight": 1
          }
       return weights_mid
    else:  # Late game! Final phases
      weights_late = {
          "corner_weight": 25,
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

    
    total_board_score = (
      corner_score 
    + corner_penalty 
    + weights["mobility_weight"] * mobility_score
    + weights["stability_weight"] * self.count_stable(board, color)
    + weights["score_weight"] * (player_score - opponent_score )
    )
    
    return total_board_score
  

# Ensure to test with:
# python simulator.py --player_1 student_agent --player_2 random_agent --display

# python simulator.py --player_1 student_agent --player_2 student_agent  --display
# python simulator.py --player_1 student_agent --player_2 human_agent  --display
# python simulator.py --player_1 student_agent --player_2 gpt_greedy_corners_agent --display --autoplay --autoplay_runs 1 --board_size_min 6 --board_size_max 10


# python simulator.py --player_1 student_agent --player_2 tester_agent --display --autoplay --autoplay_runs 1 --board_size_min 6 --board_size_max 10
# python simulator.py --player_1 student_agent --player_2 tester_agent --display