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
          "mobility_weight": 10,
          "stability_weight": 10,
          "parity_weight": 1
          }
       return weights_mid
    else:  # Late game! Final phases
      weights_late = {
          "corner_weight": 30,
          "mobility_weight": 1,
          "stability_weight": 15,
          "parity_weight": 10
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
      actual_mobility_score = 0

      if (max_actual_mobility + min_actual_mobility) != 0:
          actual_mobility_score = (
          100 * (max_actual_mobility - min_actual_mobility)/(max_actual_mobility + min_actual_mobility)
          )

      return actual_mobility_score

  def calculate_corner_control(self, board, color):
    corners = [(0, 0), (0, board.shape[1] - 1), (board.shape[0] - 1, 0), (board.shape[0] - 1, board.shape[1] - 1)]

    max_corners = sum(1 for corner in corners if board[corner] == color)
    min_corners = sum(1 for corner in corners if board[corner] == 3 - color) 
    corner_score = 0

    if (max_corners + min_corners ) != 0:
      corner_score = (
        100 * ((max_corners - min_corners)/(max_corners + min_corners))
      )

    return corner_score

  def is_stable_in_direction(self, board, stability_map, row, col, dr, dc, color):
    board_size = board.shape[0]
    r = row + dr  # Move to the neighboring cell
    c = col + dc
    while 0 <= r < board_size and 0 <= c < board_size:
        if board[r][c] == 0 or (board[r][c] != color and not stability_map[r][c]):
            return False
        r += dr
        c += dc
    return True

  def calculate_stability(self, board, color):
    board_size = board.shape[0]
    stability_map = np.zeros_like(board, dtype=bool)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]
    # Initialize stability map with corners
    corners = [(0, 0), (0, board_size - 1), 
               (board_size - 1, 0), (board_size - 1, board_size - 1)]
    stable_discs = []
    for r, c in corners:
        if board[r][c] == color:
            stability_map[r][c] = True
            stable_discs.append((r, c))

    # Propagate stability
    while stable_discs:
        r, c = stable_discs.pop()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < board_size and 0 <= nc < board_size and
                not stability_map[nr][nc] and board[nr][nc] == color):
                # Check if the disc is stable in this direction
                if self.is_stable_in_direction(board, stability_map, nr, nc, dr, dc, color):
                    stability_map[nr][nc] = True
                    stable_discs.append((nr, nc))
    # Count the stable discs
    stable_count = np.sum(stability_map)
    return stable_count


    # Implement my heuristic here!
  def heuristic_eval_board(self, board, color):
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

    # Corner Control Evaluation
    corner_score = self.calculate_corner_control(board,color)

    #Stability
    stability_score = self.calculate_stability(board, color) - self.calculate_stability(board, 3-color)

    # Dynamic Weighting
    dynamic_weights = self.dynamic_weighting(board, color)
    
    total_board_score = (
      parity_score * dynamic_weights["parity_weight"]
      + mobility_score * dynamic_weights["mobility_weight"]
      + corner_score * dynamic_weights["corner_weight"]
      + stability_score * dynamic_weights["stability_weight"]
    )
    
    # printf("") print all the heuristic values
    return total_board_score

  def alpha_beta_search(self, board, depth, alpha, beta, max_player, color, heuristic_eval_board, time_start, time_limit, ordered_moves):
    """
    Let's use alpha-beta pruning to search the game tree
    """
    # Alpha-Beta Pruning implementation inspired by
    # "Simple Minimax with Alpha-Beta Pruning" by Sebastian Lague
    # Source: https://www.youtube.com/watch?v=l-hh51ncgDI

    # Time limit check
    if time.time() - time_start >= time_limit:
        raise TimeoutError  

    # Get all valid moves for the current player
    if max_player:
        valid_moves = get_valid_moves(board, color)
    else:
        valid_moves = get_valid_moves(board, 3 - color)

    if ordered_moves:
      # If the move is in the ordered list, put at front, otherwise put it in the back of the search queue
      moves_in_order = filter(lambda m: m in ordered_moves, valid_moves)
      moves_not_in_order = filter(lambda m: m not in ordered_moves, valid_moves)

      valid_moves = sorted(moves_in_order, key=lambda m: ordered_moves.index(m)) + list(moves_not_in_order)


     # Base case: Evaluate the board when the game ends so the search algorithm can "see branch endgame scores"
    if not valid_moves or depth == 0:
        return heuristic_eval_board(board, color), None
        
    
    best_move = None
    if max_player:
      max_eval = float('-inf')
      for move in valid_moves:
        simulate_board = deepcopy(board)
        execute_move(simulate_board, move, color)

        # Recursive call for min player
        curr_eval, _ = self.alpha_beta_search(simulate_board, depth - 1, alpha, beta, False, color, heuristic_eval_board, time_start, time_limit, ordered_moves)

        if curr_eval > max_eval:
          max_eval = curr_eval
          best_move = move

        max_eval = max(max_eval, curr_eval)
        alpha = max(alpha, max_eval)
        # Prune if beta is less than or equal to alpha
        if beta <= alpha:
            break

      return max_eval, best_move
    
    else: 
      min_eval = float('inf')
      for move in valid_moves:
        simulated_board = deepcopy(board)
        execute_move(simulated_board, move, 3 - color)

        # Recursive call for the max player
        curr_eval, _ = self.alpha_beta_search(simulated_board, depth - 1, alpha, beta, True, color, heuristic_eval_board, time_start, time_limit, ordered_moves)
        if curr_eval < min_eval:
          min_eval = curr_eval
          best_move = move
        min_eval = min(min_eval, curr_eval)
        beta = min(beta, min_eval)

        # Prune if beta is less than or equal to alpha
        if beta <= alpha:
            break

      return min_eval, best_move


  def step(self, board, color, opponent):
      """
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

      # If no valid moves indicate that but abort gracefully
      valid_moves = get_valid_moves(board, color)
      if not valid_moves:
          print("No valid moves in step")
          return None 
      
      ordered_moves = []

    
      best_score = float('-inf')
      best_move = None
      try:
        # Keep iterating while we're still under 2 seconds
        while time.time() - time_start < time_limit:
          # Current depths best scores and moves
          alpha = float('-inf') 
          beta = float('inf')

          some_score, some_move = self.alpha_beta_search(board, depth, alpha, beta, True, color, self.heuristic_eval_board, time_start, time_limit, ordered_moves)
          if some_move is not None:
            best_move = some_move
            best_score = some_score

            # best move, list -> best move :: List.filter (not List.mem x list) ordered_moves
            # ordered_moves = [best_move] + [move for move in ordered_moves if move != best_move]
            ordered_moves = [best_move] + list(
                                            filter(
                                                lambda move, bm=best_move: move != bm, ordered_moves
                                            )
)

        # Increase the depth while we still have time
        depth += 1

      except TimeoutError:
        pass

      time_taken = time.time() - time_start
      # Return the best move found
      print("depth searched:", depth - 1)
      print("My AI's turn took ", time_taken, "seconds.")
      print(best_score)
      return best_move if best_move else random_move

  # TEST CASES

  # Old bot
  # python3 simulator.py --player_1 bond_agent --player_2 student_agent  --display

  # New tester
  # python simulator.py --player_1 bond_agent --player_2 tester_agent  --display

  # Testing against minimax greedy
  # python simulator.py --player_1 bond_agent --player_2 gpt_greedy_corners_agent --display --autoplay --autoplay_runs 10 --board_size_min 6 --board_size_max 8

  # python simulator.py --player_1 bond_agent --player_2 isaac_agent --display --autoplay --autoplay_runs 10 --board_size_min 6 --board_size_max 8