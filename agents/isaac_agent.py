from agents.agent import Agent
from store import register_agent
from helpers import get_valid_moves, count_capture, execute_move, check_endgame
import copy
import random

@register_agent("isaac_agent")
class StudentAgent(Agent):
    """
    A custom agent for playing Reversi/Othello using the minimax algorithm.
    """

    def __init__(self):
        super().__init__()
        self.name = "IsaacAgent"

    def step(self, board, color, opponent):
        """
        Choose a move using minimax search with a specified depth.

        Parameters:
        - board: 2D numpy array representing the game board.
        - color: Integer representing the agent's color.
        - opponent: Integer representing the opponent's color.

        Returns:
        - Tuple (x, y): The coordinates of the chosen move.
        """
        # Set the depth limit for the minimax search
        depth = 3  # You can adjust the depth based on performance requirements

        # Get all legal moves for the current player
        legal_moves = get_valid_moves(board, color)

        if not legal_moves:
            return None  # No valid moves available, pass turn

        best_move = None
        best_score = float('-inf')

        for move in legal_moves:
            simulated_board = copy.deepcopy(board)
            execute_move(simulated_board, move, color)
            move_score = self.minimax(simulated_board, depth - 1, False, color, opponent)

            if move_score > best_score:
                best_score = move_score
                best_move = move

        # Return the best move found
        return best_move if best_move else random.choice(legal_moves)

    def minimax(self, board, depth, maximizing_player, color, opponent):
        """
        Minimax search algorithm.

        Parameters:
        - board: 2D numpy array representing the game board.
        - depth: Current depth in the game tree.
        - maximizing_player: Boolean indicating whether the current layer is maximizing or minimizing.
        - color: Integer representing the agent's color.
        - opponent: Integer representing the opponent's color.

        Returns:
        - int: The evaluated score of the board.
        """
        # Terminal condition: depth is 0 or game is over
        is_endgame, player_score, opponent_score = check_endgame(board, color, opponent)
        if depth == 0 or is_endgame:
            return self.evaluate_board(board, color, player_score, opponent_score)

        if maximizing_player:
            max_eval = float('-inf')
            legal_moves = get_valid_moves(board, color)
            if not legal_moves:
                # If no moves available, pass turn
                return self.minimax(board, depth - 1, False, color, opponent)
            for move in legal_moves:
                simulated_board = copy.deepcopy(board)
                execute_move(simulated_board, move, color)
                eval_score = self.minimax(simulated_board, depth - 1, False, color, opponent)
                max_eval = max(max_eval, eval_score)
            return max_eval
        else:
            min_eval = float('inf')
            legal_moves = get_valid_moves(board, opponent)
            if not legal_moves:
                # If opponent has no moves, pass turn
                return self.minimax(board, depth - 1, True, color, opponent)
            for move in legal_moves:
                simulated_board = copy.deepcopy(board)
                execute_move(simulated_board, move, opponent)
                eval_score = self.minimax(simulated_board, depth - 1, True, color, opponent)
                min_eval = min(min_eval, eval_score)
            return min_eval

    def evaluate_board(self, board, color, player_score, opponent_score):
        """
        Evaluate the board state based on multiple factors.

        Parameters:
        - board: 2D numpy array representing the game board.
        - color: Integer representing the agent's color.
        - player_score: Score of the current player.
        - opponent_score: Score of the opponent.

        Returns:
        - int: The evaluated score of the board.
        """
        # Corner positions are highly valuable
        corners = [
            (0, 0),
            (0, board.shape[1] - 1),
            (board.shape[0] - 1, 0),
            (board.shape[0] - 1, board.shape[1] - 1)
        ]
        corner_score = sum(1 for corner in corners if board[corner] == color) * 10
        corner_penalty = sum(1 for corner in corners if board[corner] == 3 - color) * -10

        # Mobility: the number of moves the opponent can make
        opponent_moves = len(get_valid_moves(board, 3 - color))
        mobility_score = -opponent_moves

        # Disc difference
        disc_difference = player_score - opponent_score

        # Combine scores
        total_score = (
            disc_difference +
            corner_score +
            corner_penalty +
            mobility_score
        )
        return total_score
    

# python simulator.py --player_1 student_agent --player_2 gpt_greedy_corners_agent --display --autoplay --autoplay_runs 10 --board_size_min 6 --board_size_max 10

# python simulator.py --player_1 student_agent --player_2 isaac_agent --display --autoplay --autoplay_runs 10 --board_size_min 6 --board_size_max 10