from prep_data import *

def minimax(board, depth, alpha, beta, maximizing_player, model):
    
        if depth == 0 or board.is_game_over():
            # Évalue la position actuelle de fen en vector
            fen = board.fen()
            vector = fen_to_tensor(fen).reshape(1, -1)
            score = model.predict(vector, verbose=0)[0][0]
            return score if maximizing_player else -score, None

        legal_moves = list(board.legal_moves)
        best_move = None

        if maximizing_player:
            max_eval = float('-inf')
            for move in legal_moves:
                board.push(move)
                eval, _ = minimax(board, depth - 1, alpha, beta, False, model)
                board.pop()
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Élagage bêta
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for move in legal_moves:
                board.push(move)
                eval, _ = minimax(board, depth - 1, alpha, beta, True, model)
                board.pop()
                if eval < min_eval:
                    min_eval = eval
                    best_move = move
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Élagage alpha
            return min_eval, best_move