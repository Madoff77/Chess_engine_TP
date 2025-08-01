import chess.pgn
import pandas as pd
import numpy as np
 
#load le pgn et retourne une liste de parties
def load_games(pgn_path, max_games=10):
    games = []
    pgn_path = 'dataset/lichess_elite_2022-04.pgn'
    with open(pgn_path, "r", encoding="utf-8") as pgn:
        for i in range(max_games):
            game = chess.pgn.read_game(pgn)
            if game is None:
                break
            games.append(game)
    return games
 
#convertit les parties en un fichier csv
#avec les colonnes : game_id, fen, move_played, turn, result
 
def pgn_to_csv(pgn_path, max_games=10, output_csv="chess_positions.csv"):
    data = []
    with open(pgn_path) as f:
        game_id = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None or game_id >= max_games:
                break
 
            result = game.headers["Result"]
            if result == "1-0":
                label = 1
            elif result == "0-1":
                label = 0
            else:
                label = 0.5
 
            board = game.board()
            for move in game.mainline_moves():
                fen = board.fen()
                turn = fen.split()[1]  # 'w' or 'b'
                board.push(move)
                uci_move = move.uci()
 
                data.append({
                    "game_id": game_id,
                    "fen": fen,
                    "move_played": uci_move,
                    "turn": turn,
                    "result": label
                })
 
            game_id += 1
 
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Fichier CSV généré avec {len(df)} positions.")
 
 
PIECE_TO_INDEX = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}
 
def fen_to_tensor(fen):
    board_tensor = np.zeros((8, 8, 12), dtype=np.float32)
    board_fen, turn = fen.split()[0], fen.split()[1]
    rows = board_fen.split('/')
 
    for i, row in enumerate(rows):
        col = 0
        for char in row:
            if char.isdigit():
                col += int(char)
            else:
                piece_index = PIECE_TO_INDEX[char]
                board_tensor[i, col, piece_index] = 1
                col += 1
    flat_board = board_tensor.flatten()
    turn_tensor = np.array([1.0 if turn == 'w' else 0.0], dtype=np.float32)
    return np.concatenate([flat_board, turn_tensor])