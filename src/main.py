import pygame
import sys
import numpy as np
import chess

from prep_data import *
from model import *
from tensorflow.keras.models import load_model
from const import WIDTH, HEIGHT, SQSIZE
from const import *
from game import Game
from square import Square
from move import Move


class Main:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode( (WIDTH, HEIGHT) )
        pygame.display.set_caption('Chess')
        self.game = Game()
        self.model = load_model("model_eval.h5")  # Load your trained model here
    
    def evaluate_position(self,fen):
        vector = fen_to_tensor(fen).reshape(1, -1)  # (1, 768)
        score = self.model.predict(vector, verbose=0)[0][0]
        return score

    def mainloop(self):
        
        screen = self.screen
        game = self.game
        board = self.game.board
        dragger = self.game.dragger

        while True:
            # show methods
            game.show_bg(screen)
            game.show_last_move(screen)
            game.show_moves(screen)
            game.show_pieces(screen)
            game.show_hover(screen)

            if dragger.dragging:
                dragger.update_blit(screen)

            for event in pygame.event.get():

                # click
                if event.type == pygame.MOUSEBUTTONDOWN:
                    dragger.update_mouse(event.pos)

                    clicked_row = dragger.mouseY // SQSIZE
                    clicked_col = dragger.mouseX // SQSIZE

                    # if clicked square has a piece ?
                    if board.squares[clicked_row][clicked_col].has_piece():
                        piece = board.squares[clicked_row][clicked_col].piece
                        # valid piece (color) ?
                        if piece.color == game.next_player:
                            board.calc_moves(piece, clicked_row, clicked_col, bool=True)
                            dragger.save_initial(event.pos)
                            dragger.drag_piece(piece)
                            # show methods 
                            game.show_bg(screen)
                            game.show_last_move(screen)
                            game.show_moves(screen)
                            game.show_pieces(screen)
                
                # mouse motion
                elif event.type == pygame.MOUSEMOTION:
                    motion_row = event.pos[1] // SQSIZE
                    motion_col = event.pos[0] // SQSIZE

                    game.set_hover(motion_row, motion_col)

                    if dragger.dragging:
                        dragger.update_mouse(event.pos)
                        # show methods
                        game.show_bg(screen)
                        game.show_last_move(screen)
                        game.show_moves(screen)
                        game.show_pieces(screen)
                        game.show_hover(screen)
                        dragger.update_blit(screen)
                
                # click release
                elif event.type == pygame.MOUSEBUTTONUP:
                    
                    if dragger.dragging:
                        dragger.update_mouse(event.pos)

                        released_row = dragger.mouseY // SQSIZE
                        released_col = dragger.mouseX // SQSIZE

                        # create possible move
                        initial = Square(dragger.initial_row, dragger.initial_col)
                        final = Square(released_row, released_col)
                        move = Move(initial, final)

                        # valid move ?
                        if board.valid_move(dragger.piece, move):
                            # normal capture
                            captured = board.squares[released_row][released_col].has_piece()
                            board.move(dragger.piece, move)
                            # print('VALID MOVE:', move)
                            board.set_true_en_passant(dragger.piece)                            

                                                        # Évaluation par le modèle
                            # fen = board.get_fen()
                            # tensor = fen_to_tensor(fen).reshape(1, 768)
                            # eval_model = self.model.predict(tensor)[0][0]
                            # print("Évaluation du modèle :", eval_model)

                            # show methods
                            game.show_bg(screen)
                            game.show_last_move(screen)
                            game.show_pieces(screen)
                            # next turn
                            game.next_turn()
                            # if game.is_checkmate(game.board.next_turn()):
                            #     print("Échec et mat ! " + piece.color +" à gagné !" )
                            #     self.running = False
                            fen = board.get_fen()  
                            tensor = fen_to_tensor(fen) # (1, 768)
                            # vector = tensor.reshape(1, -1)
                            eval_score = self.evaluate_position(fen)
                            eval_round = round(eval_score, 2)
                            eval_str = format(eval_round, ".3f")
                            print(f"Position evaluation: {eval_str}")


                    dragger.undrag_piece()
                
                # key press
                elif event.type == pygame.KEYDOWN:
                    
                    # changing themes
                    if event.key == pygame.K_t:
                        game.change_theme()

                     # changing themes
                    if event.key == pygame.K_r:
                        game.reset()
                        game = self.game
                        board = self.game.board
                        dragger = self.game.dragger

                # quit application
                elif event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            pygame.display.update()


main = Main()
main.mainloop()