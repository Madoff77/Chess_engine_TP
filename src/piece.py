import os

class Piece:

    def __init__(self, name, color, value, texture=None, tenture_rect=None):
        self.name = name 
        self.color = color

        value_sign = 1 if color == 'white' else -1
        self.value = value * value_sign  # Adjust value based on color
        self.moves = []
        self.moved = False  # Track if the piece has moved
        self.texture = texture
        self.set_texture (size = 80)
        self.tenture_rect = tenture_rect

    def set_texture(self,size = 80):
        self.texture = os.path.join(
            f'asset/{size}px/{self.color}_{self.name}.png')
    
    def add_move(self, move):
        self.moves.append(move)

    def clear_moves(self):
        self.moves = []

class Pawn(Piece):

    def __init__(self, color):
        
        self.dir = -1 if color == 'white' else 1  # Up for white, down for black
        self.en_passant = False
        super().__init__('Pawn',color, 1.0)  

class Knight(Piece):

    def __init__(self, color):
        super().__init__('Knight',color, 3.0)  

class Bishop(Piece):

    def __init__(self, color):
        super().__init__('Bishop',color, 3.001)  

class Rook(Piece):

    def __init__(self, color):
        super().__init__('Rook',color, 5.0) 

class Queen(Piece):

    def __init__(self, color):
        super().__init__('Queen',color, 9.0) 

class King(Piece):

    def __init__(self, color):
        self.left_rook = None
        self.right_rook = None
        super().__init__('King',color, 1000000.0)  # King value is high because it's the most important piece