from prep_data import *
import pandas as pd
from model import *
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
 
 
pgn_path = 'dataset/lichess_elite_2022-04.pgn'
games = load_games(pgn_path, max_games=500)
print(f"{len(games)} parties chargées.")
 
pgn_to_csv(pgn_path, max_games=500)
 
 
 
df = pd.read_csv("chess_positions.csv")
X = np.array([fen_to_tensor(fen) for fen in df['fen']])
y = np.array(df['result'].values) #resultat de la partie
X = X.reshape((-1, 768)) #reshape pour rester sur du PMC utiliser Dense
 
 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
 
model = create_model()
reduce_lr = ReduceLROnPlateau(
    monitor="loss",    
    factor=0.5,         # réduire de moitié le LR
    patience=3,
    verbose=1
    )
 
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=1,
    restore_best_weights=True
)
callbacks = [reduce_lr, early_stop]
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.summary()
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val,y_val), callbacks=callbacks, shuffle=True)
 
model.save("model_eval.h5")