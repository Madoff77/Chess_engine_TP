from prep_data import *
import pandas as pd
from model import *
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
 

 
df = pd.read_csv("chess_positions_evaluated.csv")
X = np.array([fen_to_tensor(fen) for fen in df['fen']])
y = np.array(df['stockfish_eval'].values) # valeur de stockfish pour chaque position
# X = X.reshape((-1, 768)) #reshape pour rester sur du PMC utiliser Dense -- > mise directement dans fen_to tensor

y = y/100.0 # on normalise le score entre -10 et 10, 10 étant un echec et mat pour les blancs

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
 
model = create_model()
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",    
    factor=0.5,         # réduire de moitié le LR
    patience=3,
    verbose=1
    )
 
early_stop = EarlyStopping(
    monitor="val_loss",  # pour repérer l'overfitting
    patience=20,
    verbose=1,
    restore_best_weights=True
)
callbacks = [reduce_lr, early_stop]
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])
model.summary()
model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_val,y_val), callbacks=callbacks
           , shuffle=True # pour eviter les repetition a cause des ouverture de parties
          )
 
model.save("model_eval.h5")

# print(y.min(), y.max())