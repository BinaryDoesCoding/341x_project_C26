import os
import tensorflow as tf

# Import whatever your original script uses
from train_vww import main as orig_main

# Monkeypatch: wrap Keras fit with callbacks by intercepting tf.keras.Model.fit
# This is a safe way when we don't want to rewrite the whole file.

_original_fit = tf.keras.Model.fit

def _fit_with_ckpt(self, *args, **kwargs):
    os.makedirs("trained_models", exist_ok=True)

    callbacks = list(kwargs.get("callbacks", []))

    callbacks += [
        tf.keras.callbacks.ModelCheckpoint(
            filepath="trained_models/ckpt_epoch{epoch:02d}_val{val_accuracy:.4f}.keras",
            monitor="val_accuracy",
            save_best_only=False,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath="trained_models/best.keras",
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]

    kwargs["callbacks"] = callbacks
    return _original_fit(self, *args, **kwargs)

tf.keras.Model.fit = _fit_with_ckpt

if __name__ == "__main__":
    orig_main([])
