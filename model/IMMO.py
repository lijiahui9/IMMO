#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

class DynamicMaskGenerator:
    def __init__(self, initial_p=0.75, growth_rate=0.95, max_p=0.9):
        self.initial_p = initial_p
        self.growth_rate = growth_rate
        self.max_p = max_p

    def generate_mask(self, inputs, epoch):
        p = self.initial_p + (self.max_p - self.initial_p) * (1 - self.growth_rate ** epoch)
        p = min(p, self.max_p)
        mask = np.random.choice(2, size=inputs.shape, p=[1 - p, p]).astype(tf.keras.backend.floatx())
        return mask

def build_encoder(input_dim, dropout_rate):
    return models.Sequential([
        layers.Dense(256, activation='swish', input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(128, activation='swish'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation='swish'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
    ])

def build_decoder(output_dim, latent_dim, dropout_rate):
    return models.Sequential([
        layers.Dense(128, activation='swish', input_shape=(latent_dim,)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(256, activation='swish'),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rate),
        layers.Dense(output_dim, activation='linear'),
    ])

class MultiModalAutoencoder(models.Model):
    def __init__(self, input_dim_x, input_dim_y, input_dim_z, latent_dim=110, dropout_rate=0.473):
        super(MultiModalAutoencoder, self).__init__()
        self.input_dim_x = input_dim_x
        self.input_dim_y = input_dim_y
        self.input_dim_z = input_dim_z
        total_input_dim = input_dim_x + input_dim_y + input_dim_z

        self.encoder_x = build_encoder(input_dim_x, dropout_rate)
        self.encoder_y = build_encoder(input_dim_y, dropout_rate)
        self.encoder_z = build_encoder(input_dim_z, dropout_rate)

        self.fusion_layer = models.Sequential([
            layers.Dense(128, activation='swish'),
            layers.Dropout(dropout_rate),
            layers.Dense(latent_dim, activation='relu'),
        ])

        self.decoder = build_decoder(total_input_dim, latent_dim, dropout_rate)
        self.concat_layer = layers.Concatenate()

    def get_fused_encoding(self, x, y, z):
        encoded_x = self.encoder_x(x)
        encoded_y = self.encoder_y(y)
        encoded_z = self.encoder_z(z)
        concatenated = self.concat_layer([encoded_x, encoded_y, encoded_z])
        return self.fusion_layer(concatenated)

    def call(self, inputs, training=False):
        x, y, z = inputs
        fused = self.get_fused_encoding(x, y, z)
        return self.decoder(fused)

def masked_loss(y_true, y_pred, masks, input_dim_x, input_dim_y, input_dim_z, weights=(0.3, 0.3, 0.4)):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    masks = tf.cast(masks, tf.float32)

    y_true_x, y_true_y, y_true_z = tf.split(y_true, [input_dim_x, input_dim_y, input_dim_z], axis=1)
    y_pred_x, y_pred_y, y_pred_z = tf.split(y_pred, [input_dim_x, input_dim_y, input_dim_z], axis=1)
    mask_x, mask_y, mask_z = tf.split(masks, [input_dim_x, input_dim_y, input_dim_z], axis=1)

    mse_x = tf.reduce_mean(tf.square((y_true_x * mask_x - y_pred_x * mask_x)))
    mse_y = tf.reduce_mean(tf.square((y_true_y * mask_y - y_pred_y * mask_y)))
    mse_z = tf.reduce_mean(tf.square((y_true_z * mask_z - y_pred_z * mask_z)))

    return weights[0] * mse_x + weights[1] * mse_y + weights[2] * mse_z

def train_step(x_b, y_b, z_b, xm_b, ym_b, zm_b, mask_gen, model, optimizer, epoch):
    with tf.GradientTape() as tape:
        mask_x = mask_gen.generate_mask(x_b, epoch)
        mask_y = mask_gen.generate_mask(y_b, epoch)
        mask_z = mask_gen.generate_mask(z_b, epoch)

        x_masked = x_b * mask_x
        y_masked = y_b * mask_y
        z_masked = z_b * mask_z

        predictions = model((x_masked, y_masked, z_masked), training=True)
        y_true = tf.concat([x_b, y_b, z_b], axis=1)
        masks = tf.concat([xm_b, ym_b, zm_b], axis=1)

        loss = masked_loss(y_true, predictions, masks,
                           input_dim_x=model.input_dim_x,
                           input_dim_y=model.input_dim_y,
                           input_dim_z=model.input_dim_z)
    gradients = tape.gradient(loss, model.trainable_variables)
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, warmup=5):
        self.patience = patience
        self.min_delta = min_delta
        self.warmup = warmup
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False

    def on_epoch_end(self, epoch, current_loss):
        if epoch < self.warmup:
            return
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True

def train_model(x_train, y_train, z_train, x_m_train, y_m_train, z_m_train,
                input_dims, latent_dim=110, epochs=100, batch_size=64):

    mask_generator = DynamicMaskGenerator()
    model = MultiModalAutoencoder(*input_dims, latent_dim=latent_dim)

    lr_scheduler = optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.0018379230255141633,
        decay_steps=200,
        decay_rate=0.96,
        staircase=True
    )
    optimizer = optimizers.Adam(learning_rate=lr_scheduler)
    early_stopping = EarlyStopping()

    dataset = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train, z_train, x_m_train, y_m_train, z_m_train)
    ).shuffle(buffer_size=1024).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_loss_history = []
    best_weights = None

    for epoch in range(epochs):
        epoch_losses = []

        for batch in dataset:
            x_b, y_b, z_b, xm_b, ym_b, zm_b = batch
            loss = train_step(x_b, y_b, z_b, xm_b, ym_b, zm_b,
                              mask_generator, model, optimizer, epoch)
            epoch_losses.append(loss.numpy())

        avg_loss = np.mean(epoch_losses)
        train_loss_history.append(avg_loss)

        if avg_loss == min(train_loss_history):
            best_weights = model.get_weights()

        early_stopping.on_epoch_end(epoch, avg_loss)
        if early_stopping.stop_training:
            print(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

        current_lr = optimizer.learning_rate(epoch).numpy()
        print(f"Epoch {epoch + 1:03d}/{epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.5f}")

    if best_weights:
        model.set_weights(best_weights)
       
    plt.figure(figsize=(12, 8))
    plt.plot(range(1, len(train_loss_history)+1), train_loss_history, label='Training Loss', color='blue', linewidth=2)
    plt.title('Training Loss History', fontsize=18)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.legend(fontsize=14)
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.minorticks_on()
    plt.tick_params(which='major', length=8)
    plt.tick_params(which='minor', length=4)
    plt.tight_layout()
    plt.savefig('training_loss.png', dpi=300)  
    plt.show()
    
    return model

