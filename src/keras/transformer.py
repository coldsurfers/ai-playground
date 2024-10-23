import tensorflow as tf
from tensorflow.keras import layers

class PositionalEncoding(layers.Layer):
    def __init__(self, maxlen, embed_dim):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(maxlen, embed_dim)

    def get_angles(self, pos, i, embed_dim):
        i = tf.cast(i, tf.float32)  # Ensure 'i' is float32
        pos = tf.cast(pos, tf.float32)  # Ensure 'pos' is float32
        angle_rates = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(embed_dim, tf.float32))
        return pos * angle_rates

    def positional_encoding(self, maxlen, embed_dim):
        angle_rads = self.get_angles(
            pos=tf.range(maxlen)[:, tf.newaxis],
            i=tf.range(embed_dim)[tf.newaxis, :],
            embed_dim=embed_dim)

        # Apply sin to even indices and cos to odd indices
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        # Reconstruct the tensor by interleaving sines and cosines
        angle_rads = tf.concat([sines, cosines], axis=-1)

        pos_encoding = angle_rads[tf.newaxis, ...]

        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

# Multi-Head Self-Attention
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.projection_dim = embed_dim // num_heads

        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        attention, weights = self.scaled_dot_product_attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output

    def scaled_dot_product_attention(self, query, key, value):
        matmul_qk = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dim_key)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, value)
        return output, attention_weights


# Transformer Encoder Layer
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# Transformer Encoder Model
def build_transformer_model(vocab_size, maxlen, embed_dim, num_heads, ff_dim):
    inputs = layers.Input(shape=(maxlen,))

    # Embedding and Positional Encoding
    x = layers.Embedding(vocab_size, embed_dim)(inputs)
    x = PositionalEncoding(maxlen, embed_dim)(x)

    # Transformer block
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)

    # Make sure to pass `training` as an argument
    x = transformer_block(x, training=True)  # or pass `training` variable if inside a model's call method

    # Add the final layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# Parameters
vocab_size = 10000  # Vocabulary size
maxlen = 200        # Maximum length of input sequence
embed_dim = 32      # Embedding size for each token
num_heads = 2       # Number of attention heads
ff_dim = 32         # Hidden layer size in feed-forward network

# Build and compile the model
model = build_transformer_model(vocab_size, maxlen, embed_dim, num_heads, ff_dim)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])

# Model summary
model.summary()
