from models import Decoder, Encoder
from settings_train import (
    EMBED_DIM,
    FF_DIM,
    NUM_HEADS,
    KEY_DIM,
    VALUE_DIM,
    MAX_VOCAB_SIZE,
)

# get encoder model
# encoder = Encoder(
#     embed_dim=EMBED_DIM,
#     ff_dim=FF_DIM,
#     num_heads=NUM_HEADS,
#     key_dim=KEY_DIM,
#     value_dim=VALUE_DIM,
# )

# encoder.build_graph().summary()

# get decoder model
decoder = Decoder(
    embed_dim=EMBED_DIM,
    ff_dim=FF_DIM,
    num_heads=NUM_HEADS,
    vocab_size=MAX_VOCAB_SIZE,
    key_dim=KEY_DIM,
    value_dim=VALUE_DIM,
)

decoder.build_graph().summary()
