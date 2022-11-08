from models import (
    Decoder,
    Encoder,
)
from settings_train import EMBED_DIM, FF_DIM, NUM_HEADS

encoder = Encoder(embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM)
# print(encoder.build_graph().summary())
decoder = Decoder(
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    ff_dim=FF_DIM,
    vocab_size=20000,
)
# print(decoder.build_graph().summary())
