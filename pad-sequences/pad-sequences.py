import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    # Number of sequences
    N = len(seqs)

    # Determine maximum length
    if max_len is None:
        max_len = max((len(seq) for seq in seqs), default=0)

    # Create padded array filled with pad_value
    padded = np.full((N, max_len), pad_value)

    # Copy sequences into padded array
    for i, seq in enumerate(seqs):
        length = min(len(seq), max_len)
        padded[i, :length] = seq[:length]

    return padded


seqs = [[1,2], [3,4,5], [6]]
print(pad_sequences(seqs))
seqs1 = [[1,2,3],[4]]
print(pad_sequences(seqs1))