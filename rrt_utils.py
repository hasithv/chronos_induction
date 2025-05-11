import torch

EOS_TOKEN_ID = torch.tensor(1, dtype=torch.int)

def generate_random_token_ids(vocab, sequence_length, batch_size=1, include_eos=True):
    # random choice of vocab indices
    indices = torch.randint(0, vocab.shape[0], (batch_size, sequence_length+include_eos))

    # select the tokens
    tokens = vocab[indices]

    # add eos token to the end of each sequence
    if include_eos:
        tokens[:, -1] = EOS_TOKEN_ID
    return tokens

def repeat_sequence(token_ids, repetitions):
    return token_ids.repeat(1, repetitions)

def stack_sequences(sequences, include_eos=True):
    # stack the sequences along dimension 1 and add eos token to the end of each sequence if include_eos is True
    stacked = torch.cat(sequences, dim=1)
    if include_eos:
        stacked = torch.cat([stacked, torch.ones(stacked.shape[0], 1, dtype=torch.int) * EOS_TOKEN_ID], dim=1)
    return stacked


def generate_rrt_ids(vocab, sequence_length, repetitions=1, batch_size=1, include_eos=True):
    # repeat the token ids repetitions times and append to the end of each sequence
    token_ids = generate_random_token_ids(vocab, sequence_length, batch_size=batch_size, include_eos=False)
    token_ids = repeat_sequence(token_ids, repetitions)

    # append eos token to the end of each sequence
    if include_eos:
        token_ids = torch.cat([token_ids, torch.ones(batch_size, 1, dtype=torch.int) * EOS_TOKEN_ID], dim=1)
    return token_ids


# vocab = torch.tensor([i for i in range(4096)])
# print(generate_random_token_ids(vocab, 10, batch_size=2).shape)
# print(generate_rrt_ids(vocab, 3, repetitions=2, batch_size=2))