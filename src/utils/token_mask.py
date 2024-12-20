import random

def mask_single(i, item, batch_seed):
    """
    [!] Worker process, masking individual sequence in batch in a parralerized manner
    Performs masking on sequence.
        i: (enumerate(batch)[0]) seq position for seed control
        item: (enumerate(batch)[1]) single seq from batch
        batch_seed: (BATCH_NUM*BATCH_SIZE) unique seed for each batch
    Masking rules of 15% of each sequence:
        80%: mask
        10%: random mutation
        10%: unchanged
    Outputs a tuple of (masked_seq, mask_indices) of type (str, list(int)).
    """

    vocab = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O']

    def mutate(org):
        new = org
        while new == org:
            new = random.choice(vocab)
        return new

    seq = item['sequence']
    random.seed(batch_seed + i)
    chain = list(seq)
    seq_len = len(chain)
    num_to_mask = max(1, int(round(0.15 * seq_len)))
    mask_indices = random.sample(range(seq_len), num_to_mask)
    
    num_mask = int(round(0.8 * num_to_mask))
    num_random = int(round(0.1 * num_to_mask))
    num_unchanged = num_to_mask - num_mask - num_random

    # Adjust counts if necessary to ensure they sum to num_to_mask
    while num_mask + num_random + num_unchanged > num_to_mask:
        num_mask -= 1
    while num_mask + num_random + num_unchanged < num_to_mask:
        num_mask += 1

    # Create action labels
    actions = ['mask'] * num_mask + ['random'] * num_random + ['unchanged'] * num_unchanged
    random.shuffle(actions)

    # Apply masking according to actions
    for idx, action in zip(mask_indices, actions):
        if action == 'mask':
            chain[idx] = '<mask>'
        elif action == 'random':
            chain[idx] = mutate(chain[idx])
        else:
            pass  # Leave the token unchanged

    masked_seq = "".join(chain)

    return masked_seq, mask_indices