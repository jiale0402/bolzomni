"""Historical block mapping, returning residue tuples instead of CA coordinates."""
from native_metrics import AA3_TO_1

def map_sequence_blocks_to_residues(tokens, residues):
    """Map encoded sequence-block positions to the CA atoms represented in a PDB.

    LNR references can contain ``f###`` cap/noncanonical blocks with no CA atom,
    and a small number of references have an unresolved canonical CA.  Dynamic
    programming keeps residue order, maximizes represented atoms, and then
    prefers exact canonical residue matches over wildcard ``f###`` matches.
    """
    n_tokens = len(tokens)
    n_residues = len(residues)
    # State score is lexicographic: represented atoms, exact canonical matches,
    # and finally fewer unmatched PDB CA atoms.
    scores = [[None] * (n_residues + 1) for _ in range(n_tokens + 1)]
    previous = [[None] * (n_residues + 1) for _ in range(n_tokens + 1)]
    scores[0][0] = (0, 0, 0)

    def update(i, j, score, parent):
        if scores[i][j] is None or score > scores[i][j]:
            scores[i][j] = score
            previous[i][j] = parent

    for i in range(n_tokens + 1):
        for j in range(n_residues + 1):
            score = scores[i][j]
            if score is None:
                continue
            if i < n_tokens:
                update(i + 1, j, score, (i, j, "skip_token"))
            if j < n_residues:
                update(i, j + 1, (score[0], score[1], score[2] - 1),
                       (i, j, "skip_residue"))
            if i < n_tokens and j < n_residues:
                token = tokens[i]
                symbol = AA3_TO_1.get(residues[j][1].resname)
                is_exact = len(token) == 1 and token == symbol
                if token.startswith("f") or is_exact:
                    update(i + 1, j + 1,
                           (score[0] + 1, score[1] + int(is_exact), score[2]),
                           (i, j, "match"))

    block_map = {}
    i, j = n_tokens, n_residues
    while i or j:
        parent = previous[i][j]
        if parent is None:
            raise RuntimeError("failed to align encoded sequence blocks to PDB CA atoms")
        parent_i, parent_j, action = parent
        if action == "match":
            block_map[parent_i] = residues[parent_j]
        i, j = parent_i, parent_j
    return block_map, scores[n_tokens][n_residues]
