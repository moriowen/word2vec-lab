"""Model E. Skip-gram with negative sampling, written from scratch.

Everything gensim does in Cython, done explicitly. The five pieces that matter, each with
the failure it causes when omitted:

  subsampling         frequent words drown the signal and training is several times slower
  unigram^0.75 noise  the classic silent bug: loss falls smoothly, vectors are garbage
  dynamic window      nearby words stop being weighted more heavily than distant ones
  two matrices        W_in is kept, W_out exists only to compute the loss
  linear lr decay     late updates are as large as early ones and the model does not settle

Reference: Mikolov et al. 2013, "Distributed Representations of Words and Phrases and their
Compositionality", and the original C implementation at code.google.com/archive/p/word2vec.
"""
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Vocab:
    """Word counts, id mapping, subsampling probabilities and the negative sampling table."""

    def __init__(self, sentences, min_count=2, sample=1e-3, ns_exponent=0.75, table_size=10_000_000):
        counts = Counter(w for s in sentences for w in s)
        self.itos = [w for w, c in counts.most_common() if c >= min_count]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.counts = np.array([counts[w] for w in self.itos], dtype=np.float64)
        self.total = self.counts.sum()

        # Subsampling, Mikolov eq. 5. Keep probability sqrt(t/f) + t/f, clipped at 1.
        f = self.counts / self.total
        self.keep_prob = np.minimum((np.sqrt(f / sample) + 1) * (sample / f), 1.0)

        # Negative sampling draws from unigram^0.75, not from the raw unigram. The exponent
        # flattens the distribution so rare words are sampled more often than frequency alone
        # would allow. A precomputed table makes each draw O(1).
        w = self.counts ** ns_exponent
        w /= w.sum()
        self.neg_table = np.repeat(np.arange(len(self.itos)),
                                   np.maximum((w * table_size).astype(np.int64), 1))

    def __len__(self):
        return len(self.itos)

    def encode(self, sentences):
        return [np.array([self.stoi[w] for w in s if w in self.stoi], dtype=np.int32)
                for s in sentences]


def generate_pairs(encoded, vocab, window, rng):
    """One epoch of (center, context) pairs, subsampled and with a dynamic window."""
    centers, contexts = [], []
    for ids in encoded:
        if len(ids) < 2:
            continue
        kept = ids[rng.random(len(ids)) < vocab.keep_prob[ids]]
        if len(kept) < 2:
            continue
        # Dynamic window: the effective radius is drawn uniformly from 1..window for each
        # centre, which weights nearby words more heavily with no explicit weighting term.
        radii = rng.integers(1, window + 1, size=len(kept))
        for i, (c, r) in enumerate(zip(kept, radii)):
            lo, hi = max(0, i - r), min(len(kept), i + r + 1)
            for j in range(lo, hi):
                if j != i:
                    centers.append(c); contexts.append(kept[j])
    return np.asarray(centers, dtype=np.int64), np.asarray(contexts, dtype=np.int64)


class SGNS(nn.Module):
    """Two embedding matrices and a dot product. No hidden layer, no nonlinearity."""

    def __init__(self, vocab_size, dim, sparse=True):
        super().__init__()
        # sparse=True makes each step produce gradients only for the rows the batch touched.
        # A dense gradient would be the whole vocab_size x dim matrix on every step, which is
        # where a naive implementation spends most of its time.
        self.w_in = nn.Embedding(vocab_size, dim, sparse=sparse)   # encoder; the vectors we keep
        self.w_out = nn.Embedding(vocab_size, dim, sparse=sparse)  # decoder; discarded after training
        # Same init as the original C: W_in uniform in +/- 0.5/dim, W_out zeros.
        nn.init.uniform_(self.w_in.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.w_out.weight)

    def forward(self, centers, contexts, negatives):
        v = self.w_in(centers).unsqueeze(1)               # (B, 1, d)
        pos = torch.bmm(v, self.w_out(contexts).unsqueeze(2)).squeeze()      # (B,)
        neg = torch.bmm(self.w_out(negatives), v.transpose(1, 2)).squeeze(2)  # (B, k)
        return pos, neg

    @staticmethod
    def loss(pos, neg):
        """Mikolov eq. 4: log sigmoid(v.u) + sum over k of log sigmoid(-v.u_neg).

        Summed over the k negatives and averaged over the batch. Averaging over the
        negatives instead, which is what a plain BCEWithLogitsLoss does, divides the
        gradient by an extra factor of k and by the batch size on top of that. At B=8192
        and k=10 that leaves an effective per-pair learning rate around 3e-6 and the model
        does not move at all.
        """
        return -(F.logsigmoid(pos) + F.logsigmoid(-neg).sum(1)).mean()


def train(sentences, dim=300, window=5, negatives=10, min_count=2, sample=1e-3,
          epochs=10, alpha=0.001, min_alpha=1e-5, batch_size=8192, seed=42,
          device=None, sparse=False):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    # CPU by default, which is measured rather than assumed: at this vocabulary size MPS is
    # slower than CPU because the batch is small relative to the transfer and kernel-launch
    # overhead. On text8, with a 253k vocabulary, the balance reverses. See the run log.
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocab(sentences, min_count=min_count, sample=sample)
    encoded = vocab.encode(sentences)
    model = SGNS(len(vocab), dim, sparse=sparse).to(device)
    # Adam rather than the original's plain SGD. The C implementation updates once per
    # training pair, so its lr of 0.025 is a per-pair step; batching 8192 pairs and averaging
    # makes the equivalent SGD lr roughly alpha * batch_size, which is unstable. Adam is what
    # the PyTorch reference implementation the handout links (Andras7/word2vec-pytorch) uses,
    # and the linear decay from the original is kept on top of it.
    opt = torch.optim.Adam(model.parameters(), lr=alpha)

    epoch_pairs = [generate_pairs(encoded, vocab, window, rng) for _ in range(epochs)]
    total_batches = sum(int(np.ceil(len(c) / batch_size)) for c, _ in epoch_pairs)
    step, history = 0, []

    for centers, contexts in epoch_pairs:
        perm = rng.permutation(len(centers))
        centers, contexts = centers[perm], contexts[perm]
        running = n = 0.0
        for i in range(0, len(centers), batch_size):
            cb = torch.from_numpy(centers[i:i + batch_size]).to(device)
            xb = torch.from_numpy(contexts[i:i + batch_size]).to(device)
            nb = torch.from_numpy(
                rng.choice(vocab.neg_table, size=(len(cb), negatives))
            ).long().to(device)

            # Linear decay from alpha to min_alpha across the whole run, as in the original.
            for g in opt.param_groups:
                g["lr"] = alpha - (alpha - min_alpha) * (step / max(total_batches - 1, 1))

            pos, neg = model(cb, xb, nb)
            loss = SGNS.loss(pos, neg)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item(); n += 1; step += 1
        history.append(round(running / n, 4))

    return model, vocab, history, device


def to_keyedvectors(model, vocab):
    """Wrap W_in in a gensim KeyedVectors so every downstream tool works unchanged."""
    from gensim.models import KeyedVectors
    kv = KeyedVectors(vector_size=model.w_in.weight.shape[1])
    kv.add_vectors(vocab.itos, model.w_in.weight.detach().cpu().numpy())
    return kv
