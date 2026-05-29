# Semantle Hypersphere Solver

Automated solver for [Semantle](https://semantle.com/) using **hypersphere intersection** (dictionary filtering) in the Google News Word2Vec space.

## Algorithm

1. Load `word2vec-google-news-300` via `gensim.downloader` and build a matrix \(V \in \mathbb{R}^{N \times 300}\) with **L2-normalized** rows.
2. For each guess \(\vec{g}\), the server returns cosine similarity \(c\) to the secret word (UI scale: ×100).
3. Filter candidates with vectorized `sim = V @ g` and keep rows where \(|sim - c| < \epsilon\) (\(\epsilon = 10^{-4}\) by default).
4. Pick the next guess near the **centroid** of remaining vectors; repeat until one word remains or the API reports victory.

## Requirements

- Python 3.10+ (3.13+ uses a NumPy binary loader; install `gensim` on 3.10–3.12 for the primary path)
- ~8 GB RAM recommended (matrix is ~3.6 GB; parsing needs headroom)
- After the first successful parse, `~/.cache/semantle-solver/vectors.npy` is reused (no re-download or re-parse)
- First run downloads the Word2Vec model (~1.6 GB)

```bash
pip install -r requirements.txt
python solve_semantle.py
```

### Interactive hint mode (you play, solver suggests)

Enter each guess and the similarity Semantle shows; get the next suggested word:

```bash
python hint_semantle.py
# or
python solve_semantle.py --interactive
```

Example session:

```text
guess> article 19.20
620 candidates remain.
Suggested next guess: insurance

guess> insurance 23.41
12 candidates remain.
Suggested next guess: cover
```

## API notes

The live backend is `https://server.semantle.com`:

| Action | Method | Path |
|--------|--------|------|
| Today's puzzle metadata | GET | `/semantle/game/{game_id}/{lang}` |
| Guess feedback | GET | `/similarity/{guess}/{secret}/{lang}` |

The browser already holds `secretWord` after loading the game; the solver uses the same endpoints. Scores are normalized with `initialSimilarity / 100` for filtering (stable cosine; `similarity` may differ during UI animation).

## Options

```bash
python solve_semantle.py --initial-guess country --epsilon 1e-4 --game-id 1611 -v
```

## Disclaimer

Use for learning and personal experimentation. Respect Semantle's terms of service and avoid hammering their servers (the solver issues one HTTP request per guess).
