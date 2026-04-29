"""Trip2Vec model utilities."""

from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm


class TripCorpus:
    """Corpus reader for two-token taxi trip sentences."""

    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path

    def __iter__(self):
        with open(self.corpus_path) as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) == 2:
                    yield tokens

    def __len__(self):
        count = 0
        with open(self.corpus_path) as f:
            for _ in f:
                count += 1
        return count


class TqdmCallback(CallbackAny2Vec):
    """tqdm progress callback for gensim Word2Vec."""

    def __init__(self, total_epochs: int):
        self.bar = tqdm(total=total_epochs, desc="Trip2Vec training", unit="epoch")

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.bar.set_postfix(loss=f"{loss:.1f}")
        self.bar.update(1)

    def on_train_end(self, _model):
        self.bar.close()
