import json
from typing import Dict, Optional


class Word2VecTokenizer:
    """
    Tokenizer deterministico word-level per transcript frame-aligned.
    Versionato e serializzabile.
    """

    # --------
    # VERSIONE
    # --------
    VERSION = "1.0"

    # ----------------
    # TOKEN SPECIALI
    # ORDINE FISSO!
    # ----------------
    PAD = "<PAD>"
    SIL = "<SIL>"
    OOV = "<OOV>"

    SPECIAL_TOKENS = [PAD, SIL, OOV]

    # ----------------
    # COSTRUZIONE
    # ----------------
    def __init__(
        self,
        stoi: Dict[str, int],
        itos: Dict[int, str],
        version: str,
    ):
        self.stoi = stoi
        self.itos = itos
        self.version = version

        # ID rapidi
        self.pad_id = self.stoi[self.PAD]
        self.sil_id = self.stoi[self.SIL]
        self.oov_id = self.stoi[self.OOV]

        self.vocab_size = len(self.stoi)

    # ----------------
    # FACTORY: TRAIN
    # ----------------
    @classmethod
    def from_word2vec(cls, word2vec):
        """
        Costruisce il tokenizer a partire da Word2Vec.
        Da usare SOLO in preprocessing.
        """
        stoi = {}
        itos = {}

        idx = 0
        for tok in cls.SPECIAL_TOKENS:
            stoi[tok] = idx
            itos[idx] = tok
            idx += 1

        for word in word2vec.key_to_index:
            stoi[word] = idx
            itos[idx] = word
            idx += 1

        return cls(
            stoi=stoi,
            itos=itos,
            version=cls.VERSION,
        )

    # ----------------
    # ENCODE / DECODE
    # ----------------
    def encode(self, word: str) -> int:
        return self.stoi.get(word, self.oov_id)

    def decode(self, idx: int) -> str:
        return self.itos.get(idx, self.OOV)

    # ----------------
    # SERIALIZZAZIONE
    # ----------------
    def save(self, path: str):
        """
        Salva tokenizer + metadata.
        """
        payload = {
            "type": "WordTokenizer",
            "version": self.version,
            "special_tokens": self.SPECIAL_TOKENS,
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    # ----------------
    # DESERIALIZZAZIONE
    # ----------------
    @classmethod
    def load(cls, path: str, expected_version: Optional[str] = None):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        # ---- controlli di sicurezza
        assert payload["type"] == "WordTokenizer", "Tipo tokenizer errato"

        version = payload["version"]
        if expected_version is not None:
            assert (
                version == expected_version
            ), f"Version mismatch: {version} != {expected_version}"

        assert (
            payload["special_tokens"] == cls.SPECIAL_TOKENS
        ), "Token speciali non compatibili"

        stoi = payload["stoi"]
        itos = {int(k): v for k, v in payload["itos"].items()}

        return cls(
            stoi=stoi,
            itos=itos,
            version=version,
        )

    # ----------------
    # UTILITY
    # ----------------
    def summary(self) -> str:
        return (
            f"WordTokenizer(v={self.version}, "
            f"vocab_size={self.vocab_size}, "
            f"pad_id={self.pad_id}, "
            f"sil_id={self.sil_id}, "
            f"oov_id={self.oov_id})"
        )