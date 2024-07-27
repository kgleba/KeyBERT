from typing import List

import numpy as np
import onnxruntime as rt
from tokenizers import Tokenizer

from keybert.backend import BaseEmbedder


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=1)
    norm[norm == 0] = 1e-12
    return vector / norm[:, np.newaxis]


class ONNXBackend(BaseEmbedder):
    def __init__(self, embedding_model: str, tokenizer_config: str):
        super().__init__()

        self.tokenizer = Tokenizer.from_file(tokenizer_config)
        self.tokenizer.enable_truncation(max_length=256)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=256)

        self.model = rt.InferenceSession(embedding_model)

    def embed(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        encoded = [self.tokenizer.encode(doc) for doc in documents]

        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)

        onnx_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

        model_output = self.model.run(None, onnx_input)
        last_hidden_state = model_output[0]

        input_mask_expanded = np.broadcast_to(np.expand_dims(attention_mask, -1), last_hidden_state.shape)
        embeddings = np.sum(last_hidden_state * input_mask_expanded, 1) / np.clip(input_mask_expanded.sum(1), a_min=1e-9, a_max=None)
        embeddings = normalize(embeddings).astype(np.float32)

        return embeddings
