"""Channel Weight Learner: Learn optimal channel weights from training data.

Trains TaskSpecificAttentionHead weights using supervised learning on labeled
data (relevant/irrelevant pairs for a given task).

See ADR-0006 for architectural rationale.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

from .task_attention import TaskSpecificAttentionHead, create_task_head

logger = logging.getLogger(__name__)


class ChannelWeightLearner:
    """Learn optimal channel weights from labeled training data.

    Training data format:
        {
            "semantic": np.ndarray (hidden_dim,),
            "metadata": np.ndarray (hidden_dim,),
            "topology": np.ndarray (hidden_dim,),
            "temporal": np.ndarray (hidden_dim,),
            "label": float (1.0=relevant, 0.0=irrelevant)
        }

    Loss: Binary cross-entropy with margin ranking loss for contrastive learning.

    Parameters
    ----------
    hidden_dim : int
        Input dimension for all channels.
    output_dim : int
        Output embedding dimension.
    learning_rate : float
        Learning rate for optimizer.
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        output_dim: int = 32,
        learning_rate: float = 0.001,
    ):
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = learning_rate

        # Global model (trained on combined data)
        self._model = TaskSpecificAttentionHead(
            task_name="semantic_retrieval",
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self._optimizer = optim.Adam(self._model.parameters(), lr=learning_rate)

        # Per-task models (fine-tuned from global)
        self._task_models: Dict[str, TaskSpecificAttentionHead] = {}

        self._trained = False

    def train(
        self,
        train_data: List[Dict[str, Any]],
        epochs: int = 10,
        batch_size: int = 16,
        val_data: Optional[List[Dict[str, Any]]] = None,
        task_name: str = "semantic_retrieval",
        fine_tune: bool = False,
    ) -> Dict[str, float]:
        """Train channel weights on labeled data.

        Parameters
        ----------
        train_data : List[Dict]
            Training samples. Each dict must contain:
            - semantic, metadata, topology, temporal: np.ndarray
            - label: float (1.0 or 0.0)
        epochs : int
            Number of training epochs.
        batch_size : int
            Batch size for training.
        val_data : List[Dict], optional
            Validation data for monitoring.
        task_name : str
            Task type for task-specific training.
        fine_tune : bool
            If True, fine-tune from global model. If False, train from scratch.

        Returns
        -------
        Dict[str, float]
            Training metrics (loss, accuracy).
        """
        if not train_data:
            logger.warning("Empty training data provided.")
            return {"loss": 0.0, "accuracy": 0.0}

        # Select model: fine-tune from global or train task-specific
        if fine_tune and task_name not in self._task_models:
            self._task_models[task_name] = create_task_head(
                task_name,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
            )
            # Copy weights from global model
            self._task_models[task_name].load_state_dict(self._model.state_dict())

        model = self._task_models.get(task_name, self._model)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        all_labels = [d["label"] for d in train_data]
        pos_count = sum(1 for l in all_labels if l > 0.5)
        neg_count = len(all_labels) - pos_count

        if pos_count == 0 or neg_count == 0:
            logger.warning("Training data has only one class. Using BCE only.")
            use_margin = False
        else:
            use_margin = True

        total_loss = 0.0
        correct = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            # Shuffle data
            indices = np.random.permutation(len(train_data))

            for start in range(0, len(train_data), batch_size):
                batch_indices = indices[start : start + batch_size]
                batch = [train_data[i] for i in batch_indices]

                optimizer.zero_grad()

                # Prepare batch tensors
                sem_batch = torch.stack([
                    torch.tensor(d["semantic"], dtype=torch.float32)
                    for d in batch
                ])
                meta_batch = torch.stack([
                    torch.tensor(d["metadata"], dtype=torch.float32)
                    for d in batch
                ])
                topo_batch = torch.stack([
                    torch.tensor(d["topology"], dtype=torch.float32)
                    for d in batch
                ])
                temp_batch = torch.stack([
                    torch.tensor(d["temporal"], dtype=torch.float32)
                    for d in batch
                ])
                labels = torch.tensor(
                    [d["label"] for d in batch], dtype=torch.float32
                ).unsqueeze(1)

                # Forward pass
                embeddings = model.forward(sem_batch, meta_batch, topo_batch, temp_batch)

                # Simple BCE loss on relevance score
                # Use cosine similarity with a reference query embedding as score
                query_emb = embeddings[0:1]  # First sample as reference
                scores = torch.cosine_similarity(embeddings, query_emb.expand_as(embeddings), dim=1).unsqueeze(1)
                bce_loss = nn.BCEWithLogitsLoss()(scores, labels)

                loss = bce_loss

                # Margin ranking loss for contrastive learning
                if use_margin:
                    pos_emb = embeddings[labels.squeeze() > 0.5]
                    neg_emb = embeddings[labels.squeeze() < 0.5]
                    if len(pos_emb) > 0 and len(neg_emb) > 0:
                        pos_sim = torch.cosine_similarity(
                            pos_emb.mean(dim=0, keepdim=True),
                            query_emb,
                            dim=1
                        ).mean()
                        neg_sim = torch.cosine_similarity(
                            neg_emb.mean(dim=0, keepdim=True),
                            query_emb,
                            dim=1
                        ).mean()
                        margin_loss = torch.relu(0.5 - pos_sim + neg_sim).mean()
                        loss = loss + 0.1 * margin_loss

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

                with torch.no_grad():
                    preds = (torch.sigmoid(scores) > 0.5).float()
                    correct += (preds == labels).sum().item()

            avg_loss = epoch_loss / max(n_batches, 1)
            total_loss = avg_loss

            if epoch % 5 == 0 or epoch == epochs - 1:
                acc = correct / len(train_data)
                logger.info(
                    "Epoch %d/%d: loss=%.4f, acc=%.2f%%",
                    epoch + 1, epochs, avg_loss, acc * 100
                )

        self._trained = True
        return {
            "loss": float(total_loss),
            "accuracy": float(correct / len(train_data)),
        }

    def get_global_weights(self) -> np.ndarray:
        """Get channel weight logits from global model.

        Returns
        -------
        np.ndarray
            Raw channel weight logits (before softmax).
        """
        return self._model._channel_weight_logits.detach().cpu().numpy().copy()

    def get_learned_task_head(self, task_name: str) -> TaskSpecificAttentionHead:
        """Get a trained task head for a specific task.

        Parameters
        ----------
        task_name : str
            Task type.

        Returns
        -------
        TaskSpecificAttentionHead
            Trained task head.
        """
        if task_name in self._task_models:
            return self._task_models[task_name]
        return self._model

    def get_task_weights(self, task_name: str) -> Dict[str, float]:
        """Get normalized channel weights for a task.

        Parameters
        ----------
        task_name : str
            Task type.

        Returns
        -------
        Dict[str, float]
            Channel weights (sum to 1).
        """
        model = self.get_learned_task_head(task_name)
        return model.get_channel_weights()

    def save(self, path: str) -> None:
        """Save model state to file.

        Parameters
        ----------
        path : str
            File path.
        """
        state = {
            "global_model": self._model.state_dict(),
            "task_models": {
                name: model.state_dict()
                for name, model in self._task_models.items()
            },
            "config": {
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "lr": self.lr,
                "trained": self._trained,
            },
        }
        torch.save(state, path)
        logger.info("ChannelWeightLearner saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "ChannelWeightLearner":
        """Load model state from file.

        Parameters
        ----------
        path : str
            File path.

        Returns
        -------
        ChannelWeightLearner
            Loaded instance.
        """
        state = torch.load(path, map_location="cpu")

        config = state["config"]
        learner = cls(
            hidden_dim=config["hidden_dim"],
            output_dim=config["output_dim"],
            learning_rate=config["lr"],
        )

        learner._model.load_state_dict(state["global_model"])
        for name, model_state in state["task_models"].items():
            task_model = create_task_head(
                name,
                hidden_dim=config["hidden_dim"],
                output_dim=config["output_dim"],
            )
            task_model.load_state_dict(model_state)
            learner._task_models[name] = task_model

        learner._trained = config["trained"]
        logger.info("ChannelWeightLearner loaded from %s", path)
        return learner
