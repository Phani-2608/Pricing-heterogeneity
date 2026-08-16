"""Simple model registry: versioned save/load for the CATE model and feature builder.

Saves alongside a metadata JSON so a served prediction can be traced back
to the exact model version, config, and metrics that produced it.
"""
from __future__ import annotations

import json
import logging
import os
import pickle
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    model_version: str
    trained_at: str
    n_train: int
    metrics: dict
    config: dict


class ModelRegistry:
    """Filesystem-backed registry: `registry_dir/<version>/model.pkl + metadata.json`."""

    def __init__(self, registry_dir: str = "outputs/registry"):
        self.registry_dir = registry_dir
        os.makedirs(registry_dir, exist_ok=True)

    def _path(self, version: str) -> str:
        return os.path.join(self.registry_dir, version)

    def save(self, version: str, artifact: object, metadata: ModelMetadata) -> str:
        """Save a model + metadata under a version tag. Returns the path."""
        vpath = self._path(version)
        os.makedirs(vpath, exist_ok=True)
        with open(os.path.join(vpath, "model.pkl"), "wb") as f:
            pickle.dump(artifact, f)
        with open(os.path.join(vpath, "metadata.json"), "w") as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
        logger.info("Saved model version %s to %s", version, vpath)
        return vpath

    def load(self, version: str) -> tuple[object, ModelMetadata]:
        vpath = self._path(version)
        with open(os.path.join(vpath, "model.pkl"), "rb") as f:
            artifact = pickle.load(f)
        with open(os.path.join(vpath, "metadata.json")) as f:
            meta_dict = json.load(f)
        return artifact, ModelMetadata(**meta_dict)

    def list_versions(self) -> list[str]:
        if not os.path.isdir(self.registry_dir):
            return []
        return sorted(d for d in os.listdir(self.registry_dir) if os.path.isdir(self._path(d)))

    def latest_version(self) -> str | None:
        versions = self.list_versions()
        return versions[-1] if versions else None


def make_metadata(model_version: str, n_train: int, metrics: dict, config: dict) -> ModelMetadata:
    return ModelMetadata(
        model_version=model_version,
        trained_at=datetime.now(timezone.utc).isoformat(),
        n_train=n_train,
        metrics=metrics,
        config=config,
    )
