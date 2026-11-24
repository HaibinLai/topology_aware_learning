import numpy as np
import torch
import unittest
import tempfile
import pickle
from pathlib import Path
from types import SimpleNamespace
from torch.utils.data import TensorDataset

from src.data import CorruptedDataset, GaussianNoiseDataset, MixedGaussianNoiseDataset
from src.decentralized_client import (
    compute_metropolis_hastings_weights,
    create_centrality_dict,
)
from src.utils import save_checkpoint


class NoiseAndMetropolisHastingsTests(unittest.TestCase):
    def test_gaussian_noise_preserves_zero_one_scale_and_labels(self):
        images = torch.tensor([[[0.0, 0.25], [0.75, 1.0]]])
        labels = torch.tensor([7])
        dataset = GaussianNoiseDataset(
            TensorDataset(images, labels), noise_std=0.0, seed=3,
            value_range="zero_one",
        )

        noisy, label = dataset[0]
        self.assertTrue(torch.equal(noisy, images[0]))
        self.assertEqual(label.item(), 7)

    def test_gaussian_noise_preserves_minus_one_one_scale(self):
        images = torch.tensor([[[-1.0, -0.5], [0.5, 1.0]]])
        labels = torch.tensor([2])
        dataset = GaussianNoiseDataset(
            TensorDataset(images, labels), noise_std=0.0, seed=3,
            value_range="minus_one_one",
        )

        noisy, _ = dataset[0]
        self.assertTrue(torch.equal(noisy, images[0]))

    def test_mixed_noise_changes_only_selected_examples(self):
        images = torch.full((10, 1, 2, 2), 0.5)
        labels = torch.arange(10)
        dataset = MixedGaussianNoiseDataset(
            TensorDataset(images, labels), ratio=0.4, noise_std=0.2, seed=11,
            value_range="zero_one",
        )

        changed = {
            idx for idx in range(len(dataset))
            if not torch.equal(dataset[idx][0], images[idx])
        }
        self.assertEqual(changed, dataset.noisy_indices)
        self.assertEqual(len(changed), 4)
        self.assertTrue(
            all(dataset[idx][1].item() == idx for idx in range(len(dataset)))
        )

    def test_metropolis_hastings_matrix_is_doubly_stochastic_and_sparse(self):
        topology = np.array(
            [
                [0, 1, 1, 1],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
                [1, 0, 0, 0],
            ],
            dtype=float,
        )
        weights = compute_metropolis_hastings_weights(topology)

        self.assertTrue(np.all(weights >= 0))
        self.assertTrue(np.allclose(weights, weights.T))
        self.assertTrue(np.allclose(weights.sum(axis=0), 1.0))
        self.assertTrue(np.allclose(weights.sum(axis=1), 1.0))
        self.assertTrue(
            np.all(weights[(topology == 0) & (~np.eye(4, dtype=bool))] == 0)
        )
        self.assertTrue(np.allclose(weights[0], [0.25, 0.25, 0.25, 0.25]))
        self.assertTrue(np.allclose(weights[1], [0.25, 0.75, 0.0, 0.0]))

    def test_ood_corruptions_are_deterministic_and_preserve_labels(self):
        images = torch.full((10, 3, 8, 8), 0.5)
        labels = torch.arange(10)
        for corruption in [
            "noise",
            "gaussian_noise",
            "blur",
            "defocus",
            "weather",
            "brightness",
            "contrast",
            "pixelate",
        ]:
            with self.subTest(corruption=corruption):
                dataset = CorruptedDataset(
                    TensorDataset(images, labels),
                    corruption=corruption,
                    severity=3,
                    ratio=0.4,
                    seed=19,
                    value_range="zero_one",
                )
                self.assertEqual(len(dataset.corrupted_indices), 4)
                for idx in range(len(dataset)):
                    first, label = dataset[idx]
                    second, _ = dataset[idx]
                    self.assertTrue(torch.equal(first, second))
                    self.assertEqual(label.item(), idx)
                    self.assertEqual(first.shape, images[idx].shape)
                    self.assertGreaterEqual(float(first.min()), 0.0)
                    self.assertLessEqual(float(first.max()), 1.0)

    def test_ood_dataset_is_serializable_for_parsl(self):
        images = torch.full((2, 1, 4, 4), 0.5)
        labels = torch.arange(2)
        dataset = CorruptedDataset(
            TensorDataset(images, labels),
            corruption="blur",
            ratio=1.0,
            cache=True,
            value_range="zero_one",
        )
        restored = pickle.loads(pickle.dumps(dataset))
        self.assertTrue(torch.equal(restored[0][0], dataset[0][0]))

    def test_upstream_centrality_metrics_are_available(self):
        topology = np.array(
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float
        )
        centralities = create_centrality_dict(
            topology, np.random.default_rng(4)
        )
        self.assertTrue(
            {"degree", "betweenness", "closeness", "eigen", "random"}
            <= centralities.keys()
        )
        self.assertGreater(centralities["closeness"][1], centralities["closeness"][0])

    def test_checkpoint_is_atomically_published(self):
        model = torch.nn.Linear(2, 1)
        client = SimpleNamespace(model=model)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "4_ckpt.pth"
            save_checkpoint(4, [client], [{"test_acc": 0.5}], checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["round_idx"], 4)
            self.assertEqual(checkpoint["client_results"][0]["test_acc"], 0.5)
            self.assertFalse(list(Path(directory).glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
