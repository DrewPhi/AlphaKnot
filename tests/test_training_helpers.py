import coach
import os
import unittest
from unittest.mock import patch


class TrainingHelperTests(unittest.TestCase):
    def test_allocated_cpu_count_respects_slurm(self):
        with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "6"}):
            with patch.object(coach.config, "selfplay_workers", 0):
                self.assertEqual(coach.allocated_cpu_count(), 6)

    def test_worker_limit_caps_slurm_allocation(self):
        with patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "16"}):
            with patch.object(coach.config, "selfplay_workers", 4):
                self.assertEqual(coach.allocated_cpu_count(), 4)

    def test_ddp_shards_have_equal_lengths_and_preserve_examples(self):
        examples = list(range(5))
        shards = [
            coach.equal_ddp_shard(examples, rank, 3) for rank in range(3)
        ]

        self.assertEqual([len(shard) for shard in shards], [2, 2, 2])
        flattened = {item for shard in shards for item in shard}
        self.assertTrue(set(examples).issubset(flattened))
