# SPDX-License-Identifier: GPL-3.0-only
"""Independent FP4 codec check and bounded CUDA transport/timing checks."""
import argparse
import math
import shutil
import unittest

import torch

from power_tone import Segment, load_engine, run


@unittest.skipUnless(torch.cuda.is_available() and shutil.which("nvcc"), "CUDA Toolkit required")
class PowerToneCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = load_engine([0])

    def test_fp4_decode_against_scalar_format(self):
        magnitudes = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
        expected = []
        for scale_code in range(256):
            exponent, fraction = (scale_code >> 3) & 15, scale_code & 7
            scale = fraction / 512 if exponent == 0 else (1 + fraction / 8) * 2 ** (exponent - 7)
            if scale_code & 128:
                scale = -scale
            if scale_code & 127 == 127:
                scale = math.nan
            for code in range(16):
                expected.append((-1 if code & 8 else 1) * magnitudes[code & 7] * scale)
        packed = [(2 * index) | ((2 * index + 1) << 4) for index in range(8)] * 256
        codes = torch.tensor(packed, dtype=torch.uint8, device="cuda:0")
        scales = torch.arange(256, dtype=torch.uint8, device="cuda:0")
        output = torch.empty(4096, dtype=torch.float16, device="cuda:0")
        self.engine.decode_weights(codes, scales, output)
        actual = output.cpu()
        oracle = torch.tensor(expected, dtype=torch.float16)
        self.assertTrue(torch.all((actual == oracle) | (torch.isnan(actual) & torch.isnan(oracle))).item())

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA GPUs required")
    def test_two_voices_with_fp4_and_transport(self):
        for engine in ("resident", "burst"):
            with self.subTest(engine=engine):
                args = argparse.Namespace(engine=engine, carrier="fp4", duty=0.5,
                                          work_repeats=128, tensor_reuse=8,
                                          blocks_per_sm=4, memory_mib=64, link_gib_s=1,
                                          link_chunk_mib=4, lineout_marker=False, countdown=0)
                data = run(args, [Segment("two voices", 0.4, (440.0, 660.0)),
                                  Segment("rest", 0.1, (None, None)),
                                  Segment("two voices", 0.4, (880.0, 1320.0))], [0, 1])
                self.assertGreater(data["link_load"]["copied_bytes"], 0)
                for card in data["diagnostics"]:
                    self.assertFalse(card["cancelled"])
                    self.assertAlmostEqual(card["elapsed_s"], 0.9, delta=0.08)
                    self.assertGreater(card["weight_tiles_warp0"], 0)
                    self.assertEqual(len(card["switch_measurements"]), 2)
                    expected_edges = 0
                    for metric in card["switch_measurements"]:
                        self.assertIn(metric["event"], (0, 2))
                        self.assertAlmostEqual(metric["observed_switch_hz"] / metric["target_hz"], 1, delta=0.01)
                        expected_edges += round(metric["target_hz"] * 0.4)
                    self.assertAlmostEqual(card["rising_edges_thread0"] / expected_edges, 1, delta=0.02)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "two CUDA GPUs required")
    def test_burst_voices_rest_independently(self):
        args = argparse.Namespace(engine="burst", carrier="tensor", duty=0.5,
                                  work_repeats=512, tensor_reuse=8, blocks_per_sm=4,
                                  memory_mib=64, link_gib_s=0, link_chunk_mib=4,
                                  lineout_marker=False, countdown=0)
        segments = [Segment("first alone", .2, (440.0, None)),
                    Segment("together", .2, (440.0, 660.0)),
                    Segment("second alone", .2, (None, 660.0)),
                    Segment("new pair", .2, (880.0, 1320.0))]
        data = run(args, segments, [0, 1])
        self.assertAlmostEqual(data["voice_timelines"][0]["segments"][0]["end_s"], .4)
        self.assertAlmostEqual(data["voice_timelines"][1]["segments"][1]["start_s"], .2)
        self.assertAlmostEqual(data["voice_timelines"][1]["segments"][1]["end_s"], .6)
        for card in data["diagnostics"]:
            self.assertFalse(card["cancelled"])
            self.assertAlmostEqual(card["elapsed_s"], .8, delta=.04)
            self.assertEqual(len(card["switch_measurements"]), 2)
            for metric in card["switch_measurements"]:
                self.assertAlmostEqual(metric["observed_switch_hz"]/metric["target_hz"], 1, delta=.01)


if __name__ == "__main__":
    unittest.main()
