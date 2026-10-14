# SPDX-License-Identifier: GPL-3.0-only
"""Check signal retention, noise suppression and chunk boundaries on known audio."""

import unittest

import numpy as np

from reference_denoise import learn_noise, reduce_noise


class ReferenceDenoiseTests(unittest.TestCase):
    def test_tone_survives_noise_is_reduced_and_chunks_are_continuous(self):
        rate = 16000
        rng = np.random.default_rng(705)
        length = rate*4
        reference = rng.normal(0, .01, (length, 2))
        profile, variability = learn_noise(reference, rate)
        phase = np.arange(length)*2*np.pi*1000/rate
        tone = np.sin(phase)
        noise = rng.normal(0, .01, length)
        signal = .06*tone + noise
        audio = np.stack([signal, .7*signal], axis=1)

        for variable in (False, True):
            with self.subTest(variable_noise=variable):
                whole = reduce_noise(audio, rate, profile, variability, variable, chunk_seconds=10)
                chunks = reduce_noise(audio, rate, profile, variability, variable, chunk_seconds=.6)
                self.assertEqual(chunks.shape, audio.shape)
                self.assertTrue(np.isfinite(chunks).all())
                np.testing.assert_allclose(chunks, whole, atol=2e-7, rtol=0)
                np.testing.assert_allclose(chunks[:, 1], .7*chunks[:, 0], atol=1e-12, rtol=0)
                middle = slice(rate//2, -rate//2)
                estimate = 2*np.mean(chunks[middle, 0]*tone[middle])
                self.assertGreater(estimate, .06*10**(-2/20))
                residual = chunks[middle, 0]-estimate*tone[middle]
                self.assertLess(np.sqrt(np.mean(residual**2)), .01/3)

    def test_silent_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "静音"):
            learn_noise(np.zeros((32768, 2)), 16000)


if __name__ == "__main__":
    unittest.main()
