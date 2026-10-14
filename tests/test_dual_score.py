# SPDX-License-Identifier: GPL-3.0-only
"""Independent voices retain the MIDI timeline, crop boundaries and rests."""
import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from midi_to_score import NoteSpan, flatten_dual
from power_tone import Segment, load_dual_score, main, make_schedule, timing_arrays, voice_schedule


class DualScoreTests(unittest.TestCase):
    def test_manifest_cannot_overwrite_input_score(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duet.json"
            original = json.dumps({"events": [{"seconds": .5, "frequencies_hz": [440.0, 660.0]}]})
            path.write_text(original, encoding="utf-8")
            alias = Path(directory) / "alias.json"
            alias.symlink_to(path)
            for destination in (path, alias):
                with self.subTest(destination=destination), patch("sys.argv", [
                    "power_tone.py", "--dual-score", str(path),
                    "--manifest", str(destination), "--dry-run",
                ]):
                    with self.assertRaisesRegex(SystemExit, "不能覆盖输入乐谱"):
                        main()
                self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_diagnostics_are_rejected_as_score(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "take.json"
            path.write_text(json.dumps({"mode": "gpu_activity_modulation", "segments": []}),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "是播放诊断日志"):
                load_dual_score(path, 0)

    def test_tempo_change_crop_and_independent_leading_rest(self):
        # 100 ticks/beat: first beat = 1 second, subsequent beats = .5 seconds.
        melody = [NoteSpan(1, 0, 69, 80, 100, 200)]
        accompaniment = [NoteSpan(2, 0, 57, 80, 0, 300),
                         NoteSpan(2, 0, 60, 80, 50, 150)]
        events = flatten_dual([melody, accompaniment], 100, [(0, 1000000), (100, 500000)],
                              [0, 12], ["highest", "lowest"], .25, 1.75)
        self.assertEqual([e["frequencies_hz"] for e in events],
                         [[None, 440.0], [440.0, 440.0], [None, 440.0]])
        self.assertEqual([e["seconds"] for e in events], [.75, .5, .25])
        self.assertEqual(sum(e["seconds"] for e in events), 1.5)
        for before, after in zip(events, events[1:]):
            self.assertEqual(before["end_s"], after["start_s"])

    def test_other_voice_changes_do_not_restart_sustained_pitch(self):
        segments = [Segment("a", .2, (440.0, 660.0)),
                    Segment("b", .3, (440.0, 880.0)),
                    Segment("c", .1, (None, 880.0)),
                    Segment("d", .4, (440.0, None))]
        first = voice_schedule(segments, 0)
        second = voice_schedule(segments, 1)
        self.assertEqual([s.seconds for s in first], [.5, .1, .4])
        self.assertEqual([s.seconds for s in second], [.2, .4, .4])
        periods, _, ends = timing_arrays(first, 0, .5)
        self.assertEqual(periods.tolist(), [round(1e9/440), 0, round(1e9/440)])
        self.assertEqual(ends.tolist(), [500000000, 600000000, 1000000000])

    def test_player_loads_both_frequencies_and_transposes_both(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duet.json"
            path.write_text(json.dumps({"events": [
                {"seconds": .5, "frequencies_hz": [440.0, None]},
                {"seconds": .5, "frequencies_hz": [None, 220.0]},
            ]}), encoding="utf-8")
            args = argparse.Namespace(dual_score=path, score=None, chord_semitones=None,
                                      legato=False, transpose=12)
            segments = make_schedule(args, 2)
            self.assertEqual([s.frequencies for s in segments], [(880.0, None), (None, 440.0)])
            with self.assertRaisesRegex(ValueError, "两张卡"):
                make_schedule(args, 1)
            args.chord_semitones = 7
            with self.assertRaisesRegex(ValueError, "省略"):
                make_schedule(args, 2)


if __name__ == "__main__":
    unittest.main()
