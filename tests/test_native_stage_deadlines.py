"""Private stage envelopes do not widen the fixed public profile route."""
from pathlib import Path
import tempfile
import unittest
from dongxi_llms.native_profile_supervisor import _supervise, NATIVE_DEADLINE_ENVELOPES


class DeadlineEnvelopes(unittest.TestCase):
    def test_only_original_declared_ceilings_are_available(self):
        self.assertEqual(NATIVE_DEADLINE_ENVELOPES, {
            'profile':900, 'story-pilot':14400, 'assistant-pilot':3600,
            'preference-pilot':1800, 'reasoning-pilot':1800})

    def refuse(self, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'uncreated'
            with self.assertRaises(ValueError):
                _supervise([], output, probe=lambda excluded: None, **kwargs)
            self.assertFalse(output.exists())

    def test_fixed_profile_remains_900_seconds(self):
        self.refuse(native=True, seconds=901)

    def test_every_stage_refuses_its_own_overrun(self):
        for stage, ceiling in NATIVE_DEADLINE_ENVELOPES.items():
            self.refuse(native=True, seconds=ceiling+1, native_stage=stage)

    def test_unknown_and_nonnative_stage_refused_before_side_effects(self):
        self.refuse(native=True, seconds=1, native_stage='unbounded')
        self.refuse(native=False, seconds=1, native_stage='story-pilot')
        self.refuse(native=False, seconds=3)


if __name__ == '__main__':
    unittest.main()
