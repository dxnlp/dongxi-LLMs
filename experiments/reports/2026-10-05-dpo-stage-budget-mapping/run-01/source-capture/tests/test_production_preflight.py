"""Injected source contracts only: no real GPU, cgroup, quota or platform probe."""
from copy import deepcopy
from dataclasses import replace
import json
import unittest
from unittest.mock import patch

from dongxi_llms.production_preflight import (ADAPTERS, GIB, MAX_RESPONSE_BYTES, PreflightPolicy,
    verify_fixture_drain, verify_preflight)


def policy():
    return PreflightPolicy("assistant-dpo-smoke", "a"*64, "b"*64, "dongxi-stage-"+"c"*32,
                           1000, 2*GIB, 8, 16*1024**2, 32, 60.)


def observation(sequence=1, observed_at=100.):
    p = policy()
    return {"schema": "dongxi-preflight-observation-v1", "scope": "authored-local-fixture",
        "stage_id": p.stage_id, "preparation_sha256": p.preparation_sha256, "nonce": p.observation_nonce,
        "sequence": sequence, "observed_at": observed_at, "observer_seconds": .01,
        "host": {"available_bytes": 40*GIB, "complete": True, "measurement": "injected-sample-not-continuous"},
        "boundary": {"adapter": ADAPTERS["containment"], "instance_id": p.instance_id, "owner_uid": 1000,
            "exclusive": True, "membership_complete": True, "members": [], "hard_memory_bytes": 2*GIB, "hard_pid_limit": 8},
        "quota": {"adapter": ADAPTERS["artifact_quota"], "instance_id": p.instance_id, "owner_uid": 1000,
            "exclusive": True, "hard_bytes": 16*1024**2, "hard_entries": 32},
        "observer": {"adapter": ADAPTERS["bounded_observers"], "timeout_enforced": True, "maximum_seconds": .5,
            "maximum_bytes": MAX_RESPONSE_BYTES},
        "watchdog": {"adapter": ADAPTERS["external_watchdog"], "instance_id": p.instance_id,
            "independent": True, "armed": True, "deadline_seconds": 60., "cleanup_bounded": True},
        "gpu": {"adapter": ADAPTERS["gpu_clearance"], "ownership_complete": True, "unreadable": 0,
            "conflicts": [], "active_contexts": [], "scope": "injected-no-real-device-inspected"}}


def encode(value):
    return json.dumps(value, allow_nan=False).encode()


def pair():
    return [observation(), observation(2, 100.1)]


def drain():
    return dict(schema="dongxi-fixture-drain-v1", scope="authored-local-fixture", instance_id=policy().instance_id,
                owner_uid=1000, membership_complete=True, remaining_members=[], cleanup_errors=[], boundary_removed=True)


class PreflightTests(unittest.TestCase):
    def run_pair(self, values=None, **kwargs):
        return verify_preflight(policy(), [encode(v) for v in (values or pair())], now=100.2, **kwargs)

    def test_ready_fixture_never_launches_signals_or_claims_platform_proof(self):
        with patch("subprocess.Popen") as spawn, patch("os.kill") as kill:
            result = self.run_pair()
        spawn.assert_not_called(); kill.assert_not_called()
        self.assertFalse(result["production_ready"]); self.assertFalse(result["launch_authorized"])
        self.assertEqual(result["scope"], "authored-local-fixture")
        self.assertEqual(result["minimum_sampled_host_available_bytes"] if "minimum_sampled_host_available_bytes" in result
                         else result["evidence"]["minimum_sampled_host_available_bytes"], 40*GIB)
        self.assertTrue(all(v == "ready" for v in result["backend_gates"].values()))
        self.assertNotIn("cmdline", json.dumps(result))

    def test_actual_scope_refuses_even_valid_fixture(self):
        for scope in ("production", "actual-platform", "approved", True):
            with self.subTest(scope=scope), self.assertRaises(RuntimeError): self.run_pair(scope=scope)

    def test_missing_backend_is_not_implied_by_permission_or_sample(self):
        for section in ("boundary", "quota", "observer", "watchdog", "gpu"):
            values = pair(); values[1][section]["adapter"] = "unavailable"
            with self.subTest(section=section), self.assertRaises(ValueError): self.run_pair(values)

    def test_ownership_scope_and_parameter_limits_match_exactly(self):
        changes = [("boundary", "owner_uid", 0), ("boundary", "instance_id", "/user.slice"),
                   ("boundary", "exclusive", False), ("boundary", "hard_memory_bytes", GIB),
                   ("boundary", "hard_pid_limit", 9), ("quota", "owner_uid", True),
                   ("quota", "hard_bytes", 1), ("quota", "hard_entries", 1), ("quota", "exclusive", False)]
        for section, key, value in changes:
            values = pair(); values[1][section][key] = value
            with self.subTest(section=section,key=key), self.assertRaises(ValueError): self.run_pair(values)

    def test_incomplete_inspection_and_conflict_refuse_without_adopting_pids(self):
        changes = [("boundary", "membership_complete", False), ("boundary", "members", [1]*129),
                   ("boundary", "members", [42,42]), ("gpu", "ownership_complete", False),
                   ("gpu", "unreadable", 1), ("gpu", "conflicts", [987654]), ("gpu", "active_contexts", [987654])]
        with patch("os.kill") as kill:
            for section,key,value in changes:
                values=pair();values[1][section][key]=value
                with self.subTest(section=section,key=key),self.assertRaises(ValueError):self.run_pair(values)
        kill.assert_not_called()

    def test_unknown_fields_duplicate_keys_and_payload_envelope_fail(self):
        for key in ("cmdline", "environment", "approved", "argv"):
            values=pair();values[1][key]="private-untrusted-data"
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_pair(values)
        raw=encode(pair()[0])
        with self.assertRaises(ValueError):verify_preflight(policy(), [raw, raw[:-1]+b',"sequence":2}'],now=100.2)
        with self.assertRaises(ValueError):verify_preflight(policy(), [raw,b" "*(MAX_RESPONSE_BYTES+1)],now=100.2)
        values=pair();values[1]["observer"]["maximum_bytes"]=100
        with self.assertRaises(ValueError):self.run_pair(values)

    def test_freshness_challenge_sequence_and_ownership_races_refuse(self):
        for key, value in (("observed_at", 96.), ("observed_at", 101.), ("sequence", 1),
                           ("sequence", True), ("nonce", "d"*64), ("preparation_sha256", "e"*64)):
            values=pair();values[1][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):self.run_pair(values)
        values=pair();values[1]["boundary"]["members"]=[42]
        with self.assertRaises(ValueError):self.run_pair(values)
        with self.assertRaises(ValueError):verify_preflight(policy(),[encode(observation())]*2,now=100.2)

    def test_observer_deadline_and_watchdog_contracts_are_required_not_implemented(self):
        changes=[("observer", "timeout_enforced", False), ("observer", "maximum_seconds", 0.),
                 ("watchdog", "independent", False), ("watchdog", "armed", False),
                 ("watchdog", "cleanup_bounded", False), ("watchdog", "deadline_seconds", 61.)]
        for section,key,value in changes:
            values=pair();values[1][section][key]=value
            with self.subTest(section=section,key=key),self.assertRaises(ValueError):self.run_pair(values)
        values=pair();values[1]["observer_seconds"]=.6
        with self.assertRaises(ValueError):self.run_pair(values)
        raw=encode(observation()).replace(b'100.0',b'NaN')
        with self.assertRaises(ValueError):verify_preflight(policy(),[raw,encode(observation(2,100.1))],now=100.2)

    def test_host_reserve_is_sampled_and_cannot_be_downgraded(self):
        for section,key,value in (("host","available_bytes",24*GIB),("host","complete",False),
                                  ("host","available_bytes",True)):
            values=pair();values[0][section][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.run_pair(values)
        with self.assertRaises(ValueError):replace(policy(),minimum_host_available_bytes=24*GIB).validate()

    def test_policy_rejects_broad_target_bad_types_and_relaxed_response_bounds(self):
        for fields in ({"instance_id":"/"},{"instance_id":"user@1000.service"},{"owner_uid":True},
                       {"artifact_entries":True},{"maximum_observer_seconds":1.},
                       {"maximum_observation_age_seconds":30.},{"deadline_seconds":float("inf")},
                       {"stage_id":"arbitrary-shell"},{"preparation_sha256":"AA"}):
            with self.subTest(fields=fields),self.assertRaises(ValueError):replace(policy(),**fields).validate()

    def test_drain_requires_empty_inspectable_exact_owned_boundary_separate_from_exit(self):
        self.assertEqual(verify_fixture_drain(policy(),encode(drain()),actual_leader_exit=0)["signals_sent"],0)
        for key,value in (("remaining_members",[42]),("membership_complete",False),
                          ("cleanup_errors",["timeout"]),("boundary_removed",False),
                          ("instance_id","dongxi-stage-"+"d"*32),("owner_uid",True)):
            value_record=deepcopy(drain());value_record[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):verify_fixture_drain(policy(),encode(value_record),actual_leader_exit=0)
        for code in (7,-9,True):
            with self.subTest(code=code),self.assertRaises(ValueError):verify_fixture_drain(policy(),encode(drain()),actual_leader_exit=code)


if __name__ == "__main__":
    unittest.main()
