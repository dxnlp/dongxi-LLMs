"""Original scaling-fit microscope using explicitly authored simulations.

6ND is an approximate training-compute coordinate, not measured hardware work.
No models, hosted API, old training checkpoints or datasets are loaded here.
"""
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np


class ScalingFitError(ValueError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or []


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ScalingFitError(f"{name} must be finite and positive")
    return value


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def loss_surface(parameters, tokens, *, floor=2., model_coefficient=20., data_coefficient=20.):
    _positive(parameters, "parameters"); _positive(tokens, "tokens")
    return floor + model_coefficient/math.sqrt(parameters) + data_coefficient/math.sqrt(tokens)


def simulated_rows(*, budgets=(6e6, 24e6, 96e6, 384e6), seeds=(6601, 6602, 6603), noise=0.):
    if not math.isfinite(noise) or noise < 0 or not seeds:
        raise ValueError("Nonnegative finite noise and nonempty seeds required")
    rows = []
    for c in budgets:
        _positive(c, "budget"); optimum = math.sqrt(c/6)
        for factor in (.5, .75, 1., 1.5, 2.):
            n = round(optimum*factor); d = c/(6*n)
            for seed in seeds:
                rng = np.random.default_rng(np.random.SeedSequence([seed, int(c), n]))
                rows.append(dict(mode="authored-simulation", architecture="surface-A",
                                 data_policy="unique-token-coordinate", compute_budget=float(c),
                                 parameters=n, presented_tokens=d, unique_tokens=d, seed=seed,
                                 loss=loss_surface(n, d)+float(rng.normal(0, noise))))
    return rows


@dataclass(frozen=True)
class ScalingLaw:
    log_parameter_intercept: float
    parameter_exponent: float
    log_excess_loss_intercept: float
    excess_loss_exponent: float
    assumed_loss_floor: float
    calibration_sha256: str
    observed_budgets: tuple
    optima: tuple
    log_parameter_residuals: tuple
    contract: tuple

    def predict(self, compute):
        _positive(compute, "prediction budget")
        n = math.exp(self.log_parameter_intercept+self.parameter_exponent*math.log(compute))
        loss = self.assumed_loss_floor+math.exp(self.log_excess_loss_intercept+self.excess_loss_exponent*math.log(compute))
        return dict(compute_budget=compute, parameters=n, presented_tokens=compute/(6*n), loss=loss)


def fit_isoflops(rows, *, assumed_loss_floor=2.):
    rows = list(rows)
    if not rows:
        raise ScalingFitError("No calibration rows")
    if not math.isfinite(assumed_loss_floor):
        raise ScalingFitError("Finite assumed loss floor required")
    contracts = {(r.get("mode"), r.get("architecture"), r.get("data_policy")) for r in rows}
    if len(contracts) != 1 or any(not isinstance(x, str) or not x for x in next(iter(contracts))):
        raise ScalingFitError("Cannot silently pool different mode/architecture/data policies")
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        c = _positive(row["compute_budget"], "compute budget")
        n = _positive(row["parameters"], "parameters")
        loss = _positive(row["loss"], "loss")
        grouped[c][n].append(loss)
    if len(grouped) < 3:
        raise ScalingFitError("At least three distinct calibration budgets required")
    optima = []
    for c, sizes in sorted(grouped.items()):
        if len(sizes) < 3:
            raise ScalingFitError("At least three sizes per budget required", [dict(budget=c)])
        ns = sorted(sizes); means = {n:float(np.mean(sizes[n])) for n in ns}
        best = min(ns, key=lambda n: (means[n], n))
        if best in (ns[0], ns[-1]):
            raise ScalingFitError("Minimum at grid boundary; optimum not bracketed", [dict(budget=c, best=best, sizes=ns)])
        if means[best] <= assumed_loss_floor:
            raise ScalingFitError("Loss must exceed the explicitly assumed floor")
        optima.append(dict(compute_budget=c, parameters=best, loss=means[best]))
    logc = np.log([r["compute_budget"] for r in optima]); design = np.column_stack([np.ones(len(logc)), logc])
    logn = np.log([r["parameters"] for r in optima]); ncoef = np.linalg.lstsq(design, logn, rcond=None)[0]
    logloss = np.log([r["loss"]-assumed_loss_floor for r in optima]); lcoef = np.linalg.lstsq(design, logloss, rcond=None)[0]
    return ScalingLaw(float(ncoef[0]), float(ncoef[1]), float(lcoef[0]), float(lcoef[1]),
                      assumed_loss_floor, digest(rows), tuple(sorted(grouped)), tuple(optima),
                      tuple((logn-design@ncoef).tolist()), next(iter(contracts)))


def freeze_prediction(law, budget):
    if budget in law.observed_budgets:
        raise ScalingFitError("Prediction budget was used for calibration")
    frozen = dict(status="prediction-frozen-before-holdout", law=asdict(law),
                prediction=law.predict(budget), calibration_sha256=law.calibration_sha256,
                assumption="Loss-floor known by authored design; no claim it is known for a real LM")
    frozen["prediction_sha256"] = digest(frozen)
    return frozen


def evaluate_holdout(frozen, rows):
    protected = {key:value for key, value in frozen.items() if key != "prediction_sha256"}
    if frozen.get("prediction_sha256") != digest(protected):
        raise ScalingFitError("Frozen prediction identity changed")
    rows = list(rows); target = frozen["prediction"]["compute_budget"]
    if not rows or any(r["compute_budget"] != target for r in rows):
        raise ScalingFitError("The entire held-out budget must match the frozen target")
    if target in frozen["law"]["observed_budgets"]:
        raise ScalingFitError("Prediction budget was used for calibration")
    contract = tuple(frozen["law"]["contract"])
    if any(tuple(r.get(key) for key in ("mode", "architecture", "data_policy")) != contract for r in rows):
        raise ScalingFitError("Held-out mode/architecture/data policy differs from calibration")
    losses = defaultdict(list)
    for row in rows:
        losses[_positive(row["parameters"], "parameters")].append(_positive(row["loss"], "loss"))
    means = {n:float(np.mean(v)) for n, v in sorted(losses.items())}
    best = min(means, key=means.get); predicted = frozen["prediction"]
    return dict(heldout_sha256=digest(rows), entire_budget_rows=rows,
                mean_loss_by_size=means, observed_grid_optimum=best,
                parameter_relative_error=(predicted["parameters"]-best)/best,
                optimal_loss_error=predicted["loss"]-means[best],
                boundary_minimum=best in (min(means), max(means)),
                boundary="Grid optimum only; full holdout retained; no large-model or hardware prediction")


def bootstrap_predictions(rows, budget, *, draws=100, seed=6604, assumed_loss_floor=2.):
    if type(draws) is not int or draws <= 0:
        raise ValueError("Positive integer draws required")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["compute_budget"], row["parameters"])].append(row)
    rng = np.random.default_rng(seed); predictions, failures = [], []
    for draw in range(draws):
        sampled = [group[int(index)] for group in groups.values() for index in rng.integers(len(group), size=len(group))]
        try:
            predictions.append(fit_isoflops(sampled, assumed_loss_floor=assumed_loss_floor).predict(budget))
        except ScalingFitError as error:
            failures.append(dict(draw=draw, error=str(error), diagnostics=error.diagnostics))
    interval = None if not predictions else np.quantile([r["parameters"] for r in predictions], [.025, .975]).tolist()
    return dict(predictions=predictions, failed_fits=failures, parameter_percentile_interval=interval,
                boundary="Conditional resampling of authored replicate noise; excludes functional-form/extrapolation/hardware uncertainty")
