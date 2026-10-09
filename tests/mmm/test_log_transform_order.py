import numpy as np
import pandas as pd
import pytensor
import pytensor.tensor as pt
import pymc as pm

from pymc_marketing.mmm import MMM, NoAdstock, NoSaturation


def _find_ops(var, predicate):
    """Traverse the computational graph and collect nodes matching predicate safely."""
    # Use io_toposort to get a clean ordering of nodes leading to var
    inputs = pytensor.graph.basic.graph_inputs([var])
    nodes = pytensor.graph.basic.io_toposort(inputs, [var])
    matches = []
    for node in nodes:
        try:
            if predicate(node):
                matches.append(node)
        except Exception:
            # Ignore nodes that don't have the queried attributes
            pass
    return matches


def _op_matches(node, keywords: list[str]) -> bool:
    """Check whether a node's op or scalar_op class/name contains any keyword (case-insensitive)."""
    op = getattr(node, "op", None)
    if op is None:
        return False
    names: list[str] = []
    # Class name of the Op (e.g., Elemwise)
    names.append(op.__class__.__name__.lower())
    # Optional op.name
    op_name = getattr(op, "name", None)
    if isinstance(op_name, str):
        names.append(op_name.lower())
    # For Elemwise ops, check the scalar_op (e.g., Log, Mul)
    scalar_op = getattr(op, "scalar_op", None)
    if scalar_op is not None:
        names.append(scalar_op.__class__.__name__.lower())
    return any(any(k in n for n in names) for k in keywords)


def test_channels_log_applied_before_nosaturation_beta():
    # Minimal synthetic data
    n = 5
    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=n, freq="D"),
            "x1": np.linspace(0.0, 10.0, n),
            "y": np.linspace(1.0, 2.0, n),
        }
    )

    # Model with NoAdstock (identity) and NoSaturation (beta * x)
    mmm = MMM(
        date_column="date",
        channel_columns=["x1"],
        adstock=NoAdstock(l_max=1),
        saturation=NoSaturation(),
        log_transform_vars=["channels"],  # apply log to channels
        adstock_first=True,
    )

    X = df[["date", "x1"]]
    y = df["y"].values

    # Build the model (no sampling needed for this structural test)
    mmm.build_model(X, y)

    # Extract a scalar element from channel_contribution for graph inspection
    cc = mmm.model["channel_contribution"][0, 0]

    # Find Log and Mul nodes in the graph
    log_nodes = _find_ops(cc, lambda n: _op_matches(n, ["log"]))
    mul_nodes = _find_ops(cc, lambda n: _op_matches(n, ["mul"]))

    # Sanity: there should be at least one log and one mul (from beta * x)
    assert len(log_nodes) >= 1, "Expected a log op in the channel_contribution graph"
    assert len(mul_nodes) >= 1, "Expected a mul op (beta * x) in the channel_contribution graph"

    # Check for the WRONG pattern: log(...) applied to a product (i.e., input of log is a mul)
    wrong_pattern = False
    for ln in log_nodes:
        # If any input to the log node is the output of a mul op, it's the wrong order
        for inp in ln.inputs:
            if hasattr(inp, "owner") and inp.owner is not None:
                op_name = inp.owner.op.__class__.__name__.lower()
                if "mul" in op_name:
                    wrong_pattern = True
                    break
        if wrong_pattern:
            break

    assert not wrong_pattern, "Found log applied to a product (log(beta * x + 1)) — wrong ordering"

    # Check for the RIGHT pattern: a mul whose one input is the output of a log
    right_pattern = False
    for mn in mul_nodes:
        for inp in mn.inputs:
            if hasattr(inp, "owner") and inp.owner is not None:
                op_name = inp.owner.op.__class__.__name__.lower()
                if "log" in op_name:
                    right_pattern = True
                    break
        if right_pattern:
            break

    assert right_pattern, "Expected beta * log(adstock(x) + 1) pattern (mul of log output)"
