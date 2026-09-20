"""Test scenarios: a bug to plant, and the issue that reports it.

More than one exists on purpose. A single scenario proves the pipeline runs; it does not show
that the agent reads the *issue* rather than pattern-matching one familiar fixture. Two different
bugs, in different files, with different shapes, is the cheapest way to see that for yourself.

Each scenario is deliberately:

* **small** - one or two lines to change, so a failure is the agent's reasoning and not the size
  of the task;
* **covered by a failing test**, so success is objective rather than a matter of opinion;
* **documented**, so the intended behaviour is discoverable from the code, the way a real bug
  report and a real docstring would be.

What they are *not* is representative of real work. See the honesty note in the README: one or
two solved tasks is not a success rate.
"""

from dataclasses import dataclass, field

# An empty root conftest.py is what makes `pytest -q` work from the repository root. The console
# script does not put the working directory on sys.path, so without it `from calculator import`
# fails during collection.
CONFTEST = '"""Makes the repository root importable for pytest."""\n'

README = """\
# Agent sandbox

A throwaway repository for exercising an AI software engineering agent end to end.

Run tests with `pytest -q`.
"""


@dataclass
class Scenario:
    key: str
    summary: str

    #: Files written to the repository's default branch, path -> content.
    files: dict[str, str]

    issue_title: str
    issue_body: str

    #: Which tests should fail before the fix, for your own sanity check.
    expected_failures: list[str] = field(default_factory=list)


# --------------------------------------------------------------- 1. divide by zero

DIVIDE_SOURCE = '''\
"""Arithmetic helpers used by the billing calculator."""


def divide(numerator: float, denominator: float) -> float:
    """Divides two numbers.

    Raises:
        ValueError: if the denominator is zero.
    """
    return numerator / denominator


def average(values: list[float]) -> float:
    """Returns the mean of a list of numbers."""
    return divide(sum(values), len(values))
'''

DIVIDE_TESTS = '''\
import pytest

from calculator import average, divide


def test_divide_works():
    assert divide(10, 2) == 5


def test_average_works():
    assert average([2, 4, 6]) == 4


def test_divide_by_zero_raises_value_error():
    with pytest.raises(ValueError):
        divide(1, 0)


def test_average_of_empty_list_raises_value_error():
    with pytest.raises(ValueError):
        average([])
'''

DIVIDE = Scenario(
    key="divide-by-zero",
    summary="A missing guard clause. The docstring promises a ValueError the code never raises.",
    files={
        "calculator.py": DIVIDE_SOURCE,
        "tests/test_calculator.py": DIVIDE_TESTS,
        "conftest.py": CONFTEST,
        "requirements.txt": "pytest\n",
        "README.md": README,
    },
    issue_title="average() and divide() raise ZeroDivisionError instead of ValueError",
    issue_body="""\
Calling `average([])` crashes with `ZeroDivisionError: division by zero`.

The docstring on `divide` says it raises `ValueError` when the denominator is zero, but there is
no check, so the raw `ZeroDivisionError` reaches the caller. Our API layer maps `ValueError` to
a 422 and anything else to a 500, so an empty list currently returns a 500.

Expected: `divide(1, 0)` and `average([])` both raise `ValueError` with a clear message.
Actual: both raise `ZeroDivisionError`.

The tests in `tests/test_calculator.py` cover this and are currently failing.
""",
    expected_failures=[
        "test_divide_by_zero_raises_value_error",
        "test_average_of_empty_list_raises_value_error",
    ],
)


# ------------------------------------------------------- 2. boundary comparison

INVENTORY_SOURCE = '''\
"""Stock level rules for the warehouse service."""


def restock_needed(quantity: int, threshold: int) -> bool:
    """Whether an item needs restocking.

    An item needs restocking once its quantity has fallen to **or below** the threshold.
    """
    return quantity < threshold


def items_to_restock(stock: dict[str, int], threshold: int) -> list[str]:
    """Names of every item that needs restocking, in alphabetical order."""
    return sorted(name for name, quantity in stock.items() if restock_needed(quantity, threshold))
'''

INVENTORY_TESTS = '''\
from inventory import items_to_restock, restock_needed


def test_below_threshold_needs_restock():
    assert restock_needed(2, 5) is True


def test_above_threshold_does_not_need_restock():
    assert restock_needed(9, 5) is False


def test_exactly_at_threshold_needs_restock():
    assert restock_needed(5, 5) is True


def test_item_at_threshold_is_listed():
    stock = {"bolts": 5, "nuts": 9, "washers": 1}

    assert items_to_restock(stock, 5) == ["bolts", "washers"]
'''

INVENTORY = Scenario(
    key="boundary",
    summary="An off-by-one boundary condition. `<` where the documented rule needs `<=`.",
    files={
        "inventory.py": INVENTORY_SOURCE,
        "tests/test_inventory.py": INVENTORY_TESTS,
        "conftest.py": CONFTEST,
        "requirements.txt": "pytest\n",
        "README.md": README,
    },
    issue_title="Items sitting exactly at the restock threshold are never restocked",
    issue_body="""\
We set the threshold for bolts to 5. Stock dropped to exactly 5 and no restock was triggered, so
we ran out two days later.

The rule we want, and the one `restock_needed` documents, is "at or below the threshold". The
comparison appears to be strict, so an item resting exactly on the threshold is treated as fine
until it drops to 4.

Expected: `restock_needed(5, 5)` is `True`, and `items_to_restock({"bolts": 5, ...}, 5)` includes
`bolts`.
Actual: both treat 5 as healthy stock.

`tests/test_inventory.py` covers this and two of its cases are failing.
""",
    expected_failures=[
        "test_exactly_at_threshold_needs_restock",
        "test_item_at_threshold_is_listed",
    ],
)


SCENARIOS: dict[str, Scenario] = {DIVIDE.key: DIVIDE, INVENTORY.key: INVENTORY}

DEFAULT_SCENARIO = DIVIDE.key


def get(key: str) -> Scenario:
    if key not in SCENARIOS:
        available = ", ".join(sorted(SCENARIOS))
        raise SystemExit(f"Unknown scenario '{key}'. Available: {available}")

    return SCENARIOS[key]
