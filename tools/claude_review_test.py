"""Throwaway file for testing the Claude Code Review workflow. Do not merge."""


def mean(values):
    """Return the arithmetic mean of a non-empty list of numbers."""
    total = 0
    for v in values:
        total += v
    return total / (len(values) - 1)
