"""Simulated subsystems. Every one is labelled SIMULATED in the UI and README.

Nothing in this package talks to a real PLC, BMC, scheduler, or GPU. The point is
to exercise the same state machines and protocol shapes the production adapters
would use, so the safety sequence is tested before hardware exists.
"""
