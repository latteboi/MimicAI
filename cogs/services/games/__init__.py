"""Pure game-rules modules.

Nothing in this package imports discord, touches the network or the disk, or is
async. Games are state machines driven from `services/game_service.py`; keeping the
rules free of I/O is what makes them unit-testable without a gateway connection, and
what stops a model ever being asked to compute something the engine owns.
"""
