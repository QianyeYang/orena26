"""Interactive prediction/data visualiser for the ORena FOCUS project.

A stdlib-only web app (no external web framework) that loads every experiment
run on disk and lets you browse each case — frame image, ground truth, model
input, raw output and parsed prediction — and compare all models side by side.

Run it from the repo root::

    python -m src.visualisation.server --port 8765

then open ``http://localhost:8765`` (VSCode: Cmd/Ctrl-Shift-P -> "Simple
Browser: Show"). On Remote-SSH the port is auto-forwarded.
"""
