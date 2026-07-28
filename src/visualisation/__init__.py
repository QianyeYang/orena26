"""Interactive prediction/data visualiser for the ORena FOCUS project.

A stdlib-only web app (no external web framework) that loads explicitly selected
experiment results and lets you browse each case — frame image, ground truth,
model input, raw output and parsed prediction.

Run it from the repo root::

    ./vis-tool.sh

then open ``http://localhost:8765`` (VSCode: Cmd/Ctrl-Shift-P -> "Simple
Browser: Show"). On Remote-SSH the port is auto-forwarded.
"""
