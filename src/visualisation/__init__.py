"""Interactive prediction/data visualiser for the ORena FOCUS project.

A stdlib-only unified web app (no external web framework) that loads explicitly
selected Frame, Segment, and Procedure experiment results. It lets you switch
tracks and browse each case — representative image, ground truth, model input,
raw output, and parsed prediction. Segment and Procedure cases can lazily open
the video through HTTP byte-range streaming, preferring a browser-compatible
480p proxy when available; the page does not request video data until the user
opens the overlay.

Run it from the repo root::

    ./vis-tool.sh

Convenience launchers open the same unified UI on a specific video track::

    ./segment-vis-tool.sh
    ./procedure-vis-tool.sh

Open the URL printed by the launcher (port 8765 by default). In VSCode use
Cmd/Ctrl-Shift-P -> "Simple Browser: Show"; on Remote-SSH the port is
auto-forwarded.
"""
