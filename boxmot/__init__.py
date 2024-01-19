# VJia Li 🔥 Nano Tracking

__version__ = '10.0.43'

from boxmot.tracker_zoo import create_tracker, get_tracker_config
from boxmot.trackers.bytetrack.byte_tracker import BYTETracker
from boxmot.trackers.nanotrack.nano_tracker import NanoTracker

TRACKERS = ['bytetrack', 'nanotrack']

__all__ = ("__version__",
           "BYTETracker","NanoTracker",
           "create_tracker", "get_tracker_config")
