"""
Timestamp-based ring buffer for multi-source frame aggregation.

This module provides a generic TimestampBuffer that stores frames indexed by timestamps
and supports nearest-neighbor matching for updating fields from asynchronous sources.
"""

from __future__ import annotations

import logging
from collections import OrderedDict, deque
from copy import deepcopy
from dataclasses import dataclass
from typing import TypeVar, Generic, Optional, Callable, Dict, Any, Tuple

_logger = logging.getLogger(__name__)

# Generic type for frame data
T = TypeVar("T")


@dataclass
class BufferStats:
    """Statistics for TimestampBuffer operations."""

    total_adds: int = 0
    total_updates: int = 0
    successful_matches: int = 0
    failed_matches: int = 0
    avg_match_delay_ms: float = 0.0

    @property
    def match_success_rate(self) -> float:
        """Return match success rate as a ratio [0, 1]."""
        total = self.successful_matches + self.failed_matches
        return self.successful_matches / total if total > 0 else 0.0


class TimestampBuffer(Generic[T]):
    """
    Generic timestamp-based ring buffer with nearest-neighbor matching.

    This buffer stores frames indexed by timestamps and supports updating fields
    from asynchronous sources using nearest-neighbor timestamp matching.

    Parameters
    ----------
    max_frames : int
        Maximum number of frames to store (default: 10, ~0.5s at 15fps).
    match_threshold : float
        Maximum time difference (seconds) for nearest-neighbor matching (default: 0.066s = 66ms).
    stats_window : int
        Number of recent matches to track for delay statistics (default: 100).

    Example
    -------
    >>> from teleop_realman.comm.types import FrameData
    >>> buffer = TimestampBuffer[FrameData](max_frames=10, match_threshold=0.066)
    >>> frame = FrameData(timestamp=1.0, image_list=[img1, img2], depth_list=[d1, d2], mocap={})
    >>> buffer.add(1.0, frame)
    >>> buffer.update_field(1.002, "bbox", {"lh": [...], "rh": [...]})  # matches ts=1.0
    >>> latest = buffer.get_latest()
    """

    def __init__(
        self,
        max_frame: int = 10,
        match_threshold: float = 0.066,
        stats_window: int = 100,
    ):
        self._buffer: OrderedDict[float, T] = OrderedDict()
        self._max_frame = max_frame
        self._match_threshold = match_threshold

        # Statistics
        self._stats_window = stats_window
        self._match_delays: deque[float] = deque(maxlen=stats_window)
        self._stats = BufferStats()

    # ---- Core Operations ----

    def add(self, timestamp: float, frame: T) -> None:
        """
        Add a new frame to the buffer.

        If buffer is full, the oldest frame is removed.

        Parameters
        ----------
        timestamp : float
            Synchronized timestamp for the frame.
        frame : T
            Frame data object.
        """
        if len(self._buffer) >= self._max_frame:
            self._buffer.popitem(last=False)  # Remove oldest

        self._buffer[timestamp] = frame
        self._stats.total_adds += 1

    def update_field(
        self,
        timestamp: float,
        field_name: str,
        value: Any,
        setter: Optional[Callable[[T, Any], None]] = None,
    ) -> bool:
        """
        Update a field in the nearest matching frame.

        Uses nearest-neighbor timestamp matching within the threshold.

        Parameters
        ----------
        timestamp : float
            Timestamp of the source data.
        field_name : str
            Name of the field to update (used with setattr or dict-like access).
        value : Any
            Value to set.
        setter : Optional[Callable[[T, Any], None]]
            Custom setter function. If None, uses setattr(frame, field_name, value).

        Returns
        -------
        bool
            True if a matching frame was found and updated, False otherwise.
        """
        self._stats.total_updates += 1

        closest_ts, delay = self._find_closest_timestamp(timestamp)
        if closest_ts is None:
            self._stats.failed_matches += 1
            _logger.debug(f"No match for ts={timestamp:.4f}, threshold={self._match_threshold:.4f}s")
            return False

        # Update the frame
        frame = self._buffer[closest_ts]
        if setter is not None:
            setter(frame, value)
        else:
            # Try dict-like access first, fallback to setattr
            if hasattr(frame, "__setitem__"):
                frame[field_name] = value
            else:
                setattr(frame, field_name, value)

        # Record statistics
        self._stats.successful_matches += 1
        self._match_delays.append(delay * 1000)  # Convert to ms
        self._update_avg_delay()

        return True

    def get(self, timestamp: float) -> Optional[T]:
        """
        Get frame by exact timestamp.

        Parameters
        ----------
        timestamp : float
            Exact timestamp to look up.

        Returns
        -------
        Optional[T]
            Frame if found, None otherwise.
        """
        return self._buffer.get(timestamp)

    def get_nearest(self, timestamp: float) -> Optional[Tuple[float, T]]:
        """
        Get nearest frame within threshold.

        Parameters
        ----------
        timestamp : float
            Target timestamp.

        Returns
        -------
        Optional[Tuple[float, T]]
            Tuple of (matched_timestamp, frame) if found, None otherwise.
        """
        closest_ts, _ = self._find_closest_timestamp(timestamp)
        if closest_ts is None:
            return None
        return (closest_ts, self._buffer[closest_ts])

    def get_latest(self) -> Optional[T]:
        """
        Get the most recent frame (reference).

        Returns
        -------
        Optional[T]
            Latest frame or None if buffer is empty.
        """
        if not self._buffer:
            return None
        return next(reversed(self._buffer.values()))

    def get_latest_with_copy(self) -> Optional[T]:
        """
        Get a deep copy of the most recent frame (for visualization).

        Returns
        -------
        Optional[T]
            Deep copy of latest frame or None if buffer is empty.
        """
        frame = self.get_latest()
        if frame is None:
            return None
        return deepcopy(frame)

    def get_latest_with_field(
        self,
        field_name: str,
        checker: Optional[Callable[[T], bool]] = None,
    ) -> Optional[T]:
        """
        Get the most recent frame that has a specific field set (reference).

        Parameters
        ----------
        field_name : str
            Field name to check for non-None value.
        checker : Optional[Callable[[T], bool]]
            Custom checker function. If None, checks getattr(frame, field_name) is not None.

        Returns
        -------
        Optional[T]
            Frame with the field set, or latest frame if none found, or None if buffer empty.
        """
        if not self._buffer:
            return None

        # Search from newest to oldest
        for frame in reversed(self._buffer.values()):
            if checker is not None:
                if checker(frame):
                    return frame
            else:
                # Try dict-like access first, fallback to getattr
                if hasattr(frame, "__getitem__"):
                    try:
                        if frame[field_name] is not None:
                            return frame
                    except (KeyError, TypeError):
                        if getattr(frame, field_name, None) is not None:
                            return frame
                elif getattr(frame, field_name, None) is not None:
                    return frame

        # Fallback to latest frame
        return self.get_latest()

    def get_latest_with_field_copy(
        self,
        field_name: str,
        checker: Optional[Callable[[T], bool]] = None,
    ) -> Optional[T]:
        """
        Get a deep copy of the most recent frame with a specific field (for visualization).

        Parameters
        ----------
        field_name : str
            Field name to check.
        checker : Optional[Callable[[T], bool]]
            Custom checker function.

        Returns
        -------
        Optional[T]
            Deep copy of frame or None.
        """
        frame = self.get_latest_with_field(field_name, checker)
        if frame is None:
            return None
        return deepcopy(frame)

    # ---- Statistics ----

    def get_stats(self) -> BufferStats:
        """
        Get buffer statistics.

        Returns
        -------
        BufferStats
            Current statistics including match rates and delays.
        """
        return self._stats

    def get_avg_delay_ms(self) -> float:
        """
        Get average match delay in milliseconds.

        Returns
        -------
        float
            Average delay of recent matches, or 0.0 if no matches recorded.
        """
        return self._stats.avg_match_delay_ms

    # ---- Buffer State ----

    def __len__(self) -> int:
        """Return number of frames in buffer."""
        return len(self._buffer)

    def __contains__(self, timestamp: float) -> bool:
        """Check if exact timestamp exists in buffer."""
        return timestamp in self._buffer

    def clear(self) -> None:
        """Clear all frames from buffer."""
        self._buffer.clear()

    def timestamps(self) -> list[float]:
        """Return list of timestamps in buffer (oldest to newest)."""
        return list(self._buffer.keys())

    @property
    def oldest_timestamp(self) -> Optional[float]:
        """Return oldest timestamp in buffer."""
        if not self._buffer:
            return None
        return next(iter(self._buffer.keys()))

    @property
    def newest_timestamp(self) -> Optional[float]:
        """Return newest timestamp in buffer."""
        if not self._buffer:
            return None
        return next(reversed(self._buffer.keys()))

    # ---- Private Methods ----

    def _find_closest_timestamp(self, timestamp: float) -> Tuple[Optional[float], float]:
        """
        Find the closest timestamp within threshold.

        Parameters
        ----------
        timestamp : float
            Target timestamp.

        Returns
        -------
        Tuple[Optional[float], float]
            (closest_timestamp, delay) or (None, inf) if no match.
        """
        if not self._buffer:
            return None, float("inf")

        closest_ts = min(self._buffer.keys(), key=lambda ts: abs(ts - timestamp))
        delay = abs(closest_ts - timestamp)

        if delay <= self._match_threshold:
            return closest_ts, delay
        return None, delay

    def _update_avg_delay(self) -> None:
        """Update average match delay from recent samples."""
        if self._match_delays:
            self._stats.avg_match_delay_ms = sum(self._match_delays) / len(self._match_delays)
        else:
            self._stats.avg_match_delay_ms = 0.0
