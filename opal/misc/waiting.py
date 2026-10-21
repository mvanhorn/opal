"""blocking="auto": training waits at a decision only when the agent's answer would otherwise take effect more than
max_lag_updates updates late. The estimate is the median of the agent's last few answer times divided by the time per
update: the median over the last few check intervals, with the pauses for the agent left out (a median, so an
interval with an evaluation or a checkpoint in it does not shrink the estimate). Before the first answer, or before the
time per update is measured, training waits. Rank 0 decides; the existing broadcast at each check tells the others.

Why: without waiting, an answer lands (answer time / time per update) updates after the state the agent saw, and no
new decision is asked while one is in flight, so on a fast run most scheduled decisions are never asked."""

from __future__ import annotations

import statistics
from collections import deque

WINDOW = 5  # answers, and check intervals, in the medians


class AutoWait:
    def __init__(self, max_lag_updates):
        if isinstance(max_lag_updates, bool) or not isinstance(max_lag_updates, (int, float)) or max_lag_updates < 0:
            raise ValueError(f"opal: max_lag_updates must be a number >= 0, got {max_lag_updates!r}")
        self.max_lag = max_lag_updates
        self.answers = deque(maxlen=WINDOW)
        self.ms_per_update = None
        self._intervals = deque(maxlen=WINDOW)
        self._t = self._u = None
        self._paused = 0.0

    def tick(self, u, now):
        """At each check step (rank 0): time per update since the previous check, without the pauses in between."""
        if self._t is not None and u > self._u:
            self._intervals.append(1000 * max(0.0, now - self._t - self._paused) / (u - self._u))
            self.ms_per_update = statistics.median(self._intervals)
        self._t, self._u, self._paused = now, u, 0.0

    def paused(self, seconds):
        self._paused += seconds

    def answered(self, resp):
        """An answer's time, unless the backend failed or stopped waiting (those say nothing about the agent)."""
        if resp and not (resp.get("backend_error") or resp.get("gave_up") or resp.get("late")):
            if resp.get("seconds") is not None:
                self.answers.append(float(resp["seconds"]))

    def decide(self):
        """Whether training waits at the decision being asked, and the record of why."""
        if not self.answers or not self.ms_per_update:
            why = "no answer time measured yet" if not self.answers else "no time per update measured yet"
            return dict(wait="waited", wait_reason=why, predicted_lag_updates=None)
        s = statistics.median(self.answers)
        lag = 1000 * s / self.ms_per_update
        block = lag > self.max_lag
        return dict(
            wait="waited" if block else "did not wait",
            wait_reason=f"predicted lag {lag:.0f} updates {'>' if block else '<='} {self.max_lag:g} "
            f"(median answer {s:.1f} s, {self.ms_per_update:.1f} ms per update)",
            predicted_lag_updates=round(lag, 1),
        )
