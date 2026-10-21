"""Waiting for an agent that is unavailable for reasons that are not its own doing.

An agent is UNAVAILABLE when its CLI cannot run a turn at all: Claude Code is not logged in (or its key is invalid), a
usage, spend or rate limit is reached, the API is overloaded or the network is down, the CLI is missing; with a mailbox,
no broker is answering. Such a decision is not failed: it is kept and waited for, so that a run whose agent is down for
a while (a subscription switched, a laptop with the broker asleep) makes the same decisions as one whose agent never
was. The wait has no deadline unless `unavailable_timeout` (seconds) is given; it does not count against the decision's
own `timeout`, and the agent is told nothing about it. One line says when the wait starts (what is wrong and what to
do), a reminder follows every REMIND seconds, and one line says when it ends.

Failures of the agent's own turn (a crash inside the call, a time limit, a budget it spent) are not this: they keep the
handling of backends/claude.py (calls made again while the decision's time allows, then a failed decision).
"""

from __future__ import annotations

import datetime
import re
import time

RECHECK, RECHECK_MAX = 60.0, 300.0  # seconds before the first new try, and the longest gap between tries
REMIND = 600.0  # seconds between reminders that the run is still waiting
RESET_MARGIN = 30.0  # seconds after a reset time the message names before trying again

# Claude Code's words for the kinds of unavailability (2.1.30x; "You've hit your org's monthly spend limit · run
# /usage-credits ... · your session limit resets 1:10am (America/New_York)", "Not logged in · Please run /login",
# "API Error: 529 overloaded")
LOGIN = re.compile(
    r"not logged in|run /login|invalid api key|authentication[_ ]error|oauth token (?:has )?expired", re.I
)
# Claude Code's own words for a usage or spend limit: a call whose answer is one of these did not run, also when the CLI
# exits 0 with it as its answer
LIMIT = re.compile(
    r"hit your (?:[\w']+ ){0,3}limit|usage limit reached|spend limit reached|credit balance (?:is )?too low", re.I
)
RATE = re.compile(r"rate[_ ]limit|too many requests|\b429\b", re.I)  # in the error of a failed call only
DOWN = re.compile(
    r"overloaded|API Error: 5\d\d|\b529\b|ECONNREFUSED|ECONNRESET|ENOTFOUND|ETIMEDOUT|EAI_AGAIN|fetch failed"
    r"|socket hang up|connection (?:error|refused|reset|timed out)|network (?:error|is unreachable)|unable to connect",
    re.I,
)
HINTS = dict(
    login="log in on this machine (run `claude`, then /login); the decision goes on by itself",
    limit="wait for the limit to reset, raise it, or log in to another account; the decision goes on by itself",
    down="the API or the network is down; the decision goes on by itself once it is back",
    missing="install Claude Code or put it on PATH; the decision goes on by itself",
    broker="start a broker for this run; the decision goes on by itself once one answers",
)


def classify(texts):
    """('login' | 'limit' | 'down', the CLI's words) for the first short text that says the agent is unavailable, else
    None. Only short texts count, so that an agent's own long answer that mentions a rate limit is never taken for
    one."""
    texts = [str(t).strip() for t in texts if t and len(str(t).strip()) < 500]
    for kind, pattern in (("login", LOGIN), ("limit", LIMIT), ("limit", RATE), ("down", DOWN)):
        for t in texts:
            if pattern.search(t):
                return kind, " ".join(t.split())[:300]
    return None


_RESET = re.compile(r"resets\s+(\d{1,2})(?::(\d{2}))?\s*([ap]m)?(?:\s*\(([A-Za-z_]+/[A-Za-z_]+)\))?", re.I)


def seconds_to_reset(text, now=None):
    """Seconds until just after the reset time a limit message names ("resets 1:10am (America/New_York)", "resets
    3pm"), the next such time from now; None if it names none."""
    m = _RESET.search(str(text or ""))
    if not m:
        return None
    hour, minute, ampm, zone = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower(), m.group(4)
    if ampm == "pm" and hour != 12:
        hour += 12
    elif ampm == "am" and hour == 12:
        hour = 0
    if not (0 <= hour < 24 and 0 <= minute < 60):
        return None
    tz = None
    if zone:
        try:
            from zoneinfo import ZoneInfo

            tz = ZoneInfo(zone)
        except Exception:
            tz = None
    now_dt = datetime.datetime.fromtimestamp(time.time() if now is None else now, tz=tz)
    at = now_dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if at <= now_dt:
        at += datetime.timedelta(days=1)
    return (at - now_dt).total_seconds() + RESET_MARGIN


def _minutes(seconds):
    m = seconds / 60
    return f"{seconds:.0f} s" if seconds < 90 else f"{m:.0f} min" if m < 90 else f"{m / 60:.1f} h"


class Wait:
    """One wait for an unavailable agent. note(kind, reason) at each failed try: prints the first line, then a reminder
    every REMIND seconds (and when the reason changes), and returns the seconds to wait before the next try (None once
    `timeout` is over). end() prints the last line and returns the seconds waited. every: the caller checks again at
    this fixed interval (a mailbox's poll) instead of after 60 s, then at most every 5 min."""

    def __init__(self, say, what="Claude Code", training_waits=None, timeout=None, every=None):
        self.say, self.what, self.timeout, self.every = say, what, timeout, every
        self.training_waits = training_waits
        self.since = self.reminded = time.time()
        self.tries = 0
        self._retry_wait = min(RECHECK, RECHECK_MAX)
        self.kind = self.reason = None

    @property
    def seconds(self):
        return time.time() - self.since

    def over(self):
        return self.timeout is not None and self.seconds >= self.timeout

    def note(self, kind, reason):
        """Seconds to wait before the next try, or None when unavailable_timeout is over (the decision then fails)."""
        self.tries += 1
        changed = (kind, reason) != (self.kind, self.reason)
        self.kind, self.reason = kind, reason
        if self.tries == 1:
            run = (
                "training waits for the agent"
                if self.training_waits
                else "training continues with the current programs"
                if self.training_waits is False
                else "the request is kept"
            )
            limit = "" if self.timeout is None else f", for at most {_minutes(self.timeout)} (unavailable_timeout)"
            again = (
                f"checks again every {self.every:g} s"
                if self.every is not None
                else f"tries again in {RECHECK:.0f} s, then at most every {RECHECK_MAX / 60:.0f} min"
            )
            self.say(
                f"{self.what} is unavailable ({reason}); {run} and {again}{limit}. To go on: "
                f"{HINTS.get(kind, HINTS['down'])}"
            )
        elif time.time() - self.reminded >= REMIND or (changed and kind != "down"):
            self.reminded = time.time()
            self.say(f"still waiting for {self.what} ({_minutes(self.seconds)} so far): {reason}")
        if self.over():
            self.say(f"{self.what} stayed unavailable for {_minutes(self.seconds)} (unavailable_timeout); giving up")
            return None
        wait = self._retry_wait
        self._retry_wait = min(2 * wait, RECHECK_MAX)
        reset = seconds_to_reset(reason)
        if reset is not None:
            wait = min(wait, reset)
        if self.timeout is not None:
            wait = min(wait, max(0.0, self.timeout - self.seconds))
        return wait

    def end(self):
        waited = self.seconds
        self.say(f"{self.what} is available again after {_minutes(waited)}; the decision goes on")
        return waited
