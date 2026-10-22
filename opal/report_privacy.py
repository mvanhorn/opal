"""Publication filtering for report text only; never changes the run's source records."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

from .backends._privacy import redact_secrets


# Match locations before HTML escaping, including paths inside quoted code/JSON.
_REMOTE = re.compile(r"\b[a-zA-Z][\w+.-]*://[^\s<>\"'`]+|\barn:aws[^\s<>\"'`]+")
_HOME = re.compile(r"(?:/Users/|/home/|/homes/|[A-Za-z]:[\\/]Users[\\/])([^/\\\n\"'`]+)[/\\]")
_PATH = re.compile(
    r"(?<![\w./])(?:/(?:Users|home|homes)/[^/\n\"'`]+/[^\s<>\"'`|;,(){}\[\]]*|"
    r"(?:[A-Za-z]:[\\/]|~[\w.-]*[/\\]|/)[^\s<>\"'`|;,(){}\[\]]+)"
)
_LOGIN = re.compile(r"\b([\w.+-]+)@([\w.-]+)\b")
_UNC = re.compile(r'\\\\([^\s\\/]+)[\\/][^\s<>"\'`]+')
_LABEL = re.compile(
    r"\b(?:hostname|host_name|host|remote_host|username|user_name|user|login|bucket|bucket_name|s3_bucket)\s*[:=]\s*[\"']?([\w.@+-]+)",
    re.I,
)
_HOST = re.compile(
    r"\b(?:[\w-]+\.)+(?:com|edu|net|org|io|local|internal|lan|ac\.[a-z]{2})\b|"
    r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])",
    re.I,
)
_IDENTITY_KEYS = {
    "host",
    "hostname",
    "remotehost",
    "sshhost",
    "user",
    "username",
    "userfullname",
    "login",
    "owner",
    "author",
    "bucket",
    "bucketname",
    "s3bucket",
}
_RUN_KEYS = {"run_dir", "run_folder", "opal_run"}
_RUN_CHILD = re.compile(r"[/\\](?:workspace|decisions|programs)(?:[/\\]|$)")


class Privacy:
    """Remember identities wherever they occur in the rendered inputs, then remove repeated mentions too.

    Paths within this run (including its original location before copying) become relative paths. Other
    locations, remote authorities, login names and explicitly recorded user/host/bucket identifiers are omitted.
    The renderer uses this on every string, including dictionary keys, full messages, code and plot labels.
    """

    def __init__(self, run, records):
        self.roots = {str(Path(run).absolute()).rstrip("/"), str(Path(run).resolve()).rstrip("/")}
        self.identities = set()
        self._learn(records)
        self.roots = {root for root in self.roots if len(root) > 1}
        self.run_path = re.compile(
            r"(?<![\w./])(?:"
            + "|".join(re.escape(root) for root in sorted(self.roots, key=len, reverse=True))
            + r")(?=$|[/\\\s<>\"'`|;,(){}\[\]])(?:[/\\][^\s<>\"'`|;,(){}\[\]]*)?"
        )
        # A username discovered in one file must disappear from all other displayed files as well.
        values = sorted((s for s in self.identities if s), key=len, reverse=True)
        self.identity = (
            re.compile(r"(?<![\w])(?:" + "|".join(re.escape(s) for s in values) + r")(?![\w])", re.I)
            if values
            else None
        )

    def _learn(self, obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if re.sub(r"[_-]", "", str(key).lower()) in _IDENTITY_KEYS and isinstance(value, str):
                    self.identities.add(value)
                if (
                    str(key).lower() in _RUN_KEYS
                    and isinstance(value, str)
                    and (value.startswith("/") or re.match(r"[A-Za-z]:[\\/]", value))
                ):
                    self.roots.add(value.rstrip("/\\"))
                if str(key).lower() == "cwd" and isinstance(value, str):
                    child = _RUN_CHILD.search(value)
                    if child:
                        self.roots.add(value[: child.start()])
                self._learn(str(key))
                self._learn(value)
        elif isinstance(obj, (list, tuple)):
            for value in obj:
                self._learn(value)
        elif isinstance(obj, str):
            for match in re.finditer(r"~([\w.-]+)[/\\]", obj):
                self.identities.add(match[1])
            for match in _UNC.finditer(obj):
                self.identities.add(match[1])
            for match in _REMOTE.finditer(obj):
                if "://" in match[0]:
                    authority = match[0].split("://", 1)[1].split("/", 1)[0]
                    self.identities.add(authority)
                    try:
                        location = urlsplit(match[0])
                        self.identities.update(unquote(v) for v in (location.hostname, location.username) if v)
                        bucket = re.match(r"(.+?)\.s3[.-]", location.hostname or "")
                        if bucket:
                            self.identities.add(bucket[1])
                    except ValueError:
                        pass
                elif ":::" in match[0]:
                    self.identities.add(match[0].split(":::", 1)[1].split("/", 1)[0])
            for match in _HOME.finditer(obj):
                self.identities.add(match[1])
            for match in _LOGIN.finditer(obj):
                self.identities.update(match.groups())
            for match in _LABEL.finditer(obj):
                self.identities.add(match[1])
            for match in _PATH.finditer(obj):
                child = _RUN_CHILD.search(match[0])
                if child:
                    self.roots.add(match[0][: child.start()])

    def _path(self, match):
        path = match[0]
        # Avoid eating arithmetic such as `1 / 2`, or regex syntax such as /\w+/.
        if path in ("/", "//") or not re.search(r"[A-Za-z]", path):  # also `(u-1)/1000`: no letters, no location
            return path
        for root in sorted(self.roots, key=len, reverse=True):
            if path == root:
                return "."
            if path.startswith(root + "/") or path.startswith(root + "\\"):
                rel = path[len(root) + 1 :].replace("\\", "/")
                # A path leaving the run must not masquerade as a relative path within it.
                if ".." not in rel.split("/"):
                    return rel
        return "[path omitted]"

    def text(self, text):
        text = redact_secrets(text)
        text = _REMOTE.sub("[remote location omitted]", text)
        text = _UNC.sub("[path omitted]", text)
        text = self.run_path.sub(self._path, text)
        text = _PATH.sub(self._path, text)
        text = _LOGIN.sub("[login omitted]", text)
        text = _HOST.sub("[host omitted]", text)
        if self.identity:
            text = self.identity.sub("[identity omitted]", text)
        return text

    def clean(self, obj):
        if isinstance(obj, str):
            return self.text(obj)
        if isinstance(obj, dict):
            return {self.text(k) if isinstance(k, str) else k: self.clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.clean(v) for v in obj]
        return obj
