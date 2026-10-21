"""One isolated app-server connection. Executed as a child, so Backend owns its whole process group.

No OPAL imports: this file also runs when Python's module search path is restricted. Stdout contains JSONL records,
never login/account contents. A successful turn is accepted only after auditing the provider's actual rollout.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

if __package__:
    from ._privacy import redact_record, redact_stream
else:  # standalone transport: importing the sibling helper does not import OPAL or torch
    from _privacy import redact_record, redact_stream

FIELDS = {
    "inputTokens": "input_tokens",
    "cachedInputTokens": "cached_input_tokens",
    "outputTokens": "output_tokens",
    "reasoningOutputTokens": "reasoning_output_tokens",
}
TOOLS = {"exec_command", "write_stdin", "shell_command", "apply_patch", "exec", "wait"}


def emit(value):
    print(json.dumps(redact_record(value)), flush=True)


def toml(value):
    if isinstance(value, dict):
        return "{" + ", ".join(json.dumps(k) + " = " + toml(v) for k, v in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(toml(v) for v in value) + "]"
    return json.dumps(value)


def classify(error):
    """Codex structured errors supplement the common Claude-oriented text classifier, without changing it."""
    text = str(error)
    low = text.lower()
    for kind, words in (
        ("login", ("unauthorized", "not logged in", "authentication", "401", "invalid_api_key")),
        ("limit", ("usagelimitexceeded", "ratelimitexceeded", "usage limit", "rate limit", "429", "quota")),
        (
            "down",
            (
                "serveroverloaded",
                "httpconnectionfailed",
                "responsestream",
                "connection",
                "network",
                "fetch failed",
                "500",
                "502",
                "503",
                "504",
                "overloaded",
            ),
        ),
        ("missing", ("no such file", "not found")),
    ):
        if any(w in low for w in words):
            return [kind, text[:300]]
    return None


class IsolationError(RuntimeError):
    pass


class Client:
    def __init__(self, request):
        self.request = request
        self.n = 0
        self.events = queue.Queue(maxsize=4096)
        self.usage = None
        self.thread = None
        self.turn = None
        self.pending_turn = False
        self.messages = []
        self.worked = False
        cmd = [request["binary"], "app-server", "--stdio"]
        for key, value in request["flags"].items():
            cmd += ["-c", key + "=" + toml(value)]
        # stderr is separate from the wire transcript; it may name account paths, but never read auth contents.
        self.stderr = tempfile.TemporaryFile(mode="w+")
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
            cwd=request["workspace"],
            env=dict(os.environ, CODEX_HOME=request["home"]),
        )
        self.deadline = time.monotonic() + request.get("timeout", 3600)
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        try:
            for line in self.proc.stdout:
                self.events.put(json.loads(line))
        except Exception as exc:
            self.events.put(exc)
        finally:
            self.events.put(None)

    def send(self, message):
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def next(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Codex decision timed out")
        try:
            event = self.events.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError("Codex decision timed out") from exc
        if event is None or isinstance(event, Exception):
            raise RuntimeError("Codex app-server exited or broke the protocol")
        method = event.get("method", "")
        if method.startswith("account/") or method == "authRecovery":
            return event  # do not retain account identities or credentials
        emit(dict(type="rpc", event=event))
        if "method" in event and "id" in event:
            self.send(
                dict(id=event["id"], error=dict(code=-32601, message="OPAL never grants approvals or external tools"))
            )
            raise IsolationError("Unexpected Codex approval/tool request: " + method)
        params = event.get("params", {})
        if method == "thread/tokenUsage/updated":
            if params.get("threadId") == self.thread and (self.turn is None or params.get("turnId") == self.turn):
                self.usage = params.get("tokenUsage")
        if method == "item/completed":
            item = params.get("item", {})
            if item.get("type") == "agentMessage":
                self.messages.append(item.get("text", ""))
            elif item.get("type") in ("commandExecution", "fileChange"):
                self.worked = True
        return event

    def rpc(self, method, params, private=False):
        self.n += 1
        rid = self.n
        self.send(dict(id=rid, method=method, params=params))
        while True:
            if private:
                # Account/config responses are not logged; notifications are consumed normally thereafter.
                remaining = self.deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Codex decision timed out")
                try:
                    event = self.events.get(timeout=remaining)
                except queue.Empty as exc:
                    raise TimeoutError("Codex decision timed out") from exc
                if event is None or isinstance(event, Exception):
                    raise RuntimeError("Codex app-server exited")
                if event.get("id") != rid:
                    continue
            else:
                event = self.next()
            if event.get("id") != rid:
                continue
            if "error" in event:
                raise RuntimeError(method + ": " + json.dumps(event["error"]))
            return event["result"]

    def start(self):
        r = self.request
        init = self.rpc(
            "initialize",
            dict(clientInfo=dict(name="opal_sol", version=r["client_version"]), capabilities=dict(experimentalApi=True)),
        )
        if Path(init["codexHome"]).resolve() != Path(r["home"]).resolve():
            raise IsolationError("App-server used a different CODEX_HOME")
        self.send(dict(method="initialized"))
        account = self.rpc("account/read", dict(refreshToken=False), private=True)
        if not account.get("account"):
            raise RuntimeError("Not logged in: run CODEX_HOME=~/.codex-opal codex login --device-auth")
        cfg = self.rpc("config/read", dict(includeLayers=False), private=True).get("config", {})
        config = {"mcp_servers": {name: {"enabled": False} for name in cfg.get("mcp_servers", {})}}
        params = dict(
            model=r["model"],
            modelProvider=r["flags"]["model_provider"],
            cwd=r["workspace"],
            approvalPolicy="never",
            permissions="opal",
            runtimeWorkspaceRoots=[r["workspace"]],
            baseInstructions=r["system"],
            developerInstructions="",
            config=config,
        )
        if r.get("session"):
            params.update(threadId=r["session"], excludeTurns=True)
            out = self.rpc("thread/resume", params)
        else:
            params.update(ephemeral=False, historyMode="paginated", allowProviderModelFallback=False)
            out = self.rpc("thread/start", params)
        self.thread = out["thread"]["id"]
        self.rollout = out["thread"].get("path")
        emit(dict(type="opal_thread", session=self.thread, rollout=self.rollout))
        if r.get("session") and self.thread != r["session"]:
            raise IsolationError("Codex resumed a different thread")
        if out.get("model") != r["model"] or out.get("reasoningEffort") != r["effort"]:
            raise IsolationError("Codex model/effort differs from requested model/effort")
        if out.get("instructionSources"):
            raise IsolationError("Codex loaded instruction sources: " + str(out["instructionSources"]))
        if out.get("approvalPolicy") != "never":
            raise IsolationError("Codex approval policy changed")
        sandbox = out.get("sandbox", {})
        if (
            sandbox.get("type") != "workspaceWrite"
            or sandbox.get("networkAccess") is not False
            or sandbox.get("excludeTmpdirEnvVar") is not True
            or sandbox.get("excludeSlashTmp") is not True
        ):
            raise IsolationError("Codex workspace-write/no-network/no-global-temp policy changed")
        roots = out.get("runtimeWorkspaceRoots", [])
        if [str(Path(x).resolve()) for x in roots] != [str(Path(r["workspace"]).resolve())]:
            raise IsolationError("Codex runtime workspace roots changed")
        if out.get("activePermissionProfile", {}).get("id") != "opal":
            raise IsolationError("Codex restricted filesystem profile changed")
        if not self.rollout:
            raise IsolationError("Provider rollout path unavailable; cannot audit context")
        return out

    def run_turn(self):
        r = self.request
        self.pending_turn = True
        turn = self.rpc(
            "turn/start",
            dict(
                threadId=self.thread,
                input=[dict(type="text", text=r["message"])],
                model=r["model"],
                effort=r["effort"],
                approvalPolicy="never",
                permissions="opal",
                cwd=r["workspace"],
                runtimeWorkspaceRoots=[r["workspace"]],
            ),
        )
        self.turn = turn["turn"]["id"]
        while True:
            event = self.next()
            if event.get("method") == "turn/completed":
                params = event["params"]
                if params.get("threadId") == self.thread and params["turn"]["id"] == self.turn:
                    return params["turn"]

    def audit(self, successful):
        """Incremental provider audit, including hidden calls and all message roles. Never scan the whole home."""
        r = self.request
        path = Path(self.rollout)
        offset = r.get("audit_offset", 0)
        users, replies, kinds, tools, initial, seen_turn = [], [], [], [], bool(offset), False
        calls = set()
        # Turn completion can precede the rollout writer's flush. Bounded drain; each byte read only once.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if path.exists():
                with path.open() as stream:
                    stream.seek(offset)
                    while True:
                        before = stream.tell()
                        line = stream.readline()
                        if not line or not line.endswith("\n"):
                            stream.seek(before)
                            break
                        obj = json.loads(line)
                        p = obj.get("payload", {})
                        kind = obj.get("type")
                        if kind == "session_meta":
                            if p.get("base_instructions", {}).get("text") != r["system"]:
                                raise IsolationError("Provider baseInstructions differs from OPAL system text")
                            initial = True
                        elif kind == "turn_context":
                            if p.get("model") != r["model"] or p.get("effort") != r["effort"]:
                                raise IsolationError("Provider rollout model/effort mismatch")
                            if p.get("developer_instructions"):
                                raise IsolationError("Unexpected provider developer instructions")
                            seen_turn = True
                        elif kind == "compacted":
                            raise IsolationError(
                                "Codex compacted persistent history; full conversation no longer preserved"
                            )
                        elif kind == "world_state":
                            state = p.get("state", {})
                            if state.get("agents_md") or state.get("managed_developer_instructions"):
                                raise IsolationError("Provider world_state includes external instructions")
                            for key in ("skills", "host_skills"):
                                if state.get(key, {}).get("includeInstructions"):
                                    raise IsolationError("Provider world_state enables skills instructions")
                            if any(
                                state.get(k)
                                for k in ("apps_instructions", "plugins_instructions", "environments_instructions")
                            ):
                                raise IsolationError(
                                    "Provider world_state includes app/plugin/environment instructions"
                                )
                        elif kind == "response_item":
                            item = p.get("type")
                            kinds.append(item)
                            if item == "message":
                                role = p.get("role")
                                text = "".join(c.get("text", "") for c in p.get("content", []))
                                if role == "user":
                                    users.append(text)
                                elif role == "assistant":
                                    replies.append(text)
                                elif role != "assistant":
                                    raise IsolationError("Unexpected provider message role: " + str(role))
                            elif item == "reasoning":
                                pass
                            elif item in ("function_call", "custom_tool_call"):
                                name = p.get("name", "").removeprefix("functions.")
                                if name not in TOOLS:
                                    raise IsolationError("Unexpected provider tool: " + name)
                                calls.add(p.get("call_id"))
                                tools.append(dict(name=name, call_id=p.get("call_id")))
                            elif item in ("function_call_output", "custom_tool_call_output"):
                                if p.get("call_id") not in calls:
                                    raise IsolationError("Unmatched provider tool output")
                            else:
                                raise IsolationError("Unexpected provider non-message item: " + str(item))
                    offset = stream.tell()
            final_seen = bool(self.messages) and self.messages[-1] in replies
            if initial and seen_turn and users and (not successful or final_seen):
                break
            time.sleep(0.1)
        if not initial or not seen_turn or users != [r["message"]] or (successful and not final_seen):
            raise IsolationError("Provider context audit incomplete or user content differs from the new OPAL message")
        total = (self.usage or {}).get("total")
        previous = r.get("provider_total", {})
        if successful and total is None:
            raise IsolationError("Codex cumulative token usage missing")
        usage = {}
        if total is not None:
            for field, label in FIELDS.items():
                delta = total[field] - previous.get(field, 0)
                if delta < 0:
                    raise IsolationError("Codex cumulative token count decreased")
                usage[label] = delta
        return dict(
            passed=True,
            thread=self.thread,
            turn=self.turn,
            rollout=str(path),
            initial_base_matches=True,
            new_user_message_matches=True,
            response_item_types=kinds,
            tools=tools,
            usage=usage,
            provider_total=total or previous,
            audit_offset=offset,
        )

    def revert(self):
        """Discard an unavailable/failed turn before retry; the durable thread id stays the same."""
        self.deadline = time.monotonic() + 10
        out = self.rpc("thread/revert", dict(threadId=self.thread, beforeTurnId=self.turn))
        if out.get("thread", {}).get("id") != self.thread:
            raise IsolationError("Codex failed-turn rollback changed the conversation id")
        self.rollout = out["thread"].get("path") or self.rollout
        emit(dict(type="opal_thread", session=self.thread, rollout=self.rollout))
        return Path(self.rollout).stat().st_size

    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.stderr.seek(0)
        with Path(self.request["stderr"]).open("w") as output:
            redact_stream(self.stderr, output)
        self.stderr.close()


def main():
    request = json.load(sys.stdin)
    client = None
    result = dict(text="", returncode=-1, error=None)
    try:
        client = Client(request)
        client.start()
        turn = client.run_turn()
        success = turn["status"] == "completed"
        audit = client.audit(success)
        result.update(
            text="\n".join(client.messages),
            usage=audit["usage"],
            provider_total=audit["provider_total"],
            audit_offset=audit["audit_offset"],
            audit=audit,
            worked=client.worked,
        )
        if success:
            result.update(returncode=0)
        else:
            error = json.dumps(turn.get("error") or turn["status"])
            result["error"] = error
            down = classify(error)
            if down:
                result["unavailable"] = down
            result["audit_offset"] = client.revert()
    except Exception as exc:
        result.update(error=str(exc), returncode=-1)
        if isinstance(exc, IsolationError):
            result.update(poisoned=True)
        else:
            down = classify(str(exc))
            if down:
                result["unavailable"] = down
            if client is not None and client.turn is not None:
                try:
                    client.deadline = time.monotonic() + 10
                    client.rpc("turn/interrupt", dict(threadId=client.thread, turnId=client.turn))
                    result["audit_offset"] = client.revert()
                except Exception:
                    result.update(poisoned=True)
            elif client is not None and client.pending_turn:
                result.update(poisoned=True)
    finally:
        if client is not None:
            # Audit rejection does not erase measured consumption. A missing report stays unknown, not zero.
            total = (client.usage or {}).get("total")
            if total is not None and not result.get("usage"):
                previous = request.get("provider_total", {})
                delta = {label: total[field] - previous.get(field, 0) for field, label in FIELDS.items()}
                if all(n >= 0 for n in delta.values()):
                    result.update(usage=delta, provider_total=total)
            client.close()
    emit(dict(type="opal_result", result=result))


if __name__ == "__main__":
    main()
