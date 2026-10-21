"""Offline app-server fixture for the Codex backend. No real CLI or credentials needed."""

import json
import os
import sys
import time
from pathlib import Path

cfg = json.loads(Path(__file__).with_name("fake.json").read_text())
root = Path(cfg["root"])
root.mkdir(exist_ok=True)
if "--version" in sys.argv:
    print("codex-cli 0.160.1")
    sys.exit()
if sys.argv[1:3] == ["login", "status"]:
    sys.exit(1 if cfg.get("nologin") else 0)
launch = root / "launches"
n = int(launch.read_text()) + 1 if launch.exists() else 1
launch.write_text(str(n))
session = None
state = None
rollout = None


def emit(x):
    print(json.dumps(x), flush=True)


def append(kind, payload):
    with rollout.open("a") as f:
        f.write(json.dumps(dict(type=kind, payload=payload)) + "\n")


def save():
    (root / (session + ".json")).write_text(json.dumps(state))


for line in sys.stdin:
    req = json.loads(line)
    method, a = req["method"], req.get("params", {})
    with (root / "calls.jsonl").open("a") as f:
        f.write(json.dumps(dict(method=method, params=a, session=session)) + "\n")
    result = {}
    if method == "initialized":
        continue
    if method == "initialize":
        result = dict(codexHome=os.environ["CODEX_HOME"])
    elif method == "account/read":
        result = dict(account=None if cfg.get("nologin") or n <= cfg.get("login_fails", 0) else dict(type="chatgpt"))
    elif method == "config/read":
        result = dict(config=dict(mcp_servers={"planted": {"command": "MUST_NOT_RUN"}}))
    elif method in ("thread/start", "thread/resume"):
        if method == "thread/start":
            session = "fake-thread-" + str(n)
            rollout = root / (session + ".jsonl")
            state = dict(turns=[], total=0)
            append("session_meta", dict(id=session, base_instructions=dict(text=a["baseInstructions"])))
            save()
        else:
            session = a["threadId"]
            state = json.loads((root / (session + ".json")).read_text())
            rollout = root / (session + ".jsonl")
        result = dict(
            thread=dict(id=session, path=str(rollout)),
            model=a["model"],
            reasoningEffort=cfg.get("effort", "high"),
            approvalPolicy="never",
            instructionSources=cfg.get("sources", []),
            sandbox=dict(type="workspaceWrite", networkAccess=False, excludeTmpdirEnvVar=True, excludeSlashTmp=True),
            runtimeWorkspaceRoots=a["runtimeWorkspaceRoots"],
            activePermissionProfile=dict(id="opal"),
        )
    elif method == "turn/start":
        message = a["input"][0]["text"]
        tid = "turn-" + str(len(state["turns"]) + 1)
        before = rollout.stat().st_size
        state["turns"].append(dict(id=tid, message=message, before=before))
        append("turn_context", dict(model=a["model"], effort=a["effort"]))
        append("response_item", dict(type="message", role="user", content=[dict(type="input_text", text=message)]))
        mode = cfg.get("mode", "ok")
        if mode == "slow":
            time.sleep(cfg.get("seconds", 20))
        if cfg.get("pidfile"):
            Path(cfg["pidfile"]).write_text(str(os.getpid()))
        emit(dict(id=req["id"], result=dict(turn=dict(id=tid))))
        failed = n <= cfg.get("turn_fails", 0)
        if mode == "injected":
            append(
                "response_item",
                dict(type="message", role="developer", content=[dict(type="input_text", text="skills list")]),
            )
        if mode == "duplicate":
            append("response_item", dict(type="message", role="user", content=[dict(type="input_text", text=message)]))
        if mode == "hidden":
            append("response_item", dict(type="custom_tool_call", name="web_search", call_id="hidden", input="danger"))
        if cfg.get("updater") and not failed:
            Path("updater.py").write_text(cfg["updater"])
            if Path("analysis.py").exists():
                with open("analysis.py", "a") as f:
                    f.write("\n# fake Codex edit\n")
            if cfg.get("code_mode"):
                append(
                    "response_item",
                    dict(
                        type="custom_tool_call",
                        name="exec",
                        call_id="edit",
                        input='text(await tools.exec_command({cmd:"python edit.py"}));',
                    ),
                )
                append("response_item", dict(type="custom_tool_call_output", call_id="edit", output="ok"))
            else:
                append(
                    "response_item",
                    dict(
                        type="function_call", name="exec_command", call_id="edit", arguments='{"cmd":"python edit.py"}'
                    ),
                )
                append("response_item", dict(type="function_call_output", call_id="edit", output="ok"))
            emit(
                dict(
                    method="item/completed",
                    params=dict(threadId=session, turnId=tid, item=dict(type="commandExecution")),
                )
            )
        if not failed:
            text = "done " + str(len(state["turns"]))
            append(
                "response_item", dict(type="message", role="assistant", content=[dict(type="output_text", text=text)])
            )
            emit(
                dict(
                    method="item/completed",
                    params=dict(threadId=session, turnId=tid, item=dict(type="agentMessage", text=text)),
                )
            )
            state["total"] += 1
            total = dict(
                inputTokens=100 * state["total"],
                cachedInputTokens=40 * state["total"],
                outputTokens=30 * state["total"],
                reasoningOutputTokens=20 * state["total"],
                totalTokens=130 * state["total"],
            )
            # Last inference deliberately undercounts a tool-using turn: only cumulative deltas are correct.
            emit(
                dict(
                    method="thread/tokenUsage/updated",
                    params=dict(
                        threadId=session,
                        turnId=tid,
                        tokenUsage=dict(total=total, last=dict(total, inputTokens=1, outputTokens=1)),
                    ),
                )
            )
        if mode == "compacted":
            append("compacted", dict(message="summary replaced the conversation"))
        save()
        error = (
            dict(message=cfg.get("error", "Usage limit reached"), codexErrorInfo=cfg.get("kind", "usageLimitExceeded"))
            if failed
            else None
        )
        emit(
            dict(
                method="turn/completed",
                params=dict(
                    threadId=session, turn=dict(id=tid, status="failed" if failed else "completed", error=error)
                ),
            )
        )
        continue
    elif method == "thread/revert":
        if cfg.get("revert_fails"):
            emit(dict(id=req["id"], error=dict(code=-1, message="revert failed")))
            continue
        target = next(t for t in state["turns"] if t["id"] == a["beforeTurnId"])
        with rollout.open("r+") as f:
            f.truncate(target["before"])
        state["turns"] = state["turns"][: state["turns"].index(target)]
        save()
        result = dict(thread=dict(id=session))
    emit(dict(id=req.get("id"), result=result))
