"""The draft neutral prompt's TASK paragraph: run facts (a clause per given fact), the schedule from the actual
decision updates, and the sentence for each waiting mode."""

import torch

import opal
from opal.misc import taskfacts
from opal.schedule import decision_updates
from test_core import adamw, flat, make, train
from test_review2 import plain_run


def system(tmp_path, steps=200, backend=None, **kw):
    m, data = make()
    o, s = adamw(m, steps)
    kw = dict(dict(decisions=4, every=10, verbose=False, prompt="neutral"), **kw)
    agent = backend or (lambda ws, msg, sy: "x")
    w = opal.wrap(o, m, scheduler=s, total_steps=steps, goal="g", run_dir=tmp_path / "r", backend=agent, **kw)
    return w, m, data, s


def test_fact_clauses_are_left_out_when_not_given(tmp_path):
    assert taskfacts.run_facts(68074) == "The run is planned for 68,074 training updates."
    assert taskfacts.run_facts(1000, epochs=10) == "The run is planned for 1,000 training updates, about 10 epochs."
    full = taskfacts.run_facts(
        68074, epochs=10, train_size="435,653 sequences of 512 tokens", batch_size=64, same_order=True
    )
    assert full == (
        "The run is planned for 68,074 training updates, about 10 epochs over a training set of 435,653 sequences of "
        "512 tokens, with a batch of 64 per update; each epoch passes over the same data again."
    )
    sys_none = system(tmp_path / "a")[0].worker.system
    task = sys_none.split("SCOPE:")[0]
    assert "The run is planned for 200 training updates.\n" in task and "None" not in task and "Not given" not in task
    sys_all = system(tmp_path / "b", epochs=2, train_size="50 batches", batch_size=16, same_order_each_epoch=True)
    assert (
        "about 2 epochs over a training set of 50 batches, with a batch of 16 per update; each epoch passes over the same data again."
        in sys_all[0].worker.system
    )


def test_schedule_sentence_matches_the_decision_updates():
    at = decision_updates(200, 4, cadence=10, schedule="uniform")
    assert at == [50, 100, 150]
    assert taskfacts.schedule_sentence(at, "uniform", 200) == (
        "You are called every 50 updates: at updates 50, 100, 150, 3 scheduled decisions in all."
    )
    at = decision_updates(68074, 100, cadence=50, schedule="uniform")  # snapped to checks: gaps 650 to 700
    assert len(at) == 100 and taskfacts.schedule_sentence(at, "uniform", 68074, epochs=10) == (
        "You are called about every 680 updates (about once every 0.1 epochs): at updates "
        f"{at[0]:,}, {at[1]:,}, {at[2]:,}, ..., {at[-1]:,}, 100 scheduled decisions in all."
    )
    at = decision_updates(10000, 20, cadence=50)
    early = [u for u in at if u <= 500]
    text = taskfacts.schedule_sentence(at, "front", 10000)
    late = at[len(early) :]
    assert (
        text.startswith("You are called ")
        and f" for the first {early[-1]:,} updates ({len(early)} decisions), then " in text
    )
    assert text.endswith(
        f" from update {late[0]:,} to update {late[-1]:,} ({len(late)} decisions); {len(at)} scheduled decisions in all."
    )
    # one early decision, and the late phase's own spacing (not the jump from the early phase)
    at = decision_updates(1000, 10, cadence=50)
    assert at == [50, 200, 300, 400, 500, 600, 700, 800, 900]
    assert taskfacts.schedule_sentence(at, "front", 1000) == (
        "You are called at update 50, then every 100 updates from update 200 to update 900 (8 decisions); "
        "9 scheduled decisions in all."
    )
    assert taskfacts.schedule_sentence([50, 600], "front", 1000) == (
        "You are called at update 50, then at update 600; 2 scheduled decisions in all."
    )


def test_decision_sentence_matches_the_mode(tmp_path):
    expect = {
        True: "Training pauses until you finish, and your edits take effect from that update.",
        False: "Training keeps running while you work; your edits take effect at the first update boundary after you finish.",
        "auto": "Training waits for your answer when it would otherwise take effect more than 50 updates late",
    }
    for i, (mode, sentence) in enumerate(expect.items()):
        text = system(tmp_path / str(i), blocking=mode, schedule="uniform")[0].worker.system
        assert "Your decisions. You are called every 50 updates: at updates 50, 100, 150, 3 scheduled decisions in all." in text
        assert sentence in text
        assert [s for s in expect.values() if s in text] == [sentence]


def test_neutral_messages_give_the_epoch_and_the_next_decision(tmp_path):
    msgs = []
    w, m, data, s = system(
        tmp_path,
        backend=lambda ws, msg, sy: msgs.append(msg) or "x",
        blocking=True,
        schedule="uniform",
        epochs=4,
    )
    train(m, w, data, 200, sched=s)
    assert (
        "(25.0% of training, epoch 1.00 of 4)" in msgs[0] and "The next scheduled decision is at update 100." in msgs[0]
    )


def test_neutral_with_facts_and_keep_is_bitwise_equal_to_plain_training(tmp_path):
    w, m, data, s = system(tmp_path, steps=120, backend="keep", epochs=3, batch_size=16)
    train(m, w, data, 120, sched=s)
    assert torch.equal(flat(m), plain_run(120))


def test_starting_optimizer_summary_line_and_reference_file(tmp_path):
    w, m, data, s = system(tmp_path)
    text = w.worker.system
    line = "The starting updater.py runs the training script's own optimizer and scheduler, unchanged: AdamW, base lr 0.003,"
    assert line in text and "scheduler CosineAnnealingLR (T_max 200" in text
    assert "weight decay 0.1 on 4 tensors" in text and "group 0 (4 tensors" not in text  # the full dump is not inline
    ref = (tmp_path / "r" / "workspace" / "reference" / "starting_optimizer.txt").read_text()
    assert "optimizer 0: torch.optim.adamw.AdamW" in ref and '"lr": 0.003' in ref and "CosineAnnealingLR" in ref
    print(text.split("unchanged: ", 1)[1].split(". It", 1)[0])


def test_neutral_prompt_leaves_out_cost_replay_and_states_the_time_limit(tmp_path):
    msgs = []
    w, m, data, s = system(tmp_path, backend=lambda ws, msg, sy: msgs.append(msg) or "x", blocking=True, epochs=2)
    train(m, w, data, 200, sched=s)
    text = w.worker.system
    for word in ("replay", "rerun", "run again", "static", "from update 1", "from scratch", "budget", "$"):
        assert word not in text, word
    assert "TASK FACTS" not in text and "which is not shown to you." in text
    assert msgs and all("agent cost" not in x and "$" not in x for x in msgs)
    assert taskfacts.time_limit(3600) == "Each decision may take up to an hour."
    assert taskfacts.time_limit(21600) == "Each decision may take up to 6 hours."
