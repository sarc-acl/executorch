"""GPU campaign -- rlar with the work and the review both on another machine.

    hmz exec -f @user/gpu_campaign \
        -a actor=<model> -a reviewer=<model> \
        -e box=ssh@[<user>@<host>]/<abs path>/executorch \
        -b duration=3d,cost=100 "$(cat PROMPT.txt)"

A fork of the built-in `rlar`. The built-in flow has one environment, the workspace the run was
started in, so its reviewer reads a local directory even when the actor did its real work on a
GPU host over ssh. Here both roles are spawned on `box`, the directory on the host that `-e`
names, and the reviewer is told that this is where the work is.

Both models are arguments: `-a actor=...` and `-a reviewer=...` take any backend hmz knows, in
its `<backend>/<model>:<effort>` form. Where the two are the same model family the reviewer
shares the actor's blind spots, so the review prompt does not leave the method to the reviewer:
it names the checks the reviewer has to run itself and asks for a list of what it did not check.

It ends when the reviewer says the task is done, or when the budget is spent; `--resume` hands a
fresh actor the last review to pick up from. A turn that fails, or a review out of shape, is
taken again after a wait that grows (WAITS); GIVE_UP failures in a row end the run with the last
one. A few minutes of provider trouble therefore costs a wait, not the run.

Last run with hmz commit 474add43cf42db8881741a29cb81ef3f9a8c1a87 (hmz is installed from git
without a pin). Install with `cp -r gpu_campaign ~/.hmz/flows/`; a running hmz window has to be
quit and reopened before `/resume` picks up a changed flow file.
"""

import asyncio

from pydantic import BaseModel, Field

from hmz.flows import (
    Agent,
    AgentCollection,
    Env,
    EnvCollection,
    EnvConnectionError,
    FlowContext,
    FlowParams,
    FlowState,
    HarnessError,
    HarnessNotInstalled,
    LocalEnv,
    Permission,
    PermissionKind,
    ShellEnvMixin,
    UnsupportedOperation,
    flow,
)

GIVE_UP = 8
WAITS = (60.0, 300.0, 900.0)
PAUSE = 5.0

# What a wait may cure: a turn the CLI or its provider could not take, and a host that did not
# answer. What no wait changes is raised at once.
TRANSIENT = (HarnessError, EnvConnectionError)
HOPELESS = (HarnessNotInstalled, UnsupportedOperation)


class Actor(Agent):
    """The one that works: builds, measures and commits on the GPU host.

    Runs without hmz's write fence on the host. A campaign has to build in a container, write
    builds and raw logs outside the working copy, and take the device lock under ~/.cache; the
    default grant refuses all three. Chosen by the owner on 2026-10-04. What the actor may do is
    bounded by the task's rules, not by the kernel.
    """

    _permission = Permission(
        local=PermissionKind.ALL,
        user=PermissionKind.ALL,
        system=PermissionKind.ALL,
        online=PermissionKind.ALL,
    )


class Reviewer(Agent):
    """A fresh reader of each round, carrying the flow's own way of writing a review.

    Keeps the default grant: reads everything, writes only inside the working copy and its own
    temporary directory.
    """

    _skills = ("review-notes",)


class Agents(AgentCollection):
    """The one that works and the one that reads what it did."""

    actor: Actor
    reviewer: Reviewer


class Box(Env, ShellEnvMixin):
    """The working copy on the GPU host."""


class Envs(EnvCollection):
    """The workspace humanize requires, and the host directory where both roles work."""

    workspace: LocalEnv
    box: Box


class Review(BaseModel):
    """What one round's review comes to: whether it is over, and what the actor is told."""

    model_config = {"extra": "forbid"}

    done: bool = Field(
        description="True only if the task is completely and correctly done: everything "
        "asked for is implemented, it works, nothing was faked, stubbed or special-cased to "
        "pass, there is no next step worth taking, and you ran every one of the required "
        "checks yourself in this round. False if there is anything at all left to do or to "
        "fix, or if any required check is listed under not_checked."
    )
    notes: str = Field(
        description="The review itself, written as a message to the coding agent: what is "
        "done, what is wrong or missing, and what to do next, citing specific files, lines "
        "and commands. It is passed on word for word and is all the agent will hear from "
        "you, so leave nothing to be inferred. Give the numbers you recomputed beside the "
        "numbers that were reported. When done is true, this is what the run finishes on: "
        "say what was built and how it was checked."
    )
    not_checked: str = Field(
        description="Everything you did not check in this round, one item per line, each "
        "with the reason (no time, needs the GPU, file missing, not understood). Claims you "
        "took from a summary without recomputing them belong here. Write 'nothing' only if "
        "that is true."
    )


REVIEW_PROMPT = """You are a meticulous reviewer, running on the machine and in the working \
directory where a coding agent has been doing the task below. The work is HERE, in this \
directory and under the artifact paths the task names on this machine; it is not on whatever \
machine started this run. Use shell tools (cat, ls, git status, git log, git diff, etc.) to \
review what it has actually done against the state of this repository and its result files. \
Be skeptical: treat reward hacking -- tests weakened or special-cased, work stubbed out or \
faked, gates or goldens edited to pass -- as the thing you are most there to catch.

You may be the same model as the agent you review. What it found convincing you will find \
convincing too, so do not judge its account of the work: redo the checks. Each of the \
following you do yourself, in this round, and report with the command you ran and what it \
printed. A check you did not run is not passed; it goes into not_checked, and done is false.

1. Recompute every number the agent reports in this round (per-cell medians, gains, geometric \
means, validity counts, error figures) from the raw run files: runs.csv and the per-run logs, \
not STATUS.md, not a summary script's output that you did not rerun. Name the files. State \
each recomputed value beside the reported one, and flag any difference beyond rounding. Check \
that the baseline arm was measured in the same session as the candidate, that every claimed \
gain is outside the noise band the task fixes, and that invalid runs are still in the files \
with their reasons.

2. Run the shipped-SPIR-V golden check yourself on the build that was measured: \
sarc/tools/spirv_golden.py with that build's shader directory against sarc/golden/spirv.json. \
The shipped variants must be unchanged. Confirm that the build was made from an exported \
commit and which commit that was.

3. List the files changed outside the dev zone: git diff --name-status <parent commit> HEAD, \
filtered to everything that is not under backends/vulkan/runtime/graph/ops/glsl/sarc_dev, \
backends/vulkan/runtime/graph/ops/impl/sarc_dev, backends/vulkan/test/sarc_dev, sarc/ or \
openspec/. Print the list in your notes even when it is empty. Every entry needs an owner \
decision in the task file that covers exactly that edit. Separately list any change under \
sarc/tools, sarc/golden, to a test tolerance, a prompt or a threshold file: none is allowed.

4. Read every shader that is new or changed since the parent commit, line by line, for \
unsynchronised shared writes: more than one invocation storing to the same shared variable, \
shared array element, buffer element or image texel without a barrier or an atomic, including \
the case where every writer stores the same value (still a data race); reads of shared memory \
that can run before the barrier that follows its writes; stores made by every lane of a \
subgroup where one elected lane should store. Passing tests are not evidence here: a race of \
this kind passed every test of an earlier campaign. Name each shader you read.

5. Check the gate evidence of every candidate accepted in this round: the unmodified \
verify.sh output exists for exactly the binaries and environment that were timed and matches \
the parent's snapshot line by line apart from rates; the SDPA tiers ran with 0 mismatches; a \
candidate accepted under an owner decision is recorded as such, not as a plain pass.

Do not start GPU jobs or builds: the device belongs to the agent's queue, and a second \
workload spoils its measurements. A check that needs the GPU is asked of the agent in your \
notes and listed in not_checked.

Where you are (the flow ran this on the host before starting you):
{where}

Task (TASK.md):
"""

PICKED_UP = """{task}

Work in this repository is already under way: an earlier run here was stopped before it \
finished, and below is a reviewer's reading of the last round of it. That run may have been on \
the task above or on something else -- what carries over is the repository, not the task. You \
did not do that work and have no record of it beyond what the files now hold, so read them \
first, then carry on with the task above, taking the review as far as it bears on it.

Review of the last round:
{notes}"""

NOT_CHECKED = """{notes}

Not checked by the reviewer in this round (do not treat these as passed):
{not_checked}"""


async def _where(box: Box) -> str:
    """The host, directory, branch and head commit, so a review names what it looked at."""
    lines = []
    for label, argv in (
        ("host", ["hostname"]),
        ("directory", ["pwd"]),
        ("branch", ["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        ("head", ["git", "log", "-1", "--format=%h %s"]),
    ):
        try:
            _, out, _ = await box.exec(argv)
        except Exception as exc:  # noqa: BLE001 -- orientation only; never fail a round on it
            out = f"(unavailable: {exc})"
        lines.append(f"- {label}: {str(out).strip()}")
    return "\n".join(lines)


def _wait(failed: int) -> float:
    """How long to stay away after `failed` failures in a row: longer each time, then level."""
    if failed <= 0:
        return PAUSE
    return WAITS[min(failed, len(WAITS)) - 1]


def _failure(exc: Exception, failed: int, what: str) -> None:
    """Counts one failed turn: raises it if it is hopeless or one too many, else says so."""
    if isinstance(exc, HOPELESS) or failed >= GIVE_UP:
        raise exc
    print(
        f"{what} failed ({failed} of {GIVE_UP} in a row): {type(exc).__name__}: {exc}; "
        f"next try in {_wait(failed):.0f} s"
    )


@flow(agents=Agents, envs=Envs, params=FlowParams, resumable=True)
async def gpu_campaign(
    task: str, *, agents: Agents, envs: Envs, params: FlowParams, ctx: FlowContext
) -> str:
    """An actor works on a GPU host until a fresh reviewer on the same host says it is done."""
    state = ctx.state
    assert state is not None  # noqa: S101 -- a resumable flow is always handed its state
    actor, reviewer = agents["actor"], agents["reviewer"]
    box = envs["box"]
    # hmz moved the environment from spawn(env=...) to run(..., env=...) on 2026-10-05.
    working = await actor.spawn()
    notes: str = state["notes"] if "notes" in state else ""
    prompt = PICKED_UP.format(task=task, notes=notes) if notes else task
    failed = 0
    while True:
        try:
            worked = await actor.run(prompt, session=working, env=box)
        except TRANSIENT as exc:
            failed += 1
            _failure(exc, failed, "actor turn")
            worked = ""
        if worked:
            reading = await reviewer.spawn()
            where = await _where(box)
            try:
                review = await reviewer.run(
                    REVIEW_PROMPT.format(where=where) + task,
                    session=reading,
                    env=box,
                    output_schema=Review,
                )
            except TRANSIENT as exc:
                failed += 1
                _failure(exc, failed, "review")
                review = None
            else:
                failed = 0
            if review is not None and review.done:
                print(review.notes)
                _forget(state)
                return review.notes
            if review is not None and review.notes:
                prompt = notes = NOT_CHECKED.format(
                    notes=review.notes, not_checked=review.not_checked or "nothing"
                )
            state["rounds"] = (state["rounds"] if "rounds" in state else 0) + 1
            state["notes"] = notes
        await asyncio.sleep(_wait(failed))


def _forget(state: FlowState) -> None:
    """Drops what a run that finished kept, so the next one starts on the task alone."""
    for key in ("notes", "rounds"):
        if key in state:
            del state[key]
