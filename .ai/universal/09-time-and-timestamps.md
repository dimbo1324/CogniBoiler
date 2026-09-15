<!-- tier: extended -->

# Time and Timestamps: Never Take a Date Without Its Moment

> **Essence.** Read and report every date at full precision — hours, minutes, seconds, zone. For git use `--date=iso-strict` (`git log --pretty=format:"%h %cd %s"`); `--oneline`, `--date=short` and "2 days ago" never answer *when*.

Purpose: a date without a time cannot order two things that happened on the same day —
and on a project where several commits land in one day, that is precisely the question
being asked. Owner instruction, 2026-08-05.

## The rule

Whenever you take in information that carries a date — a commit, a file's mtime, a
history row, a log line, a release, a build, a database record — take the **full
moment** with it: hours, minutes, seconds, and the timezone (or an explicit note that
the value is UTC). Report it the same way.

Truncating is a decision, not a default. If a coarser rendering genuinely serves the
reader better in one specific place, say so at that place and explain why; never reach
for a day-granular helper because it is shorter.

Forbidden as an *answer* to "when did this happen":

- relative time alone — "2 days ago", "recently", "last week";
- a bare date — `2026-08-05` with no time;
- `git log --oneline`, which carries no time at all;
- a timestamp with no zone and no statement that it is UTC.

Relative time is fine as a *secondary* rendering next to the exact one — a tooltip, a
parenthetical. It is never the only thing shown.

## Git specifically

The commands to reach for, all read-only:

```powershell
git log -15 --date=iso-strict --pretty=format:"%h %cd %s"   # history, with committer dates
git show -s --format=fuller --date=iso-strict <ref>         # one commit, author AND committer
git log -1 --date=iso-strict --format=%cd -- <path>         # when a file last changed
```

`%cd` (committer date) is what orders history as it actually landed; `%ad` (author date)
survives a rebase or a cherry-pick unchanged, so the two disagree more often than people
expect. When the distinction could matter to the conclusion, read both and say which one
you are quoting.

`--date=iso-strict` gives `2026-08-05T14:23:07+03:00` — a real ISO 8601 instant with the
offset. Prefer it over `--date=iso` (a space instead of the `T`) and never fall back to
`--date=short` or `--date=relative`.

## In code and artifacts

- A timestamp written into an artifact, a log line, a database row, or an interface
  carries seconds. A formatting helper that drops the time of day does not belong in a
  shared utility module: without one available, truncation has to be written out
  deliberately at the call site, where a reviewer can see it.
- Store the full instant (epoch seconds, or better) even when the current display is
  coarser. Precision that was never captured cannot be recovered later.
- State the zone. If everything renders UTC, say `UTC` in the rendered string rather
  than leaving a reader to guess at their own offset.
