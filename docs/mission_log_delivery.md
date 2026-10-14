# TraceLab owned mission observations

Sprint 61 LOG-2 is tracked in the TraceLab CMOS project. The receiver contract is
[mission-log-delivery-contract.md](https://github.com/kneelinghorse/TraceLab/blob/55b5da6a7cc2334350aa40a251d0c479a3c96914/cmos/contracts/mission-log-delivery-contract.md).
The shared example is `tests/fixtures/mission-logs-v2/batch.json` in both repos.

The worker passes the claimed immutable `LeaseAttempt` to `TracelabLogHandler`.
Only after ownership confirmation and heartbeat startup does it call
`start_live_delivery()` off the event loop. That method verifies the service
principal and v2 capability marker before starting the existing five-second
and threshold transport. No constructor flag enables live writes.

Each event gets a UUID and sequence before enqueue. The same attempt proof and
identities survive retries. A response must acknowledge version, mission,
attempt, counts, ordered IDs and 200/201 semantics. Arbitrary successful HTTP
responses do not clear the queue. Default bounds are 2,000 pending entries,
200 per batch, three attempts, ten-second request timeout, 1/2-second backoff.
Overflow is counted and never evicts an in-flight event. Transport failures do
not fail research or discard paid results.

`stop_delivery()` interrupts scheduling/backoff promptly; `discard()` then waits
for the active HTTP call before releasing or requeueing ownership. Receiver 409
ends this immutable attempt's delivery. Terminal `close()` uses the durable
result-key proof after the live token clears, and retained retries keep that
proof. There is no legacy fallback. Legacy capability markers `retired` and
`terminal_only_until_LOG-2_cutover` both negotiate v2; legacy POST is never used.

`log_projection.py` is the single enum/numeric allowlist used on both sides of
the child pipe and by the HTTP sink. Raw prompt text, formatted arguments,
tracebacks and arbitrary extras do not become persisted messages. Existing
cancellation, parent-death, sandbox and artifact boundaries remain in place.

Offline worker regressions cover the live-log contract, lease/fence lifecycle,
worker exception handling and preflight. The obsolete S87 readiness hash pins
were formally retired; they are not an accepted failing baseline. CI rejects
all unit failures, errors and skips. Deployment and live receiver acceptance
require separate evidence for the deployed source and a useful authorized mission.
