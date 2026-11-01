# Input receipt timing for the next fresh acceptance cases

Status: protocol definition only; no new case has been accepted or launched by this document.

The source candidate remains the 274-file observed-query v2 freeze. The caller,
model, task and request budgets, fixture, native action, runtime, and independent
auditor are unchanged. The existing normal018 receipt files and actual owner
exit 1 remain immutable.

normal018 completed both runtime objectives and the unchanged caller final.
Its observed independent business state also passed. Attribution failed because
three application event timestamps were 2.781, 4.019 and 2.424 milliseconds after
their corresponding `Sky click` API-return timestamp. Its operator captured a
fresh native window state immediately after each click, but did not separately
sample that refresh's completion UTC. Those missing timestamps remain missing.
No normal018 interval is extended and its acceptance is not reclassified.

## New receipt meaning

For a future fresh case, `finished_utc` ends the controlled input operation:
one unchanged native click followed immediately by one fresh native observation
of the same returned target window. It is not a measurement of the click API's
execution latency. Add `timing_protocol: week6.sky_input_operation.v2` and retain
the click's own return time separately as `dispatch_returned_utc`.

The sole operator records the following timestamps directly in the same
JavaScript cell that performs that operation, in this order:

1. `started_utc`: immediately before awaiting the exact requested Sky click.
2. `dispatch_returned_utc`: immediately after that await returns successfully.
3. `post_refresh.started_utc`: immediately before awaiting the fresh native
   `get_window_state` on the same returned window.
4. `post_refresh.finished_utc`: immediately after that await returns.
5. `finished_utc`: exactly the same captured string as
   `post_refresh.finished_utc`; do not sample a later archival or inspection time.

`post_refresh` also records `method: sky.get_window_state`, the actual returned
window's `app`, `id` and `title`, and the returned `screenshot_id`. A nonempty
screenshot and the same exact target window are required. The operator inspects
that fresh result before writing the receipt exclusively with `wx` in the next
cell. Screenshot IDs are provider-local identifiers, not globally unique hashes.
The complete original request/action is copied without changing its type,
coordinates, click count, text, or ID. A native double-click remains one requested
double-click, including any independently logged ignored second physical click.

All operation intervals must be ordered and non-overlapping. There is no added
timer, arbitrary millisecond tolerance, hidden fixture-event polling, cursor
parking, extra input, model request, or task-budget increase. A failed input,
failed refresh, unknown outcome, cancellation or user stop cannot produce a
successful v2 receipt or trigger automatic input replay. Preserve the real
outcome under the existing stop and unknown-input rules.

## Evidence and acceptance

The original independent auditor continues to require every business transition
to belong to exactly one successful owned input receipt interval. It still
rejects missing or ambiguous attribution and one input credited with multiple
business transitions. Its comparison is unchanged: the new receipts state an
explicitly observed operation interval rather than the narrower API-return
interval used by normal018.

An additional read-only protocol audit must verify each actual new receipt's
version, timestamp ordering, equality of the two finish fields, same-window
post-refresh binding, nonempty screenshot identifier, exclusive original request
binding, non-overlap, and uniqueness of transition attribution. The ordinary
strict audit, real owner wait result, original/fresh frame checks, source and
dependency byte checks, Root semantic review, and live-count proof remain
required. No receipt or final verdict is written before the real operation or
the corresponding audit has occurred.

Normal and transient runs must be newly named and registered before launch.
Both use this same timing protocol. All earlier failures remain separate; their
prefixes, successful substeps, timing intervals, and outcomes are not pooled into
the new cases.
