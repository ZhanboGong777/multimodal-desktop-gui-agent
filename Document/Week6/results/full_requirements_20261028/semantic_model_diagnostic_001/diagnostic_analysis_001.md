# Semantic planning diagnosis — development evidence only

The three pre-registered local model requests completed with a real child wait exit code 0, 3 actual HTTP requests, and all 255 frozen code pins unchanged. No screenshot capture, executor or input was called. Exit 0 means the diagnostic harness finished; none of the plans was executed.

| Case | Native result | Local result | Meaning |
| --- | --- | --- | --- |
| A: full caller/snapshot + schema text | ready, 6 goals | Rejected: duplicate goal checks | Copies all final checks per goal and treats persistent titles/headings as selectable tasks. |
| B: A minus current snapshot | ready, 3 goals | Existing structural/literal checks pass | Cleaner intent sequence, but Review/Commit are unverified future UI text. This is not a live success. |
| C: B with explicitly unspecified project | ready, 3 goals | Existing checks pass | Negative control fails: the missing parameter is copied into an executable goal. |

The original normal006 saved arguments match the logged system/user messages, preserve the complete caller and final conditions, and request the intended model. Its HTTP response ends with `stop` after 2171 prompt and 91 completion tokens. There is no observed role-order, caller-loss or output-token exhaustion explanation. The logger preserves pre-wire messages/arguments and complete provider bodies, not a separate raw wire request body.

Ollama recommends providing the schema as prompt text as well as using structured output; A isolates that change. The changed output is not evidence that semantic correctness is fixed. [Official structured output documentation](https://docs.ollama.com/capabilities/structured-outputs)

B/C intentionally retain the old system sentence saying an OCR snapshot is supplied even though it is omitted. This pre-registered limitation means their failures cannot disprove a properly specified intent-only design. C still demonstrates that the current parser/model combination accepts an explicit unresolved parameter. The three requests are a development comparison, not a measured success rate.

A reasonable next restriction is intent-only semantic planning whose literal predicates may only select the caller's original final predicates, with separate positive/negative pools and exact native validation. Keep the full archived snapshot, low-level multimodal action grounding, all before/post/receipt/final checks, and all shared budgets. Reject declared unresolved placeholders before dispatch without claiming complete natural-language ambiguity detection. Let the model group adjacent requested operations rather than invent UI labels or impose a fixture-specific decomposition.

This narrower protocol still cannot promise two independently observable stages merely because a caller supplies two strings; they may become true together or be substring-equivalent. Each goal must remain false before its related inputs, and an independent acceptance event audit must check operation coverage. Observable text alone does not prove every hidden semantic operation. A new native plan, negative controls and complete real normal/transient runs are required before completion can be claimed.
