When you are done, reply with one JSON object and nothing else:
{"subagent_id": "researcher" or "operator", "status": "complete" | "partial" | "failed", "claims": [...], "open_issues": ["..."]}
Each claim is:
{"claim_id": "c1", "text": "...", "kind": "fact" | "derived" | "action_done" | "artifact", "value": <the key value, if any>, "evidence": [...], "depends_on": ["c0"], "confidence": "low" | "medium" | "high"}
Each evidence item cites one tool call:
{"type": "tool_call", "call_id": "<the id from that result's [call_id: ...] line>", "span": "<a short quote copied exactly from that result>"}
Rules:
- Cite the call whose result supports the claim. Copy the span character for character from that result.
- Write numbers in value exactly as they appear in the cited result. For a derived value, cite every call it is computed from and list the claims it depends on.
- For an action, cite the write call that performed it.
- When a claim is about a list or a set, include every item.
- Put anything you could not do in open_issues.
