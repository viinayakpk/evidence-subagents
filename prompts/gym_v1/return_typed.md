When you are done, reply with one JSON object and nothing else:
{"subagent_id": "researcher" or "operator", "status": "complete" | "partial" | "failed", "claims": [...], "open_issues": ["..."]}
Each claim is:
{"claim_id": "c1", "text": "...", "kind": "fact" | "derived" | "action_done" | "artifact", "value": <the key value, if any>, "evidence": [], "depends_on": ["c0"], "confidence": "low" | "medium" | "high"}
Rules:
- Leave evidence empty.
- Write numbers in value exactly as they appear in the tool result. For a derived value, list the claims it depends on.
- When a claim is about a list or a set, include every item.
- Put anything you could not do in open_issues.
