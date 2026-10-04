# TaskPilot

Autonomous AI task worker for the CentrAlign AI Engineering Intern submission.

TaskPilot accepts a natural-language business objective and actually executes the work inside a simulated company environment. It does not tell the user which buttons to click. It searches, extracts, writes, recovers, asks for approval when needed, and independently verifies the outcome.

## Problem

Employees describe outcomes, not click paths:

> Process the latest invoice from Acme Corporation.

A useful worker has to find the invoice, choose the latest one from real search results, create a finance record, recover if the finance session expires, pause for approval on high-impact actions, and prove the saved record matches the invoice.

## Architecture

```text
                    USER
                      │
                      ▼
                TASK MANAGER
                      │
                      ▼
                  PLANNER
                      │
                TOOL ROUTER
          ┌───────────┼───────────┐
       BROWSER      COMPANY      FILES
          └───────────┼───────────┘
                      ▼
                  OBSERVER
                      │
                STATE MANAGER
             ┌────────┴────────┐
          RECOVERY           VERIFIER
             └────────┬────────┘
                  COMPLETION
```

See [docs/architecture.md](docs/architecture.md) for the component view and
[docs/decisions.md](docs/decisions.md) for the engineering decisions behind the runtime.

| Process | Port |
| --- | --- |
| Operator UI (`apps/web`) | http://127.0.0.1:3000 |
| Agent API (`backend`) | http://127.0.0.1:8000 |
| Company sandbox (`apps/sandbox`) | http://127.0.0.1:3001 |

## Agent lifecycle

```text
Goal → Understand → Plan → Execute → Observe → Adapt → Verify → Complete
```

The runtime is an explicit loop in `backend/runtime/loop.py`. It is not a single LLM call.

1. **Understand** extracts entities and whether the request is a safe, specific outcome.
2. **Plan** stores a structured list of steps on `TaskState`.
3. **Execute** runs one validated tool at a time.
4. **Observe** records the tool result and application data.
5. **Adapt** chooses the next tool from current artifacts and errors (including picking the latest invoice from a search list).
6. **Verify** re-reads the finance or customer system. Tool success is not task success.
7. **Complete** only if verification passes.

Ambiguous goals such as “Do something with Acme.” ask for clarification instead of guessing.

## Tools

Tools are named functions with structured results:

- Browser: `browser.open|click|type|read|screenshot|go_back` (sandbox origin only)
- Company: invoice/customer/finance/login
- Files: `files.search`, `files.read`
- Memory: SQLite `memory.store` / `memory.retrieve`
- Control: `task.complete`, `task.fail`, `task.clarify`

Every result looks like:

```json
{
  "success": true,
  "tool": "company.search_invoice",
  "data": {},
  "observation": "Found 4 invoice(s).",
  "artifacts": [],
  "error": null
}
```

Unknown tools, shell access, and off-sandbox URLs are rejected.

If `LLM_API_KEY` is set, the planner can propose the next tool via an OpenAI-compatible API. Malformed or disallowed calls are rejected. Tests and demos use the observation-driven planner so runs stay reproducible without a vendor key.

## State

`TaskState` is persisted in SQLite (`data/taskpilot.db`): goal, plan, observations, artifacts, approval, verification, evidence, retry counters.

## Memory

SQLite table `memory` stores facts discovered during a run, for example `Acme Corporation → CUST-102`. There is no vector database.

## Recovery

Failure injection is deterministic (`finance_session_expired`, `verification_mismatch`, `network_error`).

On `AUTH_EXPIRED` the agent classifies the error as recoverable, re-authenticates, retries with a bound of `max_retries = 2`, and continues. Recovery is real: the finance store actually invalidates the session on the first create attempt.

## Verification

The verifier never trusts the create/update payload. It reads the record back and compares fields. A mismatch marks the task failed.

The final summary reports the verification checks, the recovery note, and the captured
evidence (record IDs, invoice/customer identifiers). Session tokens are never included.

## Human approval

Demo mode **Approval required** pauses before `company.create_finance_record` or `company.update_customer`. The API holds the task in `awaiting_approval` until `POST /api/tasks/:id/approve` or `/reject`.

## Security

- Fake sandbox credentials only (`demo` / `demo123`)
- Browser navigation allowlisted to the local sandbox
- No shell tool
- Sensitive mutations can require approval
- Tool names and arguments are validated
- Secrets are not written to the audit log

## Setup

Python 3.11+ and Node 18+ are required.

```powershell
cd "C:\Users\moham\Desktop\PROJECTS\AI Worker"
copy .env.example .env
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
```

```powershell
.\.venv\Scripts\python.exe -m uvicorn apps.sandbox.main:app --host 127.0.0.1 --port 3001
```

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
```

```powershell
cd apps\web
npm install
npm run dev
```

Or run `.\scripts\dev.ps1`.

Optional LLM:

```
LLM_API_KEY=...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

Docker Compose can start sandbox + API; the Next.js UI is still started with `npm run dev`. Local uvicorn is the supported demo path.

## Demo

Open http://127.0.0.1:3000 and http://127.0.0.1:3001 (login `demo` / `demo123`).

1. **Successful autonomy** — Demo: Normal execution  
   Goal: `Process the latest invoice from Acme Corporation.`  
   Expect INV-1042, amount ₹84,250, a finance record, verification checks.

2. **Failure recovery** — Demo: Recoverable failure  
   Same goal. Expect `Finance session expired`, re-auth, retry, success.

3. **Human approval** — Demo: Approval required  
   Same goal. Click Approve. The runtime continues. Reject terminates safely.

4. **Generalization** — Demo: Normal execution  
   Goal: `Find Acme Corporation and update its customer status to Follow Up.`  
   Same runtime, different tools.

Optional: `Find all overdue invoices from Acme Corporation and summarize them.`

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Coverage includes invoice success, session expiry recovery, verification mismatch, approval pause/continue, approval rejection, customer update, ambiguous requests, network-error recovery, the overdue-summary workflow, and secret redaction.

A Playwright browser-automation test drives the sandbox through the agent's browser tools
(login, search invoices, read the invoice page, screenshot) and asserts that off-sandbox
navigation is blocked. It starts its own sandbox server, so no manual setup is needed:

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -m browser -v
```

It uses Playwright's bundled Chromium when available and otherwise falls back to an
installed Chrome/Edge. To force a channel, set `BROWSER_CHANNEL=chrome`. To install the
bundled browser instead: `python -m playwright install chromium`.

## Limitations

This prototype:

- operates in a simulated company environment
- supports a limited number of workflows
- uses deterministic sandbox data
- does not support arbitrary external websites; browser tools are restricted to the local sandbox and are exercised by the browser test rather than the default runtime path (the planner uses the structured company tools against the same store)
- does not provide production-grade authentication
- does not provide distributed execution
- does not yet have enterprise-scale observability
- has limited long-term memory

It is a demonstration of a reusable execution architecture, not a production AI employee.

## Future work

- **Phase 1:** this prototype
- **Phase 2:** Gmail, Slack, Drive, CRM, ERP connectors
- **Phase 3:** durable queues, distributed workers, enterprise identity, audit
- **Phase 4:** company memory, workflow learning, evals, multi-agent
- **Phase 5:** production AI employee infrastructure

## License

Provided as an intern take-home prototype.
