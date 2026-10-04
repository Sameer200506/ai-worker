# Architecture

```text
                    USER
                      │
                      ▼
              TaskPilot Web (Next.js)
                      │  REST + SSE
                      ▼
                TASK MANAGER (FastAPI)
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
                STATE MANAGER (SQLite)
             ┌────────┴────────┐
          RECOVERY           VERIFIER
             └────────┬────────┘
                  COMPLETION
```

The sandbox at `http://127.0.0.1:3001` is a simulated internal company app with login, invoices, customers, and finance.

The operator dashboard at `http://127.0.0.1:3000` never talks to the sandbox directly. The agent runtime does.

Company tools execute against the same deterministic store the sandbox UI reads, so browser users and the agent see one environment.
