"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

type DemoMode = "normal" | "recoverable_failure" | "verification_failure" | "approval_required";

type EventItem = {
  type?: string;
  created_at?: string;
  message?: string;
  tool?: string;
  arguments?: Record<string, unknown>;
  data?: unknown;
  error?: { type?: string; message?: string } | null;
  error_type?: string;
  verified?: boolean;
  checks?: { field: string; expected: unknown; actual: unknown; passed: boolean }[];
  summary?: string;
  evidence?: { kind: string; value: unknown }[];
  step?: string;
  latency_ms?: number | null;
  provider?: string | null;
  model?: string | null;
  blocked?: boolean;
  retries_used?: number;
  retries_remaining?: number;
};

type Check = { field: string; expected: unknown; actual: unknown; passed: boolean };

type Task = {
  task_id: string;
  goal: string;
  status: string;
  progress: number;
  current_action?: string | null;
  summary?: string | null;
  approval_status?: string | null;
  approval_payload?: {
    action?: string;
    summary?: string;
    arguments?: Record<string, unknown>;
    impact?: Record<string, unknown>;
  } | null;
  verification_status?: string;
  verification?: { verified: boolean; observation: string; checks: Check[] } | null;
  clarification_question?: string | null;
  retry_count?: number;
  recovery_count?: number;
  step_count?: number;
  llm_provider?: string | null;
  llm_model?: string | null;
  llm_calls?: {
    step: number;
    provider?: string | null;
    model?: string | null;
    latency_ms?: number | null;
    tool?: string | null;
    validation?: string | null;
    prompt_tokens?: number | null;
    completion_tokens?: number | null;
  }[];
  evidence?: { kind: string; value: unknown; at?: string }[];
};

type AuditRow = {
  agent_step: string;
  kind: string;
  tool?: string | null;
  success?: number | null;
  latency_ms?: number | null;
  provider?: string | null;
  model?: string | null;
  created_at: string;
};

type Health = {
  ok: boolean;
  autonomous: boolean;
  llm: { provider?: string | null; model?: string | null; configured: boolean; error?: string | null };
  tools_registered: number;
  approval_policy: string;
  max_steps: number;
  max_retries: number;
};

const EXAMPLES = [
  "Process the latest invoice from Acme Corporation.",
  "Handle Acme's newest pending invoice.",
  "Find Acme Corporation and update its customer status to Follow Up.",
  "Find all overdue invoices from Acme Corporation and summarize them.",
  "Do something with Acme.",
];

const SSE_EVENTS = [
  "task.started",
  "task.resumed",
  "task.clarified",
  "ai.decision",
  "tool.started",
  "tool.completed",
  "tool.failed",
  "observation.created",
  "approval.required",
  "approval.approved",
  "approval.rejected",
  "approval.received",
  "recovery.started",
  "recovery.attempted",
  "recovery.completed",
  "verification.started",
  "verification.completed",
  "task.completed",
  "task.failed",
  "task.cancelled",
  "task.waiting_for_user",
];

const CHECK_LABELS: Record<string, string> = {
  invoice_id: "Invoice ID",
  company: "Company",
  amount: "Amount",
  due_date: "Due date",
  record_id: "Finance record",
  customer_id: "Customer ID",
  status: "Status",
  email: "Email",
  contact: "Contact",
  tools_executed: "Tools executed",
  failed_actions: "Failed actions",
};

function heading(event: EventItem): string {
  const type = event.type || "";
  if (type === "ai.decision") return "AI Decision";
  if (type === "tool.started") return `Tool · ${event.tool}`;
  if (type === "tool.completed") return `Result · ${event.tool}`;
  if (type === "tool.failed") return `Tool Failed · ${event.tool}`;
  if (type === "observation.created") return `Observation · ${event.tool}`;
  if (type === "approval.required") return "Approval Required";
  if (type.startsWith("approval.")) return "Approval";
  if (type.startsWith("recovery.")) return "Recovery";
  if (type.startsWith("verification.")) return "Independent Verification";
  if (type === "task.waiting_for_user") return "Waiting For You";
  if (type === "task.completed") return "Task Completed";
  if (type === "task.cancelled") return "Task Cancelled";
  if (type === "task.failed") return "Task Failed";
  return type;
}

function tone(type: string): string {
  if (type === "ai.decision") return "border-sky-800/70 bg-sky-950/40 text-sky-100";
  if (type === "tool.failed") return "border-rose-800/70 bg-rose-950/40 text-rose-100";
  if (type === "observation.created") return "border-teal-800/70 bg-teal-950/30 text-teal-100";
  if (type.startsWith("recovery.")) return "border-purple-800/70 bg-purple-950/40 text-purple-100";
  if (type.startsWith("verification.")) return "border-emerald-800/70 bg-emerald-950/30 text-emerald-100";
  if (type.startsWith("approval.")) return "border-amber-800/70 bg-amber-950/40 text-amber-100";
  if (type === "task.completed") return "border-emerald-700/70 bg-emerald-950/40 text-emerald-100";
  if (type === "task.failed" || type === "task.cancelled") return "border-rose-800/70 bg-rose-950/40 text-rose-100";
  return "border-slate-800 bg-slate-950/60 text-slate-300";
}

export default function HomePage() {
  const [goal, setGoal] = useState(EXAMPLES[0]);
  const [demo, setDemo] = useState<DemoMode>("normal");
  const [task, setTask] = useState<Task | null>(null);
  const [events, setEvents] = useState<EventItem[]>([]);
  const [audit, setAudit] = useState<AuditRow[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [answer, setAnswer] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const taskId = task?.task_id;

  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => (r.ok ? r.json() : null))
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const loadTask = useCallback(async (id: string) => {
    const response = await fetch(`${API}/api/tasks/${id}`);
    if (response.ok) setTask(await response.json());
  }, []);

  useEffect(() => {
    if (!taskId) return;
    const source = new EventSource(`${API}/api/tasks/${taskId}/events`);

    const push = (parsed: EventItem) => setEvents((current) => [...current, parsed]);

    source.onmessage = (message) => {
      try {
        push(JSON.parse(message.data));
      } catch {
        /* ignore malformed frames */
      }
    };
    SSE_EVENTS.forEach((name) => {
      source.addEventListener(name, (message) => {
        try {
          push(JSON.parse((message as MessageEvent).data));
        } catch {
          /* ignore malformed frames */
        }
      });
    });

    const poll = setInterval(() => loadTask(taskId).catch(() => undefined), 700);
    const auditPoll = setInterval(
      () =>
        fetch(`${API}/api/tasks/${taskId}/audit`)
          .then((r) => (r.ok ? r.json() : null))
          .then((data) => data && setAudit(data.audit))
          .catch(() => undefined),
      1200,
    );

    return () => {
      source.close();
      clearInterval(poll);
      clearInterval(auditPoll);
    };
  }, [taskId, loadTask]);

  async function runTask() {
    setBusy(true);
    setError(null);
    setEvents([]);
    setAudit([]);
    try {
      await fetch(`${API}/api/demo`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: demo }),
      });
      const response = await fetch(`${API}/api/tasks`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ goal, demo_mode: demo }),
      });
      if (!response.ok) throw new Error(`Task creation failed (${response.status})`);
      const created = await response.json();
      await loadTask(created.task_id);
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : String(exc));
    } finally {
      setBusy(false);
    }
  }

  async function decide(action: "approve" | "reject" | "cancel") {
    if (!taskId) return;
    setError(null);
    const response = await fetch(`${API}/api/tasks/${taskId}/${action}`, { method: "POST" }).catch(() => null);
    if (response && !response.ok) setError(`Request failed (${response.status}).`);
    if (response?.ok) await loadTask(taskId);
  }

  async function sendAnswer() {
    if (!taskId || !answer.trim()) return;
    const response = await fetch(`${API}/api/tasks/${taskId}/clarify`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ answer }),
    }).catch(() => null);
    if (response && !response.ok) setError(`Could not send the answer (${response.status}).`);
    if (response?.ok) {
      setAnswer("");
      await loadTask(taskId);
    }
  }

  const isTerminal = useMemo(
    () => !!task && ["completed", "failed", "cancelled", "awaiting_clarification"].includes(task.status),
    [task],
  );
  const awaitingApproval = task?.status === "awaiting_approval";
  const awaitingUser = task?.status === "awaiting_clarification";
  const running = !!task && !isTerminal && !awaitingApproval;

  return (
    <main className="mx-auto min-h-screen max-w-7xl px-6 py-8 text-slate-100">
      {/* Header */}
      <header className="mb-6 flex flex-wrap items-center justify-between gap-4 rounded-2xl border border-slate-800 bg-slate-900/90 px-6 py-4 shadow-xl">
        <div>
          <div className="flex items-center gap-2">
            <span className="inline-block h-2 w-2 rounded-full bg-sky-400" />
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-sky-400">
              LLM-driven autonomous task worker
            </p>
          </div>
          <h1 className="mt-1 text-2xl font-bold tracking-tight text-white">TaskPilot</h1>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {health && (
            <>
              <span
                className={`rounded-full border px-3 py-1 text-[11px] font-medium ${
                  health.autonomous
                    ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                    : "border-rose-500/40 bg-rose-500/10 text-rose-300"
                }`}
              >
                {health.autonomous ? "Real LLM connected" : "LLM not configured"}
              </span>
              <span className="rounded-full border border-slate-700 bg-slate-800/60 px-3 py-1 text-[11px] text-slate-300">
                {health.llm.provider} · {health.llm.model}
              </span>
              <span className="rounded-full border border-slate-700 bg-slate-800/60 px-3 py-1 text-[11px] text-slate-300">
                {health.tools_registered} tools
              </span>
            </>
          )}
          <span
            className={`inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-xs font-medium ${
              awaitingApproval
                ? "border-amber-500/40 bg-amber-500/10 text-amber-300"
                : isTerminal
                ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                : "animate-pulse border-sky-500/40 bg-sky-500/10 text-sky-300"
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                awaitingApproval ? "bg-amber-400" : isTerminal ? "bg-emerald-400" : "bg-sky-400"
              }`}
            />
            {awaitingApproval
              ? "Approval Required"
              : awaitingUser
              ? "Waiting For You"
              : isTerminal
              ? (task?.status || "Ready").replace(/_/g, " ").toUpperCase()
              : "Agent Running"}
          </span>
        </div>
      </header>

      {health && !health.autonomous && (
        <section className="mb-6 rounded-2xl border border-rose-700/60 bg-rose-950/40 p-4 text-sm text-rose-200">
          <strong>Autonomous execution is unavailable.</strong>{" "}
          {health.llm.error || "No LLM provider is configured."} Set LLM_PROVIDER, LLM_API_KEY and LLM_MODEL in
          <code className="mx-1 rounded bg-black/40 px-1.5 py-0.5 text-xs">.env</code>
          (see <code className="rounded bg-black/40 px-1.5 py-0.5 text-xs">.env.example</code>). TaskPilot never
          falls back to a deterministic planner.
        </section>
      )}

      {error && (
        <section className="mb-6 rounded-2xl border border-rose-700/60 bg-rose-950/40 p-4 text-sm text-rose-200">
          {error}
        </section>
      )}

      {/* Goal input */}
      <section className="rounded-2xl border border-slate-800 bg-slate-900/70 p-6 shadow-lg">
        <label className="block text-sm font-medium text-slate-300">User goal (natural language)</label>
        <textarea
          value={goal}
          onChange={(event) => setGoal(event.target.value)}
          placeholder="e.g. Process the latest invoice from Acme Corporation."
          rows={2}
          className="mt-2.5 w-full rounded-xl border border-slate-700 bg-slate-950/80 p-3.5 text-sm text-slate-100 placeholder-slate-500 outline-none focus:border-sky-500"
        />
        <div className="mt-3 flex flex-wrap gap-2">
          {EXAMPLES.map((example) => (
            <button
              key={example}
              onClick={() => setGoal(example)}
              className="rounded-full border border-slate-700/80 bg-slate-800/40 px-3 py-1 text-xs text-slate-300 transition hover:border-sky-400 hover:text-white"
            >
              {example}
            </button>
          ))}
        </div>

        <div className="mt-5 flex flex-wrap items-center justify-between gap-4 border-t border-slate-800/80 pt-4">
          <div className="flex items-center gap-3">
            <label className="text-xs font-medium uppercase tracking-wider text-slate-400">Demo mode</label>
            <select
              value={demo}
              onChange={(event) => setDemo(event.target.value as DemoMode)}
              className="rounded-lg border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-200 outline-none focus:border-sky-500"
            >
              <option value="normal">Normal execution</option>
              <option value="recoverable_failure">Recoverable failure (AUTH_EXPIRED)</option>
              <option value="verification_failure">Verification failure (mismatch)</option>
              <option value="approval_required">Approval required (sensitive gate)</option>
            </select>
          </div>
          <div className="flex items-center gap-3">
            {task && !isTerminal && (
              <button
                onClick={() => decide("cancel")}
                className="rounded-lg border border-slate-700 px-4 py-2 text-sm text-slate-300 transition hover:border-rose-500 hover:text-rose-300"
              >
                Cancel
              </button>
            )}
            <button
              onClick={runTask}
              disabled={busy || running || awaitingApproval || awaitingUser}
              className="rounded-lg bg-sky-600 px-6 py-2 text-sm font-semibold text-white shadow-md transition hover:bg-sky-500 disabled:opacity-40"
            >
              {busy ? "Starting…" : running ? "Agent running…" : "Run task"}
            </button>
          </div>
        </div>
        <p className="mt-3 text-[11px] text-slate-500">
          Demo mode changes only the sandbox&apos;s behaviour. The LLM, the safety layer and verification are
          identical in every mode.
        </p>
      </section>

      {/* Approval gate */}
      {awaitingApproval && task?.approval_payload && (
        <section className="mt-6 rounded-2xl border border-amber-600/70 bg-amber-950/40 p-6 shadow-xl">
          <h2 className="text-base font-semibold uppercase tracking-wide text-amber-200">
            Approval required — enforced by the backend
          </h2>
          <p className="mt-2 text-sm text-amber-100">
            {task.approval_payload.summary || "The agent proposed a state-changing action."}
          </p>
          {task.approval_payload.impact && Object.keys(task.approval_payload.impact).length > 0 && (
            <dl className="mt-4 grid gap-2 rounded-xl border border-amber-800/60 bg-black/40 p-4 sm:grid-cols-2">
              {Object.entries(task.approval_payload.impact).map(([key, value]) => (
                <div key={key} className="flex flex-col">
                  <dt className="text-[11px] uppercase tracking-wider text-amber-400/80">{key}</dt>
                  <dd className="font-mono text-sm text-amber-100">
                    {value && typeof value === "object" ? JSON.stringify(value) : String(value ?? "—")}
                  </dd>
                </div>
              ))}
            </dl>
          )}
          <details className="mt-3 text-xs text-amber-300/80">
            <summary className="cursor-pointer">Raw arguments sent to the tool</summary>
            <pre className="mt-2 overflow-auto rounded-lg bg-black/50 p-3 font-mono">
              {JSON.stringify(task.approval_payload.arguments ?? {}, null, 2)}
            </pre>
          </details>
          <div className="mt-4 flex gap-3">
            <button
              onClick={() => decide("approve")}
              className="rounded-lg bg-emerald-600 px-5 py-2 text-sm font-semibold text-white transition hover:bg-emerald-500"
            >
              Approve
            </button>
            <button
              onClick={() => decide("reject")}
              className="rounded-lg bg-rose-600 px-5 py-2 text-sm font-semibold text-white transition hover:bg-rose-500"
            >
              Reject
            </button>
          </div>
        </section>
      )}

      {/* Clarification */}
      {awaitingUser && (
        <section className="mt-6 rounded-2xl border border-sky-700/60 bg-sky-950/40 p-6 shadow-xl">
          <h2 className="text-base font-semibold uppercase tracking-wide text-sky-200">
            The agent needs clarification
          </h2>
          <p className="mt-2 text-sm text-sky-100">
            {task?.clarification_question || "Could you clarify what you would like me to do?"}
          </p>
          <div className="mt-4 flex flex-wrap gap-3">
            <input
              value={answer}
              onChange={(event) => setAnswer(event.target.value)}
              onKeyDown={(event) => event.key === "Enter" && sendAnswer()}
              placeholder="e.g. Process the latest invoice from Acme Corporation."
              className="min-w-72 flex-1 rounded-lg border border-sky-800 bg-slate-950 px-3.5 py-2 text-sm text-slate-100 outline-none focus:border-sky-500"
            />
            <button
              onClick={sendAnswer}
              disabled={!answer.trim()}
              className="rounded-lg bg-sky-600 px-5 py-2 text-sm font-semibold text-white transition hover:bg-sky-500 disabled:opacity-40"
            >
              Send answer
            </button>
          </div>
        </section>
      )}

      <section className="mt-6 grid gap-6 lg:grid-cols-3">
        {/* Live trace */}
        <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-lg lg:col-span-2">
          <div className="mb-3 flex items-center justify-between">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Live agent trace (real runtime events)
            </h2>
            {task && (
              <span className="font-mono text-[11px] text-slate-500">
                step {task.step_count} · {events.length} events
              </span>
            )}
          </div>
          <div className="max-h-[32rem] space-y-2 overflow-y-auto pr-1">
            {events.length === 0 && (
              <p className="py-10 text-center text-sm text-slate-500">
                No active task. Enter a goal and run it to watch the agent work.
              </p>
            )}
            {events.map((event, index) => (
              <div
                key={`${event.created_at}-${index}`}
                className={`rounded-xl border p-3 text-xs transition ${tone(event.type || "")}`}
              >
                <div className="mb-1 flex items-center justify-between gap-2">
                  <span className="text-[10px] font-semibold uppercase tracking-wider opacity-80">
                    {heading(event)}
                  </span>
                  <span className="flex items-center gap-2 text-[10px] text-slate-500">
                    {event.step && <span className="font-mono">{event.step}</span>}
                    {typeof event.latency_ms === "number" && <span className="font-mono">{event.latency_ms}ms</span>}
                    <span>{event.created_at?.slice(11, 19)}</span>
                  </span>
                </div>
                <p className="text-sm font-medium leading-snug">{event.message}</p>
                {event.arguments && Object.keys(event.arguments).length > 0 && (
                  <pre className="mt-1.5 overflow-auto rounded bg-black/40 p-1.5 font-mono text-[11px] text-slate-400">
                    {JSON.stringify(event.arguments)}
                  </pre>
                )}
                {event.error?.type && (
                  <p className="mt-1.5 font-mono text-[11px] text-rose-300">
                    {event.error.type}
                    {event.error.message ? `: ${event.error.message}` : ""}
                  </p>
                )}
                {event.type?.startsWith("recovery.") && typeof event.retries_remaining === "number" && (
                  <p className="mt-1.5 text-[11px] text-purple-300">retries remaining: {event.retries_remaining}</p>
                )}
                {event.type === "verification.completed" && Array.isArray(event.checks) && (
                  <ul className="mt-2 space-y-0.5 font-mono text-[11px]">
                    {event.checks.map((check) => (
                      <li key={check.field} className={check.passed ? "text-emerald-300" : "text-rose-300"}>
                        {check.passed ? "✓" : "✗"} {CHECK_LABELS[check.field] ?? check.field} · exp {String(check.expected)} /
                        act {String(check.actual)}
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            ))}
          </div>
        </div>

        <div className="flex flex-col gap-6">
          {/* Outcome */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-lg">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Outcome</h2>
              {task?.verification_status && task.verification_status !== "pending" && (
                <span
                  className={`rounded-full border px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wider ${
                    task.verification_status === "passed"
                      ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-300"
                      : "border-rose-500/40 bg-rose-500/10 text-rose-300"
                  }`}
                >
                  verification {task.verification_status}
                </span>
              )}
            </div>
            {task?.summary ? (
              <pre className="max-h-72 overflow-auto whitespace-pre-wrap rounded-xl border border-slate-800 bg-slate-950/80 p-4 font-mono text-xs leading-relaxed text-slate-200">
                {task.summary}
              </pre>
            ) : (
              <p className="py-6 text-center text-sm text-slate-500">No outcome yet.</p>
            )}

            {task?.verification?.checks && task.verification.checks.length > 0 && (
              <ul className="mt-4 space-y-1 font-mono text-xs">
                {task.verification.checks.map((check) => (
                  <li
                    key={check.field}
                    className={`flex items-center justify-between rounded px-2.5 py-1 ${
                      check.passed ? "bg-emerald-950/30 text-emerald-300" : "bg-rose-950/30 text-rose-300"
                    }`}
                  >
                    <span>
                      {check.passed ? "✓" : "✗"} {CHECK_LABELS[check.field] ?? check.field}
                    </span>
                    <span className="opacity-80">
                      exp {String(check.expected)} · act {String(check.actual)}
                    </span>
                  </li>
                ))}
              </ul>
            )}

            {task?.evidence && task.evidence.length > 0 && (
              <div className="mt-4 border-t border-slate-800/80 pt-3">
                <h3 className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-slate-400">
                  Evidence ({task.evidence.length})
                </h3>
                <div className="flex flex-wrap gap-1.5">
                  {task.evidence.map((item, index) => (
                    <span
                      key={index}
                      className="rounded bg-slate-800/80 px-2 py-0.5 font-mono text-[10px] text-slate-300"
                      title={typeof item.value === "string" ? item.value : JSON.stringify(item.value)}
                    >
                      {item.kind}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* LLM observability */}
          <div className="rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-lg">
            <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
              LLM calls &amp; latency
            </h2>
            {task?.llm_calls && task.llm_calls.length > 0 ? (
              <ul className="space-y-1.5 font-mono text-[11px]">
                {task.llm_calls.map((call, index) => (
                  <li
                    key={index}
                    className="flex items-center justify-between gap-2 rounded bg-slate-950/70 px-2.5 py-1.5"
                  >
                    <span className="text-sky-300">#{call.step}</span>
                    <span className="flex-1 truncate text-slate-300">{call.tool || "no tool call"}</span>
                    <span className="text-slate-500">{call.latency_ms ?? "?"}ms</span>
                    <span
                      className={
                        call.validation === "blocked"
                          ? "text-rose-400"
                          : call.validation === "verification_requested"
                          ? "text-emerald-400"
                          : "text-slate-500"
                      }
                    >
                      {call.validation}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="text-sm text-slate-500">No model calls recorded yet.</p>
            )}
            <p className="mt-3 text-[11px] text-slate-500">
              {task?.llm_provider || health?.llm.provider} · {task?.llm_model || health?.llm.model}
              {task ? ` · ${task.recovery_count ?? 0} recovery attempt(s)` : ""}
            </p>
          </div>
        </div>
      </section>

      {audit.length > 0 && (
        <section className="mt-6 rounded-2xl border border-slate-800 bg-slate-900/70 p-5 shadow-lg">
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">
            Observability feed (GET /api/tasks/:id/audit)
          </h2>
          <div className="max-h-64 overflow-auto">
            <table className="w-full text-left font-mono text-[11px]">
              <thead className="text-slate-500">
                <tr>
                  <th className="py-1 pr-3">step</th>
                  <th className="py-1 pr-3">kind</th>
                  <th className="py-1 pr-3">tool</th>
                  <th className="py-1 pr-3">result</th>
                  <th className="py-1 pr-3">latency</th>
                  <th className="py-1 pr-3">model</th>
                </tr>
              </thead>
              <tbody>
                {audit.map((row, index) => (
                  <tr key={index} className="border-t border-slate-800/70 text-slate-300">
                    <td className="py-1 pr-3 text-slate-500">{row.agent_step}</td>
                    <td className="py-1 pr-3">{row.kind}</td>
                    <td className="py-1 pr-3">{row.tool || "—"}</td>
                    <td className="py-1 pr-3">
                      {row.success === null ? "—" : row.success ? "ok" : "error"}
                    </td>
                    <td className="py-1 pr-3">{row.latency_ms ?? "—"}ms</td>
                    <td className="py-1 pr-3 text-slate-500">{row.model || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </main>
  );
}
