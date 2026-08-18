export class TermyteDBError extends Error {
  constructor(public status: number, public detail: unknown, public requestId?: string) {
    super(`TermyteDB request failed (${status})`);
  }
}

export type ClientOptions = { timeoutMs?: number; retries?: number; authToken?: string; fetch?: typeof globalThis.fetch };

export class TermyteDBClient {
  private readonly fetcher: typeof globalThis.fetch;
  constructor(private readonly baseUrl: string, private readonly options: ClientOptions = {}) {
    this.fetcher = options.fetch ?? globalThis.fetch;
  }

  async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const retries = Math.max(0, this.options.retries ?? 2);
    const requestId = crypto.randomUUID();
    for (let attempt = 0; attempt <= retries; attempt++) {
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), this.options.timeoutMs ?? 30000);
      try {
        const response = await this.fetcher(`${this.baseUrl.replace(/\/$/, "")}/${path.replace(/^\//, "")}`, {
          method, signal: controller.signal, headers: { accept: "application/json", "x-request-id": requestId, ...(this.options.authToken ? { authorization: `Bearer ${this.options.authToken}` } : {}), ...(body === undefined ? {} : { "content-type": "application/json" }) },
          body: body === undefined ? undefined : JSON.stringify(body),
        });
        const text = await response.text();
        const value = text ? JSON.parse(text) : undefined;
        if (response.ok) return value as T;
        if (![408, 429, 500, 502, 503, 504].includes(response.status) || attempt === retries) {
          throw new TermyteDBError(response.status, value, response.headers.get("x-request-id") ?? undefined);
        }
      } finally { clearTimeout(timer); }
      await new Promise((resolve) => setTimeout(resolve, Math.min(250 * 2 ** attempt, 2000)));
    }
    throw new Error("unreachable");
  }

  ingest(event: unknown) { return this.request("POST", "/v1/events", event); }
  process(namespaceId: string, options: Record<string, unknown> = {}) { return this.request("POST", "/v1/process", { namespace_id: namespaceId, ...options }); }
  search(namespaceId: string, query: string, options: Record<string, unknown> = {}) { return this.request("POST", "/v1/search", { namespace_id: namespaceId, query, ...options }); }
  context(namespaceId: string, query: string, options: Record<string, unknown> = {}) { return this.request("POST", "/v1/context", { namespace_id: namespaceId, query, ...options }); }
  health() { return this.request("GET", "/health"); }
  events(namespaceId: string, options: Record<string, unknown> = {}) { return this.request("GET", `/v1/events?namespace_id=${encodeURIComponent(namespaceId)}&limit=${options.limit ?? 100}&offset=${options.offset ?? 0}`); }
  evidence(namespaceId: string, options: Record<string, unknown> = {}) { return this.request("GET", `/v1/evidence?namespace_id=${encodeURIComponent(namespaceId)}&limit=${options.limit ?? 100}&offset=${options.offset ?? 0}`); }
  memories(namespaceId: string, options: Record<string, unknown> = {}) { return this.request("GET", `/v1/memories?namespace_id=${encodeURIComponent(namespaceId)}&limit=${options.limit ?? 100}&offset=${options.offset ?? 0}`); }
  jobs(namespaceId: string, options: Record<string, unknown> = {}) { return this.request("GET", `/v1/jobs?namespace_id=${encodeURIComponent(namespaceId)}&limit=${options.limit ?? 100}&offset=${options.offset ?? 0}`); }
  memory(namespaceId: string, memoryId: string) { return this.request("GET", `/v1/memories/${encodeURIComponent(memoryId)}?namespace_id=${encodeURIComponent(namespaceId)}`); }
  history(namespaceId: string, memoryId: string) { return this.request("GET", `/v1/memories/${encodeURIComponent(memoryId)}/history?namespace_id=${encodeURIComponent(namespaceId)}`); }
  invalidate(namespaceId: string, memoryId: string, reason: string) { return this.request("POST", `/v1/memories/${encodeURIComponent(memoryId)}/invalidate?namespace_id=${encodeURIComponent(namespaceId)}&reason=${encodeURIComponent(reason)}`); }
}
