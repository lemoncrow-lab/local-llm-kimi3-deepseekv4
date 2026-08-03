"""OpenAI-compatible HTTP server for DeepSeek-V4-Flash on one RTX 4090.

Why a custom engine at all: the checkpoint declares `DeepseekV4ForCausalLM` with no
`auto_map`, so transformers cannot load it, and vLLM / SGLang / llama.cpp have no such
architecture either (hyper-connections, Sinkhorn-normalised combination weights,
sqrtsoftplus routing with per-token hash layers, per-layer KV compression ratios, fp4
e2m1 experts with e8m0 block scales, DSpark MTP blocks).  The *engine* therefore has to
be the code in run.py.  Everything above it is off-the-shelf: FastAPI + uvicorn for
routing, SSE and schema validation, and the checkpoint's own encoding_dsv4 for the chat
template, including its tool-call format.

Running it as a server is worth far more here than it would be for a normal model:
starting the CLI pays ~35 s of prefill *and* rebuilds both expert caches from cold, so
the first ~30 tokens of every invocation run at a third of steady-state speed.  A
resident process pays that once.  It also keeps the KV cache, so a conversation that
extends the previous one skips prefill entirely.

    ./serve-dsv4-4090.sh                       # 127.0.0.1:8000
    curl -N localhost:8000/v1/chat/completions -H 'content-type: application/json' \
      -d '{"messages":[{"role":"user","content":"write fizzbuzz"}],"stream":true}'

One request at a time, by construction: max_batch_size is 1 and the expert cache is
shared mutable state, so generate() holds a lock.  Concurrent callers queue.
"""
import argparse
import json
import os
import sys
import threading
import time
import uuid
from typing import Any, Dict, List, Optional, Union

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run as R                                          # noqa: E402  (installs kernels)

MODEL_ID = "deepseek-v4-flash"


class Engine:
    """The model, its caches and the KV state, resident across requests."""

    def __init__(self, seq_len=16384, vram_gib=0.0, ram_gib=48.0, threads=32):
        import model as M

        torch.set_default_dtype(torch.bfloat16)
        torch.set_grad_enabled(False)
        with open(os.path.join(R.REF, "config.json")) as f:
            cfg = M.ModelArgs(**json.load(f))
        cfg.max_batch_size = 1
        cfg.max_seq_len = seq_len
        self.cfg = cfg
        self.seq_len = seq_len

        R.preflight_vram()
        t0 = time.perf_counter()
        w = R.Weights(R.MODEL_DIR)
        self.model = R.build(cfg, seq_len)
        loaded, missing = R.load_trunk(self.model, w)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.stream = R.ExpertStream(w, vram_gib=vram_gib, ram_gib=ram_gib, threads=threads)
        R.install_streaming_moe(self.model, self.stream, cfg)
        torch.set_default_device("cuda")
        print(f"loaded {loaded} trunk tensors in {time.perf_counter()-t0:.1f}s; "
              f"cache {self.stream.Ng} experts in vram, {self.stream.Nh} in pinned ram",
              flush=True)

        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(R.MODEL_DIR)
        sys.path.insert(0, os.path.join(R.MODEL_DIR, "encoding"))
        from encoding_dsv4 import encode_messages
        self.encode_messages = encode_messages

        self.eos = 1
        self.continue_max = int(os.environ.get("DSV4_CONTINUE_MAX", 256))
        self.lock = threading.Lock()
        self.state: List[int] = []          # token ids currently held in the KV cache
        self.n_prompt = self.n_gen = 0
        self.t_gen = 0.0
        self.reuse = 0

    # -- KV state ---------------------------------------------------------
    def reset(self):
        """Return every stateful attention buffer to its constructed value.

        The compressor writes kv_state/score_state at `start_pos % ratio`, so the state
        can be *extended* safely but never rewound -- a prompt that is not a prefix
        extension of the last one has to start from zero.
        """
        for name, buf in self.model.named_buffers():
            if name.endswith("score_state"):
                buf.fill_(float("-inf"))
            elif name.endswith(("kv_cache", "kv_state")):
                buf.zero_()
        self.state = []

    # -- generation -------------------------------------------------------
    def _step(self, ids, start):
        # model.py allocates bare tensors inside forward, and set_default_device is
        # thread-local -- uvicorn runs handlers on threadpool threads, so scope it here.
        with torch.device("cuda"):
            t = torch.tensor([ids], dtype=torch.long, device="cuda")
            return int(self.model.forward(t, start)[0].item())

    def generate(self, ids: List[int], max_new: int, temperature: float,
                 stop: List[str]):
        """Yields (token_id, text_so_far).  Holds the engine lock for its lifetime."""
        # A big default max_tokens must not 400 a big prompt: clamp to what is left,
        # and only refuse when the prompt alone does not fit.
        room = self.seq_len - len(ids) - 1
        if room <= 0:
            raise HTTPException(400, f"prompt is {len(ids)} tokens, seq_len is "
                                     f"{self.seq_len} (raise DSV4_SEQ_LEN)")
        max_new = min(max_new, room)
        with self.lock:
            self.model.temperature = max(float(temperature), 1e-4)
            # The reference Attention only has two modes: a chunked prefill at
            # start_pos == 0, and single-token decode (model.py:534 squeezes seqlen).
            # So a continuation has to be walked one token at a time -- 0.25 s each
            # against a flat ~35 s prefill, hence the length cut-off.
            n = len(self.state)
            t0 = time.perf_counter()
            if 0 < n < len(ids) and self.state == ids[:n] \
                    and len(ids) - n <= self.continue_max:
                self.reuse += 1
                for p in range(n, len(ids)):
                    nxt = self._step([ids[p]], p)
            else:
                self.reset()
                n = 0
                nxt = self._step(ids, 0)
            self.n_prompt += len(ids) - n
            self.state = list(ids)
            pos = len(ids)
            text, out = "", []
            try:
                for _ in range(max_new):
                    if nxt == self.eos:
                        break
                    out.append(nxt)
                    self.state.append(nxt)
                    text = self.tok.decode(out)
                    yield nxt, text          # GeneratorExit here if the client vanishes
                    if any(s and text.endswith(s) for s in stop):
                        break
                    nxt = self._step([nxt], pos)
                    pos += 1
            finally:
                self.n_gen += len(out)
                self.t_gen += time.perf_counter() - t0

    def render(self, messages, thinking_mode="chat", reasoning_effort="low", tools=None):
        kw = {"thinking_mode": thinking_mode, "reasoning_effort": reasoning_effort}
        msgs = list(messages)
        if tools:
            msgs = [{"role": "system", "content": "", "tools": tools}] + msgs \
                if msgs and msgs[0].get("role") != "system" else msgs
        return self.encode_messages(msgs, **kw)

    def stats(self):
        s = self.stream
        n = max(1, s.hit_g + s.hit_h + s.miss)
        return {
            "prompt_tokens": self.n_prompt, "generated_tokens": self.n_gen,
            "tok_per_s": round(self.n_gen / self.t_gen, 2) if self.t_gen else None,
            "prefix_reuses": self.reuse,
            "kv_tokens_resident": len(self.state),
            "expert_cache": {
                "vram_slots": s.Ng, "ram_slots": s.Nh,
                "vram_hit_pct": round(100 * s.hit_g / n, 1),
                "ram_hit_pct": round(100 * s.hit_h / n, 1),
                "disk_pct": round(100 * s.miss / n, 1),
            },
        }


# --------------------------------------------------------------------- schema
class ChatRequest(BaseModel):
    model: str = MODEL_ID
    messages: List[Dict[str, Any]]
    max_tokens: Optional[int] = 4096
    temperature: float = 0.2
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    thinking_mode: str = "chat"
    reasoning_effort: str = "low"


class CompletionRequest(BaseModel):
    model: str = MODEL_ID
    prompt: str
    max_tokens: Optional[int] = 4096
    temperature: float = 0.2
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None


app = FastAPI(title="DeepSeek-V4-Flash on a 4090")
ENGINE: Optional[Engine] = None


def _stops(stop):
    if stop is None:
        return []
    return [stop] if isinstance(stop, str) else list(stop)


def _chunk(cid, created, delta, finish=None):
    return "data: " + json.dumps({
        "id": cid, "object": "chat.completion.chunk", "created": created,
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }) + "\n\n"


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model",
                                        "owned_by": "deepseek"}]}


@app.get("/health")
def health():
    return {"status": "ok", **ENGINE.stats()}


def _run(ids, req, chat=True):
    cid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex[:24]
    created = int(time.time())
    stop = _stops(req.stop)
    gen = ENGINE.generate(ids, req.max_tokens or 512, req.temperature, stop)

    if not req.stream:
        text, n = "", 0
        for _, text in gen:
            n += 1
        for s in stop:
            if s and text.endswith(s):
                text = text[: -len(s)]
        body = {"id": cid, "created": created, "model": MODEL_ID,
                "usage": {"prompt_tokens": len(ids), "completion_tokens": n,
                          "total_tokens": len(ids) + n}}
        if chat:
            body["object"] = "chat.completion"
            body["choices"] = [{"index": 0, "finish_reason": "stop",
                                "message": {"role": "assistant", "content": text}}]
        else:
            body["object"] = "text_completion"
            body["choices"] = [{"index": 0, "finish_reason": "stop", "text": text}]
        return body

    def sse():
        if chat:
            yield _chunk(cid, created, {"role": "assistant", "content": ""})
        prev = ""
        for _, text in gen:
            delta, prev = text[len(prev):], text
            if not delta:
                continue
            if chat:
                yield _chunk(cid, created, {"content": delta})
            else:
                yield "data: " + json.dumps({
                    "id": cid, "object": "text_completion", "created": created,
                    "model": MODEL_ID,
                    "choices": [{"index": 0, "text": delta, "finish_reason": None}],
                }) + "\n\n"
        yield _chunk(cid, created, {}, "stop") if chat else "data: [DONE]\n\n"
        if chat:
            yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/v1/chat/completions")
def chat_completions(req: ChatRequest):
    text = ENGINE.render(req.messages, req.thinking_mode, req.reasoning_effort, req.tools)
    return _run(ENGINE.tok.encode(text), req, chat=True)


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    return _run(ENGINE.tok.encode(req.prompt), req, chat=False)


# ------------------------------------------------------------ anthropic /v1/messages
# Claude Code and the other Anthropic-native CLIs speak this instead of the OpenAI
# schema.  The wire format differs (content blocks, named SSE events); the engine and
# the checkpoint's own chat template are shared with the OpenAI path above.
class AnthropicRequest(BaseModel):
    model: str = MODEL_ID
    messages: List[Dict[str, Any]]
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    max_tokens: int = 4096
    temperature: float = 0.2
    stream: bool = False
    stop_sequences: Optional[List[str]] = None
    tools: Optional[List[Dict[str, Any]]] = None


def _flatten(content):
    """Anthropic content blocks -> the plain text the DeepSeek template expects."""
    if content is None or isinstance(content, str):
        return content or ""
    out = []
    for b in content:
        if not isinstance(b, dict):
            out.append(str(b))
        elif b.get("type") == "text":
            out.append(b.get("text", ""))
        elif b.get("type") == "tool_result":
            out.append(_flatten(b.get("content")))
        elif b.get("type") == "tool_use":
            out.append(json.dumps({"name": b.get("name"), "arguments": b.get("input")}))
    return "\n".join(x for x in out if x)


def _anthropic_msgs(req):
    msgs = []
    sys_txt = _flatten(req.system)
    if sys_txt:
        msgs.append({"role": "system", "content": sys_txt})
    for m in req.messages:
        msgs.append({"role": m.get("role", "user"),
                     "content": _flatten(m.get("content"))})
    # encoding_dsv4 wants OpenAI-shaped tools; Anthropic puts the schema in input_schema
    tools = [{"type": "function",
              "function": {"name": t.get("name"), "description": t.get("description", ""),
                           "parameters": t.get("input_schema", {})}}
             for t in (req.tools or [])] or None
    return msgs, tools


@app.post("/v1/messages")
def messages_endpoint(req: AnthropicRequest):
    msgs, tools = _anthropic_msgs(req)
    ids = ENGINE.tok.encode(ENGINE.render(msgs, tools=tools))
    stop = list(req.stop_sequences or [])
    gen = ENGINE.generate(ids, req.max_tokens, req.temperature, stop)
    mid = "msg_" + uuid.uuid4().hex[:24]

    if not req.stream:
        text, n = "", 0
        for _, text in gen:
            n += 1
        hit = next((s for s in stop if s and text.endswith(s)), None)
        if hit:
            text = text[: -len(hit)]
        return {"id": mid, "type": "message", "role": "assistant", "model": MODEL_ID,
                "content": [{"type": "text", "text": text}],
                "stop_reason": "stop_sequence" if hit else
                               ("max_tokens" if n >= req.max_tokens else "end_turn"),
                "stop_sequence": hit,
                "usage": {"input_tokens": len(ids), "output_tokens": n}}

    def ev(name, data):
        return f"event: {name}\ndata: {json.dumps(data)}\n\n"

    def sse():
        yield ev("message_start", {"type": "message_start", "message": {
            "id": mid, "type": "message", "role": "assistant", "model": MODEL_ID,
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": len(ids), "output_tokens": 0}}})
        yield ev("content_block_start", {"type": "content_block_start", "index": 0,
                                         "content_block": {"type": "text", "text": ""}})
        prev, n = "", 0
        for _, text in gen:
            delta, prev = text[len(prev):], text
            if not delta:
                continue
            n += 1
            yield ev("content_block_delta", {"type": "content_block_delta", "index": 0,
                                             "delta": {"type": "text_delta", "text": delta}})
        yield ev("content_block_stop", {"type": "content_block_stop", "index": 0})
        yield ev("message_delta", {"type": "message_delta",
                                   "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                   "usage": {"output_tokens": n}})
        yield ev("message_stop", {"type": "message_stop"})

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/v1/messages/count_tokens")
def count_tokens(req: AnthropicRequest):
    msgs, tools = _anthropic_msgs(req)
    return {"input_tokens": len(ENGINE.tok.encode(ENGINE.render(msgs, tools=tools)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=os.environ.get("DSV4_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("DSV4_PORT", 8000)))
    ap.add_argument("--seq-len", type=int, default=int(os.environ.get("DSV4_SEQ_LEN", 16384)))
    a = ap.parse_args()

    global ENGINE
    ENGINE = Engine(seq_len=a.seq_len,
                    vram_gib=float(os.environ.get("DSV4_VRAM_CACHE_GIB", 0)),
                    ram_gib=float(os.environ.get("DSV4_RAM_CACHE_GIB", 48)),
                    threads=int(os.environ.get("DSV4_IO_THREADS", 32)))
    print(f"listening on http://{a.host}:{a.port}  (seq_len {a.seq_len})", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, workers=1, log_level="warning",
                timeout_keep_alive=600)


if __name__ == "__main__":
    main()
