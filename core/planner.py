# agentforge/core/planner.py
import json
from typing import Any, Dict, List, Optional, Tuple


class AdvancedPlanner:
    """
    AdvancedPlanner with Tree-of-Thought (ToT) style candidate generation + scoring.

    Usage:
      plan = await planner.plan(task, history, context=context, tools=tool_registry.list_schemas())

    Configurable via:
      - n_candidates: how many candidate solutions to ask model for (default 3)
      - candidate_temperature: sampling temperature when generating candidates
      - eval_temperature: temperature used when asking model to score candidates
      - max_retries: fallback retries
    """

    def __init__(
        self,
        llm_adapter,
        n_candidates: int = 3,
        candidate_temperature: float = 0.8,
        eval_temperature: float = 0.0,
        max_retries: int = 2,
    ):
        self.llm = llm_adapter
        self.n_candidates = n_candidates
        self.candidate_temperature = candidate_temperature
        self.eval_temperature = eval_temperature
        self.max_retries = max_retries

    # ----------------------------
    # Public entry
    # ----------------------------
    async def plan(
        self,
        task: str,
        history: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        High-level plan entrypoint. Will:
         1. Build base prompt (task + memory + history)
         2. Generate N candidates (JSON list)
         3. Score each candidate with a dedicated scoring prompt
         4. Optionally self-refine the top candidate
         5. Return selected candidate normalized to {"type":"tool",...} or {"type":"final",...}
        """

        # Build prompt base (human-readable block)
        base_prompt = self._build_prompt(task, history, context)

        # If tools present and adapter supports function-calling, we could use it.
        # But candidate generation is simpler via text JSON list — keep function-calling for direct 1-shot tool calls.
        # Generate candidates
        candidates = await self._generate_candidates(base_prompt, tools=tools, n=self.n_candidates)

        # If we failed to get candidates fallback to single-step planner
        if not candidates:
            return await self._ask_with_retries(base_prompt, tools=tools)

        # Score/evaluate candidates
        scored = []
        for cand in candidates:
            score, rationale = await self._score_candidate(task, history, context, cand)
            scored.append((score, cand, rationale))

        # sort by score desc
        scored.sort(key=lambda x: x[0], reverse=True)

        # Optional: allow the top candidate to self-refine (ask model to correct or expand it)
        top_score, top_cand, top_reason = scored[0]
        refined = await self._maybe_refine_candidate(base_prompt, top_cand)

        # normalize refined candidate (or top_cand if no refine)
        chosen = refined or top_cand

        # Ensure normalized output {type: "tool"|"final", ...}
        normalized = self._normalize_candidate(chosen)

        return normalized

    # ----------------------------
    # Candidate generation
    # ----------------------------
    async def _generate_candidates(self, base_prompt: str, tools: Optional[Dict[str, Any]] = None, n: int = 3) -> List[Dict[str, Any]]:
        """
        Ask the model to produce a JSON array of candidate solutions.
        Each candidate should be one of:
          - {"type":"tool", "name":"<tool>", "input":{...}}
          - {"type":"final", "output":"..."}
        We'll instruct the model strictly to output JSON only.
        """
        tools_block = self._format_tools_block(tools) if tools else "(no tools provided)"

        gen_prompt = f"""
You are an AI planner. Produce exactly {n} distinct candidate next steps for the agent.
Each candidate must be a JSON object and the full response must be a JSON array (list).
Allowed candidate forms:
 - Call a tool:
   {{"type":"tool", "name":"<tool-name>", "input": {{ ... }}}}
 - Final answer:
   {{"type":"final", "output":"..."}}

Context / instructions:
{base_prompt}

Available tools:
{tools_block}

Return EXACTLY {n} JSON candidate objects inside a JSON array. No extra text.
"""
        # call LLM (use sampling temperature for diversity)
        raw = await self.llm.generate(gen_prompt, temperature=self.candidate_temperature)
        text = raw if isinstance(raw, str) else getattr(raw, "text", str(raw))

        parsed = self._try_parse_json(text)
        if parsed and isinstance(parsed, list):
            # keep only dict candidates
            out = [p for p in parsed if isinstance(p, dict)]
            return out[:n]
        # parsing failed -> attempt a second try with lower temperature and stricter prompt
        retry_prompt = gen_prompt + "\nIMPORTANT: Output must be valid JSON array, nothing else."
        raw2 = await self.llm.generate(retry_prompt, temperature=0.0)
        text2 = raw2 if isinstance(raw2, str) else getattr(raw2, "text", str(raw2))
        parsed2 = self._try_parse_json(text2)
        if parsed2 and isinstance(parsed2, list):
            return [p for p in parsed2 if isinstance(p, dict)][:n]
        # failed entirely
        return []

    # ----------------------------
    # Candidate scoring
    # ----------------------------
    async def _score_candidate(self, task: str, history: List[Dict[str, Any]], context: Optional[Dict[str, Any]], candidate: Dict[str, Any]) -> Tuple[float, str]:
        """
        Ask the model to score a single candidate between 0.0 and 1.0 and provide a short rationale.
        Returns (score, rationale).
        """
        cand_text = json.dumps(candidate, ensure_ascii=False)
        hist = "\n".join([f"- action: {h['action']} | result: {str(h['result'])[:200]}" for h in history])

        score_prompt = f"""
You are an evaluator. Rate how good the following candidate is for the TASK and give a numeric score 0.0-1.0 (higher is better).
TASK:
{task}

HISTORY:
{hist if hist else '(no history)'}

CONTEXT:
{(context or {})}

CANDIDATE:
{cand_text}

Provide output as JSON exactly in the form:
{{"score": <float_between_0_and_1>, "rationale": "<one-sentence explanation>"}}
"""
        raw = await self.llm.generate(score_prompt, temperature=self.eval_temperature)
        text = raw if isinstance(raw, str) else getattr(raw, "text", str(raw))
        parsed = self._try_parse_json(text)
        if parsed and isinstance(parsed, dict) and "score" in parsed:
            try:
                s = float(parsed.get("score", 0.0))
                r = str(parsed.get("rationale", ""))[:1000]
                # clamp
                s = max(0.0, min(1.0, s))
                return s, r
            except Exception:
                pass
        # fallback: if parsing failed, try to extract a number from text
        try:
            # naive: find first float-like token
            import re
            m = re.search(r"([01](?:\.\d+)?)", text)
            if m:
                s = float(m.group(1))
                s = max(0.0, min(1.0, s))
                return s, "parsed fallback"
        except Exception:
            pass
        return 0.0, "failed to parse score"

    # ----------------------------
    # Candidate self-refinement (optional)
    # ----------------------------
    async def _maybe_refine_candidate(self, base_prompt: str, candidate: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Optionally ask the model to refine/repair the top candidate into a valid normalized form.
        Return refined candidate dict or None to keep original.
        """
        cand_text = json.dumps(candidate, ensure_ascii=False)
        refine_prompt = f"""
The agent will execute the following candidate next-step:
{cand_text}

Please check it for correctness and, if needed, return a corrected candidate in JSON form.
Return either the corrected candidate object or the original candidate object as JSON (no extra text).
"""
        raw = await self.llm.generate(refine_prompt, temperature=0.0)
        text = raw if isinstance(raw, str) else getattr(raw, "text", str(raw))
        parsed = self._try_parse_json(text)
        if parsed and isinstance(parsed, dict):
            return parsed
        return None

    # ----------------------------
    # Old single-shot path (fallback)
    # ----------------------------
    async def _ask_with_retries(self, prompt: str, tools: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Backwards-compatible single-step planner (tries to parse JSON, retry with refine prompt).
        """
        for attempt in range(self.max_retries + 1):
            raw_resp = await self.llm.generate(prompt)
            text = raw_resp if isinstance(raw_resp, str) else getattr(raw_resp, "text", str(raw_resp))
            parsed = self._try_parse_json(text)
            if parsed:
                # normalize
                if isinstance(parsed, dict) and parsed.get("type") == "tool":
                    args = parsed.get("arguments") or parsed.get("input") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            args = {"__raw": args}
                    return {"type": "tool", "name": parsed.get("name"), "input": args}
                if isinstance(parsed, dict) and parsed.get("type") == "final":
                    return {"type": "final", "output": parsed.get("output")}
            # retry refine
            refine_prompt = (
                "Your previous response was not valid JSON. "
                "Please produce EXACTLY one of these formats:\n"
                '{"type":"tool","name":"<tool_name>","input":{...}}\n'
                "OR\n"
                '{"type":"final","output":"..."}\n\n'
                f"Invalid response:\n{text}\n"
            )
            text = await self.llm.generate(refine_prompt)
        return {"type": "final", "output": "[Planner failed to produce valid JSON]"}

    # ----------------------------
    # Utilities: parse / normalize / prompt build
    # ----------------------------
    def _try_parse_json(self, text: str) -> Optional[Any]:
        if not text or not isinstance(text, str):
            return None
        try:
            return json.loads(text)
        except Exception:
            # try to find JSON substring
            try:
                start = text.index("[")
                end = text.rindex("]") + 1
                return json.loads(text[start:end])
            except Exception:
                try:
                    start = text.index("{")
                    end = text.rindex("}") + 1
                    return json.loads(text[start:end])
                except Exception:
                    return None

    def _normalize_candidate(self, cand: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure candidate uses unified schema:
         - {"type":"tool","name":...,"input":{...}}
         - {"type":"final","output":"..."}
        """
        if not isinstance(cand, dict):
            return {"type": "final", "output": str(cand)}
        t = cand.get("type")
        if t == "tool":
            name = cand.get("name") or cand.get("tool") or cand.get("func") or cand.get("function")
            inp = cand.get("input") or cand.get("arguments") or cand.get("args") or {}
            if isinstance(inp, str):
                try:
                    inp = json.loads(inp)
                except Exception:
                    inp = {"__raw": inp}
            return {"type": "tool", "name": name, "input": inp}
        if t == "final":
            return {"type": "final", "output": cand.get("output") or cand.get("text") or ""}
        # fallback: if candidate looks like a function call
        if cand.get("name"):
            inp = cand.get("input") or cand.get("arguments") or {}
            return {"type": "tool", "name": cand.get("name"), "input": inp}
        # last resort: string -> final
        return {"type": "final", "output": str(cand)}

    def _format_semantic(self, semantic_hits: List[Dict[str, Any]], max_chars: int = 1000) -> str:
        if not semantic_hits:
            return "(no semantic hits)"
        lines = []
        for hit in semantic_hits:
            score = hit.get("score")
            text = hit.get("text", "")[:max_chars].replace("\n", " ")
            lines.append(f"- score:{score:.4f} text: {text}")
        return "\n".join(lines)

    def _format_episodic(self, episodic: List[Dict[str, Any]], max_chars: int = 500) -> str:
        if not episodic:
            return "(no recent episodes)"
        lines = []
        for e in episodic:
            a = e.get("action")
            r = e.get("result")
            lines.append(f"- action: {a} | result: {str(r)[:max_chars]}")
        return "\n".join(lines)

    def _build_prompt(self, task: str, history: List[Dict[str, Any]], context: Optional[Dict[str, Any]] = None) -> str:
        hist = "\n".join([f"- action: {h['action']} | result: {str(h['result'])[:200]}" for h in history])
        if context is None:
            context = {"semantic": [], "episodic": []}

        semantic_block = self._format_semantic(context.get("semantic", []))
        episodic_block = self._format_episodic(context.get("episodic", []))

        last_tool = None
        if history:
            last_tool = history[-1].get("result")

        return f"""
You are an advanced AI planning system that uses:
- ReAct reasoning (Thought → Action → Observation)
- Tree-of-Thought planning (generate multiple candidate next steps)
- JSON structured output

TASK:
{task}

RELEVANT SEMANTIC MEMORY (top matches):
{semantic_block}

RECENT EPISODIC (last steps):
{episodic_block}

LAST TOOL OUTPUT:
{last_tool if last_tool is not None else '(none)'}

HISTORY:
{hist if hist else '(no previous steps)'}
"""

    def _format_tools_block(self, tools: Optional[Dict[str, Any]]) -> str:
        if not tools:
            return "(no tools available)"
        lines = []
        # tools may be a dict name->schema or list of schemas
        if isinstance(tools, dict):
            items = tools.items()
        else:
            items = [(t.get("name"), t) for t in tools]
        for name, schema in items:
            desc = schema.get("description", "")
            params = schema.get("parameters", {})
            lines.append(f"- {name}: {desc} | params: {json.dumps(params)}")
        return "\n".join(lines)

    


    

