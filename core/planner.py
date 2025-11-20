# agentforge/core/planner.py
import json
from typing import Any, Dict, List, Optional


class AdvancedPlanner:
    """
    AdvancedPlanner with optional function-calling support.

    Usage:
      # pass tools from ToolRegistry.list_schemas()
      plan = await planner.plan(task, history, context=context, tools=tool_registry.list_schemas())
    """

    def __init__(self, llm_adapter, max_retries: int = 2):
        self.llm = llm_adapter
        self.max_retries = max_retries

    async def plan(
        self,
        task: str,
        history: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None,
        tools: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Main entry. `tools` should be a dict mapping tool_name -> schema (OpenAI-style function schema).
        If the underlying LLM adapter supports function-calling (method name generate_with_functions / call_with_functions),
        this will try to use it. Otherwise, it will inline tool schemas into the prompt.
        """
        # Build prompt (without tools block) - tools may be passed separately
        prompt = self._build_prompt(task, history, context)

        # Try function-calling path first if adapter exposes the helper
        if tools and hasattr(self.llm, "generate_with_functions"):
            return await self._plan_with_function_calling(prompt, tools)

        # Fallback: inject tool schemas into prompt text and use regular generate
        if tools:
            prompt = self._inject_tools_into_prompt(prompt, tools)

        return await self._ask_with_retries(prompt)

    # ---------------------
    # function-calling path (preferred)
    # ---------------------
    async def _plan_with_function_calling(self, prompt: str, tools: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapter'ın function-calling yeteneğini kullanarak plan üretir.
        Çıktı her zaman unified dict formatında döner:
        - tool çağrısı: {"type":"tool","name":..., "input":{...}}
        - final çıktı: {"type":"final","output":...}
        """
        functions = list(tools.values()) if isinstance(tools, dict) else tools

        try:
            raw = await self.llm.generate_with_functions(prompt, functions=functions)
        except Exception:
            # Fallback to text-based route
            return await self._ask_with_retries(self._inject_tools_into_prompt(prompt, tools))

        # --- normalize raw output ---
        if isinstance(raw, dict):
            choices = raw.get("choices") or []
            if choices:
                msg = choices[0].get("message") or {}
                func_call = msg.get("function_call") or msg.get("tool_call") or None
                if func_call:
                    fname = func_call.get("name")
                    args_raw = func_call.get("arguments") or func_call.get("input") or "{}"
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except Exception:
                        args = {"__raw": args_raw}
                    return {"type": "tool", "name": fname, "input": args}
                # Eğer final output ise
                output_text = msg.get("content") or choices[0].get("text") or ""
                return {"type": "final", "output": output_text}

        # --- fallback string output ---
        text = raw if isinstance(raw, str) else getattr(raw, "text", str(raw))
        parsed = self._try_parse_json(text)
        if parsed:
            # normalize parsed dict
            if parsed.get("type") == "tool" and parsed.get("name"):
                # arguments normalize
                args = parsed.get("arguments") or parsed.get("input") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except:
                        args = {"__raw": args}
                return {"type": "tool", "name": parsed.get("name"), "input": args}
            elif parsed.get("type") == "final" and "output" in parsed:
                return {"type": "final", "output": parsed["output"]}

        # fallback: treat as final text
        return {"type": "final", "output": text}


    # ---------------------
    # Regular prompt path with retries
    # ---------------------
    async def _ask_with_retries(self, prompt: str) -> Dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            raw_resp = await self.llm.generate(prompt)
            text = raw_resp if isinstance(raw_resp, str) else getattr(raw_resp, "text", str(raw_resp))
            parsed = self._try_parse_json(text)
            if parsed:
                # normalize tool/final output
                if parsed.get("type") == "tool":
                    args = parsed.get("arguments") or parsed.get("input") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except:
                            args = {"__raw": args}
                    return {"type": "tool", "name": parsed.get("name"), "input": args}
                elif parsed.get("type") == "final":
                    return {"type": "final", "output": parsed.get("output")}
            # retry with refine prompt
            refine_prompt = (
                "Your previous response was not valid JSON. "
                "Please produce EXACTLY one of these formats:\n"
                '{"type":"tool","name":"<tool_name>","input":{...}}\n'
                "OR\n"
                '{"type":"final","output":"..."}\n\n'
                f"Invalid response:\n{text}\n"
            )
            text = await self.llm.generate(refine_prompt)

        # fallback
        return {"type": "final", "output": "[Planner failed to produce valid JSON]"}

    # ---------------------
    # Utilities: parsing / prompt building / tools injection
    # ---------------------
    def _try_parse_json(self, text: str) -> Optional[Dict[str, Any]]:
        if not text or not isinstance(text, str):
            return None
        try:
            return json.loads(text)
        except Exception:
            # try to extract a JSON substring
            try:
                start = text.index("{")
                end = text.rindex("}") + 1
                return json.loads(text[start:end])
            except Exception:
                return None

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

        # include a short "last tool result" if present
        last_tool = None
        if history:
            last_tool = history[-1].get("result")

        return f"""
You are an advanced AI planning system that uses:
- ReAct reasoning (Thought → Action → Observation)
- Tree-of-Thought planning (multiple candidate solutions)
- JSON function calling (if supported by the model)
- Strict structured output

Your job is to decide the next step for the agent.

--------------------
TASK:
{task}
--------------------

RELEVANT SEMANTIC MEMORY (top matches):
{semantic_block}

RECENT EPISODIC (last steps):
{episodic_block}

LAST TOOL OUTPUT:
{last_tool if last_tool is not None else '(none)'}

HISTORY:
{hist if hist else '(no previous steps)'}

AVAILABLE TOOLS:
(note: when possible, we pass tool schemas using the model's function-calling API; otherwise use these descriptions)
{{TOOLS_PLACEHOLDER}}

REASONING INSTRUCTIONS:
1. Think step-by-step.
2. Prefer to use relevant memory if it helps.
3. Generate at least 1 candidate solution (you may generate more).
4. Evaluate candidate(s) briefly.
5. Decide EITHER:
   - Call a tool: {{"type":"tool","name":"<tool>","input":{{...}}}}
   - OR give final answer: {{"type":"final","output":"..."}}

OUTPUT MUST BE VALID JSON. NO EXTRA TEXT OUTSIDE THE JSON.
""".replace("{TOOLS_PLACEHOLDER}", self._format_tools_block(tools=context.get("tools") if context else None))

    def _format_tools_block(self, tools: Optional[Dict[str, Any]]) -> str:
        if not tools:
            return "(no tools available)"
        lines = []
        # tools may be a dict name->schema or list of schemas
        if isinstance(tools, dict):
            items = tools.items()
        else:
            # assume list of schemas
            items = [(t.get("name"), t) for t in tools]
        for name, schema in items:
            desc = schema.get("description", "")
            params = schema.get("parameters", {})
            lines.append(f"- {name}: {desc} | params: {json.dumps(params)}")
        return "\n".join(lines)
    


    
