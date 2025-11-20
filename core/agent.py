from __future__ import annotations
import json
from typing import List, Dict, Any, Optional
import asyncio


from agentforge.core.llm import LLMAdapter
from agentforge.core.tools import Tool, ToolRegistry
from agentforge.core.memory import MemorySystem
from agentforge.core.planner import AdvancedPlanner


class Agent:
    def __init__(
        self,
        model: str = "openai:gpt-4o-mini",
        tools: Optional[List[Tool]] = None,
        memory: Optional[MemorySystem] = None,
        max_steps: int = 7,
        semantic_k: int = 3,
        episodic_k: int = 3,
    ):
        self.model = model
        self.llm = LLMAdapter(model)

        self.tool_registry = ToolRegistry()
        for t in (tools or []):
            self.tool_registry.add(t)

        self.memory = memory or MemorySystem(self.llm)
        self.planner = AdvancedPlanner(llm_adapter=self.llm, max_retries=2)
        self.max_steps = max_steps

        # retrieval params
        self.semantic_k = semantic_k
        self.episodic_k = episodic_k

    def register_tool(self, tool: Tool):
        self.tool_registry.add(tool)

    # -------------------------
    # Non-streaming wrapper using run_stream
    # -------------------------
    async def run(self, task: str) -> str:
        final_text = ""
        async for ev in self.run_stream(task):
            if ev["type"] == "token":
                final_text += ev["text"]
            elif ev["type"] == "final":
                # if final contains consolidated text, prefer it
                if ev.get("text"):
                    final_text = ev["text"]
        return final_text

    # -------------------------
    # Streaming main loop
    # -------------------------
    async def run_stream(self, task: str):
        """
        Yields events:
        - {"type":"token","text": "..."} for streamed final outputs
        - {"type":"tool", "name":..., "input":..., "result":...} after tool execution
        - {"type":"final","text": "..."} when finished
        """
        history: List[Dict[str, Any]] = []

        for step in range(self.max_steps):
             # --- retrieve context from memory (RAG) ---
            context = await self.memory.retrieve_context(task, k_semantic=self.semantic_k, k_episodic=self.episodic_k)

            # ask planner with memory-aware context
            plan = await self.planner.plan(
                task,
                history,
                context=context,
                tool_schemas=self.tool_registry.list_schemas()
            )

            # --- FINAL RESPONSE ---
            if plan.get("type") == "final":
                final_prompt = plan.get("prompt") or plan.get("output", "")
                streamed_text = ""

                async for token in self.llm.stream(final_prompt):
                    streamed_text += token
                    yield {"type": "token", "text": token}

                # persist final to semantic memory
                await self.memory.add_knowledge(
                    text=streamed_text,
                    metadata={"source": "final_output", "task": task}
                )

                yield {"type": "final", "text": streamed_text}
                return

            # --- TOOL / FUNCTION CALL (unified) ---
            elif plan.get("type") in ("tool", "function_call"):
                tool_name = plan.get("name")
                args = plan.get("arguments") or plan.get("input") or {}

                tool = self.tool_registry.get(tool_name)
                if not tool:
                    raise RuntimeError(f"AgentError: Tool '{tool_name}' not found")

                # async/sync güvenli çalıştırma
                if asyncio.iscoroutinefunction(tool.func):
                    result = await tool.run(args)
                else:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, lambda: tool.run(args))

                # --- memory kayıtları ---
                history.append({"action": {"tool": tool_name, "input": args}, "result": result})
                await self.memory.add_episode(action=tool_name, result=result)

                # semantic memory için output normalize et
                output_text = ""
                if result.get("success"):
                    # eğer text varsa al, dict/other ise JSON stringify
                    if isinstance(result["output"], dict):
                        output_text = json.dumps(result["output"], ensure_ascii=False)
                    else:
                        output_text = str(result["output"])
                    await self.memory.add_knowledge(
                        text=output_text,
                        metadata={"source": f"tool:{tool_name}", "input": args}
                    )

                # agent → UI event olarak dön
                yield {"type": "tool", "name": tool_name, "input": args, "result": result}

            else:
                raise RuntimeError(f"AgentError: Unknown planner output type: {plan}")

        # exceeded max steps → return summary
        yield {"type": "final", "text": json.dumps(history, indent=2, ensure_ascii=False)}



        
