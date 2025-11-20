from typing import Callable, Any, Dict, Optional
import inspect
import json


class ToolExecutionError(Exception):
    pass


class Tool:
    """
    Unified Tool interface:
    - name
    - callable function (sync/async)
    - JSON schema for LLM
    - safe execution with standardized ToolResult
    """

    def __init__(self, name: str, func: Callable, description: str = ""):
        self.name = name
        self.func = func
        self.description = description
        self.schema = self._build_schema(func, name, description)

    # ------------------------------------------------------------
    # Build JSON schema from Python function signature
    # ------------------------------------------------------------
    def _build_schema(self, func: Callable, name: str, description: str) -> Dict[str, Any]:
        sig = inspect.signature(func)
        params = {}

        for pname, p in sig.parameters.items():
            params[pname] = {
                "type": self._guess_json_type(p.annotation),
                "description": f"Parameter {pname}"
            }

        return {
            "name": name,
            "description": description or f"Tool {name}",
            "parameters": {
                "type": "object",
                "properties": params,
                "required": list(params.keys()),
            },
        }

    def _guess_json_type(self, annot):
        if annot in [int, float]:
            return "number"
        if annot == bool:
            return "boolean"
        if annot == dict:
            return "object"
        if annot == list:
            return "array"
        return "string"

    # ------------------------------------------------------------
    # Safe execution wrapper → unified ToolResult
    # ------------------------------------------------------------
    def run(self, inp: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns standardized tool result:
        {
            "type": "tool",
            "name": "<tool_name>",
            "success": True/False,
            "output": dict always,
            "raw": original result
        }
        """
        try:
            result = self.func(**inp)

            # normalize output
            if isinstance(result, str):
                output = {"text": result}
            elif isinstance(result, dict):
                output = result
            else:
                output = {"value": result}

            return {
                "type": "tool",
                "name": self.name,
                "success": True,
                "output": output,
                "raw": result
            }

        except Exception as e:
            return {
                "type": "tool",
                "name": self.name,
                "success": False,
                "output": {"error": str(e)},
                "raw": None
            }

    def __repr__(self):
        return f"<Tool name={self.name}>"


# ------------------------------------------------------------
# Tool Registry
# ------------------------------------------------------------
class ToolRegistry:
    """
    - Register tools
    - Lookup by name
    - Expose schemas for LLM
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def add(self, tool: Tool):
        if tool.name in self.tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def list_schemas(self) -> Dict[str, Any]:
        return {name: t.schema for name, t in self.tools.items()}


# ------------------------------------------------------------
# Built-in tools
# ------------------------------------------------------------
def http():
    def _run(url: str) -> str:
        import requests
        resp = requests.get(url, timeout=10)
        return resp.text

    return Tool("http_fetch", _run, description="Fetch content from a URL")


def calc():
    def _run(expr: str) -> Any:
        safe_env = {"__builtins__": None}
        return eval(expr, safe_env, {})

    return Tool("calc", _run, description="Evaluate a safe math expression")


def python_function(func: Callable, name: str = None, description: str = ""):
    return Tool(name or func.__name__, func, description)
 