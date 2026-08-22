"""
Prompt Construction and Templating Engine (Topics 3.13, 3.18)
"""
from pathlib import Path


def load_system_prompt() -> str:
    prompt_path = Path(__file__).resolve().parent.parent.parent / "prompts" / "diagnostic_system.md"
    if prompt_path.exists():
        return prompt_path.read_text(encoding="utf-8")
    return "You are an authorized enterprise automotive diagnostic assistant. Answer with strict grounding."


def build_technician_prompt(
    vehicle_model: str,
    model_year: int,
    variant: str,
    region: str,
    query: str,
    retrieved_context: str = "No external context supplied."
) -> str:
    return f"""
### VEHICLE CONTEXT
- Model: {vehicle_model}
- Model Year: {model_year}
- Variant/Powertrain: {variant}
- Market/Region: {region}

### RETRIEVED MANUAL CONTEXT
{retrieved_context}

### TECHNICIAN QUERY / SYMPTOM
{query}

Generate the diagnostic assessment adhering to the provided JSON schema.
"""