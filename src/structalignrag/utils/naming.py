import re


def sanitize_model_name(name: str) -> str:
    # Keep consistent with HippoRAG: replace "/" with "_", and collapse other risky chars.
    if name is None:
        return "none"
    s = name.strip()
    s = s.replace("\\", "_").replace("/", "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "none"

