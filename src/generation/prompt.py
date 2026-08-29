from dataclasses import dataclass
from functools import lru_cache

import yaml

from src.config import REPO_ROOT

PROMPTS_DIR = REPO_ROOT / "prompts"


@dataclass(frozen=True)
class Prompt:
    version: str
    system: str
    user_template: str
    model: dict

    def render_user(self, *, context: str, question: str) -> str:
        return self.user_template.format(context=context, question=question)


@lru_cache(maxsize=8)
def load_prompt(version: str = "rag_v1") -> Prompt:
    """Load a versioned prompt from ``prompts/<version>.yaml``.

    Cached per version: the file is read once per process. A prompt change is a
    git commit to the YAML, deployed independently of code.
    """
    path = PROMPTS_DIR / f"{version}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No prompt version {version!r} at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Prompt(
        version=data["version"],
        system=data["system"].strip(),
        user_template=data["user"],
        model=data.get("model", {}),
    )
