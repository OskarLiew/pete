import json
import os
from pathlib import Path
from dotenv import load_dotenv
import typer
import yaml

from pydantic import BaseModel

from cache import CACHE_DIR


class Config(BaseModel):
    vault: Path
    cache_path: Path = CACHE_DIR


class Environment(BaseModel):
    OPENAI_API_KEY: str


CONFIG_DIR = Path(os.getenv("XDG_CONFIG_HOME", "~/.config")) / "pete"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
DOTENV_FILE = CONFIG_DIR / ".env"


def _load_config() -> Config:
    if not CONFIG_FILE.exists():
        _first_time_setup()

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        obj = yaml.safe_load(file)

    return Config.parse_obj(obj)


def _first_time_setup() -> None:
    config = Config(
        vault=typer.prompt("Vault path", type=Path),
        cache_path=typer.prompt("Cache path", default=CACHE_DIR, type=Path),
    )
    env = Environment(OPENAI_API_KEY=typer.prompt("OpenAI API key", type=str))

    typer.echo(f"First time config complete. Creating config files in {CONFIG_DIR}")
    CONFIG_DIR.mkdir(parents=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as file:
        yaml.dump(json.loads(config.json()), file)

    with open(DOTENV_FILE, "w", encoding="utf-8") as file:
        file.writelines(f"{key}={val}\n" for key, val in env.dict().items())


CONFIG = _load_config()
load_dotenv(DOTENV_FILE)
