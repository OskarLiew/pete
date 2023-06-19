import json
import os
from pathlib import Path

import typer
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, validator

if _xdg_cache := os.getenv("XDG_CACHE_HOME"):
    _DEFAULT_CACHE_DIR = Path(_xdg_cache) / "pete"
else:
    _DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pete"

if _xdg_config := os.getenv("XDG_CONFIG_HOME"):
    CONFIG_DIR = Path(_xdg_config) / "pete"
else:
    CONFIG_DIR = Path.home() / ".config" / "pete"

CONFIG_FILE = CONFIG_DIR / "config.yaml"
DOTENV_FILE = CONFIG_DIR / ".env"


class Config(BaseModel):
    vault: Path
    cache_path: Path = _DEFAULT_CACHE_DIR

    @validator("vault")
    def _vault_exists(cls, value: Path) -> Path:
        if not value.exists():
            raise ValueError(f"'{value}' does not exist")
        return value


class Environment(BaseModel):
    OPENAI_API_KEY: str


def _load_config() -> Config:
    if not CONFIG_FILE.exists():
        _first_time_setup()

    with open(CONFIG_FILE, "r", encoding="utf-8") as file:
        obj = yaml.safe_load(file)

    return Config.parse_obj(obj)


def _first_time_setup() -> None:
    typer.echo("This is the first time setup for Pete, the personal assistant")
    config = Config(
        vault=typer.prompt("Vault path", type=Path),
        cache_path=typer.prompt("Cache path", default=_DEFAULT_CACHE_DIR, type=Path),
    )
    env = Environment(OPENAI_API_KEY=typer.prompt("OpenAI API key", type=str))

    typer.echo(f"Creating config files in '{CONFIG_DIR}'")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as file:
        yaml.dump(json.loads(config.json()), file)

    with open(DOTENV_FILE, "w", encoding="utf-8") as file:
        file.writelines(f"{key}={val}\n" for key, val in env.dict().items())

    typer.echo(f"Creating cache directory '{config.cache_path}'")
    config.cache_path.mkdir(parents=True, exist_ok=True)

    typer.echo("First time setup complete")


CONFIG = _load_config()
load_dotenv(DOTENV_FILE)
