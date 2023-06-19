from contextlib import contextmanager
from typing import Generator

from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn


@contextmanager
def progress_spinner(description: str) -> Generator[Progress, None, None]:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        progress.add_task(description, total=None)
        yield progress
