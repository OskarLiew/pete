import langchain
import typer

from pete.chains.obsidian import get_obsidian_chain

app = typer.Typer()


@app.command()
def main(
    verbose: bool = typer.Option(False, "-v", "--verbose"),
) -> None:
    langchain.verbose = verbose
    chain = get_obsidian_chain()
    query: str = typer.prompt("Ask me anything", type=str)
    typer.echo(chain(query))


if __name__ == "__main__":
    app()
