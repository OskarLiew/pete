import typer

from pete.retrieval_qa import get_qa_chain

app = typer.Typer()


@app.command()
def main() -> None:
    chain = get_qa_chain()
    query: str = typer.prompt("Ask me anything", type=str)
    typer.echo(chain(query))


if __name__ == "__main__":
    app()
