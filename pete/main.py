import typer
from retrieval_qa import get_qa_chain

app = typer.Typer()


@app.command()
def main() -> None:
    chain = get_qa_chain()
    query = typer.prompt("Ask")
    typer.echo(chain(query))


if __name__ == "__main__":
    app()
