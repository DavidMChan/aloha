import click

from aloha.dataset import evaluate_dataset


@click.group()
def main() -> None:
    pass


main.add_command(evaluate_dataset)


if __name__ == "__main__":
    main()
