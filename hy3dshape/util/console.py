from rich.console import Console

console = Console()

def get_console():
    return console

def print(*args, **kwargs):
    console.log(*args, **kwargs)