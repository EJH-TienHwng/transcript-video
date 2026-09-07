from rich.table import Table
from rich.text import Text

from ..application.inspection import media_fields


def inspection_table(result: dict) -> Table:
    table = Table(title="Media inspection", expand=True)
    table.add_column("Field", style="info")
    table.add_column("Value", overflow="fold")
    for name, value in media_fields(result["metadata"]).items():
        table.add_row(name, Text(value))
    table.add_row("Path", Text(result["video"]))
    table.add_section()
    for name, path in result["artifacts"].items():
        state = result["artifact_states"][name]
        table.add_row(
            name.replace("_", " ").title(), Text(f"{state}" + (f" · {path}" if path else ""))
        )
    return table
