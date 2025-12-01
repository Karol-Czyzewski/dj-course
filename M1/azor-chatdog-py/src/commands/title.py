from cli import console

def rename_title_command(session, new_title: str) -> tuple[bool, str | None]:
    """Renames the title of the given session."""
    return session.rename_title(new_title)
