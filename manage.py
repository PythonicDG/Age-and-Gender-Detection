#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'age_gender_app.settings')

    # Set PORT for Render deployment (default to 10000 if not provided)
    port = os.environ.get("PORT", "10000")

    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc

    # If running "runserver" command, bind to 0.0.0.0 and use the correct port
    if len(sys.argv) > 1 and sys.argv[1] == "runserver":
        sys.argv = ["manage.py", "runserver", f"0.0.0.0:{port}"]

    execute_from_command_line(sys.argv)

if __name__ == '__main__':
    main()
