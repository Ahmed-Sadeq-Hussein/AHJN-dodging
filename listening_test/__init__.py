"""
listening_test -- a reusable MOS / PMOS listening test for any folder of audio.

    app.html      the rating app; open it and choose a folder, or serve it
                  next to a manifest.json
    manifest.py   builds manifest.json from a folder (same rules as the app)

See README.md in this folder for the folder convention and the manifest schema.
"""

from .manifest import (APP, DEFAULT_QUESTIONS, SCHEMA, install_app, scan,  # noqa: F401
                       validate, write)
