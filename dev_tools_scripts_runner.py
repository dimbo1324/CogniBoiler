"""Entry point only — all logic lives in scripts/runner/.

Edit that package, or scripts/runner/config/*.json, never this file.

  python dev_tools_scripts_runner.py            interactive menu
  python dev_tools_scripts_runner.py help       every script, with manuals
  python dev_tools_scripts_runner.py list       machine-readable catalog
  python dev_tools_scripts_runner.py <name>     run one script directly
"""

import sys

from scripts.runner import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
