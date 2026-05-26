#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.log import setup_logging
from src.app import App


def main():
    setup_logging()
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
