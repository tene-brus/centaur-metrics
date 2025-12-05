"""Human Signal Operations - Inter-annotator agreement metrics."""

import sys
from pathlib import Path

# Add project root to path for cli.utils imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
