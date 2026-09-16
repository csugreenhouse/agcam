set -euo pipefail

# Move to the directory containing this script (repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# run the requestor.py in the background
python3 -u plant_requests/requestor/requestor.py &
REQUESTOR_PID=$!

#command to allow running .sh file : chmod +x run.sh