set -euo pipefail

# Move to the directory containing this script (repo root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
cd "$SCRIPT_DIR"

# run the main.py in the background to start processing.
python3 -u scripts/main.py&
REQUESTOR_PID=$!
#launch gui app in web browser.
streamlit run app/Home.py&
REQUESTOR_PID=$!


#command to allow running .sh file : chmod +x run.sh