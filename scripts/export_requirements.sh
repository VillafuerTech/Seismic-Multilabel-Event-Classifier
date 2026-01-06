#!/bin/bash
# Export requirements.txt from the active conda environment
# Source of truth: environment.yml
#
# Usage: ./scripts/export_requirements.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT="$REPO_ROOT/requirements.txt"

echo "Exporting requirements from conda environment..."

# Add header
cat > "$OUTPUT" << 'EOF'
# AUTO-GENERATED from environment.yml
# Do not edit manually. Run: scripts/export_requirements.sh
# Source of truth: environment.yml
#
EOF

# Export pip packages (excludes conda-only packages)
conda list --export | grep -v "^#" | grep "pypi" | cut -d'=' -f1 >> "$OUTPUT" 2>/dev/null || true

# If no pypi packages, use pip freeze instead
if [ ! -s "$OUTPUT" ] || [ "$(wc -l < "$OUTPUT")" -le 4 ]; then
    pip freeze --local | grep -v "^-e" >> "$OUTPUT"
fi

echo "Requirements exported to: $OUTPUT"
