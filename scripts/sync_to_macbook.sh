#!/bin/bash
# Sync Results to MacBook (via git commits)
# Runs every 5 minutes to push results, status, and artifacts

set -e

PROJECT_DIR="$HOME/liquid_mono_to_3d"
cd "$PROJECT_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H%M")

echo "📤 Syncing results to MacBook at $TIMESTAMP..."

# Get current branch
CURRENT_BRANCH=$(git branch --show-current)

# Add all results, status, and monitoring files
git add results/ 2>/dev/null || true
git add status/ 2>/dev/null || true
git add monitoring/ 2>/dev/null || true
git add artifacts/ 2>/dev/null || true
git add logs/*.log 2>/dev/null || true

# Check if there are changes
if git diff --cached --quiet; then
    echo "📭 No new changes to sync"
else
    # Commit with timestamp
    git commit -m "📊 Sync: Results update from $CURRENT_BRANCH at $TIMESTAMP" || true
    
    # Push to remote
    git push origin "$CURRENT_BRANCH" || {
        echo "⚠️  Push failed, will retry next cycle"
        exit 0
    }
    
    echo "✅ Synced to GitHub - visible on MacBook"
fi

# Create sync status file
cat > monitoring/last_sync.txt << EOF
Last sync: $(date)
Branch: $CURRENT_BRANCH
Status: Success
EOF

echo "Done!"

