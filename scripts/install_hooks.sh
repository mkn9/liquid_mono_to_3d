#!/usr/bin/env bash
# Install Git hooks for TDD evidence enforcement
# Usage: bash scripts/install_hooks.sh

set -euo pipefail

# Get the directory of the Git repository
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo ".")
cd "$REPO_ROOT" || exit 1

HOOKS_DIR="$REPO_ROOT/.git/hooks"
SCRIPTS_DIR="$REPO_ROOT/scripts"

# Check if we're in a git repository
if [[ ! -d ".git" ]]; then
  echo "❌ Error: Not in a git repository root"
  echo "   Run this script from the repository root"
  exit 1
fi

# Check if hooks directory exists
if [[ ! -d "$HOOKS_DIR" ]]; then
  echo "❌ Error: Git hooks directory not found: $HOOKS_DIR"
  exit 1
fi

# Check if hook template exists
HOOK_TEMPLATE="$SCRIPTS_DIR/pre-push.hook"
if [[ ! -f "$HOOK_TEMPLATE" ]]; then
  echo "❌ Error: Hook template not found: $HOOK_TEMPLATE"
  exit 1
fi

echo "Installing Git hooks for TDD evidence enforcement..."
echo ""

# Install pre-push hook
PRE_PUSH_HOOK="$HOOKS_DIR/pre-push"

# Check if hook already exists
if [[ -f "$PRE_PUSH_HOOK" ]]; then
  # Check if it's our hook
  if grep -q "Evidence validator" "$PRE_PUSH_HOOK" 2>/dev/null; then
    echo "⚠️  Pre-push hook already installed (updating)"
    cp "$PRE_PUSH_HOOK" "$PRE_PUSH_HOOK.backup"
    echo "   Backup saved: $PRE_PUSH_HOOK.backup"
  else
    echo "⚠️  WARNING: Existing pre-push hook found!"
    echo "   Current hook: $PRE_PUSH_HOOK"
    echo ""
    echo "   Choose an option:"
    echo "   1) Backup existing hook and install new one"
    echo "   2) Append to existing hook (manual merge needed)"
    echo "   3) Cancel installation"
    echo ""
    read -p "   Your choice (1/2/3): " choice
    
    case "$choice" in
      1)
        BACKUP="$PRE_PUSH_HOOK.backup.$(date +%Y%m%d_%H%M%S)"
        cp "$PRE_PUSH_HOOK" "$BACKUP"
        echo "   Existing hook backed up to: $BACKUP"
        ;;
      2)
        echo ""
        echo "   Please manually merge these files:"
        echo "   - Existing: $PRE_PUSH_HOOK"
        echo "   - New: $HOOK_TEMPLATE"
        exit 0
        ;;
      3)
        echo "   Installation cancelled"
        exit 0
        ;;
      *)
        echo "   Invalid choice - installation cancelled"
        exit 1
        ;;
    esac
  fi
fi

# Install the hook
cp "$HOOK_TEMPLATE" "$PRE_PUSH_HOOK"
chmod +x "$PRE_PUSH_HOOK"

echo "✅ Pre-push hook installed: $PRE_PUSH_HOOK"
echo ""
echo "========================================="
echo "Installation Complete"
echo "========================================="
echo ""
echo "What this does:"
echo "  - Runs before every 'git push'"
echo "  - Checks if source/test files changed"
echo "  - Validates evidence files exist"
echo "  - Blocks push if evidence missing"
echo ""
echo "To skip validation (exploratory work):"
echo "  SKIP_EVIDENCE=1 git push"
echo ""
echo "To test the hook:"
echo "  git push --dry-run"
echo ""
echo "To uninstall:"
echo "  rm $PRE_PUSH_HOOK"
echo ""

