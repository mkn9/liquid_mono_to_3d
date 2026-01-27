#!/bin/bash
# Search chat history for a specific term
#
# Usage: ./scripts/search_chat.sh <search_term>

SEARCH_TERM="$1"

if [ -z "$SEARCH_TERM" ]; then
    echo "Usage: $0 <search_term>"
    echo ""
    echo "Examples:"
    echo "  $0 triangulation"
    echo "  $0 'camera calibration'"
    echo "  $0 bug-fix"
    exit 1
fi

echo "=========================================="
echo "Searching chat history for: $SEARCH_TERM"
echo "=========================================="
echo ""

# Search in main aggregate history
if [ -f "chat_history_complete.md" ]; then
    echo "📄 Main History (chat_history_complete.md):"
    echo "-------------------------------------------"
    grep -n -i --color=auto "$SEARCH_TERM" chat_history_complete.md | head -20
    COUNT=$(grep -i "$SEARCH_TERM" chat_history_complete.md | wc -l | tr -d ' ')
    echo ""
    echo "Found $COUNT matches"
    echo ""
fi

# Search in markdown session files
if [ -d "docs/chat_history" ]; then
    echo "📝 Session Files (docs/chat_history/):"
    echo "--------------------------------------"
    grep -r -n -i --color=auto "$SEARCH_TERM" docs/chat_history/ 2>/dev/null | head -20
    COUNT=$(grep -r -i "$SEARCH_TERM" docs/chat_history/ 2>/dev/null | wc -l | tr -d ' ')
    echo ""
    echo "Found $COUNT matches"
    echo ""
fi

# Search in JSON files (show filename only)
if [ -d ".chat_history" ]; then
    echo "🔍 JSON Files (.chat_history/):"
    echo "-------------------------------"
    FILES=$(grep -l -i "$SEARCH_TERM" .chat_history/*.json 2>/dev/null)
    if [ -n "$FILES" ]; then
        echo "$FILES" | while read -r file; do
            echo "  - $(basename "$file")"
        done
    else
        echo "  No matches"
    fi
    echo ""
fi

# Search in experiment-specific histories
echo "🔬 Experiment Histories:"
echo "------------------------"
grep -r -n -i --color=auto "$SEARCH_TERM" experiments/*/CHAT_HISTORY*.md 2>/dev/null | head -20
COUNT=$(grep -r -i "$SEARCH_TERM" experiments/*/CHAT_HISTORY*.md 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "Found $COUNT matches"
echo ""

echo "=========================================="
echo "Search complete"
echo "=========================================="

