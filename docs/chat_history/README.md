# Chat History - Markdown Exports

This directory contains human-readable markdown exports of all chat conversations for the mono_to_3d project.

## Purpose

- **Human-readable** format for easy browsing and reviewing
- **Git-friendly** with clear diffs for version control
- **Searchable** using standard text tools (grep, IDE search, etc.)
- **Archival** documentation of development decisions and discussions

## Corresponding JSON Files

Each markdown file has a corresponding JSON file in `.chat_history/` with the same base name:

```
docs/chat_history/20260120_143022_Feature.md  ← You are here
.chat_history/20260120_143022_Feature.json    ← Structured data
```

The JSON files contain:
- Structured message data
- Metadata and tags
- Timestamps
- Additional context

## File Naming Convention

```
YYYYMMDD_HHMMSS_Descriptive_Topic.md

Examples:
- 20260120_143022_3D_Tracking_Implementation.md
- 20260120_151045_Camera_Calibration_Bug.md
- 20260121_093015_Performance_Optimization.md
```

## Usage

### Browse Files

Files are automatically sorted chronologically. Use your file manager or IDE to browse.

### Search Content

```bash
# Search for a term
grep -r "triangulation" .

# Case-insensitive search
grep -ri "camera" .

# Search with context (2 lines before and after)
grep -C 2 "bug" *.md
```

Or use the provided script:

```bash
../../scripts/search_chat.sh "search term"
```

### View Index

See `INDEX.md` for a complete organized list of all conversations by:
- Chronological order
- Tags/categories
- Topics

## Creating New Entries

**Option 1: Use ChatLogger**

```python
from chat_logger import ChatLogger

logger = ChatLogger()
paths = logger.save_conversation(
    messages=[...],
    topic="Topic Name",
    tags=["tag1", "tag2"]
)
# Automatically creates both JSON and markdown
```

**Option 2: Use Helper Script**

```bash
python ../../scripts/save_chat.py
```

**Option 3: Manual Creation**

If creating manually, follow the template:

```markdown
# Topic Name

**Date:** YYYYMMDD_HHMMSS

**Tags:** tag1, tag2, tag3

**Metadata:**
- project: mono_to_3d
- phase: implementation

---

## Conversation

### User

User's question or comment...

### Assistant

Assistant's response...

### User

Follow-up question...

### Assistant

Follow-up response...
```

## Best Practices

1. **Review before committing**: Check for sensitive information
2. **Use descriptive topics**: Make files easy to find later
3. **Tag consistently**: Use standard project tags
4. **Update INDEX.md**: Run `logger.create_index()` after adding files
5. **Commit to git**: All chat history should be version controlled

## Standard Tags

- **Technical**: `3d-tracking`, `camera-calibration`, `visualization`, `testing`, `bug-fix`, `performance`, `refactoring`
- **Phases**: `planning`, `implementation`, `debugging`, `review`, `optimization`
- **Experiments**: `magvit`, `dnerf`, `neural-video`, `sensor-analysis`

## See Also

- `requirements.md` Section 3.4 - Complete Chat History Protocol (procedures, workflows, FAQ)
- `README.md` - Project overview with chat history introduction
- `.chat_history/` - JSON structured data files

