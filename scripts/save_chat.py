#!/usr/bin/env python3
"""Quick script to save chat history.

This script provides an interactive interface for saving chat conversations
to the project's chat history system.

Usage:
    python scripts/save_chat.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chat_logger import ChatLogger


def quick_save():
    """Interactive chat saver with guided prompts."""
    logger = ChatLogger()
    
    print("=" * 60)
    print("Chat History Quick Save")
    print("=" * 60)
    print()
    
    # Get topic
    topic = input("Topic (descriptive name): ").strip()
    if not topic:
        print("Error: Topic is required")
        return
    
    # Get tags
    print("\nCommon tags:")
    print("  Technical: 3d-tracking, camera-calibration, visualization,")
    print("             testing, bug-fix, performance, refactoring")
    print("  Phases: planning, implementation, debugging, review")
    print("  Experiments: magvit, dnerf, neural-video, sensor-analysis")
    tags_input = input("\nTags (comma-separated): ").strip()
    tags = [t.strip() for t in tags_input.split(",") if t.strip()]
    
    # Get messages
    print("\n" + "-" * 60)
    print("Enter conversation messages")
    print("Type 'DONE' when finished")
    print("-" * 60)
    print()
    
    messages = []
    while True:
        role = input("Role (user/assistant/DONE): ").strip().lower()
        
        if role == 'done':
            break
            
        if role not in ['user', 'assistant']:
            print("Invalid role. Use 'user' or 'assistant'")
            continue
        
        print(f"\n{role.title()} content (end with line containing only '---'):")
        lines = []
        while True:
            line = input()
            if line.strip() == '---':
                break
            lines.append(line)
        
        content = "\n".join(lines)
        if content.strip():
            messages.append({"role": role, "content": content})
            print(f"✓ Added {role} message ({len(content)} chars)")
        print()
    
    if not messages:
        print("Error: No messages provided")
        return
    
    # Save conversation
    print("\nSaving conversation...")
    paths = logger.save_conversation(
        messages=messages,
        topic=topic,
        tags=tags,
        metadata={"project": "mono_to_3d"}
    )
    
    print("\n✅ Conversation saved:")
    print(f"   JSON: {paths['json']}")
    print(f"   Markdown: {paths['markdown']}")
    
    # Append to aggregate
    append = input("\nAppend to chat_history_complete.md? (y/n): ").strip().lower()
    if append == 'y':
        json_filename = Path(paths['json']).name
        aggregate_path = logger.append_to_aggregate(json_filename)
        print(f"   Appended to: {aggregate_path}")
    
    # Update index
    print("\nUpdating index...")
    index_path = logger.create_index()
    print(f"   Index: {index_path}")
    
    # Git instructions
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Review the saved files")
    print("2. Commit to git:")
    print(f"   git add .chat_history/ docs/chat_history/")
    if append == 'y':
        print(f"   git add chat_history_complete.md")
    print(f'   git commit -m "docs: add chat history for {topic}"')
    print()


def batch_save():
    """Save conversation from a text file.
    
    File format:
        TOPIC: Topic name here
        TAGS: tag1, tag2, tag3
        
        USER:
        User message content
        
        ASSISTANT:
        Assistant response content
        
        USER:
        Another user message
        
        (etc.)
    """
    if len(sys.argv) < 2:
        print("Usage: python scripts/save_chat.py <file.txt>")
        return
    
    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return
    
    logger = ChatLogger()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Parse file
    lines = content.split('\n')
    topic = None
    tags = []
    messages = []
    current_role = None
    current_content = []
    
    for line in lines:
        if line.startswith('TOPIC:'):
            topic = line[6:].strip()
        elif line.startswith('TAGS:'):
            tags = [t.strip() for t in line[5:].split(',') if t.strip()]
        elif line.startswith('USER:'):
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": '\n'.join(current_content).strip()
                })
            current_role = 'user'
            current_content = []
        elif line.startswith('ASSISTANT:'):
            if current_role and current_content:
                messages.append({
                    "role": current_role,
                    "content": '\n'.join(current_content).strip()
                })
            current_role = 'assistant'
            current_content = []
        elif current_role:
            current_content.append(line)
    
    # Add last message
    if current_role and current_content:
        messages.append({
            "role": current_role,
            "content": '\n'.join(current_content).strip()
        })
    
    if not topic or not messages:
        print("Error: File must contain TOPIC and at least one message")
        return
    
    # Save
    paths = logger.save_conversation(
        messages=messages,
        topic=topic,
        tags=tags,
        metadata={"project": "mono_to_3d"}
    )
    
    print(f"✅ Saved from {file_path}:")
    print(f"   JSON: {paths['json']}")
    print(f"   Markdown: {paths['markdown']}")
    
    # Update index
    index_path = logger.create_index()
    print(f"   Index: {index_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        batch_save()
    else:
        quick_save()

