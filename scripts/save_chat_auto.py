#!/usr/bin/env python3
"""Non-interactive chat history saver.

Usage:
    python scripts/save_chat_auto.py --topic "My Topic" --tags "tag1,tag2" --summary "Summary text"
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chat_logger import ChatLogger


def save_auto(topic, tags, summary, metadata=None):
    """Save chat history non-interactively."""
    logger = ChatLogger()
    
    # Create a simple message structure with summary
    messages = [
        {
            "role": "user",
            "content": "Review the whole project carefully along with the chat history. I believe a fair amount of chat history is missing for some reason. We have successfully trained and demonstrated magnet. Please find the complete history of what we've done with magnet and recommend what we can leverage there to train it and integrate it thoroughly with our project now."
        },
        {
            "role": "assistant",
            "content": summary
        }
    ]
    
    if metadata is None:
        metadata = {"project": "mono_to_3d"}
    
    print("=" * 60)
    print("Chat History Auto-Save")
    print("=" * 60)
    print(f"Topic: {topic}")
    print(f"Tags: {', '.join(tags)}")
    print(f"Messages: {len(messages)}")
    print()
    
    # Save conversation
    print("Saving conversation...")
    paths = logger.save_conversation(
        messages=messages,
        topic=topic,
        tags=tags,
        metadata=metadata
    )
    
    print("\n✅ Conversation saved:")
    print(f"   JSON: {paths['json']}")
    print(f"   Markdown: {paths['markdown']}")
    
    # Update index
    print("\nUpdating index...")
    index_path = logger.create_index()
    print(f"   Index: {index_path}")
    
    return paths


def main():
    parser = argparse.ArgumentParser(description='Save chat history non-interactively')
    parser.add_argument('--topic', required=True, help='Topic/title for the conversation')
    parser.add_argument('--tags', required=True, help='Comma-separated tags')
    parser.add_argument('--summary', required=True, help='Summary of the conversation')
    parser.add_argument('--metadata', help='Additional metadata as key=value pairs')
    
    args = parser.parse_args()
    
    tags = [t.strip() for t in args.tags.split(',') if t.strip()]
    
    metadata = {"project": "mono_to_3d"}
    if args.metadata:
        for pair in args.metadata.split(','):
            if '=' in pair:
                key, value = pair.split('=', 1)
                metadata[key.strip()] = value.strip()
    
    paths = save_auto(
        topic=args.topic,
        tags=tags,
        summary=args.summary,
        metadata=metadata
    )
    
    print("\n" + "=" * 60)
    print("Next Steps:")
    print("=" * 60)
    print("1. Review the saved files")
    print("2. Commit to git:")
    print(f"   git add .chat_history/ docs/chat_history/")
    print(f'   git commit -m "docs: add chat history for {args.topic}"')
    print()


if __name__ == "__main__":
    main()

