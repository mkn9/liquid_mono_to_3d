import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

class ChatLogger:
    """A class to manage chat history for the project.
    
    Implements hybrid approach:
    - JSON files in .chat_history/ for structured data
    - Markdown files for human-readable documentation
    - Both timestamped sessions and descriptive aggregated histories
    """
    
    def __init__(self, 
                 chat_dir: str = ".chat_history",
                 markdown_dir: str = "docs/chat_history"):
        """Initialize the chat logger.
        
        Args:
            chat_dir: Directory to store JSON chat history files
            markdown_dir: Directory to store markdown exports
        """
        self.chat_dir = Path(chat_dir)
        self.markdown_dir = Path(markdown_dir)
        self.chat_dir.mkdir(exist_ok=True)
        self.markdown_dir.mkdir(parents=True, exist_ok=True)
        
    def save_conversation(
        self,
        messages: List[Dict[str, str]],
        topic: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict] = None,
        auto_export_markdown: bool = True
    ) -> Dict[str, str]:
        """Save a conversation to JSON and optionally export to markdown.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            topic: Topic/title of the conversation
            tags: Optional list of tags for categorization
            metadata: Optional dictionary of additional metadata
            auto_export_markdown: If True, automatically export to markdown
            
        Returns:
            Dictionary with 'json' and 'markdown' paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{topic.replace(' ', '_')}.json"
        
        conversation_data = {
            "timestamp": timestamp,
            "topic": topic,
            "tags": tags or [],
            "metadata": metadata or {},
            "messages": messages
        }
        
        # Save JSON
        json_path = self.chat_dir / filename
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        result = {"json": str(json_path)}
        
        # Auto-export to markdown if requested
        if auto_export_markdown:
            md_path = self.export_to_markdown(filename)
            result["markdown"] = md_path
            
        return result
    
    def load_conversation(self, filename: str) -> Dict:
        """Load a conversation from a file.
        
        Args:
            filename: Name of the conversation file
            
        Returns:
            Dictionary containing the conversation data
        """
        file_path = self.chat_dir / filename
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_conversations(
        self,
        tags: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[Dict[str, Union[str, List[str]]]]:
        """List all conversations, optionally filtered by tags and date range.
        
        Args:
            tags: Optional list of tags to filter by
            start_date: Optional start date in YYYYMMDD format
            end_date: Optional end date in YYYYMMDD format
            
        Returns:
            List of conversation metadata
        """
        conversations = []
        
        for file_path in sorted(self.chat_dir.glob("*.json")):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Filter by tags if specified
                if tags and not all(tag in data["tags"] for tag in tags):
                    continue
                
                # Filter by date range if specified
                file_date = data["timestamp"][:8]  # YYYYMMDD part
                if start_date and file_date < start_date:
                    continue
                if end_date and file_date > end_date:
                    continue
                
                conversations.append({
                    "filename": file_path.name,
                    "topic": data["topic"],
                    "timestamp": data["timestamp"],
                    "tags": data["tags"]
                })
            except (json.JSONDecodeError, KeyError):
                continue
                
        return conversations
    
    def export_to_markdown(self, filename: str, output_path: Optional[str] = None) -> str:
        """Export a conversation to markdown format.
        
        Args:
            filename: Name of the conversation file (JSON)
            output_path: Optional path to save the markdown file
            
        Returns:
            Path to the generated markdown file
        """
        conversation = self.load_conversation(filename)
        
        if output_path is None:
            # Save to markdown directory with same base name
            md_filename = f"{filename[:-5]}.md"
            output_path = self.markdown_dir / md_filename
        else:
            output_path = Path(output_path)
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        markdown_content = [
            f"# {conversation['topic']}\n\n",
            f"**Date:** {conversation['timestamp']}\n\n",
            f"**Tags:** {', '.join(conversation['tags']) if conversation['tags'] else 'None'}\n\n"
        ]
        
        # Add metadata if present
        if conversation.get('metadata'):
            markdown_content.append("**Metadata:**\n")
            for key, value in conversation['metadata'].items():
                markdown_content.append(f"- {key}: {value}\n")
            markdown_content.append("\n")
        
        markdown_content.append("---\n\n## Conversation\n\n")
        
        for msg in conversation["messages"]:
            role = msg["role"].title()
            content = msg["content"]
            markdown_content.extend([
                f"### {role}\n\n",
                f"{content}\n\n"
            ])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("".join(markdown_content))
            
        return str(output_path)
    
    def append_to_aggregate(self, 
                           session_file: str,
                           aggregate_name: str = "chat_history_complete.md") -> str:
        """Append a session to an aggregated history file.
        
        Args:
            session_file: Name of the session JSON file
            aggregate_name: Name of the aggregate markdown file
            
        Returns:
            Path to the updated aggregate file
        """
        conversation = self.load_conversation(session_file)
        aggregate_path = Path(aggregate_name)
        
        # Read existing content if file exists
        if aggregate_path.exists():
            with open(aggregate_path, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        else:
            existing_content = f"# Complete Chat History\n\n"
            existing_content += f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            existing_content += "---\n\n"
        
        # Append new session
        new_content = [
            f"## {conversation['topic']}\n\n",
            f"**Date:** {conversation['timestamp']}\n\n",
            f"**Tags:** {', '.join(conversation['tags']) if conversation['tags'] else 'None'}\n\n"
        ]
        
        for msg in conversation["messages"]:
            role = msg["role"].title()
            content = msg["content"]
            new_content.extend([
                f"### {role}\n\n",
                f"{content}\n\n"
            ])
        
        new_content.append("---\n\n")
        
        # Write combined content
        with open(aggregate_path, 'w', encoding='utf-8') as f:
            f.write(existing_content)
            f.write("".join(new_content))
        
        return str(aggregate_path)
    
    def create_index(self) -> str:
        """Create an index of all conversations.
        
        Returns:
            Path to the generated index file
        """
        index_path = self.markdown_dir / "INDEX.md"
        conversations = self.list_conversations()
        
        content = [
            "# Chat History Index\n\n",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            f"Total Conversations: {len(conversations)}\n\n",
            "---\n\n"
        ]
        
        # Group by tags
        by_tags = {}
        for conv in conversations:
            for tag in conv['tags']:
                if tag not in by_tags:
                    by_tags[tag] = []
                by_tags[tag].append(conv)
        
        # Chronological list
        content.append("## Chronological\n\n")
        for conv in sorted(conversations, key=lambda x: x['timestamp'], reverse=True):
            tags_str = f" [{', '.join(conv['tags'])}]" if conv['tags'] else ""
            content.append(f"- **{conv['timestamp']}**: {conv['topic']}{tags_str}\n")
            content.append(f"  - JSON: `.chat_history/{conv['filename']}`\n")
            md_file = conv['filename'].replace('.json', '.md')
            content.append(f"  - Markdown: `docs/chat_history/{md_file}`\n\n")
        
        # By tags
        if by_tags:
            content.append("\n## By Tags\n\n")
            for tag in sorted(by_tags.keys()):
                content.append(f"### {tag}\n\n")
                for conv in sorted(by_tags[tag], key=lambda x: x['timestamp'], reverse=True):
                    content.append(f"- **{conv['timestamp']}**: {conv['topic']}\n")
        
        with open(index_path, 'w', encoding='utf-8') as f:
            f.write("".join(content))
        
        return str(index_path)

if __name__ == "__main__":
    # Example usage
    logger = ChatLogger()
    
    # Example conversation
    messages = [
        {"role": "user", "content": "How do I implement 3D tracking?"},
        {"role": "assistant", "content": "Here's how you can implement 3D tracking...\n\n```python\ndef track_3d():\n    pass\n```"}
    ]
    
    # Save the conversation (automatically exports to markdown)
    paths = logger.save_conversation(
        messages=messages,
        topic="3D Tracking Implementation",
        tags=["3d-tracking", "computer-vision"],
        metadata={"project": "mono_to_3d"}
    )
    
    print(f"Saved conversation:")
    print(f"  JSON: {paths['json']}")
    print(f"  Markdown: {paths['markdown']}")
    
    # Optionally append to aggregate history
    json_filename = Path(paths['json']).name
    aggregate_path = logger.append_to_aggregate(json_filename)
    print(f"  Appended to: {aggregate_path}")
    
    # Create index
    index_path = logger.create_index()
    print(f"  Index updated: {index_path}")
    
    # List all conversations
    conversations = logger.list_conversations()
    print(f"\nTotal conversations: {len(conversations)}")
    for conv in conversations:
        print(f"- {conv['topic']} ({conv['timestamp']})") 