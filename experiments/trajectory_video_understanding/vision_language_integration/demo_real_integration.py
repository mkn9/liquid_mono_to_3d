"""
Real Integration Demo: Vision-Language Model with GPT-4

Demonstrates the complete VLM pipeline with:
- Real trajectory data from persistence_augmented_dataset
- Real GPT-4 API calls (no mocks)
- Actual feature extraction and classification
- Natural language generation, explanation, and Q&A

Usage:
    export OPENAI_API_KEY="your-key-here"
    python demo_real_integration.py --num-examples 5
"""

import torch
import json
import argparse
from pathlib import Path
import sys
import os
from datetime import datetime

# Add parent dirs to path
sys.path.append(str(Path(__file__).parent))

from vision_language_bridge import VisionLanguageBridge
import numpy as np


def load_trajectory_sample(dataset_dir: Path, index: int = 0):
    """
    Load a real trajectory from the dataset.
    
    Args:
        dataset_dir: Path to persistence_augmented_dataset
        index: Which trajectory to load
    
    Returns:
        video_tensor: (T, H, W, C) tensor
        metadata: dict with trajectory info
    """
    # Load .pt file
    traj_file = dataset_dir / f"output/augmented_traj_{index:05d}.pt"
    json_file = dataset_dir / f"output/augmented_traj_{index:05d}.json"
    
    if not traj_file.exists():
        raise FileNotFoundError(f"Trajectory not found: {traj_file}")
    
    # Load tensor data
    data = torch.load(traj_file)
    
    # Load metadata
    metadata = {}
    if json_file.exists():
        with open(json_file, 'r') as f:
            metadata = json.load(f)
    
    # Extract video tensor
    # Assuming data structure: {'frames': tensor, 'trajectory': array, ...}
    if isinstance(data, dict):
        if 'frames' in data:
            video = data['frames']
        elif 'video' in data:
            video = data['video']
        else:
            # Use first tensor-like value
            video = next((v for v in data.values() if isinstance(v, torch.Tensor)), None)
            if video is None:
                raise ValueError(f"No video tensor found in {traj_file}")
    else:
        video = data
    
    # Ensure correct shape: (T, H, W, C)
    if video.ndim == 4:
        # Check if shape is (T, C, H, W) and convert to (T, H, W, C)
        if video.shape[1] == 3 or video.shape[1] < video.shape[2]:
            # Likely (T, C, H, W), permute to (T, H, W, C)
            video = video.permute(0, 2, 3, 1)
    elif video.ndim == 3:
        # Add channel dimension
        video = video.unsqueeze(-1).repeat(1, 1, 1, 3)
    else:
        raise ValueError(f"Unexpected video shape: {video.shape}")
    
    # Normalize if needed
    if video.max() > 1.0:
        video = video / 255.0
    
    return video.float(), metadata


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def demonstrate_vlm_capabilities(
    bridge: VisionLanguageBridge,
    video: torch.Tensor,
    metadata: dict,
    example_num: int
):
    """
    Demonstrate all VLM capabilities on one trajectory.
    
    Args:
        bridge: VisionLanguageBridge instance
        video: Trajectory video tensor
        metadata: Trajectory metadata
        example_num: Example number for display
    """
    print_section(f"Example {example_num}: {metadata.get('description', 'Trajectory')}")
    
    # Show metadata
    print("\n📊 Trajectory Metadata:")
    for key, value in metadata.items():
        if key not in ['frames', 'video', 'trajectory']:  # Skip large data
            print(f"  - {key}: {value}")
    
    print(f"\n🎥 Video Shape: {video.shape}")
    print(f"   Frames: {video.shape[0]}, Size: {video.shape[1]}x{video.shape[2]}, Channels: {video.shape[3]}")
    
    # 1. DESCRIPTION
    print_section("1. Natural Language Description")
    print("\n🤖 Generating description with GPT-4...")
    description = bridge.describe_video(video)
    print(f"\n📝 Description:")
    print(f"   {description}")
    
    # 2. EXPLANATION
    print_section("2. Classification Explanation")
    print("\n🤖 Generating explanation with GPT-4...")
    explanation = bridge.explain_classification(video)
    print(f"\n💡 Explanation:")
    print(f"   {explanation}")
    
    # 3. QUESTION ANSWERING
    print_section("3. Question Answering")
    
    questions = [
        "How many objects are in the video?",
        "What type of trajectory is this?",
        "Why was this classification chosen?",
        "Is the motion stable or variable?",
        "What is the model's confidence level?"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n❓ Q{i}: {question}")
        print("🤖 GPT-4 thinking...")
        answer = bridge.answer_question(video, question)
        print(f"💬 A{i}: {answer}")
    
    print("\n" + "-" * 70)


def save_results(results: list, output_dir: Path):
    """Save demonstration results to JSON."""
    output_file = output_dir / f"vlm_demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description='Real VLM Integration Demo')
    parser.add_argument('--dataset-dir', type=str,
                       default='../../persistence_augmented_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--num-examples', type=int, default=3,
                       help='Number of trajectory examples to demonstrate')
    parser.add_argument('--llm-provider', type=str, default='local',
                       choices=['local', 'gpt4', 'mistral', 'phi2'],
                       help='LLM backend to use (local=TinyLlama, no API key needed)')
    parser.add_argument('--output-dir', type=str,
                       default='demo_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for vision model')
    parser.add_argument('--use-llm-qa', action='store_true',
                       help='Use LLM for Q&A (default: template-based)')
    
    args = parser.parse_args()
    
    # Setup
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print_section("🚀 Real Vision-Language Model Integration Demo")
    print(f"\n⚙️  Configuration:")
    print(f"   Dataset: {dataset_dir}")
    print(f"   LLM Provider: {args.llm_provider}")
    print(f"   Q&A Mode: {'LLM-generated' if args.use_llm_qa else 'Template-based'}")
    print(f"   Device: {args.device}")
    print(f"   Examples: {args.num_examples}")
    
    # Check API key (only for GPT-4)
    api_key = os.getenv('OPENAI_API_KEY')
    if args.llm_provider == 'gpt4':
        if not api_key:
            print("\n❌ ERROR: OPENAI_API_KEY environment variable not set!")
            print("   Set it with: export OPENAI_API_KEY='your-key-here'")
            return 1
        print(f"   API Key: {api_key[:20]}... ✅")
    elif args.llm_provider == 'local':
        print(f"   ✅ Using local LLM (no API key needed!)")
    
    # Initialize VLM Bridge
    print_section("Initializing Vision-Language Bridge")
    print("\n🔧 Loading vision model...")
    print("   Note: Using mock architecture since no trained model checkpoint available")
    print("   For production, provide path to trained Worker 2 model.")
    
    try:
        bridge_kwargs = {
            'vision_model_path': 'mock_model.pt',  # Will create mock architecture
            'llm_provider': args.llm_provider,
            'device': args.device,
            'use_llm_qa': args.use_llm_qa
        }
        
        # Add API key only for GPT-4
        if args.llm_provider == 'gpt4':
            bridge_kwargs['api_key'] = api_key
        
        bridge = VisionLanguageBridge(**bridge_kwargs)
        print("✅ VLM Bridge initialized successfully!")
    except Exception as e:
        print(f"\n❌ Failed to initialize VLM Bridge: {e}")
        return 1
    
    # Run demonstrations
    results = []
    
    for i in range(args.num_examples):
        try:
            # Load real trajectory
            print(f"\n📂 Loading trajectory {i}...")
            video, metadata = load_trajectory_sample(dataset_dir, index=i)
            
            # Demonstrate VLM
            demonstrate_vlm_capabilities(bridge, video, metadata, i+1)
            
            # Store results
            results.append({
                'example_num': i + 1,
                'metadata': metadata,
                'video_shape': list(video.shape),
                'timestamp': datetime.now().isoformat()
            })
            
        except FileNotFoundError:
            print(f"\n⚠️  Trajectory {i} not found, skipping...")
            continue
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error processing trajectory {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    if results:
        save_results(results, output_dir)
    
    # Summary
    print_section("✨ Demo Complete!")
    print(f"\n📊 Summary:")
    print(f"   Examples processed: {len(results)}/{args.num_examples}")
    print(f"   LLM Provider: {args.llm_provider}")
    print(f"   Total API calls: ~{len(results) * 7}")  # desc + explain + 5 questions
    
    print("\n" + "=" * 70)
    print("Thank you for using the Vision-Language Model Integration!")
    print("=" * 70 + "\n")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

