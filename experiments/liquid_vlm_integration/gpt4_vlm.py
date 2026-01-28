"""
Worker 5: GPT-4 VLM Integration
Uses GPT-4 for high-quality trajectory descriptions
"""
import torch
import os
from openai import OpenAI
from pathlib import Path


class GPT4VLM:
    """GPT-4 for trajectory description generation."""
    
    def __init__(self):
        """Initialize GPT-4 client."""
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("⚠️  OPENAI_API_KEY not set, using placeholder")
            self.client = None
        else:
            self.client = OpenAI(api_key=api_key)
            print(f"✅ GPT-4 API configured")
    
    def generate_description(
        self,
        embeddings: torch.Tensor,
        prompt: str = "Describe the 3D trajectory based on these visual-spatial features:"
    ) -> str:
        """
        Generate trajectory description using GPT-4.
        
        Args:
            embeddings: (1, 4096) Liquid fusion output
            prompt: Text prompt
            
        Returns:
            description: Generated text
        """
        if self.client is None:
            # Fallback for testing without API key
            return f"[GPT-4 Placeholder] A trajectory with embeddings of shape {embeddings.shape}"
        
        # Convert embeddings to summary statistics for GPT-4
        emb = embeddings.cpu().numpy().flatten()
        stats = {
            "mean": float(emb.mean()),
            "std": float(emb.std()),
            "min": float(emb.min()),
            "max": float(emb.max()),
            "norm": float((emb ** 2).sum() ** 0.5)
        }
        
        # Create prompt with embedding context
        full_prompt = f"""{prompt}

The visual-spatial embedding has the following characteristics:
- Mean activation: {stats['mean']:.3f}
- Std deviation: {stats['std']:.3f}
- L2 norm: {stats['norm']:.1f}
- Range: [{stats['min']:.3f}, {stats['max']:.3f}]

Based on these features extracted from multi-view videos and 3D reconstruction, describe the trajectory in natural language."""
        
        # Call GPT-4
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in analyzing 3D trajectories from visual-spatial features."},
                {"role": "user", "content": full_prompt}
            ],
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content.strip()
    
    def generate_batch(self, embeddings_batch: torch.Tensor) -> list:
        """Generate descriptions for multiple samples."""
        descriptions = []
        for i in range(len(embeddings_batch)):
            emb = embeddings_batch[i:i+1]
            desc = self.generate_description(emb)
            descriptions.append(desc)
        return descriptions
