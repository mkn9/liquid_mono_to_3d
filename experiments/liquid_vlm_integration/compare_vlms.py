"""
Compare TinyLlama vs GPT-4 on trajectory description task
"""
import torch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from tinyllama_vlm import TinyLlamaVLM
from gpt4_vlm import GPT4VLM
from test_fusion_integration import test_real_data_fusion


def compare_models(num_samples: int = 2):
    """
    Compare TinyLlama and GPT-4 descriptions.
    
    Returns:
        results: Dict with descriptions from both models
    """
    print(f"\n🔬 Comparing TinyLlama vs GPT-4 on {num_samples} samples...")
    
    # Initialize models
    tinyllama = TinyLlamaVLM()
    gpt4 = GPT4VLM()
    
    results = {
        "tinyllama": [],
        "gpt4": [],
        "embeddings": []
    }
    
    # Generate descriptions for each sample
    for i in range(num_samples):
        print(f"\nSample {i+1}/{num_samples}:")
        
        # Get real data with Liquid fusion
        fusion_result = test_real_data_fusion()
        embeddings = fusion_result["llm_embeddings"]
        
        # TinyLlama description
        tinyllama_desc = tinyllama.generate_description(embeddings)
        print(f"  TinyLlama: {tinyllama_desc[:80]}...")
        
        # GPT-4 description
        gpt4_desc = gpt4.generate_description(embeddings)
        print(f"  GPT-4: {gpt4_desc[:80]}...")
        
        results["tinyllama"].append(tinyllama_desc)
        results["gpt4"].append(gpt4_desc)
        results["embeddings"].append(embeddings)
    
    return results
