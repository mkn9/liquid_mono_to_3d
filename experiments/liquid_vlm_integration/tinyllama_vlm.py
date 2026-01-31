"""
Worker 4: TinyLlama VLM Integration
Generates trajectory descriptions from Liquid embeddings
Updated with improved structured prompting to reduce hallucinations
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


def create_structured_prompt():
    """
    Create structured prompt for trajectory description.
    
    This improved prompt provides explicit constraints and structure
    to reduce hallucinations and improve description quality.
    
    Returns:
        str: Structured prompt with instructions and constraints
    """
    prompt = """You are analyzing a 3D trajectory from stereo camera tracking.

Describe ONLY what you observe about:
1. Path shape: Is it straight, curved, circular, spiral, or another pattern?
2. Direction of movement: Which axis (X, Y, or Z) shows the most change? 
3. Start and end positions: Approximate coordinates where the path begins and ends
4. Motion characteristics: Is it moving at constant speed, accelerating, or decelerating?

Be factual and specific. Use only what you see in the trajectory data.

DO NOT mention:
- Videos, URLs, or web links
- Tutorials or how-to guides  
- Made-up objects or scenarios
- Information not present in the data

Focus on the geometric and kinematic properties of the trajectory path.

Trajectory description:"""
    
    return prompt


class TinyLlamaVLM:
    """TinyLlama for trajectory description generation."""
    
    def __init__(self, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize TinyLlama model."""
        print(f"📥 Loading TinyLlama: {model_name}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Projection: 4096 (Liquid) -> 2048 (TinyLlama embedding dim)
        self.visual_projector = torch.nn.Linear(4096, 2048).to(self.model.device).half()
        
        print(f"✅ TinyLlama loaded on {self.model.device}")
    
    @torch.no_grad()
    def generate_description(self, embeddings: torch.Tensor, prompt: str = None) -> str:
        """
        Generate trajectory description from Liquid embeddings.
        
        Args:
            embeddings: (1, 4096) Liquid fusion output
            prompt: Text prompt (uses structured prompt if None)
            
        Returns:
            description: Generated text
        """
        # Use structured prompt by default
        if prompt is None:
            prompt = create_structured_prompt()
        
        # Project visual embeddings to TinyLlama space
        visual_tokens = self.visual_projector(embeddings.half().to(self.model.device))
        
        # Tokenize text prompt
        text_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        text_embeddings = self.model.get_input_embeddings()(text_inputs.input_ids)
        
        # Concatenate visual + text embeddings
        combined = torch.cat([visual_tokens.unsqueeze(1), text_embeddings], dim=1)
        
        # Generate
        outputs = self.model.generate(
            inputs_embeds=combined,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        # Decode
        description = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from output
        if prompt in description:
            description = description.split(prompt)[-1].strip()
        
        return description
    
    def generate_batch(self, embeddings_batch: torch.Tensor) -> list:
        """Generate descriptions for multiple samples."""
        descriptions = []
        for i in range(len(embeddings_batch)):
            emb = embeddings_batch[i:i+1]
            desc = self.generate_description(emb)
            descriptions.append(desc)
        return descriptions
