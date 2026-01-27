"""
Abstract interface for different LLM providers.

Allows easy swapping between GPT-4, Mistral, Phi-2, etc.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np


class LLMInterface(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_equation(
        self,
        trajectory_type: int,
        sample_points: Optional[np.ndarray] = None
    ) -> str:
        """Generate symbolic equation for trajectory.
        
        Args:
            trajectory_type: 0=linear, 1=circular, 2=helical, 3=parabolic
            sample_points: Optional sample points from trajectory
        
        Returns:
            str: Symbolic equation
        """
        pass
    
    @abstractmethod
    def generate_description(
        self,
        trajectory_type: int,
        sample_points: Optional[np.ndarray] = None
    ) -> str:
        """Generate natural language description.
        
        Args:
            trajectory_type: 0=linear, 1=circular, 2=helical, 3=parabolic
            sample_points: Optional sample points from trajectory
        
        Returns:
            str: Natural language description
        """
        pass


class GPT4Interface(LLMInterface):
    """GPT-4/GPT-4o interface via OpenAI API."""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        """Initialize GPT-4 interface.
        
        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4, gpt-4o, gpt-4-turbo)
        """
        self.api_key = api_key
        self.model = model
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package required: pip install openai")
    
    def generate_equation(
        self,
        trajectory_type: int,
        sample_points: Optional[np.ndarray] = None
    ) -> str:
        """Generate equation using GPT-4."""
        trajectory_names = ["linear", "circular", "helical", "parabolic"]
        traj_name = trajectory_names[trajectory_type]
        
        prompt = f"""Given a {traj_name} trajectory in 3D space, provide the mathematical equation that describes it.

Respond with ONLY the equation, no explanation. Use standard mathematical notation.

For reference:
- Linear: p(t) = p₀ + v·t
- Circular: x = r·cos(θ), y = r·sin(θ), z = constant
- Helical: x = r·cos(θ), y = r·sin(θ), z = a·t + b
- Parabolic: each dimension follows d(t) = a·t² + b·t + c

Equation for {traj_name}:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"GPT-4 API error: {e}")
            return self._fallback_equation(trajectory_type)
    
    def generate_description(
        self,
        trajectory_type: int,
        sample_points: Optional[np.ndarray] = None
    ) -> str:
        """Generate description using GPT-4."""
        trajectory_names = ["linear", "circular", "helical", "parabolic"]
        traj_name = trajectory_names[trajectory_type]
        
        prompt = f"""Describe the motion of an object following a {traj_name} trajectory in 3D space.

Provide a clear, concise description (1-2 sentences) suitable for a technical report.

Description:"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"GPT-4 API error: {e}")
            return self._fallback_description(trajectory_type)
    
    def _fallback_equation(self, trajectory_type: int) -> str:
        """Fallback equations if API fails."""
        equations = {
            0: "p(t) = p₀ + v·t",
            1: "x = r·cos(θ), y = r·sin(θ), z = c",
            2: "x = r·cos(θ), y = r·sin(θ), z = a·t + b",
            3: "x = a·t² + b·t + c (for each dimension)"
        }
        return equations[trajectory_type]
    
    def _fallback_description(self, trajectory_type: int) -> str:
        """Fallback descriptions if API fails."""
        descriptions = {
            0: "The object moves in a straight line with constant velocity.",
            1: "The object follows a circular path, maintaining a constant radius.",
            2: "The object traces a helical spiral, combining circular and linear motion.",
            3: "The object follows a parabolic trajectory with quadratic acceleration."
        }
        return descriptions[trajectory_type]


class MistralInterface(LLMInterface):
    """Mistral-7B-Instruct interface (local model)."""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """Initialize Mistral interface.
        
        Args:
            model_name: Hugging Face model name
        """
        self.model_name = model_name
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )
        except ImportError:
            raise ImportError("transformers package required: pip install transformers")
    
    def generate_equation(
        self,
        trajectory_type: int,
        sample_points: Optional[np.ndarray] = None
    ) -> str:
        """Generate equation using Mistral."""
        trajectory_names = ["linear", "circular", "helical", "parabolic"]
        traj_name = trajectory_names[trajectory_type]
        
        prompt = f"[INST] Provide the mathematical equation for a {traj_name} trajectory in 3D. Just the equation, no explanation. [/INST]"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.3,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract just the response part (after [/INST])
        if "[/INST]" in response:
            response = response.split("[/INST]")[1].strip()
        
        return response
    
    def generate_description(
        self,
        trajectory_type: int,
        sample_points: Optional[np.ndarray] = None
    ) -> str:
        """Generate description using Mistral."""
        trajectory_names = ["linear", "circular", "helical", "parabolic"]
        traj_name = trajectory_names[trajectory_type]
        
        prompt = f"[INST] Describe the motion of an object following a {traj_name} trajectory in 3D space. 1-2 sentences. [/INST]"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in response:
            response = response.split("[/INST]")[1].strip()
        
        return response


class Phi2Interface(LLMInterface):
    """Phi-2 interface (lightweight local model)."""
    
    def __init__(self, model_name: str = "microsoft/phi-2"):
        """Initialize Phi-2 interface.
        
        Args:
            model_name: Hugging Face model name
        """
        self.model_name = model_name
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        except ImportError:
            raise ImportError("transformers package required: pip install transformers")
    
    def generate_equation(
        self,
        trajectory_type: int,
        sample_points: Optional[np.ndarray] = None
    ) -> str:
        """Generate equation using Phi-2."""
        trajectory_names = ["linear", "circular", "helical", "parabolic"]
        traj_name = trajectory_names[trajectory_type]
        
        prompt = f"Q: What is the equation for a {traj_name} trajectory in 3D?\nA:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.3,
            do_sample=False
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract answer part
        if "A:" in response:
            response = response.split("A:")[1].strip()
        
        return response
    
    def generate_description(
        self,
        trajectory_type: int,
        sample_points: Optional[np.ndarray] = None
    ) -> str:
        """Generate description using Phi-2."""
        trajectory_names = ["linear", "circular", "helical", "parabolic"]
        traj_name = trajectory_names[trajectory_type]
        
        prompt = f"Q: Describe {traj_name} motion in 3D.\nA:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            do_sample=True
        )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "A:" in response:
            response = response.split("A:")[1].strip()
        
        return response


class LocalLLMInterface(LLMInterface):
    """Local LLM interface (TinyLlama, Phi-2, etc.) - No API keys needed!"""
    
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        """Initialize local LLM.
        
        Args:
            model_name: HuggingFace model name
                - TinyLlama/TinyLlama-1.1B-Chat-v1.0 (default, most stable)
                - microsoft/phi-2 (2.7B, more capable)
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            
            print(f"📥 Loading {model_name} locally...")
            
            self.model_name = model_name
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            self.device = next(self.model.parameters()).device
            print(f"✅ Model loaded on {self.device}")
            
        except ImportError:
            raise ImportError("transformers package required: pip install transformers")
    
    def _generate(self, prompt: str, max_tokens: int = 200) -> str:
        """Generate text from prompt."""
        import torch
        
        # Use chat template for TinyLlama
        if "TinyLlama" in self.model_name:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            formatted_prompt = prompt
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only new tokens
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def generate_equation(
        self,
        trajectory_type: int,
        sample_points: Optional[np.ndarray] = None
    ) -> str:
        """Generate equation using local LLM."""
        trajectory_names = ["linear", "circular", "helical", "parabolic"]
        traj_name = trajectory_names[trajectory_type]
        
        prompt = f"What is the mathematical equation for a {traj_name} trajectory in 3D space? Provide only the equation."
        
        return self._generate(prompt, max_tokens=100)
    
    def generate_description(
        self,
        trajectory_type: int,
        sample_points: Optional[np.ndarray] = None
    ) -> str:
        """Generate description using local LLM."""
        trajectory_names = ["linear", "circular", "helical", "parabolic"]
        traj_name = trajectory_names[trajectory_type]
        
        prompt = f"Describe a {traj_name} trajectory in 2-3 sentences. What are its characteristics?"
        
        return self._generate(prompt, max_tokens=150)


def get_llm_interface(provider: str, **kwargs) -> LLMInterface:
    """Factory function to get appropriate LLM interface.
    
    Args:
        provider: "local", "gpt4", "mistral", "phi2"
        **kwargs: Provider-specific arguments
    
    Returns:
        LLMInterface: Appropriate interface instance
    """
    if provider.lower() == "local":
        return LocalLLMInterface(**kwargs)
    elif provider.lower() in ["gpt4", "gpt-4", "gpt4o"]:
        return GPT4Interface(**kwargs)
    elif provider.lower() == "mistral":
        return MistralInterface(**kwargs)
    elif provider.lower() == "phi2":
        return Phi2Interface(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}")

