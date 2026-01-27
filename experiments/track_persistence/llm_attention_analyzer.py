#!/usr/bin/env python3
"""
LLM Attention Analyzer
=====================
Uses LLM to analyze attention patterns from the Transformer model and generate insights.

This helps us understand:
- What makes a track persistent vs transient?
- Which temporal patterns indicate persistence?
- When does the model fail and why?
- What improvements could be made?
"""

import openai
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json


class LLMAttentionAnalyzer:
    """Analyzes Transformer attention patterns using LLM reasoning."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4-turbo"
    ):
        """
        Initialize LLM analyzer.
        
        Args:
            api_key: OpenAI API key (or uses OPENAI_API_KEY env var)
            model: OpenAI model to use
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.client = openai.OpenAI(api_key=self.api_key)
        self.model = model
    
    def analyze_attention_patterns(
        self,
        attention_data: List[Dict],
        track_metadata: List[Dict]
    ) -> Dict:
        """
        Analyze attention patterns across multiple tracks.
        
        Args:
            attention_data: List of attention weight dictionaries
                Each dict: {
                    'track_id': str,
                    'attention_weights': np.ndarray,  # (T,)
                    'is_persistent': bool,
                    'confidence': float,
                    'duration': int
                }
            track_metadata: Additional metadata about tracks
            
        Returns:
            analysis: Dictionary with insights
        """
        # Prepare data for LLM
        summary = self._summarize_attention_data(attention_data)
        
        # Create prompt
        prompt = self._create_analysis_prompt(summary, attention_data)
        
        # Query LLM
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in deep learning and computer vision, specializing in attention mechanisms and video understanding."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2000
        )
        
        analysis_text = response.choices[0].message.content
        
        # Structure the analysis
        analysis = {
            'summary': summary,
            'llm_insights': analysis_text,
            'patterns_identified': self._extract_patterns(analysis_text),
            'recommendations': self._extract_recommendations(analysis_text)
        }
        
        return analysis
    
    def _summarize_attention_data(
        self,
        attention_data: List[Dict]
    ) -> Dict:
        """
        Create statistical summary of attention patterns.
        
        Args:
            attention_data: List of attention weight dictionaries
            
        Returns:
            summary: Statistical summary
        """
        persistent_tracks = [d for d in attention_data if d['is_persistent']]
        transient_tracks = [d for d in attention_data if not d['is_persistent']]
        
        summary = {
            'total_tracks': len(attention_data),
            'persistent_count': len(persistent_tracks),
            'transient_count': len(transient_tracks),
            'persistent_stats': self._compute_attention_stats(persistent_tracks),
            'transient_stats': self._compute_attention_stats(transient_tracks)
        }
        
        return summary
    
    def _compute_attention_stats(
        self,
        tracks: List[Dict]
    ) -> Dict:
        """
        Compute statistics for attention weights.
        
        Args:
            tracks: List of track dictionaries
            
        Returns:
            stats: Dictionary of statistics
        """
        if not tracks:
            return {}
        
        # Collect all attention weights
        all_weights = []
        peak_positions = []  # Where is max attention?
        peak_values = []
        avg_durations = []
        
        for track in tracks:
            weights = track['attention_weights']
            if weights is not None and len(weights) > 0:
                all_weights.append(weights)
                peak_pos = np.argmax(weights)
                peak_positions.append(peak_pos / len(weights))  # Normalized position
                peak_values.append(weights[peak_pos])
                avg_durations.append(track['duration'])
        
        if not all_weights:
            return {}
        
        stats = {
            'avg_duration': float(np.mean(avg_durations)),
            'peak_position_normalized': {
                'mean': float(np.mean(peak_positions)),
                'std': float(np.std(peak_positions))
            },
            'peak_attention_value': {
                'mean': float(np.mean(peak_values)),
                'std': float(np.std(peak_values))
            },
            'attention_distribution': {
                'early_frames': float(np.mean([w[:len(w)//3].mean() for w in all_weights])),
                'middle_frames': float(np.mean([w[len(w)//3:2*len(w)//3].mean() for w in all_weights])),
                'late_frames': float(np.mean([w[2*len(w)//3:].mean() for w in all_weights]))
            }
        }
        
        return stats
    
    def _create_analysis_prompt(
        self,
        summary: Dict,
        attention_data: List[Dict]
    ) -> str:
        """
        Create prompt for LLM analysis.
        
        Args:
            summary: Statistical summary
            attention_data: Raw attention data
            
        Returns:
            prompt: Analysis prompt
        """
        prompt = f"""I've trained a Transformer model to classify 2D object tracks as persistent (long-duration, real objects) or transient (brief, noise, false positives) for 3D reconstruction.

The model uses attention mechanisms to focus on different temporal frames when making its decision.

Here's the attention pattern analysis:

**Overall Statistics:**
- Total tracks analyzed: {summary['total_tracks']}
- Persistent tracks: {summary['persistent_count']}
- Transient tracks: {summary['transient_count']}

**Persistent Track Attention Patterns:**
```json
{json.dumps(summary['persistent_stats'], indent=2)}
```

**Transient Track Attention Patterns:**
```json
{json.dumps(summary['transient_stats'], indent=2)}
```

**Example Attention Weights:**

Persistent Track Example:
- Duration: {attention_data[0]['duration']} frames
- Attention weights: {attention_data[0]['attention_weights'][:10].tolist() if attention_data[0]['attention_weights'] is not None else 'N/A'}...

Please analyze these patterns and answer:

1. **What temporal patterns distinguish persistent from transient tracks?**
   - Where does the model focus its attention for each class?
   - Are there clear differences in attention distribution?

2. **What do these attention patterns reveal about persistence?**
   - What visual/temporal cues might the model be using?
   - Does the model focus on early, middle, or late frames?

3. **What are potential failure modes?**
   - When might the model misclassify tracks?
   - What edge cases should we be aware of?

4. **What improvements would you suggest?**
   - How could we make the model more robust?
   - What additional features or architectural changes might help?

Please provide concrete, actionable insights based on the attention patterns.
"""
        
        return prompt
    
    def _extract_patterns(self, analysis_text: str) -> List[str]:
        """
        Extract identified patterns from LLM response.
        
        Args:
            analysis_text: LLM response text
            
        Returns:
            patterns: List of identified patterns
        """
        # Simple extraction - look for numbered lists or bullet points
        patterns = []
        lines = analysis_text.split('\n')
        
        in_patterns_section = False
        for line in lines:
            if 'pattern' in line.lower() or 'distinguish' in line.lower():
                in_patterns_section = True
            elif line.strip().startswith(('1.', '2.', '3.', '-', '•')):
                if in_patterns_section:
                    patterns.append(line.strip())
        
        return patterns if patterns else ['See full analysis']
    
    def _extract_recommendations(self, analysis_text: str) -> List[str]:
        """
        Extract recommendations from LLM response.
        
        Args:
            analysis_text: LLM response text
            
        Returns:
            recommendations: List of recommendations
        """
        recommendations = []
        lines = analysis_text.split('\n')
        
        in_recommendations_section = False
        for line in lines:
            if 'improve' in line.lower() or 'suggest' in line.lower() or 'recommendation' in line.lower():
                in_recommendations_section = True
            elif line.strip().startswith(('1.', '2.', '3.', '-', '•')):
                if in_recommendations_section:
                    recommendations.append(line.strip())
        
        return recommendations if recommendations else ['See full analysis']
    
    def analyze_failure_cases(
        self,
        false_positives: List[Dict],
        false_negatives: List[Dict]
    ) -> Dict:
        """
        Analyze why the model failed on specific tracks.
        
        Args:
            false_positives: Tracks incorrectly classified as persistent
            false_negatives: Tracks incorrectly classified as transient
            
        Returns:
            failure_analysis: Dictionary with insights
        """
        prompt = f"""I have a Transformer-based persistence classifier that made the following errors:

**False Positives** (classified as persistent, but actually transient):
{json.dumps([{
    'duration': fp['duration'],
    'confidence': fp['confidence'],
    'attention_weights': fp['attention_weights'][:10].tolist() if fp['attention_weights'] is not None else 'N/A'
} for fp in false_positives[:3]], indent=2)}

**False Negatives** (classified as transient, but actually persistent):
{json.dumps([{
    'duration': fn['duration'],
    'confidence': fn['confidence'],
    'attention_weights': fn['attention_weights'][:10].tolist() if fn['attention_weights'] is not None else 'N/A'
} for fn in false_negatives[:3]], indent=2)}

Please explain:
1. Why might the model have made these mistakes?
2. What patterns do the false positives have in common?
3. What patterns do the false negatives have in common?
4. How can we fix these failure modes?

Provide specific, technical insights.
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in deep learning debugging and failure analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        failure_analysis = {
            'llm_explanation': response.choices[0].message.content,
            'false_positive_count': len(false_positives),
            'false_negative_count': len(false_negatives)
        }
        
        return failure_analysis
    
    def generate_research_questions(
        self,
        analysis_results: Dict
    ) -> List[str]:
        """
        Generate research questions based on analysis.
        
        Args:
            analysis_results: Results from previous analyses
            
        Returns:
            questions: List of research questions
        """
        prompt = f"""Based on this analysis of a Transformer attention model for track persistence:

{json.dumps(analysis_results, indent=2)}

Generate 5-10 research questions that could lead to improvements or deeper understanding of the system.

Focus on:
- Architectural improvements
- Feature engineering
- Multi-modal fusion (visual + motion)
- Temporal modeling
- Robustness to edge cases

Format as a numbered list of concrete, answerable research questions.
"""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a computer vision researcher generating research questions."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=1000
        )
        
        text = response.choices[0].message.content
        
        # Extract questions
        questions = []
        for line in text.split('\n'):
            if line.strip() and (line.strip()[0].isdigit() or line.strip().startswith(('Q:', 'Question'))):
                questions.append(line.strip())
        
        return questions


def main():
    """Demo LLM attention analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze attention patterns with LLM')
    parser.add_argument('--attention-data', type=str, required=True,
                        help='Path to attention data JSON file')
    parser.add_argument('--output-dir', type=str, default='experiments/track_persistence/output/llm_analysis',
                        help='Output directory for analysis')
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenAI API key (or set OPENAI_API_KEY env var)')
    
    args = parser.parse_args()
    
    # Load attention data
    with open(args.attention_data, 'r') as f:
        attention_data = json.load(f)
    
    # Convert attention weights back to numpy arrays
    for data in attention_data:
        if data.get('attention_weights') is not None:
            data['attention_weights'] = np.array(data['attention_weights'])
    
    # Initialize analyzer
    analyzer = LLMAttentionAnalyzer(api_key=args.api_key)
    
    # Run analysis
    print("Analyzing attention patterns with LLM...")
    analysis = analyzer.analyze_attention_patterns(attention_data, [])
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'llm_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Generate research questions
    questions = analyzer.generate_research_questions(analysis)
    
    with open(output_dir / 'research_questions.txt', 'w') as f:
        f.write('\n'.join(questions))
    
    print(f"\nAnalysis complete. Results saved to {output_dir}")
    print(f"\nKey Insights:")
    print(analysis['llm_insights'][:500] + "...")


if __name__ == "__main__":
    main()

