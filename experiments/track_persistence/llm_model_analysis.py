#!/usr/bin/env python3
"""
Worker 3: LLM-Assisted Model Analysis

Uses LLM reasoning to:
1. Analyze training results and identify patterns
2. Suggest improvements and architectural changes
3. Generate hypotheses about model behavior
4. Provide insights about failure modes
5. Recommend next steps
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import anthropic
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class LLMModelAnalyzer:
    """Uses Claude to analyze model training results and provide insights."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM analyzer.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-sonnet-4-20250514"
    
    def analyze_training_results(self, results_path: Path) -> Dict[str, Any]:
        """
        Analyze training results using LLM reasoning.
        
        Args:
            results_path: Path to training_results.json
            
        Returns:
            Dictionary containing LLM analysis and recommendations
        """
        # Load training results
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # Prepare analysis prompt
        prompt = self._create_analysis_prompt(results)
        
        # Get LLM analysis
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        analysis_text = response.content[0].text
        
        # Structure the analysis
        structured_analysis = self._structure_analysis(analysis_text, results)
        
        return structured_analysis
    
    def _create_analysis_prompt(self, results: Dict) -> str:
        """Create prompt for LLM analysis."""
        
        # Extract key metrics
        train_acc = results.get('train_history', {}).get('accuracy', [])[-1] if results.get('train_history', {}).get('accuracy') else 'N/A'
        val_acc = results.get('val_history', {}).get('accuracy', [])[-1] if results.get('val_history', {}).get('accuracy') else 'N/A'
        test_acc = results.get('test_results', {}).get('accuracy', 'N/A')
        
        train_loss = results.get('train_history', {}).get('loss', [])[-1] if results.get('train_history', {}).get('loss') else 'N/A'
        val_loss = results.get('val_history', {}).get('loss', [])[-1] if results.get('val_history', {}).get('loss') else 'N/A'
        
        test_f1 = results.get('test_results', {}).get('f1', 'N/A')
        test_precision = results.get('test_results', {}).get('precision', 'N/A')
        test_recall = results.get('test_results', {}).get('recall', 'N/A')
        
        num_epochs = results.get('num_epochs', 'N/A')
        best_epoch = results.get('best_epoch', 'N/A')
        
        prompt = f"""You are an expert machine learning researcher analyzing a deep learning model for track persistence classification.

TASK: The model classifies video tracks as "persistent" (long-duration, real objects) vs "non-persistent" (brief detections, noise, clutter).

MODEL ARCHITECTURE:
- Feature Extractor: SimpleStatisticalExtractor (track length, velocity, position statistics)
- Sequence Model: Transformer (multi-head attention, 4 layers, 8 heads, 256 dim)
- Task Head: PersistenceClassificationHead (binary classification)
- Total Parameters: 236,993

DATASET:
- 2,500 synthetic videos (1,750 train / 375 val / 375 test)
- Categories: persistent tracks, brief detections, noise, mixed
- Video length: 25 frames, Resolution: 128x128

TRAINING RESULTS:
- Epochs: {num_epochs}
- Best Epoch: {best_epoch}

- Final Train Accuracy: {train_acc}
- Final Train Loss: {train_loss}

- Final Val Accuracy: {val_acc}
- Final Val Loss: {val_loss}

- Test Accuracy: {test_acc}
- Test F1 Score: {test_f1}
- Test Precision: {test_precision}
- Test Recall: {test_recall}

TARGET PERFORMANCE: 85-95% accuracy

FULL TRAINING HISTORY:
{json.dumps(results, indent=2)}

ANALYSIS REQUESTED:

1. PERFORMANCE ASSESSMENT
   - How well did the model perform vs the target?
   - Is there evidence of overfitting or underfitting?
   - What do the metrics tell us about model quality?

2. TRAINING DYNAMICS
   - Analyze the learning curves (loss, accuracy over epochs)
   - When did convergence occur?
   - Were there any unstable periods?
   - Is early stopping needed?

3. ARCHITECTURAL INSIGHTS
   - Is the simple statistical feature extractor sufficient?
   - Would visual features (MagVit) provide meaningful improvement?
   - Are there architecture bottlenecks?

4. FAILURE MODE HYPOTHESES
   - Based on the metrics, what types of errors is the model likely making?
   - Which track categories are probably hardest to classify?
   - What are the edge cases?

5. NEXT STEPS RECOMMENDATIONS
   - Should we proceed to Phase 2 (MagVit visual features)?
   - Are there hyperparameter improvements to try?
   - Should we expand the dataset?
   - Is the model ready for production deployment?

6. SURPRISING FINDINGS
   - What is unexpected about these results?
   - Are there insights that contradict initial assumptions?

Please provide detailed, technical analysis with specific reasoning. Be honest about limitations and uncertainties.
"""
        return prompt
    
    def _structure_analysis(self, analysis_text: str, results: Dict) -> Dict[str, Any]:
        """Structure the LLM analysis into organized sections."""
        
        return {
            'llm_analysis': analysis_text,
            'training_summary': {
                'test_accuracy': results.get('test_results', {}).get('accuracy'),
                'test_f1': results.get('test_results', {}).get('f1'),
                'best_epoch': results.get('best_epoch'),
                'num_epochs': results.get('num_epochs')
            },
            'timestamp': results.get('timestamp', 'unknown')
        }
    
    def analyze_attention_patterns(self, attention_data: Dict) -> str:
        """
        Analyze attention patterns from the Transformer.
        
        Args:
            attention_data: Dictionary with attention weights and metadata
            
        Returns:
            LLM analysis of attention patterns
        """
        prompt = f"""Analyze the attention patterns from a Transformer-based track persistence model.

ATTENTION DATA:
{json.dumps(attention_data, indent=2)}

CONTEXT:
- Model classifies video tracks (25 frames) as persistent vs non-persistent
- Transformer uses multi-head attention (8 heads, 4 layers)
- Attention weights show which frames the model focuses on

ANALYSIS REQUESTED:
1. Which frames receive the most attention?
2. Does the model focus on early, middle, or late frames?
3. What does this tell us about the classification strategy?
4. Are attention patterns different for different track types?
5. Are there unexpected attention patterns?

Provide specific insights about the model's learned strategy.
"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def suggest_improvements(self, results: Dict) -> str:
        """
        Generate specific improvement suggestions.
        
        Args:
            results: Training results dictionary
            
        Returns:
            LLM recommendations for improvements
        """
        prompt = f"""Given the following training results, suggest specific, actionable improvements:

RESULTS:
{json.dumps(results, indent=2)}

Provide:
1. Hyperparameter tuning suggestions (learning rate, batch size, etc.)
2. Architecture modifications
3. Data augmentation strategies
4. Training procedure improvements
5. Evaluation enhancements

Be specific and technical. Prioritize suggestions by expected impact.
"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text
    
    def generate_research_questions(self, results: Dict) -> List[str]:
        """
        Generate research questions based on results.
        
        Args:
            results: Training results dictionary
            
        Returns:
            List of research questions
        """
        prompt = f"""Based on these training results, generate 5-10 interesting research questions:

RESULTS:
{json.dumps(results, indent=2)}

Generate questions that:
- Explore model behavior and failure modes
- Investigate architectural choices
- Propose novel extensions
- Challenge assumptions

Format: One question per line, numbered.
"""
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse questions
        text = response.content[0].text
        questions = [line.strip() for line in text.split('\n') if line.strip() and any(c.isdigit() for c in line[:3])]
        
        return questions
    
    def save_analysis(self, analysis: Dict, output_path: Path):
        """Save analysis to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Also save as markdown for readability
        md_path = output_path.with_suffix('.md')
        with open(md_path, 'w') as f:
            f.write("# LLM-Assisted Model Analysis\n\n")
            f.write(f"**Generated:** {analysis.get('timestamp', 'unknown')}\n\n")
            f.write("---\n\n")
            f.write("## Training Summary\n\n")
            f.write(f"- **Test Accuracy:** {analysis['training_summary']['test_accuracy']:.4f}\n")
            f.write(f"- **Test F1 Score:** {analysis['training_summary']['test_f1']:.4f}\n")
            f.write(f"- **Best Epoch:** {analysis['training_summary']['best_epoch']}\n")
            f.write(f"- **Total Epochs:** {analysis['training_summary']['num_epochs']}\n\n")
            f.write("---\n\n")
            f.write("## LLM Analysis\n\n")
            f.write(analysis['llm_analysis'])
            f.write("\n\n---\n\n")
            
            if 'improvement_suggestions' in analysis:
                f.write("## Improvement Suggestions\n\n")
                f.write(analysis['improvement_suggestions'])
                f.write("\n\n---\n\n")
            
            if 'research_questions' in analysis:
                f.write("## Research Questions\n\n")
                for q in analysis['research_questions']:
                    f.write(f"- {q}\n")
                f.write("\n")


def main():
    """Run LLM analysis on training results."""
    import argparse
    import logging
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description='LLM-assisted model analysis')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to training_results.json')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for analysis')
    parser.add_argument('--full', action='store_true',
                       help='Run full analysis including suggestions and research questions')
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    if not results_path.exists():
        logger.error(f"Results file not found: {results_path}")
        return
    
    # Default output path
    if args.output is None:
        output_dir = results_path.parent / 'llm_analysis'
        output_path = output_dir / 'analysis.json'
    else:
        output_path = Path(args.output)
    
    logger.info("="*60)
    logger.info("Worker 3: LLM-Assisted Model Analysis")
    logger.info("="*60)
    logger.info(f"Results: {results_path}")
    logger.info(f"Output: {output_path}")
    
    # Initialize analyzer
    analyzer = LLMModelAnalyzer()
    
    # Run main analysis
    logger.info("\n🤖 Running LLM analysis...")
    analysis = analyzer.analyze_training_results(results_path)
    
    # Run additional analyses if requested
    if args.full:
        logger.info("\n💡 Generating improvement suggestions...")
        with open(results_path, 'r') as f:
            results = json.load(f)
        analysis['improvement_suggestions'] = analyzer.suggest_improvements(results)
        
        logger.info("\n❓ Generating research questions...")
        analysis['research_questions'] = analyzer.generate_research_questions(results)
    
    # Save results
    logger.info(f"\n💾 Saving analysis to {output_path}")
    analyzer.save_analysis(analysis, output_path)
    
    logger.info("\n" + "="*60)
    logger.info("✅ LLM ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Analysis saved to: {output_path}")
    logger.info(f"Markdown report: {output_path.with_suffix('.md')}")


if __name__ == '__main__':
    main()

