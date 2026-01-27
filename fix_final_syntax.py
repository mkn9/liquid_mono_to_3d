#!/usr/bin/env python3
"""
Final aggressive fix for remaining syntax errors in semantic_nerf_demo.ipynb
"""

import json
import re

def fix_final_syntax_errors():
    """Apply final fixes for the 8 remaining syntax errors."""
    
    with open('semantic_nerf_demo.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    fixes_applied = 0
    
    for cell_idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            new_source = []
            
            for line in cell['source']:
                fixed_line = line
                
                # Fix 1: Remove emoji characters that cause syntax errors
                fixed_line = re.sub(r'[🧠🔧📊📸✅❌🚀⚠️]', '', fixed_line)
                
                # Fix 2: Handle unterminated string literals more aggressively
                if 'print(f"' in fixed_line and not fixed_line.strip().endswith('")'):
                    # Find the opening quote and ensure it's properly closed
                    if fixed_line.count('"') % 2 == 1:  # Odd number of quotes
                        fixed_line = fixed_line.rstrip() + '")\n'
                
                # Fix 3: Handle malformed f-strings with complex patterns
                if re.search(r'print\(f"[^"]*\\n[^"]*$', fixed_line):
                    # F-string with newline but no closing quote
                    fixed_line = re.sub(r'(print\(f"[^"]*\\n[^"]*)', r'\1")', fixed_line)
                
                # Fix 4: Fix indentation issues
                if fixed_line.strip() and not fixed_line.startswith(' ') and cell_idx > 0:
                    # Check if this should be indented based on context
                    if any(keyword in fixed_line for keyword in ['for ', 'if ', 'with ', 'def ', 'class ']):
                        # Don't change indentation for control structures
                        pass
                    elif 'print(' in fixed_line and len([l for l in cell['source'] if l.strip()]) > 1:
                        # This might need indentation if it's part of a block
                        prev_lines = [l for l in cell['source'] if l.strip()]
                        if len(prev_lines) > 1 and any(l.strip().endswith(':') for l in prev_lines[:-1]):
                            fixed_line = '    ' + fixed_line
                
                # Fix 5: Remove completely malformed lines that can't be salvaged
                if re.search(r'print\(f"[^"]*$', fixed_line.strip()) and len(fixed_line.strip()) < 20:
                    fixed_line = '# Removed malformed print statement\n'
                
                new_source.append(fixed_line)
                
                if fixed_line != line:
                    fixes_applied += 1
            
            cell['source'] = new_source
    
    # Save the fixed notebook
    with open('semantic_nerf_demo.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    
    print(f"Applied {fixes_applied} final fixes")
    
    # Test the result
    import ast
    total_errors = 0
    for cell_idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            cell_code = ''.join(cell['source'])
            if cell_code.strip():
                try:
                    ast.parse(cell_code)
                except SyntaxError as e:
                    print(f"Cell {cell_idx}: {e}")
                    total_errors += 1
    
    if total_errors == 0:
        print("✅ All syntax errors fixed!")
    else:
        print(f"❌ {total_errors} errors remain")
    
    return total_errors

if __name__ == "__main__":
    fix_final_syntax_errors() 