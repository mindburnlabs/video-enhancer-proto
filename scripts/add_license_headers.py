#!/usr/bin/env python3
"""
Add License Headers Script

Adds MIT license headers to Python source files in the project.
"""

import os
import sys
from pathlib import Path
from typing import List, Set
import re

LICENSE_HEADER = '''"""
MIT License

Copyright (c) 2024 Video Enhancement Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
'''

def has_license_header(content: str) -> bool:
    """Check if file already has a license header"""
    # Look for common license indicators in the first few lines
    first_lines = '\n'.join(content.split('\n')[:30]).lower()
    
    indicators = [
        'mit license',
        'copyright (c)',
        'permission is hereby granted',
        'software is provided "as is"'
    ]
    
    return any(indicator in first_lines for indicator in indicators)

def has_shebang(content: str) -> bool:
    """Check if file has a shebang line"""
    return content.startswith('#!')

def get_existing_docstring_end(content: str) -> int:
    """Find the end of existing module docstring if present"""
    lines = content.split('\n')
    
    # Skip shebang and encoding lines
    start_idx = 0
    for i, line in enumerate(lines):
        if line.startswith('#!') or 'coding:' in line or 'encoding:' in line:
            start_idx = i + 1
        else:
            break
    
    # Look for docstring start
    docstring_start = None
    for i in range(start_idx, min(len(lines), start_idx + 10)):
        line = lines[i].strip()
        if line.startswith('"""') or line.startswith("'''"):
            docstring_start = i
            break
        elif line and not line.startswith('#'):
            # Hit non-comment content, no docstring
            break
    
    if docstring_start is None:
        return start_idx
    
    # Find docstring end
    quote_type = '"""' if lines[docstring_start].strip().startswith('"""') else "'''"
    
    # Single line docstring
    if lines[docstring_start].strip().count(quote_type) == 2:
        return docstring_start + 1
    
    # Multi-line docstring
    for i in range(docstring_start + 1, len(lines)):
        if quote_type in lines[i]:
            return i + 1
    
    return docstring_start + 1

def add_license_header(file_path: Path) -> bool:
    """Add license header to a Python file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if has_license_header(content):
            print(f"  ✓ Already has license header: {file_path}")
            return False
        
        lines = content.split('\n')
        
        # Find insertion point
        insert_idx = 0
        
        # Preserve shebang
        if has_shebang(content):
            insert_idx = 1
        
        # Skip encoding declarations
        while insert_idx < len(lines):
            line = lines[insert_idx].strip()
            if ('coding:' in line or 'encoding:' in line) and line.startswith('#'):
                insert_idx += 1
            else:
                break
        
        # Find end of existing docstring
        docstring_end = get_existing_docstring_end(content)
        if docstring_end > insert_idx:
            insert_idx = docstring_end
        
        # Insert license header
        lines.insert(insert_idx, '')
        lines.insert(insert_idx + 1, LICENSE_HEADER.strip())
        lines.insert(insert_idx + 2, '')
        
        # Write back to file
        new_content = '\n'.join(lines)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"  ✓ Added license header: {file_path}")
        return True
        
    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {e}")
        return False

def should_skip_file(file_path: Path) -> bool:
    """Check if file should be skipped"""
    skip_dirs = {
        '.git', '__pycache__', '.pytest_cache', 'node_modules', 
        '.venv', 'venv', 'dist', 'build', '.tox'
    }
    
    skip_files = {
        '__init__.py',  # Often just imports, license not critical
    }
    
    # Check if in skip directory
    for part in file_path.parts:
        if part in skip_dirs:
            return True
    
    # Check filename
    if file_path.name in skip_files:
        return True
    
    # Skip very small files (probably just imports)
    try:
        if file_path.stat().st_size < 100:
            return True
    except:
        pass
    
    return False

def find_python_files(root_dir: Path) -> List[Path]:
    """Find all Python files in the project"""
    python_files = []
    
    for file_path in root_dir.rglob('*.py'):
        if file_path.is_file() and not should_skip_file(file_path):
            python_files.append(file_path)
    
    return sorted(python_files)

def main():
    """Main function to add license headers"""
    project_root = Path(__file__).parent.parent
    print(f"🔍 Adding license headers to Python files in: {project_root}")
    
    # Find all Python files
    python_files = find_python_files(project_root)
    print(f"Found {len(python_files)} Python files to process")
    
    if not python_files:
        print("No Python files found to process")
        return
    
    # Process files
    added_count = 0
    skipped_count = 0
    
    for file_path in python_files:
        try:
            relative_path = file_path.relative_to(project_root)
            if add_license_header(file_path):
                added_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"  ✗ Error with {file_path}: {e}")
    
    print(f"\n📊 Summary:")
    print(f"  License headers added: {added_count}")
    print(f"  Files already had headers: {skipped_count}")
    print(f"  Total processed: {added_count + skipped_count}")
    
    if added_count > 0:
        print(f"\n✅ Successfully added license headers to {added_count} files")
    else:
        print(f"\n✅ All files already have appropriate license headers")

if __name__ == "__main__":
    main()