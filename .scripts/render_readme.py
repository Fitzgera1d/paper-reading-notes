import os
import re
from pathlib import Path

# Try to import pathspec, if not available, provide instructions.
try:
    import pathspec
except ImportError:
    print("Error: 'pathspec' library not found.")
    print("Please install it by running: pip install pathspec")
    exit(1)

def get_gitignore_spec(project_root):
    """Reads .gitignore and returns a pathspec object."""
    gitignore_path = project_root / '.gitignore'
    patterns = []
    if gitignore_path.exists():
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            patterns = f.read().splitlines()
    # Add common patterns to always ignore
    patterns.extend(['.git', '.vscode', '.idea'])
    return pathspec.PathSpec.from_lines('gitwildmatch', patterns)

def generate_tree(start_path_str, project_root):
    """
    Generates a markdown bullet list tree for .md files, respecting .gitignore.
    """
    lines = []
    start_path = Path(start_path_str)
    spec = get_gitignore_spec(project_root)
    
    # Directories to always skip
    excluded_dirs_base = {'assets', 'node_modules', '__pycache__'}

    # Walk through the directory
    for root, dirs, files in os.walk(start_path, topdown=True):
        current_path = Path(root)
        
        # Check if current path should be ignored by gitignore
        # Use as_posix() for cross-platform compatibility in spec matching
        relative_path_posix = current_path.relative_to(project_root).as_posix()
        if spec.match_file(relative_path_posix) and relative_path_posix != '.':
            dirs[:] = [] # Don't traverse further
            continue

        # Filter directories in-place
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in excluded_dirs_base and not spec.match_file( (current_path / d).relative_to(project_root).as_posix() )]
        
        # Filter files
        filtered_files = [f for f in files if f.endswith('.md') and not spec.match_file( (current_path / f).relative_to(project_root).as_posix() )]

        is_tree_root = (current_path == start_path)
        
        if is_tree_root:
            # In the root of the tree, we only want to recurse into directories, not list files.
            pass
        else:
            # For all sub-directories, list the directory name and its files.
            level = len(current_path.relative_to(start_path).parts)
            dir_indent = '  ' * (level - 1)
            lines.append(f"{dir_indent}- **{current_path.name}/**")
            
            file_indent = '  ' * level
            for f in sorted(filtered_files):
                file_path = current_path / f
                link_path = file_path.relative_to(project_root).as_posix()
                link_text = Path(f).stem
                lines.append(f"{file_indent}- [{link_text}]({link_path})")

    return "\n".join(lines)


def render_template(template_content, project_root):
    """
    Renders a template by replacing placeholders.
    """
    
    # Function to handle %content(path) replacement with indentation
    def replace_content(match):
        indent = match.group(1)
        rel_path = match.group(2)
        file_path = project_root / rel_path
        
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Add indentation to each line of the content
                indented_content = ''.join([f"{indent}{line}" for line in content.splitlines(True)])
                return indented_content
        else:
            return f"{indent}[Content file not found: {rel_path}]"

    # Use a regex that captures the leading whitespace
    content_with_indent = re.sub(r'^(\s*)%content\((.*?)\)', replace_content, template_content, flags=re.MULTILINE)

    # Function to handle %tree(path) replacement
    def replace_tree(match):
        rel_path = match.group(1)
        # Use strip to handle potential whitespace like %tree(./)
        tree_start_path = project_root / rel_path.strip()
        if tree_start_path.is_dir():
            return generate_tree(tree_start_path, project_root)
        else:
            return f"[Tree directory not found: {rel_path}]"

    final_content = re.sub(r'%tree\((.*?)\)', replace_tree, content_with_indent)

    return final_content

def main():
    """Main function to find and render templates."""
    try:
        script_path = Path(__file__).resolve()
        project_root = script_path.parent.parent
        os.chdir(project_root)
        
        templates_dir = project_root / '.templates'
        if not templates_dir.is_dir():
            print(f"Error: Templates directory not found at '{templates_dir}'")
            return

        for template_path in templates_dir.glob('*.md'):
            print(f"Rendering {template_path.name}...")
            with open(template_path, 'r', encoding='utf-8') as f:
                template_content = f.read()

            rendered_content = render_template(template_content, project_root)
            
            output_path = project_root / template_path.name
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(rendered_content)
            print(f"-> Successfully generated {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main() 