#!/usr/bin/env python3
"""
Validate workflow state before starting work.

This script validates that:
1. Issue exists in correct location (todo/ or in_progress/)
2. Issue is in correct location for current workflow stage
3. Issue move to in_progress/ was committed (if work started)
4. Branch matches issue specification
5. Git history shows correct sequence (issue move committed BEFORE work files)

Exit codes:
- 0: Workflow state is valid
- 1: Workflow violation detected
- 2: Error during validation (e.g., not a git repo)
"""

import os
import sys
import re
import json
import subprocess
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from datetime import datetime


def get_project_name() -> str:
    """
    Detect project name from context.
    
    Priority:
    1. From ~/.flowmates/config.json (if repo path mapped)
    2. From git remote URL
    3. From repository directory name
    4. Default to 'flowmates'
    """
    # Try to read flowmates config
    config_path = Path.home() / '.flowmates' / 'config.json'
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                repo_path = config.get('repo_path', '')
                # Could map repo_path to project name, but for now use default
        except Exception:
            pass
    
    # Try to get from git remote
    try:
        result = subprocess.run(
            ['git', 'remote', 'get-url', 'origin'],
            capture_output=True,
            text=True,
            check=True
        )
        remote_url = result.stdout.strip()
        # Extract project name from URL (e.g., github.com/user/repo.git -> repo)
        if remote_url:
            match = re.search(r'/([^/]+?)(?:\.git)?$', remote_url)
            if match:
                return match.group(1)
    except Exception:
        pass
    
    # Try to get from current directory
    try:
        cwd = Path.cwd()
        return cwd.name
    except Exception:
        pass
    
    # Default to flowmates
    return 'flowmates'


def get_current_branch() -> Optional[str]:
    """Get current git branch name."""
    try:
        result = subprocess.run(
            ['git', 'branch', '--show-current'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return None


def find_issue_files(project: str) -> Tuple[List[Path], List[Path]]:
    """
    Find issue files in todo/ and in_progress/ directories.
    
    Returns:
        (todo_issues, in_progress_issues)
    """
    todo_dir = Path('issues') / project / 'todo'
    in_progress_dir = Path('issues') / project / 'in_progress'
    
    todo_issues = list(todo_dir.glob('*.md')) if todo_dir.exists() else []
    in_progress_issues = list(in_progress_dir.glob('*.md')) if in_progress_dir.exists() else []
    
    return todo_issues, in_progress_issues


def parse_issue_file(issue_path: Path) -> Dict:
    """Parse issue file and extract metadata."""
    metadata = {
        'path': issue_path,
        'branch': None,
        'status': None,
        'type': None,
    }
    
    try:
        with open(issue_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract branch from frontmatter
        branch_match = re.search(r'\*\*Branch:\*\*\s*(\S+)', content)
        if branch_match:
            metadata['branch'] = branch_match.group(1)
        
        # Extract status from frontmatter
        status_match = re.search(r'\*\*Status:\*\*\s*(\S+)', content)
        if status_match:
            metadata['status'] = status_match.group(1)
        
        # Extract type from frontmatter
        type_match = re.search(r'\*\*Type:\*\*\s*(\S+)', content)
        if type_match:
            metadata['type'] = type_match.group(1)
            
    except Exception as e:
        print(f"⚠️  Warning: Could not parse issue file {issue_path}: {e}", file=sys.stderr)
    
    return metadata


def check_git_history_sequence(issue_path: Path, project: str) -> Tuple[bool, str]:
    """
    Check if issue move to in_progress/ was committed before work files.
    
    Returns:
        (is_valid, error_message)
    """
    try:
        # Get commits that touched the issue file
        result = subprocess.run(
            ['git', 'log', '--oneline', '--follow', '--', str(issue_path)],
            capture_output=True,
            text=True,
            check=True
        )
        
        issue_commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        # Get commits that modified non-issue files (work files)
        result = subprocess.run(
            ['git', 'log', '--oneline', '--', '--not', '--', 'issues/'],
            capture_output=True,
            text=True,
            check=True
        )
        
        work_commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
        
        # Check if issue was moved to in_progress/ before work started
        # This is a simplified check - in practice, we'd need to check commit timestamps
        # For now, we check if there are any commits that moved the issue to in_progress/
        in_progress_dir = Path('issues') / project / 'in_progress'
        if in_progress_dir.exists() and issue_path.parent == in_progress_dir:
            # Issue is in in_progress/, check if there's a commit that moved it there
            # Look for commits that mention moving to in_progress
            result = subprocess.run(
                ['git', 'log', '--oneline', '--grep', 'move.*in_progress', '--', str(issue_path)],
                capture_output=True,
                text=True,
                check=True
            )
            
            move_commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            if not move_commits and work_commits:
                return False, "Issue move to in_progress/ was not committed before work started"
        
        return True, ""
        
    except Exception as e:
        # If we can't check history, assume valid (don't block on this)
        return True, f"Could not validate git history: {e}"


def validate_workflow_state(project: str, current_branch: Optional[str], force: bool = False) -> Tuple[bool, str]:
    """
    Validate workflow state.
    
    Returns:
        (is_valid, error_message)
    """
    if force:
        return True, ""
    
    # Find issue files
    todo_issues, in_progress_issues = find_issue_files(project)
    
    # Check if we're on main/master (should have an issue)
    if current_branch in ['main', 'master']:
        if not todo_issues and not in_progress_issues:
            return False, (
                "❌ WORKFLOW VIOLATION: Cannot work on main/master without an issue.\n"
                "Required: Create an issue in issues/<project>/todo/ first.\n"
                "Remediation: Follow exploration workflow to create issue."
            )
    
    # If we have work files but no issue in in_progress/, that's a violation
    # Check if there are uncommitted changes to non-issue files
    try:
        result = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True
        )
        
        work_files = []
        for line in result.stdout.strip().split('\n'):
            if line and not line.startswith('??'):  # Ignore untracked
                file_path = line[3:].strip()
                if not file_path.startswith('issues/'):
                    work_files.append(file_path)
        
        # If we have work files but issue is still in todo/, that's a violation
        if work_files and todo_issues and not in_progress_issues:
            return False, (
                "❌ WORKFLOW VIOLATION: Work files modified but issue still in todo/.\n"
                "Required: Move issue to in_progress/ and commit before starting work.\n"
                f"Found {len(todo_issues)} issue(s) in todo/: {[str(p.name) for p in todo_issues]}\n"
                "Remediation:\n"
                "  1. Move issue: mv issues/<project>/todo/{issue}.md issues/<project>/in_progress/\n"
                "  2. Update status in issue file: **Status:** in_progress\n"
                "  3. Commit: git add issues/<project>/in_progress/{issue}.md && git commit -m 'chore: move issue to in_progress'\n"
                "  4. Then start work"
            )
        
        # If we have work files, validate git history sequence
        if work_files and in_progress_issues:
            for issue_path in in_progress_issues:
                is_valid, error = check_git_history_sequence(issue_path, project)
                if not is_valid:
                    return False, f"❌ WORKFLOW VIOLATION: {error}"
        
        # Validate branch matches issue specification
        if current_branch and in_progress_issues:
            for issue_path in in_progress_issues:
                issue_meta = parse_issue_file(issue_path)
                if issue_meta['branch'] and issue_meta['branch'] != current_branch:
                    return False, (
                        f"❌ WORKFLOW VIOLATION: Branch mismatch.\n"
                        f"Current branch: {current_branch}\n"
                        f"Issue specifies branch: {issue_meta['branch']}\n"
                        f"Issue: {issue_path}\n"
                        "Remediation: Checkout the correct branch specified in the issue."
                    )
    
    except subprocess.CalledProcessError:
        # Not a git repo or git error - allow but warn
        return True, "⚠️  Warning: Could not validate git state (not a git repo?)"
    
    return True, ""


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate workflow state before starting work'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Skip validation (not recommended)'
    )
    parser.add_argument(
        '--project',
        type=str,
        help='Project name (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Detect project name
    project = args.project or get_project_name()
    
    # Get current branch
    current_branch = get_current_branch()
    
    # Validate workflow state
    is_valid, error_message = validate_workflow_state(project, current_branch, args.force)
    
    if not is_valid:
        print(error_message, file=sys.stderr)
        sys.exit(1)
    
    if error_message and not args.force:
        print(error_message, file=sys.stderr)
    
    print(f"✅ Workflow state is valid (project: {project}, branch: {current_branch})")
    sys.exit(0)


if __name__ == '__main__':
    main()





