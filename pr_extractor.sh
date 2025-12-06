i#!/bin/bash

# pr_diff_extractor.sh
# Script to extract diffs, commit messages, and changed files between
# a specified branch and the main branch for PR descriptions

set -e

rm -rf pr_info

# Always use the current branch
BRANCH=$(git rev-parse --abbrev-ref HEAD)
BASE_BRANCH="main"
OUTPUT_DIR="pr_info"

# Display script usage
function show_usage {
echo "Usage: $0"
echo
echo "  Extracts changes between the current branch and main."
echo
echo "  Output will be saved to the '$OUTPUT_DIR' directory."
}

# Check if we're in a git repository
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
echo "Error: Not in a git repository." >&2
exit 1
fi

# Check if the branch exists
if ! git show-ref --verify --quiet "refs/heads/$BRANCH"; then
echo "Error: Branch '$BRANCH' does not exist." >&2
exit 1
fi

# Check if the base branch exists
if ! git show-ref --verify --quiet "refs/heads/$BASE_BRANCH"; then
echo "Error: Base branch '$BASE_BRANCH' does not exist." >&2
exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Extracting information for current branch '$BRANCH' compared to '$BASE_BRANCH'..."

# Get commit messages
echo "Extracting commit messages..."
git log "$BASE_BRANCH..$BRANCH" --pretty=format:"* %h - %s%n  %b" --reverse > "$OUTPUT_DIR/commit_messages.md"

# Get list of changed files
echo "Extracting changed files..."
git diff --name-status "$BASE_BRANCH..$BRANCH" > "$OUTPUT_DIR/changed_files.txt"

# Get diff 
echo "Extracting diff..."
git diff "$BASE_BRANCH..$BRANCH" > "$OUTPUT_DIR/branch_diff.patch"

# Generate summary stats
echo "Generating summary statistics..."
git diff --stat "$BASE_BRANCH..$BRANCH" > "$OUTPUT_DIR/diff_stats.txt"

# Generate markdown PR template
echo "Creating PR description template..."
cat > "$OUTPUT_DIR/pr_description_template.md" << EOF
# Pull Request: ${BRANCH/\//-}

## Summary

[Provide a brief description of the changes introduced in this pull request]

## Commit History

$(cat "$OUTPUT_DIR/commit_messages.md")

## Testing

[Describe the testing that has been performed]

## Additional Notes

[Any additional information that reviewers should know]
EOF

echo
echo "PR information extracted successfully to the '$OUTPUT_DIR' directory:"
echo "- Commit messages: $OUTPUT_DIR/commit_messages.md"
echo "- Changed files: $OUTPUT_DIR/changed_files.txt"
echo "- Full diff: $OUTPUT_DIR/branch_diff.patch"
echo "- Diff statistics: $OUTPUT_DIR/diff_stats.txt"
echo "- PR description template: $OUTPUT_DIR/pr_description_template.md"
echo
echo "Create a PR description using the extracted files."

# Make the script executable
chmod +x "$0"

