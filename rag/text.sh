#!/bin/bash

# Define the output file name
OUTPUT_FILE="combined_contents.txt"

# Clear the output file if it already exists
> "$OUTPUT_FILE"

# Find all files, excluding the output file itself
find . -type f ! -name "$OUTPUT_FILE" ! -name "$(basename "$0")" | while read -r file; do
    # Remove the leading './' for a cleaner look
    clean_path="${file#./}"
    
    # Write the header
    echo "========================================" >> "$OUTPUT_FILE"
    echo "FILE: $clean_path" >> "$OUTPUT_FILE"
    echo "========================================" >> "$OUTPUT_FILE"
    
    # Append the file content and add a newline for spacing
    cat "$file" >> "$OUTPUT_FILE"
    echo -e "\n" >> "$OUTPUT_FILE"
done

echo "Done! All contents saved to $OUTPUT_FILE"