# Explain Code Structure

When the user asks you to explain code structure, architecture, or data flow. Remember, you need to wait for user input on which methods or classes to explore specifically after hits the commmand:

1. **Create an ASCII flowchart** using Unicode box-drawing characters (┌─┐│└┘├┤┬┴┼) to visualize the structure
2. **Include essential code snippets** that show the key parts being explained
3. **Format the explanation** to be clear and easy to follow in a terminal/CLI environment

## Diagram Guidelines

- Use box-drawing characters to create clear flowcharts
- Show data flow with arrows (→, ▼, ▲, ◄)
- Include decision points with branching (┌─┴─┐)
- Add dimensions/shapes when explaining data structures
- Keep boxes aligned and properly connected

## Code Snippets Guidelines

- Show only the **essential parts** of the code (method signatures, key logic, important parameters)
- Use syntax highlighting markers for the language (```python, ```typescript, etc.)
- Include file paths and line numbers in the format: `filename.py:123-145`
- Add brief inline comments to explain complex parts
- Keep snippets concise - typically 5-20 lines per snippet

## Example Structure

```
┌─────────────────────────────────────────────┐
│           High-level Component              │
│  Brief description of what happens          │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│           Detailed Component                │
│ ┌─────────────────────────────────────────┐ │
│ │ Sub-component details                   │ │
│ │   - Data structure info                 │ │
│ │   - Key transformations                 │ │
│ └─────────────────────────────────────────┘ │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
```

Then include relevant code snippets:

**[filename.py:50-65](filename.py#L50-L65)**
```python
def key_method(self, param: Type) -> ReturnType:
    """Brief explanation of what this does"""
    # Key logic here
    result = important_operation(param)
    return result
```

## When to Use This Format

- Explaining class hierarchies and inheritance
- Showing data transformation pipelines
- Documenting API call chains
- Illustrating state machines or workflows
- Clarifying complex method interactions
- Mapping data structures and their relationships

Always prioritize clarity and readability over completeness. The goal is understanding, not exhaustive documentation.
