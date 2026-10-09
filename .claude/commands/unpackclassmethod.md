---
description: Analyze a class or method showing inheritance relationships and contained methods
---

Analyze a specified class or method in the codebase. You should ask the user which class or method to analyze if not specified.

For the specified class or method, provide:

**If analyzing a CLASS:**
1. **What it inherits from**: List all parent classes and their inheritance chain
2. **What inherits from it**: List all child classes that inherit from this class
3. **Methods contained**: List all methods defined in the class with their signatures
4. **Inherited methods**: List methods inherited from parent classes
5. **Method overrides**: Identify which inherited methods are overridden

**If analyzing a METHOD:**
1. **Containing class**: Which class defines this method
2. **Method signature**: Full signature including parameters and return type
3. **Overrides**: If it overrides a parent method, show the parent implementation
4. **Calls to this method**: Where in the codebase this method is called
5. **Methods it calls**: What other methods this method invokes

Present the analysis in a clear, hierarchical format that's easy to read and understand.
