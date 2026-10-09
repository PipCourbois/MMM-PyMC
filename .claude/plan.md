# Implementation Plan: Simple On/Off Logging Control via Config

## Summary
Implement a simple logging system controlled by a single config file switch. Replace all 154 print statements across 5 b2b_src files with proper logging that can be turned on/off via `mmm_config.yml`.

## Design: Simple Master Switch Approach

### Configuration (mmm_config.yml)
```yaml
logging:
  enabled: true  # Master switch: true = logs on, false = logs off
  level: "INFO"  # When enabled: DEBUG, INFO, WARNING, ERROR
```

**Simple logic:**
- `enabled: true` → All logs visible at specified level
- `enabled: false` → All logs suppressed (NullHandler)
- No per-module complexity
- No environment variables
- No file output

### Architecture

**New file:** `pymc_marketing/logging_config.py`
- Load YAML config
- Create logger factory with master switch
- Simple API: `setup_logging(config_path)` and `get_logger(name)`

**Pattern for all b2b_src files:**
```python
from pymc_marketing.logging_config import get_logger
logger = get_logger(__name__)

# Replace: print(f"Analyzing {channel}")
# With:    logger.info(f"Analyzing {channel}")
```

## Implementation Steps

### Step 1: Create Centralized Logging Module
**File:** `pymc_marketing/logging_config.py` (NEW)

Features:
- Parse YAML config logging section
- Master switch: enabled true/false
- Return configured or null logger
- Simple `get_logger(__name__)` API

### Step 2: Update Config File
**File:** `mmm_config.yml` (lines 122-136)

Simplify to:
```yaml
logging:
  enabled: true
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Step 3: Migrate All Print Statements

**3.1 b2b_src/contributions.py** (27 print statements)
- Add logger import
- Replace all print() with logger.info() or logger.debug()
- Progress messages → logger.info()
- Debug details → logger.debug()

**3.2 b2b_src/visual_tools.py** (77 print statements)
- Same pattern
- Plot generation → logger.debug()
- Analysis output → logger.info()

**3.3 b2b_src/utils.py** (6 print statements)
- Utility logs → logger.debug()

**3.4 b2b_src/ETL.py** (3 print statements)
- Data processing logs → logger.info()

**3.5 b2b_src/example_marginal_roas.py** (41 print statements)
- Example code logs → logger.info()

### Step 4: Initialize in Notebook
**File:** `mmm_tutorial log transformation...ipynb`

Add early cell:
```python
from pymc_marketing.logging_config import setup_logging
setup_logging('mmm_config.yml')
```

### Step 5: Test
- Test with `enabled: true` - verify logs appear
- Test with `enabled: false` - verify complete silence
- Test with `level: "DEBUG"` vs `level: "WARNING"`

## File Changes Summary

| File | Action | Changes |
|------|--------|---------|
| `pymc_marketing/logging_config.py` | CREATE | New logging module |
| `mmm_config.yml` | EDIT | Simplify logging section |
| `b2b_src/contributions.py` | EDIT | 27 print → logger |
| `b2b_src/visual_tools.py` | EDIT | 77 print → logger |
| `b2b_src/utils.py` | EDIT | 6 print → logger |
| `b2b_src/ETL.py` | EDIT | 3 print → logger |
| `b2b_src/example_marginal_roas.py` | EDIT | 41 print → logger |
| `mmm_tutorial...ipynb` | EDIT | Add logging setup cell |

**Total:** 1 new file, 7 modified files, 154 print→logger replacements

## Usage After Implementation

**Turn logs ON:**
```yaml
logging:
  enabled: true
  level: "INFO"  # or "DEBUG" for verbose
```

**Turn logs OFF:**
```yaml
logging:
  enabled: false
```

That's it! Simple on/off control.

## Implementation Order

1. Create `logging_config.py` with simple master switch logic
2. Update `mmm_config.yml` to simplified structure
3. Migrate `contributions.py` (27 changes)
4. Migrate `visual_tools.py` (77 changes)
5. Migrate `utils.py` (6 changes)
6. Migrate `ETL.py` (3 changes)
7. Migrate `example_marginal_roas.py` (41 changes)
8. Add logging initialization to notebook
9. Test with enabled: true/false

## Success Criteria

✅ Single config switch controls all logging
✅ No print statements remain in b2b_src
✅ Logs can be completely silenced with `enabled: false`
✅ No environment variables needed
✅ No file output complexity
✅ Simple, maintainable solution
