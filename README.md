# STAMP Control Structure Interaction Calculator

A Python script for verifying and executing mathematical calculations for missed possible control interactions in a STAMP (System-Theoretic Accident Model and Processes) hierarchical control structure model.

## Overview

This tool analyzes hierarchical control structures using STAMP methodology to identify potential missed control interactions and feedback loops. It specifically focuses on the train track zone safety case with a 3-level control hierarchy.

## Background

### STAMP Methodology

STAMP (System-Theoretic Accident Model and Processes) is a causality model based on systems theory rather than reliability theory. It views safety as a control problem rather than a failure prevention problem. The methodology analyzes:

- **Control Structures**: Hierarchical organization of system components
- **Control Actions**: Commands and information flow between components
- **Feedback Loops**: Bidirectional interactions between entities
- **Unsafe Control Actions**: Actions that can lead to hazardous states

### Unsafe Control Action Types

STAMP identifies 6 types of unsafe control actions:

1. **NOT**: Required control action not provided
2. **TOO MUCH**: Control action provided with excessive magnitude
3. **TOO LITTLE**: Control action provided with insufficient magnitude
4. **TOO EARLY**: Control action provided before it should be
5. **TOO LATE**: Control action provided after it should be
6. **WRONG ORDER**: Control actions provided in incorrect sequence

## Mathematical Formulas

### Internal Level Interactions

For a level with **n** entities:

```
Total possible interactions = n × n
Self-to-self relations = n
Unidirectional interactions = n² - n = n(n - 1)
Bidirectional feedback loops = n(n - 1) / 2
```

This formula represents the number of unique pairs of entities that can interact bidirectionally.

### Inter-level Interactions

For Level A with **a** entities and Level B with **b** entities:

```
Bidirectional feedback loops = a × b
```

Each entity in Level A can potentially interact with each entity in Level B, forming bidirectional control-feedback pairs.

### Missed Interactions

```
Missed feedback loops = Total possible - Considered important
Total missed unsafe interactions = Missed feedback loops × Unsafe action types
```

## Train Track Zone Control Structure

The case study analyzes a 3-level hierarchy:

### Level 3 (Most Concrete) - 5 Entities:
1. Roaming adversarial drone
2. Clear train tracks zone
3. Open space
4. Police force
5. Unsafe train tracks zone structure

### Level 2 (Middle) - 3 Entities:
1. Available ground-based security system
2. Police response unit
3. Reliable railway traffic control

### Level 1 (Upper) - 1 Entity:
1. TZSC (Train Zone Safety Control) system controller

## Installation

No external dependencies required. The script uses only Python standard library.

### Requirements
- Python 3.6 or higher

## Usage

### Basic Usage

Run the script directly:

```bash
python3 stamp_interaction_calculator.py
```

or make it executable:

```bash
chmod +x stamp_interaction_calculator.py
./stamp_interaction_calculator.py
```

### Using as a Module

Import and use the functions in your own code:

```python
from stamp_interaction_calculator import analyze_stamp_control_structure

results = analyze_stamp_control_structure(
    level_3_entities=5,
    level_2_entities=3,
    level_1_entities=1,
    unsafe_action_types=6,
    level_3_considered=4,
    level_2_3_considered=5,
    level_2_considered=0
)

print(f"Total missed loops: {results['total_missed_loops']}")
print(f"Total missed unsafe interactions: {results['total_missed_unsafe']}")
```

### Generalized N-Level Analysis

For custom hierarchies:

```python
from stamp_interaction_calculator import generalized_hierarchy_analysis

# Define hierarchy from top to bottom
hierarchy_levels = [1, 2, 3, 5]  # Level 1: 1 entity, Level 2: 2 entities, etc.

# Define considered loops
considered_loops = {
    'level_1': 0,
    'level_2': 0,
    'level_3': 0,
    'level_4': 4,
    'level_1_2': 0,
    'level_2_3': 0,
    'level_3_4': 5
}

results = generalized_hierarchy_analysis(hierarchy_levels, considered_loops)
```

## Output

The script produces a detailed analysis report including:

1. **Parameters**: System configuration and entities per level
2. **Calculations**: Step-by-step mathematical calculations for:
   - Level 3 internal interactions
   - Level 2-3 inter-level interactions
   - Level 2 internal interactions
3. **Totals**: Aggregate missed feedback loops and unsafe interactions
4. **Verification**: Comparison with values stated in source text
5. **Discrepancy Analysis**: Explanation of any differences

### Example Output

```
======================================================================
STAMP Control Structure Interaction Analysis
======================================================================

PARAMETERS:
----------------------------------------------------------------------
Level 3 (Most Concrete) Entities: 5
Level 2 (Middle) Entities: 3
Level 1 (Upper) Entities: 1
Total System Entities: 9
STAMP Unsafe Control Action Types: 6

CALCULATIONS:
----------------------------------------------------------------------

Level 3 (Internal Interactions):
  Total possible interactions: 5 × 5 = 25
  Self-to-self relations: 5
  Unidirectional interactions: 25 - 5 = 20
  Bidirectional feedback loops: 20 / 2 = 10
  Considered important: 4
  Missed feedback loops: 10 - 4 = 6

Level 2-3 (Inter-level Interactions):
  Possible bidirectional feedback loops: 3 × 5 = 15
  Considered necessary: 5
  Missed feedback loops: 15 - 5 = 10

Level 2 (Internal Interactions):
  Entities: 3
  Bidirectional feedback loops: 3 × (3-1) / 2 = 3
  Considered relevant: 0
  Missed feedback loops: 3 - 0 = 3

TOTALS:
----------------------------------------------------------------------
Total Missed Control Feedback Loops: 6 + 10 + 3 = 19

Compounding with STAMP Unsafe Action Types:
  Missed feedback loops: 19
  Unsafe action types: 6
  Total Missed Unsafe Interactions: 19 × 6 = 114

VERIFICATION AGAINST TEXT:
----------------------------------------------------------------------
Text states: 23 missed control feedback loops → Calculated: 19
Text states: 161 missed unsafe interactions → Calculated: 114

DISCREPANCY ANALYSIS:
----------------------------------------------------------------------
⚠ Discrepancy in missed feedback loops: 4 loop(s) difference
...
```

## Discrepancy Explanation

The script identifies a discrepancy between the text's stated values (23 missed loops, 161 missed interactions) and the mathematically calculated values (19 missed loops, 114 missed interactions).

### Root Cause

The text states "3+10+10=23", suggesting the author may have:

1. Used 10 missed loops for Level 3 instead of 6
2. Not properly subtracted the 4 considered important loops from the 10 total possible
3. Perhaps counted all possible loops without accounting for considered ones

### Correct Calculation

Using standard N² matrix analysis and STAMP methodology:

- **Level 3**: 10 possible - 4 considered = **6 missed**
- **Level 2-3**: 15 possible - 5 considered = **10 missed**
- **Level 2**: 3 possible - 0 considered = **3 missed**
- **Total**: 6 + 10 + 3 = **19 missed feedback loops**
- **Unsafe interactions**: 19 × 6 = **114 missed unsafe interactions**

## Functions

### Core Functions

#### `calculate_internal_feedback_loops(num_entities)`
Calculates bidirectional feedback loops within a single level.

**Returns:** `(total_interactions, self_relations, unidirectional, bidirectional)`

#### `calculate_interlevel_feedback_loops(level_a_entities, level_b_entities)`
Calculates bidirectional feedback loops between two levels.

**Returns:** `int` - Number of possible bidirectional feedback loops

#### `analyze_stamp_control_structure(...)`
Main analysis function for the train track zone case study.

**Returns:** `dict` - Complete analysis results

#### `generalized_hierarchy_analysis(hierarchy_levels, considered_loops, unsafe_action_types=6)`
Generalized function for analyzing N-level hierarchies.

**Returns:** `dict` - Complete analysis results for any hierarchy

## Use Cases

This tool is valuable for:

1. **Safety Engineers**: Identifying overlooked control interactions in safety-critical systems
2. **System Architects**: Comprehensive analysis of control structures
3. **Researchers**: Verification of STAMP analysis calculations
4. **Auditors**: Reviewing completeness of safety analyses
5. **Educators**: Teaching STAMP methodology and interaction analysis

## References

- Leveson, N. G. (2011). *Engineering a Safer World: Systems Thinking Applied to Safety*. MIT Press.
- STAMP Workshop materials: http://psas.scripts.mit.edu/home/
- Train Track Zone Safety Case study

## License

This script is part of the AIC PhD Appendix repository.

## Author

AIC Matrix Architect

## Contributing

For questions or improvements, please refer to the main repository documentation.
