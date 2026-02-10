#!/usr/bin/env python3
"""
STAMP Control Structure Interaction Calculator

This script verifies and executes mathematical calculations for missed possible
control interactions in a STAMP (System-Theoretic Accident Model and Processes)
hierarchical control structure model for the train track zone safety case.

Author: AIC Matrix Architect
Date: 2026
"""


def calculate_internal_feedback_loops(num_entities):
    """
    Calculate bidirectional feedback loops within a single level.
    
    Formula for internal interactions:
    - Total interactions: n × n
    - Self-to-self relations: n
    - Unidirectional interactions: n² - n
    - Bidirectional feedback loops: (n² - n) / 2 = n(n-1)/2
    
    Args:
        num_entities (int): Number of entities at the level
        
    Returns:
        tuple: (total_interactions, self_relations, unidirectional, bidirectional)
    """
    total_interactions = num_entities * num_entities
    self_relations = num_entities
    unidirectional = total_interactions - self_relations
    bidirectional = unidirectional // 2
    
    return total_interactions, self_relations, unidirectional, bidirectional


def calculate_interlevel_feedback_loops(level_a_entities, level_b_entities):
    """
    Calculate bidirectional feedback loops between two levels.
    
    Formula: For inter-level interactions, every entity in level A can interact
    with every entity in level B, forming bidirectional feedback loops.
    Number of bidirectional feedback loops = level_a_entities × level_b_entities
    
    Args:
        level_a_entities (int): Number of entities in level A
        level_b_entities (int): Number of entities in level B
        
    Returns:
        int: Number of possible bidirectional feedback loops
    """
    return level_a_entities * level_b_entities


def calculate_missed_interactions(total_possible, considered_important):
    """
    Calculate missed interactions.
    
    Args:
        total_possible (int): Total possible interactions
        considered_important (int): Interactions considered important
        
    Returns:
        int: Number of missed interactions
    """
    return total_possible - considered_important


def analyze_stamp_control_structure(
    level_3_entities,
    level_2_entities,
    level_1_entities,
    unsafe_action_types,
    level_3_considered,
    level_2_3_considered,
    level_2_considered,
    level_1_2_considered=0,
    level_1_considered=0
):
    """
    Analyze STAMP control structure and calculate missed interactions.
    
    Args:
        level_3_entities (int): Number of entities at Level 3 (most concrete)
        level_2_entities (int): Number of entities at Level 2 (middle)
        level_1_entities (int): Number of entities at Level 1 (upper)
        unsafe_action_types (int): Number of STAMP unsafe control action types
        level_3_considered (int): Feedback loops considered important at Level 3
        level_2_3_considered (int): Feedback loops considered important between Level 2-3
        level_2_considered (int): Feedback loops considered important at Level 2
        level_1_2_considered (int): Feedback loops considered important between Level 1-2
        level_1_considered (int): Feedback loops considered important at Level 1
        
    Returns:
        dict: Dictionary containing all calculation results
    """
    results = {}
    
    # Level 3 Internal Interactions
    l3_total, l3_self, l3_uni, l3_bi = calculate_internal_feedback_loops(level_3_entities)
    l3_missed = calculate_missed_interactions(l3_bi, level_3_considered)
    
    results['level_3'] = {
        'entities': level_3_entities,
        'total_interactions': l3_total,
        'self_relations': l3_self,
        'unidirectional': l3_uni,
        'bidirectional': l3_bi,
        'considered': level_3_considered,
        'missed': l3_missed
    }
    
    # Level 2-3 Inter-level Interactions
    l2_3_bi = calculate_interlevel_feedback_loops(level_2_entities, level_3_entities)
    l2_3_missed = calculate_missed_interactions(l2_3_bi, level_2_3_considered)
    
    results['level_2_3'] = {
        'level_2_entities': level_2_entities,
        'level_3_entities': level_3_entities,
        'bidirectional': l2_3_bi,
        'considered': level_2_3_considered,
        'missed': l2_3_missed
    }
    
    # Level 2 Internal Interactions
    l2_total, l2_self, l2_uni, l2_bi = calculate_internal_feedback_loops(level_2_entities)
    l2_missed = calculate_missed_interactions(l2_bi, level_2_considered)
    
    results['level_2'] = {
        'entities': level_2_entities,
        'total_interactions': l2_total,
        'self_relations': l2_self,
        'unidirectional': l2_uni,
        'bidirectional': l2_bi,
        'considered': level_2_considered,
        'missed': l2_missed
    }
    
    # Level 1-2 Inter-level Interactions (optional, for completeness)
    if level_1_entities > 0:
        l1_2_bi = calculate_interlevel_feedback_loops(level_1_entities, level_2_entities)
        l1_2_missed = calculate_missed_interactions(l1_2_bi, level_1_2_considered)
        
        results['level_1_2'] = {
            'level_1_entities': level_1_entities,
            'level_2_entities': level_2_entities,
            'bidirectional': l1_2_bi,
            'considered': level_1_2_considered,
            'missed': l1_2_missed
        }
        
        # Level 1 Internal Interactions
        l1_total, l1_self, l1_uni, l1_bi = calculate_internal_feedback_loops(level_1_entities)
        l1_missed = calculate_missed_interactions(l1_bi, level_1_considered)
        
        results['level_1'] = {
            'entities': level_1_entities,
            'total_interactions': l1_total,
            'self_relations': l1_self,
            'unidirectional': l1_uni,
            'bidirectional': l1_bi,
            'considered': level_1_considered,
            'missed': l1_missed
        }
    
    # Total Missed Feedback Loops
    total_missed_loops = l3_missed + l2_3_missed + l2_missed
    results['total_missed_loops'] = total_missed_loops
    
    # Total Missed Unsafe Interactions
    total_missed_unsafe = total_missed_loops * unsafe_action_types
    results['total_missed_unsafe'] = total_missed_unsafe
    results['unsafe_action_types'] = unsafe_action_types
    
    return results


def print_results(results, level_3_entities, level_2_entities, level_1_entities, 
                  unsafe_action_types, text_loops=23, text_unsafe=161):
    """
    Print formatted results of the STAMP analysis.
    
    Args:
        results (dict): Results from analyze_stamp_control_structure
        level_3_entities (int): Number of Level 3 entities
        level_2_entities (int): Number of Level 2 entities
        level_1_entities (int): Number of Level 1 entities
        unsafe_action_types (int): Number of unsafe action types
        text_loops (int): Value stated in text for missed loops
        text_unsafe (int): Value stated in text for missed unsafe interactions
    """
    print("=" * 70)
    print("STAMP Control Structure Interaction Analysis")
    print("=" * 70)
    print()
    
    # Parameters
    print("PARAMETERS:")
    print("-" * 70)
    print(f"Level 3 (Most Concrete) Entities: {level_3_entities}")
    print(f"Level 2 (Middle) Entities: {level_2_entities}")
    print(f"Level 1 (Upper) Entities: {level_1_entities}")
    print(f"Total System Entities: {level_3_entities + level_2_entities + level_1_entities}")
    print(f"STAMP Unsafe Control Action Types: {unsafe_action_types}")
    print()
    
    # Calculations
    print("CALCULATIONS:")
    print("-" * 70)
    print()
    
    # Level 3
    l3 = results['level_3']
    print("Level 3 (Internal Interactions):")
    print(f"  Total possible interactions: {l3['entities']} × {l3['entities']} = {l3['total_interactions']}")
    print(f"  Self-to-self relations: {l3['self_relations']}")
    print(f"  Unidirectional interactions: {l3['total_interactions']} - {l3['self_relations']} = {l3['unidirectional']}")
    print(f"  Bidirectional feedback loops: {l3['unidirectional']} / 2 = {l3['bidirectional']}")
    print(f"  Considered important: {l3['considered']}")
    print(f"  Missed feedback loops: {l3['bidirectional']} - {l3['considered']} = {l3['missed']}")
    print()
    
    # Level 2-3
    l2_3 = results['level_2_3']
    print("Level 2-3 (Inter-level Interactions):")
    print(f"  Possible bidirectional feedback loops: {l2_3['level_2_entities']} × {l2_3['level_3_entities']} = {l2_3['bidirectional']}")
    print(f"  Considered necessary: {l2_3['considered']}")
    print(f"  Missed feedback loops: {l2_3['bidirectional']} - {l2_3['considered']} = {l2_3['missed']}")
    print()
    
    # Level 2
    l2 = results['level_2']
    print("Level 2 (Internal Interactions):")
    print(f"  Entities: {l2['entities']}")
    print(f"  Bidirectional feedback loops: {l2['entities']} × ({l2['entities']}-1) / 2 = {l2['bidirectional']}")
    print(f"  Considered relevant: {l2['considered']}")
    print(f"  Missed feedback loops: {l2['bidirectional']} - {l2['considered']} = {l2['missed']}")
    print()
    
    # Totals
    print("TOTALS:")
    print("-" * 70)
    print(f"Total Missed Control Feedback Loops: {l3['missed']} + {l2_3['missed']} + {l2['missed']} = {results['total_missed_loops']}")
    print()
    print("Compounding with STAMP Unsafe Action Types:")
    print(f"  Missed feedback loops: {results['total_missed_loops']}")
    print(f"  Unsafe action types: {results['unsafe_action_types']}")
    print(f"  Total Missed Unsafe Interactions: {results['total_missed_loops']} × {results['unsafe_action_types']} = {results['total_missed_unsafe']}")
    print()
    
    # Verification
    print("VERIFICATION AGAINST TEXT:")
    print("-" * 70)
    print(f"Text states: {text_loops} missed control feedback loops → Calculated: {results['total_missed_loops']}")
    print(f"Text states: {text_unsafe} missed unsafe interactions → Calculated: {results['total_missed_unsafe']}")
    print()
    
    # Discrepancy Analysis
    print("DISCREPANCY ANALYSIS:")
    print("-" * 70)
    
    if results['total_missed_loops'] != text_loops:
        diff_loops = text_loops - results['total_missed_loops']
        print(f"⚠ Discrepancy in missed feedback loops: {diff_loops} loop(s) difference")
        print()
        print("Possible explanations for the discrepancy:")
        print("1. The text mentions '3+10+10=23', suggesting the author may have counted")
        print("   10 missed loops at Level 3 instead of 6.")
        print("   - This could occur if the author used the total possible loops (10)")
        print("     without properly accounting for the 4 considered important.")
        print("   - Alternative interpretation: The author may have counted Level 3")
        print("     differently, perhaps including unidirectional interactions.")
        print()
        print("2. The correct calculation based on standard N² matrix analysis is:")
        print(f"   - Level 3: {l3['bidirectional']} possible - {l3['considered']} considered = {l3['missed']} missed")
        print(f"   - Level 2-3: {l2_3['bidirectional']} possible - {l2_3['considered']} considered = {l2_3['missed']} missed")
        print(f"   - Level 2: {l2['bidirectional']} possible - {l2['considered']} considered = {l2['missed']} missed")
        print(f"   - Total: {l3['missed']} + {l2_3['missed']} + {l2['missed']} = {results['total_missed_loops']} missed feedback loops")
        print()
        print("3. To arrive at 23 missed loops, one would need:")
        print("   - Either 10 + 10 + 3 = 23 (assuming Level 3 internal = 10 missed)")
        print("   - Or different parameters for 'considered important' feedback loops")
        print()
    else:
        print("✓ Calculation matches the text's stated value.")
        print()
    
    if results['total_missed_unsafe'] != text_unsafe:
        diff_unsafe = text_unsafe - results['total_missed_unsafe']
        print(f"⚠ Discrepancy in missed unsafe interactions: {diff_unsafe} interaction(s) difference")
        print()
        print("This discrepancy is a direct consequence of the missed loops discrepancy:")
        print(f"  - Using text value: {text_loops} × {unsafe_action_types} = {text_loops * unsafe_action_types}")
        print(f"  - Using calculated value: {results['total_missed_loops']} × {unsafe_action_types} = {results['total_missed_unsafe']}")
        print()
        print("The calculation method is correct; the difference stems from the")
        print("interpretation of how many feedback loops were missed at Level 3.")
        print()
    else:
        print("✓ Calculation matches the text's stated value.")
        print()
    
    print("RECOMMENDATION:")
    print("-" * 70)
    print("Based on standard STAMP and N² matrix analysis methodology, the")
    print(f"mathematically correct values are {results['total_missed_loops']} missed feedback loops")
    print(f"and {results['total_missed_unsafe']} missed unsafe interactions.")
    print()


def generalized_hierarchy_analysis(hierarchy_levels, considered_loops, unsafe_action_types=6):
    """
    Generalized function for analyzing N-level hierarchies.
    
    Args:
        hierarchy_levels (list): List of integers representing entities per level,
                                ordered from top (Level 1) to bottom (Level N)
        considered_loops (dict): Dictionary with keys like 'level_N', 'level_N_M'
                                representing considered loops at each level/between levels
        unsafe_action_types (int): Number of STAMP unsafe control action types
        
    Returns:
        dict: Complete analysis results
    """
    total_missed = 0
    results = {'levels': {}, 'inter_levels': {}}
    
    # Analyze internal interactions at each level
    for i, num_entities in enumerate(hierarchy_levels, 1):
        level_key = f'level_{i}'
        _, _, _, bidirectional = calculate_internal_feedback_loops(num_entities)
        considered = considered_loops.get(level_key, 0)
        missed = calculate_missed_interactions(bidirectional, considered)
        
        results['levels'][level_key] = {
            'entities': num_entities,
            'bidirectional': bidirectional,
            'considered': considered,
            'missed': missed
        }
        total_missed += missed
    
    # Analyze inter-level interactions
    for i in range(len(hierarchy_levels) - 1):
        level_a = i + 1
        level_b = i + 2
        inter_key = f'level_{level_a}_{level_b}'
        
        bidirectional = calculate_interlevel_feedback_loops(
            hierarchy_levels[i], hierarchy_levels[i + 1]
        )
        considered = considered_loops.get(inter_key, 0)
        missed = calculate_missed_interactions(bidirectional, considered)
        
        results['inter_levels'][inter_key] = {
            'level_a': level_a,
            'level_b': level_b,
            'bidirectional': bidirectional,
            'considered': considered,
            'missed': missed
        }
        total_missed += missed
    
    results['total_missed_loops'] = total_missed
    results['total_missed_unsafe'] = total_missed * unsafe_action_types
    results['unsafe_action_types'] = unsafe_action_types
    
    return results


def main():
    """Main function to run the STAMP analysis."""
    
    # Define parameters based on the problem statement
    LEVEL_3_ENTITIES = 5  # Most concrete level
    LEVEL_2_ENTITIES = 3  # Middle level
    LEVEL_1_ENTITIES = 1  # Upper level (TZSC system controller)
    UNSAFE_ACTION_TYPES = 6  # NOT, TOO MUCH, TOO LITTLE, TOO EARLY, TOO LATE, WRONG ORDER
    
    # Considered important feedback loops
    LEVEL_3_CONSIDERED = 4
    LEVEL_2_3_CONSIDERED = 5
    LEVEL_2_CONSIDERED = 0
    
    # Values stated in the text
    TEXT_MISSED_LOOPS = 23
    TEXT_MISSED_UNSAFE = 161
    
    # Run the analysis
    results = analyze_stamp_control_structure(
        level_3_entities=LEVEL_3_ENTITIES,
        level_2_entities=LEVEL_2_ENTITIES,
        level_1_entities=LEVEL_1_ENTITIES,
        unsafe_action_types=UNSAFE_ACTION_TYPES,
        level_3_considered=LEVEL_3_CONSIDERED,
        level_2_3_considered=LEVEL_2_3_CONSIDERED,
        level_2_considered=LEVEL_2_CONSIDERED
    )
    
    # Print the results
    print_results(
        results=results,
        level_3_entities=LEVEL_3_ENTITIES,
        level_2_entities=LEVEL_2_ENTITIES,
        level_1_entities=LEVEL_1_ENTITIES,
        unsafe_action_types=UNSAFE_ACTION_TYPES,
        text_loops=TEXT_MISSED_LOOPS,
        text_unsafe=TEXT_MISSED_UNSAFE
    )
    
    # Bonus: Demonstrate generalized function
    print()
    print("=" * 70)
    print("BONUS: Generalized N-Level Hierarchy Analysis")
    print("=" * 70)
    print()
    print("Example: 4-level hierarchy with [1, 2, 3, 5] entities per level")
    print()
    
    hierarchy = [1, 2, 3, 5]  # Top to bottom
    considered = {
        'level_4': 4,  # Level 4 (bottom, 5 entities)
        'level_3_4': 5,  # Between Level 3 and 4
        'level_3': 0,  # Level 3 (3 entities)
        'level_2_3': 0,  # Between Level 2 and 3
        'level_2': 0,  # Level 2 (2 entities)
        'level_1_2': 0,  # Between Level 1 and 2
        'level_1': 0   # Level 1 (1 entity)
    }
    
    gen_results = generalized_hierarchy_analysis(hierarchy, considered)
    
    print(f"Total Missed Feedback Loops: {gen_results['total_missed_loops']}")
    print(f"Total Missed Unsafe Interactions: {gen_results['total_missed_unsafe']}")
    print()
    print("Detailed breakdown:")
    for level_key, data in gen_results['levels'].items():
        print(f"  {level_key}: {data['entities']} entities, {data['bidirectional']} possible, "
              f"{data['considered']} considered, {data['missed']} missed")
    for inter_key, data in gen_results['inter_levels'].items():
        print(f"  {inter_key}: {data['bidirectional']} possible, "
              f"{data['considered']} considered, {data['missed']} missed")


if __name__ == "__main__":
    main()
