#!/usr/bin/env python3
"""Test to verify the parser is working correctly by manually checking packed values."""

import sys
sys.path.insert(0, '/puffertank/Showdown/PufferLib')

import numpy as np
from pufferlib.ocean.showdown.showdown import Showdown
from pufferlib.ocean.showdown.py_print import ShowdownParser

def test_parser():
    """Test that the parser correctly interprets packed move values."""
    env = Showdown()
    obs, info = env.reset()
    obs = obs[0]  # Unwrap batch dimension
    
    print("Testing Parser Logic")
    print("="*60)
    
    # Create some fake packed move values to test the unpacking
    print("\n1. Testing move unpacking function:")
    test_values = [
        0,      # Should be: no move (id=0, pp=0)
        2654,   # Should unpack to something specific
        5189,   # Should unpack to something specific
        1385,   # Should unpack to something specific
    ]
    
    for packed in test_values:
        move_id = packed & 0xFF
        pp = (packed >> 8) & 0x3F
        print(f"  Packed value {packed}: move_id={move_id}, pp={pp}")
        unpacked = ShowdownParser.unpack_move(packed)
        print(f"    Parser result: {unpacked}")
    
    # Now check real observations
    print("\n2. Checking P1's active pokemon (should have all moves visible):")
    parsed = ShowdownParser.parse_observation(obs)
    p1_active = parsed['players'][0]['active_pokemon']
    p1_active_idx = parsed['players'][0]['active_index']
    
    if p1_active and p1_active_idx is not None:
        # Calculate offset for P1's active pokemon
        p1_offset = 4 + (p1_active_idx * 2 + 0) * 7
        p1_raw = obs[p1_offset:p1_offset+7]
        print(f"  P1 active slot {p1_active_idx} raw: {p1_raw}")
        print(f"  P1 moves from parser: {p1_active['moves']}")
        print(f"  P1 move count: {len(p1_active['moves'])}")
        
        # Manually unpack to verify parser
        for i in range(4):
            packed = int(p1_raw[1+i])
            if packed != 0:
                move_id = packed & 0xFF
                pp = (packed >> 8) & 0x3F
                print(f"    Manual unpack move {i}: packed={packed}, id={move_id}, pp={pp}")
    
    print("\n3. Checking P2's active pokemon (opponent - may have hidden moves):")
    p2_active = parsed['players'][1]['active_pokemon']
    p2_active_idx = parsed['players'][1]['active_index']
    
    if p2_active and p2_active_idx is not None:
        # Calculate offset for P2's active pokemon
        p2_offset = 4 + (p2_active_idx * 2 + 1) * 7
        p2_raw = obs[p2_offset:p2_offset+7]
        print(f"  P2 active slot {p2_active_idx} raw: {p2_raw}")
        print(f"  P2 moves from parser: {p2_active['moves']}")
        print(f"  P2 move count: {len(p2_active['moves'])}")
        
        # Manually unpack to verify parser
        for i in range(4):
            packed = int(p2_raw[1+i])
            if packed != 0:
                move_id = packed & 0xFF
                pp = (packed >> 8) & 0x3F
                print(f"    Manual unpack move {i}: packed={packed}, id={move_id}, pp={pp}")
            else:
                print(f"    Manual unpack move {i}: packed=0 (no move/unrevealed)")
    
    print("\n4. Running a few steps to see if P2 moves get revealed:")
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminal, truncated, info = env.step(action)
        obs = obs[0]
        
        parsed = ShowdownParser.parse_observation(obs)
        p2_active = parsed['players'][1]['active_pokemon']
        p2_active_idx = parsed['players'][1]['active_index']
        
        if p2_active and p2_active_idx is not None:
            p2_offset = 4 + (p2_active_idx * 2 + 1) * 7
            p2_raw = obs[p2_offset:p2_offset+7]
            
            print(f"\n  Step {step+1}: P2 slot {p2_active_idx}")
            print(f"    Raw: {p2_raw}")
            print(f"    Parser found {len(p2_active['moves'])} moves")
            
            # Check each slot
            revealed_count = sum(1 for i in range(4) if int(p2_raw[1+i]) != 0)
            print(f"    Manual count: {revealed_count} non-zero packed values")
            
            if terminal[0]:
                break
    
    env.close()
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("If manual unpacking shows 0s but parser also shows 0 moves,")
    print("then the bug is in the C code, NOT the parser.")
    print("If manual unpacking shows non-zero but parser shows 0 moves,")
    print("then the bug is in the parser.")
    print("="*60)

if __name__ == "__main__":
    test_parser()
