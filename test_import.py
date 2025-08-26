#!/usr/bin/env python3

print("Testing import...")
try:
    from ipc import find_ipc_hybrid
    print("✅ Import successful!")
    print("Testing function call...")
    
    # Test with a simple query
    result = find_ipc_hybrid("theft of mobile phone", top_k=3)
    print(f"✅ Function call successful! Found {len(result)} results")
    
    # Display first result
    if result:
        print(f"First result: {result[0]['IPC']} - {result[0]['offense']}")
        print(f"Score: {result[0]['score']:.4f}")
    
except ImportError as e:
    print(f"❌ Import failed: {e}")
except Exception as e:
    print(f"❌ Function call failed: {e}")
