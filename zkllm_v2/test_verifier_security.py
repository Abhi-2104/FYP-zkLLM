#!/usr/bin/env python3
"""
Test to demonstrate that current verifier is NOT doing real cryptographic verification.

This will show that the verifier accepts:
1. Corrupted proofs (shouldn't!)
2. Wrong layer commitments (shouldn't!)
3. Anything that looks like a proof file (shouldn't!)

For panel: This proves we need actual cryptographic checks.
"""

import os
import shutil
import subprocess
import sys

VERIFIER = "./verify_rmsnorm_v2"
WORKDIR = "zkllm-workdir/Llama-2-7b"
PROOF_FILE = f"{WORKDIR}/layer-0-rmsnorm-proof.bin"
BACKUP_FILE = f"{WORKDIR}/layer-0-rmsnorm-proof.bin.backup"

def run_verifier(proof_file, layer_prefix="layer-0", which="input"):
    """Run verifier and return True if it reports success."""
    cmd = [VERIFIER, proof_file, WORKDIR, layer_prefix, which]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Check if verifier reported success
    success = "✅ PROOF VERIFICATION SUCCESSFUL" in result.stdout
    return success, result.stdout

def test_original():
    """Test 1: Original valid proof should pass."""
    print("\n" + "="*80)
    print("TEST 1: Original valid proof")
    print("="*80)
    
    success, output = run_verifier(PROOF_FILE)
    print(output)
    
    if success:
        print("✅ RESULT: PASSED (expected)")
    else:
        print("❌ RESULT: FAILED (unexpected!)")
    
    return success

def test_corrupted_proof():
    """Test 2: Corrupted proof should FAIL but currently PASSES."""
    print("\n" + "="*80)
    print("TEST 2: Corrupted proof (tamper with bytes)")
    print("="*80)
    
    # Backup original
    shutil.copy(PROOF_FILE, BACKUP_FILE)
    
    try:
        # Corrupt the proof file (overwrite some bytes)
        print("Corrupting proof file (bytes 100-132)...")
        with open(PROOF_FILE, 'r+b') as f:
            f.seek(100)
            f.write(b'\xFF' * 32)  # Overwrite with garbage
        
        success, output = run_verifier(PROOF_FILE)
        print(output)
        
        if success:
            print("❌ RESULT: PASSED (This is WRONG! Corrupted proof should fail!)")
            print("   → Proves verifier is NOT checking cryptographic validity")
        else:
            print("✅ RESULT: FAILED (This is correct)")
            print("   → Verifier is doing real cryptographic checks")
        
        return not success  # Test passes if verifier correctly rejects
        
    finally:
        # Restore original
        shutil.move(BACKUP_FILE, PROOF_FILE)

def test_wrong_commitment():
    """Test 3: Proof for layer-0 verified against layer-1 commitment should FAIL."""
    print("\n" + "="*80)
    print("TEST 3: Wrong commitment (layer-0 proof vs layer-1 commitment)")
    print("="*80)
    
    print("Using layer-0 proof with layer-1 commitment...")
    print("(Like claiming proof of one layer's computation with different layer's weights)")
    
    success, output = run_verifier(PROOF_FILE, layer_prefix="layer-1")
    print(output)
    
    if success:
        print("❌ RESULT: PASSED (This is WRONG! Different commitment should fail!)")
        print("   → Proves verifier is NOT binding proof to commitment")
    else:
        print("✅ RESULT: FAILED (This is correct)")
        print("   → Verifier is checking commitment binding")
    
    return not success

def test_truncated_proof():
    """Test 4: Truncated proof file should fail."""
    print("\n" + "="*80)
    print("TEST 4: Truncated proof file")
    print("="*80)
    
    # Create truncated version
    truncated_file = f"{WORKDIR}/layer-0-rmsnorm-proof-truncated.bin"
    
    try:
        # Copy first 500 bytes only
        print("Creating truncated proof (first 500 bytes only)...")
        with open(PROOF_FILE, 'rb') as f:
            data = f.read(500)
        
        with open(truncated_file, 'wb') as f:
            f.write(data)
        
        success, output = run_verifier(truncated_file)
        print(output)
        
        if success:
            print("❌ RESULT: PASSED (This is WRONG! Truncated proof should fail!)")
            print("   → Proves verifier is NOT validating proof completeness")
        else:
            print("✅ RESULT: FAILED (This is correct)")
            print("   → Verifier is checking proof structure")
        
        return not success
        
    finally:
        if os.path.exists(truncated_file):
            os.remove(truncated_file)

def main():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                   ZKLLM VERIFIER SECURITY TEST SUITE                         ║
║                                                                              ║
║  Purpose: Demonstrate whether verifier does REAL cryptographic verification ║
║                                                                              ║
║  A SECURE verifier should:                                                   ║
║    ✓ Accept valid proofs                                                    ║
║    ✗ Reject corrupted proofs                                                ║
║    ✗ Reject proofs verified against wrong commitments                       ║
║    ✗ Reject incomplete/truncated proofs                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    if not os.path.exists(VERIFIER):
        print(f"❌ Error: Verifier executable not found: {VERIFIER}")
        print("   Run: make -f Makefile_v2 verify_rmsnorm_v2")
        return 1
    
    if not os.path.exists(PROOF_FILE):
        print(f"❌ Error: Proof file not found: {PROOF_FILE}")
        print("   Generate it first with rmsnorm_v2")
        return 1
    
    results = []
    
    # Run all tests
    results.append(("Valid proof accepted", test_original()))
    results.append(("Corrupted proof rejected", test_corrupted_proof()))
    results.append(("Wrong commitment rejected", test_wrong_commitment()))
    results.append(("Truncated proof rejected", test_truncated_proof()))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nScore: {passed}/{total}")
    
    if passed == total:
        print("\n✅ EXCELLENT: Verifier is doing REAL cryptographic verification!")
        print("   Safe to use for production and demo to panel.")
    elif passed == 1:
        print("\n⚠️  WARNING: Verifier only checks file format, NOT cryptographic validity!")
        print("   Current verifier would accept INVALID proofs from malicious prover.")
        print("   Need to implement:")
        print("   1. Sumcheck polynomial verification")
        print("   2. Commitment opening verification (call verifyWeightClaim)")
        print("   3. Binding checks between proof and commitment")
    else:
        print(f"\n⚠️  PARTIAL: Verifier has some checks but not complete.")
        print(f"   {total - passed} security checks are missing.")
    
    print("\n" + "="*80)
    print("For your panel demo:")
    print("="*80)
    if passed < total:
        print("Be honest about current state:")
        print("  ✓ Infrastructure complete (proof gen, serialization, loading)")
        print("  ✓ Verifier executable works")
        print("  ✗ Cryptographic verification not yet wired up")
        print("\nNext step: Call verifyWeightClaim() and verify sumcheck polynomials")
    else:
        print("You can confidently demonstrate:")
        print("  ✓ End-to-end ZK proof pipeline")
        print("  ✓ Cryptographically secure verification")
        print("  ✓ Proof/commitment binding")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
