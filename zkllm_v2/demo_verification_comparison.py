#!/usr/bin/env python3
"""
Simple demonstration showing the difference between fake and real verification.

This shows WHY the current verifier is not sufficient for production.
"""

import os
import sys

def banner(text):
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def test_case(name, cmd, expected_result):
    print(f"Test: {name}")
    print(f"Command: {cmd}")
    result = os.system(cmd + " > /tmp/verify_output.txt 2>&1")
    
    with open("/tmp/verify_output.txt", "r") as f:
        output = f.read()
    
    if "✅ PROOF VERIFICATION SUCCESSFUL" in output:
        actual = "PASS"
    else:
        actual = "FAIL"
    
    print(f"Expected: {expected_result}")
    print(f"Actual: {actual}")
    
    if actual == expected_result:
        print("✅ Test behavior is CORRECT")
    else:
        print("❌ Test behavior is WRONG!")
    
    return actual == expected_result

if __name__ == "__main__":
    banner("ZKLLM Verification Security Test")
    
    print("This demonstrates the difference between:")
    print("  1. Fake verification (just checks file format)")
    print("  2. Real verification (checks cryptographic validity)\n")
    
    VERIFIER = "./verify_rmsnorm_v2"
    PROOF = "zkllm-workdir/Llama-2-7b/layer-0-rmsnorm-proof.bin"
    WORKDIR = "zkllm-workdir/Llama-2-7b"
    
    if not os.path.exists(VERIFIER):
        print(f"❌ Verifier not found: {VERIFIER}")
        print("   Run: make -f Makefile_v2 verify_rmsnorm_v2")
        sys.exit(1)
    
    if not os.path.exists(PROOF):
        print(f"❌ Proof not found: {PROOF}")
        print("   Generate it first with: python3 llama-rmsnorm_v2.py 7 0 input 128 ...")
        sys.exit(1)
    
    results = []
    
    # Test 1: Correct verification
    banner("TEST 1: Valid Proof with Correct Commitment")
    print("Verifying layer-0 proof against layer-0 commitment")
    print("This SHOULD pass (proof matches commitment)\n")
    cmd = f"{VERIFIER} {PROOF} {WORKDIR} layer-0 input"
    results.append(test_case("Valid proof + correct commitment", cmd, "PASS"))
    
    # Test 2: Wrong commitment - THE CRITICAL TEST
    banner("TEST 2: Valid Proof with WRONG Commitment")
    print("Verifying layer-0 proof against layer-1 commitment")
    print("This SHOULD FAIL (different layer = different weights)")
    print("But current verifier will PASS (proving it's not checking crypto!)\n")
    cmd = f"{VERIFIER} {PROOF} {WORKDIR} layer-1 input"
    # Expect FAIL, if it PASSES that's wrong, so we invert the result
    test2_result = test_case("Valid proof + WRONG commitment", cmd, "FAIL")
    results.append(test2_result)
    
    # Test 3: Different layer entirely
    banner("TEST 3: Valid Proof with Even More Wrong Commitment")
    print("Verifying layer-0 proof against layer-5 commitment")
    print("This should DEFINITELY fail!\n")
    cmd = f"{VERIFIER} {PROOF} {WORKDIR} layer-5 input"
    test3_result = test_case("Valid proof + very wrong commitment", cmd, "FAIL")
    results.append(test3_result)
    
    # Summary
    banner("SUMMARY")
    
    total = len(results)
    passed = sum(results)
    
    print(f"Tests: {passed}/{total}")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED")
        print("\nThis means the verifier is doing REAL cryptographic verification!")
        print("It correctly:")
        print("  ✓ Accepts valid proofs with correct commitments")
        print("  ✓ Rejects valid proofs with wrong commitments")
        print("  ✓ Cryptographically binds proof to specific layer weights")
    else:
        print(f"\n⚠️  {total - passed} TESTS FAILED")
        print("\nThis means the verifier is NOT doing real verification!")
        print("Current verifier:")
        print("  ✓ Loads proofs correctly")
        print("  ✓ Loads commitments correctly")
        print("  ✗ Does NOT verify cryptographic binding")
        print("  ✗ Accepts proofs with wrong commitments")
        print("\nWhy this matters:")
        print("  • A malicious prover could use weights from a different layer")
        print("  • The verifier would incorrectly accept it")
        print("  • No cryptographic security!")
    
    banner("WHAT NEEDS TO BE FIXED")
    print("Current verifier (verify_rmsnorm_v2.cu):")
    print("  Lines 71-76: Only checks proof SIZE")
    print("  Lines 86-91: Only prints a MESSAGE")
    print("  Line 113: Returns success if proof exists")
    print("\nWhat REAL verifier needs:")
    print("  1. Recompute RMSNorm forward pass")
    print("  2. Verify hadamard sumcheck polynomials")
    print("  3. Call verifyWeightClaim() with reconstructed claim")
    print("  4. This requires saving random challenges with proof")
    print("\nEstimated fix time: ~3 hours")
    print("="*80 + "\n")
    
    sys.exit(0 if passed == total else 1)
