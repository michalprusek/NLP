#!/usr/bin/env python3
"""
Test script to verify updated Claudette prompts and templates
"""
import json
from pathlib import Path
from src.claudette_evaluator import extract_labels_from_output, get_ground_truth_labels, LABEL_MAP

def test_metadata():
    """Test that metadata.json has all 9 labels"""
    print("="*80)
    print("TEST 1: Metadata completeness")
    print("="*80)
    
    with open('datasets/tos_local/metadata.json') as f:
        metadata = json.load(f)
    
    expected_labels = set(range(9))
    actual_labels = set(int(k) for k in metadata['labels'].keys())
    
    if expected_labels == actual_labels:
        print("✅ PASS: All 9 labels present in metadata.json")
        for idx in sorted(actual_labels):
            print(f"   {idx}: {metadata['labels'][str(idx)]}")
    else:
        missing = expected_labels - actual_labels
        extra = actual_labels - expected_labels
        print(f"❌ FAIL: Label mismatch")
        if missing:
            print(f"   Missing: {missing}")
        if extra:
            print(f"   Extra: {extra}")
        return False
    
    # Check note about label 6
    if 'notes' in metadata:
        print(f"✅ Note present: {metadata['notes'][:80]}...")
    
    return True


def test_initial_prompt():
    """Test that initial.txt has correct format"""
    print("\n" + "="*80)
    print("TEST 2: Initial prompt template")
    print("="*80)
    
    prompt_file = Path('src/prompts/claudette/initial.txt')
    prompt = prompt_file.read_text(encoding='utf-8').strip()
    
    print(f"Prompt:\n{prompt}\n")
    
    # Check key elements
    checks = [
        ("FAIR/UNFAIR distinction", "FAIR" in prompt and "UNFAIR" in prompt),
        ("90% statistic", "90%" in prompt),
        ("LABELS: NONE format", "LABELS: NONE" in prompt),
        ("Multi-label support", "multiple" in prompt.lower() or "ALL" in prompt),
        ("No label 6", "6=" not in prompt),
        ("Categories 0-5, 7-8", "0=" in prompt and "5=" in prompt and "7=" in prompt and "8=" in prompt),
    ]
    
    all_passed = True
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False
    
    return all_passed


def test_gradient_prompt():
    """Test that gradient.txt has correct format"""
    print("\n" + "="*80)
    print("TEST 3: Gradient prompt template")
    print("="*80)
    
    prompt_file = Path('src/prompts/claudette/gradient.txt')
    prompt = prompt_file.read_text(encoding='utf-8').strip()
    
    # Check key elements
    checks = [
        ("FAIR terminology", "FAIR" in prompt),
        ("90% statistic", "90%" in prompt),
        ("Micro-F1 metric", "Micro-F1" in prompt or "micro_f1" in prompt),
        ("Recall warning", "Recall" in prompt),
        ("No label 6 note", "label 6 not in dataset" in prompt.lower() or "6=" not in prompt or "Note:" in prompt),
        ("Categories 0-5, 7-8", "0=" in prompt and "5=" in prompt and "7=" in prompt and "8=" in prompt),
    ]
    
    all_passed = True
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False
    
    return all_passed


def test_edit_prompt():
    """Test that edit.txt has correct format"""
    print("\n" + "="*80)
    print("TEST 4: Edit prompt template")
    print("="*80)
    
    prompt_file = Path('src/prompts/claudette/edit.txt')
    prompt = prompt_file.read_text(encoding='utf-8').strip()
    
    # Check key elements
    checks = [
        ("FAIR terminology", "FAIR" in prompt),
        ("90% statistic", "90%" in prompt),
        ("Micro-F1 metric", "Micro-F1" in prompt),
        ("BALANCE rule", "BALANCE" in prompt),
        ("Categories note", "0-5, 7-8" in prompt or "label 6" in prompt.lower()),
        ("Output format rule", "LABELS:" in prompt),
    ]
    
    all_passed = True
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False
    
    return all_passed


def test_opro_meta_prompt():
    """Test that opro_meta.txt has correct format"""
    print("\n" + "="*80)
    print("TEST 5: OPRO meta-prompt template")
    print("="*80)
    
    prompt_file = Path('src/prompts/claudette/opro_meta.txt')
    prompt = prompt_file.read_text(encoding='utf-8').strip()
    
    # Check key elements
    checks = [
        ("FAIR terminology", "FAIR" in prompt),
        ("90% statistic", "90%" in prompt),
        ("8 active categories", "8 active" in prompt.lower() or "label 6" in prompt.lower()),
        ("No label 6 in examples", "6=" not in prompt.split("EXAMPLE OUTPUT")[1] if "EXAMPLE OUTPUT" in prompt else True),
        ("Micro-F1 metric", "Micro-F1" in prompt),
        ("Multi-label emphasis", "multiple" in prompt.lower() or "MULTI-LABEL" in prompt),
    ]
    
    all_passed = True
    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")
        if not result:
            all_passed = False
    
    return all_passed


def test_default_prompts():
    """Test that evaluate_claudette.py has updated DEFAULT_PROMPTS"""
    print("\n" + "="*80)
    print("TEST 6: DEFAULT_PROMPTS in evaluate_claudette.py")
    print("="*80)
    
    # Import the module
    import evaluate_claudette
    
    prompts = evaluate_claudette.DEFAULT_PROMPTS
    
    print(f"Available prompts: {list(prompts.keys())}\n")
    
    all_passed = True
    for name, prompt in prompts.items():
        print(f"--- {name} ---")
        print(f"{prompt[:150]}...")
        
        # Check key elements
        has_fair = "FAIR" in prompt
        has_labels_format = "LABELS:" in prompt
        has_none = "NONE" in prompt
        no_label_6 = "6=" not in prompt or "6:" not in prompt
        
        status = "✅" if (has_fair and has_labels_format and has_none and no_label_6) else "❌"
        print(f"{status} FAIR: {has_fair}, LABELS: {has_labels_format}, NONE: {has_none}, No label 6: {no_label_6}\n")
        
        if not (has_fair and has_labels_format and has_none and no_label_6):
            all_passed = False
    
    return all_passed


def test_extraction():
    """Test that extraction still works with new format"""
    print("\n" + "="*80)
    print("TEST 7: Label extraction with new format")
    print("="*80)
    
    test_cases = [
        ("LABELS: NONE", set()),
        ("LABELS: 0", {0}),
        ("LABELS: 0, 3", {0, 3}),
        ("LABELS: 5, 8", {5, 8}),
        ("This is a fair clause", set()),
        ("The clause is UNFAIR. LABELS: 2", {2}),
    ]
    
    all_passed = True
    for text, expected in test_cases:
        extracted = extract_labels_from_output(text)
        passed = extracted == expected
        status = "✅" if passed else "❌"
        print(f"{status} '{text:40s}' → {sorted(extracted) if extracted else 'NONE'} (expected: {sorted(expected) if expected else 'NONE'})")
        if not passed:
            all_passed = False
    
    return all_passed


def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("CLAUDETTE PROMPT UPDATE VERIFICATION")
    print("="*80 + "\n")
    
    tests = [
        ("Metadata completeness", test_metadata),
        ("Initial prompt", test_initial_prompt),
        ("Gradient prompt", test_gradient_prompt),
        ("Edit prompt", test_edit_prompt),
        ("OPRO meta-prompt", test_opro_meta_prompt),
        ("DEFAULT_PROMPTS", test_default_prompts),
        ("Label extraction", test_extraction),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Prompts are ready for use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    exit(main())

