from analyzer import analyze_input


text = input("Enter Terms & Conditions text:\n")
results = analyze_input(text)

if results:
    print("\nDetected Risks:\n")
    for result in results:
        print(f"Sentence: {result['sentence']}")
        print(f"Risk Type: {result['risk_type']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Explanation: {result['explanation']}")
        print(f"Matched Keyword: {result['matched_keyword']}")
        print("-" * 50)

