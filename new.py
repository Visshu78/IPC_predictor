# -------------------------------
# IMPORTS
# -------------------------------
import sys
import time

print("🔄 Loading dependencies... This may take a moment on first run.")
start_time = time.time()

try:
    from ipc import find_ipc_hybrid
    load_time = time.time() - start_time
    print(f"✅ Dependencies loaded successfully in {load_time:.1f} seconds!")
except KeyboardInterrupt:
    print("\n❌ Loading interrupted by user. Please try again.")
    sys.exit(1)
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure all required packages are installed:")
    print("   pip install sentence-transformers scikit-learn spacy")
    print("   python -m spacy download en_core_web_sm")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error during import: {e}")
    print("💡 Try restarting your terminal or virtual environment.")
    sys.exit(1)

# -------------------------------
# CONFIG
# -------------------------------
# ... (previous config remains the same)
CONFIDENCE_THRESHOLD = 0.55  # Minimum confidence score to display results

# -------------------------------
# DISPLAY RESULTS
# -------------------------------
def display_results(matches, user_input):
    """Display search results in a formatted way"""
    print(f"\n🔍 Top Matches for: '{user_input}'\n" + "="*70)
    
    if not matches:
        print("❌ No relevant IPC sections found. Try a more detailed description.")
        return
    
    # Filter results with confidence above the threshold
    filtered_matches = [match for match in matches if match['score'] >= CONFIDENCE_THRESHOLD]
    
    if not filtered_matches:
        print(f"❌ No IPC sections found with confidence above {CONFIDENCE_THRESHOLD*100}%.")
        print("💡 Try providing a more detailed description of the crime.")
        
        # Show the top result even if below threshold for reference
        if matches:
            top_match = matches[0]
            print(f"\n📋 Closest match (below threshold): {top_match['IPC']} - {top_match['offense']}")
            print(f"   Confidence: {top_match['score']:.4f}")
        return
    
    for i, match in enumerate(filtered_matches, 1):
        print(f"{i}. 📌 Section: {match['IPC']} (Confidence: {match['score']:.4f})")
        print(f"   ⚖️  Offense: {match['offense']}")
        print(f"   ⛓️  Punishment: {match['punishment']}")
        print(f"   🚔 Cognizable: {match['cognizable']}")
        print(f"   🔓 Bailable: {match['bailable']}")
        print(f"   🏛️  Court: {match['court']}")
        
        # Show severity information
        crime_sev = match['crime_severity']
        punish_sev = match['punishment_severity']
        severity_match = match['severity_match']
        
        print(f"   📊 Crime Severity: {crime_sev:.2f}, Punishment Severity: {punish_sev:.2f}")
        print(f"   ✅ Severity Match: {severity_match:.2f}")
        
        if severity_match < 0.7:
            print("   ⚠️  Note: Severity mismatch detected - this section might be too harsh")
        
        print("-"*70)
    
    # Show how many results were filtered out
    if len(filtered_matches) < len(matches):
        print(f"ℹ️  Filtered out {len(matches) - len(filtered_matches)} results with confidence below {CONFIDENCE_THRESHOLD*100}%")

# -------------------------------
# EXAMPLE USAGE
# -------------------------------
if __name__ == "__main__":
    print("🔎 IPC Code Classifier - Enhanced with Severity Calibration")
    print(f"💡 Enter a crime description to find relevant IPC sections")
    print(f"   Only results with confidence above {CONFIDENCE_THRESHOLD*100}% will be shown")
    print("   Type 'quit' to exit\n")
    
    while True:
        user_query = input("📝 Enter a crime description: ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("👋 Goodbye!")
            break
            
        if not user_query:
            print("⚠️  Please enter a description.")
            continue
            
        print("⏳ Analyzing and searching...")
        # Get a larger set of results to filter
        matches = find_ipc_hybrid(user_query, top_k=20)
        
        display_results(matches, user_query)
        
        # Show confidence level of filtered results
        filtered_matches = [match for match in matches if match['score'] >= CONFIDENCE_THRESHOLD]
        if filtered_matches:
            avg_confidence = sum(match['score'] for match in filtered_matches) / len(filtered_matches)
            if avg_confidence < 0.6:
                print("💡 Medium confidence results. The matches might need verification.")
            elif avg_confidence < 0.8:
                print("✅ Good confidence results. These are likely accurate matches.")
            else:
                print("🎯 High confidence results. These are very likely accurate matches.")
