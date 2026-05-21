import sys
sys.path.insert(0, '.')
from api.main import detect_language

tests = [
    ("how do i claim", "en"),
    ("can you help", "en"),
    ("hi", "en"),
    ("what is covered", "en"),
    ("I need to file a car insurance claim for a rear-end collision that happened yesterday afternoon", "en"),
    ("Mi coche fue golpeado ayer en un accidente de trafico. El danyo estimado es de 5000 euros.", "es"),
]

all_pass = True
for msg, expected in tests:
    result = detect_language(msg)
    ok = result == expected
    if not ok:
        all_pass = False
    print(f"  [{'OK' if ok else 'FAIL'}] \"{msg[:50]}\" -> {result} (expected {expected})")

print("\nAll passed:", all_pass)
