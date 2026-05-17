import sys
sys.path.insert(0, "/Users/coyscoyscoys/shl_assignment")
from src.catalog import load_catalog
from src.retriever import build_index, retrieve_candidates

load_catalog()
build_index()

EXPECTED: dict[str, list[str]] = {
    "C1: leadership selection": [
        "Occupational Personality Questionnaire OPQ32r",
        "OPQ Universal Competency Report 2.0",
        "OPQ Leadership Report",
    ],
    "C2: Rust networking": [
        "Smart Interview Live Coding",
        "Linux Programming (General)",
        "Networking and Implementation (New)",
        "SHL Verify Interactive G+",
        "Occupational Personality Questionnaire OPQ32r",
    ],
    "C3: contact centre US": [
        "SVAR Spoken English (US) (New)",
        "Contact Center Call Simulation (New)",
        "Entry Level Customer Serv - Retail & Contact Center",
        "Customer Service Phone Simulation",
    ],
    "C4: graduate finance": [
        "SHL Verify Interactive - Numerical Reasoning",
        "Financial Accounting (New)",
        "Basic Statistics (New)",
        "Graduate Scenarios",
        "Occupational Personality Questionnaire OPQ32r",
    ],
    "C5: sales reskill": [
        "Global Skills Assessment",
        "Global Skills Development Report",
        "Occupational Personality Questionnaire OPQ32r",
        "OPQ MQ Sales Report",
        "Sales Transformation 2.0 - Individual Contributor",
    ],
    "C6: plant safety industrial": [
        "Manufac. & Indust. - Safety & Dependability 8.0",
        "Workplace Health and Safety (New)",
    ],
    "C7: healthcare HIPAA Spanish": [
        "HIPAA (Security)",
        "Medical Terminology (New)",
        "Microsoft Word 365 - Essentials (New)",
        "Dependability and Safety Instrument (DSI)",
        "Occupational Personality Questionnaire OPQ32r",
    ],
    "C8: admin Excel Word": [
        "Microsoft Excel 365 (New)",
        "Microsoft Word 365 (New)",
        "MS Excel (New)",
        "MS Word (New)",
        "Occupational Personality Questionnaire OPQ32r",
    ],
    "C9: Java Spring AWS Docker": [
        "Core Java (Advanced Level) (New)",
        "Spring (New)",
        "SQL (New)",
        "Amazon Web Services (AWS) Development (New)",
        "Docker (New)",
        "SHL Verify Interactive G+",
        "Occupational Personality Questionnaire OPQ32r",
    ],
    "C10: graduate trainee": [
        "SHL Verify Interactive G+",
        "Graduate Scenarios",
    ],
}

QUERIES: dict[str, str] = {
    "C1: leadership selection": "senior leadership selection director benchmark CXO",
    "C2: Rust networking": "senior Rust engineer networking infrastructure high performance cognitive",
    "C3: contact centre US": "entry-level contact centre screening customer service English US spoken",
    "C4: graduate finance": "graduate financial analyst numerical reasoning finance knowledge situational judgement",
    "C5: sales reskill": "sales re-skill restructuring talent audit personality development",
    "C6: plant safety industrial": "plant operator chemical facility safety dependability industrial manufacturing",
    "C7: healthcare HIPAA Spanish": "healthcare admin bilingual Spanish HIPAA patient records dependability",
    "C8: admin Excel Word": "admin assistant Excel Word simulation screening quick",
    "C9: Java Spring AWS Docker": "Senior Full-Stack Engineer Java Spring REST SQL AWS Docker microservice backend",
    "C10: graduate trainee": "graduate management trainee cognitive personality situational judgement",
}

total = 0
hit = 0
for label, expected in EXPECTED.items():
    query = QUERIES[label]
    results = retrieve_candidates(query, top_k=15)
    result_names = [r["name"] for r in results]
    expected_set = set(expected)
    found = expected_set & set(result_names)
    rec = len(found) / len(expected_set) if expected_set else 1.0

    total += rec
    status = "PASS" if rec >= 0.6 else "FAIL"
    print(f"[{status}] {label}: recall={rec:.1f}")
    for e in expected:
        marker = "OK" if e in result_names else "MISS"
        print(f"    {marker} {e}")

mean = total / len(EXPECTED)
print(f"\nMean Retrieval Recall@15: {mean:.3f}")
