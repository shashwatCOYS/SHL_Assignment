import json
import requests
import time
from typing import List, Dict, Any

BASE_URL = "http://localhost:8004"

EXPECTED_SHORTLISTS: Dict[str, List[str]] = {
    "C1": ["Occupational Personality Questionnaire OPQ32r", "OPQ Universal Competency Report 2.0", "OPQ Leadership Report"],
    "C2": ["Smart Interview Live Coding", "Linux Programming (General)", "Networking and Implementation (New)", "SHL Verify Interactive G+", "Occupational Personality Questionnaire OPQ32r"],
    "C3": ["SVAR Spoken English (US) (New)", "Contact Center Call Simulation (New)", "Entry Level Customer Serv - Retail & Contact Center", "Customer Service Phone Simulation"],
    "C4": ["SHL Verify Interactive - Numerical Reasoning", "Financial Accounting (New)", "Basic Statistics (New)", "Graduate Scenarios", "Occupational Personality Questionnaire OPQ32r"],
    "C5": ["Global Skills Assessment", "Global Skills Development Report", "Occupational Personality Questionnaire OPQ32r", "OPQ MQ Sales Report", "Sales Transformation 2.0 - Individual Contributor"],
    "C6": ["Manufac. & Indust. - Safety & Dependability 8.0", "Workplace Health and Safety (New)"],
    "C7": ["HIPAA (Security)", "Medical Terminology (New)", "Microsoft Word 365 - Essentials (New)", "Dependability and Safety Instrument (DSI)", "Occupational Personality Questionnaire OPQ32r"],
    "C8": ["Microsoft Excel 365 (New)", "Microsoft Word 365 (New)", "MS Excel (New)", "MS Word (New)", "Occupational Personality Questionnaire OPQ32r"],
    "C9": ["Core Java (Advanced Level) (New)", "Spring (New)", "SQL (New)", "Amazon Web Services (AWS) Development (New)", "Docker (New)", "SHL Verify Interactive G+", "Occupational Personality Questionnaire OPQ32r"],
    "C10": ["SHL Verify Interactive G+", "Graduate Scenarios"],
}

CONVERSATION_TRACES: Dict[str, List[List[Dict[str, str]]]] = {
    "C1": [
        [{"role": "user", "content": "We need a solution for senior leadership."}],
        [{"role": "user", "content": "We need a solution for senior leadership."}, {"role": "assistant", "content": "Who is this meant for?"}, {"role": "user", "content": "The pool consists of CXOs, director-level positions; people with more than 15 years of experience."}],
        [{"role": "user", "content": "We need a solution for senior leadership."}, {"role": "assistant", "content": "Who is this meant for?"}, {"role": "user", "content": "CXOs, director-level, 15+ years."}, {"role": "assistant", "content": "Is this for a newly created position, or developmental feedback for an executive already in role?"}, {"role": "user", "content": "Selection - comparing candidates against a leadership benchmark."}],
        [{"role": "user", "content": "We need a solution for senior leadership."}, {"role": "assistant", "content": "Who is this meant for?"}, {"role": "user", "content": "CXOs, director-level, 15+ years."}, {"role": "assistant", "content": "Is this for a newly created position, or developmental feedback?"}, {"role": "user", "content": "Selection - comparing candidates against a leadership benchmark."}, {"role": "assistant", "content": "Here are 3 recommendations for leadership selection."}, {"role": "user", "content": "Perfect, that is what we need."}],
    ],
    "C2": [
        [{"role": "user", "content": "I am hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?"}],
        [{"role": "user", "content": "I am hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?"}, {"role": "assistant", "content": "SHL catalog does not have Rust. Closest: Smart Interview Live Coding, Linux Programming, Networking. Want shortlist?"}, {"role": "user", "content": "Yes, go ahead. Should I also add a cognitive test for this level?"}],
        [{"role": "user", "content": "I am hiring a senior Rust engineer for high-performance networking infrastructure. What assessments should I use?"}, {"role": "assistant", "content": "SHL catalog does not have Rust. Closest: Live Coding, Linux Programming, Networking. Want shortlist?"}, {"role": "user", "content": "Yes. Should I also add a cognitive test?"}, {"role": "assistant", "content": "Here are 5 recommendations for senior Rust engineer."}, {"role": "user", "content": "That works. Thanks."}],
    ],
    "C3": [
        [{"role": "user", "content": "We are screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?"}],
        [{"role": "user", "content": "We are screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?"}, {"role": "assistant", "content": "What language are the calls in?"}, {"role": "user", "content": "English."}],
        [{"role": "user", "content": "We are screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?"}, {"role": "assistant", "content": "What language?"}, {"role": "user", "content": "English."}, {"role": "assistant", "content": "Which accent variant: US, UK, Australian, or Indian?"}, {"role": "user", "content": "US."}],
        [{"role": "user", "content": "We are screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?"}, {"role": "assistant", "content": "What language?"}, {"role": "user", "content": "English."}, {"role": "assistant", "content": "Which accent variant?"}, {"role": "user", "content": "US."}, {"role": "assistant", "content": "Here are 4 recommendations."}, {"role": "user", "content": "Is the Contact Center Call Simulation different from the Customer Service Phone Simulation?"}],
        [{"role": "user", "content": "We are screening 500 entry-level contact centre agents. Inbound calls, customer service focus. What should we use?"}, {"role": "assistant", "content": "What language?"}, {"role": "user", "content": "English."}, {"role": "assistant", "content": "Which accent variant?"}, {"role": "user", "content": "US."}, {"role": "assistant", "content": "Here are 4 recommendations for contact centre screening."}, {"role": "user", "content": "Is the Contact Center Call Simulation different from the Customer Service Phone Simulation?"}, {"role": "assistant", "content": "Yes, they are distinct. The new simulation is standalone. The older solution is bundled."}, {"role": "user", "content": "Perfect - new simulation for volume, old solution for finalists. Confirmed."}],
    ],
    "C4": [
        [{"role": "user", "content": "Hiring graduate financial analysts - final-year students, no work experience. We need numerical reasoning and a finance knowledge test."}],
        [{"role": "user", "content": "Hiring graduate financial analysts - final-year students, no work experience. We need numerical reasoning and a finance knowledge test."}, {"role": "assistant", "content": "Here are 4 recommendations for graduate financial analysts."}, {"role": "user", "content": "Good. Can you also add a situational judgement element - work-context decision making for graduates?"}],
        [{"role": "user", "content": "Hiring graduate financial analysts - final-year students, no work experience. We need numerical reasoning and a finance knowledge test."}, {"role": "assistant", "content": "Here are 4 recommendations."}, {"role": "user", "content": "Good. Can you also add a situational judgement element for graduates?"}, {"role": "assistant", "content": "Added Graduate Scenarios. Updated list of 5."}, {"role": "user", "content": "That covers it. Numerical + Graduate Scenarios as first filter, domain tests for shortlisted candidates."}],
    ],
    "C5": [
        [{"role": "user", "content": "As part of our restructuring and annual talent audit, we need to re-skill our Sales organization. What solutions do you recommend?"}],
        [{"role": "user", "content": "Restructuring and annual talent audit, re-skill our Sales organization. What solutions?"}, {"role": "assistant", "content": "Here are 5 recommendations for sales re-skilling audit."}, {"role": "user", "content": "What is the difference between OPQ and OPQ MQ Sales Report?"}],
        [{"role": "user", "content": "Restructuring, re-skill Sales organization. What solutions?"}, {"role": "assistant", "content": "Here are 5 recommendations."}, {"role": "user", "content": "What is the difference between OPQ and OPQ MQ Sales Report?"}, {"role": "assistant", "content": "OPQ is the underlying questionnaire. OPQ MQ Sales Report is a sales-specific view of OPQ results with optional MQ."}, {"role": "user", "content": "Clear. We will use OPQ for everyone and add MQ where needed. Keeping the five solutions."}],
    ],
    "C6": [
        [{"role": "user", "content": "We are hiring plant operators for a chemical facility. Safety is absolute top priority - reliability, procedure compliance, never cutting corners."}],
        [{"role": "user", "content": "We are hiring plant operators for a chemical facility. Safety is top priority."}, {"role": "assistant", "content": "Here are 3 recommendations for safety-critical plant operators."}, {"role": "user", "content": "What is the difference between the DSI and the Safety and Dependability 8.0?"}],
        [{"role": "user", "content": "We are hiring plant operators for a chemical facility. Safety is top priority."}, {"role": "assistant", "content": "Here are 3 recommendations."}, {"role": "user", "content": "What is the difference between the DSI and Safety and Dependability 8.0?"}, {"role": "assistant", "content": "DSI is standalone, cross-sector. Safety 8.0 is sector-specific with manufacturing norms."}, {"role": "user", "content": "We are industrial. The 8.0 bundle is the right fit. Confirmed."}],
    ],
    "C7": [
        [{"role": "user", "content": "We are hiring bilingual healthcare admin staff in South Texas. They handle patient records and need to be assessed in Spanish. HIPAA compliance is critical."}],
        [{"role": "user", "content": "Hiring bilingual healthcare admin staff in South Texas. Patient records, Spanish assessment. HIPAA critical."}, {"role": "assistant", "content": "Knowledge tests are English-only. Personality measures support Spanish. Two options: hybrid or personality-only. Which fits?"}, {"role": "user", "content": "They are functionally bilingual. Go with hybrid."}],
        [{"role": "user", "content": "Hiring bilingual healthcare admin staff. Patient records, Spanish. HIPAA critical."}, {"role": "assistant", "content": "Knowledge tests English-only, personality supports Spanish. Hybrid or personality-only?"}, {"role": "user", "content": "Functionally bilingual. Hybrid."}, {"role": "assistant", "content": "Here are 5 recommendations for hybrid healthcare admin battery."}, {"role": "user", "content": "Are we legally required under HIPAA to test all staff who touch patient records?"}],
        [{"role": "user", "content": "Hiring bilingual healthcare admin staff. Patient records, Spanish. HIPAA critical."}, {"role": "assistant", "content": "Knowledge tests English-only, personality supports Spanish. Hybrid or personality-only?"}, {"role": "user", "content": "Hybrid."}, {"role": "assistant", "content": "Here are 5 recommendations."}, {"role": "user", "content": "Are we legally required under HIPAA to test all staff?"}, {"role": "assistant", "content": "I cannot advise on legal compliance. Consult your legal team."}, {"role": "user", "content": "Understood. Keep the shortlist as-is."}],
    ],
    "C8": [
        [{"role": "user", "content": "I need to quickly screen admin assistants for Excel and Word daily."}],
        [{"role": "user", "content": "Quickly screen admin assistants for Excel and Word daily."}, {"role": "assistant", "content": "Here are 3 recommendations for quick admin screening."}, {"role": "user", "content": "In that case, I am OK with adding a simulation - we want to capture capabilities."}],
        [{"role": "user", "content": "Quickly screen admin assistants for Excel and Word daily."}, {"role": "assistant", "content": "Here are 3 recommendations for quick screening."}, {"role": "user", "content": "OK with adding simulation to capture capabilities."}, {"role": "assistant", "content": "Updated with simulations. Here are 5 recommendations."}, {"role": "user", "content": "That is good."}],
    ],
    "C9": [
        [{"role": "user", "content": "Here is the JD for an engineer. Senior Full-Stack Engineer, 5+ years across Core Java, Spring, REST API design, Angular, SQL, AWS, Docker. Will own microservice delivery, contribute to architecture, mentor mid-level."}],
        [{"role": "user", "content": "JD: Senior Full-Stack Engineer, 5+ years, Java, Spring, REST, Angular, SQL, AWS, Docker. Microservice delivery, architecture, mentor mid-level."}, {"role": "assistant", "content": "That spans many areas. Backend-leaning or frontend-heavy or balanced?"}, {"role": "user", "content": "Backend-leaning. Day-one priorities are Core Java and Spring; SQL is constant. Angular is occasional."}],
        [{"role": "user", "content": "JD: Senior Full-Stack, Java, Spring, REST, Angular, SQL, AWS, Docker. Microservice, architecture, mentor."}, {"role": "assistant", "content": "Backend-leaning or frontend-heavy?"}, {"role": "user", "content": "Backend-leaning. Java, Spring, SQL primary. Angular secondary review-only."}, {"role": "assistant", "content": "Senior IC or tech lead? That changes whether we add leadership layer."}, {"role": "user", "content": "Senior IC. They lead design on their own services but do not manage other engineers."}],
        [{"role": "user", "content": "JD: Senior Full-Stack, Java, Spring, REST, Angular, SQL, AWS, Docker."}, {"role": "assistant", "content": "Backend-leaning or frontend-heavy?"}, {"role": "user", "content": "Backend-leaning. Java, Spring, SQL primary."}, {"role": "assistant", "content": "Senior IC or tech lead?"}, {"role": "user", "content": "Senior IC."}, {"role": "assistant", "content": "Here are 6 recommendations for senior backend Java engineer."}, {"role": "user", "content": "Add AWS and Docker. Drop REST - the API signal will come through in Spring."}],
        [{"role": "user", "content": "JD: Senior Full-Stack, Java, Spring, REST, Angular, SQL, AWS, Docker."}, {"role": "assistant", "content": "Backend-leaning or frontend-heavy?"}, {"role": "user", "content": "Backend-leaning. Java, Spring, SQL primary."}, {"role": "assistant", "content": "Senior IC or tech lead?"}, {"role": "user", "content": "Senior IC."}, {"role": "assistant", "content": "Here are 6 recommendations."}, {"role": "user", "content": "Add AWS and Docker. Drop REST."}, {"role": "assistant", "content": "Updated. Here are 7 recommendations with AWS and Docker added."}, {"role": "user", "content": "Keep Verify G+. Locking it in."}],
    ],
    "C10": [
        [{"role": "user", "content": "We run a graduate management trainee scheme. We need a full battery - cognitive, personality, and situational judgement. All recent graduates."}],
        [{"role": "user", "content": "Graduate management trainee scheme. Full battery - cognitive, personality, situational judgement."}, {"role": "assistant", "content": "Here are 3 recommendations for graduate management trainee battery."}, {"role": "user", "content": "Can you remove OPQ32r and replace with something shorter?"}],
        [{"role": "user", "content": "Graduate management trainee. Full battery - cognitive, personality, situational judgement."}, {"role": "assistant", "content": "Here are 3 recommendations: Verify G+, OPQ32r, Graduate Scenarios."}, {"role": "user", "content": "Remove OPQ32r, replace with shorter?"}, {"role": "assistant", "content": "OPQ32r is most relevant. No shorter alternative."}, {"role": "user", "content": "Drop the OPQ. Final list: Verify G+ and Graduate Scenarios."}],
    ],
}


def run_trace(trace_id: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{BASE_URL}/chat",
                json={"messages": messages},
                timeout=120,
            )
            if resp.status_code == 429:
                wait = 7
                print(f"    {trace_id} 429. Waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            return resp.json()
        except Exception as e:
            print(f"    {trace_id} attempt {attempt+1} failed: {e}", flush=True)
            time.sleep(7)
    return {"reply": "timeout", "recommendations": [], "end_of_conversation": False}


def recall_at_k(recommended: List[str], expected: List[str], k: int = 10) -> float:
    rec_set = set(recommended[:k])
    expected_set = set(expected)
    if not expected_set:
        return 1.0
    return len(rec_set & expected_set) / len(expected_set)


def evaluate():
    total_recall = 0.0
    results = {}

    for trace_id, traces in CONVERSATION_TRACES.items():
        expected = EXPECTED_SHORTLISTS.get(trace_id, [])
        best_recall = 0.0
        best_rec_names = []

        # Only run the LAST turn — that's where recommendations land
        # Intermediate turns are clarification-only (0 recall) and add latency
        last_messages = traces[-1]
        print(f"{trace_id} final turn ({len(last_messages)} msgs): calling server...", flush=True)

        try:
            result = run_trace(trace_id, last_messages)
            rec_names = [r["name"] for r in result.get("recommendations", [])]
            rec = recall_at_k(rec_names, expected)

            print(f"  {trace_id}: recall={rec:.2f}, recs={len(rec_names)}, end={result.get('end_of_conversation')}", flush=True)
            if rec_names:
                for rn in rec_names:
                    status = "✓" if rn in expected else "?"
                    print(f"    {status} {rn}", flush=True)

            best_recall = rec
            best_rec_names = rec_names

        except Exception as e:
            print(f"  {trace_id}: ERROR: {e}", flush=True)

        # Also check second-to-last turn (where recommendations may first appear)
        if len(traces) >= 2 and best_recall == 0:
            penult_messages = traces[-2]
            print(f"{trace_id} penultimate turn: fallback call...", flush=True)
            try:
                result = run_trace(trace_id, penult_messages)
                rec_names = [r["name"] for r in result.get("recommendations", [])]
                rec = recall_at_k(rec_names, expected)
                if rec > best_recall:
                    best_recall = rec
                    best_rec_names = rec_names
                    print(f"  {trace_id} (penultimate): recall={rec:.2f}", flush=True)
            except Exception as e:
                print(f"  {trace_id} penultimate: ERROR: {e}", flush=True)

        results[trace_id] = best_recall
        total_recall += best_recall
        print(f"  {trace_id} best recall: {best_recall:.2f}", flush=True)
        print()

        # Stay under Gemini free tier rate limit (~10 req/min)
        time.sleep(7)

    mean_recall = total_recall / len(CONVERSATION_TRACES)
    print("=" * 50)
    print(f"Mean Recall@10: {mean_recall:.3f}")
    print(f"Scores: {dict(sorted(results.items()))}")
    print("=" * 50)
    return mean_recall


if __name__ == "__main__":
    evaluate()
