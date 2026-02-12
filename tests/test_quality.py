"""
Comprehensive RAG Quality Test - 20 Questions
Tests the quality of answers across department-related, tricky, and out-of-scope questions
"""
import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"

# Test Questions
QUESTIONS = {
    "Part 1: Department-Related Questions": [
        "I am interested in the Department of Electrical Engineering. Can you list all the programs they are offering this spring?",
        "I have completed my 16 years of education in Computer Science. Am I eligible to apply for the M.Sc. Data Science program?",
        "Who is the current Dean of the Faculty of Mechanical Engineering?",
        "Does the Department of Architecture offer a Ph.D. program, or is it only for Master's students?",
        "I am looking for the Institute of Environmental Engineering & Research. What are the specific eligibility criteria for their M.Phil. Environmental Sciences?",
        "Can you tell me which department offers the M.Sc. Mining Engineering program?",
        "I heard the Department of Petroleum & Gas Engineering is highly ranked. Is there any mention of their world ranking in the prospectus?",
        "I want to do an Executive MBA. Does the Institute of Business and Management require any professional experience for this degree?",
        "Is there any specific department that deals with Transportation Engineering?",
        "I am looking for the faculty list for the Department of Mathematics. Who is the Chairperson?"
    ],
    "Part 2: Tricky Questions": [
        "I want to apply for M.Sc. Artificial Intelligence. Should I select the Department of Computer Science on my form, or is this program offered by another department?",
        "I am interested in Geological Engineering. Is that the same as Geotechnical Engineering offered by the Civil Department, or is there a separate department for it?",
        "I live near the UET. Does the Department of Chemical Engineering offer any safety-related master's programs there?",
        "I see a program called M.Sc. Disaster Management and another called M.Sc. Disaster Mitigation Engineering. Which one is offered by the Civil Engineering department?",
        "I have a degree in Physics. Can I apply for M.Phil. Polymer Science & Technology, or is that only for chemical engineers?"
    ],
    "Part 3: Out-of-Scope Questions": [
        "What is the fee structure for the M.Sc. programs?",
        "Are there hostel facilities available for postgraduate students at the Lahore campus?",
        "What is the last date to submit my admission form for the Spring 2026 session?",
        "Does the university provide transport/bus service for students commuting from other cities?",
        "Is there an entry test required for the MS programs? If so, what is the passing score?"
    ]
}


def test_question(question: str, question_num: int, category: str):
    """Test a single question and return results"""
    print(f"\n{'='*80}")
    print(f"Q{question_num}: {question}")
    print('='*80)
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/answer",
            json={"question": question},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data['answer']
            sources = data['total_sources']
            
            print(f"\n💡 ANSWER:\n{answer}")
            print(f"\n📚 Sources Used: {sources}")
            
            # Show source departments
            if data['source_documents']:
                depts = set(doc['metadata']['department_name'] for doc in data['source_documents'])
                sections = set(doc['metadata']['section_type'] for doc in data['source_documents'])
                print(f"📍 Departments: {', '.join(depts)}")
                print(f"📑 Section Types: {', '.join(sections)}")
            
            # Quality check for out-of-scope
            if category == "Part 3: Out-of-Scope Questions":
                is_proper_response = any(phrase in answer.lower() for phrase in [
                    "don't have", "not available", "contact", "admission office"
                ])
                status = "✅ CORRECT" if is_proper_response else "⚠️ MAY HAVE HALLUCINATED"
                print(f"\nOut-of-Scope Handling: {status}")
            
            return {
                "question": question,
                "answer": answer,
                "sources": sources,
                "status": "success",
                "category": category
            }
        else:
            print(f"\n❌ ERROR: {response.status_code}")
            print(response.text)
            return {
                "question": question,
                "status": "error",
                "error": response.text,
                "category": category
            }
    
    except requests.exceptions.RequestException as e:
        print(f"\n❌ CONNECTION ERROR: {str(e)}")
        return {
            "question": question,
            "status": "connection_error",
            "error": str(e),
            "category": category
        }


def run_quality_test():
    """Run all 20 questions and generate report"""
    print("\n" + "🎯 "*40)
    print("RAG QUALITY TEST - 20 QUESTIONS")
    print("🎯 "*40)
    
    # Check if server is running
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            print("\n❌ ERROR: API server not responding properly")
            print("Start the server with: python run.py")
            return
        print(f"\n✅ Server Status: {health.json()['status']}")
        print(f"📊 Total Documents: {health.json()['total_documents']}")
    except:
        print("\n❌ ERROR: Cannot connect to API server")
        print("Start the server with: python run.py")
        return
    
    # Run all tests
    all_results = []
    question_num = 1
    
    for category, questions in QUESTIONS.items():
        print(f"\n\n{'#'*80}")
        print(f"# {category}")
        print(f"{'#'*80}")
        
        for question in questions:
            result = test_question(question, question_num, category)
            all_results.append(result)
            question_num += 1
    
    # Generate summary
    print("\n\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)
    
    successful = sum(1 for r in all_results if r['status'] == 'success')
    errors = sum(1 for r in all_results if r['status'] == 'error')
    
    print(f"\n✅ Successful: {successful}/20")
    print(f"❌ Errors: {errors}/20")
    
    # Category breakdown
    for category in QUESTIONS.keys():
        cat_results = [r for r in all_results if r['category'] == category]
        cat_success = sum(1 for r in cat_results if r['status'] == 'success')
        print(f"\n{category}: {cat_success}/{len(cat_results)} successful")
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_results_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Full results saved to: {filename}")
    
    # Final verdict
    print("\n" + "="*80)
    if successful == 20:
        print("🎉 PERFECT SCORE! All 20 questions answered successfully!")
    elif successful >= 18:
        print("🌟 EXCELLENT! Most questions answered correctly.")
    elif successful >= 15:
        print("👍 GOOD! Majority of questions handled well.")
    else:
        print("⚠️ NEEDS IMPROVEMENT. Review the failed questions.")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_quality_test()
