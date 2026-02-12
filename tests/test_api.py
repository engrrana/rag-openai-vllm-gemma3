"""
Basic API endpoint tests
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("TEST: Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200


def test_retrieve():
    """Test retrieve endpoint"""
    print("\n" + "="*60)
    print("TEST: Retrieve Endpoint")
    print("="*60)
    
    question = "What programs does Computer Science offer?"
    response = requests.post(
        f"{BASE_URL}/api/v1/retrieve",
        json={"question": question}
    )
    
    print(f"Question: {question}")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Documents Retrieved: {data['total_documents']}")
        print(f"Method: {data['retrieval_method']}")
        
        if data['retrieved_documents']:
            doc = data['retrieved_documents'][0]
            print(f"\nFirst Document:")
            print(f"  Source: {doc['metadata']['full_context']}")
            print(f"  Content: {doc['content'][:150]}...")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def test_answer():
    """Test answer endpoint"""
    print("\n" + "="*60)
    print("TEST: Answer Endpoint")
    print("="*60)
    
    question = "Who is the chairman of Electrical Engineering?"
    response = requests.post(
        f"{BASE_URL}/api/v1/answer",
        json={"question": question}
    )
    
    print(f"Question: {question}")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nAnswer: {data['answer']}")
        print(f"Sources: {data['total_sources']}")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200


def run_tests():
    """Run all basic tests"""
    print("\n" + "🧪 "*30)
    print("BASIC API TESTS")
    print("🧪 "*30)
    
    try:
        results = {
            "Health Check": test_health(),
            "Retrieve Endpoint": test_retrieve(),
            "Answer Endpoint": test_answer()
        }
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        
        for test, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test}: {status}")
        
        total = sum(results.values())
        print(f"\nTotal: {total}/{len(results)} passed")
        
        if total == len(results):
            print("\n🎉 All tests passed!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to server")
        print("Start server with: python run.py")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")


if __name__ == "__main__":
    run_tests()
