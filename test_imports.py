#!/usr/bin/env python3
"""
Quick test script to verify all imports work correctly.
"""

def test_imports():
    """Test that all required packages can be imported."""
    print("Testing imports...")
    
    try:
        import fastapi
        print("✓ FastAPI imported successfully")
    except ImportError as e:
        print(f"✗ FastAPI import failed: {e}")
        return False
    
    try:
        import llama_index
        print("✓ LlamaIndex imported successfully")
    except ImportError as e:
        print(f"✗ LlamaIndex import failed: {e}")
        return False
    
    try:
        import faiss
        print("✓ FAISS imported successfully")
    except ImportError as e:
        print(f"✗ FAISS import failed: {e}")
        return False
    
    try:
        import psycopg2
        print("✓ psycopg2 imported successfully")
    except ImportError as e:
        print(f"✗ psycopg2 import failed: {e}")
        return False
    
    try:
        import openai
        print("✓ OpenAI imported successfully")
    except ImportError as e:
        print(f"✗ OpenAI import failed: {e}")
        return False
    
    try:
        import pydantic
        print("✓ Pydantic imported successfully")
    except ImportError as e:
        print(f"✗ Pydantic import failed: {e}")
        return False
    
    print("\n✓ All imports successful!")
    return True


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        import config
        print(f"✓ Config loaded")
        print(f"  - DB_NAME: {config.DB_NAME}")
        print(f"  - FAISS_INDEX_PATH: {config.FAISS_INDEX_PATH}")
        print(f"  - LLM_MODEL: {config.LLM_MODEL}")
        
        if config.OPENAI_API_KEY and config.OPENAI_API_KEY != "your_openai_api_key_here":
            print("  - OpenAI API key: configured ✓")
        else:
            print("  - OpenAI API key: NOT configured (please set in .env)")
        
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_models():
    """Test Pydantic models."""
    print("\nTesting Pydantic models...")
    
    try:
        from models import SentenceResponse, ChatRequest, InitResponse, LLMResponse
        
        # Test SentenceResponse
        response = SentenceResponse(
            sentence_ids=[1, 2, 3],
            sentences=["Test 1", "Test 2", "Test 3"]
        )
        print(f"✓ SentenceResponse model works")
        
        # Test ChatRequest
        request = ChatRequest(question="Jak się masz?")
        print(f"✓ ChatRequest model works")
        
        # Test InitResponse
        init_resp = InitResponse(message="Success", sentences_indexed=10)
        print(f"✓ InitResponse model works")
        
        # Test LLMResponse
        llm_resp = LLMResponse(sentence_ids=[1, 2])
        print(f"✓ LLMResponse model works")
        
        return True
    except Exception as e:
        print(f"✗ Model testing failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("RAG Training - Installation Verification")
    print("=" * 60)
    
    all_pass = True
    all_pass &= test_imports()
    all_pass &= test_config()
    all_pass &= test_models()
    
    print("\n" + "=" * 60)
    if all_pass:
        print("✓ All tests passed! Installation is successful.")
        print("\nNext steps:")
        print("1. Ensure PostgreSQL is running")
        print("2. Initialize database: psql -U postgres -f init_db.sql")
        print("3. Start server: uvicorn main:app --reload")
        print("4. Initialize FAISS: curl -X POST http://localhost:8000/init")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 60)



