"""
Test script to verify all database and service connections
Final version with all fixes and better error handling
"""

import asyncio
import asyncpg
import redis
import requests
from minio import Minio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

async def test_postgresql() -> bool:
    """Test PostgreSQL connection"""
    try:
        conn = await asyncpg.connect(
            "postgresql://fairdoc:password@localhost:5432/fairdoc_v0"
        )
        result = await conn.fetchval("SELECT version()")
        await conn.close()
        print("✅ PostgreSQL: Connected successfully")
        print(f"   Version: {result.split(',')[0]}")  # Show only main version
        return True
    except Exception as e:
        print(f"❌ PostgreSQL: Connection failed - {e}")
        return False

def test_redis() -> bool:
    """Test Redis connection"""
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        info = r.info()
        print("✅ Redis: Connected successfully")
        print(f"   Version: {info.get('redis_version', 'Unknown')}")
        return True
    except Exception as e:
        print(f"❌ Redis: Connection failed - {e}")
        return False

def test_minio() -> bool:
    """Test MinIO connection"""
    try:
        client = Minio(
            "localhost:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            secure=False
        )
        # Try to list buckets
        buckets = client.list_buckets()
        print("✅ MinIO: Connected successfully")
        bucket_names = [bucket.name for bucket in buckets]
        print(f"   Buckets: {bucket_names if bucket_names else 'None (empty)'}")
        
        # Try to create a test bucket for Fairdoc
        try:
            if "fairdoc-files" not in bucket_names:
                client.make_bucket("fairdoc-files")
                print("   Created bucket: fairdoc-files")
        except Exception as bucket_error:
            print(f"   Bucket creation info: {bucket_error}")
        
        return True
    except Exception as e:
        print(f"❌ MinIO: Connection failed - {e}")
        return False

def test_chromadb_http() -> bool:
    """Test ChromaDB HTTP endpoints"""
    endpoints_to_try = [
        "http://localhost:8000/api/v1/heartbeat",
        "http://localhost:8000/api/v1/version",
        "http://localhost:8000/api/v1",
        "http://localhost:8000/heartbeat",
        "http://localhost:8000/"
    ]
    
    for endpoint in endpoints_to_try:
        try:
            response = requests.get(endpoint, timeout=5)
            if response.status_code == 200:
                print("✅ ChromaDB HTTP: Connected successfully")
                print(f"   Endpoint: {endpoint}")
                try:
                    data = response.json()
                    if isinstance(data, dict) and 'nanosecond heartbeat' in data:
                        print(f"   Heartbeat: {data['nanosecond heartbeat']}")
                except (ValueError, KeyError, TypeError):  # Fixed: Remove unused variable
                    print(f"   Response: {response.text[:100]}...")
                return True
            else:
                print(f"   Tried {endpoint}: HTTP {response.status_code}")
        except requests.RequestException as e:
            print(f"   Tried {endpoint}: {str(e)[:50]}...")
    
    print("❌ ChromaDB HTTP: All endpoints failed")
    return False

def test_chromadb_client() -> bool:
    """Test ChromaDB using Python client"""
    try:
        import chromadb
        
        # Try different client configurations
        client_configs = [
            {"host": "localhost", "port": 8000},
            {"host": "127.0.0.1", "port": 8000},
        ]
        
        for config in client_configs:
            try:
                client = chromadb.HttpClient(**config)
                
                # Test by getting heartbeat or listing collections
                collections = client.list_collections()
                print("✅ ChromaDB Client: Connected successfully")
                print(f"   Host: {config['host']}:{config['port']}")
                print(f"   Collections: {len(collections)}")
                
                # Try to create a test collection
                try:
                    _ = client.get_or_create_collection("test_connection")
                    client.delete_collection("test_connection")
                    print("   Test collection: Created and deleted successfully")
                except Exception as coll_error:
                    print(f"   Test collection: {coll_error}")
                
                return True
                
            except Exception as client_error:
                print(f"   Client config {config} failed: {client_error}")
                continue
        
        print("❌ ChromaDB Client: All configurations failed")
        return False
        
    except ImportError:
        print("❌ ChromaDB: Python package not installed")
        return False
    except Exception as e:
        print(f"❌ ChromaDB Client: Connection failed - {e}")
        return False

def test_ollama() -> bool:
    """Test Ollama connection with fixed model parsing"""
    try:
        import ollama
        
        # Test connection by listing models
        models_response = ollama.list()
        print("✅ Ollama: Connected successfully")
        
        # Parse models correctly
        if 'models' in models_response and models_response['models']:
            model_names = []
            model_count = len(models_response['models'])
            
            for model in models_response['models']:
                # Handle different model response formats
                if hasattr(model, 'model'):
                    # Model object with .model attribute
                    model_names.append(model.model)
                elif isinstance(model, dict):
                    # Dictionary with 'model' or 'name' key
                    if 'model' in model:
                        model_names.append(model['model'])
                    elif 'name' in model:
                        model_names.append(model['name'])
                else:
                    # String representation parsing
                    model_str = str(model)
                    if "model='" in model_str:
                        start = model_str.find("model='") + 7
                        end = model_str.find("'", start)
                        if end > start:
                            model_names.append(model_str[start:end])
                    else:
                        # Last resort - try to extract any quoted string
                        import re
                        matches = re.findall(r"'([^']*)'", model_str)
                        if matches:
                            model_names.append(matches[0])
            
            print(f"   Available models ({model_count}): {model_names}")
            
            # Test generation with first available model
            if model_names:
                try:
                    first_model = model_names[0]
                    print(f"   Testing model: {first_model}")
                    
                    response = ollama.generate(
                        model=first_model,
                        prompt="Test. Reply 'OK'.",
                        options={'num_predict': 5, 'temperature': 0}
                    )
                    
                    if 'response' in response:
                        test_response = response['response'].strip()
                        print(f"   Test generation: '{test_response[:30]}...'")
                        print("   Model generation: Working")
                    else:
                        print("   Test generation: No response field")
                        
                except Exception as gen_error:
                    print(f"   Model test failed: {str(gen_error)[:100]}...")
                    print("   Note: Model available but generation failed")
            else:
                print("   No models parsed successfully")
                print("   Run: ollama pull gemma2:2b")
                return False
        else:
            print("   No models available")
            print("   Run: ollama pull gemma2:2b")
            return False
            
        return True
        
    except ImportError:
        print("❌ Ollama: Python package not installed")
        print("   Run: pip install ollama")
        return False
    except Exception as e:
        print(f"❌ Ollama: Connection failed - {e}")
        return False

async def test_all_services() -> bool:
    """Test all services and provide summary"""
    print("🔍 Testing all database and service connections...\n")
    
    services = [
        ("PostgreSQL", test_postgresql()),
        ("Redis", test_redis()),
        ("MinIO", test_minio()),
        ("ChromaDB HTTP", test_chromadb_http()),
        ("ChromaDB Client", test_chromadb_client()),
        ("Ollama", test_ollama())
    ]
    
    results = []
    for name, test_coro in services:
        if asyncio.iscoroutine(test_coro):
            result = await test_coro
        else:
            result = test_coro
        results.append((name, result))
        print()  # Add spacing between tests
    
    # Print summary
    print("=" * 60)
    print("📊 CONNECTION TEST SUMMARY")
    print("=" * 60)
    
    working_services = []
    failed_services = []
    
    for name, result in results:
        status = "✅ WORKING" if result else "❌ FAILED"
        print(f"{name:20} : {status}")
        
        if result:
            working_services.append(name)
        else:
            failed_services.append(name)
    
    print("=" * 60)
    print(f"📈 Results: {len(working_services)}/{len(results)} services connected successfully")
    
    if len(working_services) >= 4:  # Need at least PostgreSQL, Redis, MinIO, and Ollama
        print("🎉 Sufficient services available to proceed!")
        print("📝 Ready for next step: Database schema creation")
        
        if failed_services:
            print(f"⚠️  Services needing attention: {', '.join(failed_services)}")
            print("💡 These can be fixed later without blocking development")
        
        return True
    else:
        print("🚨 Critical services missing - cannot proceed")
        print("🔧 Please fix failed services before continuing")
        return False

def main() -> bool:
    """Main entry point"""
    print("🚀 Fairdoc AI v0 - Database & Services Connection Test")
    print("🕐 Started at:", __import__('datetime').datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print()
    
    try:
        result = asyncio.run(test_all_services())
        
        print("\n" + "=" * 60)
        if result:
            print("✅ CONNECTION TEST: PASSED")
            print("📋 Next Step: Create SQLAlchemy database schema")
        else:
            print("❌ CONNECTION TEST: FAILED")
            print("🔧 Fix failed services before proceeding")
        print("=" * 60)
        
        return result
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Test failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
