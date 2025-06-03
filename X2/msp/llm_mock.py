import asyncio
import logging
import utils.path_setup  # Ensures paths are set up

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

try:
    from ollama import AsyncClient
    OLLAMA_AVAILABLE = True
    print("Ollama client imported successfully.")
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Ollama client not found. Chatbot will use mock responses.")

async def get_ollama_response(prompt: str, history: list = None) -> str:
    if not OLLAMA_AVAILABLE:
        await asyncio.sleep(1)  # Simulate network delay
        return "I am a mock assistant. Ollama is not available. How else can I pretend to help?"

    client = AsyncClient()  # Consider making this a global or passed-in client
    messages = []
    if history:
        for msg in history:
            # Assuming msg is a dict with 'role' and 'content'
            messages.append({'role': msg['role'], 'content': msg['content']}) 
    messages.append({'role': 'user', 'content': prompt})

    try:
        logger.info(f"Sending to Ollama (gemma:4b): {prompt}")
        response = await client.chat(
            model='gemma:4b',  # Or your specific model name
            messages=messages,
            stream=False  # For simpler handling initially
        )
        assistant_response = response['message']['content']
        logger.info(f"Received from Ollama: {assistant_response}")
        return assistant_response
    except Exception as e:
        logger.error(f"Error communicating with Ollama: {e}")
        return "I'm having trouble connecting to my brain right now. Please try again later."

if __name__ == "__main__":
    # Test function
    async def main_test():
        test_prompt = "What are common symptoms of the flu?"
        print(f"Testing Ollama with prompt: {test_prompt}")
        response = await get_ollama_response(test_prompt)
        print(f"Ollama Test Response: {response}")
    asyncio.run(main_test())
