import asyncio
import logging
import utils.path_setup

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')

try:
    from ollama import AsyncClient
    OLLAMA_AVAILABLE = True
    logger.info("Ollama client imported successfully.")
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama client not found. Chatbot will use mock responses.")

async def get_ollama_response(prompt: str, history: list = None) -> str:
    if not OLLAMA_AVAILABLE:
        await asyncio.sleep(0.5) # Simulate network delay
        mock_response = f"Mock response to: '{prompt[:50]}...'. (Ollama unavailable)"
        logger.info(f"Using mock response: {mock_response}")
        return mock_response

    client = AsyncClient() 
    messages = []
    if history: # history is expected to be list of dicts like {'role': 'user', 'content': 'Hi'}
        for msg_dict in history:
            if isinstance(msg_dict, dict) and "role" in msg_dict and "content" in msg_dict:
                 messages.append({'role': msg_dict['role'], 'content': msg_dict['content']})
            else:
                logger.warning(f"Skipping malformed history item: {msg_dict}")
                 
    messages.append({'role': 'user', 'content': prompt})

    try:
        logger.info(f"Sending to Ollama (gemma:4b) - Prompt: '{prompt}' - History items: {len(messages)-1}")
        response = await client.chat(
            model='gemma:4b', 
            messages=messages,
            stream=False 
        )
        assistant_response = response['message']['content']
        logger.info(f"Received from Ollama: '{assistant_response[:100]}...'")
        return assistant_response
    except Exception as e:
        logger.error(f"Error communicating with Ollama: {e}", exc_info=True)
        return "I'm having trouble connecting to my AI brain right now. Please try again in a moment."

if __name__ == "__main__":
    async def main_test():
        test_prompt = "What are common symptoms of the flu?"
        print(f"Testing Ollama with prompt: {test_prompt}")
        response = await get_ollama_response(test_prompt)
        print(f"Ollama Test Response: {response}")
    asyncio.run(main_test())