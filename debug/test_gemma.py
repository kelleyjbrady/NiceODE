import requests
import os

# Get the URL from the environment variable set in docker-compose
GEMMA_URL = os.getenv("GEMMA_API_URL", "http://localhost:8000")

def query_gemma(prompt: str):
    """Sends a prompt to the secure Gemma LLM service."""
    try:
        # First, check the health of the service
        health_response = requests.get(f"{GEMMA_URL}/health")
        health_response.raise_for_status()
        health_data = health_response.json()
        
        if health_data.get("status") != "ok":
            print(f"Service is not healthy: {health_data.get('detail')}")
            return None

        print(f"Gemma service is healthy. Model: {health_data.get('model_name')}. Querying...")
        
        # Now, send the prompt
        response = requests.post(f"{GEMMA_URL}/generate", json={"text": prompt})
        response.raise_for_status() # Raises an exception for 4XX/5XX errors
        
        return response.json()["generated_text"]

    except requests.exceptions.RequestException as e:
        print(f"An error occurred while communicating with the Gemma service: {e}")
        return None

if __name__ == "__main__":
    test_prompt = "Explain the difference between a Docker image and a container in simple terms."
    result = query_gemma(test_prompt)
    if result:
        print("\n--- Gemma's Response ---")
        print(result)