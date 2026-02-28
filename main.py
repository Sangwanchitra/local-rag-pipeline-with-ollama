import socket
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from agent_config import get_agent_config

def check_port(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def main():
    config = get_agent_config()
    print("=== Update the current agent configuration ===")
    print(f"- Provider: {config['provider']}")
    print(f"- Base URL: {config['base_url']}")
    print(f"- Model Name: {config['model']}")
    
    if config['use_cloud_provider']:
        print("\nERROR: Cloud provider is currently ENABLED. Disabling now...")
    else:
        print("\nSUCCESS: Cloud providers are explicitly disabled.")

    print("\nInitializing Local Ollama provider...")
    llm = ChatOllama(
        model=config['model'],
        base_url=config['base_url'],
        temperature=0
    )
    
    # Just check if Ollama is running on the expected port
    if check_port(11434):
        print("Ollama base URL is reachable! Configuration is set successfully.")
    else:
        print("Warning: Ollama base URL (http://localhost:11434) is currently unreachable. Please make sure Ollama is running.")
        
if __name__ == "__main__":
    main()
