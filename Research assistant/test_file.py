from llm import llm

def main():
    response = llm([{"role": "user", "content": "What is 2 + 2?"}])
    print(response)

if __name__ == "__main__":
    main()