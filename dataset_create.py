from dotenv import load_dotenv
load_dotenv()
import os
# === ENVIRONMENT SETUP ===
os.getenv("OPENAI_API_KEY")
os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "dataset_creation_rag"

from langsmith import Client

client = Client()
dataset = client.create_dataset("LLM Debugging Q&A", description="Common debugging questions and answers for LLM, LangChain, and LangSmith development.")

example_inputs_outputs = [
    (
        "Why is my LangChain agent stuck and not responding?",
        "This can happen if your tool is misconfigured or if a tool call hangs. Check if the tool is async, and ensure the `handle_tool_error` parameter is set to True if you're expecting failures."
    ),
    (
        "How do I fix the 'input_variables mismatch' error in LangChain?",
        "This error occurs when the PromptTemplate's input_variables do not match the keys you’re passing. Double-check the variable names in your prompt and those you provide in `.format()` or `.invoke()`."
    ),
    (
        "LangSmith is not capturing traces from my chain. What should I do?",
        "Ensure that the environment variable `LANGSMITH_TRACING` is set to 'true' and that `LANGSMITH_API_KEY` is correctly configured. Also, make sure you're using supported LangChain components like LCEL or instrumented chains."
    ),
    (
        "What does 'openai.error.InvalidRequestError: This model's maximum context length is 4097 tokens' mean?",
        "It means you're sending too much input or history for the model to handle. Reduce the prompt size or truncate messages to stay within the model’s token limit."
    ),
    (
        "Why are my OpenAI function calls not invoking the correct tool?",
        "Ensure the function name in your OpenAI function specification exactly matches the registered tool's name. Also, confirm the tool schema matches what OpenAI expects."
    ),
    (
        "How do I resolve 'MissingTemplateError' in LangChain?",
        "This occurs when you define a template with variables but forget to pass them during formatting. Check your `PromptTemplate` definition and how you're invoking it."
    ),
    (
        "My dataset upload to LangSmith failed. Why?",
        "This might happen due to malformed inputs/outputs or exceeding size limits. Ensure that your input/output pairs are dictionaries and not plain strings, and check for invalid characters."
    ),
    (
        "Why is my LangChain Runnable.map throwing 'NoneType' errors?",
        "It’s likely that one of your steps returns `None`. Add validation after each runnable step and handle missing or null values gracefully."
    ),
    (
        "I'm using memory in LangChain, but the chatbot doesn't remember past messages. Why?",
        "You need to use a memory class like `ConversationBufferMemory` and pass it into your chain or agent. Also make sure `memory_key` matches your prompt input variable."
    ),
    (
        "How can I debug a LangChain tool that’s not being called?",
        "Check that the tool is correctly passed to the agent constructor, has a valid `name`, and that your prompt allows the agent to decide when to use it. Also enable verbose mode for logs."
    ),
]

# Upload the dataset
for question, answer in example_inputs_outputs:
    client.create_example(
        inputs={"question": question},
        outputs={"answer": answer},
        dataset_id=dataset.id
    )

print("Dataset uploaded to LangSmith.")
