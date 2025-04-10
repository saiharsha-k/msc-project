from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import pathlib

load_dotenv()

# Singleton instance of LIAAgent
_lia_agent = None

class LIAAgent:
    def __init__(self):
        # Load Gemini API key from .env file
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")

        # Initialize the Gemini model via LangChain
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=self.api_key,
            temperature=0.7,
            max_output_tokens=150
        )

        # Define a custom prompt template
        prompt_template = """You are LIA (LinkedIn Intelligence Agent), a friendly and professional AI assistant designed to help users with a wide range of LinkedIn-related tasks, such as job searching, networking, profile optimization, content creation, analytics, and scheduling. Respond in a natural, conversational tone, focusing on the user's query. Do not include your role description, conversation history, or any metadata in your response. Use the conversation history for context only.

Conversation history for context:
{history}

User: {input}

LIA: """

        # Create a PromptTemplate
        self.prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=prompt_template
        )

        # Initialize conversation memory
        self.memory = ConversationBufferMemory()

        # Set up the LLM chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            verbose=False  # Set to True for debugging
        )

    def chat(self, user_message):
        """
        Process a user message, query the Gemini model via LangChain, and return the response.
        Maintains conversation history using LangChain's memory.
        """
        try:
            # Save the user message to memory
            self.memory.save_context({"input": user_message}, {"output": ""})

            # Get the conversation history
            history = self.memory.load_memory_variables({})["history"]

            # Generate the response using the LLM chain
            response = self.chain.run(history=history, input=user_message)

            # Clean up the response to remove any unwanted metadata or prompt artifacts
            response = response.strip()

            # Remove the "LIA: " prefix if present
            if response.startswith("LIA:"):
                response = response[len("LIA:"):].strip()

            # Remove any remaining prompt artifacts
            if "Conversation history for context:" in response:
                response = response.split("LIA:")[-1].strip() if "LIA:" in response else response
            if "User:" in response:
                response = response.split("User:")[0].strip()
            if "**" in response:
                response = response.split("**")[-1].strip()

            # Save the response to memory
            self.memory.save_context({"input": user_message}, {"output": response})

            return response
        except Exception as e:
            return f"Error: Failed to query the Gemini model - {str(e)}"

# Function to get the singleton instance of LIAAgent and call its chat method
def get_lia_response(user_message):
    global _lia_agent
    if _lia_agent is None:
        _lia_agent = LIAAgent()
    return _lia_agent.chat(user_message)

# Example usage (for testing)
if __name__ == "__main__":
    print("Starting conversation with LIA...")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = get_lia_response(user_input)
        print(f"LIA: {response}")