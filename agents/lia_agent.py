from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Initialize the Hugging Face model
model_id = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=100,
    device=-1  # CPU, since we don't know GPU availability
)

# Create a Hugging Face LLM using LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# Define a prompt template tailored for LinkedIn tasks for Datavalley
template = """
You are LIA (LinkedIn Intelligence Agent), an AI assistant created by Datavalley to assist with LinkedIn-related tasks. 
Your goal is to help users with professional tasks such as drafting posts, summarizing profiles, or answering LinkedIn-related queries.

**User Input**: {input}

**Response**:
"""
prompt = PromptTemplate(template=template, input_variables=["input"])

# Create the LIA agent chain
lia_chain = LLMChain(prompt=prompt, llm=llm)

def get_lia_response(user_input):
    try:
        response = lia_chain.run(input=user_input)
        return response.strip()
    except Exception as e:
        return f"Error processing request: {str(e)}"