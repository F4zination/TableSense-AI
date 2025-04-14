from converter import Converter
from langchain_openai import OpenAI
import pathlib
from langchain_core.prompts import ChatPromptTemplate
from serialization.converter import FileFormat

if __name__ == "__main__":

    # initializing the Converter class
    converter = Converter()

    # initializing the LLM
    llm = OpenAI(
        model="/models/mistral-nemo-12b",
        temperature=0,
        max_retries=2,
        max_tokens=2048,
        base_url="http://80.151.131.52:9180/v1",
        api_key="THU-I17468S973-Student-24-25-94682Y1315",
    )

    converted_content = converter.convert(pathlib.Path("../input/minimal.csv"), FileFormat.HTML, False)

    prompt = ChatPromptTemplate.from_template("""
    You are a data scientist. You have been given a CSV file containing the following data:
    
    {data}
    
    Your task is to analyze the data and provide insights. Please summarize the key findings and any interesting patterns you observe.
    """)

    formated_prompt = prompt.format(data=converted_content)

    # Use the LLM to generate a response based on the prompt and converted content
    response = llm.invoke(formated_prompt)

    # Print the response
    with open("response.md", "w") as f:
        f.write(response)



