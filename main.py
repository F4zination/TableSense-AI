from serialization.converter import TableFormat
from serialization.serialization_agent import SerializationAgent
import pathlib


def main():
    # Initialize the SerializationAgent with the required parameters
    agent = SerializationAgent(
        llm_model="/models/mistral-nemo-12b",
        temperature=0,
        max_retries=2,
        max_tokens=2048,
        base_url="http://80.151.131.52:9180/v1",
        api_key="THU-I17468S973-Student-24-25-94682Y1315",
        format_to_convert_to=TableFormat.HTML
    )

    # Example usage of the eval method
    question = "What is Peters Email?"
    dataset = pathlib.Path("input/minimal.csv")


    response = agent.eval(question, dataset, None)
    print(f"Response: {response}")


if __name__ == "__main__":
    agent = SerializationAgent(
        llm_model="/models/mistral-nemo-12b",
        temperature=0,
        max_retries=2,
        max_tokens=2048,
        base_url="http://80.151.131.52:9180/v1",
        api_key="THU-I17468S973-Student-24-25-94682Y1315",
        format_to_convert_to=TableFormat.HTML
    )

    NOT_WORKING = """You are a data scientist. You have been given the following data: 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>Name</th>
      <th>Birthday</th>
      <th>Email</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>John Smith</td>
      <td>31.3.2012</td>
      <td>john.smith@gmail.com</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>Marc Twain</td>
      <td>10.09.2000</td>
      <td>marc.twain@web.de</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>Hans Martin</td>
      <td>03.4.3029</td>
      <td>hans2.martin@future.da</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>Peter Fox</td>
      <td>02.9.1903</td>
      <td>peter.fox@fox.de</td>
    </tr>
  </tbody>
</table>.
What is Peters Email?"""

    WORKING = "Tell me a joke!"

    test = agent.invoke(NOT_WORKING)

    print(f"Test: {test}")
