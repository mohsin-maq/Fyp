import matplotlib.pyplot as plt
import warnings
import streamlit as st
import os
import tempfile
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.llms import OpenAI
import nbformat
st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings("ignore")
import seaborn as sns

# Set font family and style explicitly
plt.rcParams['font.family'] = 'sans-serif'  # Change to the desired font family
plt.rcParams['font.sans-serif'] = ['Arial']  # Change to the desired font style



os.environ['OPENAI_API_KEY'] = "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"


custom = '''
You are working with a pandas dataframe in Python. The name of the dataframe is df. You should use the tools below to answer the question posed to you:
python_repl_ast: A Python shell. Use this to execute Python commands. Input should be a valid Python command. When using this tool, sometimes output is abbreviated - make sure it does not look abbreviated before using it in your answer.

generate_plot: Generate a data visualization plot using Matplotlib and Seaborn. Input should include the Python code for creating the plot.

apply_ml_algorithm: Apply machine learning algorithms for model development. Input should include the Python code or function call for applying the algorithm.

Use the following format:


Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [python_repl_ast, generate_plot, apply_ml_algorithm]
Action Input: the input to the action (import all the libraries needed in one go, for example if i user has asked about applying train test split then import all the neccecary libraries in one response , likewise if user has asked for any classfication algorithm , import all neccessary libraries in one response , so that api is not called more than 1 time for 1 question)
Observation: the result of the action
Thought: I know the final answer
Final Answer: the final answer to the original input question

This is the result of print(df.head()): {df}

Begin!


check on the 'Question' 
if Question :
Question: Create a bar chart showing the distribution of 'column_name' in df.
then:
    Thought: I need to create a bar chart.
    Action: generate_plot
    Action Input: 
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots()
    ax.bar(categories, values)
    ax.set_xlabel('Categories')
    ax.set_ylabel('Values')
    ax.set_title('Bar Plot Example')

    # Display the bar plot in Streamlit
    st.pyplot(fig)
    """
    Observation: (No need for observation here)
    Thought: The chart has been created.
    Final Answer: Bar chart created successfully.

elif :
Question: Apply train-test split on the dataset.
then:
    Thought: I need to split the dataset for model development.
    Action: apply_ml_algorithm
    Action Input: 
    """
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    """
    Observation: (No need for observation here)
    Thought: The dataset has been split.
    Final Answer: Train-test split applied successfully.

elif :

Question : Apply classfication algorithm on this dataset 

then :
    Thought: I need to split the dataset for model development.
    Action: apply_ml_algorithm
    Action Input: 
    """
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a logistic regression model
    model = LogisticRegression()

    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)

    # Print the results
    print(f"Accuracy: {accuracy:.2f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_rep)

    """
    Observation: (No need for observation here)
    Thought: Model has been applied.
    Final Answer: Model has been applied.


Note: Try to use Seaborn for data visualization, and use scikit-learn for machine learning algorithms.
'''

@st.cache_resource
def extract_text_from_csv(csv_file, custom):
    # Assuming the OpenAI class supports specifying the model
    agent = create_csv_agent(OpenAI(temperature=0,model="gpt-3.5-turbo-instruct"),
                         csv_file,
                         template=custom,
                         verbose=True,
                         return_intermediate_steps=True)

    # st.write(agent)
    return agent

def write_jupyter(code,notebook_path='output_notebook.ipynb'):
    try:
        with open(notebook_path, 'r') as notebook_file:
            notebook_content = nbformat.read(notebook_file, as_version=4)
    except FileNotFoundError:
        notebook_content = nbformat.v4.new_notebook()

    new_cell = nbformat.v4.new_code_cell(source=code)
    notebook_content['cells'].append(new_cell)

    with open(notebook_path, 'w') as notebook_file:
        nbformat.write(notebook_content, notebook_file)
    
# Initialize Streamlit page
st.set_page_config(page_title="LLM-Data Inquiry")
st.header("LLM-Data Inquiry 📈 👨‍🔬")

# Define a list of accepted file types
accepted_file_types = ["csv"]

# Allow multiple file uploads
docs = st.sidebar.file_uploader(
    "Upload your documents", type=accepted_file_types, accept_multiple_files=True)

# Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you?"}
    ]
# Get user input and save it
if prompt := st.chat_input():
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
# Display the existing chat messages

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# Function to process user input and generate responses

# Process user input
if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            chat_history = []

            for doc in docs:
                agent = extract_text_from_csv(doc, custom)
                if agent:
                    result = agent(prompt)
                    answer=result['output']
                    
                    code=""
                    for i in range(len(result['intermediate_steps'])):
                        response=result['intermediate_steps'][i][0].tool_input
                        code+=response
                        code+="\n"
                        
                    write_jupyter(code)
                    
                
                    if 'chart' in answer or 'plot' in answer or 'visulization'  in answer or 'graph' in answer:
                        # plt.figure(figsize=(4, 4))
                        plot = st.pyplot()

                    st.write(response)
                    
                    assistant_message = {
                        "role": "assistant", "content": answer}
                    chat_history.append((prompt, answer))
                    st.session_state.messages.append(assistant_message)
