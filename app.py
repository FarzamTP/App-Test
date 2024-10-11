import streamlit as st

# Set page configuration must be called at the very top, before any other Streamlit commands
st.set_page_config(
    page_title="Lessons Learned Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

# Now import other modules and define functions
import time
import json
from datetime import datetime
from conf import ASSISTANT_ENDPOINT, ASSISTANT_API_KEY
from SearchUtils import create_suggestions_list
from openai import AzureOpenAI
import os
from io import BytesIO
import pandas as pd

# Set up the OpenAI client
client = AzureOpenAI(
    azure_endpoint=ASSISTANT_ENDPOINT,
    api_key=ASSISTANT_API_KEY,
    api_version="2024-05-01-preview",
)

# Function to find an existing assistant by name
def find_assistant_by_name(client, name):
    assistants = client.beta.assistants.list()
    for assistant in assistants:
        if assistant.name == name:
            return assistant
    return None


# Define the assistant name
assistant_name = 'LL Assistant'

# Check if the assistant already exists
existing_assistant = find_assistant_by_name(client, assistant_name)

if existing_assistant:
    assistant = existing_assistant
else:
    # Create a new assistant if it doesn't exist
    assistant = client.beta.assistants.create(
        model="GPT4O-mini",
        name=assistant_name,
        instructions=(
            "You are an intelligent assistant designed to help me retrieve information regarding Lessons Learned (LL)."
            " I have provided you with an Excel file that contains a comprehensive list of Lessons Learned."
            " When I ask you explicitly to find relevant Lessons Learned based on specific keywords, follow these steps:"
            "1. Extract the keywords from my query. The keywords are comma separated."
            "2. Confirm all extracted keywords with me to everytime."
            "3. Upon my confirmation, use the extended_search function to find the relevant Lessons Learned."
            "4. Present the results in a clear and organized tabular format for easy review, and send me this message: The file has been generated and is accessible from left side-bar under **```📂 Generated Excel Files```** section."
            "Your goal is to make the process of finding and reviewing Lessons Learned as efficient and accurate as possible."
        ),
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "extended_search",
                    "description": "Give me relevant Lessons Learned",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "A string of keywords, separated by commas",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            {"type": "code_interpreter"},
        ],
        tool_resources={
            "code_interpreter":{"file_ids": ["assistant-lkJlJudSXBnUIts7qneq65dh"]}
        },
        temperature=1,
        top_p=1,
    )

def clean_azure_from_temp_excel_files():
    for file in client.files.list():
        if file.filename.startswith('/mnt') and file.filename.split('/')[-1] in os.listdir(st.session_state['thread_docs_dir']):
            client.files.delete(file.id)

def download_excel_file_and_return_path(file_id, file_name):
    if not os.path.exists(st.session_state['thread_docs_dir']):
        os.mkdir(st.session_state['thread_docs_dir'])
    file_path = os.path.join(st.session_state['thread_docs_dir'], file_name)
    client.files.content(file_id).write_to_file(file_path)
    st.session_state['LAST_FILE_CREATED_ID'] = file_id

def save_search_results_to_file(df, keywords):
    if not os.path.exists(st.session_state['thread_docs_dir']):
        os.mkdir(st.session_state['thread_docs_dir'])
    file_name = f'relevant LLs for {keywords.replace(",", "-")}.xlsx'
    file_path = os.path.join(st.session_state['thread_docs_dir'], file_name)
    df.to_excel(file_path, index=False)
    file = client.files.create(
        file=open(file_path, "rb"),
        purpose='assistants'
    )
    st.session_state['LAST_FILE_CREATED_ID'] = file.id


# Function to get output from tool calls
def get_output_from_toolcall(tool_call):
    query = json.loads(tool_call.function.arguments)["query"]
    results_df = create_suggestions_list(query)
    if results_df is None or results_df.empty:
        return {
            "tool_call_id": tool_call.id,
            "output": "No results were found based on the given keywords."
        }
    else:
        save_search_results_to_file(results_df, query)
        results_list = results_df.to_dict(orient='records')
        results_json = json.dumps(results_list)
        return {
            "tool_call_id": tool_call.id,
            "output": results_json
        }


# Function to convert DataFrame to Excel and return as bytes
def to_excel(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Suggestions')
    processed_data = output.getvalue()
    return processed_data


def main():
    # Sidebar content
    if 'files' not in st.session_state:
        st.session_state['files'] = []

    if 'LAST_FILE_CREATED_ID' not in st.session_state:
        st.session_state['LAST_FILE_CREATED_ID'] = None

    st.sidebar.title("Settings")
    st.sidebar.write("Use the options below to manage your session.")

    if st.sidebar.button("🗑️ Start New Conversation"):
        # Clean temp excel files from Azure
        clean_azure_from_temp_excel_files()
        # Reset session state and create a new thread
        st.session_state['messages'] = []
        # Create a new thread and update the URL
        thread = client.beta.threads.create()
        st.query_params.thread_id = thread.id
        st.session_state['thread_id'] = thread.id
        st.markdown(f"""
                <script>
                    const newUrl = new URL(window.location);
                    newUrl.searchParams.set('thread_id', '{thread.id}');
                    window.history.pushState(null, '', newUrl);
                </script>
                """, unsafe_allow_html=True)
        time.sleep(1)
        st.rerun()  # Refresh the app to use the new thread_id

    # Separator
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📂 Generated Excel Files")

    st.image("data/logo.png", width=300)

    st.title("📚 Lessons Learned Assistant")
    st.markdown(
        "Welcome! Ask me anything about Lessons Learned. I can help you find relevant information based on keywords.")
    st.markdown("---")

    # Retrieve the thread_id from query parameters
    if 'thread_id' in st.query_params:
        thread = client.beta.threads.retrieve(thread_id=st.query_params['thread_id'])
    else:
        # Create a new thread and update the URL
        thread = client.beta.threads.create()
        st.query_params.thread_id = thread.id
        # JavaScript to update the URL without reloading the page
        st.markdown(f"""
                <script>
                    const newUrl = new URL(window.location);
                    newUrl.searchParams.set('thread_id', '{thread.id}');
                    window.history.pushState(null, '', newUrl);
                </script>
                """, unsafe_allow_html=True)

    st.session_state['thread_id'] = thread.id

    # Retrieve the existing thread
    thread = client.beta.threads.retrieve(thread_id=thread.id)
    thread_docs_dir = os.path.join(os.curdir, 'generated_excels', thread.id)
    st.session_state['thread_docs_dir'] = thread_docs_dir

    # Display download links for each generated Excel file
    if thread.id in os.listdir('generated_excels'):
        for idx, file_name in enumerate(os.listdir(os.path.join('generated_excels', thread.id))):
            file_path = os.path.join('generated_excels', thread.id, file_name)
            st.session_state['files'].append({
                'file_name': file_name,
                'data': to_excel(pd.read_excel(file_path)),
            })

    if st.session_state['files']:
        for idx, file_info in enumerate(st.session_state['files']):
            st.sidebar.download_button(
                label=f"📄 {file_info['file_name']}",
                data=file_info['data'],
                file_name=file_info['file_name'],
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key=f"download_button_{idx}"
            )
    else:
        st.sidebar.write("No lists has been found for this thread.")

    st.session_state['files'] = []

    st.sidebar.markdown("---")
    st.sidebar.write("For any ideas or help:")
    st.sidebar.write("📧 Farzam.taghipour.ext@siemens-energy.com")

    # Initialize messages in session state
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
        # Fetch previous messages from the assistant service
        messages = reversed(client.beta.threads.messages.list(thread_id=thread.id).data)
        for msg in messages:
            role = getattr(msg, 'role', None)
            sender = getattr(msg, 'sender', None)
            content = msg.content
            if isinstance(content, list):
                content_text = content[0].text.value
            else:
                content_text = content

            if sender == 'user' or role == 'user':
                st.session_state['messages'].append({
                    "role": "user",
                    "content": content_text,
                    "timestamp": datetime.now()
                })

            elif sender == 'assistant' or role == 'assistant':
                if len(msg.content[0].text.annotations) > 0:
                    msg.content[0].text.value = f"The file has been generated and is accessible from left side-bar under **```📂 Generated Excel Files```** section."

                st.session_state['messages'].append({
                    "role": "assistant",
                    "content": msg.content[0].text.value,
                    "timestamp": datetime.now()
                })

    # Display chat messages
    for message in (st.session_state['messages']):
        if message['role'] == 'user':
            with st.chat_message("user"):
                st.markdown(message['content'])
        elif message['role'] == 'assistant':
            with st.chat_message("assistant"):
                st.markdown(message['content'])

    # Use st.chat_input to get user input and send message on Enter
    user_input = st.chat_input("Ask me anything about the Lessons Learned database")
    if user_input:
        # Add user message to session and display it
        st.session_state['messages'].append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
        with st.chat_message("user"):
            st.markdown(user_input)

        if st.session_state['LAST_FILE_CREATED_ID']:
            attachments = [
                {
                    "file_id": st.session_state['LAST_FILE_CREATED_ID'],
                    "tools": [{"type": "code_interpreter"}]
                }
            ]
        else:
            attachments = []

        # Send user message to the assistant
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_input,
            attachments=attachments
        )

        # Show a spinner while processing
        with st.spinner("Processing your request..."):
            # Run the assistant
            run = client.beta.threads.runs.create(
                thread_id=thread.id,
                assistant_id=assistant.id
            )

            # Monitor the run status
            while run.status in ['queued', 'in_progress', 'cancelling']:
                time.sleep(1)
                run = client.beta.threads.runs.retrieve(
                    thread_id=thread.id,
                    run_id=run.id
                )

                # Handle the run output
            if run.status == 'requires_action':
                # Process tool calls
                tool_calls = run.required_action.submit_tool_outputs.tool_calls
                tool_outputs = list(map(get_output_from_toolcall, tool_calls))

                # Submit the tool outputs back to the run
                run = client.beta.threads.runs.submit_tool_outputs(
                    run_id=run.id,
                    thread_id=thread.id,
                    tool_outputs=tool_outputs
                )

                # Retrieve the updated run
                while run.status in ['queued', 'in_progress', 'cancelling']:
                    time.sleep(1)
                    run = client.beta.threads.runs.retrieve(
                        thread_id=thread.id,
                        run_id=run.id
                    )

            if run.status == 'completed':
                # Get the assistant's response
                messages = client.beta.threads.messages.list(thread_id=thread.id).data
                # Find the latest assistant message
                assistant_messages = []
                for msg in messages:
                    role = getattr(msg, 'role', None)
                    sender = getattr(msg, 'sender', None)
                    if sender == 'assistant' or role == 'assistant':
                        if len(msg.content[0].text.annotations) > 0:
                            msg.content[0].text.value = f"The file has been generated and is accessible from left side-bar under **```📂 Generated Excel Files```** section."
                        assistant_messages.append(msg)

                if assistant_messages:
                    latest_assistant_message = assistant_messages[0]
                    assistant_message_content = latest_assistant_message.content
                    if isinstance(assistant_message_content, list):
                        assistant_message = assistant_message_content[0].text.value
                        if len(assistant_message_content[0].text.annotations) > 0:
                            file_id = assistant_message_content[0].text.annotations[0].file_path.file_id
                            file_name = assistant_message_content[0].text.annotations[0].text.split('/')[-1]
                            download_excel_file_and_return_path(file_id, file_name)
                            # assistant_message = f"The file {file_name} has been generated and is accessible from left side-bar under **```📂 Generated Excel Files```** section."
                    else:
                        assistant_message = assistant_message_content
                else:
                    assistant_message = "I'm sorry, I didn't understand that."

                # Append assistant message to session state and display it
                st.session_state['messages'].append({
                    "role": "assistant",
                    "content": assistant_message,
                    "timestamp": datetime.now()
                })
                with st.chat_message("assistant"):
                    st.markdown(assistant_message)
                st.rerun()
            else:
                st.error(f"Run failed with status: {run.status}")


if __name__ == "__main__":
    main()