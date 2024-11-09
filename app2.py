import os
import logging
from typing import List, Dict
from pathlib import Path

from sqlalchemy import create_engine, text
import streamlit as st
import pandas as pd

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.tools.base import Tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    def __init__(self, database_url: str):
        """Initialize database connection"""
        self.database_url = database_url
        self.engine = create_engine(database_url)
        
    def store_dataframe(
        self, 
        file_path: str, 
        table_name: str, 
        if_exists: str = 'append'
    ) -> None:
        """Store DataFrame in database"""
        try:
            df = pd.read_excel(file_path)
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
        except Exception as e:
            logger.error(f"Error storing DataFrame: {str(e)}")
            raise
            
    def execute_query(self, query: str) -> List[str]:
        """Execute SQL query"""
        try:
            with self.engine.connect() as connection:
                print(text(query))
                result_set = connection.execute(text(query))
                return [str(row) for row in result_set]
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return str(e)
            
    def get_metadata(self, tables: List[str]) -> List[Dict]:
        """Get database metadata"""
        metadata = []
        try:
            for table in tables:
                query_columns = f"""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = '{table}';
                """
                columns = self.execute_query(query_columns)
                metadata.append({table: {'columns': columns}})
            return metadata
        except Exception as e:
            logger.error(f"Error getting metadata: {str(e)}")
            raise

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'collection_name' not in st.session_state:
        st.session_state.collection_name = "excel-embeddings"
    if 'tables' not in st.session_state:
        st.session_state.tables = []
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if 'metadata' not in st.session_state:
        st.session_state.metadata = []
    if 'authenticated_email' not in st.session_state:
        st.session_state.authenticated_email = False

def process_file(
    file_path: str,
    # rag_system: RAGSystem,
    db_manager: DatabaseManager,
    question_col: str,
    answer_col: str
) -> None:
    """Process a single Excel file"""
    table_name = "excel_data"
    st.session_state.tables.append(table_name)
    
    # df, texts = rag_system.extract_columns_from_excel(
    #     str(file_path),
    #     question_col,
    #     answer_col
    # )
    
    # embeddings = rag_system.get_embeddings(texts)
    # df = rag_system.store_data(df, texts, embeddings)
    db_manager.store_dataframe(file_path, table_name)
    
    st.session_state.metadata = db_manager.get_metadata(st.session_state.tables)

def main():
    st.set_page_config(page_title="Enhanced RAG + SQL + Email System", layout="wide")
    st.title("Enhanced :blue[SQL Agentic] System")
    
    initialize_session_state()
    
    uploaded_files = None
    question_column = None
    answer_column = None

    # Sidebar for system selection
    # with st.sidebar:
    #     st.title("System Configuration")
    #     system_type = st.radio(
    #         "Choose Processing Method",
    #         ["Direct Upload", "Email Processing"]
    #     )
    
    # Common credentials
    # api_key = st.text_input("Enter OpenAI API key", type='password', key='api_key')
    api_key = os.getenv('OPENAI_API_KEY')
    # database_url = st.text_input(
    #     "Database URL",
    #     "postgresql://test_owner:tcWI7unQ6REA@ep-yellow-recipe-a5fny139.us-east-2.aws.neon.tech:5432/test",
    #     key='db_url'
    # )
    database_url = os.getenv('DATABASE_URL')

    # if system_type == "Direct Upload":
        # Direct upload interface
    uploaded_files = st.file_uploader(
        "Upload Excel Files",
        type=["xlsx"],
        key='files',
        accept_multiple_files=True
    )
    
    col1, col2 = st.columns(2)
    # with col1:
    #     question_column = st.text_input(
    #         "Question column name",
    #         key='question_column',
    #         value='Request - Text Request'
    #     )
    # with col2:
    #     answer_column = st.text_input(
    #         "Answer column name",
    #         key='answer_column',
    #         value='Request - Text Answer'
    #     )
    question_column = 'Request - Text Request'
    answer_column = 'Request - Text Answer'
    
    if api_key and st.button("File Upload", key='process_files'):
        if uploaded_files:
            # rag_system = RAGSystem(api_key)
            db_manager = DatabaseManager(database_url)
            
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    file_path = Path(uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    process_file(
                        str(file_path),
                        # rag_system,
                        db_manager,
                        question_column,
                        answer_column
                    )
                    
                    os.remove(file_path)
            
            st.success("All files processed successfully!")
        else:
            st.error("Please upload at least one Excel file.")

    # Chat interface (available for both modes)
    if api_key:
        st.subheader("Talk with your Data")
        
        # Initialize RAG tools and agent if not already done
        if 'memory' not in st.session_state:
            st.session_state.memory = ConversationBufferWindowMemory(return_messages=True, memory_key='chat_history', input_key='input', k=5)
        if 'llm' not in st.session_state:
            st.session_state['llm'] = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, api_key=api_key)

        
        if 'agent' not in st.session_state:
            tools = [
                Tool(
                    name="execute_sql_query",
                    func=DatabaseManager(database_url).execute_query,
                    description=(
                        "Tool for executing SQL queries on a structured database. "
                    )
                )
            ]
            
            
            prompt = hub.pull("hwchase17/react-chat")
            prompt.template = """
            Assistant is a large language model trained by OpenAI.

            Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

            Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

            Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.

            TOOLS:
            ------

            Assistant has access to the following tools:

            {tools}

            To use a tool, please use the following format:

            ```
            Thought: Do I need to use a tool? Yes
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ```
            When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

            ```
            Thought: Do I need to use a tool? No
            Final Answer: [your response here]
            ```
            ### Example Session:

            ## Example Actions:
            # - **execute_query**: e.g., `execute_query('SELECT column_name FROM table_name WHERE question_id IN (...)')`. Retrieves answers from the SQL database for matched question IDs.

            ## Assistant Flow:
            Question: Hi

            Thought: The user has greeted me, so I will respond warmly.

            Final Answer: Hi! I'm here to assist you. If you have any questions feel free to ask!

            Question: How many tickets are raised?

            Thought: The user has asked a question about the number of tickets raised. This is likely a specific piece of information, so I should check the SQL database to see if there are any records related to the ticket count.

            Action: execute_query

            Action Input: 
            SELECT COUNT(*) AS ticket_count 
            FROM table_name 
            WHERE status != "closed"

            Observation: The SQL query returned a value of 42 for the ticket_count. This directly answers the user's question about the number of open tickets in the FMS system. 

            Final Answer: According to the information in the database, there are currently 42 tickets raised in the FMS system that are not in the 'closed' status.

            ```

            Begin!

            Previous conversation history:
            {chat_history}

            Question : {input}
            {agent_scratchpad}
            """
            agent = create_react_agent(st.session_state['llm'], tools, prompt)
            st.session_state['agent_executor'] = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, memory=st.session_state.memory)

        # Chat interface
        query = st.text_area("Enter your question", height=100)
        
        if st.button("Process"):
            if query:
                with st.spinner("Processing query..."):
                    # Store user message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": query
                    })
                    
                    # Generate context for the agent
                    command = f"""
                        Answer the queries from these tables:
                        {st.session_state.tables}

                        Metadata of the tables:
                        {st.session_state.metadata} 

                        User query: 
                        {query}
                    """
                    
                    # Get response from agent
                    # response = st.session_state.agent(command)
                    # response = response.split('**Answer**:')[-1].strip()
                    response = st.session_state['agent_executor'].invoke({"input": command})
                    
                    # Store assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response['output']
                    })
                    
                    # Display message
                    with st.chat_message("assistant"):
                        st.write(response['output'].strip())
                        st.write(response)
            else:
                st.warning("Please enter a question.")
        
        # Display chat history in sidebar
        with st.sidebar:
            st.header("Chat History")
            for msg in st.session_state.messages:
                with st.expander(f"{msg['role']} message"):
                    st.write(msg['content'])

if __name__ == "__main__":
    main()
    