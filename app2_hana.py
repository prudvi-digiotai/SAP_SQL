from json import load
import os
import io
import logging
from typing import List, Dict
from pathlib import Path
import datetime

from sqlalchemy import create_engine, text
import streamlit as st
import pandas as pd
from hdbcli import dbapi
from sqlalchemy.engine import URL

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.tools.base import Tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

class DatabaseManager:
    def __init__(self: str, address, port, user, password):
        """Initialize database connection"""

        self.conn = dbapi.connect(
            address=address,
            port=port,
            user=user,
            password=password
        )
        self.table_name = "excel_data"
        
    def store_dataframe(self, file_path: str, if_exists: str = 'append') -> None:
        """Store DataFrame in excel_data table using bulk insert"""
        try:
            df = pd.read_excel(file_path)
            
            # Handle NaN values: replace with None to store as NULL in DB
            df = df.applymap(lambda x: None if pd.isna(x) else x)
            
            # Dynamically generate column definitions based on DataFrame types
            column_definitions = []
            for column, dtype in df.dtypes.items():
                # Make sure column names are quoted to avoid issues with special characters
                column_name_quoted = f'"{column}"'

                if pd.api.types.is_integer_dtype(dtype):
                    column_definitions.append(f"{column_name_quoted} INTEGER")
                elif pd.api.types.is_float_dtype(dtype):
                    column_definitions.append(f"{column_name_quoted} DECIMAL")
                elif pd.api.types.is_string_dtype(dtype):
                    # Calculate max length for the column
                    max_length = df[column].astype(str).str.len().max()
                    # Use NVARCHAR with appropriate length, default to 5000 if very long
                    length = min(max_length + 100, 5000)  # add some buffer
                    column_definitions.append(f"{column_name_quoted} NVARCHAR({length})")
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    column_definitions.append(f"{column_name_quoted} DATE")
                else:
                    column_definitions.append(f"{column_name_quoted} NVARCHAR({5000})")  # Default to TEXT

            column_definitions_str = ", ".join(column_definitions)

            with self.conn.cursor() as cursor:
                # If replace mode or table doesn't exist, create new table
                if if_exists == 'replace' or not self.table_exists():
                    if self.table_exists():
                        cursor.execute(f'DROP TABLE "{self.table_name}"')
                    cursor.execute(f'CREATE TABLE "{self.table_name}" ({column_definitions_str})')
                    self.conn.commit()

                # Bulk insert data
                column_names = ", ".join([f'"{col}"' for col in df.columns])
                placeholders = ", ".join(["?" for _ in df.columns])
                insert_query = f'INSERT INTO "{self.table_name}" ({column_names}) VALUES ({placeholders})'
                
                # Convert DataFrame to list of tuples for bulk insert
                data = [tuple(row) for row in df.values]
                cursor.executemany(insert_query, data)
                self.conn.commit()

            logger.info(f"Data stored successfully in {self.table_name}")
        except Exception as e:
            logger.error(f"Error storing DataFrame: {str(e)}")
            raise

    def table_exists(self) -> bool:
        """Check if table exists"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(f"SELECT COUNT(*) FROM SYS.TABLES WHERE table_name = '{self.table_name}'")
                result = cursor.fetchone()
                return result[0] > 0
        except Exception as e:
            logger.error(f"Error checking table existence: {str(e)}")
            return False
            
    def execute_query(self, query: str) -> List[str]:
        """Execute SQL query"""
        try:
            with self.conn.cursor() as cursor:
                print("-" * 100)
                print(query)
                print("-" * 100, "HI")
                cleaned_query = query.strip().strip("'")
                print("Cleaned query:", cleaned_query)  # Debug print
                cursor.execute(cleaned_query)
                result_set = cursor.fetchall()
                return [str(row) for row in result_set]
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return str(e)
            
    def get_metadata(self):
        """Get metadata for the table columns"""
        try:
            with self.conn.cursor() as cursor:
                query = f"""
                SELECT COLUMN_NAME, DATA_TYPE_NAME, IS_NULLABLE
                FROM TABLE_COLUMNS
                WHERE SCHEMA_NAME = CURRENT_SCHEMA 
                AND TABLE_NAME = '{self.table_name}'
                ORDER BY POSITION
                """
                cursor.execute(query)
                metadata = cursor.fetchall()
                print("meatadata" , metadata)
                return metadata
        except Exception as e:
            logger.error(f"Error retrieving metadata: {str(e)}")
            raise
            
    def download_data(self) -> bytes:
        """Download all data from excel_data table as Excel file"""
        try:
            query = f"SELECT * FROM {self.table_name}"
            df = pd.read_sql(query, self.conn)
            
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False)
            output.seek(0)
            return output.getvalue()
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            raise
            
    def reset_database(self) -> None:
        """Delete all records from excel_data table"""
        try:
            with self.conn.cursor() as cursor:
                print("delete")
                cursor.execute(f'DELETE FROM "{self.table_name}"')
                self.conn.commit()
            logger.info("Table excel_data reset successfully")
            return True
        except Exception as e:
            logger.error(f"Error resetting database: {str(e)}")
            raise

    def get_record_count(self) -> int:
        """Get the current number of records in the table"""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(f'SELECT COUNT(*) FROM "{self.table_name}"')
                result = cursor.fetchone()  # Fetch the first row (should be a single value)
                return result[0] if result else 0  # Return the count, or 0 if no result
        except Exception as e:
            logger.error(f"Error getting record count: {str(e)}")
            return -1

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'collection_name' not in st.session_state:
        st.session_state.collection_name = "excel-embeddings"
    if 'tables' not in st.session_state:
        st.session_state.tables = []
    if "messages" not in st.session_state:
        st.session_state.messages = []

def process_file(
    file_path: str,
    db_manager: DatabaseManager,
    question_col: str,
    answer_col: str
) -> None:
    """Process a single Excel file"""
    table_name = "excel_dataa"
    st.session_state.tables.append(table_name)
    
    db_manager.store_dataframe(file_path, table_name)

def main():
    st.set_page_config(page_title="Enhanced RAG + SQL + Email System", layout="centered")
    st.title("Enhanced :blue[SQL Agentic] System")
    
    initialize_session_state()
    
    uploaded_files = None
    question_column = None
    answer_column = None

    api_key = os.getenv("OPENAI_API_KEY")

    address = os.getenv("HANA_ADDRESS")
    port = os.getenv("HANA_PORT")
    user = os.getenv("HANA_USER")
    password = os.getenv("HANA_PASSWORD")

    uploaded_files = st.file_uploader(
        "Upload Excel Files",
        type=["xlsx"],
        key='files',
        accept_multiple_files=True
    )

    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager(address, port, user, password)
    
    if 'metadata' not in st.session_state:
        st.session_state.metadata = st.session_state.db_manager.get_metadata()
    
    question_column = 'Request - Text Request'
    answer_column = 'Request - Text Answer'

    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("Download Data"):
            try:
                if st.session_state.db_manager.table_exists():
                    excel_data = st.session_state.db_manager.download_data()
                    st.download_button(
                        label="Click here to download",
                        data=excel_data,
                        file_name=f"excel_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key='download_button'  
                    )
                else:
                    st.warning("No data available to download.")
            except Exception as e:
                st.error(f"Error downloading data: {str(e)}")
    
    if "confirm_reset_triggered" not in st.session_state:
        st.session_state.confirm_reset_triggered = False

    if 'reset_clicked' not in st.session_state:
        st.session_state.reset_clicked = False

    # Reset database logic
    with col2:
        # Display current record count
        current_count = st.session_state.db_manager.get_record_count()
        # st.write(f"Current records: {current_count}")
        
        # First button toggles the reset_clicked state
        if not st.session_state.reset_clicked:
            if st.button("Reset Database"):
                if current_count > 0:
                    st.session_state.reset_clicked = True
                else:
                    st.warning("No data to reset.")
        
        # Show confirmation button only if reset was clicked
        if st.session_state.reset_clicked:
            st.warning(f"This will delete all {current_count} records. Are you sure?")
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                if st.button("Yes, Reset"):
                    try:
                        # Attempt reset and verify
                        success = st.session_state.db_manager.reset_database()
                        new_count = st.session_state.db_manager.get_record_count()
                        print(success, "        ", new_count)
                        
                        if success and new_count == 0:
                            st.success(f"Database reset successfully! (Verified: {new_count} records remaining)")
                            st.session_state.metadata = None
                        else:
                            st.error(f"Reset may have failed. Records remaining: {new_count}")
                        
                        st.session_state.reset_clicked = False
                        
                        # Force page rerun to update all components
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error resetting database: {str(e)}")
                        st.session_state.reset_clicked = False
            
            with col2_2:
                if st.button("Cancel"):
                    st.session_state.reset_clicked = False
                    st.rerun()

    
    if st.button("File Upload", key='process_files'):
        if uploaded_files:
            
            for uploaded_file in uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    file_path = Path(uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    process_file(
                        str(file_path),
                        st.session_state.db_manager,
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
            st.session_state.memory = ConversationBufferWindowMemory(
                ai_prefix="Assistant",
                human_prefix="User",
                return_messages=True, memory_key='chat_history', input_key='input', k=5)
        if 'llm' not in st.session_state:
            st.session_state['llm'] = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, api_key=api_key)

        
        if 'agent' not in st.session_state:

            tools = [
                Tool(
                    name="execute_sap_hana_query",
                    func=lambda x: DatabaseManager(address, port, user, password).execute_query(x.strip("'").rstrip("'")),
                    description=(
                        """
                        Tool for executing SQL queries on a SAP HANA database.                      
                        """
                    )
                )
            ]
            
            prompt = hub.pull("hwchase17/react-chat")
            prompt.template = """
            Assistant is a large language model trained by OpenAI.

            Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

            Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

            Assistant always with responds with one of ('Thought', 'Action', 'Action Input', 'Observation' or 'Final Answer')
    
            Also, Assistant is an expert data analyst and SQL query specialist, particularly skilled with SAP HANA databases. Assistant writes precise, efficient queries and provides clear, insightful analysis of the results.

            TOOLS:
            ------

            Assistant has access to the following tools:

            {tools}

            To use a tool, please use the following format:

            ```
            Thought: Do I need to use a tool? Yes
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action (no additional text)
            Observation: the result of the action
            ```
            When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

            ```
            Thought: Do I need to use a tool? No
            Final Answer: [your response here]
            ```
            ### Example Session:

            ## Example Actions:
            # - **execute_query**: e.g., `execute_query('SELECT "column_name" FROM "table_name" WHERE "question_id" IN (...)')`. Retrieves answers from the SQL database for matched question IDs.

            ## Assistant Flow:
            Question: Hi

            Thought: The user has greeted me, so I will respond warmly.

            Final Answer: Hi! I'm here to assist you. If you have any questions feel free to ask!

            Question: How many tickets are raised?

            Thought: The user has asked a question about the number of tickets raised. This is likely a specific piece of information, so I should check the SQL database to see if there are any records related to the ticket count.

            Action: execute_query

            Action Input: 'SELECT COUNT(*) AS "ticket_count" FROM "table_name" WHERE "status" != "closed"'

            Observation: The SQL query returned a value of 42 for the ticket_count. This directly answers the user's question about the number of open tickets in the FMS system. 

            Final Answer: According to the information in the database, there are currently 42 tickets raised in the FMS system that are not in the 'closed' status.

            ```

            Begin! Remember to maintain this exact format for all interactions and focus on writing clean, error-free SQL queries. Make sure to provide Final Answer to user's question.

            Previous conversation history:
            {chat_history}

            Question : {input}
            {agent_scratchpad}
            """
            agent = create_react_agent(st.session_state['llm'], tools, prompt)
            st.session_state['agent_executor'] = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, memory=st.session_state.memory, max_iterations=15)

        # Chat interface
        query = st.chat_input("Enter your question")
        
        if query:
            if query:
                with st.spinner("Processing query..."):
                    # Store user message
                    st.session_state.messages.append({
                        "role": "user",
                        "content": query
                    })
                    
                    # Generate context for the agent
                    command = f"""
                        Answer the queries from 'excel_data' tables:

                        Metadata of the tables:
                        {st.session_state.metadata} 

                        User query: 
                        {query}
                    """
                    
                    # Get response from agent
                    response = st.session_state['agent_executor'].invoke({"input": command})
                    
                    # Store assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response['output'].strip('```')
                    })

            else:
                st.warning("Please enter a question.")
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

if __name__ == "__main__":
    main()
    
