from datetime import datetime
import os
import io
import logging
from typing import List, Dict
from pathlib import Path

from sqlalchemy import create_engine, text, inspect
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
        self.table_name = "excel_data"
        
    def store_dataframe(
        self, 
        file_path: str, 
        if_exists: str = 'append'
    ) -> None:
        """Store DataFrame in excel_data table"""
        try:
            df = pd.read_excel(file_path)
            df.to_sql(self.table_name, self.engine, if_exists=if_exists, index=False)
            logger.info(f"Data stored successfully in {self.table_name}")
        except Exception as e:
            logger.error(f"Error storing DataFrame: {str(e)}")
            raise
            
    def execute_query(self, query: str) -> List[str]:
        """Execute SQL query"""
        try:
            with self.engine.connect() as connection:
                result_set = connection.execute(text(query))
                return [str(row) for row in result_set]
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return str(e)
            
    def get_metadata(self) -> Dict:
        """Get database metadata for excel_data table"""
        try:
            query_columns = f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'excel_data';
            """
            columns = self.execute_query(query_columns)
            return {"excel_data": {'columns': columns}}
        except Exception as e:
            logger.error(f"Error getting metadata: {str(e)}")
            raise
            
    def download_data(self) -> bytes:
        """Download all data from excel_data table as Excel file"""
        try:
            query = "SELECT * FROM excel_data"
            df = pd.read_sql(query, self.engine)
            
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
            with self.engine.connect() as connection:
                # First verify we have data
                result = connection.execute(text("SELECT COUNT(*) FROM excel_data"))
                count_before = result.scalar()
                logger.info(f"Records before reset: {count_before}")
                
                # Execute the truncate
                connection.execute(text("TRUNCATE TABLE excel_data"))
                connection.commit()
                
                # Verify the deletion
                result = connection.execute(text("SELECT COUNT(*) FROM excel_data"))
                count_after = result.scalar()
                logger.info(f"Records after reset: {count_after}")
                
                if count_after == 0:
                    logger.info("Table excel_data reset successfully")
                    return True
                else:
                    logger.error("Table was not properly reset")
                    return False
                    
        except Exception as e:
            logger.error(f"Error resetting database: {str(e)}")
            raise
            
    def get_record_count(self) -> int:
        """Get the current number of records in the table"""
        try:
            with self.engine.connect() as connection:
                result = connection.execute(text("SELECT COUNT(*) FROM excel_data"))
                return result.scalar()
        except Exception as e:
            logger.error(f"Error getting record count: {str(e)}")
            return -1

    def table_exists(self) -> bool:
        """Check if excel_data table exists"""
        try:
            inspector = inspect(self.engine)
            return "excel_data" in inspector.get_table_names()
        except Exception as e:
            logger.error(f"Error checking table existence: {str(e)}")
            return False

def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if 'metadata' not in st.session_state:
        st.session_state.metadata = []

def process_file(
    file_path: str,
    db_manager: DatabaseManager,
    question_col: str,
    answer_col: str
) -> None:
    """Process a single Excel file"""
    if db_manager.table_exists():
        db_manager.store_dataframe(file_path, if_exists='append')
    else:
        db_manager.store_dataframe(file_path, if_exists='replace')
    
    # st.session_state.metadata = db_manager.get_metadata()

def main():
    st.set_page_config(page_title="Enhanced RAG + SQL + Email System", layout="wide")
    st.title("Enhanced :blue[SQL Agentic] System")
    
    initialize_session_state()
    
    uploaded_files = None
    question_column = None
    answer_column = None

    # from dotenv import load_dotenv
    # load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')

    database_url = os.getenv('DATABASE_URL')

    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager(database_url)
    
    st.session_state.metadata = st.session_state.db_manager.get_metadata()

    uploaded_files = st.file_uploader(
        "Upload Excel Files",
        type=["xlsx"],
        key='files',
        accept_multiple_files=True
    )
    
    question_column = 'Request - Text Request'
    answer_column = 'Request - Text Answer'

    # Database control buttons
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
        # current_count = st.session_state.db_manager.get_record_count()
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
    
    if uploaded_files and st.button("File Upload", key='process_files'):
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
                return_messages=True, 
                memory_key='chat_history', 
                input_key='input', 
                k=5
            )

        if 'llm' not in st.session_state:
            st.session_state.llm = ChatOpenAI(
                model="gpt-4o-mini", 
                temperature=0.7, 
                api_key=api_key
            )

        if 'agent' not in st.session_state:
            tools = [
                Tool(
                    name="execute_sql_query",
                    func=st.session_state.db_manager.execute_query,
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
                        Answer the queries from the excel_data table:
                        Metadata of the table:
                        {st.session_state.metadata} 

                        User query: 
                        {query}
                    """
                    
                    response = st.session_state['agent_executor'].invoke({"input": command})
                    
                    # Store assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response['output']
                    })
                    
                    # Display message
                    with st.chat_message("assistant"):
                        st.write(response['output'].strip())
                        # st.write(response)
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
    