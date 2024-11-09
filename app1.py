import os
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

from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
from openai import OpenAI
import uuid
import io


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
            with self.engine.connect().execution_options(autocommit=True) as connection:
                connection.execute(text("TRUNCATE TABLE excel_data CASCADE"))
            logger.info("Table excel_data reset successfully")
        except Exception as e:
            logger.error(f"Error resetting database: {str(e)}")
            raise

    def table_exists(self) -> bool:
        """Check if excel_data table exists"""
        try:
            inspector = inspect(self.engine)
            return "excel_data" in inspector.get_table_names()
        except Exception as e:
            logger.error(f"Error checking table existence: {str(e)}")
            return False


class RAGSystem:
    def __init__(self, api_key: str, qdrant_url: str, qdrant_api_key: str, collection_name: str = "excel-embeddings"):
        """Initialize RAG system with OpenAI and Qdrant clients"""
        self.openai_client = OpenAI(api_key=api_key)
        self.qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        self.collection_name = collection_name
        
        # Create collection if it doesn't exist
        # all_collections = [col.name for col in self.qdrant_client.get_collections().collections]
        if self.qdrant_client.collection_exists(self.collection_name):
            self.qdrant_client.delete_collection(collection_name=self.collection_name)
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )

    def extract_columns_from_excel(
        self, 
        file_path: str, 
        question_col: str, 
        answer_col: str
    ) -> tuple[pd.DataFrame, List[str]]:
        """Extract and combine question-answer pairs from Excel"""
        try:
            df = pd.read_excel(file_path)
            raw_df = df.copy()
            
            if question_col not in df.columns or answer_col not in df.columns:
                raise ValueError(f"Required columns {question_col} and/or {answer_col} not found")
            
            df[question_col] = df[question_col].astype(str).str.strip()
            df[answer_col] = df[answer_col].astype(str).str.strip()
            df.drop_duplicates([question_col], inplace=True)
            
            selected_columns_text = df.apply(
                lambda row: f"Question: {row[question_col]}\nAnswer: {row[answer_col]}", 
                axis=1
            )
            
            return raw_df, list(selected_columns_text)
        except Exception as e:
            logger.error(f"Error processing Excel file: {str(e)}")
            raise

    def get_embeddings(
        self, 
        texts: List[str], 
        model: str = "text-embedding-3-small",
        batch_size: int = 100
    ) -> List[List[float]]:
        """Generate embeddings with batching"""
        all_embeddings = []
        try:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                retries = 3
                while retries > 0:
                    try:
                        response = self.openai_client.embeddings.create(
                            input=batch,
                            model=model,
                            dimensions=384
                        )
                        batch_embeddings = [item.embedding for item in response.data]
                        all_embeddings.extend(batch_embeddings)
                        break
                    except Exception as e:
                        retries -= 1
                        if retries == 0:
                            raise e
                        logger.warning(f"Retrying embedding generation. Error: {str(e)}")
            return all_embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise

    def store_data(
        self, 
        df: pd.DataFrame, 
        texts: List[str], 
        embeddings: List[List[float]]
    ) -> pd.DataFrame:
        """Store data in Qdrant"""
        try:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
            points = [
                models.PointStruct(
                    id=id_, 
                    vector=embedding, 
                    payload={
                        "source": "excel",
                        "timestamp": datetime.now().isoformat(),
                        "document_id": id_,
                        "text": text
                    }
                ) for id_, text, embedding in zip(ids, texts, embeddings)
            ]
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return df
        except Exception as e:
            logger.error(f"Error storing data in Qdrant: {str(e)}")
            raise

    def query_similar(
        self, 
        query_text: str, 
        top_k: int = 5
    ) -> List[Dict]:
        """Query similar documents from Qdrant"""
        try:
            query_embedding = self.get_embeddings([query_text])[0]
            results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                with_payload=True
            )
            
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'document': result.payload.get("text", ""),
                    'metadata': result.payload,
                    'similarity': result.score
                })
            return formatted_results
        except Exception as e:
            logger.error(f"Error querying similar documents: {str(e)}")
            raise


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'collection_name' not in st.session_state:
        st.session_state.collection_name = "excel-embeddings"
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if 'metadata' not in st.session_state:
        st.session_state.metadata = None
    if 'authenticated_email' not in st.session_state:
        st.session_state.authenticated_email = False


def process_file(
    file_path: str,
    rag_system: RAGSystem,
    db_manager: DatabaseManager,
    question_col: str,
    answer_col: str
) -> None:
    """Process a single Excel file"""
    df, texts = rag_system.extract_columns_from_excel(
        str(file_path),
        question_col,
        answer_col
    )
    
    embeddings = rag_system.get_embeddings(texts)
    df = rag_system.store_data(df, texts, embeddings)
    
    # Store in database with appropriate if_exists parameter
    if db_manager.table_exists():
        db_manager.store_dataframe(file_path, if_exists='append')
    else:
        db_manager.store_dataframe(file_path, if_exists='replace')
    
    st.session_state.metadata = db_manager.get_metadata()


def main():
    st.set_page_config(page_title="Enhanced RAG + SQL System", layout="wide")
    st.title("Enhanced :blue[SQL Agentic] System")
    
    initialize_session_state()
    
    api_key = os.getenv("OPENAI_API_KEY")
    database_url = os.getenv("DATABASE_URL")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api = os.getenv("QDRANT_API")

    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem(api_key, qdrant_url, qdrant_api)
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager(database_url)

    # File upload section
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
    
    with col2:
        if st.button("Reset Database"):
            try:
                if st.session_state.db_manager.table_exists():
                    st.warning("This will delete all data. Are you sure?")
                    if st.button("Confirm Reset", key='confirm_reset'):
                        st.session_state.db_manager.reset_database()
                        st.session_state.metadata = None
                        st.success("Database reset successfully!")
                else:
                    st.warning("No data to reset.")
            except Exception as e:
                st.error(f"Error resetting database: {str(e)}")

    # Process uploaded files
    if uploaded_files and st.button("Process Files", key='process_files'):
        with st.spinner("Processing files..."):
            for uploaded_file in uploaded_files:
                file_path = Path(uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                process_file(
                    str(file_path),
                    st.session_state.rag_system,
                    st.session_state.db_manager,
                    question_column,
                    answer_column
                )
                
                os.remove(file_path)
            
            st.success("All files processed successfully!")

    # Initialize chat components
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
            ),
            Tool(
                name="query_RAG",
                func=st.session_state.rag_system.query_similar,
                description="Tool to retrieve similar text when a text closely matches previous entries in RAG system."
            )
        ] 
        
        prompt = hub.pull("hwchase17/react-chat")
        prompt.template = """
        Assistant is a large language model trained by OpenAI.

        Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

        Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

        Assistant is designed to answer user queries by leveraging a SQL database and RAG sytem (a vector database for question similarity search).

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
        # - **execute_sql_query**: e.g., `execute_sql_query('SELECT column_name FROM table_name WHERE question_id IN (...)')`. Retrieves answers from the SQL database for matched question IDs.
        # - **query_RAG**: e.g., `query_RAG('user query text')`. Finds similar questions in RAG system based on the user query.

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

        Question: I need guidance on renaming segments in the FMS system.

        Thought: The user has a problem, I should first search RAG system for similar questions to the user query to see if a similar problem exists.

        Action: query_RAG
        
        Action Input: guidance on renaming segments in the FMS system

        Observation: RAG sytem returned similar questions. The corresponding answers are also provided. 

        Final Answer: Based on the information in the database, here are the steps to rename segments in the FMS system:

        1. Log into the FMS administration portal.
        2. Navigate to the Segments section.
        3. Locate the segment you want to rename.
        4. Click the "Rename" button.
        5. Enter the new segment name and save the changes.

        Let me know if you need any clarification or have additional questions!

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
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response['output']
                })
                
                with st.chat_message("assistant"):
                    st.write(response['output'].strip())
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
