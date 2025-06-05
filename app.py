import nest_asyncio
nest_asyncio.apply()

import streamlit as st
from dotenv import load_dotenv
import pdfplumber
import pandas as pd
import sqlite3
import re
from docx import Document
import io

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# --- HTML Templates for Chat UI ---
from htmlTemplates import css, bot_template, user_template
# --- End of HTML Templates ---


load_dotenv() # Load environment variables (e.g., GOOGLE_API_KEY)

def extract_text_from_documents(uploaded_files):
    """Extracts text and structured tables/data from a list of various document types."""
    all_raw_text = ""
    all_extracted_tables_as_lists = []
    all_excel_dfs = []

    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_extension == 'pdf':
                with pdfplumber.open(uploaded_file) as reader:
                    tables_found_on_file = 0
                    for page in reader.pages:
                        page_text = page.extract_text() or ""
                        all_raw_text += "\n" + page_text
                        tables = page.extract_tables()
                        for table in tables:
                            if table:
                                all_extracted_tables_as_lists.append(table)
                                tables_found_on_file += 1
                st.info(f"Extracted data from **{uploaded_file.name}** (PDF). Tables found: **{tables_found_on_file}**")

            elif file_extension == 'txt':
                string_data = uploaded_file.getvalue().decode("utf-8")
                all_raw_text += "\n" + string_data
                st.info(f"Extracted text from **{uploaded_file.name}** (TXT).")

            elif file_extension == 'docx':
                doc = Document(uploaded_file)
                doc_text = ""
                for paragraph in doc.paragraphs:
                    doc_text += paragraph.text + "\n"
                all_raw_text += "\n" + doc_text
                st.info(f"Extracted text from **{uploaded_file.name}** (DOCX).")

            elif file_extension in ['xlsx', 'xls']:
                df_excel = pd.read_excel(uploaded_file)
                all_excel_dfs.append(df_excel)
                # Add Excel data to raw text for RAG (convert to string for context)
                all_raw_text += "\n" + df_excel.to_string(index=False) 
                st.info(f"Extracted data from **{uploaded_file.name}** (Excel).")

            else:
                st.warning(f"Unsupported file type: `.{file_extension}`. Skipping **{uploaded_file.name}**.")

        except Exception as e:
            st.error(f"Error reading `{file_extension}` file **'{uploaded_file.name}'**: {e}. Please ensure it's a valid document.")
            
    return all_raw_text, all_extracted_tables_as_lists, all_excel_dfs

def process_data_to_dataframe(all_raw_text, all_extracted_tables_as_lists, all_excel_dfs):
    """Attempts to process extracted tables and Excel DataFrames into a single pandas DataFrame.
    Prioritizes merging Excel files by common IDs, then concatenates PDF tables,
    then attempts heuristic parsing of raw text if no other structured data is found.
    """
    final_df = None
    initial_dfs = []

    # Process Excel DataFrames first, aiming for merge
    if all_excel_dfs:
        st.info(f"Attempting to process and merge **{len(all_excel_dfs)}** DataFrames from Excel files.")
        
        # Sanitize column names for each Excel DF immediately for consistent merging
        cleaned_excel_dfs = []
        for i, df_excel in enumerate(all_excel_dfs):
            if df_excel.empty:
                st.warning(f"Excel DataFrame {i+1} is empty. Skipping.")
                continue
            
            # Sanitize columns
            df_excel.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col.replace(' ', '_')) for col in df_excel.columns]
            
            # Ensure unique column names within this DF
            seen = {}
            new_cols = []
            for col in df_excel.columns:
                temp_col = col
                count = 1
                while temp_col in seen:
                    temp_col = f"{col}_{count}"
                    count += 1
                seen[temp_col] = True
                new_cols.append(temp_col)
            df_excel.columns = new_cols
            
            cleaned_excel_dfs.append(df_excel)

        if cleaned_excel_dfs:
            # Try to find common ID columns for merging
            common_id_patterns = ['id', 'emp_id', 'employee_id', 'product_id', 'order_id', 'customer_id', 'person_id', 'code']
            all_cols = [col for df in cleaned_excel_dfs for col in df.columns]
            
            potential_merge_keys = []
            for pattern in common_id_patterns:
                # Find columns that match the pattern across ALL dataframes
                if all(any(re.search(f"^{pattern}$", col.lower()) or col.lower().startswith(pattern) for col in df.columns) for df in cleaned_excel_dfs):
                    # Pick the first matching column name for consistency if multiple exist
                    # This assumes the key name is consistent across files after sanitization
                    first_df_col = next((col for df in cleaned_excel_dfs for col in df.columns if re.search(f"^{pattern}$", col.lower()) or col.lower().startswith(pattern)), None)
                    if first_df_col and first_df_col not in potential_merge_keys:
                        potential_merge_keys.append(first_df_col)

            # Prioritize 'id' or 'employee_id' if present and common
            merge_key_found = False
            if potential_merge_keys:
                # Try to merge iteratively
                merged_df = cleaned_excel_dfs[0]
                st.info(f"Attempting to merge Excel files using common key(s): `{', '.join(potential_merge_keys)}`")
                
                try:
                    for i in range(1, len(cleaned_excel_dfs)):
                        # Find the actual common columns for this specific merge
                        current_merge_keys = [key for key in potential_merge_keys if key in merged_df.columns and key in cleaned_excel_dfs[i].columns]
                        if current_merge_keys:
                            merged_df = pd.merge(merged_df, cleaned_excel_dfs[i], on=current_merge_keys, how='outer', suffixes=('', f'_{i}'))
                            st.info(f"Successfully merged Excel file {i+1} using: `{', '.join(current_merge_keys)}`")
                        else:
                            st.warning(f"No common merge keys found between merged result and Excel file {i+1}. Concatenating instead.")
                            # Fallback to concat if merge keys are missing for a pair
                            merged_df = pd.concat([merged_df, cleaned_excel_dfs[i]], ignore_index=True, sort=False)
                    
                    final_df = merged_df
                    merge_key_found = True
                    st.success("All Excel files processed, attempting to merge by common ID.")
                except Exception as e:
                    st.error(f"Error during Excel merging by ID: {e}. Falling back to concatenation.")
                    # Fallback to simple concatenation if merging fails completely
                    final_df = pd.concat(cleaned_excel_dfs, ignore_index=True, sort=False)
            
            if not merge_key_found and not final_df is None: # If no merge key was found or merge failed, and data exists
                st.warning("No suitable common ID columns found across all Excel files for merging. Concatenating Excel files instead.")
                final_df = pd.concat(cleaned_excel_dfs, ignore_index=True, sort=False)
            elif not cleaned_excel_dfs: # If no excel files could be cleaned
                final_df = None
            
            if final_df is not None and not final_df.empty:
                initial_dfs.append(final_df)
                st.success(f"Processed Excel data into a DataFrame with {len(final_df)} rows.")

    # Process PDF tables (always concatenate as merge not typical for diverse PDF tables)
    if all_extracted_tables_as_lists:
        st.info(f"Processing **{len(all_extracted_tables_as_lists)}** tables extracted from PDFs/other list-based data.")
        pdf_dfs = []
        for i, table_data in enumerate(all_extracted_tables_as_lists):
            st.info(f"Processing table **{i+1}** from PDF/list data...")

            if not table_data or not isinstance(table_data, list) or len(table_data) < 1:
                st.warning(f"Table **{i+1}** is empty or malformed. Skipping.")
                continue
            
            headers_raw = table_data[0]
            if not headers_raw or not any(h is not None for h in headers_raw):
                st.warning(f"Table **{i+1}** has no valid header row. Skipping.")
                continue

            headers = [str(col).strip() if col is not None else f"Column_{j}" for j, col in enumerate(headers_raw)]
            clean_headers = [re.sub(r'[^a-zA-Z0-9_]', '', h.replace(' ', '_')) for h in headers]
            
            seen = {}
            final_headers = []
            for h in clean_headers:
                if h in seen:
                    new_h = f"{h}_{seen[h]}"
                    seen[h] += 1
                    final_headers.append(new_h)
                else:
                    seen[h] = 1
                    final_headers.append(h)
            
            data_rows = table_data[1:]
            if not data_rows:
                st.warning(f"Table **{i+1}** has headers but no data rows. Skipping.")
                continue

            try:
                padded_data_rows = []
                expected_cols = len(final_headers)
                for r_idx, row in enumerate(data_rows):
                    if row is None:
                        padded_row = [None] * expected_cols
                    else:
                        if not isinstance(row, list):
                            row = [row]
                        current_row_processed = [str(cell).strip() if cell is not None else None for cell in row]
                        
                        if len(current_row_processed) < expected_cols:
                            current_row_processed.extend([None] * (expected_cols - len(current_row_processed)))
                        elif len(current_row_processed) > expected_cols:
                            current_row_processed = current_row_processed[:expected_cols]
                    padded_data_rows.append(current_row_processed)

                df = pd.DataFrame(padded_data_rows, columns=final_headers)
                
                if df.empty:
                    st.warning(f"DataFrame created from table **{i+1}** is empty after processing. Skipping.")
                    continue

                pdf_dfs.append(df)
                st.info(f"Successfully processed table **{i+1}** into a DataFrame with **{len(df)}** rows.")

            except ValueError as e:
                st.warning(f"Skipping table **{i+1}** due to column/data mismatch or pandas error: {e}. Headers: `{final_headers}`, Sample row: `{data_rows[0] if data_rows else 'N/A'}`")
                continue
            except Exception as e:
                st.error(f"An unexpected error occurred processing table **{i+1}**: {e}")
                continue
        
        if pdf_dfs:
            try:
                combined_pdf_df = pd.concat(pdf_dfs, ignore_index=True, sort=False)
                combined_pdf_df = combined_pdf_df.dropna(how='all')
                if not combined_pdf_df.empty:
                    initial_dfs.append(combined_pdf_df)
                    st.success(f"Combined PDF table data into a DataFrame with {len(combined_pdf_df)} rows.")
                else:
                    st.warning("Combined PDF DataFrame is empty after dropping all-NA rows.")
            except Exception as e:
                st.error(f"Error concatenating PDF DataFrames: {e}.")


    # Combine all processed structured data (Excel merges, PDF concatenations)
    if initial_dfs:
        try:
            final_df = pd.concat(initial_dfs, ignore_index=True, sort=False)
            final_df = final_df.dropna(how='all')

            if final_df.empty:
                st.warning("Final combined DataFrame is empty after dropping all-NA rows. No structured data available for SQL.")
                return None
            
            # Type conversion for numeric columns
            for col in final_df.columns:
                if isinstance(col, str) and any(kw in col.lower() for kw in ['salary', 'id', 'count', 'amount', 'value', 'price', 'quantity', 'age', 'total', 'score']):
                    final_df[col] = pd.to_numeric(final_df[col], errors='coerce')
            
            st.success(f"Successfully processed a combined DataFrame with **{len(final_df)}** rows for SQL querying.")
            return final_df
        except Exception as e:
            st.error(f"Error in final combination of DataFrames: {e}.")
            return None
    
    st.info("No structured tabular data could be processed into a DataFrame from any document.")
    if all_raw_text:
        st.info("Attempting heuristic parsing of raw text for potential tabular data from unstructured text (Experimental)...")
        lines = all_raw_text.split('\n')
        data_from_text = []
        potential_headers = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Simple heuristic for comma-separated data, check for at least 3 parts and non-empty parts
            if ',' in line and len(line.split(',')) > 2 and all(part.strip() for part in line.split(',')): 
                row_vals = [s.strip() for s in line.split(',')]
                
                if potential_headers is None:
                    # Very basic header inference: if first two columns look like common headers
                    if any(h_word in row_vals[0].lower() for h_word in ['name', 'employee', 'product', 'id']) and \
                       any(h_word in row_vals[1].lower() for h_word in ['salary', 'id', 'dept', 'price', 'qty', 'value']):
                        potential_headers = row_vals
                        st.info(f"Inferred potential headers from raw text: {potential_headers}")
                        continue
                
                if potential_headers and len(row_vals) == len(potential_headers):
                    data_from_text.append(dict(zip(potential_headers, row_vals)))

        if data_from_text:
            try:
                df_from_text = pd.DataFrame(data_from_text)
                if df_from_text.empty:
                    st.warning("DataFrame created from heuristic raw text parsing is empty.")
                    return None

                df_from_text.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col.replace(' ', '_')) for col in df_from_text.columns]
                # Ensure unique column names for the heuristic DF as well
                seen = {}
                final_cols = []
                for h in df_from_text.columns:
                    if h in seen:
                        new_h = f"{h}_{seen[h]}"
                        seen[h] += 1
                        final_cols.append(new_h)
                    else:
                        seen[h] = 1
                        final_cols.append(h)
                df_from_text.columns = final_cols

                for col in df_from_text.columns:
                    if isinstance(col, str) and any(kw in col.lower() for kw in ['salary', 'id', 'count', 'amount', 'value', 'price', 'quantity', 'age', 'total', 'score']):
                        df_from_text[col] = pd.to_numeric(df_from_text[col], errors='coerce')

                st.success(f"Successfully parsed **{len(df_from_text)}** rows from raw text heuristically into a DataFrame.")
                return df_from_text
            except Exception as e:
                st.error(f"Error parsing raw text into DataFrame (heuristic method): {e}.")
                return None
            
    return None

def chunk_text(text):
    """Splits the given text into smaller, manageable chunks."""
    if not text:
        return []

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=100,
        length_function=len
    )
    return splitter.split_text(text)

def create_vectorstore(chunks):
    """Creates a FAISS vector store from text chunks using HuggingFace embeddings."""
    if not chunks:
        st.error("No text chunks available to create a vector store. Vector store will not be initialized.")
        return None

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        st.success(f"Vector store created with **{len(chunks)}** chunks.")
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {e}.")
        return None

def setup_database(df):
    """Sets up an in-memory SQLite database and loads the DataFrame into it.
    Stores connection and cursor in session_state for persistence.
    """
    
    # Close any existing connection to ensure a fresh start
    if st.session_state.conn:
        try:
            st.session_state.conn.close()
        except sqlite3.Error as e:
            st.warning(f"Error closing previous database connection: {e}")

    # CRITICAL CHANGE: Add check_same_thread=False for Streamlit compatibility
    st.session_state.conn = sqlite3.connect(":memory:", check_same_thread=False)
    st.session_state.cursor = st.session_state.conn.cursor()

    table_name = "documents_data" # Fixed table name for consistency
    
    try:
        # Sanitize column names for SQL compatibility (already done in process_data_to_dataframe for excels, but safety for others)
        df.columns = [re.sub(r'[^a-zA-Z0-9_]', '', col.replace(' ', '_')) for col in df.columns]
        # Ensure column names are unique after sanitization (already done in process_data_to_dataframe for excels, but safety for others)
        seen = {}
        new_columns = []
        for col in df.columns:
            temp_col = col
            count = 1
            while temp_col in seen:
                temp_col = f"{col}_{count}"
                count += 1
            seen[temp_col] = True
            new_columns.append(temp_col)
        df.columns = new_columns

        ### --- ADDED DEBUGGING OUTPUT HERE --- ###
        st.subheader("📊 Debug: Final Processed DataFrame for SQL")
        st.write("Columns in the database table (after sanitization):")
        st.code(str(df.columns.tolist()), language="python") # Show column names as list
        st.write("First 5 rows of the DataFrame loaded into SQL:")
        st.dataframe(df.head()) # Show a sample of the data
        ### --- END ADDED DEBUGGING OUTPUT --- ###


        df.to_sql(table_name, st.session_state.conn, if_exists="replace", index=False)
        st.success(f"DataFrame loaded into SQLite table **'{table_name}'** for SQL querying.")
        st.info(f"Table Schema (first 5 columns): {', '.join(df.columns[:5])}" + ("..." if len(df.columns) > 5 else ""))
        return True
    except Exception as e:
        st.error(f"Error loading DataFrame to SQLite database: {e}. SQL querying may not work correctly.")
        st.session_state.conn = None
        st.session_state.cursor = None
        return False

def execute_sql_query(sql_query):
    """Executes a SQL query against the in-memory SQLite database using session_state connection."""
    if not st.session_state.conn or not st.session_state.cursor:
        st.error("Database connection not established. Please ensure documents were processed successfully and structured data was found.")
        return None, None

    try:
        st.session_state.cursor.execute(sql_query)
        results = st.session_state.cursor.fetchall()
        column_names = [description[0] for description in st.session_state.cursor.description]
        return results, column_names
    except sqlite3.Error as e:
        st.error(f"SQL execution error: **{e}**. The generated SQL might be incorrect or columns/table names do not match the database schema. Ensure table name is `documents_data`.")
        st.code(f"Failed Query: {sql_query}")
        return None, None
    except Exception as e:
        st.error(f"An unexpected error occurred during SQL execution: {e}")
        return None, None

def build_qa_chain(vectorstore, k_retrieved_docs, table_name="documents_data"):
    """Builds a RetrievalQA chain using Google's Gemini Flash model with a flexible prompt."""
    if not vectorstore:
        st.error("Vector store is not available to build the QA chain. RAG functionality will be limited.")
        return None

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    # --- UPDATED PROMPT TEMPLATE ---
    prompt_template = f"""You are a helpful assistant that answers questions based on the provided document context.
    The documents contain information, which, if structured, is available in a conceptual SQLite table named '{table_name}'.
    **IMPORTANT:** Column names in this table have been sanitized to replace spaces with underscores (e.g., 'Employee ID' becomes 'Employee_ID') and ensure uniqueness.
    The columns in this table are inferred from the document content (e.g., Name, Salary, Department, Employee_ID, Product_Name, Quantity, Date, etc.).
    Your goal is to provide accurate and concise answers.

    If the user asks a question that clearly requires querying structured data (e.g., 'highest salary', 'count of employees', 'items in stock', 'sum of sales in Q1', 'list all employees'),
    and you believe the necessary columns exist in the '{table_name}' table, attempt to generate a valid SQLite SQL query for that table.
    When generating SQL, you MUST prepend it with '```sql' and append with '```'. Do not include any other text outside these markers if you generate SQL.
    **Ensure you use the sanitized column names (e.g., 'Employee_ID' instead of 'Employee ID') in your SQL queries.**

    If the user asks a question that can be answered directly from the provided text context without needing a structured database query (e.g., 'What is the company mission?', 'Explain the new policy'), answer it directly and concisely.
    If you cannot find the answer or generate appropriate SQL from the provided context, respond with 'I don't have enough information from the documents to answer that.'

    Question: {{question}}
    Context: {{context}}
    Answer:"""
    # --- END UPDATED PROMPT TEMPLATE ---

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": k_retrieved_docs}),
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def handle_user_input(user_question):
    """Handles the user's question, sends it to the LLM, and processes the response."""
    if not st.session_state.conversation:
        st.error("Please upload and process documents first before asking questions. The AI model is not ready yet.")
        return

    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        try:
            llm_response = st.session_state.conversation.run(user_question)
            sql_match = re.search(r"```sql\s*(.*?)\s*```", llm_response, re.DOTALL)

            if sql_match and st.session_state.dataframe_loaded:
                generated_sql = sql_match.group(1).strip()
                st.info(f"Generated SQL: `{generated_sql}`")
                
                results, column_names = execute_sql_query(generated_sql)

                if results is not None and column_names is not None:
                    if results:
                        result_df = pd.DataFrame(results, columns=column_names)
                        st.write(bot_template.replace("{{MSG}}", f"Here's the result from the query:"), unsafe_allow_html=True)
                        st.dataframe(result_df)
                    else:
                        st.write(bot_template.replace("{{MSG}}", "The SQL query executed successfully, but returned no results from the database."), unsafe_allow_html=True)
                else:
                    st.write(bot_template.replace("{{MSG}}", "I tried to generate and execute a SQL query, but it failed. Please check the generated SQL or the structured data in your documents. My original thought was: <br>" + llm_response), unsafe_allow_html=True)
            else:
                # If no SQL was generated or if SQL execution is not enabled/failed, display direct LLM response
                st.write(bot_template.replace("{{MSG}}", llm_response), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An unexpected error occurred during handling your question: {e}")
            st.info("Please try re-processing the documents, adjust 'k' in the sidebar, or ask a different question.")

def main():
    st.set_page_config(page_title="Multi-Doc Chatbot with SQL ✨", page_icon="📄", layout="wide")
    st.write(css, unsafe_allow_html=True)

    # Initialize session state variables
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "raw_text_debug" not in st.session_state:
        st.session_state.raw_text_debug = ""
    if "chunks_debug" not in st.session_state:
        st.session_state.chunks_debug = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "k_retrieved_docs" not in st.session_state:
        st.session_state.k_retrieved_docs = 4
    if "dataframe_loaded" not in st.session_state:
        st.session_state.dataframe_loaded = False
    
    if "conn" not in st.session_state:
        st.session_state.conn = None
    if "cursor" not in st.session_state:
        st.session_state.cursor = None

    st.title("🧠 Multi-Document Chatbot with SQL Capabilities")
    st.subheader("Upload various document types (.pdf, .txt, .docx, .xlsx, .xls), then ask questions to get direct answers or SQL results!")

    user_question = st.text_input("Ask a question about your documents:", placeholder="e.g., How many employees are there? What is John Doe's salary? Summarize the main points.")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.header("Upload Documents 📑")
        uploaded_files = st.file_uploader(
            "Upload multiple documents",
            accept_multiple_files=True,
            type=["pdf", "txt", "docx", "xlsx", "xls"],
            help="Supported formats: PDF, TXT, DOCX, XLSX, XLS. For best SQL results, use documents with clear tabular data."
        )
        
        st.markdown("---")
        st.subheader("Debugging & Configuration Options")
        
        show_raw_text = st.checkbox("Show Raw Extracted Text", value=False, help="Display the full text extracted from all uploaded documents (truncated).")
        show_chunks = st.checkbox("Show First Few Chunks", value=False, help="Display a sample of the text chunks created from the raw text, along with the total chunk count.")
        
        st.session_state.k_retrieved_docs = st.slider(
            "Number of retrieved documents (k)",
            min_value=1,
            max_value=20,
            value=st.session_state.k_retrieved_docs,
            help="This controls how many relevant document chunks are passed to the AI model for answering questions."
        )

        if st.button("Process Documents"):
            if not uploaded_files:
                st.warning("Please upload at least one document to process before clicking 'Process Documents'.")
                return

            with st.spinner("Reading & Indexing Documents..."):
                try:
                    raw_text, extracted_tables_as_lists, excel_dfs = extract_text_from_documents(uploaded_files)
                    st.session_state.raw_text_debug = raw_text
                    
                    if not raw_text:
                        st.error("No usable text could be extracted from the uploaded documents. Cannot proceed.")
                        st.session_state.dataframe_loaded = False
                        if st.session_state.conn: st.session_state.conn.close()
                        st.session_state.conn = None
                        st.session_state.cursor = None
                        return

                    # Process structured data first
                    df = process_data_to_dataframe(raw_text, extracted_tables_as_lists, excel_dfs)
                    
                    if df is not None and not df.empty:
                        st.session_state.dataframe_loaded = setup_database(df)
                    else:
                        st.session_state.dataframe_loaded = False
                        st.warning("No structured data could be processed into a DataFrame for SQL querying. The chatbot will rely *only* on direct text answers for all questions.")
                        if st.session_state.conn: st.session_state.conn.close()
                        st.session_state.conn = None
                        st.session_state.cursor = None

                    # Process unstructured text chunks
                    chunks = chunk_text(raw_text)
                    st.session_state.chunks_debug = chunks
                    if not chunks:
                        st.error("No text chunks could be created from the extracted text. The AI model will not be able to answer questions.")
                        st.session_state.conversation = None # Cannot build conversation without chunks
                        return
                    
                    vectorstore = create_vectorstore(chunks)
                    st.session_state.vectorstore = vectorstore
                    if not vectorstore:
                        st.session_state.conversation = None # Cannot build conversation without vectorstore
                        return

                    # Build the RAG/QA chain
                    st.session_state.conversation = build_qa_chain(vectorstore, st.session_state.k_retrieved_docs, table_name="documents_data")
                    
                    if st.session_state.conversation:
                        st.success("Documents processed successfully! The AI model is ready. You can now ask questions.")
                        if st.session_state.dataframe_loaded:
                            st.info("SQL execution is **enabled** for questions that can be answered from structured data.")
                        else:
                            st.info("SQL execution is **NOT enabled** as no structured data was found/processed.")
                    else:
                        st.error("Failed to build the conversational AI. Please check API keys and document content.")

                except Exception as e:
                    st.error(f"An unexpected error occurred during document processing: {e}")
                    st.info("Please review the error details, check your input files, and ensure all necessary API keys are correctly set up.")
                    if st.session_state.conn: st.session_state.conn.close()
                    st.session_state.conn = None
                    st.session_state.cursor = None
            
        if st.session_state.vectorstore:
            st.markdown("---")
            st.subheader("Manual Retrieval Test")
            st.info("Enter a query to see which document chunks are retrieved by the RAG system.")
            test_query_input = st.text_input("Query for retrieval test:", key="retrieval_test_query_sidebar")
            
            if st.button("Run Retrieval Test", key="run_retrieval_test_button"):
                if test_query_input:
                    with st.spinner("Running retrieval test..."):
                        try:
                            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": st.session_state.k_retrieved_docs})
                            retrieved_docs = retriever.invoke(test_query_input) # Use .invoke() for newer Langchain versions
                            
                            st.subheader(f"Top **{len(retrieved_docs)}** Retrieved Documents for '{test_query_input}':")
                            if retrieved_docs:
                                for i, doc in enumerate(retrieved_docs):
                                    st.markdown(f"**Document Chunk {i+1}**")
                                    st.code(doc.page_content, language="text")
                                    st.markdown("---")
                            else:
                                st.info("No relevant documents found for this query based on the current 'k' value.")
                        except Exception as e:
                            st.error(f"Error during retrieval test: {e}.")
                else:
                    st.warning("Please enter a query in the 'Query for retrieval test:' box to run the test.")

    # Debugging information display
    if st.session_state.raw_text_debug and show_raw_text:
        st.markdown("---")
        st.subheader("🔎 Raw Extracted Text Sample (first 2000 characters)")
        st.text_area("Raw Text Content", st.session_state.raw_text_debug[:2000] + ("..." if len(st.session_state.raw_text_debug) > 2000 else ""), height=300, disabled=True)
        st.info(f"Total characters extracted: **{len(st.session_state.raw_text_debug)}**")

    if st.session_state.chunks_debug and show_chunks:
        st.markdown("---")
        st.subheader("🔎 First 3 Chunks (and total count)")
        for i, chunk in enumerate(st.session_state.chunks_debug[:3]):
            st.code(f"Chunk {i+1}:\n{chunk}", language="text")
        if len(st.session_state.chunks_debug) > 3:
            st.info(f"... **{len(st.session_state.chunks_debug) - 3}** more chunks exist. **Total chunks: {len(st.session_state.chunks_debug)}**")

if __name__ == "__main__":
    main()