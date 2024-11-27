import re
import os
import json
from io import BytesIO
from typing import Any, Dict, List, Optional, Awaitable, Callable, Tuple, Type, Union
import requests
import asyncio

from collections import OrderedDict
import base64
import docx2txt
import tiktoken
import html
import time
from time import sleep
from typing import List, Tuple
from pypdf import PdfReader, PdfWriter
from dataclasses import dataclass
from sqlalchemy.engine.url import URL
from azure.core.credentials import AzureKeyCredential

from langchain.callbacks.manager import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain.pydantic_v1 import BaseModel, Field, Extra
from langchain.tools import BaseTool, StructuredTool, tool
from typing import Dict, List
from concurrent.futures import ThreadPoolExecutor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import BaseOutputParser, OutputParserException
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.tools import BaseTool, StructuredTool, tool
from langchain.prompts import PromptTemplate
from langchain.sql_database import SQLDatabase
from langchain.agents import AgentExecutor, initialize_agent, AgentType, Tool
from langchain_community.utilities import BingSearchAPIWrapper
from langchain.agents import create_sql_agent, create_openai_tools_agent
from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.callbacks.base import BaseCallbackManager
from langchain.requests import RequestsWrapper
from langchain.chains import APIChain
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.utils.json_schema import dereference_refs
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from operator import itemgetter
from typing import List



try:
    from .prompts import (AGENT_DOCSEARCH_PROMPT, MSSQL_AGENT_PREFIX)
except Exception as e:
    print(e)
    from prompts import (AGENT_DOCSEARCH_PROMPT, MSSQL_AGENT_PREFIX)

def get_search_results(query: str, indexes: list, 
                       k: int = 10,
                       reranker_threshold: int = 1,
                       sas_token: str = "") -> List[dict]:
    """Performs multi-index hybrid search and returns ordered dictionary with the combined results"""
    
    headers = {'Content-Type': 'application/json','api-key': os.environ["AZURE_SEARCH_KEY"]}
    params = {'api-version': os.environ['AZURE_SEARCH_API_VERSION']}

    pattern = r'\b(QIS|CAPA)-\d{4}-\d+\b'
    query = re.sub(pattern, r'"\g<0>"', query)

    agg_search_results = dict()
    
    for index in indexes:
        search_payload = {
            "search": query,
            "select": "id, title, content, url",
            "queryType": "semantic",
            "vectorQueries": [{"text": query, "fields": "contentVector", "kind": "text", "k": k}],
            "semanticConfiguration": "my-semantic-config",
            "captions": "extractive",
            "answers": "extractive",
            "count":"true",
            "top": k    
        }

        resp = requests.post(os.environ['AZURE_SEARCH_ENDPOINT'] + "/indexes/" + index + "/docs/search",
                         data=json.dumps(search_payload), headers=headers, params=params)

        search_results = resp.json()
        agg_search_results[index] = search_results
    
    content = dict()
    ordered_content = OrderedDict()
    
    for index,search_results in agg_search_results.items():
        for result in search_results['value']:
            if result['@search.rerankerScore'] > reranker_threshold: # Show results that are at least N% of the max possible score=4
                content[result['id']]={
                    "title": result['title'],
                    "content": result['content'],
                    "url": result['url'] + sas_token if result['url'] else "",
                    "caption": result['@search.captions'][0]['text'],
                    "score": result['@search.rerankerScore'],
                    "index": index
                }
                

    topk = k
        
    count = 0  # To keep track of the number of results added
    for id in sorted(content, key=lambda x: content[x]["score"], reverse=True):
        ordered_content[id] = content[id]
        count += 1
        if count >= topk:  # Stop after adding topK results
            break

    return ordered_content



class CustomAzureSearchRetriever(BaseRetriever):
    
    indexes: List
    topK : int
    reranker_threshold : int
    sas_token : str = ""
    
    
    def _get_relevant_documents(
        self, input: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        
        ordered_results = get_search_results(input, self.indexes, k=self.topK, reranker_threshold=self.reranker_threshold, sas_token=self.sas_token)
        
        top_docs = []
        for key,value in ordered_results.items():
            url = value["url"] if value["url"] is not None else ""
            top_docs.append(Document(page_content=value["content"], metadata={"source": url, "score":value["score"]}))

        return top_docs

    
def get_answer(llm: AzureChatOpenAI,
               retriever: CustomAzureSearchRetriever, 
               query: str,
               memory: ConversationBufferMemory = None
              ) -> Dict[str, Any]:
    
    """Gets an answer to a question from a list of Documents."""

    # Get the answer
        
    chain = (
        {
            "context": itemgetter("question") | retriever, # Passes the question to the retriever and the results are assign to context
            "question": itemgetter("question")
        }
        | AGENT_DOCSEARCH_PROMPT  # Passes the 4 variables above to the prompt template
        | llm   # Passes the finished prompt to the LLM
        | StrOutputParser()  # converts the output (Runnable object) to the desired output (string)
    )
    
    answer = chain.invoke({"question": query})

    return answer

    

#####################################################################################################
############################### AGENTS AND TOOL CLASSES #############################################
#####################################################################################################
    
class SearchInput(BaseModel):
    query: str = Field(description="should be a search query")
    return_direct: bool = Field(
        description="Whether or the result of this should be returned directly to the user without you seeing what it is",
        default=False,
    )

class GetDocSearchResults_Tool(BaseTool):
    name = "docsearch"
    description = "useful when the questions includes the term: docsearch"
    args_schema: Type[BaseModel] = SearchInput
    
    indexes: List[str] = []
    k: int = 10
    reranker_th: int = 1
    sas_token: str = "" 

    def _run(
        self, query: str,  return_direct = False, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:

        retriever = CustomAzureSearchRetriever(indexes=self.indexes, topK=self.k, reranker_threshold=self.reranker_th, 
                                               sas_token=self.sas_token, callback_manager=self.callbacks)
        results = retriever.invoke(input=query)
        
        return results

    async def _arun(
        self, query: str, return_direct = False, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        
        retriever = CustomAzureSearchRetriever(indexes=self.indexes, topK=self.k, reranker_threshold=self.reranker_th, 
                                               sas_token=self.sas_token, callback_manager=self.callbacks)
        # Please note below that running a non-async function like run_agent in a separate thread won't make it truly asynchronous. 
        # It allows the function to be called without blocking the event loop, but it may still have synchronous behavior internally.
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(ThreadPoolExecutor(), retriever.invoke, query)
        
        return results


class DocSearchAgent(BaseTool):
    """Agent to interact with for Azure AI Search """
    
    name = "docsearch"
    description = "useful when the questions includes the term: docsearch.\n"
    args_schema: Type[BaseModel] = SearchInput

    llm: AzureChatOpenAI
    indexes: List[str] = []
    k: int = 10
    reranker_th: int = 1
    sas_token: str = ""   
    
    class Config:
        extra = Extra.allow  # Allows setting attributes not declared in the model
    
    def __init__(self, **data):
        super().__init__(**data)
        tools = [GetDocSearchResults_Tool(indexes=self.indexes, k=self.k, reranker_th=self.reranker_th, sas_token=self.sas_token)]

        agent = create_openai_tools_agent(self.llm, tools, AGENT_DOCSEARCH_PROMPT)

        self.agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=self.verbose, callback_manager=self.callbacks, handle_parsing_errors=True)
        
    
    def _run(self, query: str,  return_direct = False, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            result = self.agent_executor.invoke({"question": query})
            return result['output']
        except Exception as e:
            print(e)
            return str(e)  # Return an empty string or some error indicator

    async def _arun(self, query: str,  return_direct = False, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        try:
            result = await self.agent_executor.ainvoke({"question": query})
            return result['output']
        except Exception as e:
            print(e)
            return str(e)  # Return an empty string or some error indicator
        

class SQLSearchAgent(BaseTool):
    """Agent to interact with SQL database"""
    
    name = "sqlsearch"
    description = "useful when the questions includes the term: sqlsearch.\n"
    args_schema: Type[BaseModel] = SearchInput

    llm: AzureChatOpenAI
    k: int = 10

    class Config:
        extra = Extra.allow  # Allows setting attributes not declared in the model

    def __init__(self, **data):
        super().__init__(**data)
        db_config = self.get_db_config()
        db_url = URL.create(**db_config)
        db = SQLDatabase.from_uri(db_url, schema = "qis", view_support = True,)
        toolkit = SQLDatabaseToolkit(db=db, llm=self.llm)

        self.agent_executor = create_sql_agent(
            prefix=MSSQL_AGENT_PREFIX,
            llm=self.llm,
            toolkit=toolkit,
            top_k=self.k,
            agent_type="openai-tools",
            callback_manager=self.callbacks,
            verbose=self.verbose,
        )

    def get_db_config(self):
        """Returns the database configuration."""
        return {
            'drivername': 'postgresql+psycopg2',
            'username': os.environ["SQL_SERVER_USERNAME"],
            'password': os.environ["SQL_SERVER_PASSWORD"],
            'host': os.environ["SQL_SERVER_NAME"],
            'port': 5432,
            'database': os.environ["SQL_SERVER_DATABASE"]
        }

    def _run(self, query: str, return_direct = False, run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
        try:
            # Use the initialized agent_executor to invoke the query
            result = self.agent_executor.invoke(query)
            return result['output']
        except Exception as e:
            print(e)
            return str(e)  # Return an error indicator

    async def _arun(self, query: str, return_direct = False, run_manager: Optional[AsyncCallbackManagerForToolRun] = None) -> str:
        # Note: Implementation assumes the agent_executor and its methods support async operations
        try:
            # Use the initialized agent_executor to asynchronously invoke the query
            result = await self.agent_executor.ainvoke(query)
            return result['output']
        except Exception as e:
            print(e)
            return str(e)  # Return an error indicator