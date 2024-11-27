import copy
import json
import time
import os
import logging
import uuid
import httpx
import asyncio
import uvicorn
import identity.web
from quart import (
    Blueprint,
    Quart,
    jsonify,
    make_response,
    request,
    send_from_directory,
    render_template,
    current_app,
    session,
    redirect,
    url_for
)
# from quart_session import Session
from flask_session import Session
from cachelib.file import FileSystemCache
from werkzeug.middleware.proxy_fix import ProxyFix

from openai import AsyncAzureOpenAI
from azure.identity.aio import (
    DefaultAzureCredential,
    get_bearer_token_provider
)
from backend.auth.auth_utils import get_authenticated_user_details
from backend.security.ms_defender_utils import get_msdefender_user_json
from backend.history.cosmosdbservice import CosmosConversationClient
from backend.settings import (
    app_settings,
    MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION
)
from backend.utils import (
    format_as_ndjson,
    format_stream_response
)

import os
import sys
from operator import itemgetter
from datetime import datetime
from typing import Any, Dict, TypedDict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from langserve import add_routes
from langchain.pydantic_v1 import BaseModel

from dotenv import load_dotenv
load_dotenv()

REDIRECT_PATH = os.environ["REDIRECT_PATH"]
GROUP_ID = os.environ["GROUP_ID"]
MICROSOFT_FORM_URL = os.environ["FORM_URL"]

fastapi_app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

main_bp = Blueprint("routes", __name__, static_folder="static", template_folder="static")

auth_bp = Blueprint("auth", __name__, static_folder="auth_templates/assets", template_folder="auth_templates")

cosmos_db_ready = asyncio.Event()

############################  Langchain  #################################################
##########################################################################################
import random
import requests
from operator import itemgetter
from typing import Union, List
from langchain_openai import AzureChatOpenAI
from langchain.agents import AgentExecutor, Tool, create_openai_tools_agent
from langchain_community.chat_message_histories import ChatMessageHistory, CosmosDBChatMessageHistory
from langchain.callbacks.manager import CallbackManager
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec, ConfigurableField
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import JsonOutputToolsParser
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
)
from langserve import RemoteRunnable

base_url = os.environ["LANGSERVE_BASE_URL"] # If you deployed locally
agent_chain = RemoteRunnable(base_url + "/agent/")

#custom libraries that we will use later in the app
from common.utils import (
    DocSearchAgent,
    SQLSearchAgent
)
from common.callbacks import StdOutCallbackHandler
from common.prompts import CUSTOM_CHATBOT_PROMPT 

cb_handler = StdOutCallbackHandler()
cb_manager = CallbackManager(handlers=[cb_handler])

COMPLETION_TOKENS = 4000
llm = AzureChatOpenAI(deployment_name=os.environ["AZURE_OPENAI_MODEL"], api_key=os.environ["AZURE_OPENAI_API_KEY"], temperature=0, max_tokens=COMPLETION_TOKENS, streaming=True, callback_manager=cb_manager, api_version="2024-05-01-preview")

## Azure AI Search Agent
doc_indexes = ["qa1-index-allqis-index"]
doc_search = DocSearchAgent(llm=llm, indexes=doc_indexes,
                k=15, reranker_th=1,
                sas_token="",
                name="docsearch",
                description="useful when the questions includes the term: docsearch",
                callback_manager=cb_manager, verbose=False)

sql_search = SQLSearchAgent(llm=llm, k=10, callback_manager=cb_manager,
                name="sqlsearch",
                description="useful when the questions includes the term: sqlsearch",
                verbose=False)

# Multi Agent
tools = [doc_search, sql_search]
agent = create_openai_tools_agent(llm, tools, CUSTOM_CHATBOT_PROMPT)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def get_session_history(session_id: str, user_id: str) -> CosmosDBChatMessageHistory:
    cosmos = CosmosDBChatMessageHistory(
        cosmos_endpoint=os.environ['LANGCHAIN_AZURE_COSMOSDB_ENDPOINT'],
        cosmos_database=os.environ['LANGCHAIN_AZURE_COSMOSDB_NAME'],
        cosmos_container=os.environ['LANGCHAIN_AZURE_COSMOSDB_CONTAINER_NAME'],
        connection_string=os.environ['LANGCHAIN_AZURE_COSMOSDB_CONNECTION_STRING'],
        session_id=session_id,
        user_id=user_id
        )
    # prepare the cosmosdb instance
    cosmos.prepare_cosmos()
    return cosmos

brain_agent_executor = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)

############################  Langchain End  #############################################
##########################################################################################

def create_app():  
    app = Quart(__name__)
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.config["TEMPLATES_AUTO_RELOAD"] = True
    app.config['SESSION_TYPE'] = 'filesystem'
    Session(app)
    async def init():
        try:
            app.cosmos_conversation_client = await init_cosmosdb_client()
            cosmos_db_ready.set()
        except Exception as e:
            logging.exception("Failed to initialize CosmosDB client")
            app.cosmos_conversation_client = None
            raise e 
  
    return app, init  

async def init_openai_client():
    azure_openai_client = None
    
    try:
        # API version check
        if (
            app_settings.azure_openai.preview_api_version
            < MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION
        ):
            raise ValueError(
                f"The minimum supported Azure OpenAI preview API version is '{MINIMUM_SUPPORTED_AZURE_OPENAI_PREVIEW_API_VERSION}'"
            )

        # Endpoint
        if (
            not app_settings.azure_openai.endpoint and
            not app_settings.azure_openai.resource
        ):
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_RESOURCE is required"
            )

        endpoint = (
            app_settings.azure_openai.endpoint
            if app_settings.azure_openai.endpoint
            else f"https://{app_settings.azure_openai.resource}.openai.azure.com/"
        )

        # Authentication
        aoai_api_key = app_settings.azure_openai.api_key
        ad_token_provider = None
        if not aoai_api_key:
            logging.debug("No AZURE_OPENAI_KEY found, using Azure Entra ID auth")
            async with DefaultAzureCredential() as credential:
                ad_token_provider = get_bearer_token_provider(
                    credential,
                    "https://cognitiveservices.azure.com/.default"
                )

        # Deployment
        deployment = app_settings.azure_openai.model
        if not deployment:
            raise ValueError("AZURE_OPENAI_MODEL is required")

        # Default Headers
        default_headers = {"x-ms-useragent": USER_AGENT}

        azure_openai_client = AsyncAzureOpenAI(
            api_version=app_settings.azure_openai.preview_api_version,
            api_key=aoai_api_key,
            azure_ad_token_provider=ad_token_provider,
            default_headers=default_headers,
            azure_endpoint=endpoint,
        )

        return azure_openai_client
    except Exception as e:
        logging.exception("Exception in Azure OpenAI initialization", e)
        azure_openai_client = None
        raise e
    
__version__ = "0.8.0"
@auth_bp.route("/")
async def index():
    # if not auth.get_user() and os.environ["ENABLE_AUTH"].lower()=="true":
    if not auth.get_user() and app_settings.base_settings.enable_auth:
        return redirect(url_for("auth.login"))
    # return await render_template('index_auth.html', user=auth.get_user(), version=__version__)
    return redirect(url_for("routes.index_home"))

@auth_bp.route("/auth_assets/<path:path>")
async def auth_assets(path):
    return await send_from_directory("auth_templates/auth_assets", path)

@auth_bp.route("/login")
async def login():
    return await render_template("login.html", version=__version__, **auth.log_in(
        scopes=["User.Read"], # Have user consent to scopes during log-in
        redirect_uri=url_for("auth.auth_response", _external=True), # Optional. If present, this absolute URL must match your app's redirect_uri registered in Azure Portal
        prompt="select_account",  # Optional. More values defined in  https://openid.net/specs/openid-connect-core-1_0.html#AuthRequest
        ))

@auth_bp.route("/logout")
def logout():
    return redirect(auth.log_out(url_for("auth.index", _external=True)))

@auth_bp.route(REDIRECT_PATH)
def auth_response(): 
    result = auth.complete_log_in(request.args)
    # check for access
    if GROUP_ID in result['groups']:
        session["user_authenticated"] = True
        session['user_principal_id'] = result['oid']
        session['user_name'] = result['preferred_username']
        return redirect(url_for("routes.index_home"))
    else:
        # no access
        return redirect(MICROSOFT_FORM_URL)

# @main_bp.before_request
# async def check_user_auth():
#     if "user_authenticated" not in session and request.referrer:
#         # return await render_template("login.html", version=__version__, **auth.log_in(
#         # scopes=["User.Read"], # Have user consent to scopes during log-in
#         # redirect_uri=url_for("auth.auth_response", _external=True), # Optional. If present, this absolute URL must match your app's redirect_uri registered in Azure Portal
#         # prompt="select_account",  # Optional. More values defined in  https://openid.net/specs/openid-connect-core-1_0.html#AuthRequest
#         # ))
#         # return redirect(url_for("auth.login"))

@main_bp.route("/home")
async def index_home():
    return await render_template(
        "index.html",
        title=app_settings.ui.title,
        favicon=app_settings.ui.favicon
    )


@main_bp.route("/favicon.ico")
async def favicon():
    return await main_bp.send_static_file("favicon.ico")


@main_bp.route("/assets/<path:path>")
async def assets(path):
    return await send_from_directory("static/assets", path)


# Debug settings
DEBUG = os.environ.get("DEBUG", "false")
if DEBUG.lower() == "true":
    logging.basicConfig(level=logging.DEBUG)
logging.getLogger('azure.core.pipeline.policies.http_logging_policy').setLevel(logging.WARNING)

USER_AGENT = "GitHubSampleWebApp/AsyncAzureOpenAI/1.0.0"


# Frontend Settings via Environment Variables
frontend_settings = {
    "auth_enabled": app_settings.base_settings.auth_enabled,
    "feedback_enabled": (
        app_settings.chat_history and
        app_settings.chat_history.enable_feedback
    ),
    "ui": {
        "title": app_settings.ui.title,
        "logo": app_settings.ui.logo,
        "chat_logo": app_settings.ui.chat_logo or app_settings.ui.logo,
        "chat_title": app_settings.ui.chat_title,
        "chat_description": app_settings.ui.chat_description,
        "show_share_button": app_settings.ui.show_share_button,
        "show_chat_history_button": app_settings.ui.show_chat_history_button,
    },
    "sanitize_answer": app_settings.base_settings.sanitize_answer,
    "oyd_enabled": app_settings.base_settings.datasource_type,
}


async def init_cosmosdb_client():
    cosmos_conversation_client = None
    if app_settings.chat_history:
        try:
            cosmos_endpoint = (
                f"https://{app_settings.chat_history.account}.documents.azure.com:443/"
            )

            if not app_settings.chat_history.account_key:
                async with DefaultAzureCredential() as cred:
                    credential = cred
                    
            else:
                credential = app_settings.chat_history.account_key

            cosmos_conversation_client = CosmosConversationClient(
                cosmosdb_endpoint=cosmos_endpoint,
                credential=credential,
                database_name=app_settings.chat_history.database,
                container_name=app_settings.chat_history.conversations_container,
                enable_message_feedback=app_settings.chat_history.enable_feedback,
            )
        except Exception as e:
            logging.exception("Exception in CosmosDB initialization", e)
            cosmos_conversation_client = None
            raise e
    else:
        logging.debug("CosmosDB not configured")

    return cosmos_conversation_client

async def send_chat_request(request_body, request_headers):
    filtered_messages = []
    messages = request_body.get("messages", [])
    for message in messages:
        if message.get("role") != 'tool':
            filtered_messages.append(message)
            
    request_body['messages'] = filtered_messages
    # Langchain
    # authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    # user_id = authenticated_user["user_principal_id"]
    user_id = session['user_principal_id']
    
    try:
        # Langchain
        question = request_body["messages"][-1]["content"]
        if "conversation_id" in request_body:
            session_id = request_body["conversation_id"]
        elif "conversation_id" in request_body["history_metadata"]:
            session_id = request_body["history_metadata"]["conversation_id"]
        else:
            session_id = "0000000000"
        config={"configurable": {"session_id": session_id, "user_id": user_id}}
        response = agent_chain.astream_events({"question": question}, config=config, version="v1", include_names=['AzureChatOpenAI', 'docsearch', 'sqlsearch']) 
        apim_request_id = None
    except Exception as e:
        logging.exception("Exception in send_chat_request")
        raise e

    return response, apim_request_id

async def stream_chat_request(request_body, request_headers):
    response, apim_request_id = await send_chat_request(request_body, request_headers)
    history_metadata = request_body.get("history_metadata", {})
    async def generate():
        start = False
        tool = False
        check = False
        i = 0
        async for event in response:

            # Get the tool name
            if event["event"] == 'on_chat_model_stream' and not check:
                if event['data']['chunk'].additional_kwargs:
                    tool = event['data']['chunk'].additional_kwargs['tool_calls'][0]['function']['name']
                check = True
            
            # Stream answer immediately if it is a generic query
            if event["event"] == 'on_chat_model_stream' and not tool:
                if not event['data']['chunk'].additional_kwargs:
                    yield format_stream_response(event, history_metadata, apim_request_id)
                        
            # Set 'start' as True based on the number of occurences of 'on_tool_end' event, indicating to start streaming sqlsearch or docsearch based answer
            if event["event"] == "on_tool_end":
                i+=1
                if i==1 and tool=='sqlsearch': start = True
                if i==2 and tool=='docsearch': start = True

            # Stream sqlsearch or docsearch based answer
            if event["event"] == "on_chat_model_stream" and start:
                yield format_stream_response(event, history_metadata, apim_request_id)
    
    return generate()

async def conversation_internal(request_body, request_headers):
    try:
        result = await stream_chat_request(request_body, request_headers)
        response = await make_response(format_as_ndjson(result))
        response.timeout = None
        response.mimetype = "application/json-lines"
        return response

    except Exception as ex:
        logging.exception(ex)
        if hasattr(ex, "status_code"):
            return jsonify({"error": str(ex)}), ex.status_code
        else:
            return jsonify({"error": str(ex)}), 500

class Input(TypedDict):
    question: str

class Output(BaseModel):
    output: Any

add_routes(
    fastapi_app,
    brain_agent_executor.with_types(input_type=Input, output_type=Output),
    path="/agent",
)

@main_bp.route("/conversation", methods=["POST"])
async def conversation():
    if not request.is_json:
        return jsonify({"error": "request must be json"}), 415
    request_json = await request.get_json()

    return await conversation_internal(request_json, request.headers)


@main_bp.route("/frontend_settings", methods=["GET"])
def get_frontend_settings():
    try:
        return jsonify(frontend_settings), 200
    except Exception as e:
        logging.exception("Exception in /frontend_settings")
        return jsonify({"error": str(e)}), 500


## Conversation History API ##
@main_bp.route("/history/generate", methods=["POST"])
async def add_conversation():
    await cosmos_db_ready.wait()
    # authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    # user_id = authenticated_user["user_principal_id"]
    user_id = session['user_principal_id']

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        # make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        # check for the conversation_id, if the conversation is not set, we will create a new one
        history_metadata = {}
        if not conversation_id:
            title = await generate_title(request_json["messages"])
            conversation_dict = await current_app.cosmos_conversation_client.create_conversation(
                user_id=user_id, title=title
            )
            conversation_id = conversation_dict["id"]
            history_metadata["title"] = title
            history_metadata["date"] = conversation_dict["createdAt"]

        ## Format the incoming message object in the "chat/completions" messages format
        ## then write it to the conversation history in cosmos
        messages = request_json["messages"]
        if len(messages) > 0 and messages[-1]["role"] == "user":
            createdMessageValue = await current_app.cosmos_conversation_client.create_message(
                uuid=str(uuid.uuid4()),
                conversation_id=conversation_id,
                user_id=user_id,
                input_message=messages[-1],
            )
            if createdMessageValue == "Conversation not found":
                raise Exception(
                    "Conversation not found for the given conversation ID: "
                    + conversation_id
                    + "."
                )
        else:
            raise Exception("No user message found")

        # Submit request to Chat Completions for response
        request_body = await request.get_json()
        history_metadata["conversation_id"] = conversation_id
        request_body["history_metadata"] = history_metadata
        return await conversation_internal(request_body, request.headers)

    except Exception as e:
        logging.exception("Exception in /history/generate")
        return jsonify({"error": str(e)}), 500


@main_bp.route("/history/update", methods=["POST"])
async def update_conversation():
    await cosmos_db_ready.wait()
    # authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    # user_id = authenticated_user["user_principal_id"]
    user_id = session['user_principal_id']

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        # make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        # check for the conversation_id, if the conversation is not set, we will create a new one
        if not conversation_id:
            raise Exception("No conversation_id found")

        ## Format the incoming message object in the "chat/completions" messages format
        ## then write it to the conversation history in cosmos
        messages = request_json["messages"]
        if len(messages) > 0 and messages[-1]["role"] == "assistant":
            if len(messages) > 1 and messages[-2].get("role", None) == "tool":
                # write the tool message first
                await current_app.cosmos_conversation_client.create_message(
                    uuid=str(uuid.uuid4()),
                    conversation_id=conversation_id,
                    user_id=user_id,
                    input_message=messages[-2],
                )
            # write the assistant message
            await current_app.cosmos_conversation_client.create_message(
                uuid=messages[-1]["id"],
                conversation_id=conversation_id,
                user_id=user_id,
                input_message=messages[-1],
            )
        else:
            raise Exception("No bot messages found")

        # Submit request to Chat Completions for response
        response = {"success": True}
        return jsonify(response), 200

    except Exception as e:
        logging.exception("Exception in /history/update")
        return jsonify({"error": str(e)}), 500


@main_bp.route("/history/message_feedback", methods=["POST"])
async def update_message():
    await cosmos_db_ready.wait()
    # authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    # user_id = authenticated_user["user_principal_id"]
    user_id = session['user_principal_id']

    ## check request for message_id
    request_json = await request.get_json()
    message_id = request_json.get("message_id", None)
    message_feedback = request_json.get("message_feedback", None)
    try:
        if not message_id:
            return jsonify({"error": "message_id is required"}), 400

        if not message_feedback:
            return jsonify({"error": "message_feedback is required"}), 400

        ## update the message in cosmos
        updated_message = await current_app.cosmos_conversation_client.update_message_feedback(
            user_id, message_id, message_feedback
        )
        if updated_message:
            return (
                jsonify(
                    {
                        "message": f"Successfully updated message with feedback {message_feedback}",
                        "message_id": message_id,
                    }
                ),
                200,
            )
        else:
            return (
                jsonify(
                    {
                        "error": f"Unable to update message {message_id}. It either does not exist or the user does not have access to it."
                    }
                ),
                404,
            )

    except Exception as e:
        logging.exception("Exception in /history/message_feedback")
        return jsonify({"error": str(e)}), 500


@main_bp.route("/history/delete", methods=["DELETE"])
async def delete_conversation():
    await cosmos_db_ready.wait()
    ## get the user id from the request headers
    # authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    # user_id = authenticated_user["user_principal_id"]
    user_id = session['user_principal_id']

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400

        ## make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        ## delete the conversation messages from cosmos first
        deleted_messages = await current_app.cosmos_conversation_client.delete_messages(
            conversation_id, user_id
        )

        ## Now delete the conversation
        deleted_conversation = await current_app.cosmos_conversation_client.delete_conversation(
            user_id, conversation_id
        )

        return (
            jsonify(
                {
                    "message": "Successfully deleted conversation and messages",
                    "conversation_id": conversation_id,
                }
            ),
            200,
        )
    except Exception as e:
        logging.exception("Exception in /history/delete")
        return jsonify({"error": str(e)}), 500


@main_bp.route("/history/list", methods=["GET"])
async def list_conversations():
    await cosmos_db_ready.wait()
    offset = request.args.get("offset", 0)
    # authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    # user_id = authenticated_user["user_principal_id"]
    user_id = session['user_principal_id']

    ## make sure cosmos is configured
    if not current_app.cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    ## get the conversations from cosmos
    conversations = await current_app.cosmos_conversation_client.get_conversations(
        user_id, offset=offset, limit=25
    )
    if not isinstance(conversations, list):
        return jsonify({"error": f"No conversations for {user_id} were found"}), 404

    ## return the conversation ids

    return jsonify(conversations), 200


@main_bp.route("/history/read", methods=["POST"])
async def get_conversation():
    await cosmos_db_ready.wait()
    # authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    # user_id = authenticated_user["user_principal_id"]
    user_id = session['user_principal_id']

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400

    ## make sure cosmos is configured
    if not current_app.cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    ## get the conversation object and the related messages from cosmos
    conversation = await current_app.cosmos_conversation_client.get_conversation(
        user_id, conversation_id
    )
    ## return the conversation id and the messages in the bot frontend format
    if not conversation:
        return (
            jsonify(
                {
                    "error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."
                }
            ),
            404,
        )

    # get the messages for the conversation from cosmos
    conversation_messages = await current_app.cosmos_conversation_client.get_messages(
        user_id, conversation_id
    )

    ## format the messages in the bot frontend format
    messages = [
        {
            "id": msg["id"],
            "role": msg["role"],
            "content": msg["content"],
            "createdAt": msg["createdAt"],
            "feedback": msg.get("feedback"),
        }
        for msg in conversation_messages
    ]

    return jsonify({"conversation_id": conversation_id, "messages": messages}), 200


@main_bp.route("/history/rename", methods=["POST"])
async def rename_conversation():
    await cosmos_db_ready.wait()
    # authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    # user_id = authenticated_user["user_principal_id"]
    user_id = session['user_principal_id']

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    if not conversation_id:
        return jsonify({"error": "conversation_id is required"}), 400

    ## make sure cosmos is configured
    if not current_app.cosmos_conversation_client:
        raise Exception("CosmosDB is not configured or not working")

    ## get the conversation from cosmos
    conversation = await current_app.cosmos_conversation_client.get_conversation(
        user_id, conversation_id
    )
    if not conversation:
        return (
            jsonify(
                {
                    "error": f"Conversation {conversation_id} was not found. It either does not exist or the logged in user does not have access to it."
                }
            ),
            404,
        )

    ## update the title
    title = request_json.get("title", None)
    if not title:
        return jsonify({"error": "title is required"}), 400
    conversation["title"] = title
    updated_conversation = await current_app.cosmos_conversation_client.upsert_conversation(
        conversation
    )

    return jsonify(updated_conversation), 200


@main_bp.route("/history/delete_all", methods=["DELETE"])
async def delete_all_conversations():
    await cosmos_db_ready.wait()
    ## get the user id from the request headers
    # authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    # user_id = authenticated_user["user_principal_id"]
    user_id = session['user_principal_id']

    # get conversations for user
    try:
        ## make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        conversations = await current_app.cosmos_conversation_client.get_conversations(
            user_id, offset=0, limit=None
        )
        if not conversations:
            return jsonify({"error": f"No conversations for {user_id} were found"}), 404

        # delete each conversation
        for conversation in conversations:
            ## delete the conversation messages from cosmos first
            deleted_messages = await current_app.cosmos_conversation_client.delete_messages(
                conversation["id"], user_id
            )

            ## Now delete the conversation
            deleted_conversation = await current_app.cosmos_conversation_client.delete_conversation(
                user_id, conversation["id"]
            )
        return (
            jsonify(
                {
                    "message": f"Successfully deleted conversation and messages for user {user_id}"
                }
            ),
            200,
        )

    except Exception as e:
        logging.exception("Exception in /history/delete_all")
        return jsonify({"error": str(e)}), 500


@main_bp.route("/history/clear", methods=["POST"])
async def clear_messages():
    await cosmos_db_ready.wait()
    ## get the user id from the request headers
    # authenticated_user = get_authenticated_user_details(request_headers=request.headers)
    # user_id = authenticated_user["user_principal_id"]
    user_id = session['user_principal_id']

    ## check request for conversation_id
    request_json = await request.get_json()
    conversation_id = request_json.get("conversation_id", None)

    try:
        if not conversation_id:
            return jsonify({"error": "conversation_id is required"}), 400

        ## make sure cosmos is configured
        if not current_app.cosmos_conversation_client:
            raise Exception("CosmosDB is not configured or not working")

        ## delete the conversation messages from cosmos
        deleted_messages = await current_app.cosmos_conversation_client.delete_messages(
            conversation_id, user_id
        )

        return (
            jsonify(
                {
                    "message": "Successfully deleted messages in conversation",
                    "conversation_id": conversation_id,
                }
            ),
            200,
        )
    except Exception as e:
        logging.exception("Exception in /history/clear_messages")
        return jsonify({"error": str(e)}), 500


@main_bp.route("/history/ensure", methods=["GET"])
async def ensure_cosmos():
    await cosmos_db_ready.wait()
    if not app_settings.chat_history:
        return jsonify({"error": "CosmosDB is not configured"}), 404
    try:
        success, err = await current_app.cosmos_conversation_client.ensure()
        if not current_app.cosmos_conversation_client or not success:
            if err:
                return jsonify({"error": err}), 422
            return jsonify({"error": "CosmosDB is not configured or not working"}), 500

        return jsonify({"message": "CosmosDB is configured and working"}), 200
    except Exception as e:
        logging.exception("Exception in /history/ensure")
        cosmos_exception = str(e)
        if "Invalid credentials" in cosmos_exception:
            return jsonify({"error": cosmos_exception}), 401
        elif "Invalid CosmosDB database name" in cosmos_exception:
            return (
                jsonify(
                    {
                        "error": f"{cosmos_exception} {app_settings.chat_history.database} for account {app_settings.chat_history.account}"
                    }
                ),
                422,
            )
        elif "Invalid CosmosDB container name" in cosmos_exception:
            return (
                jsonify(
                    {
                        "error": f"{cosmos_exception}: {app_settings.chat_history.conversations_container}"
                    }
                ),
                422,
            )
        else:
            return jsonify({"error": "CosmosDB is not working"}), 500


async def generate_title(conversation_messages) -> str:
    ## make sure the messages are sorted by _ts descending
    title_prompt = "Summarize the conversation so far into a 4-word or less title. Do not use any quotation marks or punctuation. Do not include any other commentary or description."

    messages = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in conversation_messages
    ]
    messages.append({"role": "user", "content": title_prompt})

    try:
        azure_openai_client = await init_openai_client()
        response = await azure_openai_client.chat.completions.create(
            model=app_settings.azure_openai.model, messages=messages, temperature=1, max_tokens=64
        )

        title = response.choices[0].message.content
        return title
    except Exception as e:
        logging.exception("Exception while generating title", e)
        return messages[-2]["content"]


app, init_cosmosdb = create_app()

auth = identity.web.Auth(
        session=session,
        authority=f"https://login.microsoftonline.com/{os.environ["TENANT_ID"]}",
        client_id=os.environ["CLIENT_ID"],
        client_credential=os.environ["CLIENT_SECRET"],
    )

@fastapi_app.on_event("startup")  
async def startup_event():  
    await init_cosmosdb() 

fastapi_app.mount("/", app)

# if __name__ == "__main__":
#     uvicorn.run("app:fastapi_app", host="localhost", port=5000, reload=True)