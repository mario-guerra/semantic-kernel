
# !python -m pip install semantic-kernel==0.2.7.dev0
# !python -m pip install qdrant-client

from typing import Tuple
import re, asyncio
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding, AzureChatCompletion
from semantic_kernel.connectors.ai.chat_request_settings import ChatRequestSettings

kernel = sk.Kernel()

deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()

embedding_generator = AzureTextEmbedding("text-embedding-ada-002", endpoint, api_key)

system_message = """
You are AZSDK_Bot, an expert on Azure SDKs. 
You can answer questions about Azure SDKs and provide links to relevant repositories based on the context provided in prompts.
Prioritize the content in the prompt when answering questions. The link to the repo should be taken from the prompt.
Whenever you see '/master/' in a prompt, replace it with '/main/'. Do not modify the URL in any other way.
"""

chat_service = AzureChatCompletion(deployment, endpoint, api_key)

# Define the async function to get embeddings. Include retry logic to account for rate limit errors.
# There are two versions of this function because the error message was originally returning as a string,
# but now it returns as an Exception object. Keeping both versions for now in case we need to revert back.
# async def create_embedding(data):
#     MAX_RETRIES = 3
#     retry_count = 0
#     while retry_count < MAX_RETRIES:
#         embeddings = await embedding_generator.generate_embeddings_async(data)
#         if "exceeded call rate limit" in str(embeddings):
#             error_message = str(embeddings)
#             delay_str = re.search(r'Please retry after (\d+)', error_message)
#             if delay_str:
#                 delay = int(delay_str.group(1))
#                 print(f"Rate limit exceeded. Retrying in {delay} seconds...")
#                 await asyncio.sleep(delay)
#                 retry_count += 1
#             else:
#                 raise Exception("Unknown error message when creating embeddings.")
#         else:
#             return embeddings

#     raise Exception("Rate limit error. All retries failed.")

async def create_embedding(data):
    MAX_RETRIES = 5
    retry_count = 0
    while retry_count < MAX_RETRIES:
        try:
            embeddings = await embedding_generator.generate_embeddings_async(data)
            return embeddings
        except Exception as e:
            error_message = str(e)
            if "exceeded call rate limit" in error_message:
                delay_str = re.search(r'Please retry after (\d+)', error_message)
                if delay_str:
                    delay = int(delay_str.group(1))
                    print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                    retry_count += 1
                else:
                    raise Exception("Unknown error message when creating embeddings.")
            else:
                raise e

    raise Exception("Rate limit error. All retries failed.")

# Define the async function to ask the chatbot. This will be used to generate a response to the user's question from relevant README content
async def ask_chatbot(input):
    messages = [("system", system_message), ("user", input)]
    reply = await chat_service.complete_chat_async(messages=messages, request_settings=ChatRequestSettings(temperature=0.7, top_p=0.8, max_tokens=2000))
    return(reply)

qdrant_client = QdrantClient(path=r"C:\Users\marioguerra\Work\semantic-kernel\python\semantic_kernel\memory")

async def query_qdrant(user_input, collection_name, language):
    # print("Querying Qdrant with input: ", user_input)
    embedding = await create_embedding(user_input)
    vector_embedding = embedding.tolist()
    search_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=vector_embedding[0],
        query_filter=Filter(
            must=[  
                FieldCondition(
                    key='language',
                    match=MatchValue(value=language)
                )
            ]
        ),
        limit=1,
        # score_threshold=0.75, # we may want to consider a threshold 
        # score if wewant to filter out low confidence results
    )
    return search_results

# Most of the work happens in the chat() function. This is where we 
# check the user input for a language match and call the query_qdrant() function.
# Summary of the code in the chat function:
# 1. First define regex patterns for different languages
# 2. Define the chat() function as an asynchronous function that takes
#    kernel, context, and previous_input as arguments
# 3. Inside the chat() function, check if the user input matches more than one
#    language and ask the user to focus on one language at a time if true
# 4. If the user input matches a single language, call the query_qdrant()
#    function with the matched language to find READMEs relevant to question
# 5. Handle the case where the user input does not match any language and
#    append the user input to the previous_input string, since this scenario
#    is most likely a follow up question about the previous SDK
# 6. If the length of the previous_input string exceeds 10,000 characters,
#    trim it to keep only the last 10,000 characters to keep within our
#    context limit (admittedly arbitrary, since I don't know the exact limit)
# 7. Update the previous_input string with the new context and provide the
#    chatbot answer at the end of each iteration

rust = r'^\/(rust).*$'
python = r'^\/(python|Python).*$'
java = r'^\/(java|Java).*$'
js = r'^\/(javascript|js|Javacript|JavaScript|JS).*$'
net = r'^\/(\.net|net|\.NET|NET|csharp|C#).*$'

async def chat(kernel: sk.Kernel, context: sk.ContextVariables, previous_input: str) -> Tuple[bool, str, str]:
    try:
        user_input = input("User:> ")
        # print(f"User:> {user_input}")

        if (user_input != ""):
            language_matches = {
                "Rust": bool(re.match(rust, user_input, re.IGNORECASE)),
                "Python": bool(re.match(python, user_input, re.IGNORECASE)),
                "Java": bool(re.match(java, user_input, re.IGNORECASE)),
                "JavaScript": bool(re.match(js, user_input, re.IGNORECASE)),
                ".NET": bool(re.match(net, user_input, re.IGNORECASE)),
            }

            matches = sum(language_matches.values())

            if matches > 1:
                print("AZSDK_Bot:> Please, one language at a time!")
                return True, user_input, previous_input

            if matches:
                language = [lang for lang, matched in language_matches.items() if matched][0]
                # print("language: ", language)

                search_results = await query_qdrant(user_input, "AzureSDKs", language)
                # print("qdrant search results: ", search_results)
                if search_results:
                    for result in search_results:
                        payload = result.payload
                        sdk = payload["SDK"]
                        # print(f"SDK:> {sdk}")
                        link_to_repo = payload["link_to_repo"]
                        # print(f"Link to repo:> {link_to_repo}")
                        language = payload["language"]
                        readme_text = payload["README_text"][:10000]
                    context_prompt = user_input + str(readme_text) + str(link_to_repo) + str(language) + str(sdk)
                else:
                    context_prompt = "No results found for this query."
            else:
                if len(previous_input) > 10000:
                    previous_input = previous_input[-10000:]
                context_prompt = previous_input + user_input        
        else:
            if not previous_input: # 
                context_prompt = "Tell me about Azure SDKs."
            else:
                context_prompt = previous_input + "\n\nTell me about this Azure SDK."

    except KeyboardInterrupt:
        print("\n\nExiting chat...")
        return False, "", ""
    except EOFError:
        print("\n\nExiting chat...")
        return False, "", ""
    if user_input == "exit":
        print("\n\nExiting chat...")
        return False, "", ""
    print("Thinking...")
    answer = await ask_chatbot(context_prompt)
    previous_input = context_prompt + answer
    print(f"AZSDK_Bot:> {answer}")
    return True, user_input, previous_input

async def main():
    context = sk.ContextVariables()
    print("Begin chatting (type 'exit' to exit):\n")
    previous_input = ""
    chatting = True
    while chatting:
        chatting, context, previous_input = await chat(kernel, context, previous_input)

if __name__ == "__main__":
    asyncio.run(main())
