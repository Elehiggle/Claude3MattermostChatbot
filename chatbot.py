import time
import anthropic
from mattermostdriver import Driver
import ssl
import certifi
import traceback
import json
import os
import threading
import re
import datetime
import logging
from bs4 import BeautifulSoup
from anthropic import Anthropic
import concurrent.futures
import base64
import httpx
from io import BytesIO
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# save the original create_default_context function so we can call it later
create_default_context_orig = ssl.create_default_context


# define a new create_default_context function that sets purpose to ssl.Purpose.SERVER_AUTH
def cdc(*args, **kwargs):
    kwargs["cafile"] = certifi.where()  # Use certifi's CA bundle
    kwargs["purpose"] = ssl.Purpose.SERVER_AUTH
    return create_default_context_orig(*args, **kwargs)


# monkey patching ssl.create_default_context to fix SSL error
ssl.create_default_context = cdc

# Anthropic API key and model
api_key = os.environ["ANTHROPIC_API_KEY"]
model = os.environ["ANTHROPIC_MODEL"]

# Mattermost server details
mattermost_url = os.environ["MATTERMOST_URL"]
mattermost_personal_access_token = os.environ["MATTERMOST_TOKEN"]

# Maximum website size
max_response_size = 1024 * 1024 * 100  # 100 MB
max_tokens = 4096
temperature = 0.0

# For filtering local links
regex_local_links = r'(?:127\.|192\.168\.|10\.|172\.1[6-9]\.|172\.2[0-9]\.|172\.3[0-1]\.|::1|[fF][cCdD]|localhost)'

# Create a driver instance
driver = Driver({
    'url': mattermost_url,
    'token': mattermost_personal_access_token,
    'scheme': 'https',
    'port': 443,
    'basepath': '/api/v4',
    'verify': True
})

# Chatbot account username, automatically fetched
chatbot_username = ""
chatbot_usernameAt = ""

# Create an Anthropic client instance
anthropic_client = Anthropic(api_key=api_key)

# Create a thread pool with a fixed number of worker threads
thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)


def get_system_instructions():
    current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return f"You are a helpful assistant. The current UTC time is {current_time}. Whenever users asks you for help you will provide them with succinct answers formatted using Markdown; do not unnecessarily greet people with their name. Do not be apologetic. Answer concisely without much bla bla. You know the user's name as it is provided within [CONTEXT, from:username] bracket at the beginning of a user-role message. Never add any CONTEXT bracket to your replies (eg. [CONTEXT, from:{chatbot_username}]). The CONTEXT bracket may also include grabbed text from a website if a user adds a link to his question."


def sanitize_username(username):
    if not re.match(r'^[a-zA-Z0-9_-]{1,64}$', username):
        username = re.sub(r'[.@!?]', '', username)[:64]
    if not re.match(r'^[a-zA-Z0-9_-]{1,64}$', username):
        username = ''.join(re.findall(r'[a-zA-Z0-9_-]', username))[:64]
    return username


def get_username_from_user_id(user_id):
    try:
        user = driver.users.get_user(user_id)
        return sanitize_username(user["username"])
    except Exception as e:
        logger.error(f"Error retrieving username for user ID {user_id}: {str(e)} {traceback.format_exc()}")
        return f"Unknown_{user_id}"


def ensure_alternating_roles(messages):
    updated_messages = []
    for i in range(len(messages)):
        if i > 0 and messages[i]["role"] == messages[i - 1]["role"]:
            updated_messages.append({
                "role": "assistant" if messages[i]["role"] == "user" else "user",
                "content": [
                    {
                        "type": "text",
                        "text": "[CONTEXT, acknowledged post]"
                    }
                ]
            })
        updated_messages.append(messages[i])
    return updated_messages


def send_typing_indicator(user_id, channel_id, parent_id=None):
    """Send a "typing" indicator to show that work is in progress."""
    options = {
        "channel_id": channel_id,
        "parent_id": parent_id
    }
    driver.client.make_request('post', f'/users/{user_id}/typing', options=options)


def send_typing_indicator_loop(user_id, channel_id, stop_event):
    while not stop_event.is_set():
        try:
            send_typing_indicator(user_id, channel_id)
            time.sleep(2)
        except Exception as e:
            logging.error(f"Error sending busy indicator: {str(e)} {traceback.format_exc()}")


def handle_typing_indicator(user_id, channel_id):
    stop_typing_event = threading.Event()
    typing_indicator_thread = threading.Thread(target=send_typing_indicator_loop,
                                               args=(user_id, channel_id, stop_typing_event))
    typing_indicator_thread.start()
    return stop_typing_event, typing_indicator_thread


def process_message(messages, channel_id, root_id, sender_name):
    stop_typing_event = None
    typing_indicator_thread = None
    try:
        logger.info("Querying Anthropic API")

        # Start the typing indicator
        stop_typing_event, typing_indicator_thread = handle_typing_indicator(driver.client.userid, channel_id)

        try:
            # Send the messages to the Anthropic API
            response = anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=get_system_instructions(),
                messages=messages,
                timeout=120,
                temperature=temperature
            )
        except anthropic.APITimeoutError:
            logging.warn("Anthropic API call timed out after 2 minutes")
            response_text = "Sorry, the API call took too long to respond."
        else:
            # Extract the text content from the ContentBlock object
            content_blocks = response.content
            response_text = ""
            for block in content_blocks:
                if block.type == 'text':
                    response_text += block.text

            # Failsafe: Remove all blocks containing [CONTEXT
            response_text = re.sub(r'(?s)\[CONTEXT.*?]', '', response_text).strip()

        # Send the API response back to the Mattermost channel as a reply to the thread or as a new thread
        driver.posts.create_post({
            "channel_id": channel_id,
            "message": response_text,
            "root_id": root_id
        })

    except Exception as e:
        logging.error(f"Error processing message: {str(e)} {traceback.format_exc()}")
    finally:
        if stop_typing_event is not None:
            stop_typing_event.set()
            typing_indicator_thread.join()


async def message_handler(event):
    try:
        event_data = json.loads(event)
        logging.info(f"Received event: {event_data}")
        if event_data.get("event") == "hello":
            logging.info("Received 'hello' event. WebSocket connection established.")
        elif event_data.get("event") == "posted":
            post = json.loads(event_data["data"]["post"])
            sender_id = post["user_id"]
            if sender_id != driver.client.userid and sender_id != "agrnjtruxjycign4wbimup8dua":  # You can ignore this or remove it, we have multiple chatbots and don't want them to answer each other in a loop
                # Remove the "@chatbot" mention from the message
                message = post["message"].replace(chatbot_usernameAt, "").strip()
                channel_id = post["channel_id"]
                sender_name = sanitize_username(event_data["data"]["sender_name"])
                root_id = post["root_id"]  # Get the root_id of the thread
                post_id = post["id"]
                channel_display_name = event_data["data"]["channel_display_name"]

                try:
                    # Retrieve the thread context
                    messages = []
                    chatbot_invoked = False
                    if root_id:
                        thread = driver.posts.get_thread(root_id)
                        # Sort the thread posts based on their create_at timestamp
                        sorted_posts = sorted(thread["posts"].values(), key=lambda x: x["create_at"])
                        for thread_post in sorted_posts:
                            if thread_post["id"] != post_id:
                                thread_sender_name = get_username_from_user_id(thread_post["user_id"])
                                thread_message = thread_post["message"]
                                role = "assistant" if thread_post["user_id"] == driver.client.userid else "user"
                                messages.append({
                                    "role": role,
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": f"[CONTEXT, from:{thread_sender_name}] {thread_message}"
                                        }
                                    ]
                                })

                                if role == "assistant":
                                    chatbot_invoked = True
                    else:
                        # If the message is not part of a thread, reply to it to create a new thread
                        root_id = post["id"]

                    # Add the current message to the messages array if "@chatbot" is mentioned, the chatbot has already been invoked in the thread or its a DM
                    if chatbot_usernameAt in post["message"] or chatbot_invoked or channel_display_name.startswith("@"):
                        links = re.findall(r'(https?://\S+)', message)  # Allow both http and https links
                        extracted_text = ""
                        total_size = 0
                        image_messages = []

                        with httpx.Client() as client:
                            for link in links:
                                if re.search(regex_local_links, link):
                                    logging.info(f"Skipping local URL: {link}")
                                    continue
                                try:
                                    with client.stream("GET", link, timeout=4, follow_redirects=True) as response:
                                        final_url = str(response.url)

                                        if re.search(regex_local_links, final_url):
                                            logging.info(f"Skipping local URL after redirection: {final_url}")
                                            continue

                                        content_type = response.headers.get('content-type', '').lower()
                                        if 'image' in content_type:
                                            # Check for compatible content types
                                            compatible_content_types = ['image/jpeg', 'image/png', 'image/gif', 'image/webp']
                                            if content_type not in compatible_content_types:
                                                raise Exception(f"Unsupported image content type: {content_type}")

                                            # Handle image content
                                            image_data = b""
                                            for chunk in response.iter_bytes():
                                                image_data += chunk
                                                total_size += len(chunk)
                                                if total_size > max_response_size:
                                                    extracted_text += "*WEBSITE SIZE EXCEEDED THE MAXIMUM LIMIT FOR THE CHATBOT, WARN THE CHATBOT USER*"
                                                    raise Exception("Response size exceeds the maximum limit at image processing")

                                            # Open the image using Pillow
                                            image = Image.open(BytesIO(image_data))

                                            # Calculate the aspect ratio of the image
                                            width, height = image.size
                                            aspect_ratio = width / height

                                            # Define the supported aspect ratios and their corresponding dimensions
                                            supported_ratios = [
                                                (1, 1, 1092, 1092),
                                                (0.75, 3 / 4, 951, 1268),
                                                (0.67, 2 / 3, 896, 1344),
                                                (0.56, 9 / 16, 819, 1456),
                                                (0.5, 1 / 2, 784, 1568)
                                            ]

                                            # Find the closest supported aspect ratio
                                            closest_ratio = min(supported_ratios,
                                                                key=lambda x: abs(x[0] - aspect_ratio))
                                            target_width, target_height = closest_ratio[2], closest_ratio[3]

                                            # Resize the image to the target dimensions
                                            resized_image = image.resize((target_width, target_height),
                                                                         Image.Resampling.LANCZOS)

                                            # Save the resized image to a BytesIO object
                                            buffer = BytesIO()
                                            resized_image.save(buffer, format=image.format, optimize=True)
                                            resized_image_data = buffer.getvalue()

                                            # Check if the resized image size exceeds 3MB
                                            if len(resized_image_data) > 3 * 1024 * 1024:
                                                # Compress the resized image to a target size of 3MB
                                                target_size = 3 * 1024 * 1024
                                                quality = 90
                                                while len(resized_image_data) > target_size:
                                                    # Reduce the image quality until the size is within the target
                                                    buffer = BytesIO()
                                                    resized_image.save(buffer, format=image.format, optimize=True, quality=quality)
                                                    resized_image_data = buffer.getvalue()
                                                    quality -= 5

                                                    if quality <= 0:
                                                        raise Exception("Image too large, can't compress")

                                            image_data_base64 = base64.b64encode(resized_image_data).decode("utf-8")
                                            image_messages.append({
                                                "type": "image",
                                                "source": {
                                                    "type": "base64",
                                                    "media_type": content_type,
                                                    "data": image_data_base64,
                                                },
                                            })
                                        else:
                                            # Handle text content
                                            content_chunks = []
                                            for chunk in response.iter_bytes():
                                                content_chunks.append(chunk)
                                                total_size += len(chunk)
                                                if total_size > max_response_size:
                                                    extracted_text += "*WEBSITE SIZE EXCEEDED THE MAXIMUM LIMIT FOR THE CHATBOT, WARN THE CHATBOT USER*"
                                                    raise Exception("Response size exceeds the maximum limit")
                                            content = b''.join(content_chunks)
                                            soup = BeautifulSoup(content, 'html.parser')
                                            extracted_text += soup.get_text()
                                except Exception as e:
                                    logging.error(f"Error extracting content from link {link}: {str(e)} {traceback.format_exc()}")

                        content = f"[CONTEXT, from:{sender_name}"
                        if extracted_text != "":
                            content += f", extracted_website_text:{extracted_text}"
                        content += f"] {message}"

                        if image_messages:
                            image_messages.append({
                                "type": "text",
                                "text": content
                            })
                            messages.append({
                                "role": "user",
                                "content": image_messages
                            })
                        else:
                            messages.append({
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": content
                                    }
                                ]
                            })

                        # Ensure alternating roles in the messages array
                        messages = ensure_alternating_roles(messages)

                        # Submit the task to the thread pool. We do this because Mattermostdriver-async is outdated
                        thread_pool.submit(process_message, messages, channel_id, root_id, sender_name)

                except Exception as e:
                    logging.error(f"Error inner message handler: {str(e)} {traceback.format_exc()}")
        else:
            # Handle other events
            pass
    except json.JSONDecodeError:
        logging.error(f"Failed to parse event as JSON: {event} {traceback.format_exc()}")
    except Exception as e:
        logging.error(f"Error message_handler: {str(e)} {traceback.format_exc()}")

try:
    # Log in to the Mattermost server
    driver.login()
    chatbot_username = driver.client.username
    chatbot_usernameAt = f"@{chatbot_username}"

    # Initialize the WebSocket connection
    while True:
        try:
            # Initialize the WebSocket connection
            driver.init_websocket(message_handler)
        except Exception as e:
            logging.error(f"Error initializing WebSocket: {str(e)} {traceback.format_exc()}")
        time.sleep(2)

except Exception as e:
    logging.error(f"Error: {str(e)} {traceback.format_exc()}")