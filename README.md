# Claude3MattermostChatbot

This project is a chatbot for Mattermost that integrates with the Anthropic API to provide helpful responses to user messages. The chatbot - like this readme - is mostly written by **Claude 3 AI**, listens for messages mentioning "@chatbot" or direct messages, processes the messages, and sends the responses back to the Mattermost channel.

## Features

- Responds to messages mentioning "@chatbot" or direct messages
- Extracts text content from links shared in the messages
- Supports the **Vision API** for describing images provided as URLs within the chat message
- Maintains context of the conversation within a thread
- Sends typing indicators to show that the chatbot is processing the message
- Utilizes a thread pool to handle multiple requests concurrently (due to `mattermostdriver-asyncio` being outdated)

## Prerequisites

- Python 3.8
- Mattermost server with API access
- Anthropic API key
- Personal access token or login/password for a dedicated Mattermost user account for the chatbot

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Claude3MattermostChatbot.git
cd Claude3MattermostChatbot
```

2. Install the required dependencies:

```bash
python3.8 -m pip install anthropic mattermostdriver ssl certifi beautifulsoup4 pillow httpx
```

3. Update the following variables in the script with your own values:

- `api_key`: Your Anthropic API key
- `mattermost_url`: The URL of your Mattermost server
- `personal_access_token`: The personal access token with relevant permissions from a dedicated Mattermost user account created specifically for the chatbot. Note that `mattermostdriver` does not support bot tokens.
- `chatbot_account`: The username of your chatbot account

Alternatively, you can use the login/password combination for the dedicated Mattermost user account if you prefer.

## Usage

Run the script:

```bash
python3.8 chatbot.py
```

The chatbot will connect to the Mattermost server and start listening for messages.
When a user mentions "@chatbot" in a message or sends a direct message to the chatbot, the chatbot will process the message, extract text content from links (if any), handle image content using the Vision API, and send the response back to the Mattermost channel.

> **Note:** If you don't trust your users, it's recommended to disable the URL/image grabbing feature, even though the chatbot filters out local addresses and IPs.

## Configuration

You can customize the behavior of the chatbot by modifying the following variables in the script:

- `model`: The Anthropic model to use for generating responses
- `max_response_size`: The maximum size of the website content to extract (in bytes)
- `max_tokens`: The maximum number of tokens to generate in the response
- `temperature`: The temperature value for controlling the randomness of the generated responses

## Known Issues

While the chatbot works great for me, there might still be some bugs lurking inside. We've done our best to address them, but if you encounter any issues, please let me know!

## Future Plans

I plan to create a similar project for integration with ChatGPT in the future.

## Monkey Patch

Please note that the monkey patch in the code is necessary due to some SSL errors that occur because of a mistake within the `mattermostdriver` library. The patch ensures that the chatbot can establish a secure connection with the Mattermost server.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Anthropic](https://www.anthropic.com/) for providing the API for generating responses
- [Mattermost](https://mattermost.com/) for the messaging platform
- [mattermostdriver](https://github.com/Vaelor/python-mattermost-driver) for the Mattermost API client library