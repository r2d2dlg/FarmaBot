<<<<<<< HEAD
# FarmaBot - Pharmacy Assistant Chatbot

FarmaBot is a bilingual (English/Spanish) chatbot designed to assist customers with pharmacy-related queries. It can provide information about medicines, store locations, and general pharmacy services.

## Features

- Bilingual support (English/Spanish)
- Medicine information lookup
- Store location and hours information
- Natural language processing
- Modern, user-friendly interface
- Secure database integration

## Prerequisites

- Python 3.9 or higher
- SQL Server with the ChatbotFarmacia database
- OpenAI API key
- ODBC Driver 17 for SQL Server

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/farmabot.git
cd farmabot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your configuration:
```env
OPENAI_API_KEY=your_openai_api_key
SERVER_NAME=your_server_name
DATABASE_NAME=ChatbotFarmacia
DRIVER=ODBC Driver 17 for SQL Server
```

## Database Setup

1. Ensure SQL Server is running
2. Create the ChatbotFarmacia database
3. Import the schema from `ChatbotFarmacia.sql`
4. Populate the database with medicine and store information

## Running the Bot

1. Start the chatbot:
```bash
python main.py
```

2. Access the interface:
- Local URL: http://127.0.0.1:7860
- Public URL: (if share=True in main.py)

## Project Structure

```
farmabot/
├── core/               # Core bot functionality
│   ├── __init__.py
│   ├── bot.py
│   └── language_manager.py
├── services/          # Business logic services
│   ├── __init__.py
│   ├── database_service.py
│   ├── medicine_service.py
│   └── store_service.py
├── interface/         # User interface
│   ├── __init__.py
│   └── gradio_interface.py
├── main.py           # Entry point
├── requirements.txt  # Dependencies
└── README.md        # Documentation
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the language models
- Gradio for the UI framework
- LangChain for the AI framework 
=======
---
title: FarmacyChatbot
emoji: 💬
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.0.1
app_file: app.py
pinned: false
short_description: Chatbot for a pharmacy in the Republic of Panama
---

An example chatbot using [Gradio](https://gradio.app), [`huggingface_hub`](https://huggingface.co/docs/huggingface_hub/v0.22.2/en/index), and the [Hugging Face Inference API](https://huggingface.co/docs/api-inference/index).
>>>>>>> 42c0345866f5d44ce526cace5c2a3f2f9b8d8160
