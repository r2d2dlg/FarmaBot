# FarmaBot - Pharmacy Assistant Chatbot

FarmaBot is a bilingual (English/Spanish) chatbot designed to assist customers with pharmacy-related queries. It can provide information about medicines, store locations, and general pharmacy services.

## Features

- Bilingual support (English/Spanish)
- Medicine information lookup
- Store location and hours information
- PDF quote generation
- Natural language processing
- Modern, user-friendly interface
- Secure database integration
- Support for both OpenAI and Google Gemini models

## Prerequisites

- Python 3.9 or higher
- SQL Server with the ChatbotFarmacia database
- API Key (OpenAI or Google depending on model choice)
- ODBC Driver 17 for SQL Server

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/farmabot.git
cd farmabot
```

2. Setup the environment (Windows):
```
# Using the provided setup script
.\setup.ps1

# Or manually:
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure your environment:
   - Copy `.env.template` to `.env`
   - Edit `.env` and add your API keys and configure the model

## Model Configuration

FarmaBot supports both OpenAI and Google Gemini models. In the `.env` file:

```
# For OpenAI models
MODEL=gpt-4-turbo  # or gpt-3.5-turbo
OPENAI_API_KEY=your_openai_api_key

# For Google Gemini models
MODEL=gemini-1.5-pro  # or gemini-1.0-pro
GOOGLE_API_KEY=your_google_api_key
```

## Database Setup

1. Ensure SQL Server is running
2. Create the ChatbotFarmacia database
3. Import the schema from `ChatbotFarmacia.sql`
4. Configure the connection string in `.env`
5. Test your connection:
```bash
python scripts/test_db.py
```

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
├── scripts/           # Utility scripts
├── main.py           # Entry point
├── requirements.txt  # Dependencies
├── setup.ps1         # Setup script
├── .env.template     # Environment template
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

- OpenAI and Google for the language models
- Gradio for the UI framework
- LangChain for the AI framework 