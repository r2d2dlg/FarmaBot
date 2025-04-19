# Deploying FarmaBot to Hugging Face Spaces

This guide explains how to deploy FarmaBot to Hugging Face Spaces.

## Prerequisites

- A Hugging Face account
- A Google API key for Gemini 2.5 Flash (recommended)
- (Alternative) An OpenAI API key if not using Gemini models

## Deployment Steps

1. **Create a new Space**
   - Go to https://huggingface.co/spaces
   - Click "Create a new Space"
   - Select "Gradio" as the SDK
   - Choose a license (e.g., Apache 2.0)
   - Set visibility to "Public" or "Private" as needed

2. **Upload Files**
   - Use the "Files" tab to upload the following files:
     - `app.py` - Main application file
     - `init_db.py` - Database initialization script
     - `requirements-huggingface.txt` as `requirements.txt` - Dependencies
     - All files in the `core/`, `services/`, and `interface/` directories
     - Any other necessary files (README-HUGGINGFACE.md as README.md, etc.)

3. **Set Environment Variables**
   - In your Space settings, add the following secrets:
     - `GOOGLE_API_KEY` - Your Google API key (recommended for using Gemini 2.5 Flash)
     - `OPENAI_API_KEY` - Your OpenAI API key (optional alternative)
     - `DB_CONNECTION_STRING` - Set to `sqlite:///farmabot.db`

4. **Configure Build**
   - In the "Settings" tab, ensure:
     - SDK: Gradio
     - Python version: 3.10 or higher
     - Set "Space hardware" to at least "CPU Medium" if possible

5. **Build and Deploy**
   - The Space will automatically build and deploy when files are uploaded
   - Check the "Factory Logs" for any build issues

## Gemini 2.5 Flash Configuration

FarmaBot is configured to use Google's Gemini 2.5 Flash model by default for:
- Faster response times
- Better handling of medical terminology
- Improved multilingual capabilities (Spanish/English)

To ensure the model works correctly:
1. Add your Google API key in the Space settings
2. The app will automatically select Gemini 2.5 Flash if the API key is available

## Troubleshooting

If you encounter issues during deployment:

1. Check the build logs for errors
2. Verify that all required dependencies are in `requirements.txt`
3. Ensure all environment variables are set correctly
4. Make sure the database is being initialized properly
5. Test locally using `python test_huggingface.py` before deployment

## Updating the Space

To update your deployed Space:

1. Make changes to your local files
2. Test them thoroughly
3. Upload the changed files to your Space
4. The Space will automatically rebuild

## Additional Resources

- [Hugging Face Spaces Documentation](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://www.gradio.app/docs/)
- [Google Gemini API Documentation](https://ai.google.dev/docs/gemini_api_overview) 