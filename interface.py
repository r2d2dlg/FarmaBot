import gradio as gr
from chatbot_logic import detect_language

def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# FarmaBot Interface")
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        submit = gr.Button("Send")

        def respond(message):
            language = detect_language(message)
            return f"Detected language: {language}"

        submit.click(respond, inputs=msg, outputs=chatbot)

    return demo