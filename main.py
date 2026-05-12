import gradio as gr
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
from typing import Optional
import webbrowser
from chat_response import ResponseHandler


class ChatbotGUI:
    def __init__(self, response_handler: ResponseHandler):
        """
        Initialize the chatbot GUI with multiple interface options.

        Args:
            response_handler: Instance of ResponseHandler for processing responses
        """
        self.response_handler = response_handler
        self.current_interface: Optional[str] = None

    def create_web_interface(self):
        """
        Create and launch a web-based interface using Gradio.
        This provides a modern, accessible interface through the web browser.
        """

        def chat_response(message, history):
            # Process message and get response
            response_data = self.response_handler.get_response(message)

            if response_data['status'] == 'error':
                return f"Error: {response_data['response']}"

            # Format response with confidence score if needed
            response = response_data['response']
            if response_data['low_confidence']:
                response += f"\n\n(Note: Confidence: {response_data['confidence']:.2f})"

            return response

        # Create Gradio chat interface with customized settings
        interface = gr.ChatInterface(
            chat_response,
            title="Mental Health Support Chatbot",
            description="I'm here to listen and provide support. While I can offer guidance, "
                        "please remember I'm not a replacement for professional mental health care.",
            examples=[
                "I've been feeling really anxious lately",
                "What are some ways to deal with stress?",
                "Can you tell me about depression symptoms?",
                "How do I help someone who is struggling?",
            ],
            retry_btn=None,
            undo_btn=None,
            clear_btn="Clear Chat"
        )

        # Launch the web interface
        interface.launch(share=False, inbrowser=True, theme="soft")

    def create_desktop_interface(self):
        """
        Create and launch a desktop interface using tkinter.
        This provides a native desktop application experience.
        """
        # Initialize main window
        self.root = tk.Tk()
        self.root.title("Mental Health Support Chatbot")
        self.root.geometry("800x600")

        # Set theme
        style = ttk.Style()
        style.theme_use('clam')

        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Create scrollable chat display
        self.chat_display = scrolledtext.ScrolledText(
            main_frame, wrap=tk.WORD, width=70, height=30)
        self.chat_display.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        # Create input field
        self.input_field = ttk.Entry(main_frame, width=60)
        self.input_field.grid(row=1, column=0, padx=5, pady=5)

        # Create send button
        send_button = ttk.Button(
            main_frame, text="Send", command=self._handle_desktop_input)
        send_button.grid(row=1, column=1, padx=5, pady=5)

        # Bind Enter key to send message
        self.input_field.bind("<Return>", lambda e: self._handle_desktop_input())

        # Display welcome message
        self._add_bot_message("Hello! I'm here to listen and provide support. "
                              "How can I help you today?")

        # Start the desktop interface
        self.root.mainloop()

    def _handle_desktop_input(self):
        """Handle user input in the desktop interface"""
        # Get and validate input
        user_input = self.input_field.get().strip()
        if not user_input:
            return

        # Display user message
        self._add_user_message(user_input)
        self.input_field.delete(0, tk.END)

        # Get and display bot response
        response_data = self.response_handler.get_response(user_input)
        if response_data['status'] == 'success':
            response = response_data['response']
            response += f"\n(Detected Intent: {response_data['intent']})" 
            if response_data['low_confidence']:
                response += f"\n(Confidence: {response_data['confidence']:.2f})"
            self._add_bot_message(response)
        else:
            self._add_bot_message(f"Error: {response_data['response']}")

    def _add_user_message(self, message: str):
        """Add a user message to the chat display"""
        self.chat_display.insert(tk.END, f"\nYou: {message}\n")
        self.chat_display.see(tk.END)

    def _add_bot_message(self, message: str):
        """Add a bot message to the chat display"""
        self.chat_display.insert(tk.END, f"\nBot: {message}\n")
        self.chat_display.see(tk.END)


def main():
    """Main function to initialize and start the chatbot application"""
    # Initialize response handler
    response_handler = ResponseHandler()

    # Initialize GUI
    gui = ChatbotGUI(response_handler)

    # Launch web interface in a separate thread
    web_thread = threading.Thread(target=gui.create_web_interface)
    web_thread.start()

    # Launch desktop interface in main thread
    gui.create_desktop_interface()


if __name__ == "__main__":
    main()


# Reference:
# https://docs.python.org/3/library/tkinter.html
# https://www.geeksforgeeks.org/python-gui-tkinter/
# https://medium.com/@owuordove/hands-on-guide-creating-a-gui-with-tkinter-in-python-cc7d5ae0b495
# https://www.geeksforgeeks.org/how-to-use-thread-in-tkinter-python/
# https://stackoverflow.com/questions/16856151/how-to-organize-threaded-gui-application-python
# https://www.geeksforgeeks.org/python-tkinter-scrolledtext-widget/
# https://medium.com/binmile/6-python-gui-frameworks-to-create-desktop-web-and-even-mobile-apps-5250af9f8258