# pip install gradio
import gradio as gr
from summarizer import summarize

gr.Interface(
    fn=summarize,                                  # your function
    inputs=gr.Textbox(label="Website URL"),
    outputs=gr.Markdown(label="Summary"),
    title="🔎 AI Website Summarizer",
).launch(share=True)   # share=True → a public link you can post! 🎉

def main():
    # Provide a website URL you want to test
    target_url = "https://www.cricbuzz.com/"
    
    print(f"Fetching and summarizing: {target_url}...\n")
    
    # Call the function from your summarizer.py file
    result = summarize(target_url)
    
    # Print the AI's markdown response
    print(result)

if __name__ == "__main__":
    main()