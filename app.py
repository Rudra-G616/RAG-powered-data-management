import gradio as gr
import os
import tempfile
import zipfile
from pathlib import Path
import base64
from rag_model import RAGModel, pipeline_function

def generate_tree(path, prefix='', is_last=False, is_root=False):
    if is_root:
        line = f"📁 {path.name}/\n"
    else:
        line = f"{prefix}└── {path.name}/\n" if is_last else f"{prefix}├── {path.name}/\n"
    
    if path.is_file():
        return f"{prefix}{'└── ' if is_last else '├── '}{path.name}\n"
    
    children = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
    for index, child in enumerate(children):
        is_child_last = index == len(children) - 1
        new_prefix = prefix + (' ' if is_last else '│ ')
        line += generate_tree(child, new_prefix, is_child_last)
    return line

def process_zip(zip_file):
    if zip_file is None:
        return "No file uploaded", gr.Dropdown(choices=[])
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            root_path = Path(temp_dir)
            tree_structure = generate_tree(root_path, is_root=True)
            all_files = list(root_path.rglob("*"))
            file_paths = [str(f.relative_to(root_path)) for f in all_files if f.is_file()]
            return tree_structure, gr.Dropdown(choices=file_paths)
        except Exception as e:
            return f"Error processing ZIP file: {str(e)}", gr.Dropdown(choices=[])

def preview_file(file_path, zip_file):
    if not file_path or not zip_file:
        return None
    
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_file.name, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        full_path = os.path.join(temp_dir, file_path)
        
        if file_path.lower().endswith('.pdf'):
            try:
                with open(full_path, 'rb') as file:
                    pdf_content = file.read()
                pdf_base64 = base64.b64encode(pdf_content).decode('utf-8')
                return f'<embed src="data:application/pdf;base64,{pdf_base64}" width="100%" height="600px" />'
            except Exception as e:
                return f"Error reading PDF file: {str(e)}"
        else:
            try:
                with open(full_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                return content
            except UnicodeDecodeError:
                return "This file is not a text file and cannot be previewed."
            except Exception as e:
                return f"Error reading file: {str(e)}"

rag_model = None

def init_rag_chatbot(zip_file):
    global rag_model
    rag_model = RAGModel()
    pipeline_function(zip_file, rag_model)
    return "RAG Chatbot initialized."

def rag_chatbot(message, history):
    global rag_model
    if rag_model is None:
        return history + [{"role": "assistant", "content": "Please initialize the RAG Chatbot first by processing a ZIP file."}]
    
    history.append({"role": "user", "content": message})
    response = rag_model.query(message)
    history.append({"role": "assistant", "content": response})
    return history

with gr.Blocks() as demo:
    gr.Markdown("# Directory Tree Structure with File Preview and RAG Chatbot")
    
    with gr.Row():
        file_input = gr.File(label="Upload ZIP file")
        process_button = gr.Button("Process ZIP")
    
    with gr.Row():
        tree_output = gr.Textbox(label="Directory Tree Structure", lines=20)
        file_dropdown = gr.Dropdown(label="Select file to preview")
    
    file_preview = gr.HTML(label="File Preview")
    init_status = gr.Textbox(label="Initialization Status")
    
    chatbot = gr.Chatbot(label="RAG Chatbot", type="messages")
    msg = gr.Textbox(label="Enter your message")
    clear = gr.Button("Clear")

    msg.submit(rag_chatbot, [msg, chatbot], [chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
    
    process_button.click(
        process_zip,
        inputs=[file_input],
        outputs=[tree_output, file_dropdown]
    )
    
    process_button.click(
        init_rag_chatbot,
        inputs=[file_input],
        outputs=[init_status]
    )
    
    file_dropdown.change(
        preview_file,
        inputs=[file_dropdown, file_input],
        outputs=[file_preview]
    )

if __name__ == "__main__":
    demo.launch(share=True)
