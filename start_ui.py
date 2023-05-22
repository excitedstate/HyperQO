import os

if __name__ == '__main__':
    # # Run the streamlit app, 输出直接到控制台, 0缓冲
    os.system('streamlit run src/simple_ui.py --server.port 8501')
