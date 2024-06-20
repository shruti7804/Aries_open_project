# Aries_open_project

Project overview : This project employs the BERT (Bidirectional Encoder Representations from Transformers) model, a cutting-edge transformer architecture. BERT has been fine-tuned on the SQuAD (Stanford Question Answering Dataset) to interpret and generate human-like responses. The main goal is to enable users to upload a PDF document, pose specific questions regarding its content, and receive precise answers in real-time.

Dependencies : streamlit,transformers,pyPDF2,os,Torch

Usage : 1.first run the script help.py to download the pre-trained model and tokenizer which will further used in the main script.<br/>
        2.then open the terminal where we are running the script and run the command 'streamlit run app.py' to host the web application.<br/>
        3.upload the file on web interface and then write your query and press enter.Now,you will get the desired answer.
