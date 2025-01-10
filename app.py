import streamlit as st
from PIL import Image
from model import Multimodal
from io import BytesIO

st.set_page_config(layout= "wide", page_title = "VQA")
st.markdown("<h1 style = 'text-align:center'>VQA Tool</h1>",unsafe_allow_html=True)


file = st.file_uploader("Upload a Image",type=["png","jpg",'jpeg'])
if file != None:
    col1, col2  = st.columns(2)
    with col1:
        st.image(file, use_container_width=True)
    
    with col2:
        question = st.text_input("Your Question")
        if file != None and question != None:
            if st.button("Ask..."):
                image = Image.open(file)
                bytes_arry = BytesIO()
                image.save(bytes_arry,format="jpeg")
                image_bytes = bytes_arry.getvalue()



                res = Multimodal().process(image=image_bytes,text=question)
                st.write(res)
