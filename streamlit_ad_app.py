import streamlit as st
from ad_generation_with_recommendation_final import *

import time
import pandas as pd
from PIL import Image

# ---- PAGE CONFIG ----
st.set_page_config(page_title="Retail Product Ad Generator & Recommendation App", page_icon="🎨", layout="centered")

# ---- STYLING ----
st.markdown(
    """
    <style>
        .title {
            text-align: center;
            background-color: #d0e8ff;
            padding: 20px;
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---- Image file for markdown
import base64

# --- Convert local image to base64 ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        b64_encoded = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_encoded}"




# ---- HEADER ----
st.markdown("""<h1 class='title'>Retail Ad Generator</h1>""", unsafe_allow_html=True)
st.write("")
st.write("🔹 Generate eye-catching product ads using **LLMs & ML-powered ads and product recommendations**.")
st.write("🔹 Powered by **Gemini, Sentence Transformers & Faiss Product Matching**.")
st.write("🔹 Recommendations using **Clustering with KMeans & Cosine Similarity**.")

st.write("---")  # Break line

query = ""

# ----session variable---#
# Example query buttons
if 'query' not in st.session_state:
    st.session_state['query'] = None

if 'button_clicked' not in st.session_state:
    st.session_state['button_clicked'] = 0

def set_query(q):
    print("print set_query ", q)
    st.session_state['query'] = q
    print("st.session_state['query'] ", st.session_state['query'])
    st.session_state['button_clicked'] = 1


# ---- USER INPUT ---- #
st.subheader("Enter a product search query")
query = st.text_input("Example: 'Affordable running shoes with high durability'")
print("query : ", type(query), query)
print("st.session_state['button_clicked'] ", st.session_state['button_clicked'])
if st.session_state['button_clicked'] == 0:
    set_query(query)

col1, col2, col3 = st.columns(3)
with col1:
    st.button("Best budget sneakers that glow in the dark", on_click=set_query, args=("Best budget sneakers that glow in the dark",))
with col2:
    st.button("Hiking boots waterproof lightweight", on_click=set_query, args=("Hiking boots waterproof lightweight",))
with col3:
    st.button("Affordable Sunscreen with 40 SPF", on_click=set_query, args=("Affordable Sunscreen with 40 SPF",))


# ---- MAIN APP  ----
if st.session_state['query'] is not None and st.session_state['query'] != "":
    st.subheader(st.session_state['query'].title())

    try:
        progress_bar = st.progress(0)
        status_text = st.empty()  # Placeholder for updating text dynamically

        # Step 1: Fetch top products
        status_text.write("🔄 Fetching top products...")
        new_query = st.session_state['query']

        llm_out_1 = process_user_input(new_query)
        
        dict_data = process_llm_output_1(llm_out_1)
        include = dict_data['Brands_include']
        exclude = dict_data['Brands_exclude']
        user_query = dict_data['Query']
        rec_product = recommend_similar_products(user_query, 50)  

        progress_bar.progress(55)
        time.sleep(2)

        # Step 2: Filter & refine results
        status_text.write("🔄 Refining product recommendations...")
        recommendations_filtered = product_filter_brand(rec_product, include, exclude)

        progress_bar.progress(65)
        time.sleep(2)

        # Step 3: Generate ad text with LLM
        status_text.write("📝 Personalizing ad content using LLMs...")
        response = get_customized_ad(new_query)  #--user's orig query for full context

        # st.write(response)

        dic_ad = process_llm_ad(response)
      

        headline = dic_ad['headline']
     
        body = dic_ad['body']
       
        action = dic_ad['action']
      
        image_desc = dic_ad['image']

        progress_bar.progress(80)
        time.sleep(2)

        # Step 4: Generate image
        status_text.write("🎨 Your ad coming right up...")
        res = get_image_for_ad(image_desc)


        progress_bar.progress(100)
        time.sleep(2)

        st.success("✅ Ad Generation Complete!")

        # ---- DISPLAY OUTPUT ----

        st.markdown(
            f"""
            <div style="text-align:center; background-color:#f0f8ff; padding:15px; border-radius:10px;">
                <h2 style="color:#007acc;">💎 {headline} 💎</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Animated Image Placeholder
        image_placeholder = st.empty()
        with st.spinner("✨ Generating your ad image..."):
            time.sleep(2)  # Simulate loading delay
            if res != 0:
                ad_image_path = "ad_image.jpg"
            else:
                ad_image_path = "default_image.jpg"  # Pre-loaded company logo

        image_placeholder.image(ad_image_path, caption=action, use_container_width =True)

        # Styled Body Text
        st.markdown(
            f"""
            <div style="padding:10px; background-color:#f9f9f9; border-left: 5px solid #007acc;">
                <p style="font-size:18px;">{body}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Call to Action
        st.markdown(
            f"""
            <div style="text-align:center; margin-top:10px;">
                <h3 style="color:#ff5733;">🌟 {action} 🌟</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display Recommended Products with Icons
        if recommendations_filtered is not None:
            st.subheader("🧺 Take a look at the matching products:")

            cols = st.columns(2)
            # --- Get base64 image source ---
            image_src = get_base64_image("default_image.jpg")

            for i in range(0, len(recommendations_filtered), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i + j < len(recommendations_filtered):
                        row = recommendations_filtered.iloc[i + j]
                        with cols[j]:
                            image_src = get_base64_image("default_image.jpg")
                            st.markdown(
                                f"""
                                <div style="background-color:#f9f9f9; padding: 1rem; border-radius: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);">
                                    <img src="{image_src}" style="width:100%; border-radius: 10px;" />
                                    <h4 style="margin-top: 0.5rem;">{row['product_n_category']}</h4>
                                    <p style="margin: 0.2rem 0;">🧢 <b>Brand:</b> {row['brand']}</p>
                                    <p style="margin: 0.2rem 0;">💰 <b>Price:</b> ${row['price']}</p>
                                    <p style="font-size: 0.9rem; color: #555;">{row['product_description'][:100]}...</p>
                                    <button style="margin-top: 0.5rem; padding: 0.5rem 1rem; background-color:#007bff; color:white; border:none; border-radius:5px;">View Product</button>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
        else:
            st.subheader("🛍️Pre-order the product now!")

        #---------------------- You may also like -----------------------#
        rec = product_rec_clustering(recommendations_filtered, k=30)
        #st.write("rec ", rec)
        if rec is not None and not rec.empty:
            st.subheader("🛒 You May Also Like")
            
            cards_html = """
            <div style="display: flex; overflow-x: auto; padding: 10px;">
            """

            for _, row in rec.iterrows():
                image_src = "https://via.placeholder.com/200"  # Replace with your actual image path or URL
                card = f"""
                <div style="flex: 0 0 auto; width: 250px; margin-right: 15px; background-color: #f9f9f9; 
                            padding: 10px; border-radius: 10px; box-shadow: 2px 2px 8px rgba(0,0,0,0.1);">
                    <img src="{image_src}" style="width:100%; border-radius: 10px;" />
                    <h4 style="margin: 0.5rem 0 0.2rem;">{row['product_n_category']}</h4>
                    <p style="margin: 0.2rem 0;">🧢 <b>Brand:</b> {row['brand']}</p>
                    <p style="margin: 0.2rem 0;">💰 <b>Price:</b> ${row['price']}</p>
                    <p style="font-size: 0.9rem; color: #555;">{row['product_description'][:100]}...</p>
                    <button style="margin-top: 0.5rem; padding: 0.5rem 1rem; background-color:#007bff; color:white; border:none; border-radius:5px;">View</button>
                </div>
                """
                cards_html += card

            cards_html += "</div>"

            st.components.v1.html(cards_html, height=350, scrolling=True)       

    except Exception as e:
        st.error(f"Just hit a snag! Error: {e}")



