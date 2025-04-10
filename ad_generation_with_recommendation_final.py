

#---------------- IMPORTS & setup------------------------#

import google.generativeai as genai


#---get from file in container----#
genai.configure(api_key= "") #--for gemini model

#---------------------------------#

import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import re
import requests
from time import sleep

# import zipfile
# with zipfile.ZipFile("product_embeddings.zip", 'r') as zip_ref:
#     zip_ref.extractall("product_embeddings.npy")

# !unzip product_embeddings.zip
#-------------------------------------------------------#

#---------------- Core Functions -----------------------------#

#-1) Process User Input

def process_user_input(user_input):
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f'''I will send you a query. Your job is to analyse it and output the following with NO OTHER explanations.
        If no input is provided then just output the word None.

        Output Format:

        1) Brands_include: (this is the brands the user wants, maybe multiple, seperate them by comma, Use None is not applicable)
        2) Brands_exclude: (this is the brand the user wants to exclude, maybe multiple, seperate them by comma, Use None is not applicable)
        3) Query: (polished query without the above components, whatever is left, reframe according to its meaning without missing keywords)

        Example:
        Query- I want hiking shoes in mid-range from Nike or addidas. Do not show from Puma.

        Output-
        1) Brands_include: Nike, addidas
        2) Brands_exclude: Puma
        3) Query: hiking shoes in mid-range price

        Heres your input:
      '''

    response = model.generate_content(prompt+ user_input)
    #---debug
    print("Response from Gemini 1st level: ", response.text)
    #-------X
    return response.text

#-2) Get matching products

def recommend_similar_products(user_query, top_k=15):
    # Load the transformer model (same as used for product embeddings)
    model = SentenceTransformer("msmarco-distilbert-base-v3")

    product_data = pd.read_csv("products_desc.csv")
    product_embeddings = np.load("./product_embeddings.npy/product_embeddings.npy")

    #-- product_embeddings (numpy array) and product_data (DataFrame) exist

    # Step 1: Create FAISS index
    embedding_dim = product_embeddings.shape[1]  # Get the embedding size
    index = faiss.IndexFlatL2(embedding_dim)  # L2 distance (works since we normalized embeddings)
    index.add(product_embeddings)  # Add product embeddings to FAISS index

    # Step 2: Convert the user query into an embedding
    query_embedding = model.encode([user_query], normalize_embeddings=True)  # Shape: (1, embedding_dim)

    # Step 3: Perform similarity search
    distances, indices = index.search(query_embedding, top_k)  # Get top-K closest products

    # Step 4: Retrieve product names from indices
    # recommended_products = product_data.iloc[indices[0]]["description"].tolist()
    recommended_products = product_data.iloc[indices[0]]  # Get full rows

     #---debug
    print("Response from Embeddings [shape]: ", recommended_products.shape)
    print(recommended_products.head())
    #-------X

    return recommended_products


#-3) Filter the top k products by brand (if specified by user in query)

def product_filter_brand(recommendations, brand_include, brand_exclude):
    recommendations_filtered = recommendations
    print("line 108 :", brand_include, brand_exclude)
  
# ----------------- including brand ----------------------#
    if isinstance(brand_include, str) and ',' in brand_include and str(brand_include).strip().lower() != 'none' and brand_include:
        brand_include_list = [b.strip() for b in brand_include.split(",") if b.strip()]
        print("line 118 ", brand_include_list)

        filtered_by_include = pd.DataFrame() #initialize empty dataframe

        for brand in brand_include_list:
            brand_products = recommendations[recommendations["brand"] == brand]
            filtered_by_include = pd.concat([filtered_by_include, brand_products.head(6)]) # take only 6 per brand
        
        recommendations_filtered = filtered_by_include
    elif brand_include and str(brand_include).strip().lower() != 'none':
        recommendations_filtered = recommendations[recommendations["brand"] == brand_include.strip()].head(6) # for single brand, take 6
    

    # ------------------- excluding brand -------------------------#
    if isinstance(brand_exclude, str) and ',' in brand_exclude and str(brand_exclude).strip().lower() != 'none' and brand_exclude:
        brand_exclude_list = [b.strip() for b in brand_exclude.split(",") if b.strip()]
        print("line 128 ", brand_exclude_list)

        recommendations_filtered = recommendations_filtered[~recommendations_filtered["brand"].isin(brand_exclude_list)]
    elif brand_exclude and str(brand_exclude).strip().lower() != 'none':
        recommendations_filtered = recommendations_filtered[recommendations_filtered["brand"] != brand_exclude.strip()]

    # ---debug
    print("Response from Filtration [shape]: ", recommendations_filtered.shape)
    # -------X

    return recommendations_filtered.head(6)


#-4) Generate ad based on user query

def get_customized_ad(query):
    prompt = f'''You are an expert ad generator and have a knack for customized ads that are enticing to the targeted audience and can be *quirky*.
    You will use the user sent query: "{query.replace("'", "''")}" to create the ad. Do your best with just 1 ad and keep
    it non-offensive.

    Output format should be (with example below):

     <<Headline: Finally, Sneakers That Fight Back Against Unseen Curbs. (And Look Good Doing It!) Headline>>
     <<Image: A cartoonish graphic of a glowing sneaker kicking a cartoonish curb. Image>>
     <<Body: Tired of tripping over rogue sidewalks in the dark? Fear not! These glow-in-the-dark New Balances are your after-dark adventure sidekick.
      Budget-friendly and visibility-enhancing, they're like having tiny headlights on your feet.
      Reduces the risk of face-planting by approximately 67% (results may vary, don't sue us). Body>>
     <<Action: Stop Tripping, Start Glowing! Get Yours Today! Action>>

    '''

    genai.configure(api_key="")  # Get this from Google AI Studio

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)

    #---debug
    print("Response from Ad Generation: ", response)
    #-------X

    return response.text

#-5) Generate Image for the ad

def get_image_for_ad(output):
  prompt = f'''Your cousin gemini had created a great ad idea. I want you to take the ad idea and create the image suggested.
  Make sure to add the logo ( a big M) in crimsone placed prominently at the bottom right in your generated image.
  Make sure the image is always friendly, cute and stylish!

  Make the image reasonably high resolution and rectangle in shape.
  Heres the output from your cousin gemini: {output}
  '''

  API_TOKEN = ""
  headers = {"Authorization": f"Bearer {API_TOKEN}"}

  payload = {
      "inputs": f"{output}",
  }

  response = requests.post(
    "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0",
    headers=headers,
    json=payload,
  )
  count = 0

  if "image" in response.headers.get("Content-Type", ""):
      with open("ad_image.jpg", "wb") as f:
          f.write(response.content)
      print("Image saved as ad_image.jpg")
  else:
      print("API Error - retrying:", response.text)  #---if error - SLEEP and then retry (msg in app meanwhile - buffer)
      if count != 3:
        sleep(3)
        get_image_for_ad(output)
        count += 1
      else:
        return 0 #--failure on UI (default image stored in file)

  return response.text


#-6) Clustering for similar yet diverse product recommendation (you may also like)

def train_clusters(k=30):
  product_embeddings = np.load("./product_embeddings.npy/product_embeddings.npy")

  product_data = pd.read_csv("products_desc.csv")

  n_clusters = k  # tune based on dataset size
  kmeans = KMeans(n_clusters=n_clusters, random_state=42)
  product_data['cluster'] = kmeans.fit_predict(product_embeddings)
  print("Cluster model trained")
  return product_data, kmeans

def product_rec_clustering(filtered_products_df, k=30):
    product_data, kmeans = train_clusters()
    print("In Rec function : ", filtered_products_df.head())
    #-- get indices for matched product to get its cluster(s)
    product_indices = filtered_products_df.index.tolist()
    print("Product indices: ", product_indices)
    selected_cluster = product_data.loc[product_indices, "cluster"]
    print("Clusters length : ", len(selected_cluster))

    if len(selected_cluster) == 1:
        
        
       #---get centroid of cluster
        selected_centroid = kmeans.cluster_centers_[selected_cluster] #--take only one FOR NOW  !!

        all_similarities = cosine_similarity(selected_centroid.reshape(1, -1), kmeans.cluster_centers_)[0]

        # Get indices of top similar clusters (excluding itself)
        similar_clusters = np.argsort(all_similarities)[::-1][1:4]

        #--get products from the similar clusters
        diverse_recs = product_data[product_data['cluster'].isin(similar_clusters)].sample(6)
        print("Single cluster")
        return diverse_recs
    else:
        final_recs = []
        for i in set(selected_cluster):
            print("i = ", i)
            selected_centroid = kmeans.cluster_centers_[i] #--take only one FOR NOW  !!
            print("selected_centroid ", selected_centroid)
            all_similarities = cosine_similarity(selected_centroid.reshape(1, -1), kmeans.cluster_centers_)[0]

            # Get indices of top similar clusters (excluding itself)
            similar_clusters = np.argsort(all_similarities)[::-1][1:4]

            #--get products from the similar clusters
            # diverse_recs = product_data[product_data['cluster'].isin(similar_clusters)].sample(3)
            diverse_cluster_data = product_data[product_data['cluster'].isin(similar_clusters)]

            if len(diverse_cluster_data) >= 3:
                diverse_recs = diverse_cluster_data.sample(3)
            else:
                diverse_recs = diverse_cluster_data
            
            final_recs.append(diverse_recs)
        print("Multiple clusters")
        return pd.concat(final_recs, ignore_index=True)
#---------------------------------------------------------------------#

#------------------------ Processing Functions ----------------------#
def clean_headline(text):
  """Removes 'Headline', '<>', and '>>' from a string."""
  text = text.replace("Headline", "")
  text = text.replace("Body", "")  
  text = text.replace(" Action", "")
  text = text.replace(r"<>", "")        
  text = text.replace(r">*", "") 
  text = text.replace(r"<*", "")        
  text = text.strip()                  
  return text


def process_llm_output_1(llm_output):

  # llm_output = """
  #         1) Brands_include: Nike, addidas
  #         2) Brands_exclude: Puma
  #         3) Query: hiking shoes in mid-range price
  # """

  # Regular expression to capture key-value pairs
  matches = re.findall(r'\d+\)\s*([\w_]+):\s*(.*)', llm_output)

  # Convert to dictionary with "None" if missing or "None" in value
  output_dict = {key: (value if value.lower() != "none" else "None") for key, value in matches}

  ''' Output
  {'Brands_include': 'Nike, addidas',
 'Brands_exclude': 'Puma',
 'Query': 'hiking shoes in mid-range price'}
 '''
  return output_dict


def process_llm_ad(llm_ad):

  '''
  <<Headline: Finally, Sneakers That Fight Back Against Unseen Curbs. (And Look Good Doing It!) >>
     <<Image: A cartoonish graphic of a glowing sneaker kicking a cartoonish curb. >>
     <<Body: Tired of tripping over rogue sidewalks in the dark? Fear not! These glow-in-the-dark New Balances are your after-dark adventure sidekick.
      Budget-friendly and visibility-enhancing, they're like having tiny headlights on your feet.
      Reduces the risk of face-planting by approximately 67% (results may vary, don't sue us). >>
     <<Action: Stop Tripping, Start Glowing! Get Yours Today! >>

    '''
  llm_ad = llm_ad.replace("\n", " ").strip()

  dic_llm = {}

  def extract_headline(pattern, text):
    """
    Extracts the headline from the given text, covering the specified cases.
    """
    headline_match = re.search(pattern, text, re.DOTALL)
    if headline_match:
        headline = headline_match.group(1).strip()
        if not re.search(r'^\w+$', headline): # exclude single word headlines
            return headline
    return None
    
  # headline = re.findall(r'<<Headline:\s*(.*)\s*(Headline)*>>', llm_ad)
  # image = re.findall(r'<<Image:\s*(.*)\s*>>', llm_ad)
  # body = re.findall(r'<<Body:\s*(.*)\s*>>', llm_ad)
  # action = re.findall(r'<<Action:(.*)\s*>>', llm_ad)

  pattern_headline = r'(?:<<Headline\s*|\bHeadline\s*)(.*?)(?:\s*Headline>>|\s*>>|\bHeadline\b)'
  pattern_body = r'(?:<<Body\s*|\bBody\s*)(.*?)(?:\s*Body>>|\s*>>|\bBody\b)'
  pattern_image = r'(?:<<Image\s*|\bImage\s*)(.*?)(?:\s*Image>>|\s*>>|\bImage\b)'
  pattern_action = r'(?:<<Action\s*|\bAction\s*)(.*?)(?:\s*Action>>|\s*>>|\bAction\b)'
  headline = extract_headline(pattern_headline, llm_ad)
  body = extract_headline(pattern_body, llm_ad)
  image = extract_headline(pattern_image, llm_ad)
  action = extract_headline(pattern_action, llm_ad)
  dic_llm = {'headline': headline.replace(":", ""), 'image': image.replace(":", ""), 'body': body.replace(":", ""), 'action': action.replace(":", "")}

  #dic_llm = {'headline': headline[0].strip(), 'image': image[0].strip(), 'body': body[0].strip(), 'action': action[0].strip()}

  return dic_llm

#-------------------------------------------------------------------------#