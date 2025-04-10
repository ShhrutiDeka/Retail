
from sentence_transformers import SentenceTransformer

# Load the model (optimized for retrieval tasks)
model = SentenceTransformer("msmarco-distilbert-base-v3")

# Load product data (assuming 'product_name' and 'description' columns)
product_data = pd.read_csv("/content/products_desc.csv")

# Combine name and description for better embeddings
product_data["combined_text"] = product_data["product_description"] + " " + product_data["product_n_category"]

# Generate embeddings
product_embeddings = model.encode(product_data["combined_text"].tolist(), normalize_embeddings=True)

# Save embeddings for future use
np.save("product_embeddings.npy", product_embeddings)

print("Product embeddings generated and saved!")
