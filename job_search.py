import cloudinary
import pytesseract
from PIL import Image
from PIL import ImageEnhance
import cloudinary.api
import re
import os
from dotenv import load_dotenv
from io import BytesIO
import requests
# run pip install pytesseract
# run pip install requests
#run "pip install sentence-transformers" to install sentence transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


load_dotenv() 

cloudinary.config(
    cloud_name=os.getenv('CLOUD_NAME'),
    api_key=os.getenv('API_KEY'),
    api_secret=os.getenv('API_SECRET'),
    secure=True
)

def get_cv(asset_id): 
    result = cloudinary.api.resource_by_asset_id(asset_id)
    
    return result 

def extract_text_from_cloudinary(asset_id):
    image_metadata = get_cv(asset_id)

    # Debug: Print the metadata to see what we're getting
    #print(f"Image metadata: {image_metadata}")

    image_url = image_metadata.get('secure_url') 

    # Debug: Print the URL
    #print(f"Image URL: {image_url}")

    if not image_url:
            raise ValueError("Could not retrieve image URL from Cloudinary")
        
    # Downloading the image
    response = requests.get(image_url)

    # Debug: Check status code and content type
    print(f"Response status: {response.status_code}")
    print(f"Content type: {response.headers.get('Content-Type')}")
    # Save the raw response to a file for inspection
    with open('debug_response.txt', 'wb') as f:
        f.write(response.content[:100])  # Save just the first 100 bytes to see what we're getting


    if response.status_code != 200:
        raise ValueError(f"Failed to download image: {response.status_code}")
    
    #debug: # Only try to open as image if it's an image content type
    if 'image' in response.headers.get('Content-Type', ''):

        # Opening the image using PIL
        image_cv = Image.open(BytesIO(response.content))

        
        # Extracting text using pytesseract
        extracted_text_cv = pytesseract.image_to_string(image_cv)
        image_cv, 
        config='--psm 6' #assuming a single block of text through page segmentation 
        
        #debugging print
        print(f"Extracted text length: {len(extracted_text_cv)}")
        print(f"First 200 characters of extracted text:\n{extracted_text_cv[:200]}")
        

        return extracted_text_cv
    else:
        return f"Error: Received non-image content: {response.headers.get('Content-Type')}"

asset_id = "cd8f9affd329aa0722e8ffb60553694a"
text_cv = extract_text_from_cloudinary(asset_id)


embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def generate_embedding(text):
    return embedding_model.encode([text])[0].tolist()

def extract_job_details(job_description):
    """
    Extract job title, company name, responsibilities, qualifications, and years of experience from a job description.
    
    Args:
        job_description (str): The raw job description text.
        
    Returns:
        tuple: (job_title, company_name, job_responsibilities, job_qualifications, experience_years)
    """
    job_title = "Unknown Job Title"
    company_name = "Unknown Company"
    job_responsibilities = ""
    job_qualifications = ""
    experience_years = 0  # Default experience years is 0

   
    title_match = re.search(r"\*\*Job Title:\*\*\s*(.+)", job_description, re.IGNORECASE)
    if title_match:
        job_title = title_match.group(1).strip()

    
    company_match = re.search(r"\*\*Company:\*\*\s*(.+)", job_description, re.IGNORECASE)
    if company_match:
        company_name = company_match.group(1).strip()

    
    responsibilities_match = re.search(r"\*\*Job Responsibilities:\*\*\n(.*?)(?=\n\*\*|\Z)", job_description, re.DOTALL)
    if responsibilities_match:
        job_responsibilities = responsibilities_match.group(1).strip()

    
    qualifications_match = re.search(r"\*\*Required Skills & Qualifications:\*\*\n(.*?)(?=\n\*\*|\Z)", job_description, re.DOTALL)
    if qualifications_match:
        job_qualifications = qualifications_match.group(1).strip()

    
    experience_match = re.search(r"(\d+)\+\s*years?", job_qualifications)
    if experience_match:
        experience_years = int(experience_match.group(1))  # Converting extracted YOE to int

    return job_title, company_name, job_responsibilities, job_qualifications, experience_years

def vector_search(user_profile_embedding, job_descriptions, top_k=5):
    """
    Perform a vector search on job descriptions based on the user profile.

    Args:
        user_profile_embedding (list): The embedding of the user profile.
        job_descriptions (list): A list of job descriptions (strings).
        top_k (int): Number of top results to return.

    Returns:
        list: A list of top-k matching job titles, company names, and their scores.
    """
    scored_jobs = []

    for job_desc in job_descriptions:
        job_title, company_name, job_responsibilities, job_qualifications, job_experience = extract_job_details(job_desc)

        # Combine responsibilities and qualifications for embedding
        job_text = job_responsibilities + "\n" + job_qualifications
        job_embedding = generate_embedding(job_text)

        # Compute similarity score
        similarity_score = cosine_similarity([user_profile_embedding], [job_embedding])[0][0]


        # Normalize score to 0-100%
        final_score = max(0, min(100, similarity_score * 100))

        scored_jobs.append((job_title, company_name, final_score))

    # Sort by score in descending order
    scored_jobs.sort(key=lambda x: x[2], reverse=True)

    # Return top-k results
    return scored_jobs[:top_k]


if __name__ == "__main__":
    # Defining the user profile, an example(this will be embedded in the system)
    user_profile = text_cv
    

    # Generate embedding for this user profile
    user_profile_embedding = generate_embedding(user_profile)

    # Loading  job descriptions from the text file
    with open("C:/Users/Gospel/Documents/InternPlace/job_description.txt", "r", encoding="utf-8") as file:
        job_descriptions = file.read().split("///")  # Split job descriptions by "///"

    # Perform vector search
    recommendations = vector_search(user_profile_embedding, job_descriptions)

    # Printing recommendations
    print("\nTop job recommendations based on your profile:")
    for i, (job_title, company_name, score) in enumerate(recommendations, start=1):
        print(f"{i}. {job_title} at {company_name} (Suitability: {score:.2f}%)")
