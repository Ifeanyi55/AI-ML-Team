import re
#run "pip install sentence-transformers" to install sentence transformers
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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

def vector_search(user_profile_embedding, user_experience, job_descriptions, top_k=5):
    """
    Perform a vector search on job descriptions based on the user profile.

    Args:
        user_profile_embedding (list): The embedding of the user profile.
        user_experience (int): The years of experience the user has.
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

        # Check if the job title contains 'intern' or if the experience required is 0
        if job_experience > 0 and 'intern' not in job_title.lower():
            # Adjust score based on experience match
            experience_penalty = 0.0
            if job_experience > 0:  # Only consider experience if specified
                if user_experience < job_experience:
                    experience_penalty = (job_experience - user_experience) * 0.05  # Small penalty per year missing
                elif user_experience >= job_experience:
                    similarity_score += 0.05  # Bonus for meeting or exceeding required experience

            # Apply experience penalty
            similarity_score -= experience_penalty

        # Normalize score to 0-100%
        final_score = max(0, min(100, similarity_score * 100))

        scored_jobs.append((job_title, company_name, final_score))

    # Sort by score in descending order
    scored_jobs.sort(key=lambda x: x[2], reverse=True)

    # Return top-k results
    return scored_jobs[:top_k]


if __name__ == "__main__":
    # Defining the user profile, an example(this will be embedded in the system)
    user_profile = "machine learning, Python, data analysis, and cloud computing"
    user_experience = 4  # Example: User has 4 years of experience

    # Generate embedding for this user profile
    user_profile_embedding = generate_embedding(user_profile)

    # Loading  job descriptions from the text file
    with open("job_descriptions.txt", "r", encoding="utf-8") as file:
        job_descriptions = file.read().split("///")  # Split job descriptions by "///"

    # Perform vector search
    recommendations = vector_search(user_profile_embedding, user_experience, job_descriptions)

    # Printing recommendations
    print("\nTop job recommendations based on your profile:")
    for i, (job_title, company_name, score) in enumerate(recommendations, start=1):
        print(f"{i}. {job_title} at {company_name} (Suitability: {score:.2f}%)")
