# install libraries requirements.txt
# pip install -q PyPDF2
# pip install -q mistralai
# pip install numpy
# pip install python-dotenv

from PyPDF2 import PdfReader
from mistralai import Mistral
from dotenv import load_dotenv
import numpy as np
import os

# loading variables from .env file
load_dotenv()

# read resumes as list
resumes = []
for file in os.listdir("resumes-directory"):
  file_path = os.path.join("resumes-directory", file)
  resumes.append(PdfReader(file_path))

# extract the text in all the resumes and place in batches of 2
resumes_text = []
for resume in resumes:
  resumes_text.append(resume.pages[0].extract_text())

resume_batches = np.array_split(resumes_text, len(resumes_text) // 2)
resume_batches = [list(resume) for resume in resume_batches]

# load job description
job = open("Job.txt", "r").read()

# match and rank resumes using Mistral AI
model = "mistral-large-latest"

# authenticate mistral client
mistral = os.getenv("MISTRAL_API_KEY")

# create user role and message
messages = [

    {
        "role": "user",
        "content": f"""In terms of skills, which of these two resumes {batch_1} best matches this job description {job}.
                    Rank the resumes and return the result as a JSON object."""
    }
]

# generate result
best_match = mistral.chat.complete(
    model = model,
    messages = messages,
    response_format = {"type":"json_object"})

print(best_match.choices[0].message.content)

# {"ranking": [
#   {
#     "name": "Emma Rodriguez",
#     "rank": 1,
#     "reason": "Emma's resume mentions 'Full Stack Developer' with 8 years of experience, which aligns with the job title and exceeds the required 3+ years of experience. Additionally, her skills in UI/UX design align with the job responsibilities. However, she does not list specific programming languages or tools mentioned in the job description."
#   },
#   {
#     "name": "Isabella Chen",
#     "rank": 2,
#     "reason": "Isabella's resume is focused on marketing and does not mention any programming languages or tools relevant to the Full-Stack Software Developer role. Her experience and skills are not a good match for the job description."
#   }
# ]}
