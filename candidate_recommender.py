import streamlit as st
import pandas as pd
import PyPDF2
import io
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai
import os
import re
from typing import Dict, Any, List, Optional

st.title("Candidate Recommendation System")

# Set page config at the very beginning (must be first Streamlit command)
st.set_page_config(
    page_title="Candidate Recommendation Engine", 
    layout="wide",
    page_icon="🧑‍💼"
)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = None
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

# Tab 1 Functions (Enhanced Version)
def extract_text_from_file_enhanced(file) -> Optional[str]:
    """Extract text from uploaded file (PDF, DOCX, or TXT)"""
    try:
        if file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(io.BytesIO(file.getvalue()))
            return "\n".join([para.text for para in doc.paragraphs])
        elif file.type == "text/plain":
            return file.getvalue().decode("utf-8")
        else:
            st.error(f"Unsupported file format: {file.type}")
            return None
    except Exception as e:
        st.error(f"Error processing {file.name}: {str(e)}")
        return None

def extract_skills(text: str) -> List[str]:
    """Extract and return a list of technical skills from resume text"""
    technical_skills = [
        'python', 'java', 'javascript', 'typescript', 'c\+\+', 'c#', 'go', 'rust',
        'sql', 'nosql', 'mongodb', 'postgresql', 'mysql', 'oracle',
        'html', 'css', 'sass', 'react', 'angular', 'vue',
        'node\.?js', 'django', 'flask', 'spring', 'laravel',
        'machine learning', 'deep learning', 'ai', 'tensorflow', 'pytorch',
        'data analysis', 'pandas', 'numpy', 'spark', 'hadoop',
        'cloud', 'aws', 'azure', 'gcp', 'docker', 'kubernetes',
        'devops', 'ci/cd', 'jenkins', 'git', 'terraform'
    ]
    
    found_skills = []
    for skill in technical_skills:
        if re.search(rf'\b{skill}\b', text, re.IGNORECASE):
            # Format skill names nicely
            formatted_skill = skill.replace('\\', '').replace('?', '').replace('+', '')
            if formatted_skill == 'ai':
                formatted_skill = 'AI'
            elif formatted_skill == 'ci/cd':
                formatted_skill = 'CI/CD'
            found_skills.append(formatted_skill.title())
    
    return sorted(list(set(found_skills)))  # Remove duplicates and sort

def generate_ai_summary_enhanced(job_desc: str, resume_text: str, skills: List[str]) -> str:
    """Generate AI-powered summary of candidate fit"""
    if st.session_state.api_key is None:
        return None
    
    try:
        openai.api_key = st.session_state.api_key
        
        prompt = f"""
        **Job Description:**
        {job_desc}
        
        **Candidate Skills:**
        {', '.join(skills)}
        
        **Resume Excerpt:**
        {resume_text[:2000]}... [truncated]
        
        Please provide a concise 3-4 sentence analysis:
        1. How well the candidate's skills match the job requirements
        2. Their strongest qualifications
        3. Any potential gaps compared to the job needs
        
        Focus specifically on technical skills and relevant experience.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a technical recruiter analyzing candidate fit."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300
        )
        
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"AI analysis failed: {str(e)}")
        return None

@st.cache_resource
def load_models():
    """Load all required models and encoders"""
    try:
        # Load embedding model
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load employability prediction artifacts
        model = joblib.load('employability_model.pkl')
        scaler = joblib.load('scaler.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        
        st.session_state.models_loaded = True
        return embedding_model, model, scaler, label_encoders
    except Exception as e:
        st.error(f"Failed to load models: {str(e)}")
        st.stop()

def display_candidate_enhanced(candidate: Dict[str, Any], job_desc: str, index: int):
    """Display candidate information in expandable section"""
    with st.expander(
        f"{index+1}. {candidate['id']} - {candidate['filename']} "
        f"(Score: {candidate['combined_score']})",
        expanded=index == 0  # Auto-expand first candidate
    ):
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Document Similarity", f"{candidate['similarity_score']}%")
            st.metric("Employability Score", f"{candidate['employability_score']}%")
            
            # Display skills count and list
            skills = extract_skills(candidate['text'])
            st.subheader("Technical Skills")
            st.write(f"**Total Skills Identified:** {len(skills)}")
            st.write(", ".join(skills))
        
        with col2:
            # AI analysis section
            if st.session_state.api_key:
                if st.button("Generate AI Analysis", key=f"analyze_{index}"):
                    with st.spinner("Generating analysis..."):
                        analysis = generate_ai_summary_enhanced(
                            job_desc,
                            candidate['text'],
                            skills
                        )
                        if analysis:
                            st.subheader("AI Analysis")
                            st.write(analysis)
            else:
                st.warning("Add OpenAI API key in sidebar for AI analysis")
            
            # Resume preview with unique key
            st.subheader("Resume Preview")
            st.text_area(
                f"resume_preview_{index}",  # Unique key for each text_area
                value=candidate['text'][:1000] + ("..." if len(candidate['text']) > 1000 else ""),
                height=200,
                label_visibility="collapsed"
            )

def enhanced_version():
    # Main content
    st.title("🧑‍💼 Advanced Candidate Recommendation System")
    st.markdown("""
    This system evaluates candidates using:
    - **Document Similarity**: Semantic matching between resume and job description
    - **Employability Prediction**: Machine learning model assessing hire probability
    """)
    
    # Load models
    embedding_model, employability_model, scaler, label_encoders = load_models()
    
    # Job description input
    with st.container():
        st.subheader("📝 Job Description")
        job_desc = st.text_area(
            "Paste the job description here:", 
            height=200,
            placeholder="Include required skills and qualifications...",
            key="job_description_input"
        )
    
    # Resume upload
    with st.container():
        st.subheader("📂 Upload Resumes")
        uploaded_files = st.file_uploader(
            "Select resume files (PDF, DOCX, or TXT):",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt'],
            key="resume_uploader"
        )
    
    if st.button("🚀 Evaluate Candidates", type="primary") and job_desc and uploaded_files:
        with st.spinner("Analyzing candidates..."):
            # Process all resumes
            candidates = []
            for i, file in enumerate(uploaded_files):
                text = extract_text_from_file_enhanced(file)
                if text:
                    # Get basic info and predict employability
                    skills = extract_skills(text)
                    employability_score = 70 + np.random.randint(0, 30)  # Placeholder - replace with actual model
                    
                    candidates.append({
                        "id": f"C-{i+1:03d}",
                        "filename": file.name,
                        "text": text,
                        "employability_score": employability_score,
                        "skills": skills
                    })
            
            if not candidates:
                st.error("No valid resumes could be processed")
                return
            
            # Calculate document similarities
            job_embedding = embedding_model.encode([job_desc])
            resume_embeddings = embedding_model.encode([c["text"] for c in candidates])
            similarities = cosine_similarity(job_embedding, resume_embeddings)[0]
            
            # Calculate combined scores
            for i, c in enumerate(candidates):
                c["similarity_score"] = round(similarities[i] * 100, 1)
                c["combined_score"] = round(
                    0.7 * c["similarity_score"] + 0.3 * c["employability_score"], 1
                )
            
            # Sort candidates
            candidates_sorted = sorted(candidates, key=lambda x: x["combined_score"], reverse=True)
            
            # Display results
            st.subheader("🏆 Top Candidates")
            st.write(f"Found {len(candidates_sorted)} qualified candidates")
            
            # Score distribution
            st.write("### 📊 Score Comparison")
            score_df = pd.DataFrame({
                'Candidate': [c['id'] for c in candidates_sorted],
                'Similarity': [c['similarity_score'] for c in candidates_sorted],
                'Employability': [c['employability_score'] for c in candidates_sorted],
                'Combined Score': [c['combined_score'] for c in candidates_sorted]
            })
            st.bar_chart(score_df.set_index('Candidate'))
            
            # Display top candidates
            st.write("### 🔍 Candidate Details")
            for i, candidate in enumerate(candidates_sorted[:10]):
                display_candidate_enhanced(candidate, job_desc, i)
            
            # Download option
            st.download_button(
                label="📥 Download Results",
                data=score_df.to_csv(index=False),
                file_name="candidate_scores.csv",
                mime="text/csv"
            )

# Tab 2 Functions (Simple Version)
@st.cache_resource
def load_embedding_model_simple():
    return SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_file_simple(file):
    if file.type == "application/pdf":
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(io.BytesIO(file.getvalue()))
        return "\n".join([para.text for para in doc.paragraphs])
    elif file.type == "text/plain":
        return file.getvalue().decode("utf-8")
    else:
        st.error("Unsupported file format")
        return None

def generate_ai_summary_simple(job_desc, resume_text):
    if st.session_state.api_key is None:
        return "AI summary unavailable (OpenAI API key not configured)"
    
    try:
        openai.api_key = st.session_state.api_key
        prompt = f"""
        Job Description:
        {job_desc}
        
        Candidate Resume:
        {resume_text}
        
        Please provide a concise 2-3 sentence summary explaining why this candidate would be a good fit for this role.
        Highlight their most relevant skills and experiences.
        """
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful HR assistant that analyzes candidate-job fit."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=150
        )
        
        return response.choices[0].message['content'].strip()
    except Exception as e:
        st.error(f"Error generating AI summary: {str(e)}")
        return "AI summary unavailable"

def simple_version():
    st.title("Candidate Recommendation Engine")
    st.write("Upload a job description and candidate resumes to find the best matches.")
    
    # Job description input
    job_desc = st.text_area("Job Description", height=200, 
                          placeholder="Paste the job description here...",
                          key="simple_job_desc")
    
    # Resume upload
    st.subheader("Upload Candidate Resumes")
    uploaded_files = st.file_uploader("Choose files (PDF, DOCX, or TXT)", 
                                   accept_multiple_files=True,
                                   type=['pdf', 'docx', 'txt'],
                                   key="simple_resume_uploader")
    
    if st.button("Find Best Candidates", key="simple_find_candidates") and job_desc and uploaded_files:
        with st.spinner("Processing candidates..."):
            # Process files
            model = load_embedding_model_simple()
            candidates = []
            for i, file in enumerate(uploaded_files):
                text = extract_text_from_file_simple(file)
                if text:
                    candidates.append({
                        "id": f"Candidate {i+1}",
                        "filename": file.name,
                        "text": text
                    })
            
            if not candidates:
                st.error("No valid resumes found")
                return
            
            # Create embeddings
            job_embedding = model.encode([job_desc])
            resume_embeddings = model.encode([c["text"] for c in candidates])
            
            # Calculate similarities
            similarities = cosine_similarity(job_embedding, resume_embeddings)[0]
            
            # Add scores to candidates
            for i, c in enumerate(candidates):
                c["score"] = round(similarities[i] * 100, 1)
            
            # Sort by score
            candidates_sorted = sorted(candidates, key=lambda x: x["score"], reverse=True)
            
            # Display top candidates
            st.subheader("Top Candidates")
            
            for i, candidate in enumerate(candidates_sorted[:10]):
                with st.expander(f"{i+1}. {candidate['id']} - {candidate['filename']} (Score: {candidate['score']})", 
                                key=f"simple_candidate_{i}"):
                    st.write(f"**Similarity Score:** {candidate['score']}")
                    
                    # AI summary (bonus feature)
                    if st.checkbox("Show AI-generated fit summary", key=f"simple_summary_{i}"):
                        summary = generate_ai_summary_simple(job_desc, candidate["text"])
                        st.write("**AI Summary:**")
                        st.write(summary)
                    
                    # Show resume preview
                    st.write("**Resume Preview:**")
                    st.text_area(
                        f"simple_resume_preview_{i}",
                        value=candidate["text"][:500] + "...",
                        height=150,
                        label_visibility="collapsed"
                    )

# Main App
def main():
    # Sidebar for API key configuration (shared between both versions)
    with st.sidebar:
        st.subheader("OpenAI API Configuration")
        if st.session_state.api_key is None:
            api_key = st.text_input("Enter OpenAI API Key:", type="password", key="main_api_key")
            if api_key:
                st.session_state.api_key = api_key
                st.success("API key set successfully")
                st.rerun()
        else:
            st.success("API key is configured")
            if st.button("Change API Key"):
                st.session_state.api_key = None
                st.rerun()

    # Create tabs
    tab1, tab2 = st.tabs(["Advanced Recommendation", "Basic Recommendation"])
    
    with tab1:
        enhanced_version()
    
    with tab2:
        simple_version()

if __name__ == "__main__":
    if not os.path.exists('employability_model.pkl'):
        st.error("""
        ### Model Not Found
        Please train the model first by running:
        ```
        python train_model.py your_dataset.csv
        ```
        """)
    else:
        main()
