from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import os
from functools import lru_cache
import logging
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
DATABASE_URL = os.getenv('DATABASE_URL', '')
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "tngtech/deepseek-r1t2-chimera:free"

# Big Five Personality Traits dimensions
PERSONALITY_DIMENSIONS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

vectorizer = TfidfVectorizer(max_features=500, stop_words='english')

@contextmanager
def get_db_connection():
    """Database connection context manager"""
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL)
        yield conn
        conn.commit()
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def init_db():
    """Initialize database tables"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS profiles (
                profile_id VARCHAR(255) PRIMARY KEY,
                name VARCHAR(255),
                bio TEXT,
                interests TEXT[],
                hobbies TEXT[],
                location VARCHAR(255),
                profession VARCHAR(255),
                values TEXT[],
                openness FLOAT DEFAULT 0.5,
                conscientiousness FLOAT DEFAULT 0.5,
                extraversion FLOAT DEFAULT 0.5,
                agreeableness FLOAT DEFAULT 0.5,
                neuroticism FLOAT DEFAULT 0.5,
                personality_analyzed BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_profile_personality 
            ON profiles(openness, conscientiousness, extraversion, agreeableness, neuroticism)
        """)
        logger.info("Database initialized successfully")

class ProfileMatcher:
    def __init__(self):
        self.profile_cache = {}
        
    def load_all_profiles(self):
        """Load all profiles from database"""
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM profiles")
            profiles = cursor.fetchall()
            
            self.profile_cache = {
                p['profile_id']: dict(p) for p in profiles
            }
        return self.profile_cache
    
    def get_profile(self, profile_id):
        """Get single profile from database"""
        with get_db_connection() as conn:
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            cursor.execute("SELECT * FROM profiles WHERE profile_id = %s", (profile_id,))
            profile = cursor.fetchone()
            return dict(profile) if profile else None
    
    def save_profile(self, profile_id, profile_data):
        """Save or update profile in database"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO profiles (
                    profile_id, name, bio, interests, hobbies, location, 
                    profession, values, openness, conscientiousness, 
                    extraversion, agreeableness, neuroticism, personality_analyzed
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (profile_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    bio = EXCLUDED.bio,
                    interests = EXCLUDED.interests,
                    hobbies = EXCLUDED.hobbies,
                    location = EXCLUDED.location,
                    profession = EXCLUDED.profession,
                    values = EXCLUDED.values,
                    openness = EXCLUDED.openness,
                    conscientiousness = EXCLUDED.conscientiousness,
                    extraversion = EXCLUDED.extraversion,
                    agreeableness = EXCLUDED.agreeableness,
                    neuroticism = EXCLUDED.neuroticism,
                    personality_analyzed = EXCLUDED.personality_analyzed,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                profile_id,
                profile_data.get('name', ''),
                profile_data.get('bio', ''),
                profile_data.get('interests', []),
                profile_data.get('hobbies', []),
                profile_data.get('location', ''),
                profile_data.get('profession', ''),
                profile_data.get('values', []),
                profile_data.get('openness', 0.5),
                profile_data.get('conscientiousness', 0.5),
                profile_data.get('extraversion', 0.5),
                profile_data.get('agreeableness', 0.5),
                profile_data.get('neuroticism', 0.5),
                profile_data.get('personality_analyzed', False)
            ))
    
    def delete_profile(self, profile_id):
        """Delete profile from database"""
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM profiles WHERE profile_id = %s", (profile_id,))
    
    def _create_profile_text(self, profile):
        """Convert profile to searchable text"""
        parts = [
            profile.get('bio', ''),
            ' '.join(profile.get('interests', [])),
            ' '.join(profile.get('hobbies', [])),
            profile.get('location', ''),
            profile.get('profession', ''),
            ' '.join(profile.get('values', []))
        ]
        return ' '.join(filter(None, parts))
    
    def calculate_personality_compatibility(self, profile1, profile2):
        """Calculate compatibility based on Big Five personality traits"""
        score = 0
        weights = {
            'openness': 0.2,
            'conscientiousness': 0.15,
            'extraversion': 0.25,
            'agreeableness': 0.25,
            'neuroticism': 0.15
        }
        
        for trait in PERSONALITY_DIMENSIONS:
            val1 = profile1.get(trait, 0.5)
            val2 = profile2.get(trait, 0.5)
            
            # Similarity for most traits (closer is better)
            if trait != 'neuroticism':
                trait_score = 1 - abs(val1 - val2)
            else:
                # For neuroticism, lower scores are generally better for compatibility
                trait_score = 1 - ((val1 + val2) / 2)
            
            score += trait_score * weights[trait]
        
        return score
    
    def find_matches(self, profile_id, top_k=10, min_personality_score=0.6):
        """Find top K matches combining text and personality"""
        target_profile = self.get_profile(profile_id)
        if not target_profile:
            return []
        
        # Load all profiles
        all_profiles = self.load_all_profiles()
        
        if len(all_profiles) <= 1:
            return []
        
        # Prepare texts for TF-IDF
        profile_ids = [pid for pid in all_profiles.keys() if pid != profile_id]
        texts = [self._create_profile_text(all_profiles[pid]) for pid in profile_ids]
        target_text = self._create_profile_text(target_profile)
        
        # Calculate text similarity
        all_texts = [target_text] + texts
        try:
            vectors = vectorizer.fit_transform(all_texts)
            text_similarities = cosine_similarity(vectors[0:1], vectors[1:])[0]
        except:
            text_similarities = np.zeros(len(texts))
        
        # Calculate personality compatibility
        matches = []
        for i, pid in enumerate(profile_ids):
            profile = all_profiles[pid]
            
            # Text similarity (40% weight)
            text_score = float(text_similarities[i])
            
            # Personality compatibility (60% weight)
            personality_score = self.calculate_personality_compatibility(target_profile, profile)
            
            # Combined score
            combined_score = (text_score * 0.4) + (personality_score * 0.6)
            
            if personality_score >= min_personality_score or text_score > 0.3:
                matches.append({
                    'profile_id': pid,
                    'score': combined_score,
                    'text_similarity': text_score,
                    'personality_compatibility': personality_score,
                    'profile': {
                        'name': profile.get('name'),
                        'bio': profile.get('bio'),
                        'interests': profile.get('interests'),
                        'location': profile.get('location')
                    }
                })
        
        # Sort by combined score
        matches.sort(key=lambda x: x['score'], reverse=True)
        return matches[:top_k]

matcher = ProfileMatcher()

@lru_cache(maxsize=50)
def analyze_personality_with_ai(profile_text):
    """Use DeepSeek to analyze personality traits"""
    if not OPENROUTER_API_KEY:
        return None
    
    prompt = f"""Analyze the personality of this person based on their profile using the Big Five personality model (OCEAN).

Profile: {profile_text[:800]}

Rate each trait from 0.0 to 1.0:
- Openness: creativity, curiosity, open to new experiences
- Conscientiousness: organized, responsible, goal-oriented
- Extraversion: outgoing, energetic, social
- Agreeableness: friendly, compassionate, cooperative
- Neuroticism: emotional instability, anxiety, moodiness

Respond ONLY in this JSON format:
{{"openness": 0.0-1.0, "conscientiousness": 0.0-1.0, "extraversion": 0.0-1.0, "agreeableness": 0.0-1.0, "neuroticism": 0.0-1.0, "summary": "brief personality summary"}}"""

    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 300
            },
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            content = data['choices'][0]['message']['content']
            
            # Extract JSON from response
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                json_str = content[start:end]
                personality = json.loads(json_str)
                return personality
        
        logger.error(f"OpenRouter error: {response.status_code}")
        return None
            
    except Exception as e:
        logger.error(f"AI personality analysis error: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM profiles")
            count = cursor.fetchone()[0]
        return jsonify({"status": "healthy", "profiles_count": count})
    except:
        return jsonify({"status": "unhealthy", "error": "Database connection failed"}), 500

@app.route('/profile', methods=['POST'])
def add_profile():
    """Add or update a profile with personality analysis"""
    try:
        data = request.json
        profile_id = data.get('profile_id')
        
        if not profile_id:
            return jsonify({"error": "profile_id required"}), 400
        
        profile_data = {
            'name': data.get('name', ''),
            'bio': data.get('bio', ''),
            'interests': data.get('interests', []),
            'hobbies': data.get('hobbies', []),
            'location': data.get('location', ''),
            'profession': data.get('profession', ''),
            'values': data.get('values', [])
        }
        
        # Analyze personality if not provided
        if data.get('analyze_personality', True) and OPENROUTER_API_KEY:
            profile_text = matcher._create_profile_text(profile_data)
            personality = analyze_personality_with_ai(profile_text)
            
            if personality:
                profile_data['openness'] = personality.get('openness', 0.5)
                profile_data['conscientiousness'] = personality.get('conscientiousness', 0.5)
                profile_data['extraversion'] = personality.get('extraversion', 0.5)
                profile_data['agreeableness'] = personality.get('agreeableness', 0.5)
                profile_data['neuroticism'] = personality.get('neuroticism', 0.5)
                profile_data['personality_analyzed'] = True
                profile_data['personality_summary'] = personality.get('summary', '')
        else:
            # Use provided personality scores or defaults
            profile_data['openness'] = data.get('openness', 0.5)
            profile_data['conscientiousness'] = data.get('conscientiousness', 0.5)
            profile_data['extraversion'] = data.get('extraversion', 0.5)
            profile_data['agreeableness'] = data.get('agreeableness', 0.5)
            profile_data['neuroticism'] = data.get('neuroticism', 0.5)
            profile_data['personality_analyzed'] = False
        
        matcher.save_profile(profile_id, profile_data)
        
        return jsonify({
            "message": "Profile added successfully",
            "profile_id": profile_id,
            "personality_analyzed": profile_data.get('personality_analyzed', False),
            "personality": {
                'openness': profile_data.get('openness'),
                'conscientiousness': profile_data.get('conscientiousness'),
                'extraversion': profile_data.get('extraversion'),
                'agreeableness': profile_data.get('agreeableness'),
                'neuroticism': profile_data.get('neuroticism')
            }
        }), 201
        
    except Exception as e:
        logger.error(f"Add profile error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/profile/<profile_id>', methods=['GET'])
def get_profile(profile_id):
    """Get a profile by ID"""
    try:
        profile = matcher.get_profile(profile_id)
        if not profile:
            return jsonify({"error": "Profile not found"}), 404
        return jsonify(profile), 200
    except Exception as e:
        logger.error(f"Get profile error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/profile/<profile_id>', methods=['DELETE'])
def delete_profile(profile_id):
    """Delete a profile"""
    try:
        matcher.delete_profile(profile_id)
        return jsonify({"message": "Profile deleted successfully"}), 200
    except Exception as e:
        logger.error(f"Delete profile error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/match/<profile_id>', methods=['GET'])
def get_matches(profile_id):
    """Get matches for a profile based on text and personality"""
    try:
        top_k = request.args.get('top_k', 10, type=int)
        min_personality = request.args.get('min_personality_score', 0.6, type=float)
        
        matches = matcher.find_matches(profile_id, top_k, min_personality)
        
        return jsonify({
            "profile_id": profile_id,
            "matches": matches,
            "count": len(matches)
        }), 200
        
    except Exception as e:
        logger.error(f"Get matches error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/profiles', methods=['GET'])
def list_profiles():
    """List all profiles"""
    try:
        profiles = matcher.load_all_profiles()
        return jsonify({
            "profiles": list(profiles.keys()),
            "count": len(profiles)
        }), 200
    except Exception as e:
        logger.error(f"List profiles error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/analyze-personality/<profile_id>', methods=['POST'])
def analyze_personality(profile_id):
    """Re-analyze personality for an existing profile"""
    try:
        profile = matcher.get_profile(profile_id)
        if not profile:
            return jsonify({"error": "Profile not found"}), 404
        
        profile_text = matcher._create_profile_text(profile)
        personality = analyze_personality_with_ai(profile_text)
        
        if personality:
            profile['openness'] = personality.get('openness', 0.5)
            profile['conscientiousness'] = personality.get('conscientiousness', 0.5)
            profile['extraversion'] = personality.get('extraversion', 0.5)
            profile['agreeableness'] = personality.get('agreeableness', 0.5)
            profile['neuroticism'] = personality.get('neuroticism', 0.5)
            profile['personality_analyzed'] = True
            
            matcher.save_profile(profile_id, profile)
            
            return jsonify({
                "message": "Personality analyzed successfully",
                "personality": personality
            }), 200
        else:
            return jsonify({"error": "Personality analysis failed"}), 500
            
    except Exception as e:
        logger.error(f"Analyze personality error: {e}")
        return jsonify({"error": str(e)}), 500

# Initialize database on startup (even when run with gunicorn)
try:
    init_db()
    logger.info("Database initialization completed successfully")
except Exception as e:
    logger.error(f"Database initialization failed: {e}")

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
