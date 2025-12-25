from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from functools import lru_cache
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
DATABASE_URL = os.getenv('DATABASE_URL', '')

# Big Five Personality Traits dimensions
PERSONALITY_DIMENSIONS = ['openness', 'conscientiousness', 'extraversion', 'agreeableness', 'neuroticism']

# Interest to personality mapping
INTEREST_TO_PERSONALITY = {
    # Openness boosters
    'art': {'openness': 0.25},
    'painting': {'openness': 0.25},
    'drawing': {'openness': 0.20},
    'music': {'openness': 0.20},
    'writing': {'openness': 0.20},
    'poetry': {'openness': 0.25},
    'travel': {'openness': 0.30, 'extraversion': 0.10},
    'photography': {'openness': 0.20},
    'movies': {'openness': 0.15},
    'theater': {'openness': 0.20},
    'dancing': {'openness': 0.20, 'extraversion': 0.15},
    'philosophy': {'openness': 0.30},
    'science': {'openness': 0.20},
    'learning': {'openness': 0.25, 'conscientiousness': 0.10},
    'exploring': {'openness': 0.30},
    'adventure': {'openness': 0.30, 'extraversion': 0.10},
    'culture': {'openness': 0.25},
    
    # Conscientiousness boosters
    'planning': {'conscientiousness': 0.30},
    'organizing': {'conscientiousness': 0.30},
    'studying': {'conscientiousness': 0.25},
    'working out': {'conscientiousness': 0.20},
    'fitness': {'conscientiousness': 0.20},
    'gym': {'conscientiousness': 0.20},
    'running': {'conscientiousness': 0.15},
    'coding': {'conscientiousness': 0.25, 'openness': 0.10},
    'programming': {'conscientiousness': 0.25, 'openness': 0.10},
    'reading': {'conscientiousness': 0.15, 'openness': 0.20},
    'gardening': {'conscientiousness': 0.20},
    'cooking': {'conscientiousness': 0.15},
    'cleaning': {'conscientiousness': 0.25},
    
    # Extraversion boosters
    'parties': {'extraversion': 0.35},
    'socializing': {'extraversion': 0.35},
    'networking': {'extraversion': 0.30},
    'public speaking': {'extraversion': 0.35},
    'team sports': {'extraversion': 0.25, 'agreeableness': 0.10},
    'dancing': {'extraversion': 0.25, 'openness': 0.10},
    'bars': {'extraversion': 0.25},
    'clubs': {'extraversion': 0.30},
    'concerts': {'extraversion': 0.20},
    'festivals': {'extraversion': 0.25},
    'meeting people': {'extraversion': 0.35},
    'hosting': {'extraversion': 0.30, 'agreeableness': 0.10},
    'karaoke': {'extraversion': 0.30},
    
    # Extraversion reducers (introverted activities)
    'reading': {'extraversion': -0.20, 'openness': 0.20},
    'meditation': {'extraversion': -0.15, 'neuroticism': -0.20},
    'yoga': {'extraversion': -0.10, 'neuroticism': -0.20},
    'gaming': {'extraversion': -0.15, 'openness': 0.10},
    'video games': {'extraversion': -0.15},
    'solo hiking': {'extraversion': -0.15, 'openness': 0.15},
    'writing': {'extraversion': -0.10, 'openness': 0.20},
    
    # Agreeableness boosters
    'volunteering': {'agreeableness': 0.35, 'conscientiousness': 0.15},
    'helping others': {'agreeableness': 0.35},
    'charity': {'agreeableness': 0.30},
    'teaching': {'agreeableness': 0.25, 'conscientiousness': 0.10},
    'mentoring': {'agreeableness': 0.30},
    'animals': {'agreeableness': 0.20},
    'pets': {'agreeableness': 0.20},
    'community': {'agreeableness': 0.25, 'extraversion': 0.10},
    'family': {'agreeableness': 0.25},
    'friends': {'agreeableness': 0.20, 'extraversion': 0.15},
    'collaboration': {'agreeableness': 0.25},
    'teamwork': {'agreeableness': 0.25},
    
    # Neuroticism reducers (emotional stability)
    'meditation': {'neuroticism': -0.30},
    'yoga': {'neuroticism': -0.25},
    'mindfulness': {'neuroticism': -0.30},
    'relaxation': {'neuroticism': -0.25},
    'nature': {'neuroticism': -0.20, 'openness': 0.15},
    'hiking': {'neuroticism': -0.15, 'openness': 0.15},
    'camping': {'neuroticism': -0.15, 'openness': 0.15},
    'spa': {'neuroticism': -0.20},
    'massage': {'neuroticism': -0.20},
    'beach': {'neuroticism': -0.15},
    
    # Mixed
    'sports': {'extraversion': 0.15, 'conscientiousness': 0.10},
    'music': {'openness': 0.20, 'agreeableness': 0.05},
    'cooking': {'conscientiousness': 0.15, 'agreeableness': 0.10},
    'baking': {'conscientiousness': 0.20, 'agreeableness': 0.10},
}

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
        
        # Create table if not exists
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
                personality_method VARCHAR(50) DEFAULT 'hybrid',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Add personality_method column if it doesn't exist (for existing tables)
        try:
            cursor.execute("""
                ALTER TABLE profiles 
                ADD COLUMN IF NOT EXISTS personality_method VARCHAR(50) DEFAULT 'hybrid'
            """)
        except Exception as e:
            logger.warning(f"Column personality_method might already exist: {e}")
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_profile_personality 
            ON profiles(openness, conscientiousness, extraversion, agreeableness, neuroticism)
        """)
        logger.info("Database initialized successfully")

def analyze_personality_from_interests(interests):
    """Analyze personality based on interests"""
    scores = {
        'openness': 0.5,
        'conscientiousness': 0.5,
        'extraversion': 0.5,
        'agreeableness': 0.5,
        'neuroticism': 0.5
    }
    
    if not interests:
        return scores
    
    # Process each interest
    for interest in interests:
        interest_lower = interest.lower().strip()
        
        # Exact match
        if interest_lower in INTEREST_TO_PERSONALITY:
            modifiers = INTEREST_TO_PERSONALITY[interest_lower]
            for trait, modifier in modifiers.items():
                scores[trait] = max(0.0, min(1.0, scores[trait] + modifier))
        else:
            # Partial match (if interest contains keyword)
            for keyword, modifiers in INTEREST_TO_PERSONALITY.items():
                if keyword in interest_lower or interest_lower in keyword:
                    for trait, modifier in modifiers.items():
                        # Apply half the modifier for partial matches
                        scores[trait] = max(0.0, min(1.0, scores[trait] + (modifier * 0.5)))
                    break
    
    return scores

def analyze_personality_from_questionnaire(questionnaire):
    """
    Analyze personality from Big Five questionnaire responses
    
    Expected format:
    {
        'openness': [5, 4, 5, 4, 5],  # 1-5 ratings
        'conscientiousness': [4, 5, 4, 4, 5],
        'extraversion': [2, 2, 1, 2, 3],
        'agreeableness': [4, 4, 3, 4, 5],
        'neuroticism': [2, 3, 2, 2, 1]
    }
    """
    scores = {}
    
    for trait, ratings in questionnaire.items():
        if trait in PERSONALITY_DIMENSIONS and ratings:
            # Calculate average rating
            avg_rating = sum(ratings) / len(ratings)
            # Normalize from 1-5 scale to 0-1 scale
            scores[trait] = (avg_rating - 1) / 4
        else:
            scores[trait] = 0.5
    
    return scores

def analyze_personality_hybrid(interests=None, questionnaire=None):
    """
    Combine interests and questionnaire for personality analysis
    Questionnaire gets 70% weight (more reliable)
    Interests get 30% weight
    """
    final_scores = {
        'openness': 0.5,
        'conscientiousness': 0.5,
        'extraversion': 0.5,
        'agreeableness': 0.5,
        'neuroticism': 0.5
    }
    
    weighted_scores = []
    
    # Interests-based (30% weight)
    if interests:
        interest_scores = analyze_personality_from_interests(interests)
        weighted_scores.append((interest_scores, 0.3))
    
    # Questionnaire-based (70% weight)
    if questionnaire:
        q_scores = analyze_personality_from_questionnaire(questionnaire)
        weighted_scores.append((q_scores, 0.7))
    
    # Calculate weighted average
    if weighted_scores:
        for trait in final_scores:
            weighted_sum = sum(scores[trait] * weight for scores, weight in weighted_scores)
            total_weight = sum(weight for _, weight in weighted_scores)
            final_scores[trait] = weighted_sum / total_weight
    
    return final_scores

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
                    extraversion, agreeableness, neuroticism, personality_method
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
                    personality_method = EXCLUDED.personality_method,
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
                profile_data.get('personality_method', 'hybrid')
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
            'openness': 0.20,
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

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM profiles")
            count = cursor.fetchone()[0]
        
        return jsonify({
            "status": "healthy",
            "profiles_count": count,
            "personality_method": "hybrid (interests + questionnaire)",
            "database_connected": True
        })
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "database_connected": False
        }), 500

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
        
        # Get questionnaire if provided
        questionnaire = data.get('questionnaire', None)
        
        # Analyze personality using hybrid method
        personality_scores = analyze_personality_hybrid(
            interests=profile_data['interests'],
            questionnaire=questionnaire
        )
        
        # Add personality scores to profile
        profile_data['openness'] = float(personality_scores['openness'])
        profile_data['conscientiousness'] = float(personality_scores['conscientiousness'])
        profile_data['extraversion'] = float(personality_scores['extraversion'])
        profile_data['agreeableness'] = float(personality_scores['agreeableness'])
        profile_data['neuroticism'] = float(personality_scores['neuroticism'])
        
        # Set method used
        if questionnaire and profile_data['interests']:
            profile_data['personality_method'] = 'hybrid'
        elif questionnaire:
            profile_data['personality_method'] = 'questionnaire'
        else:
            profile_data['personality_method'] = 'interests'
        
        matcher.save_profile(profile_id, profile_data)
        
        return jsonify({
            "message": "Profile added successfully",
            "profile_id": profile_id,
            "personality_method": profile_data['personality_method'],
            "personality": {
                'openness': profile_data['openness'],
                'conscientiousness': profile_data['conscientiousness'],
                'extraversion': profile_data['extraversion'],
                'agreeableness': profile_data['agreeableness'],
                'neuroticism': profile_data['neuroticism']
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

@app.route('/analyze-personality', methods=['POST'])
def analyze_personality_endpoint():
    """Analyze personality from interests and/or questionnaire"""
    try:
        data = request.json
        interests = data.get('interests', [])
        questionnaire = data.get('questionnaire', None)
        
        if not interests and not questionnaire:
            return jsonify({
                "error": "Either interests or questionnaire required"
            }), 400
        
        personality = analyze_personality_hybrid(interests, questionnaire)
        
        method = 'hybrid' if (interests and questionnaire) else ('questionnaire' if questionnaire else 'interests')
        
        return jsonify({
            "personality": personality,
            "method": method
        }), 200
        
    except Exception as e:
        logger.error(f"Analyze personality error: {e}")
        return jsonify({"error": str(e)}), 500

# Initialize database on startup
try:
    init_db()
    logger.info("Database initialization completed successfully")
except Exception as e:
    logger.error(f"Database initialization failed: {e}")

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
