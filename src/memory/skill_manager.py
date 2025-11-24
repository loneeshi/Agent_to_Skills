# src/memory/skill_manager.py
"""
Skill Manager
Responsible for skill storage, retrieval and management
"""

import json
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import sqlite3
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SkillManager:
    """
    Skill manager, responsible for skill storage and retrieval
    """
    
    def __init__(
        self,
        db_path: str = "assets/skill_db/skills.db",
        vectorizer_path: str = "assets/skill_db/vectorizer.pkl",
        similarity_threshold: float = 0.7,
        max_skills: int = 1000
    ):
        """
        Initialize skill manager
        
        Args:
            db_path: Database path
            vectorizer_path: Vectorizer path
            similarity_threshold: Similarity threshold
            max_skills: Maximum number of skills
        """
        self.db_path = Path(db_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.similarity_threshold = similarity_threshold
        self.max_skills = max_skills
        
        # Initialize database
        self._init_database()
        
        # Initialize vectorizer
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self._load_or_create_vectorizer()
        
    def _init_database(self):
        """Initialize skill database"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create skills table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS skills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                skill_name TEXT NOT NULL,
                task_type TEXT,
                description TEXT,
                key_actions TEXT,
                environmental_cues TEXT,
                success_conditions TEXT,
                generalizability_score REAL,
                usage_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_task_type ON skills(task_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_generalizability ON skills(generalizability_score)')
        
        conn.commit()
        conn.close()
    
    def _load_or_create_vectorizer(self):
        """Load or create vectorizer"""
        if self.vectorizer_path.exists():
            try:
                import pickle
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
            except Exception as e:
                print(f"Failed to load vectorizer: {e}, creating new vectorizer")
                self.vectorizer = TfidfVectorizer(max_features=1000)
        else:
            self.vectorizer = TfidfVectorizer(max_features=1000)
    
    def add_skill(self, skill: Dict[str, Any]) -> bool:
        """
        Add new skill
        
        Args:
            skill: Skill dictionary
            
        Returns:
            Whether addition was successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if skill already exists
            cursor.execute(
                'SELECT id FROM skills WHERE skill_name = ?',
                (skill.get('skill_name'),)
            )
            if cursor.fetchone():
                print(f"Skill '{skill.get('skill_name')}' already exists")
                return False
            
            # Check if skill count has reached limit
            cursor.execute('SELECT COUNT(*) FROM skills')
            count = cursor.fetchone()[0]
            if count >= self.max_skills:
                # Delete skill with least usage
                cursor.execute(
                    'DELETE FROM skills WHERE id = (SELECT id FROM skills ORDER BY usage_count, created_at LIMIT 1)'
                )
            
            # Insert new skill
            cursor.execute('''
                INSERT INTO skills (
                    skill_name, task_type, description, key_actions,
                    environmental_cues, success_conditions, generalizability_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                skill.get('skill_name'),
                skill.get('task_type'),
                skill.get('description'),
                json.dumps(skill.get('key_actions', [])),
                json.dumps(skill.get('environmental_cues', [])),
                json.dumps(skill.get('success_conditions', [])),
                skill.get('generalizability_score', 0.5)
            ))
            
            conn.commit()
            conn.close()
            
            # Update vectorizer
            self._update_vectorizer()
            
            return True
            
        except Exception as e:
            print(f"Failed to add skill: {e}")
            return False
    
    def retrieve_skills(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant skills
        
        Args:
            query: Query text
            top_k: Return top k skills
            
        Returns:
            List of relevant skills
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all skills
            cursor.execute('''
                SELECT id, skill_name, task_type, description, key_actions,
                       environmental_cues, success_conditions, generalizability_score
                FROM skills
                ORDER BY generalizability_score DESC, usage_count DESC
            ''')
            
            skills = cursor.fetchall()
            conn.close()
            
            if not skills:
                return []
            
            # Parse skill data
            skill_dicts = []
            skill_texts = []
            
            for skill in skills:
                skill_dict = {
                    'id': skill[0],
                    'skill_name': skill[1],
                    'task_type': skill[2],
                    'description': skill[3],
                    'key_actions': json.loads(skill[4]),
                    'environmental_cues': json.loads(skill[5]),
                    'success_conditions': json.loads(skill[6]),
                    'generalizability_score': skill[7]
                }
                skill_dicts.append(skill_dict)
                
                # Build text for similarity calculation
                skill_text = f"{skill[1]} {skill[2]} {skill[3]}"
                skill_texts.append(skill_text)
            
            # Calculate similarity
            if skill_texts:
                skill_vectors = self.vectorizer.transform(skill_texts)
                query_vector = self.vectorizer.transform([query])
                similarities = cosine_similarity(query_vector, skill_vectors)[0]
                
                # Sort by similarity
                scored_skills = list(zip(skill_dicts, similarities))
                scored_skills.sort(key=lambda x: x[1], reverse=True)
                
                # Return top k skills with similarity above threshold
                relevant_skills = []
                for skill, similarity in scored_skills[:top_k]:
                    if similarity >= self.similarity_threshold:
                        skill['similarity_score'] = float(similarity)
                        relevant_skills.append(skill)
                
                # Update usage count
                self._update_usage_count([skill['id'] for skill in relevant_skills])
                
                return relevant_skills
            
            return []
            
        except Exception as e:
            print(f"Failed to retrieve skills: {e}")
            return []
    
    def _update_usage_count(self, skill_ids: List[int]):
        """Update skill usage count"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for skill_id in skill_ids:
                cursor.execute(
                    'UPDATE skills SET usage_count = usage_count + 1, updated_at = ? WHERE id = ?',
                    (datetime.now(), skill_id)
                )
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Failed to update usage count: {e}")
    
    def _update_vectorizer(self):
        """Update vectorizer"""
        try:
            # Retrain vectorizer
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT skill_name, task_type, description FROM skills')
            skills = cursor.fetchall()
            conn.close()
            
            if skills:
                skill_texts = [f"{skill[0]} {skill[1]} {skill[2]}" for skill in skills]
                self.vectorizer.fit(skill_texts)
                
                # Save vectorizer
                import pickle
                with open(self.vectorizer_path, 'wb') as f:
                    pickle.dump(self.vectorizer, f)
                    
        except Exception as e:
            print(f"Failed to update vectorizer: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get skill statistics
        
        Returns:
            Statistics information dictionary
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total skill count
            cursor.execute('SELECT COUNT(*) FROM skills')
            total_skills = cursor.fetchone()[0]
            
            # Group by task type
            cursor.execute('''
                SELECT task_type, COUNT(*) as count
                FROM skills
                GROUP BY task_type
            ''')
            task_type_stats = dict(cursor.fetchall())
            
            # Average generalizability score
            cursor.execute('SELECT AVG(generalizability_score) FROM skills')
            avg_generalizability = cursor.fetchone()[0] or 0
            
            # Total usage count
            cursor.execute('SELECT SUM(usage_count) FROM skills')
            total_usage = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                'total_skills': total_skills,
                'task_type_statistics': task_type_stats,
                'average_generalizability_score': float(avg_generalizability),
                'total_usage_count': total_usage,
                'database_path': str(self.db_path)
            }
            
        except Exception as e:
            print(f"Failed to get statistics: {e}")
            return {}
    
    def export_skills(self, output_path: str):
        """
        Export skills to JSON file
        
        Args:
            output_path: Output file path
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT skill_name, task_type, description, key_actions,
                       environmental_cues, success_conditions, generalizability_score,
                       usage_count, created_at
                FROM skills
                ORDER BY generalizability_score DESC
            ''')
            
            skills = cursor.fetchall()
            conn.close()
            
            skill_list = []
            for skill in skills:
                skill_dict = {
                    'skill_name': skill[0],
                    'task_type': skill[1],
                    'description': skill[2],
                    'key_actions': json.loads(skill[3]),
                    'environmental_cues': json.loads(skill[4]),
                    'success_conditions': json.loads(skill[5]),
                    'generalizability_score': skill[6],
                    'usage_count': skill[7],
                    'created_at': skill[8]
                }
                skill_list.append(skill_dict)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(skill_list, f, ensure_ascii=False, indent=2)
                
            print(f"Skills exported to: {output_path}")
            
        except Exception as e:
            print(f"Failed to export skills: {e}")
