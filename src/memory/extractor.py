# src/memory/extractor.py
"""
Skill Extractor
Responsible for extracting reusable skills from successful task execution traces
"""

import os
import json
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
import yaml

from agentscope.model import DashScopeChatModel, OpenAIChatModel
from agentscope.message import Msg


class SkillExtractor:
    """
    Skill extractor, uses LLM to extract skills from execution traces
    """
    
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize skill extractor
        
        Args:
            model_config: Model configuration
        """
        self.model = OpenAIChatModel(
            model_name="DeepSeek-R1",
            # model_name="Qwen3-235B-A22B",
            api_key=os.environ["DEEPSEEK_API_KEY"],
            client_args={
                "base_url": "https://ai.api.coregpu.cn/v1"
            },
            stream=False,
        )
        self.extraction_prompt = self._load_extraction_prompt()
        
    def _load_extraction_prompt(self) -> str:
        """
        Load skill extraction prompt
        
        Returns:
            Extraction prompt template
        """
        prompt_path = Path("assets/prompts/reflection_prompts.yaml")
        if prompt_path.exists():
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompts = yaml.safe_load(f)
                return prompts.get('skill_extraction', self._get_default_prompt())
        return self._get_default_prompt()
    
    def _get_default_prompt(self) -> str:
        """
        Get default extraction prompt
        
        Returns:
            Default prompt
        """
        return """Please analyze the following successful task execution trace and extract reusable skills:

Task description: {task_description}
Execution steps: {execution_steps}

Please extract:
1. Key skill patterns
2. Generalizable action sequences  
3. Important environmental observation points
4. Potential failure avoidance strategies

Return extracted skills in JSON format as follows:
{
  "skills": [
    {
      "skill_name": "skill_name",
      "task_type": "task_type",
      "description": "skill_description",
      "key_actions": ["key_action1", "key_action2"],
      "environmental_cues": ["environmental_cue1", "environmental_cue2"],
      "success_conditions": ["success_condition1", "success_condition2"],
      "generalizability_score": 0.8
    }
  ]
}"""
    
    async def extract_skills(
        self,
        task_description: str,
        execution_trace: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract skills from execution trace
        
        Args:
            task_description: Task description
            execution_trace: Execution trace
            
        Returns:
            List of extracted skills
        """
        try:
            # Format execution steps
            execution_steps = self._format_execution_steps(execution_trace)
            
            # Build extraction prompt
            extraction_prompt = self.extraction_prompt.format(
                task_description=task_description,
                execution_steps=execution_steps
            )
            
            # Call model for extraction
            msg = Msg(
                name="system",
                content=extraction_prompt,
                role="system"
            )
            
            response = await self.model(msg)
            
            if not response or not response.content:
                return []
            
            # Parse extraction results
            skills = self._parse_extraction_response(response.content)
            
            # Validate and clean skill data
            validated_skills = self._validate_skills(skills)
            
            return validated_skills
            
        except Exception as e:
            print(f"Skill extraction failed: {e}")
            return []
    
    def _format_execution_steps(self, execution_trace: List[Dict[str, Any]]) -> str:
        """
        Format execution steps as text
        
        Args:
            execution_trace: Execution trace
            
        Returns:
            Formatted execution steps text
        """
        if not execution_trace:
            return "No execution records"
        
        steps_text = ""
        for i, step in enumerate(execution_trace, 1):
            steps_text += f"Step {i}:\n"
            steps_text += f"Task: {step.get('task', 'Unknown')}\n"
            steps_text += f"Prompt: {step.get('prompt', 'None')}\n"
            steps_text += f"Response: {step.get('response', 'None')}\n"
            steps_text += "---\n"
        
        return steps_text.strip()
    
    def _parse_extraction_response(self, response_content: str) -> List[Dict[str, Any]]:
        """
        Parse extraction response
        
        Args:
            response_content: Model response content
            
        Returns:
            List of parsed skills
        """
        try:
            # Try to extract JSON part
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                return data.get('skills', [])
            else:
                # If no JSON found, try direct parsing
                return json.loads(response_content).get('skills', [])
                
        except json.JSONDecodeError:
            print("Cannot parse model response as JSON")
            return []
        except Exception as e:
            print(f"Failed to parse response: {e}")
            return []
    
    def _validate_skills(self, skills: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate and clean skill data
        
        Args:
            skills: Raw skill list
            
        Returns:
            Validated skill list
        """
        validated_skills = []
        
        for skill in skills:
            try:
                # Validate required fields
                required_fields = ['skill_name', 'task_type', 'description']
                if not all(field in skill for field in required_fields):
                    continue
                
                # Validate field types and formats
                validated_skill = {
                    'skill_name': str(skill.get('skill_name', '')).strip(),
                    'task_type': str(skill.get('task_type', '')).strip(),
                    'description': str(skill.get('description', '')).strip(),
                    'key_actions': self._validate_list_field(skill.get('key_actions', [])),
                    'environmental_cues': self._validate_list_field(skill.get('environmental_cues', [])),
                    'success_conditions': self._validate_list_field(skill.get('success_conditions', [])),
                    'generalizability_score': self._validate_score(skill.get('generalizability_score', 0.5))
                }
                
                # Validate skill name is not empty
                if not validated_skill['skill_name']:
                    continue
                
                validated_skills.append(validated_skill)
                
            except Exception as e:
                print(f"Failed to validate skill: {e}")
                continue
        
        return validated_skills
    
    def _validate_list_field(self, field_value: Any) -> List[str]:
        """
        Validate list field
        
        Args:
            field_value: Field value
            
        Returns:
            Validated string list
        """
        if isinstance(field_value, list):
            return [str(item).strip() for item in field_value if str(item).strip()]
        elif isinstance(field_value, str):
            return [field_value.strip()] if field_value.strip() else []
        else:
            return []
    
    def _validate_score(self, score: Any) -> float:
        """
        Validate generalizability score
        
        Args:
            score: Score value
            
        Returns:
            Validated score (0-1)
        """
        try:
            score_float = float(score)
            return max(0.0, min(1.0, score_float))
        except (ValueError, TypeError):
            return 0.5  # Default score
    
    def analyze_failure(
        self,
        task_description: str,
        execution_trace: List[Dict[str, Any]],
        failure_reason: str
    ) -> Dict[str, Any]:
        """
        Analyze failure reasons
        
        Args:
            task_description: Task description
            execution_trace: Execution trace
            failure_reason: Failure reason
            
        Returns:
            Failure analysis results
        """
        try:
            execution_steps = self._format_execution_steps(execution_trace)
            
            analysis_prompt = f"""Please analyze the following failed task execution trace and summarize failure reasons:

Task description: {task_description}
Execution steps: {execution_steps}
Failure reason: {failure_reason}

Please analyze:
1. Root cause of failure
2. Strategies for improvement
3. Error patterns to avoid
4. Suggestions for next attempt

Return analysis results in JSON format."""
            
            msg = Msg(
                name="system",
                content=analysis_prompt,
                role="system"
            )
            
            response = self.model(msg)
            
            if response and response.content:
                return self._parse_analysis_response(response.content)
            else:
                return {
                    "failure_reason": failure_reason,
                    "improvement_suggestions": ["Cannot generate analysis"],
                    "error_patterns": [],
                    "next_attempt_advice": "Continue trying"
                }
                
        except Exception as e:
            print(f"Failure analysis failed: {e}")
            return {
                "failure_reason": failure_reason,
                "improvement_suggestions": ["Analysis failed"],
                "error_patterns": [],
                "next_attempt_advice": "Retry"
            }
    
    def _parse_analysis_response(self, response_content: str) -> Dict[str, Any]:
        """
        Parse analysis response
        
        Args:
            response_content: Response content
            
        Returns:
            Parsed analysis results
        """
        try:
            # Try to extract JSON
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                # If no JSON, return text analysis
                return {
                    "failure_reason": "Unknown",
                    "improvement_suggestions": [response_content],
                    "error_patterns": [],
                    "next_attempt_advice": response_content
                }
        except Exception as e:
            print(f"Failed to parse analysis response: {e}")
            return {
                "failure_reason": "Parsing failed",
                "improvement_suggestions": [response_content],
                "error_patterns": [],
                "next_attempt_advice": "Refer to above suggestions"
            }
