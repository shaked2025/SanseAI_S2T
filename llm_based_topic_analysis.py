"""
LLM-Based Topic Analysis
Uses LLM to understand full conversation context and extract main topics/issues

Approach:
1. Send entire conversation to LLM for understanding
2. LLM extracts main topics/issues (like a human would)
3. LLM creates summaries for each topic
4. Works with offline LLMs (Ollama, llama.cpp) for security
"""

import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import re
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')


class LLMBasedTopicAnalysis:
    """
    LLM-powered topic analysis that understands full conversation context
    """
    
    def __init__(self, use_llm=True, llm_model='llama2'):
        print("Loading LLM-based topic analysis system...", flush=True)
        
        # Load embedding model for fallback/supplementary analysis
        self.embedding_model = SentenceTransformer('all-mpnet-base-v2')
        
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.llm_available = False
        
        if use_llm:
            print("  Attempting to connect to offline LLM...", flush=True)
            self.llm_available = self._setup_llm()
        
        if not self.llm_available:
            print("  Warning: LLM not available - will use enhanced rule-based analysis", flush=True)
        
        print("[OK] LLM-based topic analysis ready", flush=True)
    
    def _setup_llm(self):
        """Setup offline LLM connection"""
        # Try Ollama first
        try:
            import ollama
            # Test connection
            try:
                ollama.list()
                print("    Connected to Ollama", flush=True)
                self.llm_client = 'ollama'
                self.ollama = ollama
                return True
            except:
                print("    Ollama installed but not running - start with: ollama serve", flush=True)
                return False
        except ImportError:
            pass
        
        # Try llama-cpp-python
        try:
            from llama_cpp import Llama
            print("    llama-cpp-python found (requires model file)", flush=True)
            self.llm_client = 'llama_cpp'
            # Would need model path - user can configure
            return False  # Not fully set up without model
        except ImportError:
            pass
        
        return False
    
    def analyze_conversation(self, utterances):
        """
        Analyze conversation using LLM for topic extraction
        
        Process:
        1. Prepare full conversation text
        2. Send to LLM for topic extraction
        3. Parse LLM response
        4. Create structured output
        """
        if not utterances or len(utterances) < 2:
            return {'topics': [], 'questions': [], 'utterances': utterances}
        
        print(f"\n=== LLM-BASED TOPIC ANALYSIS ===", flush=True)
        print(f"Analyzing {len(utterances)} utterances...", flush=True)
        
        # === STEP 1: Prepare Full Conversation ===
        print("\nStep 1: Preparing full conversation context...", flush=True)
        conversation_text = self._prepare_conversation_text(utterances)
        print(f"  Conversation length: {len(conversation_text)} characters", flush=True)
        
        # === STEP 2: Extract Questions ===
        print("\nStep 2: Extracting questions...", flush=True)
        questions = self._extract_questions(utterances)
        print(f"  Found {len(questions)} questions", flush=True)
        
        # === STEP 3: LLM Topic Extraction ===
        if self.llm_available:
            print("\nStep 3: Using LLM to extract main topics/issues...", flush=True)
            llm_topics = self._llm_extract_topics(conversation_text, questions)
            print(f"  LLM identified {len(llm_topics)} main topics", flush=True)
        else:
            print("\nStep 3: Using enhanced analysis (LLM not available)...", flush=True)
            llm_topics = self._enhanced_rule_based_extraction(conversation_text, questions, utterances)
            print(f"  Identified {len(llm_topics)} main topics", flush=True)
        
        # === STEP 4: Create Topic Summaries ===
        print("\nStep 4: Creating comprehensive topic summaries...", flush=True)
        topics = self._create_topic_summaries(llm_topics, utterances, questions)
        print(f"  Created {len(topics)} topic summaries", flush=True)
        
        # === STEP 5: Timeline Analysis ===
        print("\nStep 5: Analyzing temporal patterns...", flush=True)
        for topic in topics:
            self._analyze_topic_timeline(topic, utterances)
        
        print(f"\n[OK] Analysis complete: {len(topics)} topics, {len(questions)} questions", flush=True)
        
        return {
            'topics': topics,
            'questions': questions,
            'utterances': utterances
        }
    
    def _prepare_conversation_text(self, utterances):
        """Prepare full conversation text with timestamps"""
        lines = []
        for i, utt in enumerate(utterances):
            timestamp = utt.get('timestamp_str', f"[{i*10}s]")
            text = utt.get('text', '').strip()
            speaker = utt.get('speaker_role', 'Speaker')
            if text:
                lines.append(f"{timestamp} {speaker}: {text}")
        
        return '\n'.join(lines)
    
    def _extract_questions(self, utterances):
        """Extract all questions"""
        questions = []
        question_patterns = [
            re.compile(r'\b(what|where|when|who|why|how|which|whose|whom)\b.*\?', re.IGNORECASE),
            re.compile(r'\b(do|does|did|is|are|was|were|have|has|had|can|could|would|should|will|shall|may|might|must)\s+(you|he|she|they|we|it|i|there)\b', re.IGNORECASE),
            re.compile(r'\b(tell|explain|describe|clarify|elaborate|say|talk|speak)\s+(me|us|about)\b', re.IGNORECASE),
        ]
        
        for i, utt in enumerate(utterances):
            text = utt.get('text', '').strip()
            if not text:
                continue
            
            is_question = False
            if '?' in text:
                is_question = True
            else:
                for pattern in question_patterns:
                    if pattern.search(text):
                        is_question = True
                        break
            
            if is_question:
                questions.append({
                    'index': i,
                    'text': text,
                    'timestamp': utt.get('timestamp_str', ''),
                    'speaker': utt.get('speaker_role', 'Unknown')
                })
        
        return questions
    
    def _llm_extract_topics(self, conversation_text, questions):
        """
        Use LLM to extract main topics/issues from conversation
        
        This is the key improvement - LLM understands full context
        """
        # Create comprehensive prompt
        questions_text = "\n".join([f"Q{i+1}: {q['text']}" for i, q in enumerate(questions[:20])])
        
        # Limit conversation text to avoid token limits (keep last part which is often most relevant)
        if len(conversation_text) > 6000:
            conversation_text = conversation_text[-6000:]  # Keep last 6000 chars
        
        prompt = f"""You are analyzing an interview/conversation transcript. Your task is to identify the MAIN TOPICS and ISSUES that were discussed, just like a human would summarize "what was this conversation about?"

CONVERSATION TRANSCRIPT:
{conversation_text}

QUESTIONS ASKED IN THE CONVERSATION:
{questions_text}

INSTRUCTIONS:
1. Read the ENTIRE conversation to understand the full context
2. Identify the MAIN TOPICS/ISSUES (not keywords, but actual subjects/matters discussed)
3. Group related questions together under each topic
4. For each topic, write a clear summary of what was discussed

IMPORTANT:
- Focus on MAIN topics/issues, not minor details
- Topic names should be clear and descriptive (e.g., "Intelligence Organizations" not "Intelligence & Organization")
- Each topic should represent a significant part of the conversation
- Group questions that are about the same subject together

Format your response as valid JSON only:
{{
  "topics": [
    {{
      "topic_name": "Clear descriptive topic name",
      "question_indices": [1, 3, 5],
      "summary": "2-3 sentence summary of what was discussed about this topic, including key points from both questions and answers"
    }}
  ]
}}

Response (JSON only):"""
        
        try:
            if self.llm_client == 'ollama':
                # Try to use available models
                available_models = []
                try:
                    models = self.ollama.list()
                    available_models = [m['name'] for m in models.get('models', [])]
                except:
                    pass
                
                # Use best available model
                model_to_use = self.llm_model
                if model_to_use not in available_models and available_models:
                    model_to_use = available_models[0]
                    print(f"    Using available model: {model_to_use}", flush=True)
                
                response = self.ollama.generate(
                    model=model_to_use,
                    prompt=prompt,
                    options={
                        'temperature': 0.2,  # Lower temperature for more consistent, factual results
                        'top_p': 0.9,
                        'num_predict': 2000  # Allow longer responses
                    }
                )
                llm_output = response['response']
                
                # Parse JSON response
                topics = self._parse_llm_response(llm_output, questions)
                return topics
        except Exception as e:
            print(f"    LLM extraction failed: {e}, using fallback", flush=True)
            return self._enhanced_rule_based_extraction(conversation_text, questions, [])
    
    def _parse_llm_response(self, llm_output, questions):
        """Parse LLM JSON response"""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                topics = []
                
                for topic_data in data.get('topics', []):
                    topic = {
                        'topic_name': topic_data.get('topic_name', 'Unknown Topic'),
                        'question_indices': topic_data.get('question_indices', []),
                        'summary': topic_data.get('summary', ''),
                        'questions': [questions[i-1] for i in topic_data.get('question_indices', []) if 0 <= i-1 < len(questions)]
                    }
                    topics.append(topic)
                
                return topics
        except Exception as e:
            print(f"    Error parsing LLM response: {e}", flush=True)
        
        return []
    
    def _enhanced_rule_based_extraction(self, conversation_text, questions, utterances):
        """
        Enhanced rule-based extraction when LLM not available
        
        Strategy:
        1. Understand full conversation context (document-level embedding)
        2. Cluster questions by semantic similarity
        3. Extract meaningful topic names
        4. Create comprehensive summaries
        """
        if not questions:
            return []
        
        print("    Using enhanced semantic analysis (understanding full context)...", flush=True)
        
        # Step 1: Create document-level understanding
        conversation_embedding = self.embedding_model.encode([conversation_text[:4000]], show_progress_bar=False)[0]
        
        # Step 2: Embed all questions with context
        question_texts = [q['text'] for q in questions]
        question_embeddings = self.embedding_model.encode(question_texts, show_progress_bar=False)
        
        # Step 3: Cluster questions by semantic similarity
        from sklearn.cluster import AgglomerativeClustering
        
        # Determine optimal number of topics (5-10 main topics)
        n_topics = max(3, min(len(questions) // 2, 10))
        
        clustering = AgglomerativeClustering(
            n_clusters=n_topics,
            linkage='average',
            metric='cosine'
        )
        topic_ids = clustering.fit_predict(question_embeddings)
        
        # Step 4: Group questions by topic and analyze
        topic_groups = defaultdict(list)
        for i, topic_id in enumerate(topic_ids):
            topic_groups[topic_id].append(i)
        
        # Step 5: Create topics with better names and summaries
        topics = []
        for topic_id, question_indices in topic_groups.items():
            topic_questions = [questions[i] for i in question_indices]
            
            # Get answers for these questions
            answer_texts = []
            for q in topic_questions:
                q_idx = q['index']
                # Look for answers
                for i in range(q_idx + 1, min(q_idx + 4, len(utterances))):
                    answer_text = utterances[i].get('text', '').strip()
                    if answer_text and len(answer_text) > 10:
                        answer_texts.append(answer_text)
                        break
            
            # Extract topic name (better method)
            topic_name = self._extract_meaningful_topic_name(topic_questions, answer_texts)
            
            # Create comprehensive summary
            summary = self._create_comprehensive_topic_summary(topic_questions, answer_texts)
            
            topics.append({
                'topic_name': topic_name,
                'question_indices': [q['index'] for q in topic_questions],
                'questions': topic_questions,
                'summary': summary
            })
        
        # Sort by number of questions (most important first)
        topics.sort(key=lambda t: len(t['questions']), reverse=True)
        
        return topics
    
    def _extract_meaningful_topic_name(self, questions, answers):
        """Extract meaningful topic name from Q&A"""
        # Combine all text
        all_text = ' '.join([q['text'] for q in questions] + answers)
        
        # Use embedding to understand the topic
        topic_embedding = self.embedding_model.encode([all_text[:500]], show_progress_bar=False)[0]
        
        # Extract key phrases (noun phrases)
        import nltk
        from nltk.tag import pos_tag
        from nltk.tokenize import word_tokenize
        
        try:
            tokens = word_tokenize(all_text.lower())
            tagged = pos_tag(tokens)
        except:
            words = re.findall(r'\b[a-z]{4,}\b', all_text.lower())
            tagged = [(w, 'NN') for w in words]
        
        # Extract important nouns and noun phrases
        important_terms = []
        for i, (word, pos) in enumerate(tagged):
            if pos.startswith('NN') and len(word) > 4:
                # Filter common words
                if word not in ['thing', 'time', 'way', 'people', 'person', 'place', 'part', 
                              'question', 'answer', 'topic', 'what', 'where', 'when', 'who', 'why', 'how',
                              'think', 'know', 'want', 'would', 'could', 'should', 'about', 'tell', 'say']:
                    important_terms.append(word)
        
        # Count frequency
        from collections import Counter
        term_counts = Counter(important_terms)
        top_terms = [term for term, count in term_counts.most_common(3)]
        
        if top_terms:
            # Create descriptive name
            if len(top_terms) >= 2:
                topic_name = f"{top_terms[0].capitalize()} {top_terms[1].capitalize()}"
            else:
                topic_name = top_terms[0].capitalize()
            return topic_name
        else:
            # Fallback: use first meaningful words from first question
            first_q = questions[0]['text']
            words = re.findall(r'\b[a-z]{4,}\b', first_q.lower())
            meaningful = [w for w in words if w not in ['what', 'where', 'when', 'who', 'why', 'how', 'tell', 'explain']]
            if meaningful:
                return meaningful[0].capitalize()
            return "General Discussion"
    
    def _create_comprehensive_topic_summary(self, questions, answers):
        """Create comprehensive summary like LLM would"""
        summary_parts = []
        
        # Topic description
        if len(questions) == 1:
            summary_parts.append(f"TOPIC: Discussion about a specific question regarding the subject matter.")
        else:
            summary_parts.append(f"TOPIC: This section covers {len(questions)} related questions about the same subject.")
        
        # Questions
        summary_parts.append(f"\nQUESTIONS ASKED ({len(questions)}):")
        for i, q in enumerate(questions[:5], 1):  # Show up to 5
            q_text = q['text'][:120] + ('...' if len(q['text']) > 120 else '')
            summary_parts.append(f"  {i}. \"{q_text}\"")
        if len(questions) > 5:
            summary_parts.append(f"  ... and {len(questions) - 5} more question(s)")
        
        # Key information from answers
        if answers:
            combined_answers = ' '.join([a for a in answers if a and len(a) > 20])
            if combined_answers:
                # Extract key sentences
                sentences = re.split(r'[.!?]+\s+', combined_answers)
                key_sentences = sorted([s.strip() for s in sentences if len(s.strip()) > 25], 
                                      key=len, reverse=True)[:4]
                
                if key_sentences:
                    summary_parts.append(f"\nKEY INFORMATION FROM ANSWERS:")
                    for sent in key_sentences:
                        if len(sent) > 180:
                            sent = sent[:180] + "..."
                        summary_parts.append(f"  • {sent}")
        
        # Overall summary
        summary_parts.append(f"\nSUMMARY: This topic involved discussion about the subject matter, with {len(questions)} question(s) asked and responses provided.")
        
        return '\n'.join(summary_parts)
    
    def _extract_topic_name_from_questions(self, questions):
        """Extract meaningful topic name from questions"""
        # Combine all question texts
        all_text = ' '.join([q['text'] for q in questions])
        
        # Use embedding to find most representative question
        if len(questions) > 1:
            texts = [q['text'] for q in questions]
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
            centroid = np.mean(embeddings, axis=0)
            
            # Find question closest to centroid
            similarities = [np.dot(emb, centroid) / (np.linalg.norm(emb) * np.linalg.norm(centroid) + 1e-10) 
                           for emb in embeddings]
            best_idx = np.argmax(similarities)
            representative_text = questions[best_idx]['text']
        else:
            representative_text = questions[0]['text']
        
        # Extract key terms (simple approach)
        # Remove question words and common words
        words = re.findall(r'\b[a-z]{4,}\b', representative_text.lower())
        stopwords_set = {'what', 'where', 'when', 'who', 'why', 'how', 'which', 'tell', 'explain', 
                        'describe', 'think', 'know', 'want', 'would', 'could', 'should', 'about'}
        meaningful_words = [w for w in words if w not in stopwords_set]
        
        if meaningful_words:
            # Take top 2-3 meaningful words
            topic_name = ' '.join([w.capitalize() for w in meaningful_words[:2]])
            return topic_name
        else:
            return "General Discussion"
    
    def _create_topic_summary_from_questions(self, questions, utterances):
        """Create summary from questions (legacy method - use _create_comprehensive_topic_summary instead)"""
        # Get answers
        answer_texts = []
        for q in questions:
            q_idx = q['index']
            for i in range(q_idx + 1, min(q_idx + 4, len(utterances))):
                answer_text = utterances[i].get('text', '').strip()
                if answer_text and len(answer_text) > 10:
                    answer_texts.append(answer_text)
                    break
        
        return self._create_comprehensive_topic_summary(questions, answer_texts)
    
    def _create_topic_summaries(self, llm_topics, utterances, questions):
        """Create comprehensive topic summaries with natural language descriptions"""
        topics = []
        
        for topic_idx, topic_data in enumerate(llm_topics):
            # Get all utterance indices for this topic
            utterance_indices = []
            
            # Add question indices
            for q in topic_data.get('questions', []):
                utterance_indices.append(q['index'])
            
            # Find answers and related discussion for each question
            for q in topic_data.get('questions', []):
                q_idx = q['index']
                # Look for answer and follow-up discussion (next 3-5 utterances)
                for i in range(q_idx + 1, min(q_idx + 6, len(utterances))):
                    if i not in utterance_indices:
                        utterance_indices.append(i)
            
            # Also find semantically related utterances (context around questions)
            topic_question_indices = [q['index'] for q in topic_data.get('questions', [])]
            if topic_question_indices:
                # Get context window around questions
                min_idx = max(0, min(topic_question_indices) - 2)
                max_idx = min(len(utterances), max(topic_question_indices) + 5)
                for i in range(min_idx, max_idx):
                    if i not in utterance_indices:
                        utterance_indices.append(i)
            
            # Get all utterances for this topic
            topic_utterances = [utterances[i] for i in sorted(set(utterance_indices)) if i < len(utterances)]
            
            # Create natural language summary using LLM
            if self.llm_available:
                print(f"    Creating LLM summary for topic {topic_idx + 1}...", flush=True)
                summary = self._llm_create_natural_summary(topic_data, topic_utterances)
            else:
                print(f"    Creating enhanced summary for topic {topic_idx + 1}...", flush=True)
                summary = self._create_natural_summary_fallback(topic_data, topic_utterances)
            
            topic = {
                'topic_id': topic_idx,
                'label': topic_data.get('topic_name', 'Unknown Topic'),
                'questions': [q['text'] for q in topic_data.get('questions', [])],
                'utterance_indices': sorted(set(utterance_indices)),
                'mention_count': len(set(utterance_indices)),
                'summary': summary,
                'all_utterances': topic_utterances  # Store all utterances for this topic
            }
            
            topics.append(topic)
        
        return topics
    
    def _llm_create_natural_summary(self, topic_data, topic_utterances):
        """
        Use LLM to create natural language summary of what happened in this topic
        
        This is the key: understand what was discussed and summarize naturally
        """
        # Prepare full conversation text for this topic
        topic_text = []
        for utt in topic_utterances:
            timestamp = utt.get('timestamp_str', '')
            speaker = utt.get('speaker_role', 'Speaker')
            text = utt.get('text', '').strip()
            if text:
                topic_text.append(f"{timestamp} {speaker}: {text}")
        
        full_topic_conversation = '\n'.join(topic_text)
        
        # Create prompt for natural language summary
        prompt = f"""You are analyzing a conversation transcript. Create a natural, human-like summary of what happened when this topic was discussed.

TOPIC NAME: {topic_data.get('topic_name', 'Unknown Topic')}

FULL TRANSCRIPT OF THIS TOPIC DISCUSSION:
{full_topic_conversation[:5000]}

Your task: Write a natural summary (3-5 sentences) as if you're explaining to someone what happened in this part of the conversation. Include:
- What the topic was about
- What questions were asked
- What was discussed and answered
- Key points or conclusions reached

Write naturally, like you're telling a story about what happened. Be specific and clear.

Summary:"""
        
        try:
            if self.llm_client == 'ollama':
                available_models = []
                try:
                    models = self.ollama.list()
                    available_models = [m['name'] for m in models.get('models', [])]
                except:
                    pass
                
                model_to_use = self.llm_model
                if model_to_use not in available_models and available_models:
                    model_to_use = available_models[0]
                
                response = self.ollama.generate(
                    model=model_to_use,
                    prompt=prompt,
                    options={
                        'temperature': 0.3,
                        'top_p': 0.9,
                        'num_predict': 500
                    }
                )
                return response['response'].strip()
        except Exception as e:
            print(f"      LLM summary failed: {e}, using fallback", flush=True)
            return self._create_natural_summary_fallback(topic_data, topic_utterances)
        
        return self._create_natural_summary_fallback(topic_data, topic_utterances)
    
    def _create_natural_summary_fallback(self, topic_data, topic_utterances):
        """
        Create natural language summary without LLM
        
        Tries to create a human-like summary from the conversation
        """
        topic_name = topic_data.get('topic_name', 'this topic')
        questions = topic_data.get('questions', [])
        
        # Extract key information from utterances
        all_text = ' '.join([utt.get('text', '') for utt in topic_utterances if utt.get('text', '').strip()])
        
        # Find key sentences (longer, informative)
        sentences = re.split(r'[.!?]+\s+', all_text)
        key_sentences = sorted([s.strip() for s in sentences if len(s.strip()) > 40], 
                              key=len, reverse=True)[:4]
        
        # Build natural summary
        summary_parts = []
        
        # Opening - what was discussed
        if len(questions) > 1:
            summary_parts.append(f"The conversation addressed {topic_name.lower()}, with {len(questions)} questions being asked about this subject.")
        elif len(questions) == 1:
            summary_parts.append(f"The discussion focused on {topic_name.lower()}, with one main question being asked.")
        else:
            summary_parts.append(f"The conversation covered {topic_name.lower()}.")
        
        # What was asked (if we have questions)
        if questions:
            if len(questions) == 1:
                q_text = questions[0]['text'][:120] + ('...' if len(questions[0]['text']) > 120 else '')
                summary_parts.append(f"The interviewer asked: \"{q_text}\"")
            elif len(questions) <= 3:
                q_list = ', '.join([f'"{q["text"][:60]}..."' if len(q['text']) > 60 else f'"{q["text"]}"' 
                                   for q in questions[:2]])
                summary_parts.append(f"Questions included: {q_list}.")
            else:
                main_q = questions[0]['text'][:80] + ('...' if len(questions[0]['text']) > 80 else '')
                summary_parts.append(f"Multiple questions were asked, including: \"{main_q}\" and {len(questions) - 1} related questions.")
        
        # What was discussed/answered - use key sentences
        if key_sentences:
            # Take the most informative sentences
            main_discussion = key_sentences[0][:200] + ('...' if len(key_sentences[0]) > 200 else '')
            summary_parts.append(f"In response, the discussion covered: {main_discussion}")
            
            # Add additional key point if available
            if len(key_sentences) > 1 and len(key_sentences[1]) > 30:
                second_point = key_sentences[1][:150] + ('...' if len(key_sentences[1]) > 150 else '')
                summary_parts.append(f"Additionally, {second_point}")
        
        # Conclusion/context
        if len(topic_utterances) > 5:
            summary_parts.append(f"This topic was explored through {len(topic_utterances)} exchanges between the participants.")
        
        return ' '.join(summary_parts)
    
    def _create_comprehensive_summary(self, questions, answers):
        """Create comprehensive summary from Q&A"""
        summary_parts = []
        
        # Topic description
        if questions:
            summary_parts.append(f"TOPIC: {len(questions)} question(s) were asked about this subject.")
        
        # Key questions
        if len(questions) <= 3:
            summary_parts.append("\nQuestions:")
            for i, q in enumerate(questions, 1):
                summary_parts.append(f"  {i}. \"{q['text'][:100]}{'...' if len(q['text']) > 100 else ''}\"")
        else:
            summary_parts.append(f"\nMain questions ({len(questions)} total):")
            for i, q in enumerate(questions[:3], 1):
                summary_parts.append(f"  {i}. \"{q['text'][:100]}{'...' if len(q['text']) > 100 else ''}\"")
            summary_parts.append(f"  ... and {len(questions) - 3} more")
        
        # Key information from answers
        if answers:
            combined_answers = ' '.join([a for a in answers if a and len(a) > 20])
            if combined_answers:
                # Extract key sentences
                sentences = re.split(r'[.!?]+\s+', combined_answers)
                key_sentences = sorted([s.strip() for s in sentences if len(s.strip()) > 30], 
                                      key=len, reverse=True)[:3]
                
                if key_sentences:
                    summary_parts.append("\nKey information:")
                    for sent in key_sentences:
                        if len(sent) > 200:
                            sent = sent[:200] + "..."
                        summary_parts.append(f"  • {sent}")
        
        return '\n'.join(summary_parts)
    
    def _analyze_topic_timeline(self, topic, utterances):
        """Analyze temporal patterns"""
        if not topic['utterance_indices']:
            return
        
        topic_utterances = [utterances[i] for i in topic['utterance_indices'] if i < len(utterances)]
        
        timestamps = []
        for utt in topic_utterances:
            if 'timestamp' in utt:
                timestamps.append(utt['timestamp'])
            elif 'timestamp_str' in utt:
                try:
                    ts = datetime.strptime(utt['timestamp_str'], '%H:%M:%S')
                    timestamps.append(ts)
                except:
                    pass
        
        if timestamps:
            topic['first_mention'] = min(timestamps).isoformat()
            topic['last_mention'] = max(timestamps).isoformat()
            time_span = (max(timestamps) - min(timestamps)).total_seconds() / 60
            topic['total_span_minutes'] = time_span
            
            if len(timestamps) > 1:
                sorted_timestamps = sorted(timestamps)
                gaps = []
                for i in range(1, len(sorted_timestamps)):
                    gap = (sorted_timestamps[i] - sorted_timestamps[i-1]).total_seconds() / 60
                    if gap > 2.0:
                        gaps.append(gap)
                
                if gaps:
                    topic['is_revisited'] = True
                    topic['period_count'] = len(gaps) + 1
                    topic['revisit_gaps_minutes'] = gaps
                else:
                    topic['is_revisited'] = False
                    topic['period_count'] = 1

