"""
Question-Answer Based Topic Modeling
Precise topic extraction based on Q-A structure

Key improvements:
1. Extract ALL questions accurately
2. Group questions by topic (questions about same topic)
3. For each topic: include both questions AND answers
4. Create comprehensive summaries per topic
5. Use offline LLM for final summarization (optional, for security)

Based on research:
- Q-A structure is the natural unit of conversation topics
- Topics should correlate with question clusters
- Each topic should summarize both questions asked and answers given
"""

import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
import re
from collections import Counter, defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class QuestionAnswerTopicModeling:
    """
    Topic modeling based on Question-Answer structure
    More precise: topics = clusters of related Q-A pairs
    """
    
    def __init__(self, use_offline_llm=False):
        print("Loading Q-A based topic modeling system...", flush=True)
        
        # Use MPNet for better semantic understanding
        print("  Loading MPNet model for semantic understanding...", flush=True)
        self.doc_model = SentenceTransformer('all-mpnet-base-v2')  # 768-dim
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim, faster
        
        # Stopwords
        try:
            self.stopwords = set(stopwords.words('english'))
        except:
            self.stopwords = set()
        
        # Question detection patterns (comprehensive)
        self.question_patterns = [
            # WH-questions
            re.compile(r'\b(what|where|when|who|why|how|which|whose|whom)\b.*\?', re.IGNORECASE),
            # Yes/No questions
            re.compile(r'\b(do|does|did|is|are|was|were|have|has|had|can|could|would|should|will|shall|may|might|must)\s+(you|he|she|they|we|it|i|there)\b', re.IGNORECASE),
            # Imperative questions
            re.compile(r'\b(tell|explain|describe|clarify|elaborate|say|talk|speak)\s+(me|us|about)\b', re.IGNORECASE),
            # Tag questions
            re.compile(r'.*\?$', re.IGNORECASE),
            # Questions without question mark (common in speech)
            re.compile(r'^(what|where|when|who|why|how|which|whose|whom|do|does|did|is|are|was|were|have|has|had|can|could|would|should|will|shall|may|might|must|tell|explain|describe|clarify)', re.IGNORECASE),
        ]
        
        self.use_offline_llm = use_offline_llm
        if use_offline_llm:
            print("  Attempting to load offline LLM for summarization...", flush=True)
            # Try to load offline LLM (Ollama, llama.cpp, etc.)
            self.offline_llm = self._load_offline_llm()
        else:
            self.offline_llm = None
        
        print("[OK] Q-A based topic modeling ready", flush=True)
    
    def _load_offline_llm(self):
        """Try to load an offline LLM for summarization"""
        # Try Ollama first (most common)
        try:
            import ollama
            print("    Found Ollama - will use for topic summarization", flush=True)
            return 'ollama'
        except ImportError:
            pass
        
        # Try llama-cpp-python
        try:
            from llama_cpp import Llama
            print("    Found llama-cpp - will use for topic summarization", flush=True)
            return 'llama_cpp'
        except ImportError:
            pass
        
        print("    No offline LLM found - will use rule-based summarization", flush=True)
        return None
    
    def analyze_conversation(self, utterances):
        """
        Main analysis: Extract topics based on Q-A structure
        
        Process:
        1. Extract ALL questions accurately
        2. Find answers for each question
        3. Cluster Q-A pairs by topic
        4. Create topic summaries (Q + A)
        """
        if not utterances or len(utterances) < 2:
            return {'topics': [], 'questions': [], 'utterances': utterances}
        
        print(f"\n=== Q-A BASED TOPIC ANALYSIS ===", flush=True)
        print(f"Analyzing {len(utterances)} utterances...", flush=True)
        
        # === STEP 1: Extract ALL Questions ===
        print("\nStep 1: Extracting all questions...", flush=True)
        questions = self._extract_all_questions(utterances)
        print(f"  Found {len(questions)} questions", flush=True)
        
        # === STEP 2: Find Answers for Each Question ===
        print("\nStep 2: Finding answers for each question...", flush=True)
        qa_pairs = self._pair_questions_with_answers(questions, utterances)
        print(f"  Created {len(qa_pairs)} Q-A pairs", flush=True)
        
        # === STEP 3: Cluster Q-A Pairs by Topic ===
        print("\nStep 3: Clustering Q-A pairs into topics...", flush=True)
        topic_clusters = self._cluster_qa_pairs(qa_pairs, utterances)
        print(f"  Identified {len(topic_clusters)} topics", flush=True)
        
        # === STEP 4: Create Topic Summaries ===
        print("\nStep 4: Creating topic summaries (Q + A)...", flush=True)
        topics = self._create_topic_summaries(topic_clusters, utterances)
        print(f"  Created {len(topics)} topic summaries", flush=True)
        
        # === STEP 5: Timeline Analysis ===
        print("\nStep 5: Analyzing temporal patterns...", flush=True)
        for topic in topics:
            self._analyze_topic_timeline(topic, utterances)
        
        print(f"\n[OK] Analysis complete: {len(topics)} topics, {len(questions)} questions", flush=True)
        
        return {
            'topics': topics,
            'questions': questions,
            'qa_pairs': qa_pairs,
            'utterances': utterances
        }
    
    def _extract_all_questions(self, utterances):
        """
        Extract ALL questions from utterances
        
        Uses multiple detection methods:
        1. Question mark
        2. WH-words
        3. Auxiliary verb patterns
        4. Imperative question patterns
        """
        questions = []
        
        for i, utt in enumerate(utterances):
            text = utt.get('text', '').strip()
            if not text:
                continue
            
            is_question = False
            
            # Check question mark
            if '?' in text:
                is_question = True
            
            # Check patterns
            if not is_question:
                for pattern in self.question_patterns:
                    if pattern.search(text):
                        is_question = True
                        break
            
            if is_question:
                questions.append({
                    'index': i,
                    'text': text,
                    'timestamp': utt.get('timestamp_str', ''),
                    'speaker': utt.get('speaker_role', 'Unknown'),
                    'timestamp_obj': utt.get('timestamp')
                })
        
        return questions
    
    def _pair_questions_with_answers(self, questions, utterances):
        """
        Pair each question with its answer(s)
        
        Strategy:
        - Answer is usually the next utterance from a different speaker
        - Or within next 2-3 utterances
        - Answer should be semantically related to question
        """
        qa_pairs = []
        
        for q in questions:
            q_idx = q['index']
            q_text = q['text']
            q_speaker = q.get('speaker', '')
            
            # Find answer (next utterance from different speaker, or semantically similar)
            answer_indices = []
            answer_texts = []
            
            # Look ahead up to 3 utterances
            for i in range(q_idx + 1, min(q_idx + 4, len(utterances))):
                answer_utt = utterances[i]
                answer_text = answer_utt.get('text', '').strip()
                answer_speaker = answer_utt.get('speaker_role', '')
                
                if not answer_text:
                    continue
                
                # If different speaker, likely the answer
                if answer_speaker != q_speaker:
                    answer_indices.append(i)
                    answer_texts.append(answer_text)
                    break  # Take first answer from different speaker
                else:
                    # Same speaker - check if it's semantically related (might be clarification)
                    q_emb = self.sentence_model.encode([q_text], show_progress_bar=False)[0]
                    a_emb = self.sentence_model.encode([answer_text], show_progress_bar=False)[0]
                    similarity = np.dot(q_emb, a_emb) / (
                        np.linalg.norm(q_emb) * np.linalg.norm(a_emb) + 1e-10
                    )
                    
                    if similarity > 0.60:  # Semantically related
                        answer_indices.append(i)
                        answer_texts.append(answer_text)
                        break
            
            # If no answer found, look further (up to 5 utterances)
            if not answer_indices:
                for i in range(q_idx + 1, min(q_idx + 6, len(utterances))):
                    answer_utt = utterances[i]
                    answer_text = answer_utt.get('text', '').strip()
                    
                    if not answer_text or len(answer_text) < 10:
                        continue
                    
                    # Check semantic similarity
                    q_emb = self.sentence_model.encode([q_text], show_progress_bar=False)[0]
                    a_emb = self.sentence_model.encode([answer_text], show_progress_bar=False)[0]
                    similarity = np.dot(q_emb, a_emb) / (
                        np.linalg.norm(q_emb) * np.linalg.norm(a_emb) + 1e-10
                    )
                    
                    if similarity > 0.50:  # Lower threshold for further away
                        answer_indices.append(i)
                        answer_texts.append(answer_text)
                        break
            
            qa_pairs.append({
                'question': q,
                'answer_indices': answer_indices,
                'answer_texts': answer_texts,
                'qa_text': f"{q_text} {' '.join(answer_texts)}"  # Combined for clustering
            })
        
        return qa_pairs
    
    def _cluster_qa_pairs(self, qa_pairs, utterances):
        """
        Cluster Q-A pairs into topics
        
        Strategy:
        1. Embed each Q-A pair
        2. Cluster by semantic similarity
        3. Each cluster = one topic
        """
        if len(qa_pairs) < 2:
            return [qa_pairs]
        
        # Embed all Q-A pairs
        qa_texts = [pair['qa_text'] for pair in qa_pairs]
        qa_embeddings = self.doc_model.encode(qa_texts, show_progress_bar=False)
        
        # Determine number of topics (1 topic per 2-3 Q-A pairs, but at least 3 topics)
        n_topics = max(3, min(len(qa_pairs) // 2, 15))
        
        # Cluster
        clustering = AgglomerativeClustering(
            n_clusters=n_topics,
            linkage='average',
            metric='cosine'
        )
        topic_ids = clustering.fit_predict(qa_embeddings)
        
        # Group Q-A pairs by topic
        topic_clusters = defaultdict(list)
        for i, topic_id in enumerate(topic_ids):
            topic_clusters[topic_id].append(qa_pairs[i])
        
        # Convert to list and sort by first question index
        clusters_list = []
        for topic_id, pairs in topic_clusters.items():
            first_idx = min(p['question']['index'] for p in pairs)
            clusters_list.append({
                'topic_id': topic_id,
                'qa_pairs': pairs,
                'first_question_idx': first_idx
            })
        
        clusters_list.sort(key=lambda x: x['first_question_idx'])
        
        return clusters_list
    
    def _create_topic_summaries(self, topic_clusters, utterances):
        """
        Create comprehensive topic summaries
        
        For each topic:
        1. Extract topic name from Q-A pairs
        2. List all questions in this topic
        3. List all answers
        4. Create summary (using LLM if available, else rule-based)
        """
        topics = []
        
        for cluster in topic_clusters:
            qa_pairs = cluster['qa_pairs']
            
            # Get all question and answer texts
            all_questions = [pair['question']['text'] for pair in qa_pairs]
            all_answers = []
            all_utterance_indices = []
            
            for pair in qa_pairs:
                all_utterance_indices.append(pair['question']['index'])
                for idx in pair['answer_indices']:
                    all_utterance_indices.append(idx)
                    if idx < len(utterances):
                        all_answers.append(utterances[idx].get('text', ''))
            
            # Extract topic name
            topic_name = self._extract_topic_name_from_qa(qa_pairs)
            
            # Create summary
            if self.offline_llm:
                summary = self._llm_summarize_topic(all_questions, all_answers)
            else:
                summary = self._rule_based_summarize_topic(all_questions, all_answers)
            
            topic = {
                'topic_id': cluster['topic_id'],
                'label': topic_name,
                'questions': all_questions,
                'answers': all_answers,
                'qa_pairs': qa_pairs,
                'utterance_indices': sorted(set(all_utterance_indices)),
                'mention_count': len(all_utterance_indices),
                'summary': summary
            }
            
            topics.append(topic)
        
        return topics
    
    def _extract_topic_name_from_qa(self, qa_pairs):
        """
        Extract meaningful topic name from Q-A pairs
        
        Strategy:
        1. Extract key terms from questions (they define the topic)
        2. Use most common meaningful terms
        3. Create concise label
        """
        # Combine all question texts
        all_question_text = ' '.join([pair['question']['text'] for pair in qa_pairs])
        
        # Extract key terms
        try:
            tokens = word_tokenize(all_question_text.lower())
            tagged = pos_tag(tokens)
        except:
            words = re.findall(r'\b[a-z]{3,}\b', all_question_text.lower())
            tagged = [(w, 'NN') for w in words]
        
        # Extract important nouns and noun phrases
        important_terms = []
        for word, pos in tagged:
            if pos.startswith('NN') and word not in self.stopwords and len(word) > 4:
                # Filter common words
                if word not in ['thing', 'time', 'way', 'people', 'person', 'place', 'part', 
                              'topic', 'question', 'answer', 'what', 'where', 'when', 'who', 'why', 'how']:
                    important_terms.append(word)
        
        # Count and get top terms
        term_counts = Counter(important_terms)
        top_terms = [term for term, count in term_counts.most_common(3)]
        
        if top_terms:
            # Capitalize and join
            label = ' & '.join([t.capitalize() for t in top_terms[:2]])
            return label
        else:
            # Fallback: use first few words of first question
            first_q = qa_pairs[0]['question']['text']
            words = first_q.split()[:5]
            return ' '.join([w.capitalize() for w in words if len(w) > 2])
    
    def _rule_based_summarize_topic(self, questions, answers):
        """
        Create comprehensive summary using rule-based approach
        
        Creates a GPT-like summary that captures:
        1. Main topic/issue
        2. Key questions asked
        3. Main points from answers
        4. Conclusions/important information
        """
        summary_parts = []
        
        # Extract main topic from questions
        if questions:
            # Combine all questions to identify main theme
            all_question_text = ' '.join(questions)
            # Extract key nouns/entities
            try:
                tokens = word_tokenize(all_question_text.lower())
                tagged = pos_tag(tokens)
            except:
                tokens = all_question_text.lower().split()
                tagged = [(t, 'NN') for t in tokens]
            
            # Get important nouns (topic indicators)
            important_terms = []
            for word, pos in tagged:
                if pos.startswith('NN') and word not in self.stopwords and len(word) > 4:
                    if word not in ['thing', 'time', 'way', 'people', 'person', 'place', 'part', 
                                  'question', 'answer', 'topic', 'what', 'where', 'when', 'who', 'why', 'how']:
                        important_terms.append(word)
            
            topic_terms = list(set(important_terms))[:3]
            
            # Create topic description
            if topic_terms:
                topic_desc = ' and '.join([t.capitalize() for t in topic_terms])
                summary_parts.append(f"TOPIC: This section discusses {topic_desc}.")
            else:
                summary_parts.append(f"TOPIC: Discussion of questions and responses.")
        
        # Summary of questions
        if questions:
            if len(questions) == 1:
                summary_parts.append(f"\nQUESTION ASKED: \"{questions[0]}\"")
            else:
                summary_parts.append(f"\nQUESTIONS ASKED ({len(questions)}):")
                for i, q in enumerate(questions[:3], 1):  # Show first 3
                    summary_parts.append(f"  {i}. \"{q}\"")
                if len(questions) > 3:
                    summary_parts.append(f"  ... and {len(questions) - 3} more question(s)")
        
        # Summary of answers - extract key information
        if answers:
            combined_answers = ' '.join(answers)
            
            # Extract key sentences (longer, informative sentences)
            try:
                sentences = sent_tokenize(combined_answers)
            except:
                # Fallback: simple sentence splitting
                sentences = re.split(r'[.!?]+\s+', combined_answers)
                sentences = [s.strip() for s in sentences if s.strip()]
            
            # Filter and rank sentences by length and content
            meaningful_sentences = []
            for sent in sentences:
                sent_clean = sent.strip()
                if len(sent_clean) > 25 and not sent_clean.lower().startswith(('okay', 'ok', 'yes', 'no', 'well', 'so', 'then')):
                    meaningful_sentences.append(sent_clean)
            
            # Sort by length (longer = more informative)
            meaningful_sentences.sort(key=len, reverse=True)
            
            summary_parts.append(f"\nKEY INFORMATION FROM ANSWERS:")
            
            if meaningful_sentences:
                # Take top 3-4 most informative sentences
                for i, sent in enumerate(meaningful_sentences[:4], 1):
                    # Clean up sentence
                    sent_clean = sent.replace('\n', ' ').strip()
                    if len(sent_clean) > 200:
                        sent_clean = sent_clean[:200] + "..."
                    summary_parts.append(f"  • {sent_clean}")
            else:
                # Fallback: use first meaningful answer
                for answer in answers:
                    if len(answer) > 30:
                        answer_clean = answer.replace('\n', ' ').strip()
                        if len(answer_clean) > 200:
                            answer_clean = answer_clean[:200] + "..."
                        summary_parts.append(f"  • {answer_clean}")
                        break
        
        # Conclusion/Summary
        if questions and answers:
            summary_parts.append(f"\nSUMMARY: This topic involved {len(questions)} question(s) about the subject matter, with responses providing relevant information and details.")
        
        return '\n'.join(summary_parts)
    
    def _llm_summarize_topic(self, questions, answers):
        """
        Create summary using offline LLM
        """
        if not self.offline_llm:
            return self._rule_based_summarize_topic(questions, answers)
        
        # Prepare prompt
        qa_text = "\n\nQuestions:\n" + "\n".join([f"- {q}" for q in questions])
        qa_text += "\n\nAnswers:\n" + "\n".join([f"- {a}" for a in answers])
        
        prompt = f"""Summarize the following topic from an interview:

{qa_text}

Provide a concise summary (2-3 sentences) of what was discussed, including:
1. The main topic/issue
2. Key points from the answers
3. Any conclusions or important information

Summary:"""
        
        try:
            if self.offline_llm == 'ollama':
                import ollama
                response = ollama.generate(model='llama2', prompt=prompt)
                return response['response']
            elif self.offline_llm == 'llama_cpp':
                # Would need model path - skip for now
                return self._rule_based_summarize_topic(questions, answers)
        except Exception as e:
            print(f"    LLM summarization failed: {e}, using rule-based", flush=True)
            return self._rule_based_summarize_topic(questions, answers)
    
    def _analyze_topic_timeline(self, topic, utterances):
        """Analyze temporal patterns for topic"""
        if not topic['utterance_indices']:
            return
        
        topic_utterances = [utterances[i] for i in topic['utterance_indices']]
        
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

