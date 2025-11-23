"""
Document-Level Hierarchical Topic Modeling
Understanding the ENTIRE conversation first, then identifying main issues and topics

Approach:
1. Create document-level semantic representation (understand the whole conversation)
2. Identify main themes/issues from document-level understanding
3. Hierarchically break down into sub-topics
4. Focus on ISSUES and MATTERS, not just word clusters

Inspired by:
- Transformer attention mechanisms (understanding context)
- Hierarchical topic modeling (document -> themes -> topics)
- GPT-style understanding (semantic comprehension before breakdown)
"""

import numpy as np
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
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
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class DocumentLevelTopicModeling:
    """
    Document-first approach: Understand entire conversation, then extract issues/topics
    """
    
    def __init__(self):
        print("Loading document-level topic modeling system...", flush=True)
        
        # Use a larger model for better semantic understanding
        # all-mpnet-base-v2 is better for document-level understanding
        print("  Loading MPNet model for document-level understanding...", flush=True)
        self.doc_model = SentenceTransformer('all-mpnet-base-v2')  # 768-dim, better for documents
        
        # Also keep sentence model for sentence-level work
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim, faster
        
        # Stopwords for filtering
        try:
            self.stopwords = set(stopwords.words('english'))
        except:
            self.stopwords = set()
            
        print("[OK] Document-level topic modeling ready", flush=True)
    
    def analyze_conversation(self, utterances):
        """
        Main analysis: Understand document, then extract issues/topics
        
        Args:
            utterances: List of dicts with 'text', 'timestamp', etc.
            
        Returns:
            Dictionary with main issues, topics, and hierarchical structure
        """
        if not utterances or len(utterances) < 2:
            return {'topics': [], 'main_issues': [], 'utterances': utterances}
        
        print(f"\n=== DOCUMENT-LEVEL ANALYSIS ===", flush=True)
        print(f"Analyzing {len(utterances)} utterances...", flush=True)
        
        # === STEP 1: CREATE DOCUMENT-LEVEL REPRESENTATION ===
        print("\nStep 1: Creating document-level semantic representation...", flush=True)
        document_embedding, document_text = self._create_document_representation(utterances)
        print(f"  Document embedding: {document_embedding.shape[0]}-dimensional", flush=True)
        print(f"  Document length: {len(document_text.split())} words", flush=True)
        
        # === STEP 2: IDENTIFY MAIN THEMES/ISSUES (Document-Level) ===
        print("\nStep 2: Identifying main themes and issues from document understanding...", flush=True)
        main_themes = self._identify_main_themes(utterances, document_embedding, document_text)
        print(f"  Identified {len(main_themes)} main themes/issues", flush=True)
        
        # === STEP 3: HIERARCHICAL BREAKDOWN (Themes -> Topics) ===
        print("\nStep 3: Hierarchically breaking down themes into specific topics...", flush=True)
        topics = self._hierarchical_breakdown(utterances, main_themes, document_embedding)
        print(f"  Created {len(topics)} specific topics", flush=True)
        
        # === STEP 4: EXTRACT QUESTIONS AND KEY POINTS ===
        print("\nStep 4: Extracting questions and key discussion points...", flush=True)
        questions = self._extract_questions(utterances)
        print(f"  Found {len(questions)} questions", flush=True)
        
        # === STEP 5: TIMELINE ANALYSIS ===
        print("\nStep 5: Analyzing temporal patterns...", flush=True)
        for topic in topics:
            self._analyze_topic_timeline(topic, utterances)
        
        print(f"\n[OK] Analysis complete: {len(main_themes)} main issues, {len(topics)} topics", flush=True)
        
        return {
            'main_issues': main_themes,
            'topics': topics,
            'questions': questions,
            'utterances': utterances,
            'document_embedding': document_embedding
        }
    
    def _create_document_representation(self, utterances):
        """
        Create a document-level semantic representation
        
        Strategy:
        1. Combine all substantive text into one document
        2. Create document-level embedding (mean of all sentence embeddings)
        3. Also create weighted embedding (more weight to longer, more important sentences)
        """
        # Combine all text
        all_texts = [u['text'].strip() for u in utterances if u.get('text', '').strip()]
        document_text = ' '.join(all_texts)
        
        # Create sentence-level embeddings for the entire document
        # Use the better model for document understanding
        sentence_embeddings = self.doc_model.encode(all_texts, show_progress_bar=False)
        
        # Document-level embedding = weighted mean of sentence embeddings
        # Weight by sentence length (longer sentences often more important)
        weights = np.array([len(text.split()) for text in all_texts])
        weights = weights / (weights.sum() + 1e-10)  # Normalize
        
        document_embedding = np.average(sentence_embeddings, axis=0, weights=weights)
        
        return document_embedding, document_text
    
    def _identify_main_themes(self, utterances, document_embedding, document_text):
        """
        Identify main themes/issues from document-level understanding
        
        Strategy:
        1. Cluster ALL utterances into main themes (not segments)
        2. Find clusters that are semantically central to the document
        3. Extract main issues/matters from these clusters
        """
        if len(utterances) < 3:
            return []
        
        # Embed all utterances
        texts = [u['text'] for u in utterances]
        embeddings = self.doc_model.encode(texts, show_progress_bar=False)
        
        # Determine number of main themes (5-10 for a conversation)
        n_themes = min(10, max(5, len(utterances) // 8))
        
        print(f"    Clustering {len(utterances)} utterances into {n_themes} main themes...", flush=True)
        
        # Cluster utterances into main themes
        clustering = AgglomerativeClustering(
            n_clusters=n_themes,
            linkage='average',
            metric='cosine'
        )
        theme_ids = clustering.fit_predict(embeddings)
        
        # Group utterances by theme
        theme_clusters = defaultdict(list)
        for idx, theme_id in enumerate(theme_ids):
            theme_clusters[theme_id].append(idx)
        
        # Calculate centrality and create themes
        main_themes = []
        for theme_id, indices in theme_clusters.items():
            if len(indices) < 2:  # Skip very small clusters
                continue
                
            # Get utterances for this theme
            theme_utterances = [utterances[i] for i in indices]
            theme_text = ' '.join([u['text'] for u in theme_utterances])
            
            # Embed the theme
            theme_embedding = self.doc_model.encode([theme_text], show_progress_bar=False)[0]
            
            # Centrality = similarity to document embedding
            centrality = np.dot(theme_embedding, document_embedding) / (
                np.linalg.norm(theme_embedding) * np.linalg.norm(document_embedding) + 1e-10
            )
            
            # Extract label using most representative utterances
            # Find utterances closest to cluster centroid
            cluster_embeddings = embeddings[indices]
            cluster_centroid = np.mean(cluster_embeddings, axis=0)
            
            # Find most representative utterances (closest to centroid)
            similarities_to_centroid = [
                np.dot(emb, cluster_centroid) / (
                    np.linalg.norm(emb) * np.linalg.norm(cluster_centroid) + 1e-10
                )
                for emb in cluster_embeddings
            ]
            
            # Get top 3 most representative utterances
            top_indices = sorted(range(len(similarities_to_centroid)), 
                               key=lambda i: similarities_to_centroid[i], 
                               reverse=True)[:3]
            representative_utterances = [theme_utterances[i] for i in top_indices]
            
            segment = {
                'utterances': representative_utterances,  # Use only representative ones
                'utterance_indices': [indices[i] for i in top_indices]
            }
            label = self._extract_issue_label(segment)
            
            theme = {
                'label': label,
                'centrality': centrality,
                'utterance_indices': indices,
                'mention_count': len(indices),
                'embedding': theme_embedding
            }
            main_themes.append(theme)
        
        # Sort by centrality (most central themes first)
        main_themes.sort(key=lambda x: x['centrality'], reverse=True)
        
        # Limit to top themes
        main_themes = main_themes[:10]
        
        return main_themes
    
    def _create_semantic_segments(self, utterances, window_size=5, min_segment_size=3):
        """
        Create semantic segments by grouping consecutive similar utterances
        
        Uses a sliding window approach to find natural breaks
        More aggressive grouping to create meaningful segments
        """
        if len(utterances) < min_segment_size:
            return [{'utterance_indices': list(range(len(utterances))), 'utterances': utterances}]
        
        # Embed all utterances
        texts = [u['text'] for u in utterances]
        embeddings = self.sentence_model.encode(texts, show_progress_bar=False)
        
        segments = []
        current_segment = [0]
        
        for i in range(1, len(utterances)):
            # Check similarity to previous utterances in window
            window_start = max(0, i - window_size)
            window_embeddings = embeddings[window_start:i]
            current_embedding = embeddings[i]
            
            # Average similarity to window
            similarities = [
                np.dot(current_embedding, emb) / (
                    np.linalg.norm(current_embedding) * np.linalg.norm(emb) + 1e-10
                )
                for emb in window_embeddings
            ]
            avg_similarity = np.mean(similarities) if similarities else 0
            
            # More lenient threshold - only break if really different
            # Also, ensure minimum segment size
            if avg_similarity < 0.50 and len(current_segment) >= min_segment_size:  # Lower threshold, require min size
                if current_segment:
                    segments.append({
                        'utterance_indices': current_segment,
                        'utterances': [utterances[j] for j in current_segment]
                    })
                current_segment = [i]
            else:
                current_segment.append(i)
        
        # Add final segment (merge with previous if too small)
        if current_segment:
            if len(current_segment) < min_segment_size and segments:
                # Merge with last segment
                segments[-1]['utterance_indices'].extend(current_segment)
                segments[-1]['utterances'].extend([utterances[j] for j in current_segment])
            else:
                segments.append({
                    'utterance_indices': current_segment,
                    'utterances': [utterances[j] for j in current_segment]
                })
        
        return segments
    
    def _extract_issue_label(self, segment):
        """
        Extract a meaningful label for an issue/matter from a segment
        
        Strategy:
        1. Use semantic understanding - what is this segment really about?
        2. Extract key phrases, not just words
        3. Focus on the main subject/matter being discussed
        """
        # Combine all text in segment
        texts = [u['text'] for u in segment['utterances']]
        combined_text = ' '.join(texts)
        
        # Create semantic embedding of the segment
        segment_embedding = self.doc_model.encode([combined_text], show_progress_bar=False)[0]
        
        # Extract key phrases using noun phrases and important terms
        try:
            tokens = word_tokenize(combined_text.lower())
            tagged = pos_tag(tokens)
        except:
            words = re.findall(r'\b[a-z]{3,}\b', combined_text.lower())
            tagged = [(w, 'NN') for w in words]
        
        # Extract noun phrases (adj + noun, or noun + noun)
        noun_phrases = []
        i = 0
        while i < len(tagged) - 1:
            word1, pos1 = tagged[i]
            word2, pos2 = tagged[i + 1] if i + 1 < len(tagged) else ('', '')
            
            # Adjective + Noun
            if pos1.startswith('JJ') and pos2.startswith('NN'):
                if word1 not in self.stopwords and word2 not in self.stopwords:
                    noun_phrases.append(f"{word1} {word2}")
                i += 2
            # Noun + Noun (compound)
            elif pos1.startswith('NN') and pos2.startswith('NN'):
                if word1 not in self.stopwords and word2 not in self.stopwords:
                    noun_phrases.append(f"{word1} {word2}")
                i += 2
            # Single important noun
            elif pos1.startswith('NN') and word1 not in self.stopwords and len(word1) > 4:
                noun_phrases.append(word1)
                i += 1
            else:
                i += 1
        
        # Also extract important single nouns
        important_nouns = []
        for word, pos in tagged:
            if pos.startswith('NN') and word not in self.stopwords and len(word) > 4:
                # Filter out common words
                if word not in ['thing', 'time', 'way', 'people', 'person', 'place', 'part', 'work', 'year', 'day']:
                    important_nouns.append(word)
        
        # Combine and count
        all_terms = noun_phrases + important_nouns
        term_counts = Counter(all_terms)
        
        # Get top terms (prefer phrases over single words)
        top_terms = []
        for term, count in term_counts.most_common(10):
            if ' ' in term:  # Prefer phrases
                top_terms.insert(0, term)
            else:
                top_terms.append(term)
            if len(top_terms) >= 3:
                break
        
        # Create meaningful label
        if top_terms:
            # Filter out filler words and focus on meaningful terms
            meaningful_terms = []
            filler_words = {'thing', 'things', 'way', 'ways', 'time', 'times', 'part', 'parts', 
                          'going', 'tell', 'told', 'say', 'said', 'know', 'knows', 'think', 'thinks',
                          'see', 'sees', 'get', 'gets', 'come', 'comes', 'go', 'goes', 'make', 'makes',
                          'take', 'takes', 'give', 'gives', 'want', 'wants', 'need', 'needs', 'like', 'likes',
                          'topic', 'topics', 'question', 'questions', 'answer', 'answers'}
            
            for term in top_terms:
                # Skip if it's just filler words
                words_in_term = set(term.split())
                if not words_in_term.intersection(filler_words) or len(words_in_term) > 1:
                    meaningful_terms.append(term)
                if len(meaningful_terms) >= 2:
                    break
            
            if meaningful_terms:
                # Capitalize properly
                label_parts = []
                for term in meaningful_terms[:2]:  # Use top 2 meaningful terms
                    if ' ' in term:
                        # Phrase - capitalize each word
                        label_parts.append(' '.join([w.capitalize() for w in term.split()]))
                    else:
                        label_parts.append(term.capitalize())
                
                label = ' & '.join(label_parts)
                
                # Clean up common patterns
                label = re.sub(r'\b(Okay|Ok|Yes|No|Well|So|Then|Now|Just|Also|Even|Still|More|Most|Very|Really|Much|Many|Some|Any|All|Each|Every|This|That|These|Those|Topic|Topics)\b', '', label, flags=re.IGNORECASE)
                label = re.sub(r'\s+', ' ', label).strip()
                
                if len(label) > 5:
                    return label
        
        # Fallback: use first meaningful sentence fragment
        sentences = sent_tokenize(combined_text)
        if sentences:
            first_sent = sentences[0]
            # Extract first 5-7 words
            words = first_sent.split()[:7]
            label = ' '.join([w.capitalize() for w in words if len(w) > 2])
            if len(label) > 10:
                return label
        
        return "General Discussion"
    
    def _hierarchical_breakdown(self, utterances, main_themes, document_embedding):
        """
        Break down main themes into specific topics
        
        For each main theme:
        1. Find all utterances related to that theme
        2. Cluster those utterances into specific sub-topics
        3. Create topic labels
        """
        all_topics = []
        
        for theme_idx, theme in enumerate(main_themes):
            # Get utterances for this theme
            theme_utterances = [utterances[i] for i in theme['utterance_indices']]
            
            if len(theme_utterances) < 2:
                # Single utterance = single topic
                topic = {
                    'topic_id': theme_idx,
                    'theme_id': theme_idx,
                    'label': theme['label'],
                    'utterance_indices': theme['utterance_indices'],
                    'mention_count': len(theme['utterance_indices']),
                    'is_main_theme': True
                }
                all_topics.append(topic)
                continue
            
            # Cluster utterances within this theme into sub-topics
            theme_texts = [u['text'] for u in theme_utterances]
            theme_embeddings = self.sentence_model.encode(theme_texts, show_progress_bar=False)
            
            # Determine number of sub-topics (1-3 per theme)
            n_subtopics = min(3, max(1, len(theme_utterances) // 3))
            
            if n_subtopics == 1:
                # Single topic for this theme
                topic = {
                    'topic_id': len(all_topics),
                    'theme_id': theme_idx,
                    'label': theme['label'],
                    'utterance_indices': theme['utterance_indices'],
                    'mention_count': len(theme['utterance_indices']),
                    'is_main_theme': True
                }
                all_topics.append(topic)
            else:
                # Cluster into sub-topics
                clustering = AgglomerativeClustering(
                    n_clusters=n_subtopics,
                    linkage='average',
                    metric='cosine'
                )
                subtopic_ids = clustering.fit_predict(theme_embeddings)
                
                # Create topics for each sub-cluster
                for subtopic_id in range(n_subtopics):
                    subtopic_indices = [
                        theme['utterance_indices'][i]
                        for i in range(len(theme_utterances))
                        if subtopic_ids[i] == subtopic_id
                    ]
                    
                    subtopic_utterances = [utterances[i] for i in subtopic_indices]
                    subtopic_label = self._extract_issue_label({
                        'utterances': subtopic_utterances,
                        'utterance_indices': subtopic_indices
                    })
                    
                    topic = {
                        'topic_id': len(all_topics),
                        'theme_id': theme_idx,
                        'label': f"{theme['label']}: {subtopic_label}",
                        'utterance_indices': subtopic_indices,
                        'mention_count': len(subtopic_indices),
                        'is_main_theme': False
                    }
                    all_topics.append(topic)
        
        return all_topics
    
    def _extract_questions(self, utterances):
        """
        Extract questions asked during the conversation
        """
        questions = []
        
        for i, utt in enumerate(utterances):
            text = utt.get('text', '').strip()
            if not text:
                continue
            
            # Check if it's a question
            if text.endswith('?') or text.startswith(('what', 'who', 'when', 'where', 'why', 'how', 'did', 'do', 'does', 'can', 'could', 'would', 'will')):
                questions.append({
                    'index': i,
                    'text': text,
                    'timestamp': utt.get('timestamp_str', ''),
                    'speaker': utt.get('speaker_role', 'Unknown')
                })
        
        return questions
    
    def _analyze_topic_timeline(self, topic, utterances):
        """
        Analyze when topic was discussed (timeline analysis)
        """
        if not topic['utterance_indices']:
            return
        
        topic_utterances = [utterances[i] for i in topic['utterance_indices']]
        
        # Get timestamps
        timestamps = []
        for utt in topic_utterances:
            if 'timestamp' in utt:
                timestamps.append(utt['timestamp'])
            elif 'timestamp_str' in utt:
                # Parse timestamp string if needed
                try:
                    ts = datetime.strptime(utt['timestamp_str'], '%H:%M:%S')
                    timestamps.append(ts)
                except:
                    pass
        
        if timestamps:
            topic['first_mention'] = min(timestamps).isoformat()
            topic['last_mention'] = max(timestamps).isoformat()
            
            # Calculate time span
            time_span = (max(timestamps) - min(timestamps)).total_seconds() / 60
            topic['total_span_minutes'] = time_span
            
            # Check if topic was revisited
            if len(timestamps) > 1:
                sorted_timestamps = sorted(timestamps)
                gaps = []
                for i in range(1, len(sorted_timestamps)):
                    gap = (sorted_timestamps[i] - sorted_timestamps[i-1]).total_seconds() / 60
                    if gap > 2.0:  # More than 2 minutes gap
                        gaps.append(gap)
                
                if gaps:
                    topic['is_revisited'] = True
                    topic['period_count'] = len(gaps) + 1
                    topic['revisit_gaps_minutes'] = gaps
                else:
                    topic['is_revisited'] = False
                    topic['period_count'] = 1

