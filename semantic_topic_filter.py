"""
Semantic Topic Filter - Advanced Conversation Analysis
Filters out meta-discourse, conversation management, and transition phrases
to identify REAL topics being discussed.

Based on research:
- Discourse Analysis and Conversation Management (Schiffrin, 1987)
- Meta-discourse markers in conversation (Hyland, 2005)
- Semantic coherence for topic extraction (Barzilay & Lee, 2004)
- Conversation summarization (Gillick et al., 2019)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter
import re


class SemanticTopicFilter:
    """
    Filters utterances to identify REAL topics vs. conversation management
    
    Key insight: "Let's move on to the next topic" is NOT a topic,
    it's conversation management. We need to understand semantics.
    """
    
    def __init__(self):
        # Load semantic model for understanding utterance meaning
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Conversation management patterns (meta-discourse)
        # Comprehensive list based on discourse analysis research
        self.meta_discourse_patterns = [
            # Topic transitions
            r"let'?s?\s+move\s+on",
            r"move\s+on\s+to",
            r"next\s+topic",
            r"let'?s?\s+talk\s+about",
            r"let'?s?\s+discuss",
            r"moving\s+on",
            r"changing\s+topic",
            r"switching\s+to",
            r"let'?s?\s+go\s+to",
            r"let'?s?\s+get\s+back\s+to",
            r"let'?s?\s+continue",
            r"let'?s?\s+proceed",
            r"moving\s+forward",
            r"on\s+to\s+the\s+next",
            
            # Conversation management
            r"give\s+me\s+a\s+moment",
            r"just\s+a\s+moment",
            r"hold\s+on",
            r"wait\s+a\s+minute",
            r"can\s+you\s+repeat",
            r"i\s+don'?t\s+understand",
            r"what\s+do\s+you\s+mean",
            r"i\s+need\s+a\s+moment",
            r"just\s+give\s+me",
            r"one\s+moment",
            r"bear\s+with\s+me",
            r"let\s+me\s+think",
            r"i\s+need\s+to\s+think",
            
            # Filler and acknowledgments
            r"^okay\s*$",
            r"^ok\s*$",
            r"^yeah\s*$",
            r"^yes\s*$",
            r"^no\s*$",
            r"^uh\s+huh\s*$",
            r"^mm\s+hmm\s*$",
            r"^uh\s*$",
            r"^um\s*$",
            r"^hmm\s*$",
            r"^right\s*$",
            r"^sure\s*$",
            r"^alright\s*$",
            
            # Topic navigation
            r"what\s+did\s+we\s+talk\s+about",
            r"going\s+back\s+to",
            r"returning\s+to",
            r"as\s+i\s+said",
            r"like\s+i\s+mentioned",
            r"as\s+i\s+mentioned",
            r"like\s+i\s+said",
            r"going\s+back",
            r"back\s+to\s+what",
            r"earlier\s+you\s+said",
            r"you\s+mentioned\s+earlier",
            
            # Closing/opening phrases
            r"thank\s+you",
            r"thanks",
            r"goodbye",
            r"see\s+you",
            r"have\s+a\s+good\s+day",
            r"nice\s+to\s+meet\s+you",
            r"nice\s+meeting\s+you",
            r"talk\s+to\s+you\s+later",
            r"catch\s+you\s+later",
            r"bye\s+bye",
            r"see\s+you\s+later",
            r"take\s+care",
            
            # Repetition/confirmation
            r"can\s+you\s+say\s+that\s+again",
            r"repeat\s+that",
            r"i\s+didn'?t\s+catch\s+that",
            r"say\s+that\s+again",
            r"come\s+again",
            r"pardon\s+me",
            r"excuse\s+me",
            r"what\s+was\s+that",
            r"what\s+did\s+you\s+say",
            
            # Clarification requests (without content)
            r"^what\s+do\s+you\s+mean\s*$",
            r"^can\s+you\s+clarify\s*$",
            r"^i\s+don'?t\s+follow\s*$",
            r"^i\s+don'?t\s+get\s+it\s*$",
            
            # Topic management phrases
            r"that'?s\s+all\s+for\s+now",
            r"we'?re\s+done\s+with",
            r"finished\s+with",
            r"done\s+talking\s+about",
            r"that\s+concludes",
        ]
        
        # Compile patterns for efficiency
        self.meta_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.meta_discourse_patterns]
        
        # Semantic embeddings for common meta-discourse phrases
        self.meta_discourse_examples = [
            "Let's move on to the next topic",
            "Give me a moment",
            "Can you repeat that?",
            "What do you mean?",
            "Okay, let's continue",
            "Moving on",
            "Next topic",
            "Thank you",
            "Goodbye",
            "See you later",
            "I don't understand",
            "Hold on a minute",
            "Just a moment",
            "What did we talk about?",
            "Going back to what you said",
            "As I mentioned before",
            "Let me think about that",
            "I need a moment",
            "Can you clarify?",
            "What was that again?",
        ]
        
        # Pre-compute embeddings for meta-discourse examples
        self.meta_embeddings = self.semantic_model.encode(self.meta_discourse_examples)
        
    def is_meta_discourse(self, utterance_text):
        """
        Check if utterance is meta-discourse (conversation management, not content)
        
        Uses both pattern matching and semantic similarity
        """
        text_lower = utterance_text.lower().strip()
        
        # Check 1: Pattern matching (fast, catches obvious cases)
        for pattern in self.meta_patterns:
            if pattern.search(text_lower):
                return True, "pattern_match"
        
        # Check 2: Semantic similarity (catches variations)
        utterance_embedding = self.semantic_model.encode([utterance_text])[0]
        
        # Calculate similarity to known meta-discourse examples
        similarities = np.dot(self.meta_embeddings, utterance_embedding)
        max_similarity = np.max(similarities)
        
        # If very similar to meta-discourse (>0.75), it's likely meta-discourse
        if max_similarity > 0.75:
            return True, "semantic_similarity"
        
        # Check 3: Length and content (very short utterances are often filler)
        words = text_lower.split()
        if len(words) <= 3:
            # Check if it's just filler words
            filler_words = {'okay', 'ok', 'yes', 'no', 'yeah', 'uh', 'huh', 'mm', 'hmm', 'well', 'so', 'um'}
            if set(words).issubset(filler_words):
                return True, "filler_words"
        
        return False, None
    
    def has_substantive_content(self, utterance_text):
        """
        Check if utterance contains substantive content (actual topic discussion)
        
        Criteria:
        1. Not meta-discourse
        2. Contains meaningful nouns/entities
        3. Has semantic content beyond conversation management
        """
        # Already checked meta-discourse
        is_meta, reason = self.is_meta_discourse(utterance_text)
        if is_meta:
            return False, f"meta_discourse: {reason}"
        
        # Check for meaningful content
        words = utterance_text.lower().split()
        
        # Too short to be substantive
        if len(words) < 4:
            return False, "too_short"
        
        # Check for question words only (questions need answers to be topics)
        question_only = all(word in {'what', 'where', 'when', 'why', 'how', 'who', 'which', 'do', 'does', 'did', 'is', 'are', 'was', 'were', 'can', 'could', 'would', 'should', 'will'} for word in words[:3])
        if question_only and len(words) < 8:
            return False, "question_only"
        
        return True, "substantive"
    
    def filter_utterances(self, utterances):
        """
        Filter utterances to keep only those with substantive content
        
        Returns:
            filtered_utterances: List of utterances with substantive content
            meta_utterances: List of meta-discourse utterances (for reference)
        """
        filtered = []
        meta = []
        
        for utt in utterances:
            text = utt.get('text', '')
            if not text or len(text.strip()) < 3:
                continue
            
            is_meta, reason = self.is_meta_discourse(text)
            if is_meta:
                utt['_meta_reason'] = reason
                meta.append(utt)
            else:
                has_content, content_reason = self.has_substantive_content(text)
                if has_content:
                    filtered.append(utt)
                else:
                    utt['_meta_reason'] = content_reason
                    meta.append(utt)
        
        return filtered, meta


class SemanticCoherenceAnalyzer:
    """
    Analyzes semantic coherence to identify real topics
    
    Key insight: Real topics have semantic coherence - multiple utterances
    discussing the same subject matter, not just similar words.
    """
    
    def __init__(self, semantic_model):
        self.semantic_model = semantic_model
        
    def calculate_coherence(self, utterances):
        """
        Calculate semantic coherence of a group of utterances
        
        High coherence = utterances discuss the same topic
        Low coherence = utterances are unrelated
        """
        if len(utterances) < 2:
            return 1.0  # Single utterance is perfectly coherent
        
        texts = [u.get('text', '') for u in utterances]
        embeddings = self.semantic_model.encode(texts)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        # Coherence = average pairwise similarity
        coherence = np.mean(similarities) if similarities else 0.0
        
        return coherence
    
    def is_coherent_topic(self, utterances, min_coherence=0.50):
        """
        Check if a group of utterances forms a coherent topic
        
        Args:
            utterances: List of utterances
            min_coherence: Minimum coherence threshold (0.50 = moderate)
        """
        coherence = self.calculate_coherence(utterances)
        return coherence >= min_coherence, coherence


class ContextAwareTopicExtractor:
    """
    Extracts topics with context awareness
    
    Understands that:
    - "Let's talk about cyber security" is conversation management
    - "Cyber security is important for companies" is the actual topic
    - Context matters - same words, different semantic roles
    """
    
    def __init__(self):
        self.semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.filter = SemanticTopicFilter()
        self.coherence_analyzer = SemanticCoherenceAnalyzer(self.semantic_model)
        
    def extract_real_topics(self, utterances):
        """
        Extract REAL topics from conversation
        
        Process:
        1. Filter out meta-discourse
        2. Group by semantic similarity
        3. Check coherence
        4. Extract topic labels from coherent groups
        """
        # Step 1: Filter meta-discourse
        print("  Filtering meta-discourse and conversation management...")
        substantive_utterances, meta_utterances = self.filter.filter_utterances(utterances)
        
        print(f"    Substantive utterances: {len(substantive_utterances)}")
        print(f"    Meta-discourse filtered: {len(meta_utterances)}")
        
        if len(substantive_utterances) < 2:
            return {
                'topics': [],
                'filtered_utterances': substantive_utterances,
                'meta_utterances': meta_utterances
            }
        
        # Step 2: Semantic clustering of substantive content
        print("  Clustering substantive content by semantic similarity...")
        texts = [u.get('text', '') for u in substantive_utterances]
        embeddings = self.semantic_model.encode(texts)
        
        # Use hierarchical clustering with semantic similarity
        from sklearn.cluster import AgglomerativeClustering
        
        # Determine number of clusters (aim for 5-10 main topics)
        n_clusters = min(max(5, len(substantive_utterances) // 10), 15)
        
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(embeddings)
        
        # Step 3: Group by cluster and check coherence
        print("  Analyzing semantic coherence of clusters...")
        clusters = {}
        for idx, label in enumerate(cluster_labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(substantive_utterances[idx])
        
        # Step 4: Filter clusters by coherence
        coherent_topics = []
        for cluster_id, cluster_utterances in clusters.items():
            is_coherent, coherence_score = self.coherence_analyzer.is_coherent_topic(
                cluster_utterances, min_coherence=0.45
            )
            
            if is_coherent and len(cluster_utterances) >= 2:
                # This is a real topic
                topic = {
                    'cluster_id': cluster_id,
                    'utterances': cluster_utterances,
                    'coherence': coherence_score,
                    'mention_count': len(cluster_utterances)
                }
                coherent_topics.append(topic)
        
        print(f"    Coherent topics found: {len(coherent_topics)}")
        
        return {
            'topics': coherent_topics,
            'filtered_utterances': substantive_utterances,
            'meta_utterances': meta_utterances
        }

