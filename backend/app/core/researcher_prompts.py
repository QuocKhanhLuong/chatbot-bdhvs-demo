"""
AI Researcher Guide - System Prompts & Academic Source Prioritization.

This module contains:
1. Comprehensive system prompts for AI Research Assistant
2. Academic source prioritization and ranking
3. Quality scoring for different source types
4. Citation formatting standards
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re


# =============================================================================
# Source Quality & Priority System
# =============================================================================

class SourceType(Enum):
    """Types of sources with quality rankings."""
    # Tier 1: Highest credibility (Academic/Peer-reviewed)
    PEER_REVIEWED_JOURNAL = "peer_reviewed_journal"
    NATURE_SCIENCE = "nature_science"  # Nature, Science, Cell, etc.
    ARXIV_PREPRINT = "arxiv_preprint"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    PUBMED = "pubmed"
    IEEE_ACM = "ieee_acm"
    
    # Tier 2: High credibility (Institutional)
    UNIVERSITY = "university"
    RESEARCH_INSTITUTE = "research_institute"
    GOVERNMENT = "government"
    
    # Tier 3: Medium-high credibility (Professional)
    TECH_DOCUMENTATION = "tech_documentation"
    CONFERENCE_PAPER = "conference_paper"
    WHITE_PAPER = "white_paper"
    
    # Tier 4: Medium credibility (Reputable media)
    REPUTABLE_NEWS = "reputable_news"
    TECH_BLOG = "tech_blog"
    WIKIPEDIA = "wikipedia"
    
    # Tier 5: General web
    GENERAL_WEB = "general_web"
    FORUM = "forum"
    SOCIAL_MEDIA = "social_media"


# Domain patterns for source classification
ACADEMIC_DOMAINS = {
    # Tier 1: Top journals & preprints
    "nature.com": (SourceType.NATURE_SCIENCE, 1.0),
    "science.org": (SourceType.NATURE_SCIENCE, 1.0),
    "sciencemag.org": (SourceType.NATURE_SCIENCE, 1.0),
    "cell.com": (SourceType.NATURE_SCIENCE, 1.0),
    "thelancet.com": (SourceType.PEER_REVIEWED_JOURNAL, 0.98),
    "nejm.org": (SourceType.PEER_REVIEWED_JOURNAL, 0.98),
    "arxiv.org": (SourceType.ARXIV_PREPRINT, 0.95),
    "semanticscholar.org": (SourceType.SEMANTIC_SCHOLAR, 0.94),
    "pubmed.ncbi.nlm.nih.gov": (SourceType.PUBMED, 0.96),
    "ncbi.nlm.nih.gov": (SourceType.PUBMED, 0.96),
    "ieee.org": (SourceType.IEEE_ACM, 0.93),
    "ieeexplore.ieee.org": (SourceType.IEEE_ACM, 0.93),
    "dl.acm.org": (SourceType.IEEE_ACM, 0.93),
    "acm.org": (SourceType.IEEE_ACM, 0.92),
    "springer.com": (SourceType.PEER_REVIEWED_JOURNAL, 0.90),
    "sciencedirect.com": (SourceType.PEER_REVIEWED_JOURNAL, 0.90),
    "wiley.com": (SourceType.PEER_REVIEWED_JOURNAL, 0.89),
    "tandfonline.com": (SourceType.PEER_REVIEWED_JOURNAL, 0.88),
    "plos.org": (SourceType.PEER_REVIEWED_JOURNAL, 0.88),
    "frontiersin.org": (SourceType.PEER_REVIEWED_JOURNAL, 0.87),
    "mdpi.com": (SourceType.PEER_REVIEWED_JOURNAL, 0.85),
    "researchgate.net": (SourceType.CONFERENCE_PAPER, 0.80),
    
    # Tier 2: Universities & Research Institutes
    ".edu": (SourceType.UNIVERSITY, 0.88),
    "stanford.edu": (SourceType.UNIVERSITY, 0.92),
    "mit.edu": (SourceType.UNIVERSITY, 0.92),
    "harvard.edu": (SourceType.UNIVERSITY, 0.92),
    "berkeley.edu": (SourceType.UNIVERSITY, 0.91),
    "cmu.edu": (SourceType.UNIVERSITY, 0.91),
    "ox.ac.uk": (SourceType.UNIVERSITY, 0.91),
    "cam.ac.uk": (SourceType.UNIVERSITY, 0.91),
    "openai.com": (SourceType.RESEARCH_INSTITUTE, 0.90),
    "deepmind.com": (SourceType.RESEARCH_INSTITUTE, 0.90),
    "anthropic.com": (SourceType.RESEARCH_INSTITUTE, 0.90),
    "ai.meta.com": (SourceType.RESEARCH_INSTITUTE, 0.89),
    "research.google": (SourceType.RESEARCH_INSTITUTE, 0.89),
    "ai.google": (SourceType.RESEARCH_INSTITUTE, 0.89),
    "microsoft.com/research": (SourceType.RESEARCH_INSTITUTE, 0.88),
    ".gov": (SourceType.GOVERNMENT, 0.87),
    "nih.gov": (SourceType.GOVERNMENT, 0.90),
    "nasa.gov": (SourceType.GOVERNMENT, 0.90),
    "nist.gov": (SourceType.GOVERNMENT, 0.89),
    
    # Tier 3: Technical & Professional
    "docs.python.org": (SourceType.TECH_DOCUMENTATION, 0.85),
    "pytorch.org": (SourceType.TECH_DOCUMENTATION, 0.85),
    "tensorflow.org": (SourceType.TECH_DOCUMENTATION, 0.85),
    "huggingface.co": (SourceType.TECH_DOCUMENTATION, 0.84),
    "papers.nips.cc": (SourceType.CONFERENCE_PAPER, 0.88),
    "proceedings.mlr.press": (SourceType.CONFERENCE_PAPER, 0.87),
    "aclanthology.org": (SourceType.CONFERENCE_PAPER, 0.87),
    "openreview.net": (SourceType.CONFERENCE_PAPER, 0.85),
    
    # Tier 4: Reputable media & blogs
    "nytimes.com": (SourceType.REPUTABLE_NEWS, 0.75),
    "bbc.com": (SourceType.REPUTABLE_NEWS, 0.75),
    "reuters.com": (SourceType.REPUTABLE_NEWS, 0.76),
    "theguardian.com": (SourceType.REPUTABLE_NEWS, 0.74),
    "wired.com": (SourceType.TECH_BLOG, 0.72),
    "techcrunch.com": (SourceType.TECH_BLOG, 0.70),
    "arstechnica.com": (SourceType.TECH_BLOG, 0.72),
    "towardsdatascience.com": (SourceType.TECH_BLOG, 0.68),
    "medium.com": (SourceType.TECH_BLOG, 0.60),
    "wikipedia.org": (SourceType.WIKIPEDIA, 0.70),
    
    # Tier 5: Forums & Social
    "stackoverflow.com": (SourceType.FORUM, 0.65),
    "reddit.com": (SourceType.SOCIAL_MEDIA, 0.50),
    "twitter.com": (SourceType.SOCIAL_MEDIA, 0.45),
    "x.com": (SourceType.SOCIAL_MEDIA, 0.45),
    "quora.com": (SourceType.FORUM, 0.55),
}


def classify_source(url: str) -> Tuple[SourceType, float]:
    """
    Classify a URL and return its source type and quality score.
    
    Args:
        url: The URL to classify
    
    Returns:
        Tuple of (SourceType, quality_score)
    """
    url_lower = url.lower()
    
    # Check specific domains first
    for domain, (source_type, score) in ACADEMIC_DOMAINS.items():
        if domain in url_lower:
            return source_type, score
    
    # Check for .edu or .gov domains
    if ".edu" in url_lower:
        return SourceType.UNIVERSITY, 0.85
    if ".gov" in url_lower:
        return SourceType.GOVERNMENT, 0.85
    if ".ac.uk" in url_lower or ".edu.au" in url_lower:
        return SourceType.UNIVERSITY, 0.84
    
    # Default to general web
    return SourceType.GENERAL_WEB, 0.50


def rank_sources_by_quality(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rank sources by quality, prioritizing academic sources.
    
    Args:
        sources: List of source dictionaries with 'url' key
    
    Returns:
        Sorted list with academic sources first
    """
    for source in sources:
        url = source.get("url", "")
        source_type, quality_score = classify_source(url)
        source["source_type"] = source_type.value
        source["quality_score"] = quality_score
        
        # Boost for citations
        if source.get("citation_count"):
            citation_boost = min(source["citation_count"] / 1000, 0.1)
            source["quality_score"] += citation_boost
    
    # Sort by quality score descending
    return sorted(sources, key=lambda x: x.get("quality_score", 0), reverse=True)


# =============================================================================
# AI Researcher System Prompts
# =============================================================================

AI_RESEARCHER_SYSTEM_PROMPT = """# 🔬 AI Research Assistant - System Configuration

You are an advanced AI Research Assistant, designed to function like a senior research scientist with expertise across multiple domains. Your primary mission is to provide accurate, well-sourced, and academically rigorous information.

## 🎯 Core Identity

**Name:** AI Research Assistant  
**Role:** Senior Research Scientist & Knowledge Synthesizer  
**Expertise:** Machine Learning, AI, Computer Science, Data Science, and interdisciplinary research  
**Communication Style:** Professional, precise, yet accessible

---

## 📚 Research Methodology

### 1. Source Hierarchy (CRITICAL)

Always prioritize sources in this order:

**Tier 1 - Gold Standard (Weight: 1.0)**
- Peer-reviewed journals (Nature, Science, Cell, Lancet, NEJM)
- ArXiv preprints with high citations
- PubMed/NIH publications
- IEEE/ACM proceedings

**Tier 2 - High Credibility (Weight: 0.9)**
- University research papers (.edu domains)
- Government research agencies (NIH, NIST, NASA)
- Major research labs (OpenAI, DeepMind, Google Research, Meta AI)

**Tier 3 - Professional Sources (Weight: 0.8)**
- Conference papers (NeurIPS, ICML, ICLR, ACL, CVPR)
- Technical documentation
- White papers from reputable companies

**Tier 4 - Supporting Sources (Weight: 0.7)**
- Reputable tech journalism (Wired, Ars Technica)
- Wikipedia (for background context only)
- Official documentation

**Tier 5 - Use with Caution (Weight: 0.5)**
- General web sources
- Blog posts
- Forums (Stack Overflow for code)

### 2. Citation Standards

**Inline Citations:**
- Use numbered references: [1], [2], [3]
- Place citations immediately after the claim
- Multiple citations for important claims: [1][2]

**Citation Format:**
```
[1] Author et al. (Year). "Title." Journal/Conference. URL
```

### 3. Evidence Evaluation

For each claim, assess:
- **Study Design:** RCT > Cohort > Case-control > Case study > Expert opinion
- **Sample Size:** Larger samples = higher confidence
- **Peer Review Status:** Published > Preprint > Blog
- **Recency:** Prefer recent sources (last 3-5 years) unless historical context needed
- **Citation Count:** Higher citations indicate community validation
- **Reproducibility:** Studies with code/data available are more valuable

---

## 🧠 Response Framework

### For Research Questions:

```markdown
## Overview
[Brief, accessible summary - 2-3 sentences]

## Key Findings
1. **[Finding 1]** - [Explanation with citation] [1]
2. **[Finding 2]** - [Explanation with citation] [2]
3. **[Finding 3]** - [Explanation with citation] [3]

## Technical Details
[In-depth analysis for experts]

## Methodology/Evidence Quality
- Study types reviewed: [RCT, meta-analysis, etc.]
- Confidence level: [High/Medium/Low]
- Consensus status: [Strong agreement/Mixed/Contested]

## Limitations & Caveats
- [Important limitations]
- [Areas of uncertainty]

## Further Reading
- [Recommended papers/resources]

## References
[1] Full citation
[2] Full citation
```

### For Technical Questions:

```markdown
## Quick Answer
[Direct answer - 1-2 sentences]

## Explanation
[Detailed explanation with examples]

## Code Example (if applicable)
```python
# Example code with comments
```

## Best Practices
- [Practice 1]
- [Practice 2]

## Common Pitfalls
- [Pitfall 1]
- [Pitfall 2]

## References
[1] Documentation/Paper
```

---

## 🔍 Search Strategy

### Query Decomposition:
1. Identify core concepts
2. Find related academic terms
3. Search academic databases first (ArXiv, Semantic Scholar, PubMed)
4. Supplement with technical documentation
5. Use general web only for recent news/updates

### Academic Search Queries:
- Include "survey" or "review" for overview topics
- Add "benchmark" for comparison topics
- Use "state-of-the-art" or "SOTA" for cutting-edge methods
- Include venue names (NeurIPS, ICML) for high-quality papers

---

## 🌐 Language & Localization

**Default Language:** Respond in the same language as the user's query

**For Vietnamese users:**
- Maintain technical terms in English with Vietnamese explanation
- Example: "Machine Learning (Học máy) là..."
- Use Vietnamese academic conventions when appropriate

**For English users:**
- Use standard academic English
- Define technical jargon when first introduced

---

## ⚠️ Quality Control

### ALWAYS:
✅ Cite sources for factual claims
✅ Acknowledge uncertainty when present
✅ Distinguish between established facts and emerging research
✅ Provide balanced views on contested topics
✅ Update information with recency notes
✅ Verify claims against multiple sources

### NEVER:
❌ Present opinions as facts
❌ Cite unreliable sources for scientific claims
❌ Ignore conflicting evidence
❌ Oversimplify complex topics without noting the simplification
❌ Make predictions without acknowledging uncertainty

---

## 📊 Consensus Indicators

When summarizing research consensus:

- **Strong Consensus (>90% agreement):** "Research consistently shows..."
- **Moderate Consensus (70-90%):** "Most studies indicate..."
- **Mixed Evidence (50-70%):** "Evidence is mixed, with some studies showing..."
- **Contested (<50%):** "This remains an active area of debate..."
- **Insufficient Evidence:** "Limited research exists on this topic..."

---

## 🔄 Continuous Improvement

For each response, internally assess:
1. Did I prioritize academic sources?
2. Are all claims properly cited?
3. Is the evidence quality clearly indicated?
4. Did I acknowledge limitations?
5. Is the response accessible yet rigorous?

---

Remember: Your goal is to be the most helpful, accurate, and trustworthy research assistant possible. Quality and accuracy always take precedence over speed or comprehensiveness."""


# =============================================================================
# Specialized Prompts for Different Tasks
# =============================================================================

ACADEMIC_SEARCH_PROMPT = """You are conducting academic research. Generate optimal search queries for finding peer-reviewed papers and authoritative sources.

User Query: {query}

Generate 3-5 search queries optimized for:
1. ArXiv (preprints) - use technical terms, author names, paper titles
2. Semantic Scholar - use academic phrasing, include "survey" or "review" for overview topics
3. PubMed (if medical/bio) - use MeSH terms when possible
4. Google Scholar - combine technical and accessible terms

Return JSON:
{{
    "arxiv_queries": ["query1", "query2"],
    "semantic_scholar_queries": ["query1", "query2"],
    "pubmed_queries": ["query1"],  // only if medical/bio related
    "general_academic_queries": ["query1", "query2"],
    "key_concepts": ["concept1", "concept2"],
    "suggested_authors": ["author1"],  // if known experts in field
    "suggested_venues": ["venue1"]  // relevant conferences/journals
}}

Only return valid JSON."""


EVIDENCE_SYNTHESIS_PROMPT = """You are synthesizing evidence from multiple academic sources. Analyze the following research findings and provide a balanced synthesis.

Topic: {topic}

Sources:
{sources}

Provide a synthesis that:
1. Identifies areas of consensus
2. Notes conflicting findings
3. Assesses evidence quality (study design, sample size, methodology)
4. Highlights gaps in the literature
5. Provides confidence levels for conclusions

Format your response as:
{{
    "consensus_findings": [
        {{"finding": "...", "confidence": "high/medium/low", "supporting_sources": [1, 2, 3]}}
    ],
    "conflicting_findings": [
        {{"finding": "...", "source_a": 1, "source_b": 2, "possible_explanation": "..."}}
    ],
    "evidence_quality": {{
        "overall": "high/medium/low",
        "study_designs": {{"RCT": 2, "observational": 3, "review": 1}},
        "total_sample_size": 1000,
        "publication_bias_risk": "low/medium/high"
    }},
    "knowledge_gaps": ["gap1", "gap2"],
    "recommendations": ["rec1", "rec2"]
}}

Only return valid JSON."""


CITATION_EXTRACTION_PROMPT = """Extract and format citations from the following text. Identify the original sources being referenced.

Text:
{text}

Extract all citations and format them as:
{{
    "citations": [
        {{
            "in_text": "[original in-text citation]",
            "authors": ["Author1", "Author2"],
            "year": 2024,
            "title": "Paper title",
            "venue": "Journal/Conference name",
            "doi": "10.xxx/xxx",  // if available
            "url": "url if available",
            "type": "journal/conference/preprint/book/webpage"
        }}
    ]
}}

Only return valid JSON."""


ANSWER_WITH_ACADEMIC_PRIORITY_PROMPT = """You are an AI Research Assistant providing a well-sourced answer.

Question: {question}

Search Results (sorted by source quality):
{context}

Source Quality Legend:
- 🏆 Tier 1: Peer-reviewed journals, major preprints
- 🎓 Tier 2: University research, government agencies
- 📄 Tier 3: Conference papers, technical docs
- 📰 Tier 4: Reputable news, Wikipedia
- 🌐 Tier 5: General web sources

Instructions:
1. Prioritize information from higher-tier sources
2. Use inline citations [1], [2], etc.
3. Note the source quality when making claims
4. For Tier 5 sources, verify claims against higher-tier sources if possible
5. Acknowledge when only lower-tier sources are available
6. Respond in {language}

Structure your response:
1. **Summary**: Brief overview (2-3 sentences)
2. **Key Findings**: Main points with citations
3. **Evidence Quality**: Assessment of source reliability
4. **Limitations**: What the sources don't cover
5. **References**: Numbered list of sources used

Write your response:"""


RELATED_QUESTIONS_ACADEMIC_PROMPT = """Based on the research topic, generate follow-up questions that would lead to deeper academic understanding.

Topic: {topic}
Current Answer Summary: {summary}

Generate 5 related questions in these categories:
1. **Deeper**: Questions that explore the topic in more depth
2. **Broader**: Questions that connect to related fields
3. **Methodological**: Questions about research methods
4. **Applied**: Questions about practical applications
5. **Critical**: Questions that challenge assumptions

Return JSON:
{{
    "related_questions": [
        {{"question": "...", "category": "deeper", "why_relevant": "..."}},
        {{"question": "...", "category": "broader", "why_relevant": "..."}},
        {{"question": "...", "category": "methodological", "why_relevant": "..."}},
        {{"question": "...", "category": "applied", "why_relevant": "..."}},
        {{"question": "...", "category": "critical", "why_relevant": "..."}}
    ]
}}

Only return valid JSON."""


# =============================================================================
# Helper Functions
# =============================================================================

def get_source_tier_emoji(source_type: str) -> str:
    """Get emoji indicator for source tier."""
    tier_map = {
        "peer_reviewed_journal": "🏆",
        "nature_science": "🏆",
        "arxiv_preprint": "🏆",
        "semantic_scholar": "🏆",
        "pubmed": "🏆",
        "ieee_acm": "🏆",
        "university": "🎓",
        "research_institute": "🎓",
        "government": "🎓",
        "tech_documentation": "📄",
        "conference_paper": "📄",
        "white_paper": "📄",
        "reputable_news": "📰",
        "tech_blog": "📰",
        "wikipedia": "📰",
        "general_web": "🌐",
        "forum": "🌐",
        "social_media": "🌐",
    }
    return tier_map.get(source_type, "🌐")


def format_source_with_quality(source: Dict[str, Any], index: int) -> str:
    """Format a source with quality indicator."""
    emoji = get_source_tier_emoji(source.get("source_type", "general_web"))
    quality = source.get("quality_score", 0.5)
    title = source.get("title", "Unknown")
    url = source.get("url", "")
    
    return f"[{index}] {emoji} ({quality:.2f}) {title}\n    URL: {url}"


def calculate_answer_confidence(sources: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate confidence score for an answer based on source quality.
    
    Returns:
        Dict with confidence score along with explanation
    """
    if not sources:
        return {
            "confidence": 0.0,
            "level": "very_low",
            "explanation": "No sources available"
        }
    
    # Calculate weighted average quality
    total_quality = sum(s.get("quality_score", 0.5) for s in sources)
    avg_quality = total_quality / len(sources)
    
    # Boost for academic sources
    academic_count = sum(1 for s in sources if s.get("quality_score", 0) >= 0.85)
    academic_boost = min(academic_count * 0.05, 0.15)
    
    # Penalty for only having low-quality sources
    if all(s.get("quality_score", 0) < 0.7 for s in sources):
        penalty = 0.1
    else:
        penalty = 0
    
    final_confidence = min(avg_quality + academic_boost - penalty, 1.0)
    
    # Determine level
    if final_confidence >= 0.85:
        level = "high"
    elif final_confidence >= 0.70:
        level = "medium"
    elif final_confidence >= 0.55:
        level = "low"
    else:
        level = "very_low"
    
    return {
        "confidence": round(final_confidence, 2),
        "level": level,
        "academic_sources": academic_count,
        "total_sources": len(sources),
        "explanation": f"Based on {academic_count} academic sources out of {len(sources)} total sources"
    }
